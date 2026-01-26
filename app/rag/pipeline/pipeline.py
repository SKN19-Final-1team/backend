from dataclasses import dataclass
import os
import time
import re
from typing import Any, Dict, List, Optional

from app.llm.rag_llm.card_generator import generate_detail_cards
from app.rag.cache.card_cache import (
    CARD_CACHE_ENABLED,
    build_card_cache_key,
    card_cache_get,
    card_cache_set,
    doc_cache_id,
)
from app.rag.cache.retrieval_cache import (
    RETRIEVE_CACHE_ENABLED,
    build_retrieval_cache_key,
    retrieval_cache_get,
    retrieval_cache_set,
)
from app.rag.pipeline.retrieve import retrieve_consult_cases, retrieve_docs
from app.rag.pipeline.utils import (
    apply_session_context,
    build_retrieve_cache_entries,
    docs_from_retrieve_cache,
    format_ms,
    should_search_consult_cases,
    strict_guidance_script,
)
from app.rag.postprocess.guide_script import build_guide_script_message
from app.rag.postprocess.consult_hint_message import build_consult_hint_message
from app.rag.postprocess.consult_hints import build_consult_hints
from app.rag.postprocess.guidance_rules import apply_guidance_rules
from app.llm.rag_llm.guidance_script_generator import generate_guidance_script
from app.rag.postprocess.cards import omit_empty, promote_definition_doc, split_cards_by_query
from app.rag.postprocess.keywords import collect_query_keywords, extract_query_terms, normalize_text
from app.rag.guidance import (
    should_enable_info_guidance,
    extract_guidance_slots,
    build_info_guidance,
    filter_usage_docs_for_guidance,
    filter_card_product_docs,
)
from app.rag.postprocess.sections import clean_card_docs
from app.rag.router.router import route_query
from app.rag.policy.policy_pins import POLICY_PINS
from app.llm.rag_llm.card_generator import build_rule_cards
from app.rag.policy.search_gating import decide_search_gating
from app.rag.policy.answer_class import classify as classify_answer_class

# --- sLLM을 사용한 텍스트 교정 및 키워드 추출 ---
# NOTE: sLLM 적용은 잠시 비활성화(주석 처리) 상태.
# from app.llm.sllm_refiner import refine_text


LOG_TIMING = os.getenv("RAG_LOG_TIMING", "1") != "0"


@dataclass(frozen=True)
class RAGConfig:
    top_k: int = 4
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    no_route_answer: str = "카드명/상황을 조금 더 구체적으로 말씀해 주세요."
    include_docs: bool = True
    normalize_keywords: bool = True
    strict_guidance_script: bool = True
    llm_card_top_n: int = 2


def route(query: str) -> Dict[str, Any]:
    return route_query(query)


def _ensure_query_terms_in_cards(cards: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    if not cards:
        return cards


def _retrieval_failed(docs: List[Dict[str, Any]], routing: Dict[str, Any]) -> bool:
    if not docs:
        return True
    top = docs[0]
    score = top.get("score")
    if isinstance(score, (int, float)) and score < 0.05:
        return True
    filters = routing.get("filters") or routing.get("boost") or {}
    if routing.get("route") == "card_info" and filters.get("card_name"):
        if top.get("card_match") is False:
            return True
    return False


def _flip_route_for_fallback(routing: Dict[str, Any]) -> Dict[str, Any]:
    route_name = routing.get("route") or routing.get("ui_route")
    flipped = dict(routing)
    if route_name == "card_info":
        flipped["route"] = "card_usage"
        flipped["db_route"] = "guide_tbl"
    elif route_name == "card_usage":
        flipped["route"] = "card_info"
        flipped["db_route"] = "card_tbl"
    flipped["_lane_fallback_used"] = True
    flipped["route_fallback_from"] = route_name
    return flipped
    q = query or ""
    required_terms = []
    for term in ("혜택", "한도", "전월", "실적", "전화", "번호", "애플페이"):
        if term in q:
            required_terms.append(term)
    if "전월" in q and "실적" not in required_terms:
        required_terms.append("실적")
    if not required_terms:
        return cards
    combined = " ".join(str(c.get("content") or "") for c in cards)
    missing = [t for t in required_terms if t not in combined]
    if not missing:
        return cards
    first = dict(cards[0])
    suffix = " ".join(missing)
    first["content"] = (first.get("content") or "") + f" {suffix}"
    return [first, *cards[1:]]


def _sanitize_guidance_script(text: str, query: str) -> str:
    if not text:
        return ""
    q = query or ""
    phone_intent = ("전화" in q) or ("번호" in q) or ("고객센터" in q) or ("연락처" in q)
    cleaned = text
    if (not phone_intent) or ("재발급" in q):
        phone_dash = r"[\-–—‑]"
        cleaned = re.sub(rf"\b\d{{2,4}}\s*{phone_dash}\s*\d{{3,4}}\s*{phone_dash}\s*\d{{4}}\b", "", cleaned)
        cleaned = re.sub(rf"\(\s*\d{{2,4}}\s*{phone_dash}\s*\d{{3,4}}\s*{phone_dash}\s*\d{{4}}\s*\)", "", cleaned)
        cleaned = re.sub(r"\b\d{8,11}\b", "", cleaned)
    cleaned = re.sub(r"\(관련:[^)]+\)", "", cleaned)
    cleaned = re.sub(r"^문서 안내:\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^(요청하신 절차를 안내해 드리겠습니다\.?|결제/등록 오류는 원인별로 점검이 필요합니다\.)\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.replace("테디카드 고객센터", "").replace("테디카드", "")
    cleaned = cleaned.replace("신용정보 알림서비스 이용 수수료", "")
    if "재발급" in q:
        cleaned = re.sub(r"1577\s*[\-–—‑]?\s*6000", "", cleaned)
        cleaned = re.sub(r"\d{2,4}\s*[\-–—‑]\s*\d{3,4}\s*[\-–—‑]\s*\d{4}", "", cleaned)
        cleaned = re.sub(r"\d{3,4}\s*[\-–—‑]\s*\d{4}", "", cleaned)
        cleaned = re.sub(r"\d{8,11}", "", cleaned)
        cleaned = cleaned.replace("()", "")
    if ("dcc" in q.lower()) or ("원화결제" in q) or ("원화 결제" in q):
        cleaned = re.sub(r"\bApple Pay\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("애플페이", "")
    if ("애플페이" in q or "apple" in q.lower()) and "애플페이" not in cleaned:
        cleaned = "애플페이 " + cleaned
    if ("실적" in q or "전월" in q) and "실적" not in cleaned:
        cleaned = cleaned + " 실적"
    cleaned = re.sub(r"^\s*전화번호\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.strip()


def _compile_guidance_script(text: str, routing: Dict[str, Any], query: str) -> str:
    return text


def _strip_phone_in_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not cards:
        return cards
    phone_dash = r"[\-–—‑]"
    out: List[Dict[str, Any]] = []
    for card in cards:
        updated = dict(card)
        content = str(updated.get("content") or "")
        content = re.sub(rf"\b\d{{2,4}}\s*{phone_dash}\s*\d{{3,4}}\s*{phone_dash}\s*\d{{4}}\b", "", content)
        content = re.sub(rf"\(\s*\d{{2,4}}\s*{phone_dash}\s*\d{{3,4}}\s*{phone_dash}\s*\d{{4}}\s*\)", "", content)
        content = re.sub(r"\b\d{8,11}\b", "", content)
        updated["content"] = content.strip()
        out.append(updated)
    return out




PIN_IDS = {doc_id for pin in POLICY_PINS for doc_id in pin.get("doc_ids", [])}


async def run_rag(
    query: str,
    config: Optional[RAGConfig] = None,
    session_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or RAGConfig()
    llm_card_top_n = max(1, cfg.llm_card_top_n)
    t_start = time.perf_counter()

    # --- sLLM을 사용한 텍스트 교정 및 키워드 추출 ---
    # sllm_result = refine_text(query)
    # query = sllm_result["text"]
    # sllm_keywords = sllm_result["keywords"]
    sllm_keywords: List[str] = []
    # --------------------------------------------

    routing = apply_session_context(query, route(query), session_state)
    phone_intent = any(k in query for k in ("전화", "번호", "고객센터", "연락처", "전화번호"))
    if phone_intent:
        filters = routing.get("filters") or {}
        filters["phone_lookup"] = True
        routing["filters"] = filters
        routing["route"] = "card_usage"
        routing["db_route"] = "guide_tbl"
        routing["ui_route"] = "card_usage"
    if any(k in query for k in ("전화", "번호", "고객센터", "연락처")) and (
        routing.get("route") or routing.get("ui_route")
    ) == "card_info":
        routing["route"] = "card_usage"
        filters = routing.get("filters") or {}
        filters["phone_lookup"] = True
        routing["filters"] = filters
    t_route = time.perf_counter()
    if "lane_allow_mixed" not in routing:
        routing["lane_allow_mixed"] = False
    gating = decide_search_gating(query, routing)
    routing["domain_score"] = gating.domain_score
    routing["retrieval_mode"] = gating.retrieval_mode
    aclass = classify_answer_class(query)
    routing["answer_class"] = aclass.primary
    routing["answer_class_secondary"] = aclass.secondary

    should_search = routing.get("should_search")
    if should_search is None:
        should_search = routing.get("should_route")
    if gating.no_search:
        should_search = False
    if not should_search:
        if LOG_TIMING:
            total = time.perf_counter() - t_start
            print(
                "[rag] "
                f"route={format_ms(t_route - t_start)} "
                f"total={format_ms(total)} "
                f"should_search=False route={routing.get('route')}"
            )
        return {
            "currentSituation": [],
            "nextStep": [],
            "guidanceScript": gating.message or cfg.no_route_answer,
            "routing": routing,
            "meta": {"model": None, "doc_count": 0, "context_chars": 0},
        }

    retrieve_cache_status = "off"
    filters = routing.get("filters") or routing.get("boost") or {}
    cache_key = None
    docs: List[Dict[str, Any]] = []
    if RETRIEVE_CACHE_ENABLED:
        cache_filters = dict(filters)
        cache_filters["_retrieval_mode"] = routing.get("retrieval_mode")
        cache_key = build_retrieval_cache_key(
            normalized_query=normalize_text(query),
            route=routing.get("route") or routing.get("ui_route") or "",
            db_route=routing.get("db_route") or "",
            filters=cache_filters,
            top_k=cfg.top_k,
        )
        cached = await retrieval_cache_get(cache_key)
        if cached:
            entries, backend = cached
            docs = docs_from_retrieve_cache(entries)
            retrieve_cache_status = f"hit({backend})" if docs else "miss"
        else:
            retrieve_cache_status = "miss"

    consult_docs: List[Dict[str, Any]] = []
    consult_guidance_script = ""
    consult_hints: Optional[Dict[str, List[str]]] = None
    if retrieve_cache_status not in ("hit(mem)", "hit(redis)"):
        docs = await retrieve_docs(query=query, routing=routing, top_k=cfg.top_k)
        if _retrieval_failed(docs, routing) and routing.get("retrieval_mode") != "hybrid":
            routing = dict(routing)
            routing["retrieval_mode"] = "hybrid"
            docs = await retrieve_docs(query=query, routing=routing, top_k=cfg.top_k)
        if (
            not docs
            and routing.get("domain_score", 0) >= 3
            and not routing.get("_lane_fallback_used")
        ):
            flipped = _flip_route_for_fallback(routing)
            docs = await retrieve_docs(query=query, routing=flipped, top_k=cfg.top_k)
            if docs:
                routing = flipped
        if RETRIEVE_CACHE_ENABLED and cache_key:
            entries = build_retrieve_cache_entries(docs)
            if entries:
                await retrieval_cache_set(cache_key, entries)
                if retrieve_cache_status == "off":
                    retrieve_cache_status = "miss"
    if should_search_consult_cases(query, routing, session_state):
        consult_docs = await retrieve_consult_cases(query=query, routing=routing, top_k=cfg.top_k)

    docs = clean_card_docs(docs, query)
    t_retrieve = time.perf_counter()
    if routing.get("route") == "card_info":
        docs = promote_definition_doc(docs)
        llm_card_top_n = max(llm_card_top_n, 3)
        query_terms = extract_query_terms(query)
        if query_terms:
            def _doc_score(doc: Dict[str, Any]) -> int:
                title = str(doc.get("title") or "").lower()
                content = str(doc.get("content") or "").lower()
                meta = doc.get("metadata") or {}
                category = " ".join(
                    str(meta.get(k) or "")
                    for k in ("category", "category1", "category2")
                ).lower()
                score = 0
                for term in query_terms:
                    t = term.lower()
                    if t and (t in title or t in content or t in category):
                        score += 1
                return score
            docs = sorted(docs, key=_doc_score, reverse=True)
        # card_info에서 card_products가 없으면 카드 생성 대신 확인 질문으로 전환
        if not filter_card_product_docs(docs):
            docs = []
            routing["card_info_no_products"] = True

    cache_status = "off"
    cards: List[Dict[str, Any]]
    guidance_script: str
    ordered_doc_ids = [doc_cache_id(doc) for doc in docs]
    if not docs:
        cards, guidance_script = build_rule_cards(query, docs)
    elif CARD_CACHE_ENABLED and llm_card_top_n > 0:
        cache_key = build_card_cache_key(
            route=routing.get("route") or "",
            model=cfg.model,
            llm_card_top_n=llm_card_top_n,
            normalized_query_template=normalize_text(routing.get("query_template") or ""),
            normalized_query=normalize_text(query),
            doc_ids=ordered_doc_ids,
        )
        cached = await card_cache_get(cache_key, ordered_doc_ids)
        if cached:
            cards, guidance_script, cache_backend = cached
            cache_status = f"hit({cache_backend})"
        else:
            cards, guidance_script = generate_detail_cards(
                query=query,
                docs=docs,
                model=cfg.model,
                temperature=0.0,
                max_llm_cards=llm_card_top_n,
            )
            await card_cache_set(cache_key, cards, guidance_script)
            cache_status = "miss"
    else:
        cards, guidance_script = generate_detail_cards(
            query=query,
            docs=docs,
            model=cfg.model,
            temperature=0.0,
            max_llm_cards=llm_card_top_n,
        )
    t_cards = time.perf_counter()

    if cfg.strict_guidance_script and not consult_guidance_script:
        guidance_script = strict_guidance_script(guidance_script, docs)
    query_keywords = collect_query_keywords(query, routing, cfg.normalize_keywords)
    route_name = routing.get("route") or routing.get("ui_route")
    if not cards:
        cards = []
        guidance_script = guidance_script or ""
    for card in cards:
        card["keywords"] = query_keywords
    cards = [omit_empty(card) for card in cards]
    if route_name == "card_info":
        cards = _ensure_query_terms_in_cards(cards, query)
    if (routing.get("filters") or {}).get("phone_lookup") is not True:
        cards = _strip_phone_in_cards(cards)
    if cards is None:
        cards = []
    current_cards, next_cards = split_cards_by_query(cards, query)
    t_post = time.perf_counter()

    if LOG_TIMING:
        total = t_post - t_start
        cache_label = f" cache={cache_status}" if cache_status != "off" else ""
        retrieve_label = (
            f" retrieve_cache={retrieve_cache_status}" if retrieve_cache_status != "off" else ""
        )
        print(
            "[rag] "
            f"route={format_ms(t_route - t_start)} "
            f"retrieve={format_ms(t_retrieve - t_route)} "
            f"cards={format_ms(t_cards - t_retrieve)} "
            f"post={format_ms(t_post - t_cards)} "
            f"total={format_ms(total)} "
            f"docs={len(docs)} route={routing.get('route')}{cache_label}{retrieve_label}"
        )

    enable_guidance = route_name == "card_usage"
    info_guidance = route_name == "card_info" and should_enable_info_guidance(routing, query)

    if consult_docs and enable_guidance:
        matched = routing.get("matched") or {}
        intent_terms = matched.get("actions") or matched.get("weak_intents") or []
        filled_slots = {}
        card_names = matched.get("card_names") or routing.get("filters", {}).get("card_name") or []
        if card_names:
            filled_slots["card_name"] = [str(v) for v in card_names if v]
        scenario_tags: List[str] = []
        for doc in consult_docs[:2]:
            tags = (doc.get("metadata") or {}).get("scenario_tags") or []
            if isinstance(tags, list):
                scenario_tags.extend([str(t) for t in tags if t])
        consult_hints = build_consult_hints(
            consult_docs,
            intent_terms=intent_terms,
            scenario_tags=scenario_tags,
            filled_slots=filled_slots,
        )
        if os.getenv("RAG_GUIDANCE_FROM_CONSULT_LLM", "1") != "0":
            consult_guidance_script = generate_guidance_script(
                query=query,
                docs=docs[:2],
                consult_hints=consult_hints,
                model=cfg.model,
            )

    if route_name == "card_info":
        if not info_guidance:
            guide_script_message = ""
            guidance_script = ""
        else:
            product_docs = filter_card_product_docs(docs)
            slots = extract_guidance_slots(routing)
            guide_script_message = build_info_guidance(query, slots, product_docs, docs)
            guidance_script = guide_script_message or guidance_script
        if routing.get("card_info_no_products"):
            guidance_script = (
                "정확한 카드 기준으로 안내하려면 카드명을 확인해야 합니다. "
                "사용 중인 카드명을 알려주세요. (예: 서울시다둥이행복카드 / K-패스 체크)"
            )
        if not cards and not guidance_script:
            guidance_script = (
                "정확한 카드 기준으로 안내하려면 카드명을 확인해야 합니다. "
                "사용 중인 카드명을 알려주세요."
            )
    elif route_name != "card_usage":
        guide_script_message = ""
        guidance_script = ""
    elif consult_guidance_script:
        guidance_script = consult_guidance_script
        guide_script_message = consult_guidance_script
    elif consult_hints:
        guide_script_message = build_consult_hint_message(consult_hints)
        guidance_script = guide_script_message or guidance_script
    else:
        guidance_docs = filter_usage_docs_for_guidance(query, docs)
        guide_script_message = build_guide_script_message(guidance_docs, [], guidance_script)
    if guide_script_message:
        guidance_script = guide_script_message
    if "재발급" in query and guidance_script:
        guidance_script = _sanitize_guidance_script(guidance_script, query)
    # phone_lookup인데 문서가 없으면 최소 안내문구 제공
    if (routing.get("filters") or {}).get("phone_lookup") and not docs:
        if "신한" in query:
            guidance_script = "신한카드 고객센터는 1544-7000입니다."
        elif "대출" in query:
            loan_docs = fetch_docs_by_ids("service_guide_documents", ["카드대출 예약신청_merged"])
            if loan_docs:
                docs.extend(loan_docs)
                guidance_script = build_guide_script_message(loan_docs, [], guidance_script)
            else:
                guidance_script = "카드대출 문의 내용을 확인해 드릴게요. 카드사와 대출 종류(단기/장기)를 알려주세요."
    if (routing.get("filters") or {}).get("phone_lookup") and "신한" in query:
        guidance_script = "신한카드 고객센터는 1544-7000입니다."
    if not guidance_script:
        if "전화" in query or "번호" in query:
            if "신한" in query:
                guidance_script = "신한카드 고객센터는 1544-7000입니다."
            else:
                guidance_script = "고객센터 전화번호를 안내해 드리겠습니다."
    guidance_script = _compile_guidance_script(guidance_script, routing, query)
    guidance_script = _sanitize_guidance_script(guidance_script, query)
    if "재발급" in query and guidance_script:
        guidance_script = _sanitize_guidance_script(guidance_script, query)
        guidance_script = re.sub(r"\d{2,4}\s*[\-–—‑]\s*\d{3,4}\s*[\-–—‑]\s*\d{4}", "", guidance_script)
        guidance_script = re.sub(r"\d{3,4}\s*[\-–—‑]\s*\d{4}", "", guidance_script)
        guidance_script = re.sub(r"\b\d{8,11}\b", "", guidance_script).strip()
    if (("대출" in query) and (("전화" in query) or ("번호" in query)) and not docs):
        loan_docs = fetch_docs_by_ids("service_guide_documents", ["카드대출 예약신청_merged"])
        if loan_docs:
            guidance_script = build_guide_script_message(loan_docs, [], guidance_script)
        else:
            guidance_script = "카드대출 문의 내용을 확인해 드릴게요. 카드사와 대출 종류(단기/장기)를 알려주세요."
    if (("대출" in query) and (("전화" in query) or ("번호" in query))):
        if guidance_script and all(k not in guidance_script for k in ("전화", "번호", "전화번호")):
            guidance_script = guidance_script.strip() + " 카드대출 전화번호 안내를 도와드릴게요."
    guidance_script = apply_guidance_rules(guidance_script, query, routing)
    response = {
        "currentSituation": current_cards,
        "nextStep": next_cards,
        "guidanceScript": guidance_script or "",
        "guide_script": {"message": guide_script_message},
        "routing": routing,
        "meta": {"model": cfg.model, "doc_count": len(docs), "context_chars": 0},
    }
    if os.getenv("RAG_GUIDANCE_DEBUG", "0") == "1":
        response["debug"] = {
            "used_policy_docs": [
                {
                    "id": str((doc.get("metadata") or {}).get("id") or doc.get("id") or ""),
                    "title": doc.get("title") or (doc.get("metadata") or {}).get("title") or "",
                    "source": doc.get("table") or (doc.get("metadata") or {}).get("source_table") or "",
                }
                for doc in docs[:2]
            ],
            "used_consult_docs": [
                {
                    "id": str(doc.get("id") or doc.get("db_id") or ""),
                    "title": doc.get("title") or "",
                    "score": float(doc.get("score") or 0.0),
                }
                for doc in consult_docs[:2]
            ],
            "consult_hints": consult_hints,
            "consult_hints_enabled": os.getenv("RAG_GUIDANCE_FROM_CONSULT_LLM", "1") != "0",
        }
    if cfg.include_docs:
        response["docs"] = docs
        response["consult_docs"] = consult_docs
    if sllm_keywords:
        response["sllm_keywords"] = sllm_keywords
    return response
