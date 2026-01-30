from __future__ import annotations

from typing import Any, Dict, List
import os
import re
import time

from app.llm.rag_llm.card_generator import generate_detail_cards, build_rule_cards
from app.llm.rag_llm.guidance_script_generator import generate_guidance_script
from app.rag.cache.card_cache import (
    CARD_CACHE_ENABLED,
    build_card_cache_key,
    card_cache_get,
    card_cache_set,
    doc_cache_id,
)
from app.rag.pipeline.utils import format_ms, strict_guidance_script
from app.rag.postprocess.cards import omit_empty, promote_definition_doc, split_cards_by_query
from app.rag.postprocess.keywords import collect_query_keywords, extract_query_terms, normalize_text
from app.rag.postprocess.sections import clean_card_docs
from app.rag.guidance import (
    should_enable_info_guidance,
    extract_guidance_slots,
    build_info_guidance,
    filter_card_product_docs,
    filter_guidance_docs,
)


LOG_TIMING = os.getenv("RAG_LOG_TIMING", "1") != "0"

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


async def build_guidance_response(
    *,
    query: str,
    routing: Dict[str, Any],
    docs: List[Dict[str, Any]],
    consult_docs: List[Dict[str, Any]],
    config: Any,
    t_start: float,
    t_route: float,
    t_retrieve: float,
    retrieve_cache_status: str,
) -> Dict[str, Any]:
    # phone lookup은 RAG/LLM을 우회하고 정적 안내만 제공
    if (routing.get("filters") or {}).get("phone_lookup") is True:
        if "신한" in query:
            guidance_script = "신한카드 고객센터는 1544-7000입니다."
        else:
            guidance_script = "고객센터 전화번호를 안내해 드리겠습니다."
        response = {
            "currentSituation": [],
            "nextStep": [],
            "guidanceScript": guidance_script,
            "guide_script": {"message": guidance_script},
            "routing": routing,
            "meta": {"model": config.model, "doc_count": 0, "context_chars": 0},
        }
        if getattr(config, "include_docs", False):
            response["docs"] = []
            if getattr(config, "include_consult_docs", False):
                response["consult_docs"] = []
        return response

    llm_card_top_n = max(1, config.llm_card_top_n)

    docs = clean_card_docs(docs, query)
    route_name = routing.get("route") or routing.get("ui_route")
    llm_docs = docs
    if route_name == "card_usage":
        llm_card_top_n = 1
        if docs:
            def _pin_sort_key(doc: Dict[str, Any]) -> tuple[int, int, float]:
                pinned = 1 if doc.get("_pinned") else 0
                pin_rank = doc.get("_pin_rank")
                pin_rank_key = -pin_rank if isinstance(pin_rank, int) else -10**9
                score = float(doc.get("score") or 0)
                return (pinned, pin_rank_key, score)
            llm_docs = sorted(docs, key=_pin_sort_key, reverse=True)[:1]
    if routing.get("route") == "card_info":
        docs = promote_definition_doc(docs)
        llm_docs = docs
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
            llm_docs = docs
        # card_info에서 card_products가 없으면 카드 생성 대신 확인 질문으로 전환
        if not filter_card_product_docs(docs):
            docs = []
            llm_docs = []
            routing["card_info_no_products"] = True

    cache_status = "off"
    cards: List[Dict[str, Any]]
    guidance_script: str
    ordered_doc_ids = [doc_cache_id(doc) for doc in llm_docs]
    if not llm_docs:
        cards, guidance_script = build_rule_cards(query, docs)
    elif CARD_CACHE_ENABLED and llm_card_top_n > 0:
        cache_key = build_card_cache_key(
            route=routing.get("route") or "",
            model=config.model,
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
                docs=llm_docs,
                model=config.model,
                temperature=0.0,
                max_llm_cards=llm_card_top_n,
            )
            await card_cache_set(cache_key, cards, guidance_script)
            cache_status = "miss"
    else:
        cards, guidance_script = generate_detail_cards(
            query=query,
            docs=llm_docs,
            model=config.model,
            temperature=0.0,
            max_llm_cards=llm_card_top_n,
        )
    t_cards = time.perf_counter()

    if config.strict_guidance_script:
        guidance_script = strict_guidance_script(guidance_script, docs)
    query_keywords = collect_query_keywords(query, routing, config.normalize_keywords)
    if not cards:
        cards = []
        guidance_script = guidance_script or ""
    for card in cards:
        card["keywords"] = query_keywords
    cards = [omit_empty(card) for card in cards]
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
    raw_guide_docs = [
        doc
        for doc in docs
        if str(doc.get("table") or (doc.get("metadata") or {}).get("source_table") or "")
        == "service_guide_documents"
    ]
    guidance_docs = filter_guidance_docs(query, docs, routing=routing) if enable_guidance else []
    if enable_guidance:
        # 카드 생성 LLM의 guidance 문구를 사용하지 않고 별도 가이드로 생성
        guidance_script = ""
        if guidance_docs:
            selected = [
                {
                    "id": str((doc.get("metadata") or {}).get("id") or doc.get("id") or ""),
                    "title": str(doc.get("title") or (doc.get("metadata") or {}).get("title") or ""),
                    "score": float(doc.get("score") or 0.0),
                }
                for doc in guidance_docs
            ]
            print(f"[guide_docs] selected={selected}")
    guide_script_message = ""

    if route_name == "card_info":
        if not info_guidance:
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
        guidance_script = ""
    else:
        if enable_guidance and guidance_docs:
            filters = routing.get("filters") or {}
            intent_terms: List[str] = []
            for key in ("intent", "weak_intent"):
                val = filters.get(key)
                if isinstance(val, list):
                    intent_terms.extend(val)
                elif isinstance(val, str):
                    intent_terms.append(val)
            matched = routing.get("matched") or {}
            card_names = matched.get("card_names") or []
            filled_slots = {"card_name": card_names} if card_names else None
            guidance_script = generate_guidance_script(
                query=query,
                docs=guidance_docs,
                model=config.model,
            )
            if not guidance_script:
                print("[guide_fallback] reason=llm_empty")
        elif enable_guidance and not guidance_docs:
            if not raw_guide_docs:
                print("[guide_fallback] reason=no_docs_raw")
            else:
                print(f"[guide_fallback] reason=no_docs_after_filter raw={len(raw_guide_docs)} filtered=0")
                # Relax filter: keep top docs by score to avoid empty guidance
                guidance_docs = sorted(
                    raw_guide_docs,
                    key=lambda d: float(d.get("score") or 0.0),
                    reverse=True,
                )[:2]
                if guidance_docs:
                    selected = [
                        {
                            "id": str((doc.get("metadata") or {}).get("id") or doc.get("id") or ""),
                            "title": str(doc.get("title") or (doc.get("metadata") or {}).get("title") or ""),
                            "score": float(doc.get("score") or 0.0),
                        }
                        for doc in guidance_docs
                    ]
                    print(f"[guide_docs] relaxed_selected={selected}")
        if enable_guidance and not guidance_script:
            guidance_script = (
                "정확한 안내를 위해 상황을 조금 더 구체적으로 알려주세요.\n"
                "현재 겪고 계신 문제나 원하시는 안내를 말씀해 주실 수 있을까요?"
            )
    final_guidance = guidance_script or ""
    response = {
        "currentSituation": current_cards,
        "nextStep": next_cards,
        "guidanceScript": final_guidance,
        "guide_script": {"message": final_guidance},
        "routing": routing,
        "meta": {"model": config.model, "doc_count": len(docs), "context_chars": 0},
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
        }
    if config.include_docs:
        response["docs"] = docs
        if getattr(config, "include_consult_docs", False):
            response["consult_docs"] = consult_docs
    return response
