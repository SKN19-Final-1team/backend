from dataclasses import dataclass
import os
import time
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
from app.rag.postprocess.consult_hints import build_consult_hints
from app.llm.rag_llm.guidance_script_generator import generate_guidance_script
from app.rag.postprocess.cards import omit_empty, promote_definition_doc, split_cards_by_query
from app.rag.postprocess.keywords import collect_query_keywords, normalize_text
from app.rag.postprocess.sections import clean_card_docs
from app.rag.router.router import route_query
from app.rag.policy.policy_pins import POLICY_PINS
from app.llm.rag_llm.card_generator import build_rule_cards

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
    t_route = time.perf_counter()

    should_search = routing.get("should_search")
    if should_search is None:
        should_search = routing.get("should_route")
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
            "guidanceScript": cfg.no_route_answer,
            "routing": routing,
            "meta": {"model": None, "doc_count": 0, "context_chars": 0},
        }

    retrieve_cache_status = "off"
    filters = routing.get("filters") or routing.get("boost") or {}
    cache_key = None
    docs: List[Dict[str, Any]] = []
    if RETRIEVE_CACHE_ENABLED:
        cache_key = build_retrieval_cache_key(
            normalized_query=normalize_text(query),
            route=routing.get("route") or routing.get("ui_route") or "",
            db_route=routing.get("db_route") or "",
            filters=filters,
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
    if retrieve_cache_status not in ("hit(mem)", "hit(redis)"):
        docs = await retrieve_docs(query=query, routing=routing, top_k=cfg.top_k)
        if RETRIEVE_CACHE_ENABLED and cache_key:
            entries = build_retrieve_cache_entries(docs)
            if entries:
                await retrieval_cache_set(cache_key, entries)
                if retrieve_cache_status == "off":
                    retrieve_cache_status = "miss"
    if should_search_consult_cases(query, routing, session_state):
        consult_docs = await retrieve_consult_cases(query=query, routing=routing, top_k=cfg.top_k)
        if os.getenv("RAG_GUIDANCE_FROM_CONSULT_LLM", "1") != "0":
            matched = routing.get("matched") or {}
            intent_terms = matched.get("actions") or matched.get("weak_intents") or []
            consult_hints = build_consult_hints(consult_docs, intent_terms=intent_terms)
            consult_guidance_script = generate_guidance_script(
                query=query,
                docs=docs[:2],
                consult_hints=consult_hints,
                model=cfg.model,
            )

    docs = clean_card_docs(docs, query)
    t_retrieve = time.perf_counter()
    if routing.get("route") == "card_info":
        docs = promote_definition_doc(docs)

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
    for card in cards:
        card["keywords"] = query_keywords
    cards = [omit_empty(card) for card in cards]
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

    if consult_guidance_script:
        guidance_script = consult_guidance_script
        guide_script_message = consult_guidance_script
    else:
        guide_script_message = build_guide_script_message(docs, consult_docs, guidance_script)
    if guide_script_message:
        guidance_script = guide_script_message
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
            "consult_hints": consult_hints if "consult_hints" in locals() else {},
        }
    if cfg.include_docs:
        response["docs"] = docs
        response["consult_docs"] = consult_docs
    if sllm_keywords:
        response["sllm_keywords"] = sllm_keywords
    return response
