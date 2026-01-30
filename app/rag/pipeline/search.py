from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import asyncio
import os
import time

from app.rag.cache.retrieval_cache import (
    RETRIEVE_CACHE_ENABLED,
    build_retrieval_cache_key,
    retrieval_cache_get,
    retrieval_cache_set,
)
from app.rag.pipeline.retrieve import retrieve_consult_cases, retrieve_docs, retrieve_docs_card_info
from app.rag.pipeline.utils import (
    apply_session_context,
    build_retrieve_cache_entries,
    docs_from_retrieve_cache,
    format_ms,
    should_search_consult_cases,
)
from app.rag.postprocess.keywords import normalize_text
from app.rag.router.router import route_query
from app.rag.policy.search_gating import decide_search_gating
from app.rag.policy.answer_class import classify as classify_answer_class


LOG_TIMING = os.getenv("RAG_LOG_TIMING", "1") != "0"
LOG_RETRIEVER_DEBUG = os.getenv("RAG_LOG_RETRIEVER_DEBUG") == "1"
RETRIEVE_BUDGET_MS = int(os.getenv("RAG_RETRIEVE_BUDGET_MS", "950"))
RETRIEVE_MAX_STAGES = int(os.getenv("RAG_RETRIEVE_MAX_STAGES", "2"))


@dataclass(frozen=True)
class SearchResult:
    routing: Dict[str, Any]
    docs: List[Dict[str, Any]]
    consult_docs: List[Dict[str, Any]]
    retrieve_cache_status: str
    should_search: bool
    no_search_message: Optional[str]
    t_start: float
    t_route: float
    t_retrieve: float


def route(query: str) -> Dict[str, Any]:
    return route_query(query)


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


async def run_search(
    query: str,
    *,
    top_k: int,
    enable_consult_search: bool = True,
    session_state: Optional[Dict[str, Any]] = None,
) -> SearchResult:
    t_start = time.perf_counter()
    routing = apply_session_context(query, route(query), session_state)
    phone_intent = any(k in query for k in ("전화", "번호", "고객센터", "연락처", "전화번호"))
    if phone_intent:
        filters = routing.get("filters") or {}
        filters["phone_lookup"] = True
        routing["filters"] = filters
        routing["route"] = "card_usage"
        routing["db_route"] = "guide_tbl"
        routing["ui_route"] = "card_usage"
        if (routing.get("route") or routing.get("ui_route")) == "card_info":
            routing["route"] = "card_usage"
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
            # print(
            #     "[rag] "
            #     f"route={format_ms(t_route - t_start)} "
            #     f"total={format_ms(total)} "
            #     f"should_search=False route={routing.get('route')}"
            # )
        return SearchResult(
            routing=routing,
            docs=[],
            consult_docs=[],
            retrieve_cache_status="off",
            should_search=False,
            no_search_message=gating.message,
            t_start=t_start,
            t_route=t_route,
            t_retrieve=t_route,
        )

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
            top_k=top_k,
        )
        cached = await retrieval_cache_get(cache_key)
        if cached:
            entries, backend = cached
            docs = docs_from_retrieve_cache(entries)
            retrieve_cache_status = f"hit({backend})" if docs else "miss"
        else:
            retrieve_cache_status = "miss"

    consult_docs: List[Dict[str, Any]] = []
    consult_task: Optional[asyncio.Task] = None
    if enable_consult_search and (routing.get("route") or routing.get("ui_route")) == "card_usage":
        if should_search_consult_cases(query, routing, session_state, commit=False):
            consult_task = asyncio.create_task(
                retrieve_consult_cases(query=query, routing=dict(routing), top_k=top_k)
            )
    if retrieve_cache_status not in ("hit(mem)", "hit(redis)"):
        allow_fallback = (routing.get("route") or routing.get("ui_route")) != "card_info"
        retrieve_stage = 0
        retrieve_start = time.perf_counter()
        route_name = routing.get("route") or routing.get("ui_route")
        effective_top_k = top_k
        if route_name == "card_usage":
            effective_top_k = min(effective_top_k, 2)
        if route_name == "card_info":
            docs = await retrieve_docs_card_info(
                query=query,
                routing=routing,
                top_k=effective_top_k,
                log_scores=LOG_RETRIEVER_DEBUG,
                budget_ms=RETRIEVE_BUDGET_MS,
                start_ts=retrieve_start,
            )
            retrieve_stage = 2
        else:
            docs = await retrieve_docs(query=query, routing=routing, top_k=effective_top_k)
            retrieve_stage = 1
        elapsed_ms = (time.perf_counter() - retrieve_start) * 1000
        budget_exceeded = elapsed_ms >= RETRIEVE_BUDGET_MS
        stage_exceeded = retrieve_stage >= RETRIEVE_MAX_STAGES
        if allow_fallback:
            if (not budget_exceeded) and (not stage_exceeded) and _retrieval_failed(docs, routing) and routing.get("retrieval_mode") != "hybrid":
                routing = dict(routing)
                routing["retrieval_mode"] = "hybrid"
                docs = await retrieve_docs(query=query, routing=routing, top_k=effective_top_k)
                retrieve_stage += 1
            elif (
                (not budget_exceeded)
                and (not stage_exceeded)
                and not docs
                and routing.get("domain_score", 0) >= 3
                and not routing.get("_lane_fallback_used")
            ):
                flipped = _flip_route_for_fallback(routing)
                docs = await retrieve_docs(query=query, routing=flipped, top_k=effective_top_k)
                if docs:
                    routing = flipped
                retrieve_stage += 1
        if RETRIEVE_CACHE_ENABLED and cache_key:
            entries = build_retrieve_cache_entries(docs)
            if entries:
                await retrieval_cache_set(cache_key, entries)
                if retrieve_cache_status == "off":
                    retrieve_cache_status = "miss"
    if consult_task:
        consult_docs = await consult_task
        if (routing.get("route") or routing.get("ui_route")) != "card_usage":
            consult_docs = []
        else:
            if session_state is not None:
                session_state["consult_last_search_at"] = time.time()
                session_state["consult_last_query"] = query

    t_retrieve = time.perf_counter()
    return SearchResult(
        routing=routing,
        docs=docs,
        consult_docs=consult_docs,
        retrieve_cache_status=retrieve_cache_status,
        should_search=True,
        no_search_message=None,
        t_start=t_start,
        t_route=t_route,
        t_retrieve=t_retrieve,
    )
