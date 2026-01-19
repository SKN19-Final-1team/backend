from dataclasses import dataclass
import os
import re
import time
from typing import Any, Dict, List, Optional

from app.llm.card_generator import generate_detail_cards
from app.rag.cache.card_cache import (
    CARD_CACHE_ENABLED,
    build_card_cache_key,
    card_cache_get,
    card_cache_set,
    doc_cache_id,
)
from app.rag.postprocess.cards import omit_empty, promote_definition_doc, split_cards_by_query
from app.rag.postprocess.keywords import collect_query_keywords, normalize_text
from app.rag.retriever import retrieve_multi
from app.rag.router import route_query
from app.llm.sllm_refiner import refine_stt_text


LOG_TIMING = os.getenv("RAG_LOG_TIMING", "1") != "0"


# --- 설정 ---
@dataclass(frozen=True)
class RAGConfig:
    top_k: int = 5
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    no_route_answer: str = "카드명/상황을 조금 더 구체적으로 말씀해 주세요."
    include_docs: bool = True
    normalize_keywords: bool = False
    strict_guidance_script: bool = True
    llm_card_top_n: int = 2


# --- 라우팅 ---
def route(query: str) -> Dict[str, Any]:
    return route_query(query)


# --- 안내 스크립트 정제 ---
def _strict_guidance_script(script: str, docs: List[Dict[str, Any]]) -> str:
    if not script:
        return ""
    content = " ".join(doc.get("content") or "" for doc in docs).strip()
    if not content:
        return ""
    normalized_content = normalize_text(content)
    sentences = [s.strip() for s in re.split(r"[.!?\\n]+", script) if s.strip()]
    for sentence in sentences:
        if normalize_text(sentence) not in normalized_content:
            return ""
    return script


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


# --- 검색 ---
async def retrieve(
    query: str,
    routing: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    filters = routing.get("filters") or routing.get("boost") or {}
    route_name = routing.get("route") or routing.get("ui_route")

    sources = set()
    if route_name == "card_info":
        sources.add("card_tbl")
    elif filters.get("card_name"):
        sources.update({"card_tbl", "guide_tbl"})
    if filters.get("payment_method"):
        sources.update({"card_tbl", "guide_tbl"})
    if filters.get("intent") or filters.get("weak_intent"):
        sources.add("guide_tbl")
    if not sources:
        sources.update({"card_tbl", "guide_tbl"})

    return await retrieve_multi(
        query=query,
        routing=routing,
        tables=sorted(sources),
        top_k=top_k,
    )


# --- 답변 생성 ---
# --- 파이프라인 ---
async def run_rag(query: str, config: Optional[RAGConfig] = None) -> Dict[str, Any]:
    cfg = config or RAGConfig()
    t_start = time.perf_counter()
    
    # Refine STT text using sLLM (if enabled)
    refined_query = refine_stt_text(query)
    
    routing = route(refined_query)
    t_route = time.perf_counter()

    should_search = routing.get("should_search")
    if should_search is None:
        should_search = routing.get("should_route")
    if not should_search:
        if LOG_TIMING:
            total = time.perf_counter() - t_start
            print(
                "[rag] "
                f"route={_format_ms(t_route - t_start)} "
                f"total={_format_ms(total)} "
                f"should_search=False route={routing.get('route')}"
            )
        return {
            "currentSituation": [],
            "nextStep": [],
            "guidanceScript": cfg.no_route_answer,
            "routing": routing,
            "meta": {"model": None, "doc_count": 0, "context_chars": 0},
        }

    docs = await retrieve(query=refined_query, routing=routing, top_k=cfg.top_k)
    t_retrieve = time.perf_counter()
    if routing.get("route") == "card_info":
        docs = promote_definition_doc(docs)
    cache_status = "off"
    cards: List[Dict[str, Any]]
    guidance_script: str
    ordered_doc_ids = [doc_cache_id(doc) for doc in docs]
    if CARD_CACHE_ENABLED and cfg.llm_card_top_n > 0:
        cache_key = build_card_cache_key(
            route=routing.get("route") or "",
            model=cfg.model,
            llm_card_top_n=cfg.llm_card_top_n,
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
                query=refined_query,
                docs=docs,
                model=cfg.model,
                temperature=0.0,
                max_llm_cards=cfg.llm_card_top_n,
            )
            await card_cache_set(cache_key, cards, guidance_script)
            cache_status = "miss"
    else:
        cards, guidance_script = generate_detail_cards(
            query=refined_query,
            docs=docs,
            model=cfg.model,
            temperature=0.0,
            max_llm_cards=cfg.llm_card_top_n,
        )
    t_cards = time.perf_counter()
    if cfg.strict_guidance_script:
        guidance_script = _strict_guidance_script(guidance_script, docs)
    query_keywords = collect_query_keywords(refined_query, routing, cfg.normalize_keywords)
    for card in cards:
        card["keywords"] = query_keywords
    cards = [omit_empty(card) for card in cards]
    current_cards, next_cards = split_cards_by_query(cards, refined_query)
    t_post = time.perf_counter()

    if LOG_TIMING:
        total = t_post - t_start
        cache_label = f" cache={cache_status}" if cache_status != "off" else ""
        print(
            "[rag] "
            f"route={_format_ms(t_route - t_start)} "
            f"retrieve={_format_ms(t_retrieve - t_route)} "
            f"cards={_format_ms(t_cards - t_retrieve)} "
            f"post={_format_ms(t_post - t_cards)} "
            f"total={_format_ms(total)} "
            f"docs={len(docs)} route={routing.get('route')}{cache_label}"
        )

    response = {
        "currentSituation": current_cards,
        "nextStep": next_cards,
        "guidanceScript": guidance_script or "",
        "routing": routing,
        "meta": {"model": cfg.model, "doc_count": len(docs), "context_chars": 0},
    }
    if cfg.include_docs:
        response["docs"] = docs
    return response
