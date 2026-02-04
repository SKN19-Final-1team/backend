from typing import Any, Dict, Optional
import asyncio
import os

from app.guide.guide_pipeline import build_guide_response
from app.rag.pipeline.config import RAGConfig
from app.rag.pipeline.card_pipeline import build_card_response
from app.rag.pipeline.search import run_search
from app.rag.cache.doc_title_cache import record_doc_titles
from app.rag.router.signals import has_vocab_match

async def run_rag(
    query: str,
    config: Optional[RAGConfig] = None,
    session_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or RAGConfig()
    require_vocab_match = os.getenv("RAG_REQUIRE_VOCAB_MATCH", "1") != "0"
    if require_vocab_match and not has_vocab_match(query):
        return {
            "currentSituation": [],
            "nextStep": [],
            "guidanceScript": "",
            "guide_script": {"message": ""},
            "routing": {
                "should_search": False,
                "should_route": False,
                "should_trigger": False,
                "route": None,
                "ui_route": None,
            },
            "meta": {"model": None, "doc_count": 0, "context_chars": 0},
        }
    search = await run_search(
        query,
        top_k=cfg.top_k,
        enable_consult_search=cfg.enable_consult_search,
        session_state=session_state,
    )
    if not search.should_search:
        return {
            "currentSituation": [],
            "nextStep": [],
            "guidanceScript": "",
            "guide_script": {"message": ""},
            "routing": search.routing,
            "meta": {"model": None, "doc_count": 0, "context_chars": 0},
        }

    record_doc_titles(search.docs)

    card_routing = dict(search.routing)
    guide_routing = dict(search.routing)

    card_task = asyncio.create_task(
        build_card_response(
            query=query,
            routing=card_routing,
            docs=search.docs,
            config=cfg,
            t_start=search.t_start,
            t_route=search.t_route,
            t_retrieve=search.t_retrieve,
            retrieve_cache_status=search.retrieve_cache_status,
        )
    )
    guide_task = asyncio.create_task(
        build_guide_response(
            query=query,
            routing=guide_routing,
            docs=search.docs,
            consult_docs=search.consult_docs,
            t_start=search.t_start,
            t_route=search.t_route,
            t_retrieve=search.t_retrieve,
        )
    )

    card_result, guide_result = await asyncio.gather(card_task, guide_task, return_exceptions=True)

    if isinstance(card_result, Exception):
        card_result = {
            "currentSituation": [],
            "nextStep": [],
            "routing": card_routing,
            "meta": {"model": cfg.model, "doc_count": len(search.docs), "context_chars": 0},
        }
    if isinstance(guide_result, Exception):
        guide_result = {"guidanceScript": "", "guide_script": {"message": ""}, "meta": {}}

    response = {
        "currentSituation": card_result.get("currentSituation", []),
        "nextStep": card_result.get("nextStep", []),
        "guidanceScript": guide_result.get("guidanceScript", ""),
        "guide_script": guide_result.get("guide_script", {"message": ""}),
        "routing": card_result.get("routing", card_routing),
        "meta": card_result.get("meta", {"model": cfg.model, "doc_count": len(search.docs), "context_chars": 0}),
    }
    if cfg.include_docs:
        response["docs"] = search.docs
        if getattr(cfg, "include_consult_docs", False):
            response["consult_docs"] = search.consult_docs
    return response
