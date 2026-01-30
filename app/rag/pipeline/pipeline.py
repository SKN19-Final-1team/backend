from typing import Any, Dict, Optional

from app.llm.guide_pipeline import build_guidance_response
from app.rag.pipeline.config import RAGConfig
from app.rag.pipeline.search import run_search

# --- sLLM을 사용한 텍스트 교정 및 키워드 추출 ---
# NOTE: sLLM 적용은 잠시 비활성화(주석 처리) 상태.
# from app.llm.sllm_refiner import refine_text


async def run_rag(
    query: str,
    config: Optional[RAGConfig] = None,
    session_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or RAGConfig()
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
            "guidanceScript": search.no_search_message or cfg.no_route_answer,
            "routing": search.routing,
            "meta": {"model": None, "doc_count": 0, "context_chars": 0},
        }
    return await build_guidance_response(
        query=query,
        routing=search.routing,
        docs=search.docs,
        consult_docs=search.consult_docs,
        config=cfg,
        t_start=search.t_start,
        t_route=search.t_route,
        t_retrieve=search.t_retrieve,
        retrieve_cache_status=search.retrieve_cache_status,
    )
