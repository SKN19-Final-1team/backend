from __future__ import annotations

from typing import Any, Dict, List
import asyncio
import os
import time

from app.guide.guide_client import get_guide_model_name
from app.guide.guide_generator import generate_guide_message
from app.rag.postprocess.sections import clean_card_docs
from app.rag.pipeline.utils import format_ms

LOG_TIMING = os.getenv("RAG_LOG_TIMING", "1") != "0"


async def build_guide_response(
    *,
    query: str,
    docs: List[Dict[str, Any]],
    consult_docs: List[Dict[str, Any]],
    routing: Dict[str, Any],
    t_start: float,
    t_route: float,
    t_retrieve: float,
) -> Dict[str, Any]:
    guide_start = time.perf_counter()
    docs = clean_card_docs(docs, query)
    message = await asyncio.to_thread(generate_guide_message, query, docs, consult_docs)
    guide_end = time.perf_counter()

    if LOG_TIMING:
        total = guide_end - t_start
        print(
            "[guide] "
            f"route={format_ms(t_route - t_start)} "
            f"retrieve={format_ms(t_retrieve - t_route)} "
            f"guide={format_ms(guide_end - guide_start)} "
            f"total={format_ms(total)} "
            f"docs={len(docs)} route={routing.get('route')}"
        )

    return {
        "guidanceScript": message or "",
        "guide_script": {"message": message or ""},
        "meta": {"guide_model": get_guide_model_name()},
    }


__all__ = ["build_guide_response"]
