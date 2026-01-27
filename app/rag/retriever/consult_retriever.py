from __future__ import annotations

from typing import Any, Dict, List

from app.rag.retriever.consult_cases import search_consultation_documents


def retrieve_consult_docs(
    query_text: str,
    intent: str | None,
    categories: List[str] | None,
    top_k: int,
) -> List[Dict[str, Any]]:
    routing = {
        "matched": {"actions": [intent] if intent else []},
        "consult_category_candidates": categories or [],
    }
    return search_consultation_documents(query=query_text, routing=routing, top_k=top_k)


__all__ = ["retrieve_consult_docs"]
