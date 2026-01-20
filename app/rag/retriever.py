from typing import Dict, List, Optional
import os

from app.rag.router import route_query as _route_query
from app.rag.retriever_db import _is_card_table, _safe_table, text_search, vector_search
from app.rag.retriever_rank import _collect_candidates, _finalize_candidates
from app.rag.retriever_terms import _build_search_context, _priority_terms

LOG_RETRIEVER_DEBUG = os.getenv("RAG_LOG_RETRIEVER_DEBUG") == "1"


# Router entry

def route_query(query: str) -> Dict[str, Optional[object]]:
    return _route_query(query)


def _fetch_k(top_k: int) -> int:
    return max(top_k * 3, top_k + 5)


async def retrieve_docs(
    query: str,
    routing: Dict[str, object],
    top_k: int = 5,
    table: Optional[str] = None,
    allow_fallback: bool = True,
) -> List[Dict[str, object]]:
    tables = ["card_products", "service_guide_documents"] if table is None else [_safe_table(table)]
    if allow_fallback and table in (None, "card_products", "card_tbl"):
        if "service_guide_documents" not in tables and "guide_tbl" not in tables:
            tables.append("service_guide_documents")
    return await retrieve_multi(query=query, routing=routing, tables=tables, top_k=top_k)


async def retrieve_multi(
    query: str,
    routing: Dict[str, object],
    tables: List[str],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    context = _build_search_context(query, routing)
    fetch_k = _fetch_k(top_k)
    candidates: List[tuple[int, float, Dict[str, object]]] = []

    def _fetch_rows(
        safe_table: str,
    ) -> List[tuple[object, str, Dict[str, object], float]]:
        rows: List[tuple[object, str, Dict[str, object], float]] = []
        if _is_card_table(safe_table) and context.category_terms:
            category_filters = dict(context.filters)
            category_filters["category"] = context.category_terms
            rows = vector_search(
                context.query_text,
                table=safe_table,
                limit=fetch_k,
                filters=category_filters,
            )
            if not rows:
                rows = vector_search(
                    context.query_text,
                    table=safe_table,
                    limit=fetch_k,
                    filters=context.filters,
                )
        else:
            rows = vector_search(
                context.query_text,
                table=safe_table,
                limit=fetch_k,
                filters=context.filters,
            )
        if _is_card_table(safe_table) and context.category_terms and context.search_mode in {"ISSUE", "BENEFIT"}:
            priority_terms = _priority_terms(context.category_terms)
            if priority_terms:
                rows.extend(
                    text_search(
                        table=safe_table,
                        terms=priority_terms,
                        limit=fetch_k,
                        filters=context.filters,
                    )
                )
        if _is_card_table(safe_table) and context.card_terms and not context.intent_terms and not context.payment_terms and not context.category_terms:
            loose_filters = dict(context.filters)
            loose_filters.pop("card_name", None)
            loose_rows = vector_search(context.query_text, table=safe_table, limit=fetch_k, filters=loose_filters)
            if loose_rows:
                rows.extend(loose_rows)
        return rows

    for table in tables:
        safe_table = _safe_table(table)
        rows = _fetch_rows(safe_table)
        table_candidates = _collect_candidates(safe_table, rows, context, fetch_k)
        candidates.extend(table_candidates)
        if LOG_RETRIEVER_DEBUG:
            print(
                "[retriever] "
                f"table={safe_table} vec_rows={len(rows)} "
                f"cand_added={len(table_candidates)} total_cand={len(candidates)}"
            )

    def _doc_key(doc: Dict[str, object]) -> str:
        title = doc.get("title")
        return title if title else f"__no_title__{doc.get('table')}:{doc.get('id')}"

    docs = _finalize_candidates(candidates, _doc_key)
    if LOG_RETRIEVER_DEBUG and docs:
        top = docs[0]
        print(
            "[retriever] "
            f"top_title={top.get('title')} "
            f"score={top.get('score')} rrf={top.get('rrf_score')} "
            f"title_score={top.get('title_score')}"
        )
    return docs[:top_k]
