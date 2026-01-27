from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging
import time

from pgvector import Vector
from pgvector.psycopg2 import register_vector

from app.rag.retriever.db import _db_conn, _escape_pyformat_percent, embed_query


logger = logging.getLogger(__name__)

_DEFAULT_TEXT_WEIGHT = 0.2


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v]
    return [str(value)]


def _unique(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        if not v:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _collect_category_candidates(routing: Dict[str, Any]) -> List[str]:
    filters = routing.get("filters") or routing.get("boost") or {}
    matched = routing.get("matched") or {}
    candidates = []
    candidates.extend(_as_list(routing.get("consult_category_candidates")))
    candidates.extend(_as_list(filters.get("category")))
    candidates.extend(_as_list(filters.get("intent")))
    candidates.extend(_as_list(filters.get("weak_intent")))
    candidates.extend(_as_list(matched.get("actions")))
    candidates.extend(_as_list(matched.get("weak_intents")))
    return _unique([c.strip() for c in candidates if isinstance(c, str) and c.strip()])


def _build_category_filter(categories: List[str], params: List[object]) -> str:
    if not categories:
        return ""
    exact = categories
    like = [f"%{c}%" for c in categories]
    params.extend([exact, like])
    return "(category = ANY(%s) OR category ILIKE ANY(%s))"


def search_consultation_documents(
    query: str,
    routing: Dict[str, Any],
    top_k: int,
    text_weight: float = _DEFAULT_TEXT_WEIGHT,
) -> List[Dict[str, Any]]:
    categories = _collect_category_candidates(routing)
    emb = Vector(embed_query(query))
    text_query = (query or "").strip()

    def _run_query(apply_category_filter: bool) -> List[Tuple[Any, ...]]:
        category_params: List[object] = []
        where_parts = ["embedding IS NOT NULL"]
        if apply_category_filter:
            category_clause = _build_category_filter(categories, category_params)
            if category_clause:
                where_parts.append(category_clause)
        where_sql = " WHERE " + " AND ".join(where_parts)
        params: List[object] = [emb, text_query, *category_params, emb, top_k]
        sql = (
            "SELECT id, consultation_id, title, content, category, metadata, usage_count, "
            "effectiveness_score, "
            "1 - (embedding <=> %s) AS vscore, "
            "ts_rank_cd("
            "to_tsvector('simple', COALESCE(title, '') || ' ' || COALESCE(content, '')),"
            "plainto_tsquery('simple', %s)"
            ") AS tscore "
            "FROM consultation_documents"
            f"{where_sql} "
            "ORDER BY (embedding <=> %s) ASC "
            "LIMIT %s"
        )
        with _db_conn() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(_escape_pyformat_percent(sql), params)
                return cur.fetchall()

    start = time.perf_counter()
    rows: List[Tuple[Any, ...]] = _run_query(apply_category_filter=True)
    if categories and not rows:
        rows = _run_query(apply_category_filter=False)
    if not rows:
        params: List[object] = [text_query, top_k]
        sql = (
            "SELECT id, consultation_id, title, content, category, metadata, usage_count, "
            "effectiveness_score, "
            "0.0 AS vscore, "
            "ts_rank_cd("
            "to_tsvector('simple', COALESCE(title, '') || ' ' || COALESCE(content, '')),"
            "plainto_tsquery('simple', %s)"
            ") AS tscore "
            "FROM consultation_documents "
            "WHERE COALESCE(title, '') <> '' OR COALESCE(content, '') <> '' "
            "ORDER BY tscore DESC NULLS LAST "
            "LIMIT %s"
        )
        with _db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_escape_pyformat_percent(sql), params)
                rows = cur.fetchall()
    exec_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "[consult_retriever] vector+text exec_ms=%.1f rows=%d categories=%d",
        exec_ms,
        len(rows),
        len(categories),
    )

    docs: List[Dict[str, Any]] = []
    for (
        doc_id,
        consultation_id,
        title,
        content,
        category,
        metadata,
        usage_count,
        effectiveness_score,
        vscore,
        tscore,
    ) in rows:
        meta = metadata if isinstance(metadata, dict) else {}
        meta = {
            **meta,
            "title": title,
            "category": category,
            "consultation_id": consultation_id,
            "usage_count": usage_count,
            "effectiveness_score": effectiveness_score,
        }
        vscore = float(vscore or 0.0)
        tscore = float(tscore or 0.0)
        combined = (1.0 - text_weight) * vscore + text_weight * min(tscore, 1.0)
        docs.append(
            {
                "id": str(doc_id),
                "db_id": doc_id,
                "title": title,
                "content": content or "",
                "metadata": meta,
                "table": "consultation_documents",
                "score": combined,
                "vector_score": vscore,
                "text_score": tscore,
            }
        )
    return docs


__all__ = ["search_consultation_documents"]
