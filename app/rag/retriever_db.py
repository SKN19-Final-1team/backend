from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import os
import threading

import psycopg2
from psycopg2 import pool as pg_pool
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg2 import register_vector

from app.llm.base import get_openai_client
from app.rag.retriever_terms import (
    _as_list,
    _expand_action_terms,
    _expand_card_terms,
    _expand_payment_terms,
    _extract_query_terms,
    _unique_in_order,
)

load_dotenv()

_DB_POOL_ENABLED = os.getenv("RAG_DB_POOL", "1") != "0"
_TRGM_ENABLED = os.getenv("RAG_TRGM_RANK", "1") != "0"
_TRGM_MAX_TERMS = int(os.getenv("RAG_TRGM_MAX_TERMS", "3"))
_DB_POOL: Optional[pg_pool.ThreadedConnectionPool] = None
_DB_POOL_LOCK = threading.Lock()
CARD_TABLES = {"card_tbl", "card_products"}
GUIDE_TABLES = {"guide_tbl", "service_guide_documents"}
TABLE_ALIASES = {"card_tbl": "card_products", "guide_tbl": "service_guide_documents"}


def _is_card_table(name: str) -> bool:
    return name in CARD_TABLES


def _is_guide_table(name: str) -> bool:
    return name in GUIDE_TABLES


def _resolve_table(name: str) -> str:
    return TABLE_ALIASES.get(name, name)


def _db_config() -> Dict[str, object]:
    host = os.getenv("DB_HOST_IP") or os.getenv("DB_HOST")
    cfg = {
        "host": host,
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", "0")) or None,
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise ValueError(f"Missing DB settings: {missing}")
    return cfg


def _db_pool() -> Optional[pg_pool.ThreadedConnectionPool]:
    global _DB_POOL
    if not _DB_POOL_ENABLED:
        return None
    if _DB_POOL is None:
        with _DB_POOL_LOCK:
            if _DB_POOL is None:
                minconn = int(os.getenv("RAG_DB_POOL_MIN", "1"))
                maxconn = int(os.getenv("RAG_DB_POOL_MAX", "4"))
                _DB_POOL = pg_pool.ThreadedConnectionPool(minconn, maxconn, **_db_config())
    return _DB_POOL


@contextmanager
def _db_conn():
    db_pool = _db_pool()
    if db_pool is None:
        conn = psycopg2.connect(**_db_config())
        try:
            yield conn
        finally:
            conn.close()
        return
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        try:
            conn.rollback()
        finally:
            db_pool.putconn(conn)


def _safe_table(name: str) -> str:
    if name not in CARD_TABLES | GUIDE_TABLES:
        raise ValueError(f"Unsupported table: {name}")
    return name


def embed_query(text: str, model: str = "text-embedding-3-small") -> List[float]:
    client = get_openai_client()
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def _source_sql(table: str, include_embedding: bool) -> str:
    actual = _resolve_table(table)
    if actual == "card_products":
        content_expr = (
            "COALESCE(name, '')"
            " || E'\\n\\n' || COALESCE(main_benefits, '')"
            " || E'\\n\\n' || COALESCE(performance_condition, '')"
            " || E'\\n\\n' || COALESCE(metadata->>'full_content', '')"
        )
        metadata_expr = (
            "COALESCE(metadata, '{}'::jsonb) || jsonb_build_object("
            "'title', name, "
            "'card_name', name, "
            "'category', card_type::text, "
            "'category1', card_type::text, "
            "'category2', brand::text, "
            "'source_table', 'card_products'"
            ")"
        )
        embedding_expr = "NULL::vector(1536) AS embedding"
    elif actual == "service_guide_documents":
        content_expr = "content"
        metadata_expr = (
            "COALESCE(metadata, '{}'::jsonb) || jsonb_build_object("
            "'title', title, "
            "'category', category, "
            "'category1', document_type, "
            "'source_table', 'service_guide_documents'"
            ")"
        )
        embedding_expr = "embedding"
    else:
        content_expr = "content"
        metadata_expr = "metadata"
        embedding_expr = "embedding"
    select_parts = ["id", f"{content_expr} AS content", f"{metadata_expr} AS metadata"]
    if include_embedding:
        select_parts.append(embedding_expr)
    return f"SELECT {', '.join(select_parts)} FROM {actual}"


def fetch_docs_by_ids(table: str, ids: List[str]) -> List[Dict[str, object]]:
    if not ids:
        return []
    safe_table = _safe_table(table)
    sql = _source_sql(safe_table, include_embedding=False) + " WHERE id = ANY(%s)"
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (ids,))
            rows = cur.fetchall()
    docs: List[Dict[str, object]] = []
    for doc_id, content, metadata in rows:
        meta = metadata if isinstance(metadata, dict) else {}
        title = meta.get("title") or meta.get("name") or meta.get("card_name")
        docs.append(
            {
                "id": str(doc_id),
                "db_id": doc_id,
                "title": title,
                "content": content or "",
                "metadata": meta,
                "table": safe_table,
            }
        )
    return docs


def _build_like_group(terms: List[str], params: List[str]) -> Optional[str]:
    if not terms:
        return None
    term_clauses = []
    for term in terms:
        term_clauses.append("(content ILIKE %s OR metadata->>'title' ILIKE %s)")
        params.extend([f"%{term}%", f"%{term}%"])
    return "(" + " OR ".join(term_clauses) + ")"


def build_where_clause(
    filters: Optional[Dict[str, object]],
    table: str,
) -> Tuple[str, List[str]]:
    filters = filters or {}
    clauses: List[str] = []
    params: List[str] = []
    has_category = False

    for key in ("category1", "category2"):
        values = _as_list(filters.get(key))
        if not values:
            continue
        placeholders = ", ".join(["%s"] * len(values))
        clauses.append(f"metadata->>'{key}' IN ({placeholders})")
        params.extend(values)
        has_category = True

    category_terms = _as_list(filters.get("category"))
    if category_terms and _is_card_table(table):
        term_clauses = []
        for term in category_terms:
            term_clauses.append(
                "(metadata->>'category' ILIKE %s OR metadata->>'category1' ILIKE %s OR metadata->>'category2' ILIKE %s)"
            )
            params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
        if term_clauses:
            clauses.append("(" + " OR ".join(term_clauses) + ")")
            has_category = True

    card_values = _as_list(filters.get("card_name"))
    card_terms = _expand_card_terms(card_values)
    intent_terms = _expand_action_terms(_as_list(filters.get("intent")))
    payment_terms = _expand_payment_terms(_as_list(filters.get("payment_method")))
    has_intent = bool(intent_terms) or bool(_as_list(filters.get("weak_intent")))

    if _is_guide_table(table):
        pass
    else:
        card_meta_clause = None
        if card_values:
            term_clauses = []
            for value in card_values:
                value_str = str(value)
                value_norm = value_str.replace(" ", "")
                term_clauses.append(
                    "(replace(metadata->>'card_name', ' ', '') ILIKE %s OR metadata->>'card_name' ILIKE %s)"
                )
                params.extend([f"%{value_norm}%", f"%{value_str}%"])
            if term_clauses:
                card_meta_clause = "(" + " OR ".join(term_clauses) + ")"
        card_group = None
        payment_group = None
        if not card_meta_clause:
            card_group = _build_like_group(card_terms, params)
            if not card_group:
                payment_only = payment_terms and not has_intent and not has_category
                if payment_only:
                    payment_group = None
                else:
                    payment_group = _build_like_group(payment_terms, params)
        if card_meta_clause:
            clauses.append(card_meta_clause)
        elif card_group:
            clauses.append(card_group)
        if payment_group:
            clauses.append(payment_group)

    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params


def vector_search(
    query: str,
    table: str,
    limit: int,
    filters: Optional[Dict[str, object]] = None,
) -> List[Tuple[object, str, Dict[str, object], float]]:
    table = _safe_table(table)
    actual_table = _resolve_table(table)
    if actual_table == "card_products":
        terms = _extract_query_terms(query)
        if not terms and query.strip():
            terms = [query.strip()]
        return text_search(table=table, terms=terms, limit=limit, filters=filters)
    emb = Vector(embed_query(query))
    where_sql, where_params = build_where_clause(filters, table)
    with _db_conn() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            def _run(where_sql: str, where_params: List[str]):
                sql = (
                    "WITH source AS ("
                    f"{_source_sql(table, include_embedding=True)}"
                    ") "
                    "SELECT id, content, metadata, 1 - (embedding <=> %s) AS score "
                    f"FROM source{where_sql} ORDER BY embedding <=> %s LIMIT %s"
                )
                params = [emb, *where_params, emb, limit]
                try:
                    cur.execute(sql, params)
                except Exception:
                    conn.rollback()
                    sql = (
                        "WITH source AS ("
                        f"{_source_sql(table, include_embedding=True)}"
                        ") "
                        "SELECT id, content, metadata, 1 - (embedding <-> %s) AS score "
                        f"FROM source{where_sql} ORDER BY embedding <-> %s LIMIT %s"
                    )
                    cur.execute(sql, params)
                return cur.fetchall()

            results = _run(where_sql, where_params)
            if not results and where_sql and filters:
                results = _run("", [])
            return results


def text_search(
    table: str,
    terms: List[str],
    limit: int,
    filters: Optional[Dict[str, object]] = None,
) -> List[Tuple[object, str, Dict[str, object], float]]:
    if not terms:
        return []
    table = _safe_table(table)
    filters = filters or {}
    terms = _unique_in_order([t for t in terms if t])
    if _TRGM_MAX_TERMS > 0:
        terms = terms[:_TRGM_MAX_TERMS]
    def _build_term_clauses(items: List[str]):
        trgm_params: List[str] = []
        like_params: List[str] = []
        trgm_clauses = []
        like_clauses = []
        for term in items:
            trgm_clauses.append(
                "("
                "COALESCE(content, '') % %s OR "
                "COALESCE(metadata->>'title', '') % %s OR "
                "COALESCE(metadata->>'category', '') % %s OR "
                "COALESCE(metadata->>'category1', '') % %s OR "
                "COALESCE(metadata->>'category2', '') % %s"
                ")"
            )
            trgm_params.extend([term] * 5)
            like_clauses.append(
                "("
                "content ILIKE %s OR "
                "metadata->>'title' ILIKE %s OR "
                "metadata->>'category' ILIKE %s OR "
                "metadata->>'category1' ILIKE %s OR "
                "metadata->>'category2' ILIKE %s"
                ")"
            )
            like_params.extend([f"%{term}%"] * 5)
        trgm_where = "(" + " OR ".join(trgm_clauses) + ")" if trgm_clauses else ""
        like_where = "(" + " OR ".join(like_clauses) + ")" if like_clauses else ""
        return trgm_where, like_where, trgm_params, like_params

    trgm_where, like_where, trgm_params, like_params = _build_term_clauses(terms)
    if _is_card_table(table):
        card_values = _as_list(filters.get("card_name"))
        if card_values:
            placeholders = ", ".join(["%s"] * len(card_values))
            if trgm_where:
                trgm_where = f"({trgm_where}) AND metadata->>'card_name' IN ({placeholders})"
                trgm_params.extend(card_values)
            if like_where:
                like_where = f"({like_where}) AND metadata->>'card_name' IN ({placeholders})"
                like_params.extend(card_values)
    if not trgm_where and not like_where:
        return []

    score_cols = (
        "content",
        "metadata->>'title'",
        "metadata->>'category'",
        "metadata->>'category1'",
        "metadata->>'category2'",
    )
    score_params: List[str] = []
    score_parts = []
    if _TRGM_ENABLED and terms:
        for term in terms:
            score_parts.append(
                "GREATEST(" + ", ".join([f"similarity(COALESCE({col}, ''), %s)" for col in score_cols]) + ")"
            )
            score_params.extend([term] * len(score_cols))
    score_expr = " + ".join(score_parts) if score_parts else "0.0"
    with _db_conn() as conn:
        with conn.cursor() as cur:
            def _run(where_sql: str, where_params: List[str]):
                sql = (
                    "WITH source AS ("
                    f"{_source_sql(table, include_embedding=False)}"
                    ") "
                    f"SELECT id, content, metadata, {score_expr} AS score "
                    f"FROM source WHERE {where_sql} "
                    "ORDER BY score DESC LIMIT %s"
                )
                params = [*score_params, *where_params, limit]
                cur.execute(sql, params)
                return cur.fetchall()

            results: List[Tuple[object, str, Dict[str, object], float]] = []
            if _TRGM_ENABLED and trgm_where:
                try:
                    results = _run(trgm_where, trgm_params)
                except Exception:
                    conn.rollback()
                    results = []
            if not results and like_where:
                results = _run(like_where, like_params)
            return results
