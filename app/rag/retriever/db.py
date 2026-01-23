from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import logging
import os
import threading
import time

import psycopg2
from psycopg2 import pool as pg_pool
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg2 import register_vector

from app.llm.base import get_openai_client
from app.rag.common.doc_source_filters import ALLOWED_SCOPE_FILTERS, DOC_SOURCE_FILTERS
from app.rag.retriever.terms import (
    _as_list,
    _expand_action_terms,
    _expand_card_terms,
    _expand_guide_terms,
    _expand_payment_terms,
    _extract_query_terms,
    _unique_in_order,
)

logger = logging.getLogger(__name__)


def _escape_pyformat_percent(sql: str) -> str:
    """Escape literal '%' characters so psycopg2 does not treat them as placeholders."""
    escaped: list[str] = []
    i = 0
    length = len(sql)
    while i < length:
        ch = sql[i]
        if ch == "%":
            if i + 1 < length and sql[i + 1] == "s":
                escaped.append("%s")
                i += 2
                continue
            escaped.append("%%")
            i += 1
            continue
        escaped.append(ch)
        i += 1
    return "".join(escaped)

load_dotenv()

_DB_POOL_ENABLED = os.getenv("RAG_DB_POOL", "1") != "0"
_TRGM_ENABLED = os.getenv("RAG_TRGM_RANK", "1") != "0"
_TRGM_MAX_TERMS = int(os.getenv("RAG_TRGM_MAX_TERMS", "3"))
_TRGM_MIN_LEN = int(os.getenv("RAG_TRGM_MIN_LEN", "3"))
_EXPLAIN_ENABLED = os.getenv("RAG_ENABLE_EXPLAIN", "0") == "1"
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


def _is_scope_filter_allowed(scope_filter: Optional[object]) -> bool:
    if not scope_filter:
        return False
    return str(scope_filter) in ALLOWED_SCOPE_FILTERS


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


def _build_title_like_group(terms: List[str], params: List[str]) -> Optional[str]:
    if not terms:
        return None
    term_clauses = []
    for term in terms:
        term_clauses.append("(metadata->>'title' ILIKE %s)")
        params.append(f"%{term}%")
    return "(" + " OR ".join(term_clauses) + ")"


def build_where_clause(
    filters: Optional[Dict[str, object]],
    table: str,
) -> Tuple[str, List[str]]:
    filters = filters or {}
    clauses: List[str] = []
    params: List[str] = []
    has_category = False

    # 스코프 필터 (문서 소스별 제한)
    scope_filter = filters.get("_scope_filter")
    filter_keys = list(filters.keys())
    logger.debug(
        "build_where_clause table=%s scope_filter_present=%s filter_keys=%s",
        table,
        bool(scope_filter),
        filter_keys,
    )
    if scope_filter:
        if not _is_scope_filter_allowed(scope_filter):
            raise ValueError(f"Unsupported scope filter: {scope_filter}")
        clauses.append(str(scope_filter))

    # APPLEPAY: ID 프리픽스 필터 (애플페이 문서만 검색)
    id_prefix = filters.get("id_prefix")
    if id_prefix:
        clauses.append("id LIKE %s")
        params.append(f"{id_prefix}%")

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
    intent_values = _as_list(filters.get("intent"))
    weak_values = _as_list(filters.get("weak_intent"))
    card_terms = _expand_card_terms(card_values)
    intent_terms = _expand_action_terms(intent_values)
    payment_terms = _expand_payment_terms(_as_list(filters.get("payment_method")))
    weak_terms = _expand_action_terms(weak_values)
    has_intent = bool(intent_terms) or bool(weak_terms)

    if _is_guide_table(table):

        guide_terms = _expand_guide_terms(_unique_in_order([*intent_values, *weak_values]))
        guide_group = _build_like_group(guide_terms, params)
        if guide_group:
            clauses.append(guide_group)
        if filters.get("exclude_card_specific"):
            clauses.append(
                "COALESCE(metadata->>'original_card_name', '') = '' "
                "AND COALESCE(metadata->>'card_name', '') = ''"
            )
    else:
        card_meta_clause = None
        if card_values:
            # 모든 card_values를 ILIKE ANY로 한 번에 처리
            norm_terms = [f"%{str(v).replace(' ', '')}%" for v in card_values]
            norm_all = [f"%{str(v)}%" for v in card_values]
            card_meta_clause = (
                "(replace(metadata->>'card_name', ' ', '') ILIKE ANY(%s) OR "
                "metadata->>'card_name' ILIKE ANY(%s))"
            )
            params.extend([norm_terms, norm_all])
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
    # card_products는 embedding이 없으므로 text_search로 처리
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
                logger.debug("[vector_search] FINAL_SQL where_clause: %s", where_sql)
                logger.debug("[vector_search] FINAL_PARAMS: %s", params)
                def _execute(sql_text: str, params_list: List[object]) -> List[tuple]:
                    exec_start = time.perf_counter()
                    cur.execute(_escape_pyformat_percent(sql_text), params_list)
                    exec_ms = (time.perf_counter() - exec_start) * 1000
                    rows = cur.fetchall()
                    print(f"[retriever_db] table={table} mode=vector exec_ms={exec_ms:.1f} rows={len(rows)}")
                    return rows

                try:
                    return _execute(sql, params)
                except Exception:
                    conn.rollback()
                    sql = (
                        "WITH source AS ("
                        f"{_source_sql(table, include_embedding=True)}"
                        ") "
                        "SELECT id, content, metadata, 1 - (embedding <-> %s) AS score "
                        f"FROM source{where_sql} ORDER BY embedding <-> %s LIMIT %s"
                    )
                    return _execute(sql, params)

            results = _run(where_sql, where_params)
            if not results and where_sql and filters:
                card_values = _as_list(filters.get("card_name"))
                intent_values = _as_list(filters.get("intent"))
                weak_values = _as_list(filters.get("weak_intent"))
                intent_only = _is_guide_table(table) and (intent_values or weak_values) and not card_values
                if intent_only:
                    fallback_terms = _expand_guide_terms(_unique_in_order([*intent_values, *weak_values]))
                    fallback_params: List[str] = []
                    fallback_group = _build_title_like_group(fallback_terms, fallback_params)
                    if fallback_group:
                        results = _run(" WHERE " + fallback_group, fallback_params)
                else:
                    results = _run("", [])
            return results


def _and_conditions(where: str, condition: str) -> str:
    """조건을 AND로 결합 (값 없음)"""
    if not where:
        return condition
    return f"({where}) AND {condition}"


def _and_param_condition(where: str, condition: str, param: object, params: List) -> str:
    """조건을 AND로 결합 (파라미터 추가)"""
    params.append(param)
    return _and_conditions(where, condition)


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
    scope_filter = (filters or {}).get("_scope_filter")
    guide_with_terms_filter = DOC_SOURCE_FILTERS.get("guide_with_terms")
    use_trgm = _TRGM_ENABLED and not _is_card_table(table)
    if scope_filter and guide_with_terms_filter and str(scope_filter) == guide_with_terms_filter:
        use_trgm = False
    terms = _unique_in_order([t for t in terms if t])
    trgm_terms = [t for t in terms if len(t) >= _TRGM_MIN_LEN] if use_trgm else []
    if use_trgm and _TRGM_MAX_TERMS > 0:
        trgm_terms = trgm_terms[:_TRGM_MAX_TERMS]

    def _build_term_clauses(trgm_items: List[str], like_items: List[str]):
        """
        WHERE 절 조건 생성
        - trgm: PostgreSQL trigram 연산자 (%)
        - like: ILIKE 연산자
        """
        trgm_params: List[str] = []
        like_params: List[str] = []
        trgm_clauses: List[str] = []
        like_clauses: List[str] = []
        
        # Trigram 검색 (각 term)
        for term in trgm_items:
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
        
        # ILIKE 검색 (각 term)
        for term in like_items:
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

    trgm_where, like_where, trgm_params, like_params = _build_term_clauses(trgm_terms, terms)
    # 카드 테이블(card_products 등) 2단계 검색: 1차 card_name 인덱스 기반 후보 추출, 2차 본문 검색
    if _is_card_table(table):
        actual_table = _resolve_table(table)
        card_values = _as_list(filters.get("card_name"))
        if card_values:
            # 1차: normalized card_name 인덱스 기반 후보 추출
            eq_any = [str(v).replace(' ', '').lower() for v in card_values]
            prefix_any = [f"{str(v)}%" for v in card_values]
            # 후보 추출 쿼리
            id_candidates = []
            with _db_conn() as conn:
                with conn.cursor() as cur:
                    sql = (
                        "SELECT id FROM " + actual_table +
                        " WHERE LOWER(REPLACE(metadata->>'card_name',' ','')) = ANY(%s) "
                        "OR metadata->>'card_name' LIKE ANY(%s) LIMIT 20"
                    )
                    cur.execute(sql, [eq_any, prefix_any])
                    id_candidates = [row[0] for row in cur.fetchall()]
            if not id_candidates:
                like_any = [f"%{str(v)}%" for v in card_values]
                no_space_any = [f"%{str(v).replace(' ', '')}%" for v in card_values]
                with _db_conn() as conn:
                    with conn.cursor() as cur:
                        sql = (
                            "SELECT id FROM " + actual_table +
                            " WHERE (replace(metadata->>'card_name',' ','') ILIKE ANY(%s) OR metadata->>'card_name' ILIKE ANY(%s)) LIMIT 20"
                        )
                        cur.execute(sql, [no_space_any, like_any])
                        id_candidates = [row[0] for row in cur.fetchall()]
            # 후보가 없으면 terms 기반 본문(content) 검색을 추가로 시도
            if not id_candidates:
                if terms:
                    # terms 기반 본문 검색 (카드명 없이)
                    terms_clause = " OR ".join(["content ILIKE %s" for _ in terms])
                    params = [f"%{t}%" for t in terms]
                    source_sql = (
                        "SELECT id, "
                        "COALESCE(name, '') || E'\n\n' || COALESCE(main_benefits, '') || E'\n\n' || COALESCE(performance_condition, '') AS content, "
                        "metadata FROM " + actual_table
                    )
                    with _db_conn() as conn:
                        with conn.cursor() as cur:
                            sql = (
                                "WITH source AS (" + source_sql + ") "
                                "SELECT id, content, metadata, 0.0 AS score FROM source "
                                "WHERE " + terms_clause +
                                " LIMIT %s"
                            )
                            params.append(limit)
                            cur.execute(sql, params)
                            rows = cur.fetchall()
                    return rows
                else:
                    return []
            # 2차: 본문(content/main_benefits) 검색 (후보 20개 내외)
            params = [id_candidates]
            like_clause = ""
            if terms:
                like_clause = " WHERE " + " OR ".join(["content ILIKE %s" for _ in terms])
                params.extend([f"%{t}%" for t in terms])
            source_sql = (
                "SELECT id, "
                "COALESCE(name, '') || E'\n\n' || COALESCE(main_benefits, '') || E'\n\n' || COALESCE(performance_condition, '') AS content, "
                "metadata FROM " + actual_table +
                " WHERE id = ANY(%s)"
            )
            with _db_conn() as conn:
                with conn.cursor() as cur:
                    sql = (
                        "WITH source AS (" + source_sql + ") "
                        "SELECT id, content, metadata, 0.0 AS score FROM source"
                        + like_clause +
                        " LIMIT %s"
                    )
                    params.append(limit)
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    if not rows and like_clause:
                        # terms 필터로 모두 걸러진 경우 id 후보 전체를 반환
                        cur.execute(
                            "WITH source AS (" + source_sql + ") SELECT id, content, metadata, 0.0 AS score FROM source LIMIT %s",
                            [id_candidates, limit],
                        )
                        rows = cur.fetchall()
            return rows
    
    id_prefix = filters.get("id_prefix")
    if id_prefix:
        id_prefix_str = str(id_prefix)
        id_prefix_condition = f"id LIKE '{id_prefix_str}%'"
        if use_trgm and trgm_where:
            trgm_where = _and_conditions(trgm_where, id_prefix_condition)
        elif use_trgm and trgm_where == "":
            trgm_where = id_prefix_condition
        if like_where:
            like_where = _and_conditions(like_where, id_prefix_condition)
        else:
            like_where = id_prefix_condition
    
    if scope_filter:
        scope_filter_sql = str(scope_filter)
        if use_trgm and trgm_where:
            trgm_where = _and_conditions(trgm_where, scope_filter_sql)
        elif use_trgm and trgm_where == "":
            trgm_where = scope_filter_sql
        if like_where:
            like_where = _and_conditions(like_where, scope_filter_sql)
        else:
            like_where = scope_filter_sql
    
    if not trgm_where and not like_where:
        return []

    has_trgm = bool(trgm_where)
    has_like = bool(like_where)
    if has_trgm and has_like:
        submode = "trgm_like"
    elif has_trgm:
        submode = "trgm"
    else:
        submode = "like"

    score_cols = (
        "content",
        "metadata->>'title'",
        "metadata->>'category'",
        "metadata->>'category1'",
        "metadata->>'category2'",
    )
    score_params: List[str] = []
    score_parts = []
    if use_trgm and trgm_terms:
        for term in trgm_terms:
            score_parts.append(
                "GREATEST(" + ", ".join([f"similarity(COALESCE({col}, ''), %s)" for col in score_cols]) + ")"
            )
            score_params.extend([term] * len(score_cols))
    merged_bonus = ""
    if table == "service_guide_documents" and guide_with_terms_filter and str(scope_filter) == guide_with_terms_filter:
        merged_bonus = (
            " + CASE WHEN id LIKE '%_merged' THEN 0.6 ELSE 0 END"
            " + CASE WHEN id LIKE '카드상품별_거래조건_이자율__수수료_등__merged' THEN 1.5 ELSE 0 END"
        )
    score_expr = (" + ".join(score_parts) if score_parts else "0.0") + merged_bonus
    # DB 연결 및 SQL 실행 (EXPLAIN 포함)
    with _db_conn() as conn:
        with conn.cursor() as cur:
            def _run(where_sql: str, where_params: List[str]):
                source_sql_text = _source_sql(table, include_embedding=False)
                sql = (
                    "WITH source AS ("
                    + source_sql_text
                    + ") "
                    + "SELECT id, content, metadata, " + score_expr + " AS score "
                    + "FROM source WHERE " + where_sql + " "
                    + "ORDER BY score DESC LIMIT %s"
                )
                params = [*score_params, *where_params, limit]
                # SQL 및 EXPLAIN 로그: mogrify 결과만 출력, 파싱 없음
                if _EXPLAIN_ENABLED:
                    try:
                        explain_sql = cur.mogrify(f"EXPLAIN (ANALYZE, BUFFERS) {sql}", params)
                        print(f"[retriever_db][EXPLAIN_SQL] {explain_sql.decode('utf-8')}")
                        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {sql}", params)
                        explain_result = cur.fetchall()
                        if explain_result:
                            print("[retriever_db][EXPLAIN]", "\n".join(str(row[0]) for row in explain_result))
                        else:
                            print("[retriever_db][EXPLAIN] (no result)")
                    except Exception as exc:
                        print(f"[retriever_db][EXPLAIN ERROR] {exc}")
                        conn.rollback()
                exec_start = time.perf_counter()
                cur.execute(_escape_pyformat_percent(sql), params)
                exec_ms = (time.perf_counter() - exec_start) * 1000
                rows = cur.fetchall()
                print(f"[retriever_db] table={table} mode=text submode={submode} exec_ms={exec_ms:.1f} rows={len(rows)}")
                return rows

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
