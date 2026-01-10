from typing import Dict, List, Optional, Tuple

import os
import re

import psycopg2
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg2 import register_vector

from app.llm.base import get_openai_client
from app.rag.router import route_query as _route_query
from app.rag.vocab.rules import ACTION_SYNONYMS, CARD_NAME_SYNONYMS, PAYMENT_SYNONYMS, STOPWORDS, WEAK_INTENT_SYNONYMS

load_dotenv()


def route_query(query: str) -> Dict[str, Optional[object]]:
    return _route_query(query)


def _as_list(value: Optional[object]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [item for item in value if item]


def _unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _expand_payment_terms(terms: List[str]) -> List[str]:
    expanded = []
    for canonical in terms:
        expanded.extend(PAYMENT_SYNONYMS.get(canonical, []))
    return _unique_in_order([*terms, *expanded])


def _expand_card_terms(terms: List[str]) -> List[str]:
    expanded = []
    for canonical in terms:
        expanded.extend(CARD_NAME_SYNONYMS.get(canonical, []))
    return _unique_in_order([*terms, *expanded])


def _expand_action_terms(terms: List[str]) -> List[str]:
    expanded = []
    for canonical in terms:
        expanded.extend(ACTION_SYNONYMS.get(canonical, []))
    return _unique_in_order([*terms, *expanded])


def _expand_weak_terms(terms: List[str]) -> List[str]:
    expanded = []
    for canonical in terms:
        expanded.extend(WEAK_INTENT_SYNONYMS.get(canonical, []))
    return _unique_in_order([*terms, *expanded])


def _title_match_score(title: Optional[str], terms: List[str], weight: int) -> int:
    if not title:
        return 0
    lowered = title.lower()
    score = 0
    for term in terms:
        if term and term.lower() in lowered:
            score += weight
    return score


def _build_like_group(terms: List[str], params: List[str]) -> Optional[str]:
    if not terms:
        return None
    term_clauses = []
    for term in terms:
        term_clauses.append("(content ILIKE %s OR metadata->>'title' ILIKE %s)")
        params.extend([f"%{term}%", f"%{term}%"])
    return "(" + " OR ".join(term_clauses) + ")"


def _build_query_text(query: str, query_template: Optional[str]) -> str:
    if query_template:
        merged = f"{query_template} {query}".strip()
        return merged or query
    return query


_TERM_WS_RE = re.compile(r"\s+")


def _extract_query_terms(query: str) -> List[str]:
    text = _TERM_WS_RE.sub(" ", query.strip().lower())
    raw_terms = [term for term in text.split(" ") if term]
    stopwords = {word.lower() for word in STOPWORDS}
    terms = []
    for term in raw_terms:
        if term.isdigit():
            continue
        if len(term) < 2:
            continue
        if term in stopwords:
            continue
        terms.append(term)
    return _unique_in_order(terms)


def build_where_clause(
    filters: Optional[Dict[str, object]],
    table: str,
) -> Tuple[str, List[str]]:
    filters = filters or {}
    clauses: List[str] = []
    params: List[str] = []

    for key in ("category1", "category2"):
        values = _as_list(filters.get(key))
        if not values:
            continue
        placeholders = ", ".join(["%s"] * len(values))
        clauses.append(f"metadata->>'{key}' IN ({placeholders})")
        params.extend(values)

    card_terms = _expand_card_terms(_as_list(filters.get("card_name")))
    intent_terms = _expand_action_terms(_as_list(filters.get("intent")))
    payment_terms = _expand_payment_terms(_as_list(filters.get("payment_method")))

    if table == "guide_tbl":
        intent_group = _build_like_group(intent_terms, params)
        if intent_group:
            clauses.append(intent_group)
    else:
        card_group = _build_like_group(card_terms, params)
        payment_group = None
        if not card_group:
            payment_group = _build_like_group(payment_terms, params)
        if card_group:
            clauses.append(card_group)
        if payment_group:
            clauses.append(payment_group)

    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params


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


def _safe_table(name: str) -> str:
    if name not in ("card_tbl", "guide_tbl"):
        raise ValueError(f"Unsupported table: {name}")
    return name


def _choose_table(filters: Dict[str, object]) -> str:
    if filters.get("card_name"):
        return "card_tbl"
    if filters.get("intent") or filters.get("weak_intent"):
        return "guide_tbl"
    return "card_tbl"


def embed_query(text: str, model: str = "text-embedding-3-small") -> List[float]:
    client = get_openai_client()
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def vector_search(
    query: str,
    table: str,
    limit: int,
    filters: Optional[Dict[str, object]] = None,
) -> List[Tuple[object, str, Dict[str, object], float]]:
    table = _safe_table(table)
    emb = Vector(embed_query(query))
    where_sql, where_params = build_where_clause(filters, table)
    with psycopg2.connect(**_db_config()) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            sql = (
                f"SELECT id, content, metadata, 1 - (embedding <=> %s) AS score "
                f"FROM {table}{where_sql} ORDER BY embedding <=> %s LIMIT %s"
            )
            params = [emb, *where_params, emb, limit]
            try:
                cur.execute(sql, params)
            except Exception:
                conn.rollback()
                sql = (
                    f"SELECT id, content, metadata, 1 - (embedding <-> %s) AS score "
                    f"FROM {table}{where_sql} ORDER BY embedding <-> %s LIMIT %s"
                )
                cur.execute(sql, params)
            return cur.fetchall()


async def retrieve_docs(
    query: str,
    routing: Dict[str, object],
    top_k: int = 5,
    table: Optional[str] = None,
    allow_fallback: bool = True,
) -> List[Dict[str, object]]:
    query_text = _build_query_text(query, routing.get("query_template"))
    filters = routing.get("filters") or {}

    card_terms = _expand_card_terms(_as_list(filters.get("card_name")))
    intent_terms = _expand_action_terms(_as_list(filters.get("intent")))
    weak_terms = _expand_weak_terms(_as_list(filters.get("weak_intent")))
    payment_terms = _expand_payment_terms(_as_list(filters.get("payment_method")))
    query_terms = _extract_query_terms(query)
    rank_terms = _unique_in_order([*intent_terms, *payment_terms, *weak_terms, *query_terms])

    table = _safe_table(table) if table else _choose_table(filters)
    fetch_k = max(top_k * 3, top_k + 5)
    rows = vector_search(query_text, table=table, limit=fetch_k, filters=filters)
    if table == "card_tbl" and card_terms and not intent_terms and not payment_terms:
        loose_filters = dict(filters)
        loose_filters.pop("card_name", None)
        loose_rows = vector_search(query_text, table=table, limit=fetch_k, filters=loose_filters)
        if loose_rows:
            rows.extend(loose_rows)
    if allow_fallback and not rows and table == "card_tbl":
        table = "guide_tbl"
        rows = vector_search(query_text, table=table, limit=fetch_k, filters=filters)

    candidates: List[Tuple[int, float, Dict[str, object]]] = []
    for doc_id, content, metadata, score in rows:
        title = metadata.get("title") if isinstance(metadata, dict) else None
        title_score = _title_match_score(title, card_terms, 2)
        title_score += _title_match_score(title, rank_terms, 1)
        candidates.append(
            (
                title_score,
                float(score),
                {
                    "id": doc_id,
                    "title": title,
                    "content": content,
                    "metadata": metadata,
                    "score": float(score),
                    "title_score": title_score,
                    "table": table,
                },
            )
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    has_title_hits = any(score > 0 for score, _, _ in candidates)
    if has_title_hits:
        candidates = [item for item in candidates if item[0] > 0]

    best_by_title: Dict[str, Tuple[Tuple[int, float], Dict[str, object]]] = {}
    for _, score, doc in candidates:
        title = doc.get("title")
        key = title if title else f"__no_title__{doc.get('id')}"
        content_len = len(doc.get("content") or "")
        rank_key = (content_len, score)
        existing = best_by_title.get(key)
        if not existing or rank_key > existing[0]:
            best_by_title[key] = (rank_key, doc)

    docs = [item[1] for item in best_by_title.values()]
    docs.sort(key=lambda item: (item.get("title_score", 0), item.get("score", 0.0)), reverse=True)
    return docs[:top_k]
