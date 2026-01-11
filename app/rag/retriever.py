from typing import Dict, List, Optional, Tuple

import os
import re
import json

import psycopg2
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg2 import register_vector

from app.llm.base import get_openai_client
from app.rag.router import route_query as _route_query
from app.rag.vocab.rules import ACTION_SYNONYMS, CARD_NAME_SYNONYMS, PAYMENT_SYNONYMS, STOPWORDS, WEAK_INTENT_SYNONYMS

load_dotenv()

RRF_K = 60
QUERY_TITLE_WEIGHT = 2
QUERY_CONTENT_WEIGHT = 1
CARD_META_WEIGHT = 3
TITLE_SCORE_WEIGHT = 0.001
ISSUANCE_HINT_TOKENS = ("발급", "신청", "재발급", "대상", "서류")
CATEGORY_MATCH_TOKENS = ("발급", "신청", "재발급", "대상", "서류", "적립", "혜택")
ISSUANCE_TITLE_BONUS = {
    "발급 대상": 6,
    "발급": 4,
    "신청": 3,
    "구비": 2,
    "서류": 2,
    "신규": 2,
}
ISSUANCE_TITLE_DEMOTE = ("유의사항", "공통 제외", "적립", "혜택")
CATEGORY_TITLE_BONUS = {
    "적립 서비스": 4,
    "일상 생활비": 3,
    "필수 생활비": 3,
    "포인트 적립": 2,
}
CATEGORY_TITLE_DEMOTE = ("유의사항", "공통 제외")
KEYWORD_STOPWORDS = {"카드"}
PRIORITY_TERMS_BY_CATEGORY = {
    "발급": ["발급 대상"],
    "신청": ["발급 대상"],
    "재발급": ["발급 대상"],
    "적립": ["적립 서비스", "일상 생활비 적립", "필수 생활비 적립", "포인트 적립"],
}
ISSUE_TERMS = {"발급", "신청", "재발급", "대상", "서류"}
BENEFIT_TERMS = {"적립", "혜택", "할인", "포인트"}
REISSUE_TERMS = {"재발급", "재발행"}
REISSUE_TITLE_PENALTY = 4
MIN_GUIDE_CONTENT_LEN = 60
_BROKEN_JSON_RE = re.compile(r"\}\s*,\s*\{")


# --- 라우터 진입점 ---
def route_query(query: str) -> Dict[str, Optional[object]]:
    return _route_query(query)


# --- 공통 유틸 ---
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


# --- 용어 확장/분류 ---
def _expand_no_space_terms(terms: List[str]) -> List[str]:
    out = []
    for term in terms:
        compact = term.replace(" ", "")
        if compact and compact != term:
            out.append(compact)
    return out


def _expand_payment_terms(terms: List[str]) -> List[str]:
    expanded = []
    for canonical in terms:
        expanded.extend(PAYMENT_SYNONYMS.get(canonical, []))
    combined = _unique_in_order([*terms, *expanded])
    return _unique_in_order([*combined, *_expand_no_space_terms(combined)])


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


def _extract_category_terms(terms: List[str]) -> List[str]:
    hits: List[str] = []
    for term in terms:
        for hint in CATEGORY_MATCH_TOKENS:
            if hint in term:
                hits.append(hint)
    if "발급" in hits and "대상" in hits:
        hits.append("발급 대상")
    return _unique_in_order(hits)


# --- 제목/카테고리 점수 ---
def _issuance_title_bonus(title: Optional[str], category_terms: List[str]) -> int:
    if not title or not category_terms:
        return 0
    if not any(term in ISSUANCE_HINT_TOKENS for term in category_terms):
        return 0
    score = 0
    for token, bonus in ISSUANCE_TITLE_BONUS.items():
        if token in title:
            score += bonus
    if any(token in title for token in ISSUANCE_TITLE_DEMOTE):
        score -= 2
    return score


def _category_title_bonus(title: Optional[str], category_terms: List[str]) -> int:
    if not title or not category_terms:
        return 0
    if not any(term in ("적립", "혜택") for term in category_terms):
        return 0
    score = 0
    for token, bonus in CATEGORY_TITLE_BONUS.items():
        if token in title:
            score += bonus
    if any(token in title for token in CATEGORY_TITLE_DEMOTE):
        score -= 2
    return score


def _priority_terms(category_terms: List[str]) -> List[str]:
    terms: List[str] = []
    for term in category_terms:
        terms.extend(PRIORITY_TERMS_BY_CATEGORY.get(term, []))
    return _unique_in_order(terms)


def _select_search_mode(terms: List[str]) -> str:
    if any(term in ISSUE_TERMS for term in terms):
        return "ISSUE"
    if any(term in BENEFIT_TERMS for term in terms):
        return "BENEFIT"
    return "GENERAL"


# --- guide 문서 정리 ---
def _is_noisy_guide_doc(title: Optional[str], content: str) -> bool:
    if not title:
        return True
    if not content or len(content.strip()) < MIN_GUIDE_CONTENT_LEN:
        return True
    if _BROKEN_JSON_RE.search(content):
        return True
    stripped = content.lstrip()
    if stripped.startswith(("}", "]")):
        return True
    return False


# --- 매칭 점수 ---
def _title_match_score(title: Optional[str], terms: List[str], weight: int) -> int:
    if not title:
        return 0
    lowered = title.lower()
    score = 0
    for term in terms:
        if term and term.lower() in lowered:
            score += weight
    return score


def _content_match_score(content: str, terms: List[str], weight: int) -> int:
    if not content:
        return 0
    lowered = content.lower()
    score = 0
    for term in terms:
        if term and term.lower() in lowered:
            score += weight
    return score


def _card_meta_score(metadata: Dict[str, object], card_values: List[str]) -> int:
    if not card_values:
        return 0
    card_name = metadata.get("card_name")
    if not card_name:
        return 0
    card_name_str = str(card_name)
    card_name_norm = card_name_str.replace(" ", "")
    for value in card_values:
        value_str = str(value)
        if card_name_str == value_str:
            return CARD_META_WEIGHT
        value_norm = value_str.replace(" ", "")
        if card_name_norm == value_norm:
            return CARD_META_WEIGHT
        if value_str and value_str in card_name_str:
            return CARD_META_WEIGHT
        if value_norm and value_norm in card_name_norm:
            return CARD_META_WEIGHT
    return 0


def _card_term_match(title: Optional[str], content: str, card_terms: List[str]) -> bool:
    if card_terms and _title_match_score(title, card_terms, 1) > 0:
        return True
    if card_terms and _content_match_score(content, card_terms, 1) > 0:
        return True
    return False


def _category_match_score(meta: Dict[str, object], terms: List[str]) -> int:
    if not meta or not terms:
        return 0
    parts = []
    for key in ("category", "category1", "category2"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    if not parts:
        return 0
    category_text = " ".join(parts)
    return _title_match_score(category_text, terms, 1)


# --- SQL 조건 구성 ---
def _build_like_group(terms: List[str], params: List[str]) -> Optional[str]:
    if not terms:
        return None
    term_clauses = []
    for term in terms:
        term_clauses.append("(content ILIKE %s OR metadata->>'title' ILIKE %s)")
        params.extend([f"%{term}%", f"%{term}%"])
    return "(" + " OR ".join(term_clauses) + ")"


# --- 쿼리 파싱 ---
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


# --- 콘텐츠 정규화 ---
def _parse_json_content(content: str) -> Dict[str, object]:
    if not content:
        return {}
    text = content.lstrip()
    if text.startswith("{") or text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                data = data[0] if data else {}
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def _extract_json_value(content: str, key: str) -> Optional[str]:
    if not content:
        return None
    match = re.search(rf'"{re.escape(key)}"\\s*:\\s*"([^"]+)"', content)
    if not match:
        return None
    value = match.group(1)
    return value.replace("\\n", "\n")


def _normalize_doc_fields(
    content: str,
    metadata: Optional[object],
) -> Tuple[Optional[str], str, Dict[str, object]]:
    meta = metadata if isinstance(metadata, dict) else {}
    title = meta.get("title")
    normalized_content = content or ""
    id_candidate: Optional[str] = None

    if not title:
        parsed = _parse_json_content(normalized_content)
        if parsed:
            if not title:
                title = parsed.get("title") or (parsed.get("metadata") or {}).get("title")
            if not id_candidate:
                id_candidate = parsed.get("id") or (parsed.get("metadata") or {}).get("id")
            body = parsed.get("content") or parsed.get("text")
            if isinstance(body, str) and body:
                normalized_content = body
            parsed_meta = parsed.get("metadata")
            if isinstance(parsed_meta, dict):
                meta = {**parsed_meta, **meta}
        else:
            title = _extract_json_value(normalized_content, "title") or title
            body = _extract_json_value(normalized_content, "content")
            if not body:
                body = _extract_json_value(normalized_content, "text")
            if body:
                normalized_content = body
            if not id_candidate:
                id_candidate = _extract_json_value(normalized_content, "id")

    if not meta.get("id") and id_candidate:
        meta["id"] = id_candidate

    return title, normalized_content, meta


# --- DB 필터 구성 ---
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
    if category_terms and table == "card_tbl":
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

    if table == "guide_tbl":
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


# --- DB 설정/벡터 검색 ---
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
    params: List[str] = []
    term_clauses = []
    for term in terms:
        term_clauses.append(
            "("
            "content ILIKE %s OR "
            "metadata->>'title' ILIKE %s OR "
            "metadata->>'category' ILIKE %s OR "
            "metadata->>'category1' ILIKE %s OR "
            "metadata->>'category2' ILIKE %s"
            ")"
        )
        params.extend([f"%{term}%"] * 5)
    where_sql = "(" + " OR ".join(term_clauses) + ")" if term_clauses else ""
    if table == "card_tbl":
        card_values = _as_list(filters.get("card_name"))
        if card_values:
            placeholders = ", ".join(["%s"] * len(card_values))
            where_sql = f"({where_sql}) AND metadata->>'card_name' IN ({placeholders})"
            params.extend(card_values)
    if not where_sql:
        return []
    with psycopg2.connect(**_db_config()) as conn:
        with conn.cursor() as cur:
            sql = f"SELECT id, content, metadata, 0.0 AS score FROM {table} WHERE {where_sql} LIMIT %s"
            params.append(limit)
            cur.execute(sql, params)
            return cur.fetchall()


# --- 후보 랭킹 ---
def _build_candidates_from_rows(
    vec_rows: List[Tuple[object, str, Dict[str, object], float]],
    kw_rows: List[Tuple[object, str, Dict[str, object], float]],
    table: str,
    card_values: List[str],
    card_terms: List[str],
    rank_terms: List[str],
    query_terms: List[str],
    category_terms: List[str],
    search_mode: str,
    wants_reissue: bool,
) -> List[Tuple[int, float, Dict[str, object]]]:
    vec_rank: Dict[str, int] = {}
    kw_rank: Dict[str, int] = {}
    vec_docs: Dict[str, Dict[str, object]] = {}
    kw_docs: Dict[str, Dict[str, object]] = {}

    for idx, (doc_id, content, metadata, score) in enumerate(vec_rows, 1):
        key = f"{table}:{doc_id}"
        if key in vec_docs:
            continue
        title, normalized_content, normalized_meta = _normalize_doc_fields(content, metadata)
        if table == "guide_tbl" and _is_noisy_guide_doc(title, normalized_content):
            continue
        output_id = normalized_meta.get("id") or doc_id
        vec_docs[key] = {
            "id": str(output_id) if output_id is not None else "",
            "db_id": doc_id,
            "title": title,
            "content": normalized_content,
            "metadata": normalized_meta,
            "vector_score": float(score),
            "table": table,
        }
        vec_rank[key] = idx

    for idx, (doc_id, content, metadata, _) in enumerate(kw_rows, 1):
        key = f"{table}:{doc_id}"
        if key in kw_docs:
            continue
        title, normalized_content, normalized_meta = _normalize_doc_fields(content, metadata)
        if table == "guide_tbl" and _is_noisy_guide_doc(title, normalized_content):
            continue
        output_id = normalized_meta.get("id") or doc_id
        kw_docs[key] = {
            "id": str(output_id) if output_id is not None else "",
            "db_id": doc_id,
            "title": title,
            "content": normalized_content,
            "metadata": normalized_meta,
            "vector_score": 0.0,
            "table": table,
        }
        kw_rank[key] = idx

    candidates: List[Tuple[int, float, Dict[str, object]]] = []
    for key in set(vec_docs.keys()) | set(kw_docs.keys()):
        doc = vec_docs.get(key) or kw_docs.get(key)
        if not doc:
            continue
        rrf_score = 0.0
        if key in vec_rank:
            rrf_score += 1.0 / (RRF_K + vec_rank[key])
        if key in kw_rank:
            rrf_score += 1.0 / (RRF_K + kw_rank[key])
        title = doc.get("title")
        meta = doc.get("metadata") or {}
        card_meta_score = _card_meta_score(meta, card_values)
        title_score = _title_match_score(title, card_terms, 2)
        title_score += _title_match_score(title, rank_terms, 1)
        title_score += _title_match_score(title, query_terms, QUERY_TITLE_WEIGHT)
        title_score += _content_match_score(doc.get("content") or "", query_terms, QUERY_CONTENT_WEIGHT)
        card_match_base = card_meta_score > 0 or _card_term_match(
            title,
            doc.get("content") or "",
            card_terms,
        )
        category_score = 0
        issuance_bonus = 0
        category_bonus = 0
        if search_mode == "ISSUE":
            issuance_bonus = _issuance_title_bonus(title, category_terms)
        elif search_mode == "BENEFIT":
            category_score = _category_match_score(meta, query_terms)
            category_bonus = _category_title_bonus(title, category_terms)
        if card_values and not card_match_base:
            category_score = 0
            issuance_bonus = 0
            category_bonus = 0
        reissue_penalty = 0
        if title and "재발급" in title:
            if wants_reissue:
                reissue_penalty = REISSUE_TITLE_PENALTY
            else:
                reissue_penalty = -REISSUE_TITLE_PENALTY
        title_score += category_score
        title_score += issuance_bonus
        title_score += category_bonus
        title_score += reissue_penalty
        title_score += card_meta_score
        doc["card_meta_score"] = card_meta_score
        doc["issuance_bonus"] = issuance_bonus
        doc["category_bonus"] = category_bonus
        doc["reissue_penalty"] = reissue_penalty
        if card_values:
            doc["card_match"] = card_match_base
        else:
            doc["card_match"] = True
        final_score = rrf_score + (title_score * TITLE_SCORE_WEIGHT)
        doc["score"] = final_score
        doc["rrf_score"] = rrf_score
        doc["title_score"] = title_score
        candidates.append((title_score, rrf_score, doc))

    return candidates


def _build_search_context(
    query: str,
    routing: Dict[str, object],
) -> Dict[str, object]:
    query_text = _build_query_text(query, routing.get("query_template"))
    filters = routing.get("filters") or {}

    card_values = _as_list(filters.get("card_name"))
    card_terms = _expand_card_terms(card_values)
    intent_terms = _expand_action_terms(_as_list(filters.get("intent")))
    weak_terms = _expand_weak_terms(_as_list(filters.get("weak_intent")))
    payment_terms = _expand_payment_terms(_as_list(filters.get("payment_method")))
    query_terms = _extract_query_terms(query)
    category_terms = _extract_category_terms([*query_terms, *weak_terms, *intent_terms])
    search_mode = _select_search_mode([*category_terms, *query_terms, *weak_terms, *intent_terms])
    wants_reissue = any(term in REISSUE_TERMS for term in [*query_terms, *intent_terms])
    rank_terms = _unique_in_order([*intent_terms, *payment_terms, *weak_terms, *query_terms])
    payment_only = bool(payment_terms) and not card_terms and not intent_terms
    payment_norm = {term.lower().replace(" ", "") for term in payment_terms}
    extra_terms = [
        term
        for term in query_terms
        if term.lower().replace(" ", "") not in payment_norm
    ]

    return {
        "query_text": query_text,
        "filters": filters,
        "card_values": card_values,
        "card_terms": card_terms,
        "intent_terms": intent_terms,
        "weak_terms": weak_terms,
        "payment_terms": payment_terms,
        "query_terms": query_terms,
        "category_terms": category_terms,
        "search_mode": search_mode,
        "wants_reissue": wants_reissue,
        "rank_terms": rank_terms,
        "payment_only": payment_only,
        "extra_terms": extra_terms,
    }


def _fetch_k(top_k: int) -> int:
    return max(top_k * 3, top_k + 5)


def _keyword_rows(
    table: str,
    context: Dict[str, object],
    limit: int,
) -> List[Tuple[object, str, Dict[str, object], float]]:
    search_mode = context.get("search_mode", "GENERAL")
    if table == "guide_tbl":
        if search_mode != "ISSUE":
            return []
    else:
        if not (context.get("payment_only") or search_mode in {"ISSUE", "BENEFIT"}):
            return []
    extra_terms: List[str] = list(context.get("extra_terms") or [])
    if context.get("category_terms"):
        extra_terms.extend(context["category_terms"])
    extra_terms = _unique_in_order(extra_terms)
    extra_terms = [term for term in extra_terms if term not in KEYWORD_STOPWORDS]
    if not extra_terms:
        return []
    return text_search(table=table, terms=extra_terms, limit=limit, filters=context.get("filters"))


def _collect_candidates(
    table: str,
    vec_rows: List[Tuple[object, str, Dict[str, object], float]],
    context: Dict[str, object],
    limit: int,
) -> List[Tuple[int, float, Dict[str, object]]]:
    return _build_candidates_from_rows(
        vec_rows=vec_rows,
        kw_rows=_keyword_rows(table, context, limit),
        table=table,
        card_values=context["card_values"],
        card_terms=context["card_terms"],
        rank_terms=context["rank_terms"],
        query_terms=context["query_terms"],
        category_terms=context["category_terms"],
        search_mode=context.get("search_mode", "GENERAL"),
        wants_reissue=bool(context.get("wants_reissue")),
    )


def _finalize_candidates(
    candidates: List[Tuple[int, float, Dict[str, object]]],
    key_fn,
) -> List[Dict[str, object]]:
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    has_card_match = any(doc.get("card_match") for _, _, doc in candidates)
    if has_card_match:
        candidates = [item for item in candidates if item[2].get("card_match")]

    best_by_title: Dict[str, Tuple[Tuple[int, float], Dict[str, object]]] = {}
    for _, score, doc in candidates:
        key = key_fn(doc)
        content_len = len(doc.get("content") or "")
        rank_key = (content_len, score)
        existing = best_by_title.get(key)
        if not existing or rank_key > existing[0]:
            best_by_title[key] = (rank_key, doc)

    docs = [item[1] for item in best_by_title.values()]
    docs.sort(key=lambda item: (item.get("title_score", 0), item.get("score", 0.0)), reverse=True)
    return docs


# --- 공개 검색 API ---
async def retrieve_docs(
    query: str,
    routing: Dict[str, object],
    top_k: int = 5,
    table: Optional[str] = None,
    allow_fallback: bool = True,
) -> List[Dict[str, object]]:
    tables = ["card_tbl", "guide_tbl"] if table is None else [_safe_table(table)]
    if allow_fallback and table == "card_tbl":
        if "guide_tbl" not in tables:
            tables.append("guide_tbl")
    return await retrieve_multi(query=query, routing=routing, tables=tables, top_k=top_k)


async def retrieve_multi(
    query: str,
    routing: Dict[str, object],
    tables: List[str],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    context = _build_search_context(query, routing)
    fetch_k = _fetch_k(top_k)
    candidates: List[Tuple[int, float, Dict[str, object]]] = []

    for table in tables:
        safe_table = _safe_table(table)
        rows: List[Tuple[object, str, Dict[str, object], float]] = []
        if safe_table == "card_tbl" and context.get("category_terms"):
            category_filters = dict(context["filters"])
            category_filters["category"] = context["category_terms"]
            rows = vector_search(
                context["query_text"],
                table=safe_table,
                limit=fetch_k,
                filters=category_filters,
            )
            if not rows:
                rows = vector_search(
                    context["query_text"],
                    table=safe_table,
                    limit=fetch_k,
                    filters=context["filters"],
                )
        else:
            rows = vector_search(
                context["query_text"],
                table=safe_table,
                limit=fetch_k,
                filters=context["filters"],
            )
        if (
            safe_table == "card_tbl"
            and context.get("category_terms")
            and context.get("search_mode") in {"ISSUE", "BENEFIT"}
        ):
            priority_terms = _priority_terms(context["category_terms"])
            if priority_terms:
                rows.extend(
                    text_search(
                        table=safe_table,
                        terms=priority_terms,
                        limit=fetch_k,
                        filters=context.get("filters"),
                    )
                )
        if (
            safe_table == "card_tbl"
            and context["card_terms"]
            and not context["intent_terms"]
            and not context["payment_terms"]
            and not context.get("category_terms")
        ):
            loose_filters = dict(context["filters"])
            loose_filters.pop("card_name", None)
            loose_rows = vector_search(context["query_text"], table=safe_table, limit=fetch_k, filters=loose_filters)
            if loose_rows:
                rows.extend(loose_rows)
        candidates.extend(_collect_candidates(safe_table, rows, context, fetch_k))

    def _doc_key(doc: Dict[str, object]) -> str:
        title = doc.get("title")
        return title if title else f"__no_title__{doc.get('table')}:{doc.get('id')}"

    docs = _finalize_candidates(candidates, _doc_key)
    return docs[:top_k]
