from typing import Dict, List, Optional
import os
import re
import time

from app.rag.router.router import route_query as _route_query
from app.rag.common.text_utils import unique_in_order
from app.rag.retriever.db import _is_card_table, _safe_table, text_search, vector_search
from app.rag.retriever.rank import _collect_candidates, _finalize_candidates
from app.rag.retriever.terms import (
    _as_list,
    _build_search_context,
    _expand_guide_terms,
    _filter_guide_query_terms,
    _priority_terms,
)

from app.rag.common.doc_source_filters import DOC_SOURCE_FILTERS

LOG_RETRIEVER_DEBUG = os.getenv("RAG_LOG_RETRIEVER_DEBUG") == "1"
ENABLE_PRIORITY_TERMS = os.getenv("RAG_ENABLE_PRIORITY_TERMS", "0") == "1"
USE_VECTOR = os.getenv("RAG_USE_VECTOR", "1") != "0"
USE_KEYWORD = os.getenv("RAG_USE_KEYWORD", "1") != "0"
MAX_DB_CALLS = int(os.getenv("RAG_MAX_DB_CALLS", "2"))
MAX_DOCUMENT_SOURCES = min(int(os.getenv("RAG_MAX_DOCUMENT_SOURCES", "2")), 2)

# 문서 소스 필터 패턴
_TERM_SEP_RE = re.compile(r"[\s\-/·]+")
_LOSS_INTENT_KEYS = {"분실", "도난", "분실도난", "도난분실", "잃어버"}
_CARD_INFO_ENTITY_MAP = {
    "다둥이": "서울시다둥이행복카드",
    "국민행복": "국민행복카드",
    "k패스": "K-패스",
    "k-패스": "K-패스",
    "k 패스": "K-패스",
    "나라사랑": "나라사랑카드",
    "으랏차차": "KT 으랏차차",
}
_PAYMENT_INTENT_KEYS = {
    "결제",
    "승인",
    "오류",
    "안돼",
    "안됨",
    "실패",
    "삼성페이",
    "티머니",
    "등록",
    "연동",
    "카카오페이",
    "애플페이",
}
_PAYMENT_BLOCK_TERMS = [
    "k패스",
    "k-패스",
    "다둥이",
    "혜택",
    "연회비",
    "발급",
    "추천",
    "전월",
    "실적",
]
_PAYMENT_BLOCK_PATTERNS = [
    "%k패스%",
    "%K-패스%",
    "%다둥이%",
    "%혜택%",
    "%연회비%",
    "%발급%",
    "%추천%",
    "%전월실적%",
]


def _normalize_match_key(value: str) -> str:
    return _TERM_SEP_RE.sub("", value.strip().lower())


def _should_strict_guide_filter(terms: List[str]) -> bool:
    normalized = {_normalize_match_key(term) for term in terms if term}
    return bool(normalized & _LOSS_INTENT_KEYS)


def _has_any_keyword(text: str, keywords: List[str] | set[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    compact = _normalize_match_key(lower)
    for key in keywords:
        if not key:
            continue
        key_lower = key.lower()
        if key_lower in lower:
            return True
        key_compact = _normalize_match_key(key_lower)
        if key_compact and key_compact in compact:
            return True
    return False


def _row_has_blocked_term(row: tuple[object, str, Dict[str, object], float], blocked: List[str]) -> bool:
    _, content, metadata, _ = row
    meta = metadata if isinstance(metadata, dict) else {}
    title = str(meta.get("title") or meta.get("name") or meta.get("card_name") or "")
    text = f"{title} {content or ''}"
    return _has_any_keyword(text, blocked)


def _should_block_payment_noise(terms: List[str]) -> bool:
    normalized = {_normalize_match_key(term) for term in terms if term}
    return bool(normalized & _PAYMENT_INTENT_KEYS)


def _row_has_any_term(
    row: tuple[object, str, Dict[str, object], float],
    terms: List[str],
) -> bool:
    _, content, metadata, _ = row
    meta = metadata if isinstance(metadata, dict) else {}
    title = str(meta.get("title") or meta.get("name") or meta.get("card_name") or "")
    text = f"{title} {content or ''}".lower()
    compact = _normalize_match_key(text)
    for term in terms:
        if not term:
            continue
        term_lower = term.lower()
        if term_lower in text:
            return True
        term_compact = _normalize_match_key(term)
        if term_compact and term_compact in compact:
            return True
    return False


def _is_card_specific_meta(metadata: Dict[str, object]) -> bool:
    original = str(metadata.get("original_card_name") or "").strip()
    card_name = str(metadata.get("card_name") or "").strip()
    return bool(original or card_name)


# Router entry

def route_query(query: str) -> Dict[str, Optional[object]]:
    return _route_query(query)


def _fetch_k(top_k: int) -> int:
    return max(top_k * 2, top_k + 3)


def _get_document_source_filter(source: str) -> Optional[str]:
    """문서 소스에 해당하는 SQL 필터 조건을 반환합니다."""
    return DOC_SOURCE_FILTERS.get(source)


async def retrieve_docs(
    query: str,
    routing: Dict[str, object],
    top_k: int = 5,
    table: Optional[str] = None,
) -> List[Dict[str, object]]:
    filters = routing.get("filters") or routing.get("boost") or {}
    route_name = routing.get("route") or routing.get("ui_route")
    db_route = routing.get("db_route")
    if route_name == "card_info":
        # 엔티티 하드 필터: 매칭 실패 시 후보 제거
        if not _as_list(filters.get("card_name")):
            lowered = query.lower()
            for key, canonical in _CARD_INFO_ENTITY_MAP.items():
                if key in lowered:
                    filters["card_name"] = [canonical]
                    break
        if _as_list(filters.get("card_name")):
            filters["require_card_name_match"] = True
        routing["filters"] = filters
    # APPLEPAY: 애플페이 intent 감지 시 guide 문서만 검색
    applepay_intent = routing.get("applepay_intent")
    if applepay_intent:
        tables = ["service_guide_documents"]
    elif table is not None:
        tables = [_safe_table(table)]
    elif db_route == "card_tbl":
        tables = ["card_products"]
    elif db_route == "guide_tbl":
        tables = ["service_guide_documents"]
    elif db_route == "both":
        tables = ["card_products", "service_guide_documents"]
    elif route_name == "card_usage" and not _as_list(filters.get("card_name")):
        tables = ["service_guide_documents"]
    elif route_name == "card_info":
        tables = ["card_products"]
    elif _as_list(filters.get("card_name")):
        tables = ["card_products", "service_guide_documents"]
    else:
        tables = ["card_products", "service_guide_documents"]
    return await retrieve_multi(query=query, routing=routing, tables=tables, top_k=top_k)


async def retrieve_multi(
    query: str,
    routing: Dict[str, object],
    tables: List[str],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    context = _build_search_context(query, routing)
    route_name = routing.get("route") or routing.get("ui_route")
    fetch_k = _fetch_k(top_k)
    retrieval_mode = routing.get("retrieval_mode") or "keyword_only"
    use_vector = USE_VECTOR and retrieval_mode != "keyword_only"
    use_keyword = USE_KEYWORD
    if route_name == "card_info":
        fetch_k = max(fetch_k, 20)
    candidates: List[tuple[float, int, Dict[str, object]]] = []
    
    # 문서 소스 및 제외 목록
    document_sources = routing.get("document_sources", ["guide_merged", "guide_general"])
    exclude_sources = routing.get("exclude_sources", ["terms"])
    applepay_intent = routing.get("applepay_intent")
    db_calls = 0
    db_calls_limit_reached = False
    last_fetch_elapsed_ms = 0.0
    if document_sources and MAX_DOCUMENT_SOURCES > 0:
        document_sources = document_sources[:MAX_DOCUMENT_SOURCES]

    def _fetch_rows(safe_table: str, source_filter: Optional[str] = None) -> List[tuple]:
        nonlocal db_calls, db_calls_limit_reached
        if MAX_DB_CALLS > 0 and db_calls >= MAX_DB_CALLS:
            db_calls_limit_reached = True
            last_fetch_elapsed_ms = 0.0
            return []
        if routing.get("skip_guide_with_terms_query") and source_filter == DOC_SOURCE_FILTERS.get("guide_with_terms"):
            last_fetch_elapsed_ms = 0.0
            return []
        db_calls += 1
        fetch_start = time.perf_counter()
        search_filters = dict(context.filters)
        def _finish(rows: List[tuple]) -> List[tuple]:
            nonlocal last_fetch_elapsed_ms
            last_fetch_elapsed_ms = (time.perf_counter() - fetch_start) * 1000
            return rows
        
        # 소스 필터 추가
        if source_filter:
            search_filters["_scope_filter"] = source_filter

        if route_name == "card_info" and _is_card_table(safe_table) and context.card_values:
            search_filters["require_card_name_match"] = True
        
        # 애플페이 필터
        if applepay_intent and safe_table == "service_guide_documents":
            search_filters["id_prefix"] = "hyundai_applepay"

        # 카드 테이블 + 카테고리 필터 → 먼저 카테고리로 검색
        if _is_card_table(safe_table) and context.category_terms:
            cat_filters = dict(search_filters)
            cat_filters["category"] = context.category_terms
            if use_vector:
                rows = vector_search(context.query_text, table=safe_table, limit=fetch_k, filters=cat_filters)
            else:
                rows = []
            if not rows and use_keyword:
                rows = text_search(table=safe_table, terms=context.query_terms, limit=fetch_k, filters=cat_filters)
            if not rows and use_vector:
                rows = vector_search(context.query_text, table=safe_table, limit=fetch_k, filters=search_filters)
            return _finish(rows)
        
        # Intent 기반 검색 (guide 전용)
        intent_only = bool(context.intent_terms or context.weak_terms) and not context.card_terms
        if intent_only and not _is_card_table(safe_table):
            intent_vals = _as_list(search_filters.get("intent", []))
            weak_vals = _as_list(search_filters.get("weak_intent", []))
            guide_terms = _expand_guide_terms(unique_in_order([*intent_vals, *weak_vals]))
            if not guide_terms:
                guide_terms = _filter_guide_query_terms(context.query_terms)
            
            rows = []
            if use_keyword:
                rows = text_search(table=safe_table, terms=guide_terms or context.query_terms, limit=fetch_k, filters=search_filters)
            elif use_vector:
                rows = vector_search(context.query_text, table=safe_table, limit=fetch_k, filters=search_filters)
            
            # Loss/theft 쿼리 엄격한 필터
            if guide_terms and _should_strict_guide_filter(guide_terms):
                filtered = [r for r in rows if _row_has_any_term(r, guide_terms)]
                if filtered:
                    rows = filtered
                
                # 카드 특정 메타 제거
                if not context.card_values:
                    generic = [r for r in rows if not _is_card_specific_meta(r[2] if isinstance(r[2], dict) else {})]
                    if generic:
                        rows = generic
            
            return _finish(rows)
        
        # 기본 검색
        if _is_card_table(safe_table):
            rows = vector_search(context.query_text, table=safe_table, limit=fetch_k, filters=search_filters) if use_vector else []
        else:
            rows = []
            if use_keyword:
                rows = text_search(table=safe_table, terms=context.query_terms, limit=fetch_k, filters=search_filters)
            if not rows and use_vector:
                rows = vector_search(context.query_text, table=safe_table, limit=fetch_k, filters=search_filters)
        
        if (ENABLE_PRIORITY_TERMS and use_keyword and _is_card_table(safe_table) 
            and context.category_terms and context.search_mode in {"ISSUE", "BENEFIT"}):
            priority_terms = _priority_terms(context.category_terms)
            if priority_terms:
                extra = text_search(table=safe_table, terms=priority_terms, limit=fetch_k, filters=search_filters)
                rows.extend(extra)
        
        if (use_vector and _is_card_table(safe_table) and context.card_terms 
            and not context.intent_terms and not context.payment_terms and not context.category_terms):
            loose = dict(search_filters)
            loose.pop("card_name", None)
            if not search_filters.get("require_card_name_match"):
                extra = vector_search(context.query_text, table=safe_table, limit=fetch_k, filters=loose)
                if extra:
                    rows.extend(extra)

        return _finish(rows)

    for table in tables:
        safe_table = _safe_table(table)
        
        if safe_table == "service_guide_documents" and document_sources:
            if len(document_sources) >= 2 and "guide_merged" in document_sources and "guide_general" in document_sources:
                source_filter = _get_document_source_filter("guide_all")
                source_label = "guide_all"
            elif document_sources:
                source_filter = _get_document_source_filter(document_sources[0])
                source_label = document_sources[0]
            else:
                source_filter = None
                source_label = "guide_default"
            
            rows = _fetch_rows(safe_table, source_filter=source_filter)
            table_candidates = _collect_candidates(safe_table, rows, context, fetch_k)
            candidates.extend(table_candidates)
            fetch_ms = last_fetch_elapsed_ms
            break_hit = db_calls_limit_reached
            guide_quota = max(top_k * 2, top_k + 3)
            print(
                f"[retriever] source={source_label} fetch_ms={fetch_ms:.1f} cand_added={len(table_candidates)} "
                f"total_cand={len(candidates)} break_hit={break_hit}"
            )
            if LOG_RETRIEVER_DEBUG:
                print(
                    f"[retriever] source={source_label} rows={len(rows)} "
                    f"table_candidates={len(table_candidates)} "
                    f"guide_quota={guide_quota} guide_candidates={len(table_candidates)}"
                )
        else:
            # 기타 테이블: 일반 검색
            rows = _fetch_rows(safe_table)
            table_candidates = _collect_candidates(safe_table, rows, context, fetch_k)
            candidates.extend(table_candidates)
            fetch_ms = last_fetch_elapsed_ms
            break_hit = db_calls_limit_reached
            print(
                f"[retriever] table={safe_table} fetch_ms={fetch_ms:.1f} cand_added={len(table_candidates)} "
                f"total_cand={len(candidates)} break_hit={break_hit}"
            )
        
        if LOG_RETRIEVER_DEBUG and table_candidates:
            print(f"[retriever] table={safe_table} rows={len(rows)} candidates={len(table_candidates)}")

    def _doc_key(doc: Dict[str, object]) -> str:
        title = doc.get("title")
        return title if title else f"__no_title__{doc.get('table')}:{doc.get('id')}"

    # 결과 구성: card_info는 카드/가이드 최소 1개씩 포함하도록 구성
    card_docs = [d for d in candidates if d[2].get("table") == "card_products"]
    guide_docs = [d for d in candidates if d[2].get("table") == "service_guide_documents"]
    lane_allow_mixed = bool(routing.get("lane_allow_mixed"))
    if route_name == "card_info":
        if not lane_allow_mixed:
            docs = _finalize_candidates(card_docs[:top_k], _doc_key, context)
            return docs[:top_k]
        ordered: List[Tuple[float, int, Dict[str, object]]] = []
        if card_docs:
            ordered.extend(card_docs[:max(1, top_k - 1)])
        if guide_docs and len(ordered) < top_k:
            ordered.extend(guide_docs[:1])
        if len(ordered) < top_k:
            rest = [c for c in candidates if c not in ordered]
            ordered.extend(rest[: max(0, top_k - len(ordered))])
        docs = _finalize_candidates(ordered, _doc_key, context)
    else:
        if not lane_allow_mixed:
            docs = _finalize_candidates(guide_docs[:top_k], _doc_key, context)
            return docs[:top_k]
        diverse_docs = []
        if card_docs:
            diverse_docs.append(card_docs[0])
        if guide_docs:
            diverse_docs.append(guide_docs[0])
        rest = [c for c in candidates if c not in diverse_docs]
        diverse_docs.extend(rest[:max(0, top_k - len(diverse_docs))])
        docs = _finalize_candidates(diverse_docs, _doc_key, context)
    if LOG_RETRIEVER_DEBUG and docs:
        top = docs[0]
        print(
            "[retriever] "
            f"top_title={top.get('title')} "
            f"score={top.get('score')} rrf={top.get('rrf_score')} "
            f"title_score={top.get('title_score')}"
        )
    return docs[:top_k]
