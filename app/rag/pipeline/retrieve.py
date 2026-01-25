from __future__ import annotations

from typing import Any, Dict, List, Set
import time

from app.rag.common.doc_source_filters import DOC_SOURCE_FILTERS
from app.rag.pipeline.utils import GUIDE_INTENT_TOKENS, should_expand_card_info, text_has_any_compact
from app.rag.policy.policy_pins import POLICY_PINS
from app.rag.retriever.retriever import retrieve_multi
from app.rag.retriever.db import fetch_docs_by_ids
from app.rag.retriever.consult_retriever import retrieve_consult_docs


DOCUMENT_SOURCE_POLICY_MAP = {
    "A": ["guide_merged", "guide_general"],
    "B": ["guide_general", "guide_merged"],
    "C": ["guide_general"],
}
DEFAULT_DOCUMENT_SOURCES = ["guide_merged", "guide_general"]


def _normalize_text(text: str) -> str:
    return (text or "").lower()


def _compact_text(text: str) -> str:
    return (text or "").replace(" ", "")


def _is_kpass_query(normalized: str, compact: str) -> bool:
    return "k패스" in normalized or "k-패스" in normalized or "kpass" in compact


def _is_urachacha_query(normalized: str) -> bool:
    return any(term in normalized for term in ("으랏차차", "으랏차"))


def _is_reservation_query(normalized: str) -> bool:
    return any(term in normalized for term in ("예약신청", "카드대출", "대출", "카드론"))


def _has_token(tokens: set[str], *values: str) -> bool:
    normalized = [val.lower() for val in values if val]
    token_set = {token.lower() for token in tokens}
    return any(any(token in text for token in token_set) for text in normalized)


def _collect_pinned_entries(query: str, routing: Dict[str, Any]) -> List[Dict[str, Any]]:
    pins: List[Dict[str, Any]] = []
    normalized = _normalize_text(query)
    matched = routing.get("matched", {}) or {}
    card_names = [name.lower() for name in matched.get("card_names") or []]
    actions = [a.lower() for a in matched.get("actions") or []]
    payments = [p.lower() for p in matched.get("payments") or []]

    for policy in POLICY_PINS:
        policy_card_names = [name.lower() for name in policy.get("card_names", [])]
        if policy_card_names:
            if not any(req in card_name for card_name in card_names for req in policy_card_names):
                continue
        policy_excludes = [name.lower() for name in policy.get("exclude_card_names", [])]
        if policy_excludes and any(ex in card_name for card_name in card_names for ex in policy_excludes):
            continue
        policy_tokens = set(policy.get("tokens", []))
        if policy_tokens and not _has_token(policy_tokens, normalized, *actions, *payments):
            continue
        doc_ids = policy.get("doc_ids") or []
        if not doc_ids:
            continue
        pins.append({"table": policy["table"], "ids": doc_ids})

    norm_query = _compact_text(normalized)

    def _pin(table: str, ids: list[str]) -> None:
        if ids:
            pins.append({"table": table, "ids": ids})

    if _is_kpass_query(normalized, norm_query):
        _pin("service_guide_documents", ["k패스_2", "k패스_13", "k패스_14"])
        _pin("card_products", ["CARD-SHINHAN-K-패스-신한카드-체크"])

    if _is_urachacha_query(normalized):
        _pin("card_products", ["CARD-SHINHAN-KT-으랏차차-신한카드"])

    if _is_reservation_query(normalized):
        _pin("service_guide_documents", ["카드대출 예약신청_merged"])
    return pins


def _doc_unique_id(doc: Dict[str, Any]) -> str:
    return str(doc.get("id") or doc.get("db_id") or "")


async def retrieve_docs(
    query: str,
    routing: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    start = time.perf_counter()
    filters = routing.get("filters") or routing.get("boost") or {}
    route_name = routing.get("route") or routing.get("ui_route")
    db_route = routing.get("db_route")
    routing_for_retrieve = routing
    normalized_query = (query or "").lower()
    compact_query = _compact_text(normalized_query)
    _FINANCIAL_TERMS = {
        "이자",
        "수수료",
        "연체",
        "리볼빙",
        "약관",
        "요율",
        "거래조건",
        "한도",
        "금리",
        "현금서비스",
        "단기대출",
    }
    if route_name == "card_usage" and any(term in normalized_query for term in _FINANCIAL_TERMS):
        routing_for_retrieve = dict(routing_for_retrieve)
        routing_for_retrieve["document_sources"] = ["guide_with_terms"]
        routing_for_retrieve["skip_guide_with_terms_query"] = True
    
    sources = set()
    if db_route == "card_tbl":
        sources.add("card_products")
    elif db_route == "guide_tbl":
        sources.add("service_guide_documents")
    elif db_route == "both":
        sources.update({"card_products", "service_guide_documents"})
    else:
        # router가 명시하지 않은 경우, 라우트/필터 기반 최소 소스 선택
        if route_name == "card_usage" and not filters.get("card_name"):
            sources.add("service_guide_documents")
        elif route_name == "card_info":
            sources.add("card_products")
        elif filters.get("card_name"):
            sources.update({"card_products", "service_guide_documents"})
    if route_name == "card_info" and should_expand_card_info(query, routing, filters):
        sources.update({"card_products", "service_guide_documents"})
        if "allow_guide_without_card_match" not in routing:
            routing_for_retrieve = dict(routing)
            routing_for_retrieve["allow_guide_without_card_match"] = True
    if (
        filters.get("card_name")
        and text_has_any_compact(query, GUIDE_INTENT_TOKENS)
        and "allow_guide_without_card_match" not in routing
    ):
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        routing_for_retrieve["allow_guide_without_card_match"] = True
    if filters.get("payment_method") and not (route_name == "card_usage" and not filters.get("card_name")):
        sources.update({"card_products", "service_guide_documents"})
    if filters.get("intent") or filters.get("weak_intent"):
        sources.add("service_guide_documents")

    # K-패스 쿼리는 card_name 필터를 보강해 카드/가이드 매칭을 강제
    if _is_kpass_query(normalized_query, compact_query):
        card_names = filters.get("card_name")
        if not card_names:
            card_names = ["k패스"]
        elif isinstance(card_names, list):
            if "k패스" not in card_names:
                card_names = [*card_names, "k패스"]
        else:
            card_names = [str(card_names), "k패스"]
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        filters_copy = dict(routing_for_retrieve.get("filters", routing_for_retrieve.get("boost", {})))
        filters_copy["card_name"] = card_names
        routing_for_retrieve["filters"] = filters_copy
        routing_for_retrieve["boost"] = filters_copy
    
    # APPLEPAY: 애플페이 intent 감지 시 guide 문서만 사용
    applepay_intent = routing.get("applepay_intent")
    if applepay_intent:
        sources = {"service_guide_documents"}
    
    # 스코프 필터 적용: routing에서 받은 document_sources 기반으로 필터 생성
    document_source_policy = routing.get("document_source_policy")
    document_sources = routing.get("document_sources")
    if not document_sources:
        document_sources = DOCUMENT_SOURCE_POLICY_MAP.get(document_source_policy, DEFAULT_DOCUMENT_SOURCES)

    pinned_entries = _collect_pinned_entries(query, routing)
    pinned_docs: List[Dict[str, Any]] = []
    pinned_doc_ids: Set[str] = set()
    for entry in pinned_entries:
        fetched = fetch_docs_by_ids(entry["table"], entry["ids"])
        for doc in fetched:
            doc_id = _doc_unique_id(doc)
            if doc_id and doc_id not in pinned_doc_ids:
                pinned_doc_ids.add(doc_id)
                pinned_docs.append(doc)
    if document_sources and "service_guide_documents" in sources:
        has_merged = "guide_merged" in document_sources
        has_general = "guide_general" in document_sources
        has_terms_all = "guide_with_terms" in document_sources
        
        if has_terms_all:
            guide_filter = DOC_SOURCE_FILTERS.get("guide_with_terms")
        elif has_merged and has_general:
            guide_filter = DOC_SOURCE_FILTERS.get("guide_all")
            if "guide_merged" in document_sources:
                document_sources.remove("guide_merged")
            if "guide_general" in document_sources:
                document_sources.remove("guide_general")
            if "guide_all" not in document_sources:
                document_sources.append("guide_all")
        elif has_merged:
            guide_filter = DOC_SOURCE_FILTERS.get("guide_merged")
        elif has_general:
            guide_filter = DOC_SOURCE_FILTERS.get("guide_general")
        else:
            guide_filter = None

        if guide_filter:
            if routing_for_retrieve is routing:
                routing_for_retrieve = dict(routing)
            # filters에 스코프 필터 추가
            filters_copy = dict(routing_for_retrieve.get("filters", routing_for_retrieve.get("boost", {})))
            filters_copy["_scope_filter"] = guide_filter
            routing_for_retrieve["filters"] = filters_copy
            routing_for_retrieve["boost"] = filters_copy
    
    if not sources:
        sources.update({"card_products", "service_guide_documents"})

    retrieved_docs = await retrieve_multi(
        query=query,
        routing=routing_for_retrieve,
        tables=sorted(sources),
        top_k=top_k,
    )
    filtered_docs = []
    for doc in retrieved_docs:
        doc_id = _doc_unique_id(doc)
        if doc_id and doc_id in pinned_doc_ids:
            continue
        filtered_docs.append(doc)
    total_docs = len(pinned_docs) + len(filtered_docs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(
        f"[pipeline_retrieve] retrieve_ms={elapsed_ms:.1f} doc_count={total_docs} route={route_name}"
    )
    return pinned_docs + filtered_docs


async def retrieve_consult_cases(
    query: str,
    routing: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not routing.get("need_consult_case_search"):
        return []
    try:
        matched = routing.get("matched") or {}
        intent = None
        actions = matched.get("actions") or []
        if actions:
            intent = str(actions[0])
        categories = routing.get("consult_category_candidates") or []
        return retrieve_consult_docs(
            query_text=query,
            intent=intent,
            categories=categories,
            top_k=top_k,
        )
    except Exception as exc:
        print(f"[pipeline_retrieve][consult] error={exc}")
        return []
