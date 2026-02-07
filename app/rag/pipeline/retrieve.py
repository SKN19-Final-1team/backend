from __future__ import annotations

from typing import Any, Dict, List
import time

from app.rag.common.doc_source_filters import DOC_SOURCE_FILTERS
from app.rag.pipeline.utils import text_has_any_compact
from app.rag.policy.policy_pins import build_pin_requests
from app.rag.retriever.db import fetch_docs_by_ids
from app.rag.retriever.consult_retriever import retrieve_consult_docs


DOCUMENT_SOURCE_POLICY_MAP = {
    "A": ["guide_merged", "guide_general"],
    "B": ["guide_general", "guide_merged"],
    "C": ["guide_general"],
}
DEFAULT_DOCUMENT_SOURCES = ["guide_merged", "guide_general"]

# Pin 허용 threshold
_PIN_SCORE_THRESHOLD = 0.3  # vector search 기준으로 조정


def _normalize_text(text: str) -> str:
    return (text or "").lower()


def _compact_text(text: str) -> str:
    return _normalize_text(text).replace(" ", "").replace("-", "")


def post_filter_docs(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    최소한의 후처리 필터 - vector search가 대부분 처리하므로 필수 제외 로직만 유지
    """
    if not docs:
        return docs

    q = _normalize_text(query)

    def _doc_text(doc: Dict[str, Any]) -> str:
        title = str(doc.get("title") or "")
        content = str(doc.get("content") or "")
        return f"{title} {content}".lower()

    def _is_applepay_doc(doc: Dict[str, Any]) -> bool:
        doc_id = str(doc.get("id") or doc.get("db_id") or "").lower()
        if "hyundai_applepay" in doc_id:
            return True
        text = _doc_text(doc)
        return any(t in text for t in ("애플페이", "applepay", "apple pay"))

    # 애플페이가 아닌 일반 서비스 문의에서 애플페이 문서 제외 (오염 방지)
    if not any(t in q for t in ("애플페이", "applepay", "apple pay")):
        if any(t in q for t in ("해지", "사용내역", "이용내역", "한도", "결제일", "조회", "취소")):
            filtered = [d for d in docs if not _is_applepay_doc(d)]
            if filtered:
                docs = filtered

    return docs


def _pin_allowed(
    retrieved_docs: List[Dict[str, Any]],
    budget_ms: int | None,
    start_ts: float | None,
    force: bool = False,
) -> bool:
    if force:
        return True
    if budget_ms is not None and start_ts is not None:
        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        if elapsed_ms >= budget_ms:
            return False
    if not retrieved_docs:
        return False
    top_score = retrieved_docs[0].get("score")
    if not isinstance(top_score, (int, float)):
        return False
    return top_score >= _PIN_SCORE_THRESHOLD


async def retrieve_docs_card_info(
    query: str,
    routing: Dict[str, Any],
    top_k: int,
    log_scores: bool = False,
    budget_ms: int | None = None,
    start_ts: float | None = None,
) -> List[Dict[str, Any]]:
    """
    카드 정보 검색 - 단순화된 버전, vector-first 전략 사용
    """
    return await retrieve_docs(
        query=query,
        routing=routing,
        top_k=top_k,
        budget_ms=budget_ms,
        start_ts=start_ts,
    )


async def retrieve_docs(
    query: str,
    routing: Dict[str, Any],
    top_k: int,
    budget_ms: int | None = None,
    start_ts: float | None = None,
) -> List[Dict[str, Any]]:
    """
    단순화된 검색 로직 - simple_retrieve를 활용하여 복잡도 감소
    """
    start = time.perf_counter()
    filters = routing.get("filters") or routing.get("boost") or {}
    route_name = routing.get("route") or routing.get("ui_route")
    phone_lookup = bool(filters.get("phone_lookup"))
    db_route = routing.get("db_route")
    routing_for_retrieve = routing
    normalized_query = (query or "").lower()

    # 테이블 선택 (card_products vs service_guide_documents)
    sources = set()
    if db_route == "card_tbl":
        sources.add("card_products")
        # 혜택/할인 키워드가 있으면 guide 문서도 포함
        benefit_keywords = ("할인", "혜택", "적립", "캐시백", "포인트", "마일리지")
        if any(kw in normalized_query for kw in benefit_keywords):
            sources.add("service_guide_documents")
    elif db_route == "guide_tbl":
        sources.add("service_guide_documents")
    elif db_route == "both":
        sources.update({"card_products", "service_guide_documents"})
    else:
        # 기본 라우팅
        if route_name == "card_usage":
            sources.add("service_guide_documents")
            if filters.get("card_name"):
                sources.add("card_products")
        elif route_name == "card_info":
            sources.add("card_products")
            if filters.get("intent") or filters.get("weak_intent"):
                sources.add("service_guide_documents")
        else:
            sources.update({"card_products", "service_guide_documents"})

    # APPLEPAY: 애플페이 문의는 전용 문서만 검색 (단, 분실/도난은 제외)
    applepay_intent = routing.get("applepay_intent")
    if not applepay_intent and text_has_any_compact(query, ["애플페이", "apple pay", "applepay"]):
        applepay_intent = "applepay_general"

    # 분실/도난 쿼리는 hyundai_applepay 제한 적용 안 함
    is_loss_query = any(term in normalized_query for term in ("분실", "도난", "잃어버"))

    if applepay_intent and not is_loss_query:
        sources = {"service_guide_documents"}
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        filters_copy = dict(routing_for_retrieve.get("filters", {}))
        filters_copy["id_prefix"] = "hyundai_applepay"
        routing_for_retrieve["filters"] = filters_copy
        routing_for_retrieve["document_sources"] = ["hyundai_applepay"]
    elif applepay_intent and is_loss_query:
        # Apple Pay + 분실/도난: intent 필터 완전 제거
        # "5. 보안_merged" 같은 일반 보안 문서 검색을 위해
        # vector search가 알아서 관련 문서를 찾도록 함
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        filters_copy = dict(routing_for_retrieve.get("filters", {}))
        # intent 필터 완전 제거 (너무 엄격함)
        filters_copy.pop("intent", None)
        filters_copy.pop("weak_intent", None)
        # payment_method는 유지 (애플페이 관련성 보장)
        routing_for_retrieve["filters"] = filters_copy

    # phone_lookup: 전화번호 검색은 guide만
    if phone_lookup:
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        routing_for_retrieve["document_sources"] = ["guide_with_terms"]
        routing_for_retrieve["db_route"] = "guide_tbl"
        sources = {"service_guide_documents"}

    # 문서 소스 스코프 필터 적용
    document_source_policy = routing.get("document_source_policy")
    document_sources = routing.get("document_sources")
    if not document_sources:
        document_sources = DOCUMENT_SOURCE_POLICY_MAP.get(document_source_policy, DEFAULT_DOCUMENT_SOURCES)

    if document_sources and "service_guide_documents" in sources:
        has_merged = "guide_merged" in document_sources
        has_general = "guide_general" in document_sources
        has_terms_all = "guide_with_terms" in document_sources

        if has_terms_all:
            guide_filter = DOC_SOURCE_FILTERS.get("guide_with_terms")
        elif has_merged and has_general:
            guide_filter = DOC_SOURCE_FILTERS.get("guide_all")
        elif has_merged:
            guide_filter = DOC_SOURCE_FILTERS.get("guide_merged")
        elif has_general:
            guide_filter = DOC_SOURCE_FILTERS.get("guide_general")
        else:
            guide_filter = None

        if guide_filter:
            if routing_for_retrieve is routing:
                routing_for_retrieve = dict(routing)
            filters_copy = dict(routing_for_retrieve.get("filters", routing_for_retrieve.get("boost", {})))
            filters_copy["_scope_filter"] = guide_filter
            routing_for_retrieve["filters"] = filters_copy

    if not sources:
        sources.update({"card_products", "service_guide_documents"})

    # 검색 실행: simple_retrieve (vector-first)
    from app.rag.retriever.simple_retriever import simple_retrieve
    retrieved_docs = simple_retrieve(
        query=query,
        routing=routing_for_retrieve,
        tables=sorted(sources),
        top_k=top_k,
    )

    # card_usage: 결과가 없거나 점수가 낮으면 vector 재시도
    if (
        route_name == "card_usage"
        and routing_for_retrieve.get("retrieval_mode") != "vector"
        and (routing_for_retrieve.get("document_sources") or []) != ["guide_with_terms"]
    ):
        top_score = retrieved_docs[0].get("score") if retrieved_docs else None
        if (not retrieved_docs) or (isinstance(top_score, (int, float)) and top_score < 0.1):
            vector_routing = dict(routing_for_retrieve)
            vector_routing["retrieval_mode"] = "vector"
            retrieved_docs = simple_retrieve(
                query=query,
                routing=vector_routing,
                tables=sorted(sources),
                top_k=top_k,
            )

    # 후처리 필터 적용
    if route_name == "card_usage":
        retrieved_docs = post_filter_docs(query, retrieved_docs)

    # 점수 정규화 및 정렬
    for doc in retrieved_docs:
        if not isinstance(doc.get("score"), (int, float)):
            doc["score"] = 0.0
    retrieved_docs.sort(key=lambda d: d.get("score", 0.0), reverse=True)

    # 핀 로직: 중요 문서 보강
    loss_terms = ("분실", "도난", "잃어버", "분실신고", "도난신고")
    critical_pin = False
    matched_entity = ""

    # 특수 엔티티 매칭 (핀 로직에만 사용)
    special_entities = {
        "다둥이": ["다둥이", "서울시다둥이"],
        "국민행복": ["국민행복"],
        "K-패스": ["k패스", "k-패스", "kpass"],
        "나라사랑": ["나라사랑"],
        "으랏차차": ["으랏차차", "으랏차"],
    }
    for entity, tokens in special_entities.items():
        if any(token in normalized_query for token in tokens):
            matched_entity = entity
            break

    if route_name == "card_usage":
        if any(term in normalized_query for term in loss_terms):
            critical_pin = True
        if "나라사랑" in normalized_query:
            critical_pin = True
        if any(term in normalized_query for term in ("예약신청", "카드대출", "리볼빙", "수수료", "이자")):
            critical_pin = True
        if matched_entity:
            critical_pin = True
        if phone_lookup or ("전화" in normalized_query) or ("번호" in normalized_query):
            critical_pin = True
    elif route_name == "card_info" and matched_entity:
        critical_pin = True

    pin_max = 2 if critical_pin else 1
    if route_name == "card_info" and matched_entity == "K-패스":
        pin_max = max(pin_max, 3)
    pin_allowed = _pin_allowed(retrieved_docs, budget_ms, start_ts, force=critical_pin)
    pinned_added = 0

    def _append_pins(pinned_docs: List[Dict[str, Any]]) -> None:
        nonlocal pinned_added, retrieved_docs
        if pinned_added >= pin_max or not pinned_docs:
            return
        retrieved_index: Dict[str, int] = {}
        for idx, doc in enumerate(retrieved_docs):
            doc_id = str(doc.get("id") or doc.get("db_id") or "")
            if doc_id:
                retrieved_index[doc_id] = idx
        for doc in pinned_docs:
            doc_id = str(doc.get("id") or doc.get("db_id") or "")
            if not doc_id:
                continue
            if doc_id in retrieved_index:
                existing = dict(retrieved_docs[retrieved_index[doc_id]])
                existing["_pinned"] = True
                if "_pin_rank" in doc:
                    existing["_pin_rank"] = doc["_pin_rank"]
                retrieved_docs[retrieved_index[doc_id]] = existing
                continue
            if pinned_added < pin_max:
                pinned_doc = dict(doc)
                pinned_doc["_pinned"] = True
                retrieved_docs.append(pinned_doc)
                pinned_added += 1
                retrieved_index[doc_id] = len(retrieved_docs) - 1
                if pinned_added >= pin_max:
                    return

    def _mark_pin_rank(pinned_docs: List[Dict[str, Any]], pin_ids: List[str]) -> List[Dict[str, Any]]:
        if not pinned_docs:
            return pinned_docs
        rank_map = {str(pid): idx for idx, pid in enumerate(pin_ids)}
        marked: List[Dict[str, Any]] = []
        for doc in pinned_docs:
            doc_id = str(doc.get("id") or doc.get("db_id") or "")
            marked_doc = dict(doc)
            if doc_id in rank_map:
                marked_doc["_pin_rank"] = rank_map[doc_id]
            marked.append(marked_doc)
        marked.sort(key=lambda d: d.get("_pin_rank", 10**9))
        return marked

    # 분실/도난 핀
    if pin_allowed and route_name == "card_usage" and any(term in normalized_query for term in loss_terms):
        if "나라사랑" in normalized_query:
            pin_ids = ["narasarang_faq_005", "narasarang_faq_006", "카드분실_도난_관련피해_예방_및_대응방법_merged"]
        else:
            pin_ids = ["카드분실_도난_관련피해_예방_및_대응방법_merged"]
        pinned = fetch_docs_by_ids("service_guide_documents", pin_ids)
        _append_pins(_mark_pin_rank(pinned or [], pin_ids))

    # 정책 기반 핀
    for table, pin_ids in build_pin_requests(
        route_name=route_name,
        normalized_query=normalized_query,
        matched_entity=matched_entity,
        pin_allowed=pin_allowed,
    ):
        pinned = fetch_docs_by_ids(table, pin_ids)
        _append_pins(_mark_pin_rank(pinned or [], pin_ids))

    elapsed_ms = (time.perf_counter() - start) * 1000
    return retrieved_docs


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
    except Exception:
        return []
