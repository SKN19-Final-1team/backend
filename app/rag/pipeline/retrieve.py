from __future__ import annotations

from typing import Any, Dict, List
import time

from app.rag.common.doc_source_filters import DOC_SOURCE_FILTERS
from app.rag.pipeline.utils import text_has_any_compact
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
    return _normalize_text(text).replace(" ", "").replace("-", "")


async def retrieve_docs(
    query: str,
    routing: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    start = time.perf_counter()
    filters = routing.get("filters") or routing.get("boost") or {}
    route_name = routing.get("route") or routing.get("ui_route")
    phone_lookup = bool(filters.get("phone_lookup"))
    db_route = routing.get("db_route")
    routing_for_retrieve = routing
    normalized_query = (query or "").lower()
    compact_query = _compact_text(query)
    loss_terms = ("분실", "도난", "잃어버", "분실신고", "도난신고")
    special_entities = {
        "다둥이": ["다둥이", "서울시다둥이"],
        "국민행복": ["국민행복"],
        "K-패스": ["k패스", "k-패스", "kpass", "k패스카드", "k패스체크"],
        "나라사랑": ["나라사랑"],
        "으랏차차": ["으랏차차", "으랏차"],
    }
    matched_entity = ""
    for entity, tokens in special_entities.items():
        if any(token in normalized_query or token in compact_query for token in tokens):
            matched_entity = entity
            break
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
        routing_for_retrieve["db_route"] = "guide_tbl"
        routing_for_retrieve.pop("skip_guide_with_terms_query", None)
    
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
    if filters.get("intent") or filters.get("weak_intent"):
        sources.add("service_guide_documents")
    # 분실/도난 질문은 가이드 문서만 사용해 오염을 방지
    if route_name == "card_usage" and any(term in normalized_query for term in loss_terms):
        sources = {"service_guide_documents"}
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        routing_for_retrieve["db_route"] = "guide_tbl"
        routing_for_retrieve["document_sources"] = ["guide_merged", "guide_general"]
        routing_for_retrieve["exclude_sources"] = ["card_products", "terms"]
        filters_copy = dict(routing_for_retrieve.get("filters", {}))
        filters_copy["exclude_title_terms"] = [
            "K-패스",
            "k패스",
            "다둥이",
            "혜택",
            "연회비",
            "발급",
            "추천",
            "전월",
        ]
        routing_for_retrieve["filters"] = filters_copy

    # card_info에서 card_name이 있으면 card_products만 사용
    if route_name == "card_info" and filters.get("card_name"):
        sources = {"card_products", "service_guide_documents"}
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        routing_for_retrieve["db_route"] = "card_tbl"
        routing_for_retrieve.pop("allow_guide_without_card_match", None)

    # 특수 카드 엔티티는 guide 문서도 함께 포함
    if route_name == "card_info" and matched_entity:
        sources.update({"card_products", "service_guide_documents"})
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        routing_for_retrieve["document_sources"] = ["guide_merged", "guide_general"]
        filters_copy = dict(routing_for_retrieve.get("filters", {}))
        filters_copy["card_name"] = [matched_entity]
        routing_for_retrieve["filters"] = filters_copy
        routing_for_retrieve["boost"] = filters_copy

    # card_info 기본 소스는 card_products. guide 문서는 필요 시에만 혼합
    if route_name == "card_info" and not phone_lookup:
        sources.add("card_products")
        if matched_entity or filters.get("card_name") or filters.get("intent") or filters.get("weak_intent"):
            sources.add("service_guide_documents")
    # DCC/원화결제 차단은 Apple Pay 문서 오염을 방지
    if ("dcc" in compact_query) or ("원화결제" in normalized_query) or ("원화 결제" in normalized_query):
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        filters_copy = dict(routing_for_retrieve.get("filters", {}))
        filters_copy["exclude_title_terms"] = list(
            set((filters_copy.get("exclude_title_terms") or []) + ["Apple Pay", "애플페이"])
        )
        routing_for_retrieve["filters"] = filters_copy
        routing_for_retrieve["document_sources"] = ["guide_merged", "guide_general"]

    # APPLEPAY: 애플페이 intent 감지 시 guide 문서만 사용
    applepay_intent = routing.get("applepay_intent")
    if not applepay_intent and text_has_any_compact(query, ["애플페이", "apple pay", "applepay"]):
        applepay_intent = "applepay_general"
    if applepay_intent:
        sources = {"service_guide_documents"}
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        filters_copy = dict(routing_for_retrieve.get("filters", {}))
        # 애플페이 전용 문서만 보도록 ID 프리픽스 필터 강제
        filters_copy["id_prefix"] = "hyundai_applepay"
        routing_for_retrieve["filters"] = filters_copy
        routing_for_retrieve["document_sources"] = ["hyundai_applepay"]
        routing_for_retrieve["exclude_sources"] = ["terms", "card_products"]

    # phone_lookup은 검색 폭을 최소화 (guide 문서만, 단일 스코프)
    if phone_lookup:
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        routing_for_retrieve["document_sources"] = ["guide_with_terms"]
        routing_for_retrieve["db_route"] = "guide_tbl"
        filters_copy = dict(routing_for_retrieve.get("filters", {}))
        filters_copy["exclude_title_terms"] = ["신용정보 알림서비스"]
        routing_for_retrieve["filters"] = filters_copy
        sources = {"service_guide_documents"}

    # 통신/할인/한도 질문에서는 불필요한 신용정보 알림서비스 문서 제외
    telecom_terms = {"통신", "통신요금", "자동납부", "할인", "한도", "전월", "실적"}
    if any(term in normalized_query for term in telecom_terms) and route_name == "card_info":
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        filters_copy = dict(routing_for_retrieve.get("filters", {}))
        filters_copy["exclude_title_terms"] = ["신용정보 알림서비스"]
        routing_for_retrieve["filters"] = filters_copy
        # card_info에서는 guide 문서 혼합 최소화
        routing_for_retrieve["db_route"] = "card_tbl"
    
    # 스코프 필터 적용: routing에서 받은 document_sources 기반으로 필터 생성
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
    # 분실/도난 질문은 핵심 문서를 반드시 포함
    if route_name == "card_usage" and any(term in normalized_query for term in loss_terms):
        pin_ids = ["카드분실_도난_관련피해_예방_및_대응방법_merged", "재발급 안내_merged"]
        if "나라사랑" in normalized_query:
            pin_ids.extend(["narasarang_faq_005", "narasarang_faq_006"])
        pinned = fetch_docs_by_ids("service_guide_documents", pin_ids)
        if pinned:
            retrieved_ids = {str(doc.get("id") or doc.get("db_id") or "") for doc in retrieved_docs}
            for doc in pinned:
                doc_id = str(doc.get("id") or doc.get("db_id") or "")
                if doc_id and doc_id not in retrieved_ids:
                    retrieved_docs.append(doc)
    if route_name == "card_usage" and ("나라사랑" in normalized_query) and ("재발급" in normalized_query):
        pinned = fetch_docs_by_ids("service_guide_documents", ["narasarang_faq_006"])
        if pinned:
            retrieved_ids = {str(doc.get("id") or doc.get("db_id") or "") for doc in retrieved_docs}
            for doc in pinned:
                doc_id = str(doc.get("id") or doc.get("db_id") or "")
                if doc_id and doc_id not in retrieved_ids:
                    retrieved_docs.append(doc)
    # 엔티티 guide 문서를 최소 1개 보강
    if route_name == "card_info" and matched_entity:
        guide_ids = []
        if matched_entity == "K-패스":
            guide_ids = ["k패스_2", "k패스_13", "k패스_14"]
        elif matched_entity == "다둥이":
            guide_ids = ["dadungi_013"]
        elif matched_entity == "국민행복":
            guide_ids = ["국민행복카드_28"]
        elif matched_entity == "나라사랑":
            guide_ids = ["narasarang_faq_005", "narasarang_faq_006"]
        if guide_ids:
            pinned = fetch_docs_by_ids("service_guide_documents", guide_ids)
            if pinned:
                retrieved_ids = {str(doc.get("id") or doc.get("db_id") or "") for doc in retrieved_docs}
                for doc in pinned:
                    doc_id = str(doc.get("id") or doc.get("db_id") or "")
                    if doc_id and doc_id not in retrieved_ids:
                        retrieved_docs.append(doc)
    # 예약/대출/수수료/이자 관련은 필수 문서 핀으로 보강
    if any(term in normalized_query for term in ("예약신청", "카드대출", "카드론", "현금서비스", "리볼빙", "수수료", "이자", "약관")):
        pin_ids = [
            "카드대출 예약신청_merged",
            "카드상품별_거래조건_이자율__수수료_등__merged",
            "sinhan_terms_credit_신용카드_개인회원_약관_039",
            "sinhan_terms_credit_신용카드_개인회원_약관_040",
        ]
        pinned = fetch_docs_by_ids("service_guide_documents", pin_ids)
        if pinned:
            retrieved_ids = {str(doc.get("id") or doc.get("db_id") or "") for doc in retrieved_docs}
            for doc in pinned:
                doc_id = str(doc.get("id") or doc.get("db_id") or "")
                if doc_id and doc_id not in retrieved_ids:
                    retrieved_docs.append(doc)
    filtered_docs = retrieved_docs
    total_docs = len(filtered_docs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(
        f"[pipeline_retrieve] retrieve_ms={elapsed_ms:.1f} doc_count={total_docs} route={route_name}"
    )
    return filtered_docs


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
