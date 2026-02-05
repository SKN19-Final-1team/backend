from __future__ import annotations

from typing import Any, Dict, List
import time

from app.rag.common.doc_source_filters import DOC_SOURCE_FILTERS
from app.rag.pipeline.utils import text_has_any_compact
from app.rag.policy.policy_pins import build_pin_requests
from app.rag.retriever.retriever import retrieve_multi
from app.rag.retriever.db import fetch_docs_by_ids
from app.rag.retriever.consult_retriever import retrieve_consult_docs


_CARD_INFO_T_HIGH = 0.35
_CARD_INFO_GAP = 0.08
_CARD_INFO_T_LOW = 0.22
_CARD_INFO_GAP_LOW = 0.04

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


def post_filter_docs(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not docs:
        return docs
    q = _normalize_text(query)
    qc = _compact_text(query)

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

    loss_terms = ("분실", "도난", "잃어버", "분실신고")
    loan_terms = ("대출", "현금서비스", "카드대출", "리볼빙", "이자", "수수료")
    dcc_terms = ("dcc", "원화결제", "원화 결제", "해외 원화")
    apple_terms = ("애플페이", "applepay", "apple pay")
    samsung_terms = ("삼성페이", "samsungpay", "samsung pay")
    kakao_terms = ("카카오페이", "kakaopay", "kakao pay")
    tmoney_terms = ("티머니", "tmoney", "t-money", "교통카드")

    loss_ids = {
        "narasarang_faq_005",
        "narasarang_faq_006",
        "카드분실_도난_관련피해_예방_및_대응방법_merged",
    }
    loan_ids = {
        "카드대출 예약신청_merged",
        "카드상품별_거래조건_이자율__수수료_등__merged",
        "sinhan_terms_credit_신용카드_개인회원_약관_039",
        "sinhan_terms_credit_신용카드_개인회원_약관_040",
    }

    def _filter_by_terms(terms: tuple[str, ...], id_whitelist: set[str]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for doc in docs:
            doc_id = str(doc.get("id") or doc.get("db_id") or "")
            text = _doc_text(doc)
            if doc_id in id_whitelist or any(t in text for t in terms):
                filtered.append(doc)
        return filtered

    # loss intent
    if any(t in q for t in loss_terms):
        filtered = _filter_by_terms(loss_terms, loss_ids)
        if "나라사랑" in q:
            filtered = [
                d
                for d in filtered
                if "narasarang" in str(d.get("id") or d.get("db_id") or "").lower()
                or "나라사랑" in _doc_text(d)
                or str(d.get("id") or d.get("db_id") or "") in loss_ids
            ] or filtered
        if filtered:
            docs = filtered
        for doc in docs:
            doc_id = str(doc.get("id") or doc.get("db_id") or "")
            if doc_id == "narasarang_faq_005":
                doc["score"] = float(doc.get("score") or 0.0) + 2.0
            elif doc_id in loss_ids:
                doc["score"] = float(doc.get("score") or 0.0) + 1.0
            elif "나라사랑" in q and "narasarang" in doc_id.lower():
                doc["score"] = float(doc.get("score") or 0.0) + 0.5

    # reissue intent (나라사랑 재발급 문서 상단 고정)
    if "나라사랑" in q and any(t in q for t in ("재발급", "재발행")):
        for doc in docs:
            doc_id = str(doc.get("id") or doc.get("db_id") or "")
            if doc_id == "narasarang_faq_006":
                doc["score"] = float(doc.get("score") or 0.0) + 2.0

    # loan intent (skip for cancellation/close flows)
    if any(t in q for t in loan_terms) and not any(t in q for t in ("취소", "해지", "해제", "철회")):
        filtered = _filter_by_terms(loan_terms, loan_ids)
        if filtered:
            docs = filtered
        for doc in docs:
            doc_id = str(doc.get("id") or doc.get("db_id") or "")
            if doc_id in loan_ids:
                doc["score"] = float(doc.get("score") or 0.0) + 1.0

    # cancel/usage/limit/billing intents: remove generic fee/terms docs when not asked
    if any(t in q for t in ("해지", "취소", "사용내역", "이용내역", "거래내역", "결제일", "한도", "조회")):
        if not any(t in q for t in ("이자", "수수료", "금리", "연체")):
            fee_terms = ("이자율", "수수료", "거래조건", "카드상품별")
            filtered = [d for d in docs if not any(t in _doc_text(d) for t in fee_terms)]
            if filtered:
                docs = filtered

    # billing day intent
    if any(t in q for t in ("결제일", "결제일자", "납부일", "청구일")):
        filtered = _filter_by_terms(
            (
                "결제일",
                "결제일자",
                "결제예정일",
                "결제 예정일",
                "결제일 변경",
                "결제일 안내",
                "납부일",
                "청구일",
                "청구 예정일",
            ),
            set(),
        )
        if filtered:
            docs = filtered

    # limit intent
    if any(t in q for t in ("한도", "이용한도", "신용한도", "한도조회")):
        filtered = _filter_by_terms(
            ("한도", "이용한도", "신용한도", "한도조회", "한도 조회", "한도 상향"),
            set(),
        )
        if filtered:
            docs = filtered

    # usage history intent
    if any(t in q for t in ("사용내역", "이용내역", "거래내역", "결제내역")):
        filtered = _filter_by_terms(
            ("사용내역", "이용내역", "거래내역", "결제내역", "내역 조회"),
            set(),
        )
        if filtered:
            docs = filtered

    # cancel/close intent
    if any(t in q for t in ("해지", "탈회", "해제")):
        filtered = _filter_by_terms(
            ("카드 해지", "해지", "탈회", "해제", "회원 탈회", "해지 절차"),
            set(),
        )
        if filtered:
            docs = filtered

    # revolving intent
    if any(t in q for t in ("리볼빙", "일부결제")):
        filtered = _filter_by_terms(
            (
                "리볼빙",
                "일부결제",
                "일부결제금액이월",
                "일부결제대금이월약정",
                "일부결제금액이월약정",
            ),
            set(),
        )
        if filtered:
            docs = filtered
        if any(t in q for t in ("취소", "해지")):
            filtered = [
                d
                for d in docs
                if any(t in _doc_text(d) for t in ("취소", "해지", "해제", "철회", "중단"))
            ]
            if filtered:
                docs = filtered

    # payment intents
    if any(t in q for t in apple_terms):
        filtered = [
            d
            for d in docs
            if "hyundai_applepay" in str(d.get("id") or d.get("db_id") or "").lower()
            or any(t in _doc_text(d) for t in apple_terms)
        ]
        if filtered:
            docs = filtered
    else:
        # 비(애플페이) 문의에서 애플페이 문서가 상위로 뜨는 경우 제거
        if any(t in q for t in ("해지", "사용내역", "이용내역", "한도", "결제일", "리볼빙", "조회", "취소")):
            filtered = [d for d in docs if not _is_applepay_doc(d)]
            if filtered:
                docs = filtered
    if any(t in q for t in samsung_terms):
        filtered = [d for d in docs if any(t in _doc_text(d) for t in samsung_terms)]
        if filtered:
            docs = filtered
    if any(t in q for t in kakao_terms):
        filtered = [d for d in docs if any(t in _doc_text(d) for t in kakao_terms)]
        if filtered:
            docs = filtered
    if any(t in q or t in qc for t in tmoney_terms):
        filtered = [d for d in docs if "티머니" in _doc_text(d) or "tmoney" in _doc_text(d)]
        if filtered:
            docs = filtered
    if any(t in q for t in dcc_terms):
        filtered = [d for d in docs if any(t in _doc_text(d) for t in dcc_terms)]
        if filtered:
            docs = filtered

    # 조회 intent: 조회/확인 문구 포함 문서 우선
    if "조회" in q:
        filtered = [d for d in docs if any(t in _doc_text(d) for t in ("조회", "확인"))]
        if filtered:
            docs = filtered

    def _apply_term_bias(terms: tuple[str, ...], boost: float, penalty: float) -> None:
        for doc in docs:
            text = _doc_text(doc)
            score = float(doc.get("score") or 0.0)
            if any(t in text for t in terms):
                doc["score"] = score + boost
            else:
                doc["score"] = score - penalty

    # intent-specific biasing to surface relevant docs
    if "리볼빙" in q and any(t in q for t in ("취소", "해지")):
        _apply_term_bias(("취소", "해지", "해제", "철회"), boost=0.6, penalty=0.4)
        # avoid fee/terms heavy docs when asking to cancel
        fee_terms = ("거래조건", "이자율", "수수료", "약관")
        for doc in docs:
            text = _doc_text(doc)
            if any(t in text for t in fee_terms):
                doc["score"] = float(doc.get("score") or 0.0) - 0.8
    if any(t in q for t in ("해지", "탈회")) and "리볼빙" not in q:
        _apply_term_bias(("해지", "탈회", "해제"), boost=0.5, penalty=0.3)
    if any(t in q for t in ("사용내역", "이용내역", "거래내역", "결제내역")):
        _apply_term_bias(("사용내역", "이용내역", "거래내역", "결제내역"), boost=0.6, penalty=0.4)
    if "한도" in q and "조회" in q:
        _apply_term_bias(("한도", "이용한도", "한도조회", "한도 조회"), boost=0.5, penalty=0.3)

    return docs


def _card_group_key(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata") or {}
    card_name = meta.get("card_name") or meta.get("original_card_name") or doc.get("title") or ""
    return str(card_name).replace(" ", "").lower()


def _card_info_should_stop_lex(docs: List[Dict[str, Any]]) -> bool:
    if not docs:
        return False
    top1 = docs[0]
    top1_score = top1.get("score")
    if not isinstance(top1_score, (int, float)):
        return False
    top2_score = None
    if len(docs) > 1:
        top2 = docs[1]
        if isinstance(top2.get("score"), (int, float)):
            top2_score = float(top2.get("score"))
    if top1_score >= _CARD_INFO_T_HIGH:
        if top2_score is None:
            return True
        if (top1_score - top2_score) >= _CARD_INFO_GAP:
            return True
    if len(docs) >= 3:
        keys = [_card_group_key(d) for d in docs[:3]]
        if keys[0] and all(k == keys[0] for k in keys[1:]):
            return True
    return False


def _card_info_should_vector(docs: List[Dict[str, Any]]) -> bool:
    if not docs:
        return True
    top1_score = docs[0].get("score")
    top2_score = docs[1].get("score") if len(docs) > 1 else None
    if not isinstance(top1_score, (int, float)):
        return True
    if top1_score < _CARD_INFO_T_LOW:
        return True
    if top2_score is not None and (top1_score - top2_score) < _CARD_INFO_GAP_LOW:
        return True
    keys = [_card_group_key(d) for d in docs[:3]]
    if len({k for k in keys if k}) > 1:
        return True
    return False


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
    return top_score >= _CARD_INFO_T_HIGH


async def retrieve_docs_card_info(
    query: str,
    routing: Dict[str, Any],
    top_k: int,
    log_scores: bool = False,
    budget_ms: int | None = None,
    start_ts: float | None = None,
) -> List[Dict[str, Any]]:
    first_pass_routing = dict(routing)
    first_pass_routing["retrieval_mode"] = "keyword_only"
    docs = await retrieve_docs(
        query=query,
        routing=first_pass_routing,
        top_k=min(top_k, 3),
        budget_ms=budget_ms,
        start_ts=start_ts,
    )
    if not _card_info_should_stop_lex(docs) and _card_info_should_vector(docs):
        if budget_ms is not None and start_ts is not None:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000
            if elapsed_ms >= budget_ms:
                return docs
        vector_routing = dict(routing)
        vector_routing["retrieval_mode"] = "vector"
        docs = await retrieve_docs(
            query=query,
            routing=vector_routing,
            top_k=min(top_k + 2, 6),
            budget_ms=budget_ms,
            start_ts=start_ts,
        )
    return docs


async def retrieve_docs(
    query: str,
    routing: Dict[str, Any],
    top_k: int,
    budget_ms: int | None = None,
    start_ts: float | None = None,
) -> List[Dict[str, Any]]:
    start = time.perf_counter()
    filters = routing.get("filters") or routing.get("boost") or {}
    if routing.get("route") == "card_info" or routing.get("ui_route") == "card_info":
        if "_skip_db_fallback" not in filters:
            filters = dict(filters)
            filters["_skip_db_fallback"] = True
            routing = dict(routing)
            routing["filters"] = filters
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

    # service intents should avoid terms-heavy docs (unless explicitly asking about fees)
    _SERVICE_INTENT_TERMS = (
        "해지",
        "취소",
        "사용내역",
        "이용내역",
        "거래내역",
        "결제일",
        "한도",
        "조회",
    )
    _FEE_INTENT_TERMS = ("이자", "수수료", "금리", "연체", "약관", "요율", "거래조건")
    if (
        route_name == "card_usage"
        and any(term in normalized_query for term in _SERVICE_INTENT_TERMS)
        and not any(term in normalized_query for term in _FEE_INTENT_TERMS)
    ):
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        routing_for_retrieve["db_route"] = "guide_tbl"
        routing_for_retrieve["document_sources"] = ["guide_merged", "guide_general"]
        routing_for_retrieve["exclude_sources"] = ["terms"]
        filters_copy = dict(routing_for_retrieve.get("filters", {}))
        exclude_terms = list(filters_copy.get("exclude_title_terms") or [])
        for term in ("거래조건", "이자율", "수수료", "약관"):
            if term not in exclude_terms:
                exclude_terms.append(term)
        filters_copy["exclude_title_terms"] = exclude_terms
        routing_for_retrieve["filters"] = filters_copy
    
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
        routing_for_retrieve["allow_guide_without_card_match"] = True
        filters_copy = dict(routing_for_retrieve.get("filters", {}))
        if not filters_copy.get("intent"):
            filters_copy["intent"] = ["분실도난"]
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

    # 비(애플페이) 문의에서 애플페이 문서 제외
    if route_name == "card_usage" and not any(t in normalized_query for t in ("애플페이", "apple pay", "applepay")):
        if any(t in normalized_query for t in ("해지", "사용내역", "이용내역", "한도", "결제일", "리볼빙", "조회", "취소")):
            if routing_for_retrieve is routing:
                routing_for_retrieve = dict(routing)
            filters_copy = dict(routing_for_retrieve.get("filters", {}))
            exclude_like_any = filters_copy.get("exclude_like_any") or []
            if not isinstance(exclude_like_any, list):
                exclude_like_any = list(exclude_like_any)
            exclude_like_any.extend(["hyundai_applepay%", "%애플페이%", "%apple pay%", "%applepay%"])
            filters_copy["exclude_like_any"] = exclude_like_any
            routing_for_retrieve["filters"] = filters_copy

    # card_info에서 card_name이 있으면 기본은 card_products, 다만 혜택/실적/할인/통신 등은 guide도 허용
    if route_name == "card_info" and filters.get("card_name"):
        benefit_tokens = ("혜택", "할인", "전월", "실적", "통신", "자동납부", "적립", "캐시백")
        needs_guide = text_has_any_compact(query, benefit_tokens)
        sources = {"card_products", "service_guide_documents"}
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        routing_for_retrieve["db_route"] = "both" if needs_guide else "card_tbl"
        if needs_guide:
            routing_for_retrieve["allow_guide_without_card_match"] = True
        else:
            routing_for_retrieve.pop("allow_guide_without_card_match", None)

    # 특수 카드 엔티티는 guide 문서도 함께 포함
    if route_name == "card_info" and matched_entity:
        sources.update({"card_products", "service_guide_documents"})
        if routing_for_retrieve is routing:
            routing_for_retrieve = dict(routing)
        routing_for_retrieve["document_sources"] = ["guide_merged", "guide_general"]
        filters_copy = dict(routing_for_retrieve.get("filters", {}))
        filters_copy["card_name"] = [matched_entity]
        if matched_entity == "K-패스" and "다자녀" not in normalized_query:
            exclude_terms = list(filters_copy.get("exclude_title_terms") or [])
            if "다자녀" not in exclude_terms:
                exclude_terms.append("다자녀")
            filters_copy["exclude_title_terms"] = exclude_terms
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
    # card_usage는 조건부로 vector 1회만 허용
    if (
        route_name == "card_usage"
        and routing_for_retrieve.get("retrieval_mode") != "vector"
        and (routing_for_retrieve.get("document_sources") or []) != ["guide_with_terms"]
    ):
        top_score = retrieved_docs[0].get("score") if retrieved_docs else None
        if (not retrieved_docs) or (isinstance(top_score, (int, float)) and top_score < 0.1):
            vector_routing = dict(routing_for_retrieve)
            vector_routing["retrieval_mode"] = "vector"
            retrieved_docs = await retrieve_multi(
                query=query,
                routing=vector_routing,
                tables=sorted(sources),
                top_k=top_k,
            )
    # filter noisy K-패스 docs for loss/loan queries early
    def _is_kpass_doc(doc: Dict[str, Any]) -> bool:
        title = str(doc.get("title") or "").lower()
        content = str(doc.get("content") or "").lower()
        return "k패스" in title or "k-패스" in title or "k패스" in content or "k-패스" in content

    loan_terms = ("대출", "현금서비스", "카드대출", "리볼빙")
    loss_terms_query = ("분실", "도난", "잃어버")
    if any(term in normalized_query for term in loan_terms + loss_terms_query):
        filtered = [doc for doc in retrieved_docs if not _is_kpass_doc(doc)]
        retrieved_docs = filtered or retrieved_docs

    if route_name == "card_usage":
        retrieved_docs = post_filter_docs(query, retrieved_docs)

    # ensure score is numeric and sort by score desc
    for doc in retrieved_docs:
        if not isinstance(doc.get("score"), (int, float)):
            doc["score"] = 0.0
    retrieved_docs.sort(key=lambda d: d.get("score", 0.0), reverse=True)

    critical_pin = False
    if route_name == "card_usage":
        if any(term in normalized_query for term in loss_terms):
            critical_pin = True
        if "나라사랑" in normalized_query:
            critical_pin = True
        if any(
            term in normalized_query
            for term in ("예약신청", "카드대출", "카드론", "현금서비스", "리볼빙", "수수료", "이자", "약관")
        ):
            critical_pin = True
        if matched_entity in {"다둥이", "국민행복", "나라사랑", "K-패스"}:
            critical_pin = True
        if phone_lookup or ("전화" in normalized_query) or ("번호" in normalized_query) or ("고객센터" in normalized_query):
            critical_pin = True
    elif route_name == "card_info" and matched_entity:
        # 카드명/프로그램 매칭이 잡힌 card_info는 핀을 강제로 보강
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
                # 이미 있는 문서는 핀 마킹만 갱신
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

    # 분실/도난 질문은 핵심 문서를 반드시 포함
    if pin_allowed and route_name == "card_usage" and any(term in normalized_query for term in loss_terms):
        if "나라사랑" in normalized_query:
            pin_ids = ["narasarang_faq_005", "narasarang_faq_006", "카드분실_도난_관련피해_예방_및_대응방법_merged", "재발급 안내_merged"]
        else:
            pin_ids = ["카드분실_도난_관련피해_예방_및_대응방법_merged", "재발급 안내_merged"]
        pinned = fetch_docs_by_ids("service_guide_documents", pin_ids)
        _append_pins(_mark_pin_rank(pinned or [], pin_ids))
    for table, pin_ids in build_pin_requests(
        route_name=route_name,
        normalized_query=normalized_query,
        matched_entity=matched_entity,
        pin_allowed=pin_allowed,
    ):
        pinned = fetch_docs_by_ids(table, pin_ids)
        _append_pins(_mark_pin_rank(pinned or [], pin_ids))
    filtered_docs = retrieved_docs
    total_docs = len(filtered_docs)
    elapsed_ms = (time.perf_counter() - start) * 1000
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
    except Exception:
        return []
