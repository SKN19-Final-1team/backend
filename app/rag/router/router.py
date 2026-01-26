from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.rag.router.rules import decide_route, match_force_rule, ROUTER_FORCE_RULES
from app.rag.router.signals import extract_signals, Signals
from app.rag.vocab.keyword_dict import ROUTE_CARD_USAGE


@dataclass(frozen=True)
class RouterResult:
    route: Optional[str]
    filters: Dict[str, Any]
    ui_route: Optional[str]
    db_route: Optional[str]
    boost: Dict[str, Any]
    query_template: Optional[str]
    matched: Dict[str, Any]
    applepay_intent: Optional[str]
    should_search: bool
    should_trigger: bool
    should_route: bool
    document_sources: list
    exclude_sources: list
    document_source_policy: str
    need_consult_case_search: bool
    consult_category_candidates: list
    consult_keyword_hits: int


_CONSULT_DOMAIN_KEYWORDS = {
    "분실",
    "재발급",
    "승인",
    "취소",
    "수수료",
    "한도",
    "현금서비스",
    "카드론",
    "리볼빙",
    "결제",
    "오류",
    "에러",
    "불가",
    "거절",
}


_PHONE_LOOKUP_TERMS = {
    "전화번호",
    "고객센터",
    "콜센터",
    "ars",
    "연락처",
    "상담원",
    "대표번호",
    "문의전화",
    "문의 전화",
}

_CARDINFO_TERMS = {
    "괜찮",
    "좋아",
    "혜택",
    "추천",
    "연회비",
    "조건",
    "신청",
    "발급",
    "한도",
    "실적",
    "할인",
    "서류",
    "방법",
}

_LOSS_STRONG_TERMS = {
    "분실",
    "잃어버",
    "도난",
    "훔",
    "없어졌",
    "주운",
    "주웠",
}

_LOSS_ACTION_TERMS = {
    "정지",
    "신고",
    "재발급",
    "부정사용",
    "승인",
    "결제됐",
    "피해",
}

_CARDNAME_BAD_TOKENS = {
    "좋아요",
    "괜찮",
    "추천",
    "혜택",
    "연회비",
    "조건",
    "신청",
    "서류",
    "방법",
    "얼마",
    "한도",
    "할인",
}

_KPASS_TERMS = {"k패스", "k-pass", "케이패스"}
_KPASS_BENEFIT = {
    "다자녀": ["다자녀", "2자녀", "세자녀", "자녀", "미성년 자녀"],
    "체크": ["체크", "체크카드"],
    "청년": ["청년", "만 19", "만19", "만 34", "만34"],
}
_REGION_MAP = {
    "경기": ["경기", "경기도"],
    "충남": ["충남", "충청남도"],
    "충북": ["충북", "충청북도"],
    "서울": ["서울", "서울시"],
}


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _is_phone_lookup(normalized: str) -> bool:
    return _contains_any(normalized, _PHONE_LOOKUP_TERMS)


def _is_loss_intent(normalized: str) -> bool:
    strong = _contains_any(normalized, _LOSS_STRONG_TERMS)
    action = _contains_any(normalized, _LOSS_ACTION_TERMS)
    info_like = _contains_any(normalized, _CARDINFO_TERMS)
    if strong and not info_like:
        return True
    return strong and action


def _is_plausible_card_name(name: str) -> bool:
    lowered = (name or "").lower()
    if any(token in lowered for token in _CARDNAME_BAD_TOKENS):
        return False
    return len(lowered) < 18


def _extract_kpass_region(normalized: str) -> Optional[str]:
    for key, pats in _REGION_MAP.items():
        if any(p in normalized for p in pats):
            return key
    return None


def _extract_kpass_benefits(normalized: str) -> list[str]:
    hits = []
    for key, pats in _KPASS_BENEFIT.items():
        if any(p in normalized for p in pats):
            hits.append(key)
    return hits


def _count_domain_keyword_hits(normalized: str) -> int:
    if not normalized:
        return 0
    hits = {term for term in _CONSULT_DOMAIN_KEYWORDS if term in normalized}
    return len(hits)


def _build_consult_category_candidates(signals: Signals) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for items in (signals.actions, signals.weak_intents, signals.pattern_hits):
        for item in items or []:
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def route_query(query: str) -> Dict[str, Optional[object]]:
    signals = extract_signals(query)
    force_rule = match_force_rule(signals.normalized)
    normalized = signals.normalized
    card_names = [name for name in signals.card_names if _is_plausible_card_name(name)]
    actions = list(signals.actions)
    if actions and not _is_loss_intent(normalized):
        actions = [a for a in actions if "분실" not in a and "도난" not in a]
    consult_keyword_hits = _count_domain_keyword_hits(signals.normalized)
    consult_category_candidates = _build_consult_category_candidates(signals)
    need_consult_case_search = bool(
        actions or signals.payments or signals.weak_intents
    )

    if _is_phone_lookup(normalized):
        return RouterResult(
            route="card_usage",
            filters={"intent": ["phone_lookup"], "phone_lookup": True},
            ui_route="phone_lookup",
            db_route="guide_tbl",
            boost={"intent": ["phone_lookup"], "phone_lookup": True},
            query_template=None,
            matched={
                "card_names": card_names,
                "actions": actions,
                "payments": signals.payments,
                "weak_intents": signals.weak_intents,
            },
            applepay_intent=signals.applepay_intent,
            should_search=True,
            should_trigger=True,
            should_route=True,
            document_sources=["guide_merged", "guide_general"],
            exclude_sources=["terms"],
            document_source_policy="B",
            need_consult_case_search=False,
            consult_category_candidates=consult_category_candidates,
            consult_keyword_hits=consult_keyword_hits,
        ).__dict__

    if force_rule:
        return RouterResult(
            route=force_rule["route"],
            filters={},
            ui_route=force_rule["route"],
            db_route="card_tbl",
            boost={},
            query_template=None,
            matched={
                "card_names": card_names,
                "actions": actions,
                "payments": signals.payments,
                "weak_intents": signals.weak_intents,
            },
            applepay_intent=signals.applepay_intent,
            should_search=True,
            should_trigger=True,
            should_route=True,
            document_sources=[],
            exclude_sources=["terms"],
            document_source_policy="C",
            need_consult_case_search=need_consult_case_search,
            consult_category_candidates=consult_category_candidates,
            consult_keyword_hits=consult_keyword_hits,
        ).__dict__

    (
        ui_route,
        db_route,
        boost,
        query_template,
        should_trigger,
        should_search,
        document_sources,
        exclude_sources,
        document_source_policy,
    ) = decide_route(signals)
    if ui_route == ROUTE_CARD_USAGE:
        need_consult_case_search = True

    filters = dict(boost or {})
    # sanitize card_name/intent hints for info-like queries
    if "card_name" in filters:
        raw_names = filters.get("card_name") or []
        if isinstance(raw_names, str):
            raw_names = [raw_names]
        cleaned = [name for name in raw_names if _is_plausible_card_name(str(name))]
        if cleaned:
            filters["card_name"] = cleaned
        else:
            filters.pop("card_name", None)
    if not _is_loss_intent(normalized):
        for key in ("intent", "weak_intent"):
            values = filters.get(key) or []
            if isinstance(values, str):
                values = [values]
            filtered = [
                v for v in values if isinstance(v, str) and ("분실" not in v and "도난" not in v)
            ]
            if filtered:
                filters[key] = filtered
            else:
                filters.pop(key, None)
        if _contains_any(normalized, _CARDINFO_TERMS):
            filters.pop("intent", None)
            filters.pop("weak_intent", None)
    if _contains_any(normalized, _KPASS_TERMS):
        filters = dict(filters)
        filters.setdefault("card_name", ["K-패스"])
        region = _extract_kpass_region(normalized)
        if region:
            filters["region"] = [region]
        benefits = _extract_kpass_benefits(normalized)
        if benefits:
            filters["benefit_type"] = benefits
        boost = filters
    else:
        boost = filters

    matched_card_names = list(card_names)
    filter_card_names = filters.get("card_name") or []
    if isinstance(filter_card_names, str):
        filter_card_names = [filter_card_names]
    for name in filter_card_names:
        if name and name not in matched_card_names:
            matched_card_names.append(name)

    return RouterResult(
        route=ui_route,
        filters=filters,
        ui_route=ui_route,
        db_route=db_route,
        boost=boost,
        query_template=query_template,
        matched={
            "card_names": matched_card_names,
            "actions": actions,
            "payments": signals.payments,
            "weak_intents": signals.weak_intents,
        },
        applepay_intent=signals.applepay_intent,
        should_search=should_search,
        should_trigger=should_trigger,
        should_route=should_trigger,
        document_sources=document_sources,
        exclude_sources=exclude_sources,
        document_source_policy=document_source_policy,
        need_consult_case_search=need_consult_case_search,
        consult_category_candidates=consult_category_candidates,
        consult_keyword_hits=consult_keyword_hits,
    ).__dict__


__all__ = ["route_query", "RouterResult", "ROUTER_FORCE_RULES", "Signals"]
