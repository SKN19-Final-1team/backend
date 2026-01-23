from typing import List, Optional, Tuple

from app.rag.common.doc_source_filters import DOC_SOURCE_FILTERS
from app.rag.vocab.keyword_dict import ROUTE_CARD_INFO, ROUTE_CARD_USAGE

_LOSS_THEFT_INTENTS = {"분실", "도난", "분실도난", "도난분실", "잃어버", "잃음"}
_ERROR_INTENTS = {"오류", "에러", "안돼", "불가", "거절", "되지않음"}
_REGISTRATION_INTENTS = {"등록", "추가", "신청", "인증"}
_TERMS_TRIGGERS = {
    "이자",
    "수수료",
    "연체",
    "리볼빙",
    "약관",
    "요율",
    "적용",
    "적용대상",
    "거래조건",
    "한도",
    "금리",
}


def decide_document_sources(
    applepay_intent: Optional[str],
    ui_route: str,
    actions: List[str],
    card_names: List[str],
    terms_trigger: bool,
) -> Tuple[List[str], List[str]]:
    if applepay_intent:
        return (["hyundai_applepay", "guide_general"], ["terms", "card_products"])
    if ui_route == ROUTE_CARD_USAGE and terms_trigger:
        return (["guide_with_terms"], [])
    if any((intent in _LOSS_THEFT_INTENTS) for intent in actions):
        return (["guide_merged", "guide_general"], ["terms"])
    if any((intent in _ERROR_INTENTS) or (intent in _REGISTRATION_INTENTS) for intent in actions):
        return (["guide_merged", "guide_general"], ["terms"])
    if ui_route == ROUTE_CARD_INFO:
        return (["card_products", "guide_merged", "guide_general"], ["terms"])
    return (["guide_merged", "guide_general"], ["terms"])


def document_source_policy(
    applepay_intent: Optional[str],
    route: Optional[str],
    card_names: List[str],
    actions: List[str],
    payments: List[str],
) -> str:
    if applepay_intent:
        return "A"
    if route == ROUTE_CARD_INFO and card_names:
        return "A"
    if card_names and (actions or payments):
        return "B"
    if actions:
        return "B"
    return "C"


__all__ = [
    "decide_document_sources",
    "document_source_policy",
    "_TERMS_TRIGGERS",
]
