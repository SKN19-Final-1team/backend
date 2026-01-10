import re
from typing import Dict, List, Optional

from flashtext import KeywordProcessor

# vocab 규칙 정의
from app.rag.vocab.rules import (
    ACTION_ALLOWLIST,
    ACTION_SYNONYMS,
    CARD_NAME_SYNONYMS,
    PAYMENT_ALLOWLIST,
    PAYMENT_SYNONYMS,
    WEAK_INTENT_ROUTE_HINTS,
    WEAK_INTENT_SYNONYMS,
    ROUTE_CARD_INFO,
    ROUTE_CARD_USAGE,
)

#     리스트에서 중복을 제거, 최초 등장 순서는 유지
def _unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


_WS_RE = re.compile(r"\s+")

#   사용자 입력 문장 정규화 = 중복 공백 제거, 소문자 변환
def _normalize_query(text: str) -> str:
    text = _WS_RE.sub(" ", text.strip())
    return text.lower()

#     FlashText KeywordProcessor 생성 = 동의어 canonical 값으로 매핑
def _build_processor(synonyms: Dict[str, List[str]]) -> KeywordProcessor:
    kp = KeywordProcessor(case_sensitive=False)
    for canonical, terms in synonyms.items():
        kp.add_keyword(canonical, canonical)
        for term in terms:
            kp.add_keyword(term, canonical)
    return kp

# FlashText 프로세서 사전 생성 (모듈 로딩 시 1회)
_CARD_KP = _build_processor(CARD_NAME_SYNONYMS)
_ACTION_KP = _build_processor(ACTION_SYNONYMS)
_PAYMENT_KP = _build_processor(PAYMENT_SYNONYMS)
_WEAK_INTENT_KP = _build_processor(WEAK_INTENT_SYNONYMS)


def route_query(query: str) -> Dict[str, Optional[object]]:
    normalized = _normalize_query(query)
    card_names = _unique_in_order(_CARD_KP.extract_keywords(normalized))
    actions = _unique_in_order(_ACTION_KP.extract_keywords(normalized))
    payments = _unique_in_order(_PAYMENT_KP.extract_keywords(normalized))
    weak_intents = _unique_in_order(_WEAK_INTENT_KP.extract_keywords(normalized))

    route = None
    filters: Dict[str, List[str]] = {}
    query_template = None
    should_route = False

    if card_names and actions:
        route = ROUTE_CARD_USAGE
        filters = {"card_name": card_names, "intent": actions}
        if payments:
            filters["payment_method"] = payments
        if weak_intents:
            filters["weak_intent"] = weak_intents
        query_template = f"{card_names[0]} {actions[0]} 방법"
        should_route = True
    elif card_names and payments:
        route = ROUTE_CARD_USAGE
        filters = {"card_name": card_names, "payment_method": payments}
        query_template = f"{card_names[0]} {payments[0]} 사용 방법"
        should_route = True
    elif card_names and weak_intents:
        route = WEAK_INTENT_ROUTE_HINTS.get(weak_intents[0], ROUTE_CARD_USAGE)
        filters = {"card_name": card_names, "weak_intent": weak_intents}
        if route == ROUTE_CARD_INFO:
            query_template = f"{card_names[0]} {weak_intents[0]}"
        else:
            query_template = f"{card_names[0]} {weak_intents[0]} 방법"
        should_route = True
    elif card_names:
        route = ROUTE_CARD_INFO
        filters = {"card_name": card_names}
        query_template = f"{card_names[0]} 카드 정보"
        should_route = True
    elif actions:
        route = ROUTE_CARD_USAGE
        filters = {"intent": actions}
        if payments:
            filters["payment_method"] = payments
        query_template = f"카드 {actions[0]} 방법"
        should_route = any(action in ACTION_ALLOWLIST for action in actions)
    elif payments:
        route = ROUTE_CARD_USAGE
        filters = {"payment_method": payments}
        query_template = f"{payments[0]} 사용 방법"
        should_route = any(payment in PAYMENT_ALLOWLIST for payment in payments)

    return {
        "route": route,
        "filters": filters,
        "query_template": query_template,
        "matched": {
            "card_names": card_names,
            "actions": actions,
            "payments": payments,
            "weak_intents": weak_intents,
        },
        "should_route": should_route,
    }
