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


def _fallback_contains(synonyms: Dict[str, List[str]], text: str) -> List[str]:
    hits = []
    compact_text = text.replace(" ", "")
    for canonical, terms in synonyms.items():
        for term in [canonical, *terms]:
            if not term:
                continue
            lowered = term.lower()
            if lowered in text or lowered.replace(" ", "") in compact_text:
                hits.append(canonical)
                break
    return hits

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

    if not card_names:
        card_names = _unique_in_order(_fallback_contains(CARD_NAME_SYNONYMS, normalized))
    if not actions:
        actions = _unique_in_order(_fallback_contains(ACTION_SYNONYMS, normalized))
    if not payments:
        payments = _unique_in_order(_fallback_contains(PAYMENT_SYNONYMS, normalized))
    if not weak_intents:
        weak_intents = _unique_in_order(_fallback_contains(WEAK_INTENT_SYNONYMS, normalized))

    ui_route = None
    db_route = None  # "card_tbl" | "guide_tbl" | "both"
    boost: Dict[str, List[str]] = {}
    query_template = None

    # 검색은 항상 태우되, 실시간 트리거 여부만 제한
    should_search = True
    should_trigger = False

    # 1) 카드 + 액션: 둘 다 있으니 가장 강함
    if card_names and actions:
        ui_route = ROUTE_CARD_USAGE
        db_route = "both"
        boost = {"card_name": card_names, "intent": actions}
        if payments:
            boost["payment_method"] = payments
        if weak_intents:
            boost["weak_intent"] = weak_intents
        query_template = f"{card_names[0]} {actions[0]} 방법"
        should_trigger = True

    # 2) 카드 + 결제수단
    elif card_names and payments:
        ui_route = ROUTE_CARD_USAGE
        db_route = "card_tbl"
        boost = {"card_name": card_names, "payment_method": payments}
        query_template = f"{card_names[0]} {payments[0]} 사용 방법"
        should_trigger = True

    # 3) 카드 + 약한의도
    elif card_names and weak_intents:
        ui_route = WEAK_INTENT_ROUTE_HINTS.get(weak_intents[0], ROUTE_CARD_USAGE)
        db_route = "both"
        boost = {"card_name": card_names, "weak_intent": weak_intents}
        if ui_route == ROUTE_CARD_INFO:
            query_template = f"{card_names[0]} {weak_intents[0]}"
        else:
            query_template = f"{card_names[0]} {weak_intents[0]} 방법"
        should_trigger = True

    # 4) 카드만
    elif card_names:
        ui_route = ROUTE_CARD_INFO
        db_route = "card_tbl"
        boost = {"card_name": card_names}
        query_template = f"{card_names[0]} 정보"
        should_trigger = True

    # 5) 액션만 (중요! allowlist로 검색을 막지 말기)
    elif actions:
        ui_route = ROUTE_CARD_USAGE
        db_route = "guide_tbl"
        boost = {"intent": actions}
        if payments:
            boost["payment_method"] = payments
        query_template = f"카드 {actions[0]} 방법"
        should_trigger = any(a in ACTION_ALLOWLIST for a in actions)

    # 6) 결제수단만
    elif payments:
        ui_route = ROUTE_CARD_USAGE
        db_route = "card_tbl"
        boost = {"payment_method": payments}
        query_template = f"{payments[0]} 사용 방법"
        should_trigger = any(p in PAYMENT_ALLOWLIST for p in payments)

    # 7) 아무것도 못 잡으면: fallback 검색
    else:
        ui_route = ROUTE_CARD_USAGE
        db_route = "both"
        boost = {}
        query_template = None
        should_trigger = False

    return {
        "route": ui_route,
        "filters": boost,
        "ui_route": ui_route,
        "db_route": db_route,
        "boost": boost,
        "query_template": query_template,
        "matched": {
            "card_names": card_names,
            "actions": actions,
            "payments": payments,
            "weak_intents": weak_intents,
        },
        "should_search": should_search,
        "should_trigger": should_trigger,
        "should_route": should_trigger,  # 기존 키 유지
    }
