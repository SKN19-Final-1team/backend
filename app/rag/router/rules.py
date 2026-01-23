import os
from typing import Dict, List, Optional, Tuple

from app.rag.vocab.keyword_dict import (
    ACTION_ALLOWLIST,
    PAYMENT_ALLOWLIST,
    WEAK_INTENT_ROUTE_HINTS,
    ROUTE_CARD_INFO,
    ROUTE_CARD_USAGE,
)
from app.rag.router.signals import Signals, _extract_tokens, _has_any_term, first, route_tuple
from app.rag.router.sources import document_source_policy as compute_document_source_policy, decide_document_sources, _TERMS_TRIGGERS

ROUTER_FORCE_RULES = [
    {
        "name": "rebate_usage",
        "subjects": ["리볼빙", "일부결제금액이월", "이월약정"],
        "actions": ["신청", "해지", "변경", "방법", "가능", "절차"],
        "route": ROUTE_CARD_USAGE,
    },
    {
        "name": "applepay_usage",
        "subjects": ["애플페이", "apple pay"],
        "actions": ["등록", "추가", "인증", "안돼", "오류", "실패"],
        "route": ROUTE_CARD_USAGE,
    },
    {
        "name": "reissue_usage",
        "subjects": ["재발급", "재발행", "재교부", "재발급신청"],
        "actions": ["방법", "신청", "절차", "해당"],
        "route": ROUTE_CARD_USAGE,
    },
]

_BENEFIT_ROUTE_TOKENS = {
    "혜택",
    "할인",
    "자동납부",
    "자동이체",
    "통신료",
    "통신요금",
    "전월실적",
    "적립",
    "캐시백",
}

_REISSUE_TOKENS = {"재발급", "재발행", "재교부"}

STRICT_SEARCH = os.getenv("RAG_ROUTER_STRICT_SEARCH", "1") != "0"
MIN_QUERY_LEN = int(os.getenv("RAG_ROUTER_MIN_QUERY_LEN", "2"))


def match_force_rule(normalized: str) -> Optional[Dict[str, any]]:
    for rule in ROUTER_FORCE_RULES:
        if any(subject in normalized for subject in rule["subjects"]) and any(
            action in normalized for action in rule["actions"]
        ):
            return rule
    return None


def decide_route(signals: Signals) -> Tuple[str, str, Dict[str, List[str]], Optional[str], bool, bool]:
    normalized = signals.normalized
    card_names = signals.card_names
    actions = signals.actions
    payments = signals.payments
    weak_intents = signals.weak_intents
    pattern_hits = signals.pattern_hits
    applepay_intent = signals.applepay_intent
    info_hint = signals.info_hint
    usage_strong = signals.usage_strong
    issuance_hint = signals.issuance_hint

    benefit_route_hint = _has_any_term(normalized, _BENEFIT_ROUTE_TOKENS)
    if not card_names and "국민행복" in normalized:
        card_names = ["국민행복"]
    reissue_intent = any(term in normalized for term in _REISSUE_TOKENS)

    strong_signal = signals.strong_signal
    should_search = strong_signal and len(normalized) >= MIN_QUERY_LEN if STRICT_SEARCH else True
    single_token_noise = not strong_signal and len(_extract_tokens(normalized)) == 1

    ui_route, db_route, boost, query_template, should_trigger = route_tuple(ROUTE_CARD_USAGE, "both")

    cases = [
        (reissue_intent and not applepay_intent,
         route_tuple(ROUTE_CARD_USAGE, "guide_tbl", {"intent": ["재발급"]}, None, True)),
        (benefit_route_hint and not applepay_intent,
         route_tuple(ROUTE_CARD_INFO, "card_tbl" if card_names else "both", {"card_name": card_names} if card_names else {}, None, True)),
        (info_hint and not usage_strong and not payments and not pattern_hits,
         route_tuple(
             ROUTE_CARD_INFO,
             "both" if actions or weak_intents else ("card_tbl" if card_names else "both"),
             {k: v for k, v in {
                 "card_name": card_names or None,
                 "intent": actions or None,
                 "weak_intent": weak_intents or None,
             }.items() if v},
             (f"{first(card_names)} 정보" if card_names else (f"카드 {first(actions)} 정보" if actions else None)),
             True,
         )),
        (issuance_hint and not card_names,
         route_tuple(ROUTE_CARD_INFO, "card_tbl", {"intent": ["발급"]}, "카드 발급 조건", True)),
        (card_names and actions,
         route_tuple(
             ROUTE_CARD_USAGE,
             "both",
             {k: v for k, v in {
                 "card_name": card_names,
                 "intent": actions,
                 "payment_method": payments or None,
                 "weak_intent": weak_intents or None,
             }.items() if v},
             f"{first(card_names)} {first(actions)} 방법" if card_names and actions else None,
             True,
         )),
        (card_names and payments,
         route_tuple(ROUTE_CARD_USAGE, "card_tbl", {"card_name": card_names, "payment_method": payments}, f"{first(card_names)} {first(payments)} 사용 방법" if card_names and payments else None, True)),
        (card_names and weak_intents,
         route_tuple(
             WEAK_INTENT_ROUTE_HINTS.get(first(weak_intents), ROUTE_CARD_USAGE) if weak_intents else ROUTE_CARD_USAGE,
             "both",
             {"card_name": card_names, "weak_intent": weak_intents},
             (
                 f"{first(card_names)} {first(weak_intents)}"
                 if WEAK_INTENT_ROUTE_HINTS.get(first(weak_intents), ROUTE_CARD_USAGE) == ROUTE_CARD_INFO
                 else f"{first(card_names)} {first(weak_intents)} 방법"
             ) if card_names and weak_intents else None,
             True,
         )),
        (card_names,
         route_tuple(ROUTE_CARD_INFO, "card_tbl", {"card_name": card_names}, f"{first(card_names)} 정보" if card_names else None, True)),
        (actions,
         route_tuple(
             ROUTE_CARD_USAGE,
             "guide_tbl",
             {k: v for k, v in {
                 "intent": actions,
                 "payment_method": payments or None,
             }.items() if v},
             f"카드 {first(actions)} 방법" if actions else None,
             any(a in ACTION_ALLOWLIST for a in actions),
         )),
        (payments,
         route_tuple(ROUTE_CARD_USAGE, "card_tbl", {"payment_method": payments}, f"{first(payments)} 사용 방법" if payments else None, any(p in PAYMENT_ALLOWLIST for p in payments))),
    ]

    for cond, values in cases:
        if cond:
            ui_route, db_route, boost, query_template, should_trigger = values
            break

    if single_token_noise or (card_names and not actions and not payments and not weak_intents and len(normalized.split()) == 1):
        should_search = False
        should_trigger = False

    document_sources, exclude_sources = decide_document_sources(
        applepay_intent=applepay_intent,
        ui_route=ui_route,
        actions=actions,
        card_names=card_names,
        terms_trigger=_has_any_term(normalized, _TERMS_TRIGGERS),
    )
    document_source_policy = compute_document_source_policy(
        applepay_intent=applepay_intent,
        route=ui_route,
        card_names=card_names,
        actions=actions,
        payments=payments,
    )

    return ui_route, db_route, boost, query_template, should_trigger, should_search, document_sources, exclude_sources, document_source_policy


__all__ = [
    "match_force_rule",
    "decide_route",
    "STRICT_SEARCH",
    "MIN_QUERY_LEN",
]
