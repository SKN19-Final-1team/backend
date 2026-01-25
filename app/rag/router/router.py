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
    consult_keyword_hits = _count_domain_keyword_hits(signals.normalized)
    consult_category_candidates = _build_consult_category_candidates(signals)
    need_consult_case_search = bool(
        signals.actions or signals.payments or signals.weak_intents
    )

    if force_rule:
        return RouterResult(
            route=force_rule["route"],
            filters={},
            ui_route=force_rule["route"],
            db_route="card_tbl",
            boost={},
            query_template=None,
            matched={
                "card_names": signals.card_names,
                "actions": signals.actions,
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

    return RouterResult(
        route=ui_route,
        filters=boost,
        ui_route=ui_route,
        db_route=db_route,
        boost=boost,
        query_template=query_template,
        matched={
            "card_names": signals.card_names,
            "actions": signals.actions,
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
