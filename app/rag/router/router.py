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


def route_query(query: str) -> Dict[str, Optional[object]]:
    signals = extract_signals(query)
    force_rule = match_force_rule(signals.normalized)

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
    ).__dict__


__all__ = ["route_query", "RouterResult", "ROUTER_FORCE_RULES", "Signals"]
