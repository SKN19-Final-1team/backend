from typing import Any, Dict


def should_enable_info_guidance(routing: Dict[str, Any], query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    matched = routing.get("matched") or {}
    intent_terms = []
    for key in ("actions", "weak_intents"):
        intent_terms.extend(matched.get(key) or [])
    intent_terms.extend((routing.get("filters") or {}).get("intent") or [])
    intent_terms = [str(t) for t in intent_terms if t]
    trigger_terms = (
        "추천",
        "비교",
        "얼마",
        "조건",
        "가능",
        "연회비",
        "혜택",
        "신청",
        "발급",
        "한도",
        "전월",
        "실적",
        "통신",
        "자동납부",
        "통신사",
        "편의점",
        "배달",
    )
    if any(t in q for t in trigger_terms):
        return True
    if any(t for t in intent_terms if any(k in t for k in trigger_terms)):
        return True
    filters = routing.get("filters") or {}
    if filters.get("card_name") and not (filters.get("intent") or filters.get("benefit_type")):
        return False
    return False
