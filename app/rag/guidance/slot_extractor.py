from typing import Any, Dict, List


def extract_guidance_slots(routing: Dict[str, Any]) -> Dict[str, Any]:
    matched = routing.get("matched") or {}
    filters = routing.get("filters") or {}
    card_names = matched.get("card_names") or filters.get("card_name") or []
    if isinstance(card_names, str):
        card_names = [card_names]
    card_names = [str(n) for n in card_names if n]
    card_names = sorted(set(card_names), key=len)

    region = ""
    regions = filters.get("region") or []
    if isinstance(regions, str):
        region = regions
    elif regions:
        region = str(regions[0])

    benefit_types = filters.get("benefit_type") or []
    if isinstance(benefit_types, str):
        benefit_types = [benefit_types]
    benefit_types = [str(b) for b in benefit_types if b]
    benefit_types = sorted(set(benefit_types))

    return {
        "card_names": card_names,
        "region": region,
        "benefit_types": benefit_types,
    }
