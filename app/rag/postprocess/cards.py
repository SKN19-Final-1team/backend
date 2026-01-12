from typing import Any, Dict, List, Optional, Tuple

from app.rag.postprocess.keywords import BENEFIT_FILTER_TOKENS, ISSUE_FILTER_TOKENS, extract_query_terms, text_has_any


def is_definition_title(title: str) -> bool:
    if not title:
        return False
    lowered = title.lower()
    for marker in ("란", "정의", "소개", "무엇", "무엇인가요"):
        if marker in lowered:
            return True
    return False


def promote_definition_doc(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for idx, doc in enumerate(docs):
        title = doc.get("title") or ""
        if is_definition_title(title):
            if idx == 0:
                return docs
            return [docs[idx], *docs[:idx], *docs[idx + 1 :]]
    return docs


def split_cards_by_query(
    cards: List[Dict[str, Any]],
    query: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    terms = set(extract_query_terms(query))
    mode: Optional[str] = None
    if terms.intersection(ISSUE_FILTER_TOKENS):
        mode = "ISSUE"
    elif terms.intersection(BENEFIT_FILTER_TOKENS):
        mode = "BENEFIT"

    if not mode:
        return cards[:2], cards[2:4]

    def _blocked(card: Dict[str, Any]) -> bool:
        text = f"{card.get('title') or ''} {card.get('content') or ''}"
        if mode == "ISSUE":
            return text_has_any(text, BENEFIT_FILTER_TOKENS)
        return text_has_any(text, ISSUE_FILTER_TOKENS)

    kept = [card for card in cards if not _blocked(card)]
    blocked = [card for card in cards if _blocked(card)]

    current = kept[:2]
    next_step = kept[2:4]
    if len(current) < 2:
        needed = 2 - len(current)
        current.extend(blocked[:needed])
        blocked = blocked[needed:]
    if len(next_step) < 2:
        needed = 2 - len(next_step)
        next_step.extend(blocked[:needed])
    return current, next_step


def omit_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, val in value.items():
            cleaned_val = omit_empty(val)
            if cleaned_val in ("", None, [], {}):
                continue
            cleaned[key] = cleaned_val
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            cleaned_item = omit_empty(item)
            if cleaned_item in ("", None, [], {}):
                continue
            cleaned_list.append(cleaned_item)
        return cleaned_list
    return value
