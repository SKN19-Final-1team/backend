from dataclasses import dataclass
from typing import Dict, List
import re

from app.rag.retriever.config import (
    BENEFIT_TERMS,
    CATEGORY_MATCH_TOKENS,
    ISSUE_TERMS,
    PRIORITY_TERMS_BY_CATEGORY,
    REISSUE_TERMS,
)
from app.rag.vocab.keyword_dict import (
    ACTION_SYNONYMS,
    PAYMENT_SYNONYMS,
    STOPWORDS,
    WEAK_INTENT_SYNONYMS,
    get_card_name_synonyms,
)

_STOPWORDS_LOWER = {word.lower() for word in STOPWORDS}
_TERM_WS_RE = re.compile(r"\s+")
_TERM_SEP_RE = re.compile(r"[\s\-/·]+")
_GUIDE_GENERIC_TERMS = {
    "카드",
    "카드사",
    "서비스",
    "이용",
    "방법",
    "안내",
    "문의",
}
_GUIDE_INTENT_SAFE_TERMS = {
    "분실도난": [
        "분실",
        "도난",
        "잃어버",
        "분실신고",
        "도난신고",
    ],
}


def _as_list(value: object | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [item for item in value if item]


def _unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _expand_no_space_terms(terms: List[str]) -> List[str]:
    out = []
    for term in terms:
        compact = term.replace(" ", "")
        if compact and compact != term:
            out.append(compact)
    return out


def _normalize_term_key(term: str) -> str:
    return _TERM_SEP_RE.sub("", term.strip().lower())


def _expand_loss_separator_variants(term: str) -> List[str]:
    compact = _normalize_term_key(term)
    if "분실" not in compact or "도난" not in compact:
        return []
    if compact not in ("분실도난", "도난분실"):
        return []
    return [
        "분실 도난",
        "분실·도난",
        "분실/도난",
        "분실-도난",
        "도난 분실",
        "도난·분실",
        "도난/분실",
        "도난-분실",
    ]


def _filter_generic_guide_terms(terms: List[str]) -> List[str]:
    if len(terms) <= 1:
        return terms
    filtered = [
        term
        for term in terms
        if _normalize_term_key(term) not in _GUIDE_GENERIC_TERMS
    ]
    return filtered or terms


def _filter_guide_query_terms(terms: List[str]) -> List[str]:
    if not terms:
        return []
    return _filter_generic_guide_terms(terms)


def _expand_payment_terms(terms: List[str]) -> List[str]:
    expanded = []
    for canonical in terms:
        expanded.extend(PAYMENT_SYNONYMS.get(canonical, []))
    combined = _unique_in_order([*terms, *expanded])
    return _unique_in_order([*combined, *_expand_no_space_terms(combined)])


def _expand_card_terms(terms: List[str]) -> List[str]:
    expanded = []
    card_synonyms = get_card_name_synonyms()
    for canonical in terms:
        expanded.extend(card_synonyms.get(canonical, []))
    return _unique_in_order([*terms, *expanded])


def _expand_action_terms(terms: List[str]) -> List[str]:
    expanded = []
    for canonical in terms:
        expanded.extend(ACTION_SYNONYMS.get(canonical, []))
    return _unique_in_order([*terms, *expanded])


def _expand_guide_terms(terms: List[str]) -> List[str]:
    if not terms:
        return []
    normalized_keys = {_normalize_term_key(term) for term in terms if term}
    safe_terms: List[str] = []
    safe_keys = { _normalize_term_key(canon) for canon in _GUIDE_INTENT_SAFE_TERMS.keys() }
    for canonical, terms_list in _GUIDE_INTENT_SAFE_TERMS.items():
        if _normalize_term_key(canonical) in normalized_keys:
            safe_terms.extend(terms_list)
    if safe_terms:
        expanded = [
            term
            for term in terms
            if _normalize_term_key(term) in safe_keys
        ]
        expanded = _unique_in_order([*expanded, *safe_terms])
    else:
        expanded = _unique_in_order(_expand_action_terms(terms))

    normalized: List[str] = []
    for term in expanded:

        if term in {"분실", "도난", "분실도난", "도난분실", "잃어버", "분실/도난", "분실·도난", "분실-도난", "분실 도난", "도난 분실"}:
            if "분실도난" not in normalized:
                normalized.append("분실도난")
        else:
            normalized.append(term)
        compact = _normalize_term_key(term)
        if compact and compact != term and compact not in normalized:
            normalized.append(compact)
        for v in _expand_loss_separator_variants(term):
            if v not in normalized:
                normalized.append(v)
    normalized = _unique_in_order(normalized)[:3]
    return _filter_generic_guide_terms(normalized)


def _expand_weak_terms(terms: List[str]) -> List[str]:
    expanded = []
    for canonical in terms:
        expanded.extend(WEAK_INTENT_SYNONYMS.get(canonical, []))
    return _unique_in_order([*terms, *expanded])


def _extract_category_terms(terms: List[str]) -> List[str]:
    hits: List[str] = []
    for term in terms:
        for hint in CATEGORY_MATCH_TOKENS:
            if hint in term:
                hits.append(hint)
    if "발급" in hits and "대상" in hits:
        hits.append("발급 대상")
    return _unique_in_order(hits)


def _build_query_text(query: str, query_template: str | None) -> str:
    if query_template:
        merged = f"{query_template} {query}".strip()
        return merged or query
    return query


def _extract_query_terms(query: str) -> List[str]:
    text = _TERM_WS_RE.sub(" ", query.strip().lower())
    raw_terms = [term for term in text.split(" ") if term]
    terms = []
    for term in raw_terms:
        if term.isdigit():
            continue
        if len(term) < 2:
            continue
        if term in _STOPWORDS_LOWER:
            continue
        terms.append(term)
    return _unique_in_order(terms)


def _priority_terms(category_terms: List[str]) -> List[str]:
    terms: List[str] = []
    for term in category_terms:
        terms.extend(PRIORITY_TERMS_BY_CATEGORY.get(term, []))
    return _unique_in_order(terms)


def _select_search_mode(terms: List[str]) -> str:
    if any(term in ISSUE_TERMS for term in terms):
        return "ISSUE"
    if any(term in BENEFIT_TERMS for term in terms):
        return "BENEFIT"
    return "GENERAL"


@dataclass(frozen=True)
class SearchContext:
    query_text: str
    filters: Dict[str, object]
    allow_guide_without_card_match: bool
    card_name_matched: bool
    route_name: str
    card_values: List[str]
    card_terms: List[str]
    intent_terms: List[str]
    weak_terms: List[str]
    payment_terms: List[str]
    query_terms: List[str]
    category_terms: List[str]
    search_mode: str
    wants_reissue: bool
    rank_terms: List[str]
    payment_only: bool
    extra_terms: List[str]


def _build_search_context(query: str, routing: Dict[str, object]) -> SearchContext:
    query_text = _build_query_text(query, routing.get("query_template"))
    filters = routing.get("filters") or {}
    allow_guide_without_card_match = bool(routing.get("allow_guide_without_card_match"))
    route_name = str(routing.get("route") or routing.get("ui_route") or "")

    card_values = _as_list(filters.get("card_name"))
    card_terms = _expand_card_terms(card_values)
    intent_terms = _expand_action_terms(_as_list(filters.get("intent")))
    weak_terms = _expand_weak_terms(_as_list(filters.get("weak_intent")))
    payment_terms = _expand_payment_terms(_as_list(filters.get("payment_method")))
    query_terms = _extract_query_terms(query)
    category_terms = _extract_category_terms([*query_terms, *weak_terms, *intent_terms])
    search_mode = _select_search_mode([*category_terms, *query_terms, *weak_terms, *intent_terms])
    wants_reissue = any(term in REISSUE_TERMS for term in [*query_terms, *intent_terms])
    rank_terms = _unique_in_order([*intent_terms, *payment_terms, *weak_terms, *query_terms])
    payment_only = bool(payment_terms) and not card_terms and not intent_terms
    payment_norm = {term.lower().replace(" ", "") for term in payment_terms}
    extra_terms = [
        term
        for term in query_terms
        if term.lower().replace(" ", "") not in payment_norm
    ]

    return SearchContext(
        query_text=query_text,
        filters=filters,
        allow_guide_without_card_match=allow_guide_without_card_match,
        card_name_matched=bool(card_values),
        route_name=route_name,
        card_values=card_values,
        card_terms=card_terms,
        intent_terms=intent_terms,
        weak_terms=weak_terms,
        payment_terms=payment_terms,
        query_terms=query_terms,
        category_terms=category_terms,
        search_mode=search_mode,
        wants_reissue=wants_reissue,
        rank_terms=rank_terms,
        payment_only=payment_only,
        extra_terms=extra_terms,
    )
