from dataclasses import dataclass
from typing import Dict, List
import re

from app.rag.retriever_config import (
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
