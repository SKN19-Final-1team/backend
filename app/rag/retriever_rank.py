import os
import re
from typing import Dict, List, Optional, Tuple

from app.rag.retriever_config import (
    CARD_META_WEIGHT,
    KEYWORD_STOPWORDS,
    MIN_GUIDE_CONTENT_LEN,
    RRF_K,
)
from app.rag.retriever_db import _is_guide_table, text_search
from app.rag.retriever_terms import SearchContext, _unique_in_order

_BOOST_ENABLED = os.getenv("RAG_RRF_BOOST", "1") != "0"
_BOOST_CARD = float(os.getenv("RAG_RRF_BOOST_CARD", "0.2"))
_BOOST_CARD_GUIDE_REDUCE = float(os.getenv("RAG_RRF_BOOST_CARD_REDUCE", "1.0"))
_BOOST_INTENT = float(os.getenv("RAG_RRF_BOOST_INTENT", "0.15"))
_BOOST_PAYMENT = float(os.getenv("RAG_RRF_BOOST_PAYMENT", "0.1"))
_BOOST_WEAK = float(os.getenv("RAG_RRF_BOOST_WEAK", "0.05"))
_BOOST_CATEGORY = float(os.getenv("RAG_RRF_BOOST_CATEGORY", "0.05"))
_BOOST_GUIDE = float(os.getenv("RAG_RRF_BOOST_GUIDE", "0.004"))
_BOOST_GUIDE_COVERAGE = float(os.getenv("RAG_RRF_BOOST_GUIDE_COVERAGE", "0.01"))
_BOOST_INTENT_TITLE = float(os.getenv("RAG_RRF_BOOST_INTENT_TITLE", "0.02"))
_PENALTY_CARD_GUIDE = float(os.getenv("RAG_RRF_PENALTY_CARD_GUIDE", "0.06"))
_BOOST_GUIDE_TOKENS = tuple(
    token.strip()
    for token in os.getenv(
        "RAG_RRF_BOOST_GUIDE_TOKENS",
        "다자녀,신청,방법,대상,서류,등록,인증,환급,혜택,적립",
    ).split(",")
    if token.strip()
)

_CARD_NORM_RE = re.compile(r"[^0-9a-zA-Z가-힣]+")


def _normalize_card_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return _CARD_NORM_RE.sub("", text.lower())


def _is_noisy_guide_doc(title: Optional[str], content: str) -> bool:
    if not title:
        return True
    if not content or len(content.strip()) < MIN_GUIDE_CONTENT_LEN:
        return True
    return False


def _title_match_score(title: Optional[str], terms: List[str], weight: int) -> int:
    if not title:
        return 0
    lowered = title.lower()
    score = 0
    for term in terms:
        if term and term.lower() in lowered:
            score += weight
    return score


def _content_match_score(content: str, terms: List[str], weight: int) -> int:
    if not content:
        return 0
    lowered = content.lower()
    score = 0
    for term in terms:
        if term and term.lower() in lowered:
            score += weight
    return score


def _card_meta_score(metadata: Dict[str, object], card_values: List[str]) -> int:
    if not card_values:
        return 0
    card_name = metadata.get("card_name")
    if not card_name:
        return 0
    card_name_str = str(card_name)
    card_name_norm = card_name_str.replace(" ", "")
    card_name_compact = _normalize_card_text(card_name_str)
    for value in card_values:
        value_str = str(value)
        if card_name_str == value_str:
            return CARD_META_WEIGHT
        value_norm = value_str.replace(" ", "")
        if card_name_norm == value_norm:
            return CARD_META_WEIGHT
        if value_str and value_str in card_name_str:
            return CARD_META_WEIGHT
        if value_norm and value_norm in card_name_norm:
            return CARD_META_WEIGHT
        value_compact = _normalize_card_text(value_str)
        if value_compact and card_name_compact:
            if value_compact == card_name_compact:
                return CARD_META_WEIGHT
            if value_compact in card_name_compact or card_name_compact in value_compact:
                return CARD_META_WEIGHT
    return 0


def _card_term_match(title: Optional[str], content: str, card_terms: List[str]) -> bool:
    if card_terms and _title_match_score(title, card_terms, 1) > 0:
        return True
    if card_terms and _content_match_score(content, card_terms, 1) > 0:
        return True
    if not card_terms:
        return False
    normalized_title = _normalize_card_text(title or "")
    normalized_content = _normalize_card_text(content)
    for term in card_terms:
        term_norm = _normalize_card_text(term)
        if not term_norm:
            continue
        if term_norm in normalized_title or term_norm in normalized_content:
            return True
    return False


def _category_match_score(meta: Dict[str, object], terms: List[str]) -> int:
    if not meta or not terms:
        return 0
    parts = []
    for key in ("category", "category1", "category2"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    if not parts:
        return 0
    category_text = " ".join(parts)
    return _title_match_score(category_text, terms, 1)


def _doc_has_token(doc: Dict[str, object], tokens: List[str]) -> bool:
    if not tokens:
        return False
    title = (doc.get("title") or "").lower()
    content = (doc.get("content") or "").lower()
    meta = doc.get("metadata") or {}
    parts: List[str] = []
    for key in ("category", "category1", "category2"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    category_text = " ".join(parts).lower()
    for token in tokens:
        token_lower = token.lower()
        if token_lower in title or token_lower in content or token_lower in category_text:
            return True
    return False


def _count_term_matches(doc: Dict[str, object], terms: List[str]) -> int:
    if not terms:
        return 0
    title = (doc.get("title") or "").lower()
    content = (doc.get("content") or "").lower()
    meta = doc.get("metadata") or {}
    parts: List[str] = []
    for key in ("category", "category1", "category2"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    category_text = " ".join(parts).lower()
    normalized_title = _normalize_card_text(title)
    normalized_content = _normalize_card_text(content)
    normalized_category = _normalize_card_text(category_text)
    hits = 0
    for term in terms:
        term_lower = term.lower()
        if term_lower in title or term_lower in content or term_lower in category_text:
            hits += 1
            continue
        term_norm = _normalize_card_text(term)
        if term_norm and (
            term_norm in normalized_title
            or term_norm in normalized_content
            or term_norm in normalized_category
        ):
            hits += 1
    return hits


def _guide_tokens(context: SearchContext) -> List[str]:
    tokens = _unique_in_order(
        [
            *context.weak_terms,
            *context.category_terms,
            *context.intent_terms,
            *context.query_terms,
        ]
    )
    if not tokens:
        return []
    if _BOOST_GUIDE_TOKENS:
        return [token for token in tokens if token in _BOOST_GUIDE_TOKENS]
    return tokens


def _intent_title_terms(intent_terms: List[str]) -> List[str]:
    if not intent_terms:
        return []
    expanded: List[str] = []
    for term in intent_terms:
        expanded.append(term)
        if "분실" in term:
            expanded.append("분실")
        if "도난" in term:
            expanded.append("도난")
    return _unique_in_order(expanded)


def _normalize_doc_fields(
    content: str,
    metadata: Optional[object],
) -> Tuple[Optional[str], str, Dict[str, object]]:
    meta = metadata if isinstance(metadata, dict) else {}
    title = meta.get("title") or meta.get("name") or meta.get("card_name")
    normalized_content = content or ""
    return title, normalized_content, meta


def _rows_to_docs(
    rows: List[Tuple[object, str, Dict[str, object], float]],
    table: str,
    use_vector_score: bool,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, int]]:
    docs: Dict[str, Dict[str, object]] = {}
    ranks: Dict[str, int] = {}
    for idx, (doc_id, content, metadata, score) in enumerate(rows, 1):
        key = f"{table}:{doc_id}"
        if key in docs:
            continue
        title, normalized_content, normalized_meta = _normalize_doc_fields(content, metadata)
        if _is_guide_table(table) and _is_noisy_guide_doc(title, normalized_content):
            continue
        output_id = normalized_meta.get("id") or doc_id
        docs[key] = {
            "id": str(output_id) if output_id is not None else "",
            "db_id": doc_id,
            "title": title,
            "content": normalized_content,
            "metadata": normalized_meta,
            "vector_score": float(score) if use_vector_score else 0.0,
            "table": table,
        }
        ranks[key] = idx
    return docs, ranks


def _score_candidate(
    doc: Dict[str, object],
    context: SearchContext,
    rrf_score: float,
) -> Tuple[int, float]:
    title = doc.get("title")
    meta = doc.get("metadata") or {}
    content = doc.get("content") or ""
    card_meta_score = _card_meta_score(meta, context.card_values)
    card_match_base = card_meta_score > 0 or _card_term_match(
        title,
        content,
        context.card_terms,
    )
    doc["card_meta_score"] = card_meta_score
    title_score = 0
    if context.card_values:
        doc["card_match"] = card_match_base
    else:
        doc["card_match"] = True
    boost_score = 0.0
    if _BOOST_ENABLED:
        guide_tokens = _guide_tokens(context)
        is_guide_doc = _is_guide_table(str(doc.get("table")))
        if context.card_values and not card_match_base:
            boost_score = 0.0
        else:
            if context.card_values and card_match_base:
                card_boost = _BOOST_CARD
                if guide_tokens and not is_guide_doc:
                    card_boost *= max(0.0, 1.0 - _BOOST_CARD_GUIDE_REDUCE)
                boost_score += card_boost
            if context.intent_terms and (
                _title_match_score(title, context.intent_terms, 1)
                or _content_match_score(content, context.intent_terms, 1)
            ):
                boost_score += _BOOST_INTENT
            if _BOOST_INTENT_TITLE > 0 and is_guide_doc and context.intent_terms:
                intent_title_terms = _intent_title_terms(context.intent_terms)
                if _title_match_score(title, intent_title_terms, 1):
                    boost_score += _BOOST_INTENT_TITLE
            if context.payment_terms and (
                _title_match_score(title, context.payment_terms, 1)
                or _content_match_score(content, context.payment_terms, 1)
            ):
                boost_score += _BOOST_PAYMENT
            if context.weak_terms and (
                _title_match_score(title, context.weak_terms, 1)
                or _content_match_score(content, context.weak_terms, 1)
            ):
                boost_score += _BOOST_WEAK
            if context.category_terms and _category_match_score(meta, context.category_terms) > 0:
                boost_score += _BOOST_CATEGORY
            if _BOOST_GUIDE > 0 and is_guide_doc and context.card_values and card_match_base:
                if guide_tokens and _doc_has_token(doc, guide_tokens):
                    boost_score += _BOOST_GUIDE
                    if _BOOST_GUIDE_COVERAGE > 0 and context.query_terms:
                        match_count = _count_term_matches(doc, context.query_terms)
                        if match_count >= 2:
                            boost_score += _BOOST_GUIDE_COVERAGE * match_count
            if guide_tokens and not is_guide_doc and context.card_values and card_match_base:
                boost_score -= _PENALTY_CARD_GUIDE
    doc["rrf_boost"] = boost_score
    final_score = rrf_score + boost_score
    doc["score"] = final_score
    doc["rrf_score"] = rrf_score
    doc["title_score"] = title_score
    return title_score, final_score


def _keyword_rows(
    table: str,
    context: SearchContext,
    limit: int,
) -> List[Tuple[object, str, Dict[str, object], float]]:
    def _build_extra_terms(ctx: SearchContext) -> List[str]:
        terms: List[str] = list(ctx.extra_terms)
        if ctx.category_terms:
            terms.extend(ctx.category_terms)
        terms = _unique_in_order(terms)
        return [term for term in terms if term not in KEYWORD_STOPWORDS]

    search_mode = context.search_mode
    if _is_guide_table(table):
        if search_mode not in {"ISSUE", "BENEFIT"}:
            return []
    else:
        if not (context.payment_only or search_mode in {"ISSUE", "BENEFIT"}):
            return []
    extra_terms = _build_extra_terms(context)
    if not extra_terms:
        return []
    return text_search(table=table, terms=extra_terms, limit=limit, filters=context.filters)


def _build_candidates_from_rows(
    vec_rows: List[Tuple[object, str, Dict[str, object], float]],
    kw_rows: List[Tuple[object, str, Dict[str, object], float]],
    table: str,
    context: SearchContext,
) -> List[Tuple[float, int, Dict[str, object]]]:
    vec_docs, vec_rank = _rows_to_docs(vec_rows, table, use_vector_score=True)
    kw_docs, kw_rank = _rows_to_docs(kw_rows, table, use_vector_score=False)

    candidates: List[Tuple[float, int, Dict[str, object]]] = []
    for key in set(vec_docs.keys()) | set(kw_docs.keys()):
        doc = vec_docs.get(key) or kw_docs.get(key)
        if not doc:
            continue
        rrf_score = 0.0
        if key in vec_rank:
            rrf_score += 1.0 / (RRF_K + vec_rank[key])
        if key in kw_rank:
            rrf_score += 1.0 / (RRF_K + kw_rank[key])
        title_score, final_score = _score_candidate(doc, context, rrf_score)
        candidates.append((final_score, title_score, doc))

    return candidates


def _collect_candidates(
    table: str,
    vec_rows: List[Tuple[object, str, Dict[str, object], float]],
    context: SearchContext,
    limit: int,
) -> List[Tuple[float, int, Dict[str, object]]]:
    return _build_candidates_from_rows(
        vec_rows=vec_rows,
        kw_rows=_keyword_rows(table, context, limit),
        table=table,
        context=context,
    )


def _finalize_candidates(
    candidates: List[Tuple[float, int, Dict[str, object]]],
    key_fn,
    context: SearchContext,
) -> List[Dict[str, object]]:
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    has_card_match = any(doc.get("card_match") for _, _, doc in candidates)
    if has_card_match:
        if context.allow_guide_without_card_match:
            candidates = [
                item
                for item in candidates
                if item[2].get("card_match") or _is_guide_table(str(item[2].get("table")))
            ]
        else:
            candidates = [item for item in candidates if item[2].get("card_match")]

    best_by_title: Dict[str, Tuple[Tuple[int, float], Dict[str, object]]] = {}
    for final_score, _, doc in candidates:
        key = key_fn(doc)
        content_len = len(doc.get("content") or "")
        rank_key = (content_len, final_score)
        existing = best_by_title.get(key)
        if not existing or rank_key > existing[0]:
            best_by_title[key] = (rank_key, doc)

    docs = [item[1] for item in best_by_title.values()]
    docs.sort(key=lambda item: (item.get("score", 0.0), item.get("title_score", 0)), reverse=True)
    return docs
