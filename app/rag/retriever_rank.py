from typing import Dict, List, Optional, Tuple

from app.rag.retriever_config import (
    CARD_META_WEIGHT,
    CATEGORY_TITLE_BONUS,
    CATEGORY_TITLE_DEMOTE,
    ISSUANCE_HINT_TOKENS,
    ISSUANCE_TITLE_BONUS,
    ISSUANCE_TITLE_DEMOTE,
    KEYWORD_STOPWORDS,
    MIN_GUIDE_CONTENT_LEN,
    QUERY_CONTENT_WEIGHT,
    QUERY_TITLE_WEIGHT,
    REISSUE_TITLE_PENALTY,
    RRF_K,
    TITLE_SCORE_WEIGHT,
)
from app.rag.retriever_db import _is_guide_table, text_search
from app.rag.retriever_terms import SearchContext, _unique_in_order


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
    return 0


def _card_term_match(title: Optional[str], content: str, card_terms: List[str]) -> bool:
    if card_terms and _title_match_score(title, card_terms, 1) > 0:
        return True
    if card_terms and _content_match_score(content, card_terms, 1) > 0:
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


def _issuance_title_bonus(title: Optional[str], category_terms: List[str]) -> int:
    if not title or not category_terms:
        return 0
    if not any(term in ISSUANCE_HINT_TOKENS for term in category_terms):
        return 0
    score = 0
    for token, bonus in ISSUANCE_TITLE_BONUS.items():
        if token in title:
            score += bonus
    if any(token in title for token in ISSUANCE_TITLE_DEMOTE):
        score -= 2
    return score


def _category_title_bonus(title: Optional[str], category_terms: List[str]) -> int:
    if not title or not category_terms:
        return 0
    if not any(term in ("적립", "혜택") for term in category_terms):
        return 0
    score = 0
    for token, bonus in CATEGORY_TITLE_BONUS.items():
        if token in title:
            score += bonus
    if any(token in title for token in CATEGORY_TITLE_DEMOTE):
        score -= 2
    return score


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
    title_score = _title_match_score(title, context.card_terms, 2)
    title_score += _title_match_score(title, context.rank_terms, 1)
    title_score += _title_match_score(title, context.query_terms, QUERY_TITLE_WEIGHT)
    title_score += _content_match_score(content, context.query_terms, QUERY_CONTENT_WEIGHT)
    card_match_base = card_meta_score > 0 or _card_term_match(
        title,
        content,
        context.card_terms,
    )
    category_score = 0
    issuance_bonus = 0
    category_bonus = 0
    if context.search_mode == "ISSUE":
        issuance_bonus = _issuance_title_bonus(title, context.category_terms)
    elif context.search_mode == "BENEFIT":
        category_score = _category_match_score(meta, context.query_terms)
        category_bonus = _category_title_bonus(title, context.category_terms)
    if context.card_values and not card_match_base:
        category_score = 0
        issuance_bonus = 0
        category_bonus = 0
    reissue_penalty = 0
    if title and "재발급" in title:
        reissue_penalty = REISSUE_TITLE_PENALTY if context.wants_reissue else -REISSUE_TITLE_PENALTY
    title_score += category_score
    title_score += issuance_bonus
    title_score += category_bonus
    title_score += reissue_penalty
    title_score += card_meta_score
    doc["card_meta_score"] = card_meta_score
    doc["issuance_bonus"] = issuance_bonus
    doc["category_bonus"] = category_bonus
    doc["reissue_penalty"] = reissue_penalty
    if context.card_values:
        doc["card_match"] = card_match_base
    else:
        doc["card_match"] = True
    final_score = rrf_score + (title_score * TITLE_SCORE_WEIGHT)
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
        if search_mode != "ISSUE":
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
) -> List[Tuple[int, float, Dict[str, object]]]:
    vec_docs, vec_rank = _rows_to_docs(vec_rows, table, use_vector_score=True)
    kw_docs, kw_rank = _rows_to_docs(kw_rows, table, use_vector_score=False)

    candidates: List[Tuple[int, float, Dict[str, object]]] = []
    for key in set(vec_docs.keys()) | set(kw_docs.keys()):
        doc = vec_docs.get(key) or kw_docs.get(key)
        if not doc:
            continue
        rrf_score = 0.0
        if key in vec_rank:
            rrf_score += 1.0 / (RRF_K + vec_rank[key])
        if key in kw_rank:
            rrf_score += 1.0 / (RRF_K + kw_rank[key])
        title_score, _ = _score_candidate(doc, context, rrf_score)
        candidates.append((title_score, rrf_score, doc))

    return candidates


def _collect_candidates(
    table: str,
    vec_rows: List[Tuple[object, str, Dict[str, object], float]],
    context: SearchContext,
    limit: int,
) -> List[Tuple[int, float, Dict[str, object]]]:
    return _build_candidates_from_rows(
        vec_rows=vec_rows,
        kw_rows=_keyword_rows(table, context, limit),
        table=table,
        context=context,
    )


def _finalize_candidates(
    candidates: List[Tuple[int, float, Dict[str, object]]],
    key_fn,
) -> List[Dict[str, object]]:
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    has_card_match = any(doc.get("card_match") for _, _, doc in candidates)
    if has_card_match:
        candidates = [item for item in candidates if item[2].get("card_match")]

    best_by_title: Dict[str, Tuple[Tuple[int, float], Dict[str, object]]] = {}
    for _, score, doc in candidates:
        key = key_fn(doc)
        content_len = len(doc.get("content") or "")
        rank_key = (content_len, score)
        existing = best_by_title.get(key)
        if not existing or rank_key > existing[0]:
            best_by_title[key] = (rank_key, doc)

    docs = [item[1] for item in best_by_title.values()]
    docs.sort(key=lambda item: (item.get("title_score", 0), item.get("score", 0.0)), reverse=True)
    return docs
