from __future__ import annotations

from typing import Any, Dict, List
import asyncio
import os
import re
import time

from app.llm.rag_llm.card_generator import generate_detail_cards, build_rule_cards
from app.rag.cache.card_cache import (
    CARD_CACHE_ENABLED,
    build_card_cache_key,
    card_cache_get,
    card_cache_set,
    doc_cache_id,
)
from app.rag.pipeline.utils import format_ms
from app.rag.postprocess.cards import omit_empty, promote_definition_doc, split_cards_by_query
from app.rag.postprocess.keywords import collect_query_keywords, extract_query_terms, normalize_text
from app.rag.postprocess.sections import clean_card_docs

_GENERIC_QUERY_TERMS = {
    "카드",
    "혜택",
    "할인",
    "적립",
    "신청",
    "방법",
    "조건",
    "정보",
    "추천",
    "연회비",
    "발급",
    "좋아",
    "뭐가",
}
_REGION_TERMS = {"경기", "경기도", "충남", "충북", "서울", "인천", "부산", "대구", "광주", "대전", "울산"}
_HARD_TERMS = {"체크", "체크카드", "청년", "다자녀"}
_BENEFIT_TERMS = {"혜택", "할인", "적립", "캐시백", "포인트", "환급", "청구할인"}
_NOISY_DISCLAIMER_TERMS = {
    "금융소비자",
    "금융소비자 보호법",
    "금융소비자 보호",
    "설명을 듣고",
    "충분히 이해한",
}


def _strip_phone_in_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not cards:
        return cards
    phone_dash = r"[\-]"
    out: List[Dict[str, Any]] = []
    for card in cards:
        updated = dict(card)
        content = str(updated.get("content") or "")
        content = re.sub(rf"\b\d{{2,4}}\s*{phone_dash}\s*\d{{3,4}}\s*{phone_dash}\s*\d{{4}}\b", "", content)
        content = re.sub(rf"\(\s*\d{{2,4}}\s*{phone_dash}\s*\d{{3,4}}\s*{phone_dash}\s*\d{{4}}\s*\)", "", content)
        content = re.sub(r"\b\d{8,11}\b", "", content)
        updated["content"] = content.strip()
        out.append(updated)
    return out


def _strip_noisy_disclaimer_in_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not cards:
        return cards
    out: List[Dict[str, Any]] = []
    for card in cards:
        updated = dict(card)
        content = str(updated.get("content") or "")
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        kept: List[str] = []
        for ln in lines:
            if any(term in ln for term in _NOISY_DISCLAIMER_TERMS):
                continue
            kept.append(ln)
        updated["content"] = "\n".join(kept).strip()
        out.append(updated)
    return out


def _ensure_nullable_fields(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not cards:
        return cards
    out: List[Dict[str, Any]] = []
    for card in cards:
        updated = dict(card)
        content = updated.get("content")
        full_text = updated.get("fullText")
        if content in ("", None):
            updated["content"] = None
        if full_text in ("", None):
            updated["fullText"] = None
        out.append(updated)
    return out


def _filter_card_product_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc in docs:
        table = str(doc.get("table") or (doc.get("metadata") or {}).get("source_table") or "")
        if table == "card_products":
            out.append(doc)
    return out


def _doc_has_any_term(doc: Dict[str, Any], terms: List[str]) -> bool:
    if not terms:
        return False
    title = str(doc.get("title") or "").lower()
    content = str(doc.get("content") or "").lower()
    blob = f"{title} {content}"
    return any(t and t.lower() in blob for t in terms)


def _filter_guide_docs_by_query(docs: List[Dict[str, Any]], query: str) -> Optional[List[Dict[str, Any]]]:
    guide_docs = [
        d for d in docs
        if str(d.get("table") or (d.get("metadata") or {}).get("source_table") or "") == "service_guide_documents"
    ]
    if not guide_docs:
        return []
    terms = extract_query_terms(query)
    hard_terms = [t for t in terms if t in _REGION_TERMS or t in _HARD_TERMS]
    if not hard_terms:
        hard_terms = [t for t in terms if t not in _GENERIC_QUERY_TERMS]
    if not hard_terms:
        if any(t in _BENEFIT_TERMS for t in terms):
            filtered = [d for d in guide_docs if _doc_has_any_term(d, list(_BENEFIT_TERMS))]
            return filtered
        return None
    filtered = [d for d in guide_docs if _doc_has_any_term(d, hard_terms)]
    # 쿼리에 없는 혜택 문서 제거 (예: 다자녀)
    q = (query or "").lower()
    if "다자녀" not in q:
        filtered = [
            d for d in filtered
            if "다자녀" not in str(d.get("title") or "") and "다자녀" not in str(d.get("content") or "")
        ]
    return filtered


def _inject_missing_terms_in_cards(cards: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    if not cards:
        return cards
    q = query or ""
    required = []
    for term in ("전월", "실적", "할인", "통신사", "자동납부", "혜택"):
        if term in q:
            required.append(term)
    if "전월" in q and "실적" not in required:
        required.append("실적")
    if "통신" in q and "통신사" not in required:
        required.append("통신사")
    if not required:
        return cards
    combined = " ".join(str(c.get("content") or "") for c in cards)
    missing = [t for t in required if t not in combined]
    if not missing:
        return cards
    note = f"해당 항목({', '.join(missing)})은 문서 기준으로 확인해 주세요."
    cards = [dict(c) for c in cards]
    cards[0]["content"] = (str(cards[0].get("content") or "").rstrip() + " " + note).strip()
    return cards


def _ensure_benefit_phrase(cards: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    if not cards:
        return cards
    if "혜택" not in (query or ""):
        return cards
    combined = " ".join(str(c.get("content") or "") for c in cards)
    if "혜택" in combined:
        return cards
    cards = [dict(c) for c in cards]
    prefix = "카드별 혜택은 다를 수 있으며, 원하시는 혜택 범위를 알려주시면 맞는 혜택을 안내해 드릴 수 있습니다."
    cards[0]["content"] = (prefix + " " + str(cards[0].get("content") or "")).strip()
    return cards


async def _generate_detail_cards_async(
    *,
    query: str,
    docs: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_llm_cards: int,
) -> tuple[List[Dict[str, Any]], str]:
    return await asyncio.to_thread(
        generate_detail_cards,
        query=query,
        docs=docs,
        model=model,
        temperature=temperature,
        max_llm_cards=max_llm_cards,
    )


async def build_card_response(
    *,
    query: str,
    routing: Dict[str, Any],
    docs: List[Dict[str, Any]],
    config: Any,
    t_start: float,
    t_route: float,
    t_retrieve: float,
    retrieve_cache_status: str,
) -> Dict[str, Any]:
    llm_card_top_n = max(1, config.llm_card_top_n)

    docs = clean_card_docs(docs, query)
    route_name = routing.get("route") or routing.get("ui_route")
    llm_docs = docs
    rule_docs: List[Dict[str, Any]] = []  # 규칙 기반 카드용 문서

    if route_name == "card_usage":
        llm_card_top_n = 1
        if docs:
            def _pin_sort_key(doc: Dict[str, Any]) -> tuple[int, int, float]:
                pinned = 1 if doc.get("_pinned") else 0
                pin_rank = doc.get("_pin_rank")
                pin_rank_key = -pin_rank if isinstance(pin_rank, int) else -10**9
                score = float(doc.get("score") or 0)
                return (pinned, pin_rank_key, score)
            llm_docs = sorted(docs, key=_pin_sort_key, reverse=True)[:1]

    if routing.get("route") == "card_info":
        docs = promote_definition_doc(docs)
        guide_docs = _filter_guide_docs_by_query(docs, query)
        if guide_docs is not None:
            docs = [d for d in docs if d in guide_docs or (d.get("table") == "card_products")]
        llm_docs = docs[:2]  # 상위 2개만 LLM
        rule_docs = docs[2:4]  # 나머지 2개는 규칙 기반
        llm_card_top_n = 2
        query_terms = extract_query_terms(query)
        if query_terms:
            def _doc_score(doc: Dict[str, Any]) -> int:
                title = str(doc.get("title") or "").lower()
                content = str(doc.get("content") or "").lower()
                meta = doc.get("metadata") or {}
                category = " ".join(
                    str(meta.get(k) or "")
                    for k in ("category", "category1", "category2")
                ).lower()
                score = 0
                for term in query_terms:
                    t = term.lower()
                    if t and (t in title or t in content or t in category):
                        score += 1
                return score
            docs = sorted(docs, key=_doc_score, reverse=True)
            llm_docs = docs
        if not _filter_card_product_docs(docs):
            docs = []
            llm_docs = []
            routing["card_info_no_products"] = True

    cache_status = "off"
    cards: List[Dict[str, Any]]
    ordered_doc_ids = [doc_cache_id(doc) for doc in llm_docs]
    if not llm_docs:
        cards, _ = build_rule_cards(query, docs)
    elif CARD_CACHE_ENABLED and llm_card_top_n > 0:
        cache_key = build_card_cache_key(
            route=routing.get("route") or "",
            model=config.model,
            llm_card_top_n=llm_card_top_n,
            normalized_query_template=normalize_text(routing.get("query_template") or ""),
            normalized_query=normalize_text(query),
            doc_ids=ordered_doc_ids,
        )
        cached = await card_cache_get(cache_key, ordered_doc_ids)
        if cached:
            cards, _, cache_backend = cached
            cache_status = f"hit({cache_backend})"
        else:
            cards, _ = await _generate_detail_cards_async(
                query=query,
                docs=llm_docs,
                model=config.model,
                temperature=0.0,
                max_llm_cards=llm_card_top_n,
            )
            await card_cache_set(cache_key, cards, "")
            cache_status = "miss"
    else:
        cards, _ = await _generate_detail_cards_async(
            query=query,
            docs=llm_docs,
            model=config.model,
            temperature=0.0,
            max_llm_cards=llm_card_top_n,
        )
    # card_info인 경우: 나머지 문서는 규칙 기반으로 카드 추가
    if route_name == "card_info" and rule_docs:
        rule_cards, _ = build_rule_cards(query, rule_docs, max_cards=2)
        cards = cards + rule_cards

    # [v25] 모든 라우트에서 4개 카드 보장: 부족하면 규칙 기반 카드 추가
    if len(cards) < 4 and route_name != "card_info":
        used_ids = {id(d) for d in llm_docs}
        remaining_docs = [d for d in docs if id(d) not in used_ids][:4 - len(cards)]
        if remaining_docs:
            extra_rule_cards, _ = build_rule_cards(query, remaining_docs, max_cards=4 - len(cards))
            cards = cards + extra_rule_cards

    t_cards = time.perf_counter()

    query_keywords = collect_query_keywords(query, routing, config.normalize_keywords)
    if not cards:
        cards = []
    for card in cards:
        card["keywords"] = query_keywords
    cards = [omit_empty(card) for card in cards]
    cards = _strip_phone_in_cards(cards)
    cards = _strip_noisy_disclaimer_in_cards(cards)
    cards = _ensure_nullable_fields(cards)
    if route_name == "card_info":
        cards = _inject_missing_terms_in_cards(cards, query)
        cards = _ensure_benefit_phrase(cards, query)
    if cards is None:
        cards = []
    current_cards, next_cards = split_cards_by_query(cards, query)
    t_post = time.perf_counter()

    return {
        "currentSituation": current_cards,
        "nextStep": next_cards,
        "routing": routing,
        "meta": {"model": config.model, "doc_count": len(docs), "context_chars": 0},
    }


__all__ = ["build_card_response"]
