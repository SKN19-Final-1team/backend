from typing import Any, Dict, List, Optional
import re

from app.rag.postprocess.keywords import extract_query_terms
from app.rag.common.text_utils import unique_in_order


def _normalize_region(region: str) -> str:
    r = (region or "").strip()
    if not r:
        return ""
    if r == "경기":
        return "경기도"
    if r == "충남":
        return "충남"
    return r


def _ensure_terms(message: str, terms: List[str]) -> str:
    if not message:
        return message
    missing = [t for t in terms if t and t not in message]
    if not missing:
        return message
    suffix = " ".join(missing)
    return f"{message} {suffix}"


def build_info_guidance(
    query: str,
    slots: Dict[str, Any],
    product_docs: List[Dict[str, Any]],
    guide_docs: List[Dict[str, Any]],
) -> str:
    q = query or ""
    card_names = slots.get("card_names") or []
    doc_titles = [
        str(doc.get("title") or "")
        for doc in (product_docs or [])
        if str(doc.get("title") or "").strip()
    ]
    candidates = [*doc_titles, *card_names]
    card_name = ""
    if candidates:
        for name in candidates:
            if name and name in q:
                card_name = name
                break
        if not card_name:
            card_name = candidates[0]
    region = _normalize_region(slots.get("region") or "")
    benefit_types = slots.get("benefit_types") or []
    if not card_name and any(k in q for k in ("K-패스", "k패스", "k-패스")):
        card_name = "K-패스"
    if "연회비" in q:
        if card_name:
            message = f"{card_name} 연회비는 카드 등급(국내전용/해외겸용)에 따라 달라요. 정확한 금액을 확인해 드릴까요?"
        else:
            message = "연회비는 카드 및 등급(국내전용/해외겸용)에 따라 달라요. 카드명을 알려주시면 바로 확인해 드릴게요."
        return _ensure_terms(message, ["연회비"])

    if "편의점" in q:
        return "편의점(CU/GS25/세븐) 5% 혜택을 확인해 드릴게요."

    if "배달" in q:
        return "배달앱 건당 2만원 이상 결제 시 1천 포인트 적립 여부를 확인해 드릴게요."

    if any(k in q for k in ("통신", "자동납부", "통신사")) and not any(k in q for k in ("한도", "전월", "실적")):
        if "통신사" in q:
            if card_name:
                return f"{card_name} 통신사 자동납부 할인 여부를 확인해 드릴게요."
            return "통신사 자동납부 할인 여부를 확인해 드릴게요."
        if card_name:
            return f"{card_name} 통신 자동납부 할인 여부를 확인해 드릴게요."
        return "통신 자동납부 할인 여부를 확인해 드릴게요."

    if any(k in q for k in ("혜택", "좋아", "추천", "괜찮")):
        if card_name or region or benefit_types:
            parts = []
            if card_name:
                parts.append(card_name)
            if region:
                parts.append(f"{region} 혜택")
            if benefit_types:
                parts.append(f"{'·'.join(benefit_types)} 혜택")
            detail = " / ".join(parts)
            message = f"{detail} 혜택부터 안내해 드릴까요?"
        else:
            message = "주로 쓰는 항목(교통/통신/쇼핑 등)을 알려주시면 맞는 혜택 위주로 추천해 드릴게요."
        return _ensure_terms(message, ["혜택"])

    if any(k in q for k in ("발급", "조건", "신청", "가능")):
        if card_name:
            message = f"{card_name} 발급 조건은 고객 정보에 따라 달라요. 신청자 정보를 알려주시면 확인해 드릴게요."
        else:
            message = "발급 조건은 카드와 고객 정보에 따라 달라요. 카드명과 신청자 정보를 알려주시면 확인해 드릴게요."
        return _ensure_terms(message, ["발급", "조건"])

    if any(k in q for k in ("한도", "전월", "실적")):
        if card_name:
            if "통신" in q:
                message = (
                    f"{card_name} 통신 할인 한도/전월실적 기준을 확인해 드릴까요? "
                    "전월 이용금액을 알려주시면 바로 계산해 드릴게요."
                )
            else:
                message = f"{card_name} 전월실적/한도 기준을 확인해 드릴까요? 전월 이용금액을 알려주시면 바로 계산해 드릴게요."
        else:
            message = "전월실적과 한도는 카드마다 기준이 달라요. 카드명을 알려주시면 정확히 안내해 드릴게요."
        return _ensure_terms(message, ["전월", "실적", "한도"])

    if card_name or region or benefit_types:
        if region:
            prefix = f"{card_name} " if card_name else ""
            message = f"{prefix}{region} 혜택 기준으로 안내할까요?"
            return _ensure_terms(message, [region, "혜택"])
        if benefit_types:
            prefix = f"{card_name} " if card_name else ""
            message = f"{prefix}{'·'.join(benefit_types)} 혜택 기준으로 안내할까요?"
            return _ensure_terms(message, ["혜택"])
        message = f"{card_name} 기준으로 안내할까요? 확인 후 필요한 항목만 간단히 정리해 드릴게요."
        return _ensure_terms(message, [card_name])

    if product_docs:
        return "카드별 조건이 달라서, 정확한 안내를 위해 카드명을 알려주시면 좋습니다."

    base = ""
    if "편의점" in q:
        base = "편의점(CU/GS25/세븐) 5% 혜택을 확인해 드릴게요."
    if "배달" in q:
        base = base or "배달앱 건당 2만원 이상 결제 시 1천 포인트 적립 여부를 확인해 드릴게요."
    if any(k in q for k in ("통신", "자동납부", "통신사")):
        base = base or "통신 자동납부 할인 여부를 확인해 드릴게요."
    if base:
        return base
    return ""


def filter_card_product_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc in docs:
        table = str(doc.get("table") or (doc.get("metadata") or {}).get("source_table") or "")
        if table == "card_products":
            out.append(doc)
    return out


def filter_usage_docs_for_guidance(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return docs or []


_TAG_KEYWORDS = {
    "loss": ("분실", "도난", "잃어버", "신고", "정지", "해제", "재발급"),
    "loan": ("대출", "현금서비스", "카드론", "리볼빙", "이자", "수수료", "금리"),
    "payment": ("결제", "승인", "오류", "취소", "환불", "청구", "납부", "연체", "미납"),
    "apply": ("신청", "발급", "서류", "준비물", "본인확인", "소득", "재직", "제출"),
    "usage": ("사용처", "가맹점", "이용처", "사용 가능", "사용가능"),
    "phone": ("전화", "번호", "고객센터", "콜센터", "연락처"),
    "applepay": ("애플페이", "apple pay"),
    "samsungpay": ("삼성페이",),
    "kakaopay": ("카카오페이", "kakaopay"),
    "tmoney": ("티머니", "tmoney"),
    "dcc": ("dcc", "원화결제", "원화 결제"),
}

# Core intent tokens used for hard-gating/allowlisting to avoid off-topic docs.
_INTENT_CORE_TOKENS = {
    "loan": ("현금서비스", "단기카드대출", "리볼빙", "카드론", "대출"),
    "payment": (
        "승인",
        "결제",
        "오류",
        "실패",
        "미승인",
        "거절",
        "취소",
        "환불",
        "청구",
        "납부",
        "연체",
        "미납",
        "한도",
        "정지",
    ),
    "apply": ("신청", "발급", "서류", "준비물", "본인확인", "소득", "재직", "제출"),
    "phone": ("고객센터", "콜센터", "전화", "연락처", "대표번호", "번호"),
}

# Hard brand/type tokens to prevent obvious mismatches (e.g., 티머니 ↔ 나라사랑).
_BRAND_TOKENS = (
    "티머니",
    "tmoney",
    "카카오페이",
    "kakaopay",
    "삼성페이",
    "애플페이",
    "apple pay",
    "dcc",
    "원화결제",
    "원화 결제",
    "나라사랑",
    "테디카드",
    "k패스",
    "k-패스",
    "k pass",
    "k-pass",
    "국민행복",
    "국민행복카드",
)

_QUERY_STOPWORDS = {
    "카드",
    "안내",
    "문의",
    "확인",
    "방법",
    "절차",
    "가능",
    "관련",
    "정보",
    "사용",
    "이용",
    "고객센터",
    "센터",
    "전화",
    "번호",
    "처리",
}


def _intent_terms(query: str) -> set[str]:
    lowered = (query or "").lower()
    terms: set[str] = set()
    for tag, tokens in _TAG_KEYWORDS.items():
        if any(token in lowered for token in tokens):
            terms.update(tokens)
    return terms


def _pin_score(doc: Dict[str, Any]) -> tuple[int, int, float]:
    pinned = 1 if doc.get("_pinned") else 0
    pin_rank = doc.get("_pin_rank")
    pin_rank_key = -pin_rank if isinstance(pin_rank, int) else -10**9
    score = float(doc.get("score") or 0)
    return (pinned, pin_rank_key, score)


def _tag_text(text: str) -> set[str]:
    if not text:
        return set()
    lowered = text.lower()
    tags: set[str] = set()
    for tag, tokens in _TAG_KEYWORDS.items():
        if any(token in lowered for token in tokens):
            tags.add(tag)
    return tags


def _doc_tag_text(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata") or {}
    tags = meta.get("tags") or meta.get("scenario_tags") or []
    if isinstance(tags, list):
        tags_text = " ".join(str(t) for t in tags if t)
    else:
        tags_text = str(tags or "")
    parts = [
        str(doc.get("title") or ""),
        str(doc.get("id") or ""),
        str(meta.get("title") or ""),
        str(meta.get("category") or ""),
        str(meta.get("category1") or ""),
        str(meta.get("category2") or ""),
        tags_text,
    ]
    return " ".join([p for p in parts if p]).lower()


def _doc_title_text(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata") or {}
    parts = [
        str(doc.get("title") or ""),
        str(doc.get("id") or ""),
        str(meta.get("title") or ""),
        str(meta.get("category") or ""),
        str(meta.get("category1") or ""),
        str(meta.get("category2") or ""),
    ]
    return " ".join([p for p in parts if p]).lower()


def _doc_brand_text(doc: Dict[str, Any]) -> str:
    base = _doc_tag_text(doc)
    content = str(doc.get("content") or "")
    return f"{base} {content}".lower().strip()


def _tag_doc(doc: Dict[str, Any]) -> set[str]:
    return _tag_text(_doc_tag_text(doc))


def _tag_query(query: str, routing: Optional[Dict[str, Any]]) -> set[str]:
    text = (query or "").lower()
    tags = _tag_text(text)
    if (routing or {}).get("filters", {}).get("phone_lookup") is True:
        tags.add("phone")
    if not tags:
        tags.add("general")
    return tags


def _brand_tokens(text: str) -> set[str]:
    if not text:
        return set()
    lowered = text.lower()
    return {token for token in _BRAND_TOKENS if token in lowered}


def filter_guidance_docs(
    query: str,
    docs: List[Dict[str, Any]],
    max_docs: int = 4,
    routing: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not docs:
        return []
    guide_docs = [
        doc
        for doc in docs
        if str(doc.get("table") or (doc.get("metadata") or {}).get("source_table") or "")
        == "service_guide_documents"
    ]
    if not guide_docs:
        return []
    query_tags = _tag_query(query, routing)
    if "general" not in query_tags:
        tagged = [doc for doc in guide_docs if _tag_doc(doc) & query_tags]
        if tagged:
            guide_docs = tagged

    # Intent hard-gate + allowlist: keep only docs whose title/id matches core tokens.
    gate_tokens: List[str] = []
    for tag in query_tags:
        gate_tokens.extend(_INTENT_CORE_TOKENS.get(tag, ()))
    gate_tokens = unique_in_order(gate_tokens)
    if gate_tokens:
        gated = [doc for doc in guide_docs if any(tok in _doc_title_text(doc) for tok in gate_tokens)]
        if gated:
            guide_docs = gated
    brand_tokens = _brand_tokens(query)
    if brand_tokens:
        brand_filtered = [
            doc
            for doc in guide_docs
            if any(token in _doc_brand_text(doc) for token in brand_tokens)
        ]
        if brand_filtered:
            guide_docs = brand_filtered
    else:
        # If query has no brand, prefer brand-neutral docs to avoid vendor-specific leakage.
        neutral_docs = [
            doc
            for doc in guide_docs
            if not _brand_tokens(_doc_brand_text(doc))
        ]
        if neutral_docs:
            guide_docs = neutral_docs

    filters = (routing or {}).get("filters") or {}
    intent_terms: List[str] = []
    for key in ("intent", "weak_intent"):
        val = filters.get(key)
        if isinstance(val, list):
            intent_terms.extend([str(v) for v in val if v])
        elif isinstance(val, str) and val:
            intent_terms.append(val)
    expanded_intent_terms: List[str] = []
    for term in intent_terms:
        lowered = str(term).strip().lower()
        if not lowered:
            continue
        if lowered in _TAG_KEYWORDS:
            expanded_intent_terms.extend(_TAG_KEYWORDS[lowered])
        else:
            expanded_intent_terms.append(lowered)
    expanded_intent_terms.extend(sorted(_intent_terms(query)))
    expanded_intent_terms = unique_in_order(expanded_intent_terms)

    # Strict intent filter using titles/ids only to avoid topic leakage.
    title_intent_counts: Dict[str, int] = {}
    if expanded_intent_terms:
        for doc in guide_docs:
            lowered = _doc_title_text(doc)
            title_intent_counts[str(doc.get("id") or doc.get("metadata", {}).get("id") or "")] = sum(
                1 for term in expanded_intent_terms if term in lowered
            )
        max_title_intent = max(title_intent_counts.values() or [0])
        if max_title_intent > 0:
            guide_docs = [
                doc
                for doc in guide_docs
                if title_intent_counts.get(
                    str(doc.get("id") or doc.get("metadata", {}).get("id") or ""), 0
                )
                > 0
            ]
    terms = unique_in_order([*extract_query_terms(query), *expanded_intent_terms])
    match_counts: Dict[str, int] = {}
    if terms:
        for doc in guide_docs:
            text = _doc_tag_text(doc)
            content = str(doc.get("content") or "")
            lowered = f"{text} {content}".lower()
            match_counts[str(doc.get("id") or doc.get("metadata", {}).get("id") or "")] = sum(
                1 for term in terms if term in lowered
            )
    intent_counts: Dict[str, int] = {}
    if expanded_intent_terms:
        for doc in guide_docs:
            lowered = _doc_brand_text(doc)
            intent_counts[str(doc.get("id") or doc.get("metadata", {}).get("id") or "")] = sum(
                1 for term in expanded_intent_terms if term in lowered
            )
    pinned = [doc for doc in guide_docs if doc.get("_pinned")]
    others = [doc for doc in guide_docs if not doc.get("_pinned")]
    if others and intent_counts:
        max_intent = max(intent_counts.values() or [0])
        if max_intent > 0:
            others = [
                doc
                for doc in others
                if intent_counts.get(str(doc.get("id") or doc.get("metadata", {}).get("id") or ""), 0)
                > 0
            ]
    if others and match_counts:
        max_match = max(match_counts.values() or [0])
        if max_match > 0:
            others = [
                doc
                for doc in others
                if match_counts.get(str(doc.get("id") or doc.get("metadata", {}).get("id") or ""), 0) > 0
            ]
    if others:
        non_negative = [doc for doc in others if float(doc.get("score") or 0.0) >= 0.0]
        if non_negative:
            others = non_negative

    pinned_sorted = sorted(pinned, key=lambda d: d.get("_pin_rank", 10**9))
    if others:
        def _rank(doc: Dict[str, Any]) -> tuple[int, float]:
            doc_id = str(doc.get("id") or doc.get("metadata", {}).get("id") or "")
            return (match_counts.get(doc_id, 0), float(doc.get("score") or 0.0))

        others_sorted = sorted(others, key=_rank, reverse=True)
    else:
        others_sorted = []

    selected = (pinned_sorted + others_sorted)[:max_docs]

    def _entity_match(doc: Dict[str, Any], query_text: str) -> bool:
        brand_tokens = _brand_tokens(query_text)
        if not brand_tokens:
            return False
        text = f"{_doc_title_text(doc)} {str(doc.get('id') or '').lower()}"
        return any(tok in text for tok in brand_tokens)

    if selected:
        filtered = [
            doc
            for doc in selected
            if float(doc.get("score") or 0.0) > 0.0 or _entity_match(doc, query)
        ]
        if filtered:
            selected = filtered

    return selected
