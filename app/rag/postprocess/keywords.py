from typing import Any, Dict, List

import re

from app.rag.vocab.keyword_dict import STOPWORDS

_TERM_WS_RE = re.compile(r"\s+")
_TERM_CLEAN_RE = re.compile(r"[^\w가-힣]+")
_PARTICLE_SUFFIXES = (
    "으로",
    "에서",
    "에게",
    "으로",
    "로",
    "와",
    "과",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "도",
    "만",
    "요",
    "죠",
)
_KEYWORD_STOPWORDS = {
    "무엇",
    "무엇인가요",
    "뭔가요",
    "뭐야",
    "뭐에요",
    "뭐예요",
    "뭔지",
    "어떤",
    "어떻게",
    "왜",
    "언제",
    "어디",
    "가능",
    "되나요",
    "되요",
    "해주세요",
    "알려줘",
    "알려주세요",
    "알려줘요",
    "방법",
    "소개",
    "정의",
    "란",
    "카드",
}
ISSUE_FILTER_TOKENS = ("발급", "신청", "재발급", "대상", "서류")
BENEFIT_FILTER_TOKENS = ("적립", "혜택", "유의", "제외", "포인트", "할인")


def _unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _strip_particle(term: str) -> str:
    for suffix in _PARTICLE_SUFFIXES:
        if term.endswith(suffix) and len(term) > len(suffix) + 1:
            return term[: -len(suffix)]
    return term


def normalize_text(text: str) -> str:
    return _TERM_WS_RE.sub(" ", text.strip().lower())


def extract_query_terms(query: str) -> List[str]:
    text = _TERM_WS_RE.sub(" ", query.strip().lower())
    raw_terms = [term for term in text.split(" ") if term]
    stopwords = {word.lower() for word in STOPWORDS}
    terms = []
    for term in raw_terms:
        term = _TERM_CLEAN_RE.sub("", term)
        if not term:
            continue
        if term in stopwords or term in _KEYWORD_STOPWORDS:
            continue
        if term.endswith("란") and len(term) > 1:
            term = term[:-1]
        term = _strip_particle(term)
        if not term:
            continue
        if term in stopwords or term in _KEYWORD_STOPWORDS:
            continue
        if term.isdigit():
            continue
        if len(term) < 2:
            continue
        terms.append(term)
    return _unique_in_order(terms)


def collect_query_keywords(query: str, routing: Dict[str, Any], normalize: bool) -> List[str]:
    if normalize:
        matched = routing.get("matched") or {}
        keywords: List[str] = []
        for key in ("card_names", "actions", "payments", "weak_intents"):
            keywords.extend(matched.get(key) or [])
    else:
        keywords = extract_query_terms(query)
    normalized = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        if not kw.startswith("#"):
            kw = f"#{kw}"
        normalized.append(kw)
    return _unique_in_order(normalized)


def text_has_any(text: str, tokens: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(token in lowered for token in tokens)
