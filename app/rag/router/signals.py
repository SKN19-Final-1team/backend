import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from flashtext import KeywordProcessor

from app.rag.common.text_utils import unique_in_order
from app.rag.vocab.keyword_dict import (
    ACTION_SYNONYMS,
    PAYMENT_SYNONYMS,
    WEAK_INTENT_SYNONYMS,
    get_card_name_synonyms,
    get_compound_patterns,
)

try:
    from rapidfuzz import fuzz, process  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fuzz = None
    process = None

_WS_RE = re.compile(r"\s+")
_FUZZY_CLEAN_RE = re.compile(r"[^\w가-힣]+")
_TOKEN_RE = re.compile(r"[0-9a-zA-Z가-힣]+")

_STRONG_ACTION_TOKENS = {
    "방법",
    "어떻게",
    "신청",
    "등록",
    "추가",
    "설정",
    "변경",
    "해지",
    "취소",
    "안됨",
    "오류",
    "실패",
    "인증",
    "재발급",
    "재발행",
    "재교부",
    "분실",
}

_INFO_HINT_TERMS = {
    "혜택",
    "연회비",
    "조건",
    "추천",
    "비교",
    "좋아",
    "좋은",
    "괜찮",
    "정보",
    "소개",
    "어떤",
    "뭐가",
}
_USAGE_STRONG_TERMS = {
    "분실",
    "도난",
    "재발급",
    "발급",
    "신청",
    "해지",
    "결제",
    "승인",
    "취소",
    "환불",
    "오류",
    "에러",
    "안돼",
    "불가",
    "거절",
    "신고",
    "등록",
    "인증",
    "서류",
    "사용",
    "사용처",
    "교통",
    "충전",
    "연체",
    "수수료",
    "이자",
    "이자율",
    "대출",
    "현금서비스",
    "카드론",
    "리볼빙",
    "입금",
    "납부",
    "한도",
}

_ISSUANCE_TERMS = {
    "발급",
    "요건",
    "자격",
    "대상",
    "심사",
    "조건",
    "등록",
    "신청",
}

_CARD_TOKEN_STOPWORDS = {
    "카드",
    "연회비",
    "발급",
    "신청",
    "조건",
    "가능",
    "여부",
    "있어요",
    "있나요",
    "있나",
    "뭐에요",
    "뭐예요",
    "뭐야",
    "어떻게",
    "어떤",
}


@dataclass(frozen=True)
class Signals:
    normalized: str
    card_names: List[str]
    actions: List[str]
    payments: List[str]
    weak_intents: List[str]
    pattern_hits: List[str]
    applepay_intent: Optional[str]
    info_hint: bool
    usage_strong: bool
    issuance_hint: bool

    @property
    def strong_signal(self) -> bool:
        return bool(
            self.card_names
            or self.actions
            or self.payments
            or self.pattern_hits
            or self.weak_intents
            or self.info_hint
        )


def _normalize_query(text: str) -> str:
    text = _WS_RE.sub(" ", text.strip())
    return text.lower()


def _compact_text(text: str) -> str:
    return _FUZZY_CLEAN_RE.sub("", text.lower())


def _extract_tokens(text: str) -> List[str]:
    tokens = []
    for token in _TOKEN_RE.findall(text.lower()):
        if len(token) < 2 or token in _CARD_TOKEN_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _expand_token_variants(token: str) -> List[str]:
    out = {token}
    if token.endswith("카드") and len(token) > 2:
        out.add(token[:-2])
    if token.startswith("카드") and len(token) > 2:
        out.add(token[2:])
    return [t for t in out if t and t not in _CARD_TOKEN_STOPWORDS]


def _has_any_term(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _build_processor(synonyms: Dict[str, List[str]]) -> KeywordProcessor:
    kp = KeywordProcessor(case_sensitive=False)
    for canonical, terms in synonyms.items():
        kp.add_keyword(canonical, canonical)
        for term in terms:
            kp.add_keyword(term, canonical)
    return kp


def _fallback_contains(synonyms: Dict[str, List[str]], text: str) -> List[str]:
    hits = []
    compact_text = text.replace(" ", "")
    for canonical, terms in synonyms.items():
        for term in [canonical, *terms]:
            if not term:
                continue
            lowered = term.lower()
            if lowered in text or lowered.replace(" ", "") in compact_text:
                hits.append(canonical)
                break
    return hits


def _action_has_nonweak_term(action: str, text: str, compact_text: str, weak_terms: set[str]) -> bool:
    terms = ACTION_SYNONYMS.get(action) or []
    for term in terms:
        if not term:
            continue
        lowered = term.lower()
        if lowered in weak_terms:
            continue
        if lowered in text or lowered.replace(" ", "") in compact_text:
            return True
    return False


def _filter_actions_with_weak_intents(
    actions: List[str],
    weak_intents: List[str],
    text: str,
    weak_terms: set[str],
) -> List[str]:
    if not actions or not weak_intents:
        return actions
    compact_text = text.replace(" ", "")
    return [action for action in actions if _action_has_nonweak_term(action, text, compact_text, weak_terms)]


_CARD_KP = None
_CARD_KP_SIZE = -1
_ACTION_SYNONYMS_WITH_ERROR = {**ACTION_SYNONYMS, **{"오류": ["에러", "오류가", "오류다", "에러가", "에러네", "안돼", "안돼요", "안되네", "안됨", "불가", "되지않음", "작동안함", "작동안돼", "등록안돼", "등록안됨", "결제안돼", "결제오류", "승인안됨", "인증안됨"]}}
_ACTION_KP = _build_processor(_ACTION_SYNONYMS_WITH_ERROR)
_PAYMENT_KP = _build_processor(PAYMENT_SYNONYMS)
_WEAK_INTENT_KP = _build_processor(WEAK_INTENT_SYNONYMS)
_WEAK_TERMS = {
    term.lower()
    for terms in WEAK_INTENT_SYNONYMS.values()
    for term in terms
    if isinstance(term, str)
}

_CARD_FUZZY = None
_CARD_FUZZY_SIZE = -1
_ACTION_FUZZY = None
_PAYMENT_FUZZY = None

FUZZY_ENABLED = os.getenv("RAG_ROUTER_FUZZY", "1") != "0"
FUZZY_TOP_N = int(os.getenv("RAG_ROUTER_FUZZY_TOP_N", "3"))
FUZZY_THRESHOLD = int(os.getenv("RAG_ROUTER_FUZZY_THRESHOLD", "85"))
FUZZY_CARD_THRESHOLD = int(os.getenv("RAG_ROUTER_FUZZY_CARD_THRESHOLD", "78"))
FUZZY_MAX_CANDIDATES = int(os.getenv("RAG_ROUTER_FUZZY_MAX_CANDIDATES", "1000"))
FUZZY_MIN_LEN = int(os.getenv("RAG_ROUTER_FUZZY_MIN_LEN", "3"))
CARD_TOKEN_MIN_SCORE = int(os.getenv("RAG_ROUTER_CARD_TOKEN_MIN_SCORE", "3"))
CARD_TOKEN_MAX_HITS = int(os.getenv("RAG_ROUTER_CARD_TOKEN_MAX_HITS", "3"))


def _build_fuzzy_candidates(synonyms: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, str]]:
    candidates: List[str] = []
    mapping: Dict[str, str] = {}
    for canonical, terms in synonyms.items():
        for term in [canonical, *terms]:
            if not term or term in mapping:
                continue
            candidates.append(term)
            mapping[term] = canonical
    return candidates, mapping


def _ensure_card_kp() -> KeywordProcessor:
    global _CARD_KP, _CARD_KP_SIZE
    synonyms = get_card_name_synonyms()
    size = len(synonyms)
    if _CARD_KP is None or size != _CARD_KP_SIZE:
        _CARD_KP = _build_processor(synonyms)
        _CARD_KP_SIZE = size
    return _CARD_KP


def _ensure_card_fuzzy():
    global _CARD_FUZZY, _CARD_FUZZY_SIZE
    synonyms = get_card_name_synonyms()
    size = len(synonyms)
    if _CARD_FUZZY is None or size != _CARD_FUZZY_SIZE:
        _CARD_FUZZY = _build_fuzzy_candidates(synonyms)
        _CARD_FUZZY_SIZE = size
    return _CARD_FUZZY


def _ensure_action_fuzzy():
    global _ACTION_FUZZY
    if _ACTION_FUZZY is None:
        _ACTION_FUZZY = _build_fuzzy_candidates(_ACTION_SYNONYMS_WITH_ERROR)
    return _ACTION_FUZZY


def _ensure_payment_fuzzy():
    global _PAYMENT_FUZZY
    if _PAYMENT_FUZZY is None:
        _PAYMENT_FUZZY = _build_fuzzy_candidates(PAYMENT_SYNONYMS)
    return _PAYMENT_FUZZY


def _fuzzy_match(
    query: str,
    candidates: List[str],
    mapping: Dict[str, str],
    scorer=None,
    processor=None,
    threshold: Optional[int] = None,
) -> List[str]:
    if not FUZZY_ENABLED or fuzz is None or process is None:
        return []
    if not query or len(query) < FUZZY_MIN_LEN or not candidates:
        return []
    if len(candidates) > FUZZY_MAX_CANDIDATES:
        candidates = candidates[:FUZZY_MAX_CANDIDATES]
    cutoff = FUZZY_THRESHOLD if threshold is None else threshold
    results = process.extract(
        query,
        candidates,
        scorer=scorer or fuzz.WRatio,
        processor=processor,
        limit=FUZZY_TOP_N,
        score_cutoff=cutoff,
    )
    hits = []
    for term, _, _ in results:
        canon = mapping.get(term)
        if canon:
            hits.append(canon)
    return unique_in_order(hits)


def _card_token_match(query: str, synonyms: Dict[str, List[str]]) -> List[str]:
    tokens = _extract_tokens(query)
    if not tokens:
        return []
    variants = []
    for token in tokens:
        variants.extend(_expand_token_variants(token))
    if not variants:
        return []
    token_weights = {}
    for token in variants:
        weight = len(token)
        if any(ch.isascii() and ch.isalnum() for ch in token):
            weight += 2
        token_weights[token] = weight
    best_score = 0
    hits: List[str] = []
    for name in synonyms.keys():
        name_compact = _compact_text(name)
        score = 0
        for token in variants:
            if token and token in name_compact:
                score += token_weights.get(token, len(token))
        if score <= 0:
            continue
        if score > best_score:
            best_score = score
            hits = [name]
        elif score == best_score:
            hits.append(name)
    if best_score < CARD_TOKEN_MIN_SCORE or not hits:
        return []
    if len(hits) > CARD_TOKEN_MAX_HITS:
        hits = sorted(hits, key=len)[:CARD_TOKEN_MAX_HITS]
    return hits


def _match_compound_patterns(text: str) -> List[str]:
    hits = []
    for rule in get_compound_patterns():
        if rule.pattern.search(text):
            hits.append(rule.category)
    return hits


def _detect_applepay_intent(normalized: str, payments: List[str]) -> Optional[str]:
    has_applepay_term = any(term in normalized for term in ["애플페이", "apple pay", "applepay", "지갑", "wallet"])
    if not has_applepay_term:
        return None
    if ("애플페이" in payments or not payments):
        if any(t in normalized for t in ["등록", "추가", "카드", "지갑"]) and any(
            t in normalized for t in ["안돼", "오류", "안됨", "완료", "안되", "되지", "못"]
        ):
            return "applepay_add_card"
        if any(t in normalized for t in ["등록", "추가", "카드", "지갑", "추가하"]):
            return "applepay_add_card"
        if any(t in normalized for t in ["결제", "결제하"]):
            return "applepay_payment"
        if any(t in normalized for t in ["티머니", "교통", "충전", "탑승"]):
            return "applepay_transport"
        if any(t in normalized for t in ["사용처", "어디", "가능", "가맹점", "이용처", "어디서"]):
            return "applepay_where_to_use"
        if any(t in normalized for t in ["분실", "도난", "보안", "삭제", "분실했", "도난당"]):
            return "applepay_security"
        return "applepay_general"
    return None


def extract_signals(query: str) -> Signals:
    normalized = _normalize_query(query)
    card_kp = _ensure_card_kp()
    card_names = unique_in_order(card_kp.extract_keywords(normalized))
    actions = unique_in_order(_ACTION_KP.extract_keywords(normalized))
    payments = unique_in_order(_PAYMENT_KP.extract_keywords(normalized))
    weak_intents = unique_in_order(_WEAK_INTENT_KP.extract_keywords(normalized))

    if not card_names:
        card_names = unique_in_order(_fallback_contains(get_card_name_synonyms(), normalized))
    if not actions:
        actions = unique_in_order(_fallback_contains(ACTION_SYNONYMS, normalized))
    if not payments:
        payments = unique_in_order(_fallback_contains(PAYMENT_SYNONYMS, normalized))
    if not weak_intents:
        weak_intents = unique_in_order(_fallback_contains(WEAK_INTENT_SYNONYMS, normalized))

    if not card_names:
        synonyms = get_card_name_synonyms()
        card_names = unique_in_order(_card_token_match(normalized, synonyms))
        if not card_names and len(normalized) >= FUZZY_MIN_LEN and fuzz is not None and process is not None:
            candidates, mapping = _ensure_card_fuzzy()
            card_names = unique_in_order(
                _fuzzy_match(
                    normalized,
                    candidates,
                    mapping,
                    scorer=fuzz.partial_ratio,
                    processor=_compact_text,
                    threshold=FUZZY_CARD_THRESHOLD,
                )
            )

    if not actions and len(normalized) >= FUZZY_MIN_LEN and fuzz is not None and process is not None:
        candidates, mapping = _ensure_action_fuzzy()
        actions = unique_in_order(_fuzzy_match(normalized, candidates, mapping))

    if not payments and len(normalized) >= FUZZY_MIN_LEN and fuzz is not None and process is not None:
        candidates, mapping = _ensure_payment_fuzzy()
        payments = unique_in_order(_fuzzy_match(normalized, candidates, mapping))

    actions = _filter_actions_with_weak_intents(actions, weak_intents, normalized, _WEAK_TERMS)

    pattern_hits = _match_compound_patterns(query)
    if pattern_hits:
        actions = unique_in_order([*actions, *pattern_hits])

    applepay_intent = _detect_applepay_intent(normalized, payments)
    info_hint = _has_any_term(normalized, _INFO_HINT_TERMS)
    usage_strong = _has_any_term(normalized, _USAGE_STRONG_TERMS)
    issuance_hint = _has_any_term(normalized, _ISSUANCE_TERMS)

    return Signals(
        normalized=normalized,
        card_names=card_names,
        actions=actions,
        payments=payments,
        weak_intents=weak_intents,
        pattern_hits=pattern_hits,
        applepay_intent=applepay_intent,
        info_hint=info_hint,
        usage_strong=usage_strong,
        issuance_hint=issuance_hint,
    )


def first(items: List[str]) -> Optional[str]:
    return items[0] if items else None


def route_tuple(route: str, db_route: str, boost: Optional[Dict[str, List[str]]] = None, template: Optional[str] = None, trigger: bool = False):
    return route, db_route, boost or {}, template, trigger
