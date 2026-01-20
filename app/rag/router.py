import os
import re
from typing import Dict, List, Optional, Tuple

from flashtext import KeywordProcessor

# vocab 규칙 정의
from app.rag.vocab.keyword_dict import (
    ACTION_ALLOWLIST,
    ACTION_SYNONYMS,
    PAYMENT_ALLOWLIST,
    PAYMENT_SYNONYMS,
    WEAK_INTENT_ROUTE_HINTS,
    WEAK_INTENT_SYNONYMS,
    ROUTE_CARD_INFO,
    ROUTE_CARD_USAGE,
    get_card_name_synonyms,
    get_compound_patterns,
)

try:
    from rapidfuzz import fuzz, process  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fuzz = None
    process = None

#     리스트에서 중복을 제거, 최초 등장 순서는 유지
def _unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


_WS_RE = re.compile(r"\s+")
_FUZZY_CLEAN_RE = re.compile(r"[^\w가-힣]+")
_TOKEN_RE = re.compile(r"[0-9a-zA-Z가-힣]+")

# Router 설정 (환경변수로 조정 가능)
STRICT_SEARCH = os.getenv("RAG_ROUTER_STRICT_SEARCH", "1") != "0"
MIN_QUERY_LEN = int(os.getenv("RAG_ROUTER_MIN_QUERY_LEN", "2"))
FUZZY_ENABLED = os.getenv("RAG_ROUTER_FUZZY", "1") != "0"
FUZZY_TOP_N = int(os.getenv("RAG_ROUTER_FUZZY_TOP_N", "3"))
FUZZY_THRESHOLD = int(os.getenv("RAG_ROUTER_FUZZY_THRESHOLD", "85"))
FUZZY_CARD_THRESHOLD = int(os.getenv("RAG_ROUTER_FUZZY_CARD_THRESHOLD", "78"))
FUZZY_MAX_CANDIDATES = int(os.getenv("RAG_ROUTER_FUZZY_MAX_CANDIDATES", "1000"))
FUZZY_MIN_LEN = int(os.getenv("RAG_ROUTER_FUZZY_MIN_LEN", "3"))
CARD_TOKEN_MIN_SCORE = int(os.getenv("RAG_ROUTER_CARD_TOKEN_MIN_SCORE", "3"))
CARD_TOKEN_MAX_HITS = int(os.getenv("RAG_ROUTER_CARD_TOKEN_MAX_HITS", "3"))

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

#   사용자 입력 문장 정규화 = 중복 공백 제거, 소문자 변환
def _normalize_query(text: str) -> str:
    text = _WS_RE.sub(" ", text.strip())
    return text.lower()


def _compact_text(text: str) -> str:
    return _FUZZY_CLEAN_RE.sub("", text.lower())


def _extract_tokens(text: str) -> List[str]:
    tokens = []
    for token in _TOKEN_RE.findall(text.lower()):
        if len(token) < 2:
            continue
        if token in _CARD_TOKEN_STOPWORDS:
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

#     FlashText KeywordProcessor 생성 = 동의어 canonical 값으로 매핑
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

# FlashText 프로세서 사전 생성 (카드는 DB 기반이므로 지연 로딩)
_CARD_KP = None
_CARD_KP_SIZE = -1
_ACTION_KP = _build_processor(ACTION_SYNONYMS)
_PAYMENT_KP = _build_processor(PAYMENT_SYNONYMS)
_WEAK_INTENT_KP = _build_processor(WEAK_INTENT_SYNONYMS)

_CARD_FUZZY = None
_CARD_FUZZY_SIZE = -1
_ACTION_FUZZY = None
_PAYMENT_FUZZY = None


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
        _ACTION_FUZZY = _build_fuzzy_candidates(ACTION_SYNONYMS)
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
    if not query or len(query) < FUZZY_MIN_LEN:
        return []
    if not candidates:
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
    return _unique_in_order(hits)


def _card_token_match(query: str, synonyms: Dict[str, List[str]]) -> List[str]:
    tokens = _extract_tokens(query)
    if not tokens:
        return []
    variants = []
    for token in tokens:
        variants.extend(_expand_token_variants(token))
    if not variants:
        return []
    best_score = 0
    hits: List[str] = []
    for name in synonyms.keys():
        name_compact = _compact_text(name)
        score = 0
        for token in variants:
            if token and token in name_compact:
                score += len(token)
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
        return []
    return hits


def _match_compound_patterns(text: str) -> List[str]:
    hits = []
    for rule in get_compound_patterns():
        if rule.pattern.search(text):
            hits.append(rule.category)
    return hits


def route_query(query: str) -> Dict[str, Optional[object]]:
    normalized = _normalize_query(query)
    card_kp = _ensure_card_kp()
    card_names = _unique_in_order(card_kp.extract_keywords(normalized))
    actions = _unique_in_order(_ACTION_KP.extract_keywords(normalized))
    payments = _unique_in_order(_PAYMENT_KP.extract_keywords(normalized))
    weak_intents = _unique_in_order(_WEAK_INTENT_KP.extract_keywords(normalized))

    if not card_names:
        card_names = _unique_in_order(_fallback_contains(get_card_name_synonyms(), normalized))
    if not actions:
        actions = _unique_in_order(_fallback_contains(ACTION_SYNONYMS, normalized))
    if not payments:
        payments = _unique_in_order(_fallback_contains(PAYMENT_SYNONYMS, normalized))
    if not weak_intents:
        weak_intents = _unique_in_order(_fallback_contains(WEAK_INTENT_SYNONYMS, normalized))

    if not card_names:
        synonyms = get_card_name_synonyms()
        card_names = _unique_in_order(_card_token_match(normalized, synonyms))
        if not card_names:
            candidates, mapping = _ensure_card_fuzzy()
            card_names = _unique_in_order(
                _fuzzy_match(
                    normalized,
                    candidates,
                    mapping,
                    scorer=fuzz.partial_ratio if fuzz else None,
                    processor=_compact_text,
                    threshold=FUZZY_CARD_THRESHOLD,
                )
            )
    if not actions:
        candidates, mapping = _ensure_action_fuzzy()
        actions = _unique_in_order(_fuzzy_match(normalized, candidates, mapping))
    if not payments:
        candidates, mapping = _ensure_payment_fuzzy()
        payments = _unique_in_order(_fuzzy_match(normalized, candidates, mapping))

    pattern_hits = _match_compound_patterns(query)
    if pattern_hits:
        actions = _unique_in_order([*actions, *pattern_hits])

    ui_route = None
    db_route = None  # "card_tbl" | "guide_tbl" | "both"
    boost: Dict[str, List[str]] = {}
    query_template = None

    strong_signal = bool(card_names or actions or payments or pattern_hits)
    if STRICT_SEARCH:
        should_search = strong_signal and len(normalized) >= MIN_QUERY_LEN
    else:
        should_search = True
    should_trigger = False

    # 1) 카드 + 액션: 둘 다 있으니 가장 강함
    if card_names and actions:
        ui_route = ROUTE_CARD_USAGE
        db_route = "both"
        boost = {"card_name": card_names, "intent": actions}
        if payments:
            boost["payment_method"] = payments
        if weak_intents:
            boost["weak_intent"] = weak_intents
        query_template = f"{card_names[0]} {actions[0]} 방법"
        should_trigger = True

    # 2) 카드 + 결제수단
    elif card_names and payments:
        ui_route = ROUTE_CARD_USAGE
        db_route = "card_tbl"
        boost = {"card_name": card_names, "payment_method": payments}
        query_template = f"{card_names[0]} {payments[0]} 사용 방법"
        should_trigger = True

    # 3) 카드 + 약한의도
    elif card_names and weak_intents:
        ui_route = WEAK_INTENT_ROUTE_HINTS.get(weak_intents[0], ROUTE_CARD_USAGE)
        db_route = "both"
        boost = {"card_name": card_names, "weak_intent": weak_intents}
        if ui_route == ROUTE_CARD_INFO:
            query_template = f"{card_names[0]} {weak_intents[0]}"
        else:
            query_template = f"{card_names[0]} {weak_intents[0]} 방법"
        should_trigger = True

    # 4) 카드만
    elif card_names:
        ui_route = ROUTE_CARD_INFO
        db_route = "card_tbl"
        boost = {"card_name": card_names}
        query_template = f"{card_names[0]} 정보"
        should_trigger = True

    # 5) 액션만 (중요! allowlist로 검색을 막지 말기)
    elif actions:
        ui_route = ROUTE_CARD_USAGE
        db_route = "guide_tbl"
        boost = {"intent": actions}
        if payments:
            boost["payment_method"] = payments
        query_template = f"카드 {actions[0]} 방법"
        should_trigger = any(a in ACTION_ALLOWLIST for a in actions)

    # 6) 결제수단만
    elif payments:
        ui_route = ROUTE_CARD_USAGE
        db_route = "card_tbl"
        boost = {"payment_method": payments}
        query_template = f"{payments[0]} 사용 방법"
        should_trigger = any(p in PAYMENT_ALLOWLIST for p in payments)

    # 7) 아무것도 못 잡으면: fallback 검색
    else:
        ui_route = ROUTE_CARD_USAGE
        db_route = "both"
        boost = {}
        query_template = None
        should_trigger = False

    return {
        "route": ui_route,
        "filters": boost,
        "ui_route": ui_route,
        "db_route": db_route,
        "boost": boost,
        "query_template": query_template,
        "matched": {
            "card_names": card_names,
            "actions": actions,
            "payments": payments,
            "weak_intents": weak_intents,
        },
        "should_search": should_search,
        "should_trigger": should_trigger,
        "should_route": should_trigger,  # 기존 키 유지
    }
