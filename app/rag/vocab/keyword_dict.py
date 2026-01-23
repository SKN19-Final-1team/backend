from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Pattern, Set
import json
import os
import re

from dotenv import load_dotenv


KEYWORD_DICT_PATH = Path(__file__).with_name("keywords_dict_v2_with_patterns.json")

# 라우팅 힌트용
ROUTE_CARD_INFO = "card_info"
ROUTE_CARD_USAGE = "card_usage"

# ACTION 중에서도 즉시 대응 가능한 intent만 허용
ACTION_ALLOWLIST = {"분실", "분실도난", "오류"}

# 문제/오류 관련 intent (고급 검색용)
ERROR_INTENT_SYNONYMS: Dict[str, List[str]] = {
    "오류": ["에러", "오류가", "오류다", "에러가", "에러네", "안돼", "안돼요", "안되네", "안됨", "불가", "되지않음", "작동안함", "작동안돼", "등록안돼", "등록안됨", "결제안돼", "결제오류", "승인안됨", "인증안됨"],
}

# 애플페이 전용 intent (source-specific routing용)
APPLEPAY_INTENT_SYNONYMS: Dict[str, List[str]] = {
    "applepay_add_card": ["등록", "추가", "카드추가", "카드등록", "지갑추가", "카드를 추가", "카드 추가", "카드 등록"],
    "applepay_payment": ["결제", "오프라인", "온라인", "할부", "취소", "리더기", "태그", "NFC"],
    "applepay_transport": ["티머니", "교통", "충전", "교통카드", "탑승", "익스프레스", "버스", "지하철"],
    "applepay_where_to_use": ["사용처", "어디서", "가능", "매장", "이용처", "가능 지점", "가맹점"],
    "applepay_security": ["분실", "도난", "보안", "삭제", "비밀번호", "분실모드"],
}

# 결제 수단 관련 키워드 (사전에서 누락된 항목 보완)
PAYMENT_SYNONYMS: Dict[str, List[str]] = {
    "iM유페이": ["iM 유페이", "iM유 페이", "im유페이"],
    "네이버페이": ["naver pay", "네이버 페이"],
    "삼성페이": ["samsung pay", "삼성 페이"],
    "애플페이": ["apple pay", "applepay", "애플 페이"],
    "카카오페이": ["kakao pay", "카카오 페이"],
    "티머니": ["t-money", "tmoney", "t머니", "티 머니"],
}
PAYMENT_ALLOWLIST = set(PAYMENT_SYNONYMS.keys())

# 약한 의도 단독 등장 시 기본 라우팅 힌트
WEAK_INTENT_ROUTE_HINTS = {
    "혜택": ROUTE_CARD_INFO,
    "발급": ROUTE_CARD_USAGE,
    "신청": ROUTE_CARD_USAGE,
    "사용": ROUTE_CARD_USAGE,
    "사용처": ROUTE_CARD_USAGE,
}

WEAK_INTENT_CANONICALS = tuple(WEAK_INTENT_ROUTE_HINTS.keys())

STOPWORDS = {"n/a", "na", "none", "문의", "안내"}

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CompoundPattern:
    category: str
    pattern: Pattern[str]


def _expand_variants(term: str) -> Set[str]:
    if not term:
        return set()
    base = term.strip()
    if not base:
        return set()
    variants = {base, base.lower()}
    variants.add(_WS_RE.sub("", base))
    variants.add(base.replace("-", " "))
    variants.add(base.replace(" - ", "-"))
    variants.add(base.replace("·", " "))
    variants.add(base.replace("·", ""))
    return {v for v in variants if v}


def _expand_card_variants(term: str) -> Set[str]:
    variants = _expand_variants(term)
    if not term:
        return variants
    variants.add(term.replace("-", ""))
    variants.add(term.replace(" ", "").replace("-", ""))
    return {v for v in variants if v}


def _collect_terms(key: str, entry: Dict[str, object]) -> Set[str]:
    terms = set()
    for term in [
        key,
        entry.get("canonical"),
        *list(entry.get("synonyms") or []),
        *list(entry.get("variations") or []),
    ]:
        if isinstance(term, str):
            terms.update(_expand_variants(term))
    for pat in entry.get("compound_patterns") or []:
        for term in pat.get("keywords") or []:
            if isinstance(term, str):
                terms.update(_expand_variants(term))
    return {t for t in terms if t}


def _choose_primary_category(key: str, categories: List[Dict[str, object]]) -> str | None:
    if not categories:
        return None
    for cat in categories:
        if cat.get("category") == key:
            return key
    best_name = None
    best_priority = -1
    for cat in categories:
        name = cat.get("category")
        if not name:
            continue
        priority = cat.get("priority", 0)
        try:
            score = int(priority)
        except (TypeError, ValueError):
            score = 0
        if score > best_priority:
            best_priority = score
            best_name = name
    return best_name


@lru_cache(maxsize=1)
def _keyword_entries() -> Dict[str, Dict[str, object]]:
    try:
        raw = KEYWORD_DICT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    keywords = data.get("keywords")
    return keywords if isinstance(keywords, dict) else {}


@lru_cache(maxsize=1)
def get_action_synonyms() -> Dict[str, List[str]]:
    out: Dict[str, Set[str]] = {}
    for key, entry in _keyword_entries().items():
        if key in WEAK_INTENT_CANONICALS:
            continue
        categories = entry.get("categories") or []
        primary = _choose_primary_category(key, categories)
        if not primary:
            continue
        terms = _collect_terms(key, entry)
        bucket = out.setdefault(primary, set())
        bucket.update(terms)
        bucket.add(primary)
    return {k: sorted(v) for k, v in out.items()}


@lru_cache(maxsize=1)
def get_weak_intent_synonyms() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    keywords = _keyword_entries()
    for canon in WEAK_INTENT_CANONICALS:
        terms = {canon}
        entry = keywords.get(canon)
        if entry:
            terms.update(_collect_terms(canon, entry))
        out[canon] = sorted(terms)
    return out


@lru_cache(maxsize=1)
def get_compound_patterns() -> List[CompoundPattern]:
    patterns: List[CompoundPattern] = []
    for entry in _keyword_entries().values():
        for rule in entry.get("compound_patterns") or []:
            pattern = rule.get("pattern")
            category = rule.get("category")
            if not pattern or not category:
                continue
            try:
                patterns.append(CompoundPattern(category=category, pattern=re.compile(pattern, re.I)))
            except re.error:
                continue
    return patterns


_CARD_NAME_CACHE: Dict[str, List[str]] | None = None


def get_card_name_synonyms() -> Dict[str, List[str]]:
    global _CARD_NAME_CACHE
    if _CARD_NAME_CACHE is not None:
        return _CARD_NAME_CACHE
    load_dotenv()
    host = os.getenv("DB_HOST_IP") or os.getenv("DB_HOST")
    cfg = {
        "host": host,
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", "0")) or None,
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        return {}
    try:
        import psycopg2  # lazy import
    except Exception:
        return {}
    try:
        with psycopg2.connect(connect_timeout=3, **cfg) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT name FROM card_products WHERE name IS NOT NULL ORDER BY 1;")
                names = [row[0] for row in cur.fetchall() if row and row[0]]
                cur.execute(
                    "SELECT DISTINCT metadata->>'card_name' "
                    "FROM service_guide_documents "
                    "WHERE metadata ? 'card_name' "
                    "AND metadata->>'card_name' IS NOT NULL "
                    "AND metadata->>'card_name' <> '' "
                    "ORDER BY 1;"
                )
                guide_names = [row[0] for row in cur.fetchall() if row and row[0]]
    except Exception:
        return {}
    combined = {name for name in [*names, *guide_names] if name}
    _CARD_NAME_CACHE = {
        name: sorted(_expand_card_variants(name) - {name})
        for name in combined
    }
    return _CARD_NAME_CACHE


ACTION_SYNONYMS = get_action_synonyms()
WEAK_INTENT_SYNONYMS = get_weak_intent_synonyms()

# ERROR_INTENT_SYNONYMS를 ACTION_SYNONYMS에 병합 - 안돼요/오류 등을 action으로 인식
ACTION_SYNONYMS.update(ERROR_INTENT_SYNONYMS)


def get_vocab_groups() -> List[Dict[str, object]]:
    return [
        {
            "type": "CARD/PROGRAM",
            "route": "card_tbl",
            "cooldown_sec": 8,
            "synonyms": get_card_name_synonyms(),
            "filter_key": "card_name",
        },
        {
            "type": "INTENT",
            "route": "guide_tbl",
            "cooldown_sec": 5,
            "synonyms": ACTION_SYNONYMS,
            "filter_key": "intent",
        },
        {
            "type": "WEAK_INTENT",
            "route": "guide_tbl",
            "cooldown_sec": 5,
            "synonyms": WEAK_INTENT_SYNONYMS,
            "filter_key": "weak_intent",
        },
        {
            "type": "PAYMENT",
            "route": "card_tbl",
            "cooldown_sec": 6,
            "synonyms": PAYMENT_SYNONYMS,
            "filter_key": "payment_method",
        },
    ]
