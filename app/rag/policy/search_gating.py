from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List


_ACK_PATTERNS = re.compile(r"^(네|아니요|아뇨|응|어|음|흠|하|헉|헐|ㅋㅋ+|ㅎㅎ+|ㄷㄷ+|오케이|ok|okay)$")
_DOMAIN_KEYWORDS = {
    "분실",
    "도난",
    "재발급",
    "해지",
    "발급",
    "신청",
    "연회비",
    "혜택",
    "수수료",
    "이자",
    "결제",
    "승인",
    "오류",
    "안돼",
    "안됨",
    "한도",
    "청구",
    "할부",
    "포인트",
    "적립",
    "카드",
    "등록",
    "연동",
    "고객센터",
    "전화번호",
}


def _compact(text: str) -> str:
    return (text or "").replace(" ", "")


def domain_signal_score(query: str, routing: Dict[str, Any]) -> int:
    normalized = (query or "").lower()
    compact = _compact(normalized)
    score = 0
    for term in _DOMAIN_KEYWORDS:
        if term in normalized or term in compact:
            score += 1
    matched = routing.get("matched") or {}
    card_names = matched.get("card_names") or []
    if card_names:
        score += 2
    actions = matched.get("actions") or []
    if actions:
        score += 1
    return score


def _is_ack_like(query: str) -> bool:
    normalized = (query or "").strip().lower()
    if not normalized:
        return True
    if _ACK_PATTERNS.match(normalized):
        return True
    return False


@dataclass(frozen=True)
class GatingDecision:
    no_search: bool
    domain_score: int
    retrieval_mode: str
    message: str


def decide_search_gating(query: str, routing: Dict[str, Any]) -> GatingDecision:
    normalized = (query or "").strip()
    domain_score = domain_signal_score(query, routing)
    retrieval_mode = "keyword_only" if domain_score < 5 else "hybrid"

    no_search = False
    if len(normalized) <= 3:
        no_search = domain_score == 0
    elif _is_ack_like(normalized) and domain_score == 0:
        no_search = True

    if no_search:
        message = "무엇을 도와드릴까요? (예: 분실/재발급/연회비/혜택)"
    else:
        message = ""
    return GatingDecision(
        no_search=no_search,
        domain_score=domain_score,
        retrieval_mode=retrieval_mode,
        message=message,
    )
