from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


CLASS_KEYWORDS: Dict[str, List[str]] = {
    "TROUBLESHOOTING": ["오류", "안돼", "안됨", "실패", "결제", "승인", "등록", "연동"],
    "PROCEDURE_STEPS": ["분실", "도난", "재발급", "해지", "발급", "신청"],
    "FEE_INTEREST": ["연회비", "수수료", "이자", "연체", "요율", "금리"],
    "ELIGIBILITY": ["조건", "대상", "가능", "자격", "제한"],
    "BENEFIT_SUMMARY": ["혜택", "할인", "적립", "포인트", "캐시백"],
    "DEFINITION": ["무엇", "뭐", "의미", "설명", "정의"],
}

PRIORITY = [
    "TROUBLESHOOTING",
    "PROCEDURE_STEPS",
    "FEE_INTEREST",
    "ELIGIBILITY",
    "BENEFIT_SUMMARY",
    "DEFINITION",
]


@dataclass(frozen=True)
class AnswerClassResult:
    primary: str
    secondary: List[str]


def classify(query: str) -> AnswerClassResult:
    normalized = (query or "").lower()
    matches: List[str] = []
    for cls, keywords in CLASS_KEYWORDS.items():
        if any(term in normalized for term in keywords):
            matches.append(cls)
    if not matches:
        return AnswerClassResult(primary="BENEFIT_SUMMARY", secondary=[])

    # priority
    primary = None
    for cls in PRIORITY:
        if cls in matches:
            primary = cls
            break
    secondary = [cls for cls in matches if cls != primary]
    return AnswerClassResult(primary=primary or matches[0], secondary=secondary)
