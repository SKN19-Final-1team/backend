from __future__ import annotations

from typing import Dict, List


def build_consult_hint_message(hints: Dict[str, List[str]]) -> str:
    flow = hints.get("flow_steps") or []
    questions = hints.get("common_questions") or []
    parts: List[str] = []
    if flow:
        parts.append("안내 순서: " + ", ".join(flow[:3]))
    if questions:
        parts.append("확인 질문: " + questions[0])
    return " ".join(parts).strip()


__all__ = ["build_consult_hint_message"]
