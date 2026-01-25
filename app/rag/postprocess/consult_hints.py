from __future__ import annotations

from typing import Any, Dict, List, Optional


_ROLE_PREFIXES = ("상담사:", "손님:", "고객:", "고객님:")


def _strip_role_prefix(line: str) -> str:
    line = (line or "").strip()
    for prefix in _ROLE_PREFIXES:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return line


def _is_low_signal(line: str) -> bool:
    if not line:
        return True
    if line in {"네", "네.", "예", "예.", "알겠습니다", "감사합니다", "감사합니다."}:
        return True
    if any(line.startswith(prefix) for prefix in _ROLE_PREFIXES):
        return True
    return len(line) < 6


_FALLBACK_INTENT_HINTS = {
    "분실": (
        ["분실 카드 확인", "분실 장소/일시 확인", "카드 정지 및 재발급 안내"],
        ["분실하신 카드 종류를 확인해도 될까요?", "분실 장소와 날짜를 확인해도 될까요?"],
    ),
    "도난": (
        ["도난 카드 확인", "도난 장소/일시 확인", "카드 정지 및 재발급 안내"],
        ["도난 카드 종류를 확인해도 될까요?", "도난 장소와 날짜를 확인해도 될까요?"],
    ),
    "재발급": (
        ["카드 재발급 대상 확인", "재발급 신청 진행", "배송/수령 안내"],
        ["재발급할 카드 종류를 확인해도 될까요?", "수령 주소를 확인해도 될까요?"],
    ),
    "승인": (
        ["결제 승인 내역 확인", "이용 본인 여부 확인", "필요 시 차단/정지 안내"],
        ["해당 승인 건이 본인 이용인지 확인해도 될까요?"],
    ),
    "취소": (
        ["결제 취소 요청 확인", "취소 처리 경로 안내", "환불 일정 안내"],
        ["취소 요청하신 결제 건을 확인해도 될까요?"],
    ),
    "수수료": (
        ["수수료 발생 사유 확인", "적용 기준 안내", "추가 문의 경로 안내"],
        ["확인하실 수수료 유형을 알려주실 수 있을까요?"],
    ),
    "한도": (
        ["현재 한도 확인", "한도 변경 가능 여부 안내", "필요 서류/절차 안내"],
        ["확인하실 한도 종류(일/월)를 알려주실 수 있을까요?"],
    ),
    "현금서비스": (
        ["현금서비스 이용 가능 여부 확인", "이용 절차 안내", "수수료/이자 안내"],
        ["현금서비스 이용 금액을 알려주실 수 있을까요?"],
    ),
    "분실도난": (
        ["분실/도난 카드 확인", "분실/도난 장소·일시 확인", "카드 정지 및 재발급 안내"],
        ["분실하신 카드 종류를 확인해도 될까요?", "분실 장소와 날짜를 확인해도 될까요?"],
    ),
    "도난/분실": (
        ["분실/도난 카드 확인", "분실/도난 장소·일시 확인", "카드 정지 및 재발급 안내"],
        ["분실하신 카드 종류를 확인해도 될까요?", "분실 장소와 날짜를 확인해도 될까요?"],
    ),
}


def _fallback_from_intents(intents: List[str], max_steps: int, max_qs: int) -> Dict[str, List[str]]:
    flow_steps: List[str] = []
    questions: List[str] = []
    for intent in intents:
        for key, (steps, qs) in _FALLBACK_INTENT_HINTS.items():
            if key in intent:
                flow_steps.extend(steps)
                questions.extend(qs)
    return {
        "flow_steps": flow_steps[:max_steps],
        "common_questions": questions[:max_qs],
    }


def build_consult_hints(
    consult_docs: List[Dict[str, Any]],
    intent_terms: Optional[List[str]] = None,
    max_steps: int = 3,
    max_qs: int = 2,
) -> Dict[str, List[str]]:
    flow_steps: List[str] = []
    questions: List[str] = []

    for doc in consult_docs[:2]:
        content = doc.get("content") or ""
        for raw in content.splitlines():
            line = _strip_role_prefix(raw)
            if _is_low_signal(line):
                continue
            # 질문은 확인 질문 후보로 분리
            if "?" in line and len(questions) < max_qs:
                questions.append(line)
                continue
            if len(flow_steps) < max_steps:
                flow_steps.append(line)
            if len(flow_steps) >= max_steps and len(questions) >= max_qs:
                break
        if len(flow_steps) >= max_steps and len(questions) >= max_qs:
            break

    hints = {"flow_steps": flow_steps, "common_questions": questions}
    if (not flow_steps and not questions) and intent_terms:
        return _fallback_from_intents(intent_terms, max_steps, max_qs)
    return hints


__all__ = ["build_consult_hints"]
