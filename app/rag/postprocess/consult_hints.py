from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json


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


_CARD_QUESTION_TOKENS = ("어떤 카드", "카드 종류", "카드명", "카드 이름", "재발급할 카드")


def _filter_questions(questions: List[str], filled_slots: Dict[str, List[str]]) -> List[str]:
    if not questions:
        return []
    card_names = filled_slots.get("card_name") or []
    if card_names:
        return [q for q in questions if not any(token in q for token in _CARD_QUESTION_TOKENS)]
    return questions


_RULES_PATH = Path(__file__).with_name("consult_hint_rules.json")


def _load_rules() -> Dict[str, Any]:
    try:
        raw = _RULES_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"scenario_tags": {}, "intent_keywords": {}, "fallback": {"flow_steps": [], "common_questions": []}}


def _merge_hints(
    base: Dict[str, List[str]],
    add_steps: List[str],
    add_questions: List[str],
    max_steps: int,
    max_qs: int,
) -> Dict[str, List[str]]:
    flow_steps = list(base.get("flow_steps") or [])
    questions = list(base.get("common_questions") or [])
    for step in add_steps:
        if step and step not in flow_steps and len(flow_steps) < max_steps:
            flow_steps.append(step)
    for q in add_questions:
        if q and q not in questions and len(questions) < max_qs:
            questions.append(q)
    return {"flow_steps": flow_steps, "common_questions": questions}


def _lookup_intent_rules(intents: List[str], rules: Dict[str, Any], max_steps: int, max_qs: int) -> Dict[str, List[str]]:
    intent_rules = rules.get("intent_keywords") or {}
    hints: Dict[str, List[str]] = {"flow_steps": [], "common_questions": []}
    for intent in intents:
        for key, payload in intent_rules.items():
            if key in intent:
                steps = payload.get("flow_steps") or []
                questions = payload.get("common_questions") or []
                hints = _merge_hints(hints, steps, questions, max_steps, max_qs)
    return hints


def _lookup_tag_rules(tags: List[str], rules: Dict[str, Any], max_steps: int, max_qs: int) -> Dict[str, List[str]]:
    tag_rules = rules.get("scenario_tags") or {}
    hints: Dict[str, List[str]] = {"flow_steps": [], "common_questions": []}
    for tag in tags:
        payload = tag_rules.get(tag)
        if not isinstance(payload, dict):
            continue
        steps = payload.get("flow_steps") or []
        questions = payload.get("common_questions") or []
        hints = _merge_hints(hints, steps, questions, max_steps, max_qs)
    return hints


def build_consult_hints(
    consult_docs: List[Dict[str, Any]],
    intent_terms: Optional[List[str]] = None,
    scenario_tags: Optional[List[str]] = None,
    filled_slots: Optional[Dict[str, List[str]]] = None,
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

    rules = _load_rules()
    merged = {"flow_steps": [], "common_questions": []}
    if scenario_tags:
        merged = _lookup_tag_rules(scenario_tags, rules, max_steps, max_qs)
    if intent_terms:
        intent_hints = _lookup_intent_rules(intent_terms, rules, max_steps, max_qs)
        merged = _merge_hints(
            merged,
            intent_hints.get("flow_steps") or [],
            intent_hints.get("common_questions") or [],
            max_steps,
            max_qs,
        )
    merged = _merge_hints(merged, flow_steps, questions, max_steps, max_qs)
    if filled_slots:
        merged["common_questions"] = _filter_questions(
            merged.get("common_questions") or [],
            filled_slots,
        )[:max_qs]
    if merged.get("flow_steps") or merged.get("common_questions"):
        return merged
    fallback = rules.get("fallback") or {}
    return {
        "flow_steps": list(fallback.get("flow_steps") or [])[:max_steps],
        "common_questions": list(fallback.get("common_questions") or [])[:max_qs],
    }


__all__ = ["build_consult_hints"]
