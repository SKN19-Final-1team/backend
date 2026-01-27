from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

_RULES_PATH = Path(__file__).with_name("guidance_rules.json")


def _load_rules() -> Dict[str, Any]:
    with _RULES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f) or {}


def apply_guidance_rules(text: str, query: str, routing: Dict[str, Any]) -> str:
    if not text:
        return ""
    rules = _load_rules()
    cleaned = text
    for pattern in rules.get("banned_patterns", []):
        cleaned = re.sub(pattern, "", cleaned)
    for phrase in rules.get("banned_phrases", []):
        cleaned = cleaned.replace(phrase, "")
    for src, dst in (rules.get("replacements") or {}).items():
        cleaned = cleaned.replace(src, dst)

    # split to lines/sentences
    parts = [p.strip() for p in re.split(r"[\n]+", cleaned) if p.strip()]
    if len(parts) == 1:
        parts = [p.strip() for p in re.split(r"[.!?]+|[。！？]+", parts[0]) if p.strip()]

    # pick one core sentence + optional one question sentence, then merge to 1 line
    loss_terms = ("분실", "도난", "잃어버")
    loss_keys = ("정지", "분실", "도난", "신고", "재발급")
    question = next((p for p in parts if p.endswith("?") or p.endswith("？")), "")
    core = ""
    if any(t in query for t in loss_terms):
        core = next((p for p in parts if any(k in p for k in loss_keys)), "")
    if not core:
        core = parts[0] if parts else ""

    if question and question != core:
        cleaned = f"{core}. {question}".strip()
    else:
        cleaned = core.strip()

    if any(k in query for k in ("전화", "번호", "전화번호")) and not re.search(r"(전화|번호)", cleaned):
        cleaned = cleaned + " 전화번호 안내입니다."
    if any(k in query for k in ("신청", "발급")) and "신청" not in cleaned:
        cleaned = cleaned + " 신청은 카드사 심사 후 가능합니다."

    max_lines = int(rules.get("max_lines") or 2)
    cleaned_lines = [p.strip() for p in re.split(r"[\n]+", cleaned) if p.strip()]
    cleaned = "\n".join(cleaned_lines[:max_lines]).strip()

    return cleaned.strip()
