from __future__ import annotations

import re
from typing import Any, Dict, List


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    for sep in (". ", "! ", "? ", "\n"):
        if sep in text:
            text = text.split(sep, 1)[0]
            break
    return text.strip()


def _clean_line(text: str, limit: int = 120) -> str:
    text = (text or "").strip()
    # 전화번호/고객센터 등 불필요한 숫자 노출 제거
    phone_dash = r"[\-–—‑]"
    text = re.sub(rf"\b\d{{2,4}}{phone_dash}\d{{3,4}}{phone_dash}\d{{4}}\b", "", text)
    text = re.sub(rf"\b\d{{2,4}}{phone_dash}\d{{4}}\b", "", text)
    text = re.sub(r"\b\d{8,11}\b", "", text)
    text = re.sub(r"(고객센터|콜센터|문의|연락처)[^\n]*", "", text)
    text = text.replace("테디카드", "").replace("신용정보 알림서비스", "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def build_guide_script_message(
    docs: List[Dict[str, Any]],
    consult_docs: List[Dict[str, Any]],
    guidance_script: str,
) -> str:
    doc_line = ""
    if docs:
        doc_line = _first_sentence(docs[0].get("content") or docs[0].get("title") or "")
    consult_line = ""
    if consult_docs:
        consult_line = _first_sentence(consult_docs[0].get("content") or consult_docs[0].get("title") or "")

    parts = []
    if doc_line:
        parts.append("문서 안내: " + _clean_line(doc_line))
    if consult_line:
        parts.append("상담 사례: " + _clean_line(consult_line))
    if not parts and guidance_script:
        parts.append(_clean_line(guidance_script, 160))

    if not parts:
        return ""
    message = "\n".join(parts)
    if not message.endswith("요"):
        message = message + " 필요하시면 추가 확인해 드릴게요."
    return message


__all__ = ["build_guide_script_message"]
