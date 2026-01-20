from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

from app.rag.postprocess.keywords import extract_query_terms


_SECTION_RE = re.compile(r"^#{1,3}\s*(.+)$")
_BLOCK_SPLIT_RE = re.compile(r"\n{2,}")
_BLOCK_NORM_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[0-9a-zA-Z가-힣]+")

_SECTION_HINTS = {
    "연회비",
    "발급",
    "대상",
    "신청",
    "혜택",
    "적립",
    "할인",
    "사용처",
    "유의사항",
    "조건",
    "자격",
    "수수료",
}


def _normalize_block(text: str) -> str:
    return _BLOCK_NORM_RE.sub(" ", text.strip().lower())


def _split_blocks(text: str) -> List[str]:
    return [b.strip() for b in _BLOCK_SPLIT_RE.split(text) if b.strip()]


def _dedupe_blocks(text: str) -> str:
    blocks = _split_blocks(text)
    seen = set()
    out = []
    for block in blocks:
        key = _normalize_block(block)
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
    return "\n\n".join(out)


def _split_sections(text: str) -> List[Tuple[Optional[str], str]]:
    lines = text.splitlines()
    sections: List[Tuple[Optional[str], str]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        match = _SECTION_RE.match(stripped)
        if match:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = match.group(1).strip()
            current_lines = [stripped]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def _query_section_terms(query: str) -> List[str]:
    terms = [t for t in extract_query_terms(query) if t]
    filtered = [t for t in terms if t in _SECTION_HINTS]
    if filtered:
        return filtered
    # fallback: use any short tokens in query
    return [t for t in terms if len(t) <= 6]


def _extract_matching_section(text: str, query: str) -> Optional[str]:
    if not text:
        return None
    sections = _split_sections(text)
    if not sections:
        return None
    terms = _query_section_terms(query)
    if not terms:
        return None
    lowered_terms = [t.lower() for t in terms]
    matched = []
    for title, body in sections:
        if not title:
            continue
        title_l = title.lower()
        if any(term in title_l for term in lowered_terms):
            matched.append(body)
    if not matched:
        return None
    return "\n\n".join(matched[:2])


def _is_card_doc(doc: Dict[str, Any]) -> bool:
    meta = doc.get("metadata") or {}
    source_table = meta.get("source_table")
    if source_table == "card_products":
        return True
    category = (meta.get("category") or "").lower()
    category1 = (meta.get("category1") or "").lower()
    if category in {"credit", "debit"} or category1 in {"credit", "debit"}:
        return True
    card_name = meta.get("card_name")
    if card_name and str(doc.get("id", "")).startswith("CARD-"):
        return True
    return False


def clean_card_docs(docs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for doc in docs:
        if not _is_card_doc(doc):
            cleaned.append(doc)
            continue
        content = doc.get("content") or ""
        trimmed = _dedupe_blocks(content)
        section = _extract_matching_section(trimmed, query)
        if section:
            doc = dict(doc)
            doc["content"] = section
        else:
            doc = dict(doc)
            doc["content"] = trimmed
        cleaned.append(doc)
    return cleaned
