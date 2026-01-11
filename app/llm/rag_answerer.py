from typing import Any, Dict, List, Optional, Tuple

import json
import re

from app.llm.base import get_openai_client

DEFAULT_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_CONTEXT_CHARS = 1600
MIN_BODY_CHARS = 60
DEFAULT_SOURCES_HEADER = "[출처]"
MAX_CARD_DOC_CHARS = 1200
_WS_RE = re.compile(r"\s+")
_TIME_PATTERNS = [
    re.compile(r"\b\d{1,2}:\d{2}\s*[~-]\s*\d{1,2}:\d{2}\b"),
    re.compile(r"\b\d+\s*[~-]\s*\d+\s*영업일\b"),
    re.compile(r"\b\d+\s*영업일\b"),
    re.compile(r"\b약\s*\d+\s*(분|시간)\b"),
    re.compile(r"\b\d+\s*(분|시간)\b"),
]
_SYSTEM_PATH_RE = re.compile(r"[가-힣A-Za-z0-9\\s]+(?:>\\s*[가-힣A-Za-z0-9\\s]+){2,}")
_EXCEPTION_START = ("단,", "다만", "예외", "주의")
_EXCEPTION_PATTERNS = (
    re.compile(r"^단,\\s*.+"),
    re.compile(r"^다만\\s*.+"),
    re.compile(r"^예외\\s*.*"),
    re.compile(r"^주의\\s*.*"),
    re.compile(r".+\\b제외\\b.+"),
    re.compile(r".+\\b불가\\b.+"),
)


def build_context(docs: List[Dict[str, Any]], max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> str:
    if not docs:
        return ""
    parts = []
    total = 0
    for idx, doc in enumerate(docs, 1):
        title = doc.get("title") or "(no title)"
        snippet = doc.get("content", "")
        block = f"[{idx}] {title}\n{snippet}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def _extract_body(text: str) -> str:
    if not text:
        return ""
    if "내용:" in text:
        return text.split("내용:", 1)[1].strip()
    if text.startswith("제목:"):
        parts = text.split("\n", 1)
        return parts[1].strip() if len(parts) > 1 else ""
    return text.strip()


def format_sources(
    docs: List[Dict[str, Any]],
    header: str = DEFAULT_SOURCES_HEADER,
) -> str:
    if not docs:
        return ""
    lines = []
    for idx, doc in enumerate(docs, 1):
        title = doc.get("title") or "(no title)"
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        source_id = meta.get("id") or doc.get("id")
        if source_id:
            lines.append(f"- [{idx}] {title} ({source_id})")
        else:
            lines.append(f"- [{idx}] {title}")
    return header + "\n" + "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _base_card(doc: Dict[str, Any]) -> Dict[str, Any]:
    content = doc.get("content") or ""
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    card_id = meta.get("id") or doc.get("id") or ""
    return {
        "id": str(card_id),
        "title": doc.get("title") or "",
        "keywords": [],
        "content": _truncate(content, 140),
        "systemPath": "",
        "requiredChecks": [],
        "exceptions": [],
        "regulation": "",
        "detailContent": content,
        "time": "",
        "note": "",
        "relevanceScore": float(doc.get("score") or 0.0),
    }


def _normalize_for_match(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


def _in_doc(text: str, content: str) -> bool:
    if not text or not content:
        return False
    return _normalize_for_match(text) in _normalize_for_match(content)


def _filter_list_items(items: Any, content: str) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value:
            continue
        if _in_doc(value, content):
            out.append(item)
    return out


def _extract_time(content: str) -> str:
    if not content:
        return ""
    for pattern in _TIME_PATTERNS:
        match = pattern.search(content)
        if match:
            return match.group(0)
    return ""


def _extract_system_path(content: str) -> str:
    if not content:
        return ""
    match = _SYSTEM_PATH_RE.search(content)
    if not match:
        return ""
    return match.group(0).strip()


def _extract_exceptions(content: str) -> List[str]:
    if not content:
        return []
    lines: List[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if len(line) > 180:
            continue
        if " - " in line:
            parts = [part.strip() for part in line.split(" - ") if part.strip()]
        else:
            parts = [line]
        for part in parts:
            if len(part) > 160:
                continue
            if part.startswith(_EXCEPTION_START):
                lines.append(part)
                continue
            if any(pattern.search(part) for pattern in _EXCEPTION_PATTERNS):
                lines.append(part)
    seen = set()
    out = []
    for item in lines:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _build_card_prompt(query: str, docs: List[Dict[str, Any]]) -> str:
    parts = []
    for idx, doc in enumerate(docs, 1):
        content = doc.get("content") or ""
        doc_id = doc.get("id") or ""
        title = doc.get("title") or ""
        parts.append(
            f"[{idx}] id={doc_id}\n"
            f"title={title}\n"
            f"content={_truncate(content, MAX_CARD_DOC_CHARS)}"
        )
    joined = "\n\n".join(parts) if parts else "문서 없음"
    doc_count = len(docs)
    return (
        "다음은 카드 상담용 문서입니다. 사용자 질문과 문서 내용을 참고해 카드 상세 정보를 생성하세요.\n"
        "정확한 정보가 없으면 합리적인 상담 시나리오를 간단히 구성해도 됩니다.\n"
        "반드시 JSON 객체만 반환하세요. 추가 텍스트는 금지합니다.\n"
        f"카드 수는 {doc_count}개이며, 같은 순서로 cards 배열을 채우세요.\n"
        "모든 카드에 모든 필드를 채우되, 알 수 없으면 빈 문자열/빈 배열로 채우세요.\n\n"
        f"[사용자 질문]\n{query}\n\n"
        "[문서]\n"
        f"{joined}\n\n"
        "[JSON 스키마]\n"
        "{\n"
        "  \"cards\": [\n"
        "    {\n"
        "      \"id\": \"문서 id 그대로\",\n"
        "      \"title\": \"문서 title 그대로\",\n"
        "      \"keywords\": [\"#키워드1\", \"#키워드2\"],\n"
        "      \"content\": \"사용자에게 보여줄 1~2문장 요약\",\n"
        "      \"systemPath\": \"업무 경로(없으면 빈 문자열)\",\n"
        "      \"requiredChecks\": [\"필수 확인사항\"],\n"
        "      \"exceptions\": [\"예외 사항\"],\n"
        "      \"regulation\": \"관련 규정/약관(없으면 빈 문자열)\",\n"
        "      \"time\": \"처리 시간(없으면 빈 문자열)\",\n"
        "      \"note\": \"추가 메모(없으면 빈 문자열)\"\n"
        "    }\n"
        "  ],\n"
        "  \"guidanceScript\": \"상담원이 읽을 간단 스크립트(없으면 빈 문자열)\"\n"
        "}\n"
        "\n"
        "[추가 규칙]\n"
        "- guidanceScript는 문서에서 그대로 발췌한 문장만 사용하세요.\n"
        "- 문서에 없는 내용은 절대 추가하지 마세요.\n"
    )

def _parse_cards_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def generate_detail_cards(
    query: str,
    docs: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tuple[List[Dict[str, Any]], str]:
    if not docs:
        return [], ""
    base_cards = [_base_card(doc) for doc in docs]
    prompt = _build_card_prompt(query, docs)
    client = get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 카드 상담 업무용 카드 생성기다. "
                        "문서 내용과 사용자 질문을 기반으로 카드 정보를 생성한다."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 카드 상담 업무용 카드 생성기다. "
                        "문서 내용과 사용자 질문을 기반으로 카드 정보를 생성한다."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
    raw = resp.choices[0].message.content or ""
    payload = _parse_cards_payload(raw)
    if not payload:
        return base_cards, ""
    parsed = payload.get("cards") if isinstance(payload, dict) else None
    guidance_script = payload.get("guidanceScript") if isinstance(payload, dict) else ""
    if not isinstance(parsed, list):
        return base_cards, guidance_script or ""

    out: List[Dict[str, Any]] = []
    for idx, base in enumerate(base_cards):
        generated = parsed[idx] if idx < len(parsed) and isinstance(parsed[idx], dict) else {}
        merged = {**base, **generated}
        merged["id"] = base["id"]
        merged["title"] = base["title"]
        merged["detailContent"] = base["detailContent"]
        merged["relevanceScore"] = base["relevanceScore"]
        content = base.get("detailContent") or ""
        merged["requiredChecks"] = _filter_list_items(merged.get("requiredChecks"), content)
        merged["exceptions"] = _filter_list_items(merged.get("exceptions"), content)
        for field in ("note", "time", "regulation", "systemPath"):
            value = merged.get(field)
            if not isinstance(value, str) or not _in_doc(value, content):
                merged[field] = ""
        if not merged.get("time"):
            extracted_time = _extract_time(content)
            if extracted_time:
                merged["time"] = extracted_time
        if not merged.get("systemPath"):
            extracted_path = _extract_system_path(content)
            if extracted_path:
                merged["systemPath"] = extracted_path
        if not merged.get("exceptions"):
            extracted_exceptions = _extract_exceptions(content)
            if extracted_exceptions:
                merged["exceptions"] = extracted_exceptions
        out.append(merged)
    return out, guidance_script or ""


def generate_answer(
    query: str,
    docs: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> Dict[str, Any]:
    context = build_context(docs, max_chars=max_context_chars)
    body_chars = sum(len(_extract_body(doc.get("content", ""))) for doc in docs)
    system = (
    "당신은 카드 고객지원 도우미입니다.\n"
    "아래에 제공된 [출처 문서]만을 근거로 답변해야 합니다.\n"
    "출처에 없는 정보, 절차, 조건, 예시는 절대 만들어내지 마세요.\n"
    "문서 내용만 요약하거나 그대로 설명하세요.\n"
    "문서에 질문에 대한 정보가 없거나 불충분하면 "
    "'제공된 문서 기준으로는 확인할 수 없습니다'라고 명확히 말하세요.\n"
    "모든 답변은 한국어로 작성하세요."
)
    user = (
    f"사용자 질문:\n{query}\n\n"
    f"[출처 문서] (번호로 인용):\n"
    f"{context if context else '제공된 문서 없음'}\n\n"
    "[답변 작성 규칙]\n"
    "- 출처 문서에 명시된 내용만 사용하세요.\n"
    "- 문서에 절차나 단계가 명확히 나와 있을 때만 번호(1,2,3)를 사용하세요.\n"
    "- 문서에 없는 절차, 조건, 기준을 새로 만들어 설명하지 마세요.\n"
    "- 문장을 인용하거나 요약할 때는 문장 끝에 [번호] 형식으로 출처를 표시하세요.\n"
    "- 간결하고 사실 위주로 답변하세요."
)


    if not context or body_chars < MIN_BODY_CHARS:
        answer = (
            "현재 적재된 문서에 질문을 해결할 만큼의 상세 내용이 없습니다. "
            "관련 안내가 누락되어 있어 구체적인 답변을 드리기 어렵습니다."
        )
    else:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )

        answer = resp.choices[0].message.content
    sources = format_sources(docs)
    if sources:
        answer = f"{answer}\n\n{sources}"

    return {
        "answer": answer,
        "sources": sources,
        "model": model,
        "context_chars": len(context),
        "doc_count": len(docs),
    }
