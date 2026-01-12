from typing import Any, Dict, List, Optional, Tuple

import json
import os
import time

from app.llm.base import get_openai_client
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.2
MAX_CARD_DOC_CHARS = 450
CARD_RETRY_BACKOFF_SEC = 0.6
CARD_PROMPT_VERSION = os.getenv("RAG_CARD_PROMPT_VERSION", "v2-content-only")


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _base_card(doc: Dict[str, Any]) -> Dict[str, Any]:
    content = doc.get("content") or ""
    meta = doc.get("metadata") or {}
    card_id = meta.get("id") or doc.get("id") or ""
    return {
        "id": str(card_id),
        "title": doc.get("title") or meta.get("title") or "",
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
        "다음은 카드 상담용 문서입니다. 사용자 질문과 문서 내용을 참고해 카드 요약(content)만 생성하세요.\n"
        "반드시 JSON 객체만 반환하세요. 추가 텍스트는 금지합니다.\n"
        f"카드 수는 {doc_count}개이며, 같은 순서로 cards 배열을 채우세요.\n"
        "각 card는 content 필드만 포함하세요. 문서에 없는 내용은 쓰지 마세요.\n\n"
        f"[사용자 질문]\n{query}\n\n"
        "[문서]\n"
        f"{joined}\n\n"
        "[JSON 스키마]\n"
        "{\n"
        "  \"cards\": [\n"
        "    {\"content\": \"1~2문장 요약\"}\n"
        "  ]\n"
        "}\n"
        "\n"
        "[규칙]\n"
        "- 문서에 없는 내용은 절대 추가하지 마세요.\n"
        "- content 외의 필드는 출력하지 마세요.\n"
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


def _is_response_format_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "response_format" in message or "json_object" in message:
        return True
    if isinstance(exc, (BadRequestError, UnprocessableEntityError)):
        return True
    return False


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return bool(status and status >= 500)
    return False


def generate_detail_cards(
    query: str,
    docs: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_llm_cards: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    if not docs:
        return [], ""
    base_cards = [_base_card(doc) for doc in docs]
    llm_count = len(docs) if max_llm_cards is None else max(0, min(max_llm_cards, len(docs)))
    if llm_count == 0:
        return base_cards, ""
    docs_for_llm = docs[:llm_count]
    prompt = _build_card_prompt(query, docs_for_llm)
    client = get_openai_client()
    messages = [
        {
            "role": "system",
            "content": (
                "너는 카드 상담 업무용 카드 생성기다. "
                "문서 내용과 사용자 질문을 기반으로 카드 정보를 생성한다."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        if _is_response_format_error(exc):
            print("[cards] response_format failed:", repr(exc))
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        elif _is_transient_error(exc):
            print("[cards] transient error, retrying:", repr(exc))
            time.sleep(CARD_RETRY_BACKOFF_SEC)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        else:
            raise
    raw = resp.choices[0].message.content or ""
    payload = _parse_cards_payload(raw)
    if not payload:
        return base_cards, ""
    parsed = payload.get("cards") if isinstance(payload, dict) else None
    guidance_script = payload.get("guidanceScript") if isinstance(payload, dict) else ""
    if not isinstance(parsed, list):
        return base_cards, guidance_script or ""

    out = list(base_cards)
    for idx, base in enumerate(base_cards[:llm_count]):
        generated = parsed[idx] if idx < len(parsed) and isinstance(parsed[idx], dict) else {}
        merged = {**base, **generated}
        merged["id"] = base["id"]
        merged["title"] = base["title"]
        merged["detailContent"] = base["detailContent"]
        merged["relevanceScore"] = base["relevanceScore"]
        out[idx] = merged
    return out, guidance_script or ""
