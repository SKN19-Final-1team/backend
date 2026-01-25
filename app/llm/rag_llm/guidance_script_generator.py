from __future__ import annotations

from typing import Any, Dict, List, Optional

import json

from app.llm.base import get_openai_client

DEFAULT_MODEL = "gpt-4.1-mini"
MAX_DOCS = 2
MAX_SNIPPET_CHARS = 900


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _build_prompt(query: str, consult_docs: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for idx, doc in enumerate(consult_docs[:MAX_DOCS], 1):
        title = doc.get("title") or ""
        category = (doc.get("metadata") or {}).get("category") or ""
        content = _truncate(doc.get("content") or "", MAX_SNIPPET_CHARS)
        parts.append(
            f"[사례 {idx}]\n"
            f"제목: {title}\n"
            f"카테고리: {category}\n"
            f"내용: {content}"
        )
    return (
        "다음 상담 사례를 참고해 상담사가 그대로 읽을 수 있는 안내 문구를 작성해줘. "
        "대화 요약이 아니라 안내문이어야 하며, 역할 표기(상담사/고객/손님:)는 금지. "
        "존댓말로 1~2문장, 140자 이내, 구체적 절차 요약 중심. "
        "추측/불확실한 정보는 넣지 말고, 개인정보 요청은 하지 마.\n\n"
        f"고객 질문: {query}\n\n"
        + "\n\n".join(parts)
        + "\n\n"
        "JSON 형식으로만 출력: {\"guidanceScript\": \"...\"}"
    )


def generate_guidance_script_from_consult(
    query: str,
    consult_docs: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> str:
    if not consult_docs:
        return ""
    prompt = _build_prompt(query, consult_docs)
    client = get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=model or DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "너는 카드 상담 가이드 문구 생성기다.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        print("[guidance_script] LLM error:", repr(exc))
        return ""

    raw = resp.choices[0].message.content or ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    guidance_script = payload.get("guidanceScript")
    if not isinstance(guidance_script, str):
        return ""
    return guidance_script.strip()


__all__ = ["generate_guidance_script_from_consult"]
