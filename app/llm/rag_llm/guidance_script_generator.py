from __future__ import annotations

from typing import Any, Dict, List, Optional

import json

from app.llm.rag_llm.guide_client import generate_guide_text

MAX_DOCS = 2
MAX_SNIPPET_CHARS = 520
_POLICY_EXCLUDE_TOKENS = ("무기명", "책임지지", "면책", "약관")


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _filter_policy_snippet(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    kept: List[str] = []
    for line in lines:
        if any(token in line for token in _POLICY_EXCLUDE_TOKENS):
            continue
        kept.append(line)
    return "\n".join(kept)


def _build_doc_snippets(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for doc in docs[:MAX_DOCS]:
        title = doc.get("title") or (doc.get("metadata") or {}).get("title") or ""
        content = _truncate(doc.get("content") or "", MAX_SNIPPET_CHARS)
        content = _filter_policy_snippet(content)
        out.append({"title": title, "snippet": content})
    return out


def _build_prompt(query: str, doc_snippets: List[Dict[str, str]], consult_hints: Dict[str, List[str]]) -> str:
    parts: List[str] = []
    if doc_snippets:
        docs_block = []
        for idx, doc in enumerate(doc_snippets, 1):
            docs_block.append(f"- ({idx}) {doc.get('title', '')}: {doc.get('snippet', '')}")
        parts.append("[정책 근거]\n" + "\n".join(docs_block))
    if consult_hints:
        flow = consult_hints.get("flow_steps") or []
        questions = consult_hints.get("common_questions") or []
        if flow:
            parts.append("[상담 흐름]\n" + "\n".join(f"- {item}" for item in flow))
        if questions:
            parts.append("[확인 질문]\n" + "\n".join(f"- {item}" for item in questions))

    question_rule = ""
    if consult_hints.get("common_questions"):
        question_rule = "확인 질문을 1개 포함해. "
    return (
        "다음 정보를 바탕으로 상담사가 그대로 읽을 수 있는 안내 문구를 작성해줘. "
        "대화 요약이 아니라 안내문이어야 하며, 역할 표기(상담사/고객/손님:)는 금지. "
        "존댓말로 2~4문장, 200자 이내, 구체적 절차 요약 중심. "
        "정책 근거는 [정책 근거]만 사용하고, 말투/진행 순서는 [상담 흐름]을 참고해. "
        "첫 문장은 처리/안내 선언으로 시작하고, 마지막 문장은 확인 질문 1개로 끝내. "
        + question_rule
        + "추측/불확실한 정보는 넣지 말고, 개인정보 요청은 하지 마.\n\n"
        f"고객 질문: {query}\n\n"
        + "\n\n".join(parts)
        + "\n\n"
        "JSON 형식으로만 출력: {\"guidanceScript\": \"...\"}"
    )


def generate_guidance_script(
    query: str,
    docs: List[Dict[str, Any]],
    consult_hints: Dict[str, List[str]],
    model: Optional[str] = None,
) -> str:
    _ = model
    if not docs and not consult_hints:
        return ""
    doc_snippets = _build_doc_snippets(docs)
    prompt = _build_prompt(query, doc_snippets, consult_hints)
    system_prompt = "너는 카드 상담 가이드 문구 생성기다."
    try:
        raw = generate_guide_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=180,
            json_output=True,
        )
    except Exception as exc:
        print("[guidance_script] LLM error:", repr(exc))
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    guidance_script = payload.get("guidanceScript")
    if not isinstance(guidance_script, str):
        return ""
    guidance_script = guidance_script.strip()
    if any(token in guidance_script for token in ("상담사:", "손님:", "고객:", "고객님:")):
        return ""
    return guidance_script


__all__ = ["generate_guidance_script"]
