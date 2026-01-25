from typing import Any, Dict, List

from app.llm.base import get_openai_client

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_CONTEXT_CHARS = 1600
MIN_BODY_CHARS = 60
DEFAULT_SOURCES_HEADER = "[출처]"


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
        meta = doc.get("metadata") or {}
        source_id = meta.get("id") or doc.get("id")
        if source_id:
            lines.append(f"- [{idx}] {title} ({source_id})")
        else:
            lines.append(f"- [{idx}] {title}")
    return header + "\n" + "\n".join(lines)


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
