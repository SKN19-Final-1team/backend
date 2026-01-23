from typing import Any, Dict, List, Optional, Tuple

import json
import os
import re
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
# 최적화: LLM 입력 문자 수 제한 (280자로 유지, 품질 확보)
DOC_SNIPPET_CHARS = int(os.getenv("RAG_DOC_SUMMARY_SNIPPET_CHARS", "280"))
DOC_SNIPPET_FALLBACK_CHARS = int(os.getenv("RAG_DOC_SUMMARY_FALLBACK_CHARS", "420"))
MAX_CARD_CONTEXT_CHARS = int(os.getenv("RAG_CARD_CONTEXT_CHARS", "850"))
RULE_SUMMARY_CHARS = int(os.getenv("RAG_RULE_SUMMARY_CHARS", "220"))
DOC_SNIPPET_MIN_TERM_LEN = int(os.getenv("RAG_DOC_SUMMARY_MIN_TERM_LEN", "2"))
DOC_SUMMARY_PROMPT_VERSION = os.getenv("RAG_DOC_SUMMARY_PROMPT_VERSION", "v1")
MAX_CARD_DOC_CHARS = DOC_SNIPPET_CHARS
CARD_RETRY_BACKOFF_SEC = 0.6
CARD_PROMPT_VERSION = os.getenv("RAG_CARD_PROMPT_VERSION", "v2-content-only")
DOC_SUMMARY_CACHE_ENABLED = False  # doc summary cache removed; keep flag for compatibility

def build_doc_summary_cache_key(*args, **kwargs):
    return None

def doc_summary_cache_get(key):
    return None

def doc_summary_cache_set(key, summary):
    return None

_TERM_RE = re.compile(r"[A-Za-z0-9가-힣]+")
_SECTION_CUT_PATTERNS = [
    re.compile(r"^#\\s*금융소비자\\s*보호제도\\s*안내", re.I),
    re.compile(r"^#\\s*기타\\s*안내", re.I),
    re.compile(r"^#\\s*해외이용\\s*확인사항", re.I),
    re.compile(r"^#\\s*연회비\\s*반환\\s*기준", re.I),
]
_CONTACT_LINE_RE = re.compile(r"(고객센터|콜센터|센터|문의|연락처)\\s*[:：]?\\s*\\d{2,4}-\\d{3,4}-\\d{4}")
_PHONE_RE = re.compile(r"\\b\\d{2,4}-\\d{3,4}-\\d{4}\\b")


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


def _unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_compact(text: str) -> str:
    return re.sub(r"[\\s\\-_/]+", "", (text or "").lower())


def _extract_query_terms(query: str) -> List[str]:
    raw_terms = _TERM_RE.findall(query or "")
    terms = [term.lower() for term in raw_terms if len(term) >= DOC_SNIPPET_MIN_TERM_LEN]
    return _unique_in_order(terms)


def _extract_relevant_snippets(query: str, content: str, limit: int) -> str:
    if not content:
        return ""
    # Remove noisy tail sections and contact-heavy lines before extraction.
    cleaned_lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        if any(pattern.search(stripped) for pattern in _SECTION_CUT_PATTERNS):
            break
        if _CONTACT_LINE_RE.search(stripped) or _PHONE_RE.search(stripped):
            continue
        cleaned_lines.append(line)
    content = "\n".join(cleaned_lines).strip()
    if not content:
        return ""
    terms = _extract_query_terms(query)
    if not terms:
        return _truncate(content, limit)
    paragraphs = [p.strip() for p in re.split(r"\\n{2,}", content) if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in re.split(r"\\n+", content) if p.strip()]
    if not paragraphs:
        return _truncate(content, limit)
    matched_indexes: List[int] = []
    for idx, paragraph in enumerate(paragraphs):
        compact = _normalize_compact(paragraph)
        if any(term in compact for term in terms):
            matched_indexes.append(idx)
    if not matched_indexes:
        return _truncate(content, DOC_SNIPPET_FALLBACK_CHARS)
    selected = set()
    for idx in matched_indexes:
        selected.add(idx)
        if idx - 1 >= 0:
            selected.add(idx - 1)
        if idx + 1 < len(paragraphs):
            selected.add(idx + 1)
    picked: List[str] = []
    total = 0
    for idx in sorted(selected):
        paragraph = paragraphs[idx]
        if not paragraph:
            continue
        if picked and total + len(paragraph) + 1 > limit:
            break
        picked.append(paragraph)
        total += len(paragraph) + 1
        if total >= limit:
            break
    return _truncate("\\n".join(picked), limit)


def _build_rule_summary(query: str, content: str) -> str:
    if not content:
        return ""
    # 1~2문장, 160자 이내, 불릿/인사/문의/전화 등 제거
    summary = _extract_relevant_snippets(query, content, 160)
    # 불릿/인사/문의/전화 패턴 제거
    summary = re.sub(r"^[\-•·\*\d\s]+", "", summary, flags=re.MULTILINE)
    summary = re.sub(r"(문의|연락처|전화|고객센터|콜센터)[^\n]*", "", summary)
    summary = re.sub(r"(^|\n)[가-힣]{2,5}님[\s,]*", "", summary)
    summary = summary.strip()
    return summary[:160]


def build_rule_cards(query: str, docs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """LLM을 거치지 않고 룰 기반 요약만으로 카드 생성."""
    if not docs:
        return [], ""
    cards = [_base_card(doc) for doc in docs]
    for idx, doc in enumerate(docs):
        cards[idx]["content"] = _build_rule_summary(query, doc.get("content") or "")
    return cards, ""


def _build_card_prompt(query: str, docs: List[Dict[str, Any]]) -> str:
    parts = []
    remaining = MAX_CARD_CONTEXT_CHARS
    for idx, doc in enumerate(docs, 1):
        if remaining <= 0:
            break
        per_limit = min(DOC_SNIPPET_CHARS, max(80, remaining // max(1, len(docs) - idx + 1)))
        content = _extract_relevant_snippets(query, doc.get("content") or "", per_limit)
        remaining -= len(content)
        doc_id = doc.get("id") or ""
        title = doc.get("title") or ""
        parts.append(
            f"[{idx}] id={doc_id}\n"
            f"title={title}\n"
            f"content={content}"
        )
    joined = "\n\n".join(parts) if parts else "문서 없음"
    doc_count = len(docs)
    return (
        "다음은 카드 상담용 문서입니다. 사용자 질문과 문서 내용을 참고해 카드 요약(content)만 생성하세요.\n"
        "반드시 JSON 객체만 반환하세요. 추가 텍스트는 금지합니다.\n"
        f"카드 수는 {doc_count}개이며, 같은 순서로 cards 배열을 채우세요.\n"
        "각 card는 content 필드만 포함하세요. 문서에 없는 내용은 쓰지 마세요.\n"
        "- 각 content는 1~2문장, 160자 이내\n"
        "- 불릿/인사/마크다운 금지, 핵심 사실만 요약\n\n"
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
    # LLM 호출: 1회 실패 시 fallback만 적용, 불필요한 루프 제거
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
    # pins 기반 확정 답변은 LLM 생략
    try:
        from app.rag.policy.policy_pins import POLICY_PINS
        PIN_IDS = {doc_id for pin in POLICY_PINS for doc_id in pin.get("doc_ids", [])}
    except Exception:
        PIN_IDS = set()
    all_pin_docs = True
    for doc in docs:
        doc_id = str((doc.get("metadata") or {}).get("id") or doc.get("id") or "")
        if doc_id not in PIN_IDS:
            all_pin_docs = False
            break
    if all_pin_docs and PIN_IDS:
        return base_cards, ""
    # 최적화: LLM 입력을 2개 문서로 제한
    max_llm_cards = max_llm_cards or 2
    llm_count = len(docs) if max_llm_cards is None else max(0, min(max_llm_cards, len(docs)))
    if llm_count == 0:
        return base_cards, ""

    doc_id_to_index: Dict[str, int] = {}
    for idx, doc in enumerate(docs):
        doc_id = str((doc.get("metadata") or {}).get("id") or doc.get("id") or "")
        if doc_id:
            doc_id_to_index[doc_id] = idx

    cache_hits = {"redis": 0, "mem": 0}
    cache_miss = 0
    cached_doc_ids = set()
    if DOC_SUMMARY_CACHE_ENABLED:
        for doc in docs:
            doc_id = str((doc.get("metadata") or {}).get("id") or doc.get("id") or "")
            table = str(doc.get("table") or (doc.get("metadata") or {}).get("source_table") or "")
            key = build_doc_summary_cache_key(table, doc_id, model, DOC_SUMMARY_PROMPT_VERSION)
            cached = doc_summary_cache_get(key)
            if cached:
                summary, backend = cached
                idx = doc_id_to_index.get(doc_id)
                if idx is not None and summary:
                    base_cards[idx]["content"] = summary
                    cached_doc_ids.add(doc_id)
                    cache_hits[backend] = cache_hits.get(backend, 0) + 1
            else:
                cache_miss += 1

    # Fill non-cached cards with rule-based summaries so non-LLM docs stay relevant.
    for idx, doc in enumerate(docs):
        doc_id = str((doc.get("metadata") or {}).get("id") or doc.get("id") or "")
        if doc_id in cached_doc_ids:
            continue
        base_cards[idx]["content"] = _build_rule_summary(query, doc.get("content") or "")


    docs_for_llm: List[Dict[str, Any]] = []
    doc_ids_for_llm: List[str] = []
    # LLM을 반드시 태우도록: 최소 1개는 LLM에 전달
    docs_for_llm_candidates = [doc for doc in docs if str((doc.get("metadata") or {}).get("id") or doc.get("id") or "") not in cached_doc_ids]
    if not docs_for_llm_candidates:
        docs_for_llm_candidates = docs[:1] if docs else []
    for doc in docs_for_llm_candidates[:max(1, llm_count)]:
        doc_id = str((doc.get("metadata") or {}).get("id") or doc.get("id") or "")
        docs_for_llm.append(doc)
        doc_ids_for_llm.append(doc_id)

    if DOC_SUMMARY_CACHE_ENABLED:
        total_hits = sum(cache_hits.values())
        if total_hits or cache_miss:
            hit_label = ",".join(f"{k}:{v}" for k, v in cache_hits.items() if v)
            hit_label = hit_label or "0"
            print(f"[cards] doc_summary_cache hit({hit_label}) miss={cache_miss}")

    prompt = _build_card_prompt(query, docs_for_llm)
    # 로깅: LLM 입력 길이/문서별 길이
    doc_ids = []
    doc_chars = []
    ctx_total = 0
    for doc in docs_for_llm:
        doc_id = str((doc.get("metadata") or {}).get("id") or doc.get("id") or "")
        snippet = _extract_relevant_snippets(query, doc.get("content") or "", DOC_SNIPPET_CHARS)
        doc_ids.append(doc_id)
        doc_chars.append(len(snippet))
        ctx_total += len(snippet)
    print(
        f"[cards] llm_input_chars={len(prompt)} ctx_chars={ctx_total} "
        f"doc_ids={doc_ids} doc_chars={doc_chars}"
    )
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
            max_tokens=350,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        # 단 1회만 재시도, 실패 시 fallback
        if _is_response_format_error(exc) or _is_transient_error(exc):
            print("[cards] LLM error, fallback once:", repr(exc))
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=350,
                )
            except Exception as exc2:
                print("[cards] LLM fallback failed:", repr(exc2))
                return base_cards, ""
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
    for idx, doc_id in enumerate(doc_ids_for_llm):
        generated = parsed[idx] if idx < len(parsed) and isinstance(parsed[idx], dict) else {}
        card_index = doc_id_to_index.get(doc_id)
        if card_index is None:
            continue
        base = out[card_index]
        merged = {**base, **generated}
        merged["id"] = base["id"]
        merged["title"] = base["title"]
        merged["detailContent"] = base["detailContent"]
        merged["relevanceScore"] = base["relevanceScore"]
        out[card_index] = merged
        summary = str(merged.get("content") or "")
        table = str(docs[card_index].get("table") or (docs[card_index].get("metadata") or {}).get("source_table") or "")
        key = build_doc_summary_cache_key(table, doc_id, model, DOC_SUMMARY_PROMPT_VERSION)
        doc_summary_cache_set(key, summary)
    return out, guidance_script or ""
