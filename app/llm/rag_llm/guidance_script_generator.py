from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from app.llm.rag_llm.guide_client import generate_guide_text

# =========================
# Config
# =========================
MAX_DOCS = 4
MAX_SNIPPET_CHARS = 360

_ROLE_PREFIXES = ("상담사:", "손님:", "고객:", "고객님:")

_PII_PATTERN = re.compile(
    r"(전화번호|연락처|주소|계좌|카드번호|비밀번호|주민|생년월일|인증번호|CVC|CVV|유효기간)"
)

_PHONE_INTENT_PATTERN = re.compile(
    r"(전화번호|연락처|고객센터|콜센터|대표번호|ars|번호\s*알려|번호\s*좀|전화\s*좀)",
    re.IGNORECASE,
)

_PHONE_PATTERN = re.compile(
    r"\b\d{2,4}\s*[\-–—-]\s*\d{3,4}\s*[\-–—-]\s*\d{4}\b|\b\d{8,11}\b"
)

_URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\S+\.(com|kr)\b)", re.IGNORECASE)

_CONTACT_WORDS_RE = re.compile(
    r"(콜센터|고객센터|ars|홈페이지|웹사이트|사이트|영업점|지점|방문|전화\s*문의|전화로)",
    re.IGNORECASE,
)

_QUESTION_HINT_RE = re.compile(r"(어떻게|어느|어떤|무엇|언제|어디|왜|맞으|원하시|하시겠)")

_DECLARATIVE_Q_RE = re.compile(
    r"(입니다|됩니다|가능합니다|소요됩니다|소요될\s*수\s*있습니다|필요합니다|수\s*있습니다|있습니다|없습니다)\?\s*$"
)

_NOT_QUESTION_TAIL = re.compile(
    r"(참고|주의|유의|권장|바랍니다|부탁드립니다).*(해\s*주세요|하세요|바랍니다)\?\s*$"
)
_ADVICE_Q_RE = re.compile(r"(필요합니다|바랍니다|참고하세요|유의하세요|주의하세요)\?\s*$")

_HEADING_LIKE = re.compile(
    r"^(제\d+조.*|재발급\s*안내|이용방법\s*안내|.*안내|.*방법|.*세부사항|\d+\.?)$"
)
_BULLET_PREFIX = re.compile(r"^\s*[-*•]\s+")
_EMPTY_COLONISH = re.compile(r"^\s*(신청경로|가능시간|세부내용|참조)\s*[:：]\s*\.?\s*$", re.IGNORECASE)

# 질문 앵커 토큰 (질문 다양성의 씨앗)
_QUESTION_ACTION_TOKENS = (
    "분실", "도난", "신고", "정지", "해제", "재발급",
    "대출", "현금서비스", "카드론", "리볼빙", "이자", "수수료", "금리", "한도",
    "결제", "승인", "환불", "취소",
    "등록", "변경", "충전",
    "티머니", "애플페이", "삼성페이", "카카오페이", "K-패스",
)

_TIME_ANCHOR_PATTERNS = (
    re.compile(r"\b\d{1,2}:\d{2}\s*~\s*\d{1,2}:\d{2}\b"),
    re.compile(r"\b24\s*시간\b"),
    re.compile(r"\b365\s*일\b"),
    re.compile(r"\b월\s*\d+\s*회\b"),
    re.compile(r"\b약?\s*\d+\s*주일\b"),
)

# =========================
# Small utils
# =========================
def _truncate_plain(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _build_doc_snippets(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for doc in docs[:MAX_DOCS]:
        title = doc.get("title") or (doc.get("metadata") or {}).get("title") or ""
        content = _truncate_plain(doc.get("content") or "", MAX_SNIPPET_CHARS)
        out.append({"title": title.strip(), "snippet": content.strip()})
    return out


def _strip_role_labels(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    cleaned = cleaned.replace("고객 질문:", "").replace("고객질문:", "")
    cleaned = cleaned.replace("정책 근거:", "").replace("정책근거:", "")
    cleaned = cleaned.replace("[정책 근거]", "").replace("[정책근거]", "")
    cleaned = cleaned.replace("[상담 흐름]", "").replace("[상담흐름]", "")

    first_hit = None
    for prefix in _ROLE_PREFIXES:
        idx = cleaned.find(prefix)
        if idx != -1 and (first_hit is None or idx < first_hit):
            first_hit = idx
    if first_hit is not None:
        cleaned = cleaned[:first_hit]

    cleaned = re.sub(r"\[[^\]]+#\d+\]", "", cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _split_lines(text: str) -> List[str]:
    if not text:
        return []
    lines = [l.strip() for l in text.splitlines()]
    return [l for l in lines if l]


def _sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def _limit_sentences(text: str, max_sentences: int = 1) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[\.!?])\s+|\n+", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) <= max_sentences:
        return text.strip()
    return " ".join(parts[:max_sentences]).strip()


def _is_heading_like(line: str) -> bool:
    if not line:
        return True
    s = _BULLET_PREFIX.sub("", line.strip().replace("_", " ")).strip()
    if len(s) <= 2:
        return True
    if _HEADING_LIKE.match(s):
        return True
    if _EMPTY_COLONISH.match(s):
        return True
    if not re.search(r"(합니다|하세요|돼요|됩니다|가능|불가|필요|권장|이용|신청|신고|등록|변경|재발급|확인)", s):
        if len(s) <= 25:
            return True
    return False


def _normalize_line(line: str) -> str:
    s = (line or "").strip()
    if not s:
        return ""
    s = _BULLET_PREFIX.sub("", s).strip()
    if not s.endswith(("?", ".", "!", "…")):
        s += "."
    s = re.sub(r"\.\.+$", ".", s)
    s = re.sub(r"\?\.+$", "?", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


# =========================
# Dedup / similarity
# =========================
def _norm_for_dedupe(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'“”‘’]", "", s)
    s = s.replace("발급 은행(카드사)", "은행(카드사)")
    s = s.replace("해당 은행(카드사)", "은행(카드사)")
    s = s.replace("해당 카드사", "카드사")
    s = s.strip(" .!?")
    return s


def _dedupe_lines(lines: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for l in lines:
        key = _norm_for_dedupe(l)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(l)
    return out


# =========================
# Anchor & question building
# =========================
def _pick_question_anchor(query: str, doc_snippets: List[Dict[str, str]]) -> str:
    # 1) doc 기반
    for doc in doc_snippets:
        text = f"{doc.get('title','')} {doc.get('snippet','')}"
        for token in _QUESTION_ACTION_TOKENS:
            if token and token in text:
                return token

    # 2) query 기반
    for token in _QUESTION_ACTION_TOKENS:
        if token and token in (query or ""):
            return token

    # 3) 시간 힌트
    for doc in doc_snippets:
        text = f"{doc.get('title','')} {doc.get('snippet','')}"
        for pattern in _TIME_ANCHOR_PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(0)

    return "안내"


def _anchor_to_choice_question(anchor: str) -> str:
    a = (anchor or "").strip()

    if a in {"분실", "도난", "신고", "정지", "해제", "재발급"}:
        return "분실 신고부터 진행하실까요, 아니면 재발급 방법을 먼저 확인해드릴까요?"
    if a in {"대출", "현금서비스", "카드론", "리볼빙", "이자", "수수료", "금리", "한도"}:
        return "신청 방법이 궁금하신가요, 아니면 수수료·이자율·한도를 먼저 확인해드릴까요?"
    if a in {"결제", "승인", "환불", "취소"}:
        return "결제가 안 된 상황 확인부터 할까요, 아니면 취소·환불 처리 경로를 먼저 안내해드릴까요?"
    if a in {"애플페이", "삼성페이", "카카오페이"}:
        return "지금 ‘등록’ 단계에서 막히셨나요, 아니면 ‘결제/인증’ 단계에서 막히셨나요?"
    if a in {"티머니", "K-패스", "충전", "등록", "변경"}:
        return "지금 ‘등록’ 단계에서 막히셨나요, 아니면 ‘유효성 체크/변경’ 단계에서 막히셨나요?"
    if re.search(r"\d", a):
        return f"안내된 시간/기간({a}) 기준으로 진행하시면 될까요?"
    return "어떤 부분을 먼저 확인해드릴까요?"


def _build_summary_line(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "문의 주신 내용을 확인해드릴게요."
    # 너무 일반적인 고정문 대신, 짧은 query는 그대로 반영
    if len(q) <= 18:
        return f"'{q}' 관련해서 안내 도와드릴게요."
    return "문의 주신 내용으로 안내 도와드릴게요."


def _build_question_line(anchor: str) -> str:
    q = _anchor_to_choice_question(anchor)
    if not q.endswith("?"):
        q = q.rstrip(".") + "?"
    return q


# =========================
# Doc sentence picking
# =========================
def _pick_doc_sentence(doc_snippets: List[Dict[str, str]]) -> str:
    weak_re = re.compile(r"(다를\s*수|문의해\s*주세요|참고해\s*주세요|확인해\s*주세요|바랍니다|권장)")
    for doc in doc_snippets:
        snippet = doc.get("snippet", "")
        for s in _sentence_split(snippet):
            s2 = _BULLET_PREFIX.sub("", s).strip()
            if not s2:
                continue
            if _is_heading_like(s2):
                continue
            if weak_re.search(s2):
                continue
            if len(s2) < 10:
                continue
            return s2
    return ""


# =========================
# Sanitization (NO REWRITE)
# =========================
def _sanitize_contacts_sentence(sentence: str, allow_phone: bool) -> str:
    """
    절대 '문장 재작성' 하지 않는다.
    - URL 제거
    - allow_phone=False면 전화번호 제거
    - 채널 단어는 유지하되, 번호/URL만 제거해서 문장 파손을 최소화
    """
    if not sentence:
        return ""
    s = sentence.strip()
    s = _URL_PATTERN.sub("", s).strip()
    if not allow_phone:
        s = _PHONE_PATTERN.sub("", s).strip()

    # 남은 찌꺼기 정리
    s = re.sub(r"\(\s*\)", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = s.strip(" ,·")
    return s


# =========================
# Prompt
# =========================
def _build_prompt(query: str, doc_snippets: List[Dict[str, str]], question_anchor: str) -> str:
    docs_block = []
    for idx, doc in enumerate(doc_snippets, 1):
        title = doc.get("title", "")
        snip = doc.get("snippet", "")
        docs_block.append(f"- ({idx}) {title}: {snip}")

    return (
        "다음 정보를 바탕으로 상담사가 읽을 수 있는 3줄 답변을 작성해줘.\n"
        "요구사항:\n"
        "- 정확히 3줄, 각 줄 1문장\n"
        "- 1줄: 짧은 공감/상황 요약(메타 표현 금지)\n"
        "- 2줄: 문서 근거 핵심 안내(문서 내용을 자연어 1문장으로)\n"
        "- 3줄: 다음 행동을 묻는 자연스러운 질문(반드시 '?'로 끝)\n"
        "- 문서 제목만 읽지 말고 문장으로 풀어쓰기\n"
        "- 3줄 질문은 아래 [질문 앵커]의 단어를 문장에 포함\n"
        "- 불릿/리스트/번호 매기기 금지('-','1.' 등)\n"
        "- 개인정보 요청 금지(주소/계좌/카드번호/생년월일 등)\n"
        "- 전화번호/URL/특정 고객센터 안내는 사용자가 '번호/연락처'를 요청한 경우에만 포함\n\n"
        f"[질문 앵커]\n- {question_anchor}\n\n"
        f"[정책 근거]\n" + "\n".join(docs_block) + "\n\n"
        f"고객 질문: {query}\n"
    )


# =========================
# Main
# =========================
def generate_guidance_script(
    query: str,
    docs: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> str:
    query = (query or "").strip()
    if not query:
        return ""
    if not docs:
        return ""

    phone_intent = bool(_PHONE_INTENT_PATTERN.search(query))
    doc_snippets = _build_doc_snippets(docs)

    anchor = _pick_question_anchor(query, doc_snippets)
    prompt = _build_prompt(query=query, doc_snippets=doc_snippets, question_anchor=anchor)

    try:
        llm_out = generate_guide_text(prompt=prompt, model=model)  # type: ignore
    except TypeError:
        try:
            llm_out = generate_guide_text(prompt=prompt)  # type: ignore
        except TypeError:
            llm_out = generate_guide_text(prompt)  # type: ignore

    text = _strip_role_labels((llm_out or "").strip())

    # ---- hard fallback if empty
    if not text:
        s1 = _normalize_line(_sanitize_contacts_sentence(_build_summary_line(query), allow_phone=phone_intent))
        s2 = _normalize_line(_sanitize_contacts_sentence(_pick_doc_sentence(doc_snippets) or "관련 기준에 따라 안내해드릴게요.", allow_phone=phone_intent))
        q3 = _sanitize_contacts_sentence(_build_question_line(anchor), allow_phone=phone_intent)
        if not q3.endswith("?"):
            q3 = q3.rstrip(".") + "?"
        return f"{s1}\n{s2}\n{q3}".strip()

    # ---- parse lines
    lines = _split_lines(text)

    # bullets/numbering 제거
    if any(_BULLET_PREFIX.match(l) for l in lines) or any(re.match(r"^\s*\d+\.\s*", l) for l in lines):
        cleaned = "\n".join(_BULLET_PREFIX.sub("", l).strip() for l in lines)
        cleaned = re.sub(r"^\s*\d+\.\s*", "", cleaned, flags=re.MULTILINE)
        lines = _split_lines(cleaned)

    if len(lines) < 3:
        lines = _sentence_split(text)[:3]

    s1 = _limit_sentences(lines[0], 1) if len(lines) > 0 else _build_summary_line(query)
    s2 = _limit_sentences(lines[1], 1) if len(lines) > 1 else (_pick_doc_sentence(doc_snippets) or "관련 기준에 따라 안내해드릴게요.")
    s3 = _limit_sentences(lines[2], 1) if len(lines) > 2 else _build_question_line(anchor)

    # heading-like 교체
    if _is_heading_like(s1):
        s1 = _build_summary_line(query)
    if _is_heading_like(s2) or _EMPTY_COLONISH.match(s2):
        s2 = _pick_doc_sentence(doc_snippets) or "관련 기준에 따라 안내해드릴게요."

    # sanitize (NO rewrite)
    s1 = _sanitize_contacts_sentence(s1, allow_phone=phone_intent)
    s2 = _sanitize_contacts_sentence(s2, allow_phone=phone_intent)
    s3 = _sanitize_contacts_sentence(s3, allow_phone=phone_intent)

    # normalize punctuation
    s1 = _normalize_line(s1)
    s2 = _normalize_line(s2)

    # question validation
    fallback_q = _build_question_line(anchor)
    q = (s3 or "").strip()
    q = _BULLET_PREFIX.sub("", q).strip()

    if (
        "?" not in q
        or _NOT_QUESTION_TAIL.search(q)
        or _ADVICE_Q_RE.search(q)
        or (_DECLARATIVE_Q_RE.search(q) and not _QUESTION_HINT_RE.search(q))
        or _PII_PATTERN.search(q)
        or (anchor and anchor not in q)
    ):
        q = fallback_q

    if not q.endswith("?"):
        q = q.rstrip(".") + "?"

    # ---- dedupe lines (핵심)
    merged = _dedupe_lines([s1, s2, q])

    # dedupe로 줄 수가 줄어들면 채우기
    if len(merged) < 3:
        # 부족분은 "문서 1문장"과 "질문" 우선
        candidates = []
        candidates.append(_pick_doc_sentence(doc_snippets) or "관련 기준에 따라 안내해드릴게요.")
        candidates.append(_build_question_line(anchor))
        candidates.append(_build_summary_line(query))
        for c in candidates:
            if len(merged) >= 3:
                break
            if _norm_for_dedupe(c) not in {_norm_for_dedupe(x) for x in merged}:
                if c.endswith("?"):
                    merged.append(c if c.endswith("?") else c.rstrip(".") + "?")
                else:
                    merged.append(_normalize_line(c))

    # 최종 3줄 강제
    merged = merged[:3]
    # 마지막 줄은 무조건 질문
    if not merged[-1].endswith("?"):
        merged[-1] = _build_question_line(anchor)

    return "\n".join(merged).strip()


__all__ = ["generate_guidance_script"]
