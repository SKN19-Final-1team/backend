from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

# =========================================================
# Simple, stable guidance script generator (3 lines)
# - Query-first intent scoring
# - Pick ONE best grounding sentence by anchor relevance
# - No A/B "또는" questions (single question only)
# - Strong noise filtering (promo/title/legal fragments)
# - Phone numbers kept only if user asked for them
# =========================================================

MAX_DOCS = 3
MAX_SNIPPET_CHARS = 420

# -------------------------
# Safety / Redaction
# -------------------------
PII_PATTERN = re.compile(
    r"(전화번호|연락처|주소|계좌|카드번호|비밀번호|주민|생년월일|인증번호|CVC|CVV|유효기간)",
    re.IGNORECASE,
)
PHONE_INTENT_PATTERN = re.compile(
    r"(전화번호|연락처|고객센터|콜센터|대표번호|ars|번호\s*알려|번호\s*좀|전화\s*좀)",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(
    r"\b\d{2,4}\s*[\-–—-]\s*\d{3,4}\s*[\-–—-]\s*\d{4}\b|\b\d{8,11}\b"
)
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\S+\.(com|kr)\b)", re.IGNORECASE)

# -------------------------
# Sentence tools
# -------------------------
SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+|\n+")
LEADING_ENUM = re.compile(r"^\s*([0-9]+[\)\.]|[①②③④⑤⑥⑦⑧⑨]|[-•]\s+)\s*")

TITLEISH = re.compile(r"^(제\d+조.*|.*안내$|.*방법$|.*세부사항$)$")
LEGALISH = re.compile(r"(약관|조문|규정|제\d+조|제\d+항|제\d+호)", re.IGNORECASE)

# “홍보/헤더/쓸모없는 상담 멘트” 강하게 컷
NOISE = re.compile(
    r"(문의하신|확인해볼게요|안내해\s*드릴게요|안내\s*도와드릴게요|"
    r"이렇게\s.*(하세요|해요)|예방하세요|피해\s*예방|참고해\s*주세요|문의해\s*주세요)",
    re.IGNORECASE,
)

# 질문형/지시형 문장(근거로 쓰기 별로)
BAD_TAIL = re.compile(r"(하셨나요|하셨습니까|되었나요|되셨나요)\?$", re.IGNORECASE)

# -------------------------
# Anchor / Intent
# -------------------------
# 각 앵커는 query에서 점수화 (doc는 보조)
ANCHOR_DEFS: list[dict[str, Any]] = [
    {
        "label": "결제 오류",
        "query": [r"(결제|승인|오류|에러|안\s*돼|실패|거절)"],
        "kw": ["결제", "승인", "오류", "에러", "실패", "거절"],
        "summary": "결제가 잘 안 되어 불편하셨겠어요.",
        "question": "오류가 나온 시점이 ‘결제 시도’인지 ‘인증 단계’인지 알려주실 수 있을까요?",
    },
    {
        "label": "취소/환불",
        "query": [r"(환불|취소|반품|승인취소|매입취소)"],
        "kw": ["취소", "환불", "반품", "승인", "매입"],
        "summary": "취소/환불 관련으로 문의 주셨군요.",
        "question": "취소/환불은 ‘승인 취소’인지 ‘매입 이후 환불’인지 어느 쪽인가요?",
    },
    {
        "label": "분실 신고",
        "query": [r"(분실|잃어버|도난)"],
        "kw": ["분실", "도난", "신고", "정지", "사용", "차단"],
        "summary": "카드 분실로 문의 주셨군요.",
        "question": "지금 바로 분실 신고(사용 정지)부터 진행해드릴까요?",
    },
    {
        "label": "재발급",
        "query": [r"(재발급|재발행|다시\s*발급)"],
        "kw": ["재발급", "재발행", "수령", "소요", "신청"],
        "summary": "재발급 관련으로 문의 주셨군요.",
        "question": "재발급 신청 방법부터 안내드릴까요?",
    },
    {
        "label": "등록",
        "query": [r"(등록|추가|연동|연결)"],
        "kw": ["등록", "추가", "연동", "연결", "인증", "유효성"],
        "summary": "등록 과정에서 불편이 있으셨겠어요.",
        "question": "지금 ‘등록’ 단계에서 막히셨나요, ‘인증/유효성 체크’ 단계에서 막히셨나요?",
    },
    {
        "label": "카드 변경",
        "query": [r"(변경|교체|갱신|재등록)"],
        "kw": ["변경", "교체", "갱신", "재등록"],
        "summary": "카드 변경 관련 문의 주셨군요.",
        "question": "카드 변경은 앱에서 진행 중이신가요, 홈페이지에서 진행 중이신가요?",
    },
    {
        "label": "대출 이용",
        "query": [r"(현금서비스|단기카드대출|카드대출|카드론|대출)"],
        "kw": ["대출", "현금서비스", "신청", "이체", "ARS", "ATM", "시간"],
        "summary": "대출 이용 관련 문의 주셨군요.",
        "question": "신청은 앱/홈페이지/ARS 중 어떤 방법으로 진행하실까요?",
    },
    {
        "label": "리볼빙",
        "query": [r"(리볼빙|일부결제금액이월)"],
        "kw": ["리볼빙", "일부결제", "이월", "약정", "수수료"],
        "summary": "리볼빙 관련 문의 주셨군요.",
        "question": "리볼빙 약정 신청 경로부터 안내드릴까요?",
    },
    {
        "label": "수수료/이자/한도",
        "query": [r"(수수료|이자율|이자|연회비|한도)"],
        "kw": ["수수료", "이자", "이자율", "연회비", "한도"],
        "summary": "수수료/이자/한도 관련 문의 주셨군요.",
        "question": "수수료·이자율·한도 중 어떤 항목이 가장 궁금하신가요?",
    },
    {
        "label": "사용 가능 여부",
        "query": [r"(사용처|어디서|가능|불가|이용처)"],
        "kw": ["사용", "이용", "가능", "불가", "사용처", "이용처"],
        "summary": "사용 가능 여부를 확인해드릴게요.",
        "question": "어느 사용처(가맹점/기관/해외/교통 등)에서 사용 가능한지 확인해드릴까요?",
    },
]

DEFAULT_ANCHOR = {
    "label": "진행 절차",
    "kw": [],
    "summary": "문의하신 내용을 확인해드릴게요.",
    "question": "어떤 부분부터 도와드릴까요?",
}

# -------------------------
# Core helpers
# -------------------------
def _truncate(text: str, limit: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= limit else t[:limit].rstrip()

def _build_doc_snippets(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for doc in docs[:MAX_DOCS]:
        title = (doc.get("title") or (doc.get("metadata") or {}).get("title") or "").strip()
        snippet = _truncate(doc.get("content") or "", MAX_SNIPPET_CHARS)
        out.append({"title": title, "snippet": snippet})
    return out

def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT.split(text or "") if p and p.strip()]
    out: List[str] = []
    for p in parts:
        p = LEADING_ENUM.sub("", p).strip()
        if len(p) >= 8:
            out.append(p)
    return out

def _normalize_sentence(s: str, is_question: bool = False) -> str:
    s = (s or "").strip()
    s = LEADING_ENUM.sub("", s).strip()
    if not s:
        return s
    if is_question:
        s = s.rstrip(".")
        return s if s.endswith("?") else s + "?"
    if not s.endswith((".", "!", "?", "…")):
        s += "."
    return s

def _redact(text: str, allow_phone: bool) -> str:
    t = (text or "").strip()
    t = URL_PATTERN.sub("", t)
    if not allow_phone:
        t = PHONE_PATTERN.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"\(\s*\)", "", t).strip()
    return t

def _is_bad_grounding(s: str) -> bool:
    if not s:
        return True
    if TITLEISH.match(s):
        return True
    if BAD_TAIL.search(s):
        return True
    if NOISE.search(s):
        return True
    # 너무 짧은 조문/약관 조각은 근거로 쓰기 별로
    if LEGALISH.search(s) and len(s) < 45:
        return True
    if PII_PATTERN.search(s):
        return True
    return False

def _score_anchor(query: str, doc_snips: List[Dict[str, str]], anchor_def: dict[str, Any]) -> int:
    """
    query hit: +10 per pattern
    doc hit: +2 per keyword occurrence (light)
    """
    score = 0
    q = query or ""
    blob = " ".join([q] + [f"{d.get('title','')} {d.get('snippet','')}" for d in doc_snips])

    for pat in anchor_def.get("query", []):
        if re.search(pat, q, flags=re.IGNORECASE):
            score += 10

    for kw in anchor_def.get("kw", []):
        if kw and kw in blob:
            score += 2

    return score

def _choose_anchor(query: str, doc_snips: List[Dict[str, str]]) -> dict[str, Any]:
    best = DEFAULT_ANCHOR
    best_score = 0
    for a in ANCHOR_DEFS:
        sc = _score_anchor(query, doc_snips, a)
        if sc > best_score:
            best = a
            best_score = sc
    return best

def _pick_grounding_sentence(doc_snips: List[Dict[str, str]], anchor_kw: List[str], allow_phone: bool) -> str:
    """
    Pick sentence with max keyword overlap.
    If nothing decent, fall back to a safe title-based line.
    """
    best_s = ""
    best_sc = -1

    for d in doc_snips:
        for s in _split_sentences(d.get("snippet", "")):
            if _is_bad_grounding(s):
                continue

            sc = 0
            if anchor_kw:
                sc += sum(1 for kw in anchor_kw if kw and kw in s)

            # 질문/명령문보단 "사실 안내" 문장 선호 (약한 패널티)
            if s.endswith("?"):
                sc -= 1

            # 너무 일반적인 “~가능합니다/~입니다”만 있는 문장도 패널티
            if re.fullmatch(r".*(가능합니다|입니다|됩니다)\.?", s) and len(s) < 25:
                sc -= 1

            if sc > best_sc:
                best_sc = sc
                best_s = s

            # 충분히 관련성 높은 문장 만나면 조기 종료
            if sc >= 2:
                best_s = s
                best_sc = sc
                break

    if best_s:
        best_s = _redact(best_s, allow_phone=allow_phone)
        return best_s

    # title fallback
    for d in doc_snips:
        title = (d.get("title") or "").strip()
        if title:
            title = _redact(title, allow_phone=allow_phone)
            if title:
                return f"관련 안내는 '{title}' 기준으로 진행하시면 됩니다."
    return "문서 안내 기준에 따라 순서대로 진행하시면 됩니다."

def _final_guard(summary: str, guide: str, question: str) -> tuple[str, str, str]:
    # 중복 방지 + 로봇 문구 방지
    a = re.sub(r"\s+", " ", summary).strip()
    b = re.sub(r"\s+", " ", guide).strip()
    c = re.sub(r"\s+", " ", question).strip()

    if b == a:
        b = "관련 안내를 문서 기준으로 정리해드릴게요."
    if c == a or c == b:
        c = "어떤 부분부터 도와드릴까요?"

    # 질문에 "또는/혹은/ /" 섞여 있으면 단일 질문으로 교체
    if re.search(r"(또는|혹은|/)", c):
        c = "지금 어떤 단계에서 막히셨는지 알려주실 수 있을까요?"

    return a, b, c

# -------------------------
# Public API
# -------------------------
def generate_guidance_script(
    query: str,
    docs: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> str:
    # kept for signature compatibility
    _ = model

    query = (query or "").strip()
    if not query or not docs:
        return ""

    allow_phone = bool(PHONE_INTENT_PATTERN.search(query))
    doc_snips = _build_doc_snippets(docs)

    anchor = _choose_anchor(query, doc_snips)
    summary = anchor.get("summary", DEFAULT_ANCHOR["summary"])
    question = anchor.get("question", DEFAULT_ANCHOR["question"])

    guide = _pick_grounding_sentence(
        doc_snips=doc_snips,
        anchor_kw=anchor.get("kw", []),
        allow_phone=allow_phone,
    )

    line1 = _redact(_normalize_sentence(summary, is_question=False), allow_phone=allow_phone)
    line2 = _redact(_normalize_sentence(guide, is_question=False), allow_phone=allow_phone)
    line3 = _redact(_normalize_sentence(question, is_question=True), allow_phone=allow_phone)

    # 질문에 PII 유도 단어가 끼면 교체
    if PII_PATTERN.search(line3):
        line3 = "어떤 부분부터 도와드릴까요?"

    line1, line2, line3 = _final_guard(line1, line2, line3)
    return "\n".join([line1, line2, line3]).strip()

__all__ = ["generate_guidance_script"]
