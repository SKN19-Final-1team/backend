"""
guide_generator.py와 card_generator.py에서 공통으로 사용되는
텍스트 처리 유틸리티 함수 및 정규표현식 패턴들
"""
from __future__ import annotations

from typing import Any, Dict, List
import os
import re

# 환경변수 설정
MAX_DOCS = int(os.getenv("GUIDE_MAX_DOCS", "2"))
MAX_CONSULT_DOCS = int(os.getenv("GUIDE_MAX_CONSULT_DOCS", "1"))
MAX_SNIPPET_CHARS = int(os.getenv("GUIDE_MAX_SNIPPET_CHARS", "600"))

# 정규표현식 패턴들
_PHONE_PATTERN = re.compile(r"\b\d{2,4}-\d{3,4}-\d{4}\b|\b\d{8,13}\b")
_URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\S+\.(com|kr|net|org)\b)", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PLACEHOLDER_PATTERN = re.compile(r"\[[^\]]+#\d+\]")
_SPEAKER_PATTERN = re.compile(r"(손님|고객|상담사)\s*:\s*", re.IGNORECASE)
_FILLER_PATTERN = re.compile(
    r"(잠시만\s*기다려\s*주|기다려\s*주셔서\s*감사|확인\s*후\s*안내|확인해\s*보겠|확인해\s*보니)",
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
_GUIDE_TERM_RE = re.compile(r"[A-Za-z0-9가-힣]+")
_UNIT_WITHOUT_NUMBER = re.compile(r"(?<![0-9가-힣])(원|일|월|시|분|개월|건|%)(?![0-9가-힣])")
_CLAUSE_PATTERN = re.compile(r"제\s*\d+\s*조")
_CLAUSE_ITEM_PATTERN = re.compile(r"[①-⑳]|(?:^|\s)\d+\s*[.)]")
_DOC_TITLE_PATTERN = re.compile(r"\b\w*(?:_\w*){2,}\b")
_DOC_TRAIL_PATTERN = re.compile(r"[가-힣A-Za-z0-9_]*대응방법")
_ACCOUNT_ASSERT_PATTERN = re.compile(
    r"(되어\s*있|등록되어\s*있|가입되어\s*있|완료되었|완료되어|처리해\s*드렸|처리되었습니다)",
    re.IGNORECASE,
)
_GIFT_TERMS_PATTERN = re.compile(
    r"(gift|기프트|선불|테디카드|인터넷\s*이용\s*등록|소득공제\s*신청)",
    re.IGNORECASE,
)
_HARD_ASSERT_PATTERN = re.compile(
    r"(바로|즉시).*(진행|처리|조치|신고|정지).*(해\s*드려야|해드려야|해\s*드려서|해드려서|해\s*드리면|해드리면|드립니다|하겠습니다|합니다)",
    re.IGNORECASE,
)
_LOSS_HARD_PATTERN = re.compile(
    r"(바로|즉시).*(분실|도난).*(신고|정지|차단|처리).*(해야|하셔야|합니다)",
    re.IGNORECASE,
)
_INSTANT_BLOCK_PATTERN = re.compile(r"즉시\s*(정지|차단)", re.IGNORECASE)
_INSTANT_HANDLE_PATTERN = re.compile(
    r"(바로|즉시).*(접수|처리|조치|정지|중지).*(해\s*드리|해드리|하겠|합니다)",
    re.IGNORECASE,
)
_INSTANT_ASSERT_PATTERN = re.compile(r"즉시\s*(정지|중지|차단|처리)\w*", re.IGNORECASE)
_GARBLED_PATTERN = re.compile(r"기재확인이\s*필요합니다지")
_QUESTION_ALLOWED_PATTERNS = [
    re.compile(r"(어느|어떤).*(카드사|은행)"),
    re.compile(r"(진행).*(해\s*드릴까요|해드릴까요|하시겠|하실까요|진행하실까요)"),
    re.compile(r"(분실|도난).*확인|분실인지\s*도난인지"),
]
_FILLER_TOKENS = {
    "네",
    "예",
    "아",
    "음",
    "그럼",
    "그리고",
    "혹시",
    "지금",
    "손님",
    "고객",
}

_BAD_DETAIL_PATTERN = re.compile(
    r"(아래\s*표|다음\s*표|표와\s*같|고객센터|콜센터|문의|연락처|전화|"
    r"금융소비자|유의사항|주의사항|예방\s*수칙|비밀번호|주민등록번호|생일|"
    r"연속된\s*숫자|제\s*\d+\s*조|[①-⑳])",
    re.IGNORECASE,
)


def doc_text(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata") or {}
    return str(doc.get("detailContent") or meta.get("full_content") or doc.get("content") or "")


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def redact(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = _URL_PATTERN.sub("", cleaned)
    cleaned = _EMAIL_PATTERN.sub("", cleaned)
    cleaned = _PHONE_PATTERN.sub("", cleaned)
    cleaned = _PLACEHOLDER_PATTERN.sub("", cleaned)
    cleaned = _SPEAKER_PATTERN.sub("", cleaned)
    cleaned = _FILLER_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def extract_query_terms_for_guide(query: str) -> List[str]:
    raw_terms = _GUIDE_TERM_RE.findall(query or "")
    terms = [term.lower() for term in raw_terms if len(term) >= 2]
    seen = set()
    out: List[str] = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out


def extract_relevant_snippet_for_guide(query: str, content: str, limit: int) -> str:
    if not content:
        return ""
    terms = extract_query_terms_for_guide(query)
    sents = [s.strip() for s in _SENT_SPLIT.split(content) if s and s.strip()]
    if not sents:
        return ""
    picked: List[str] = []
    total = 0
    for s in sents:
        if _BAD_DETAIL_PATTERN.search(s):
            continue
        if terms and not any(t in s.lower() for t in terms):
            continue
        chunk = redact(s)
        if not chunk:
            continue
        if picked and total + len(chunk) + 1 > limit:
            break
        picked.append(chunk)
        total += len(chunk) + 1
        if total >= limit:
            break
    if not picked:
        for s in sents:
            if _BAD_DETAIL_PATTERN.search(s):
                continue
            chunk = redact(s)
            if chunk:
                return truncate(chunk, limit)
        return truncate(redact(sents[0]), limit)
    return truncate(" ".join(picked), limit)


def pick_doc_detail(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return ""
    content = redact(doc_text(docs[0]))
    if not content:
        return ""
    sents = [s.strip() for s in _SENT_SPLIT.split(content) if s and s.strip()]
    if not sents:
        return ""
    for s in sents:
        if _BAD_DETAIL_PATTERN.search(s):
            continue
        if re.search(r"\d", s):
            return truncate(s, 140)
    for s in sents:
        if _BAD_DETAIL_PATTERN.search(s):
            continue
        return truncate(s, 140)
    return ""


def summarize_consult_snippet(text: str, limit: int = 180) -> str:
    if not text:
        return ""
    t = re.sub(r"(?<=\.)\s+", " ", text)
    t = _SENT_SPLIT.sub(" ", t)
    t = redact(t)
    if not t:
        return ""
    sents = [s.strip() for s in _SENT_SPLIT.split(t) if s and s.strip()]
    picked = sents[:2] if sents else [t]
    summary = " ".join(picked).strip()
    return summary[:limit].rstrip()


def sort_docs_for_guide(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not docs:
        return docs
    q = (query or "").lower()
    loss_terms = ["분실", "도난", "잃어버"]
    query_terms = [t for t in loss_terms if t in q]
    if not query_terms:
        def _score_general(doc: Dict[str, Any]) -> tuple[float, str]:
            score = float(doc.get("score") or 0.0)
            doc_id = str((doc.get("metadata") or {}).get("id") or doc.get("id") or "")
            return (score, doc_id)

        return sorted(docs, key=_score_general, reverse=True)

    def _score(doc: Dict[str, Any]) -> tuple[int, float, str]:
        title = str(doc.get("title") or "").lower()
        content = doc_text(doc).lower()
        text = f"{title} {content}"
        hit = sum(1 for t in query_terms if t in text)
        score = float(doc.get("score") or 0.0)
        doc_id = str((doc.get("metadata") or {}).get("id") or doc.get("id") or "")
        return (hit, score, doc_id)

    return sorted(docs, key=_score, reverse=True)


def detect_intent(query: str) -> str:
    q = (query or "").lower()
    if any(term in q for term in ["분실", "도난", "잃어버"]):
        return "loss"
    if any(term in q for term in ["재발급", "재발행"]):
        return "reissue"
    if any(term in q for term in ["대출", "현금서비스", "카드대출", "리볼빙"]):
        return "loan"
    if any(term in q for term in ["해외", "dcc", "원화결제"]):
        return "overseas"
    if any(term in q for term in ["애플페이", "삼성페이", "카카오페이", "티머니", "교통카드"]):
        return "pay"
    return "general"


def filter_docs_by_intent(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not docs:
        return docs
    intent = detect_intent(query)
    if intent == "general":
        return docs
    q_lower = (query or "").lower()
    if intent in {"loss", "reissue"} and not any(term in q_lower for term in ["gift", "기프트", "선불", "테디"]):
        filtered = []
        for doc in docs:
            title = str(doc.get("title") or "").lower()
            content = doc_text(doc).lower()
            text = f"{title} {content}"
            if any(term in text for term in ["gift", "기프트", "선불", "테디카드"]):
                continue
            filtered.append(doc)
        docs = filtered or docs
    if intent == "loss" and not any(term in q_lower for term in ["재발급", "재발행", "수령", "입대", "훈련소", "전역"]):
        filtered = []
        for doc in docs:
            title = str(doc.get("title") or "").lower()
            content = doc_text(doc).lower()
            text = f"{title} {content}"
            if any(term in text for term in ["재발급", "재발행", "수령", "신청 후", "입대", "훈련소"]):
                continue
            filtered.append(doc)
        docs = filtered or docs
    intent_terms_map = {
        "loss": ["분실", "도난", "잃어버"],
        "reissue": ["재발급", "재발행", "재신청"],
        "loan": ["대출", "현금서비스", "리볼빙", "카드대출"],
        "overseas": ["해외", "dcc", "원화결제"],
        "pay": ["애플페이", "삼성페이", "카카오페이", "티머니", "교통카드"],
    }
    intent_terms = intent_terms_map.get(intent, [])
    if not intent_terms:
        return docs
    filtered = []
    for doc in docs:
        title = str(doc.get("title") or "").lower()
        content = doc_text(doc).lower()
        text = f"{title} {content}"
        if any(term in text for term in intent_terms):
            filtered.append(doc)
    return filtered or docs


def filter_consult_by_intent(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return filter_docs_by_intent(query, docs)


def is_low_content_sentence(sentence: str) -> bool:
    s = sentence.strip()
    if not s:
        return True
    if s in {"진행하실 수 있습니다.", "진행하실 수 있습니다"}:
        return True
    hangul_count = sum(1 for ch in s if "가" <= ch <= "힣")
    if hangul_count < 8:
        return True
    tokens = re.findall(r"[가-힣]+", s)
    meaningful = [t for t in tokens if len(t) >= 2 and t not in _FILLER_TOKENS]
    return not meaningful


def build_doc_block(query: str, docs: List[Dict[str, Any]], max_docs: int) -> str:
    parts: List[str] = []
    for idx, doc in enumerate(docs[:max_docs], 1):
        title = (doc.get("title") or (doc.get("metadata") or {}).get("title") or "").strip()
        content = doc_text(doc)
        snippet = extract_relevant_snippet_for_guide(query, content, MAX_SNIPPET_CHARS)
        title = redact(title)
        if not title and not snippet:
            continue
        parts.append(f"[Doc {idx}]\nTitle: {title or 'N/A'}\nContent: {snippet or 'N/A'}")
    return "\n\n".join(parts).strip()


def build_consult_block(docs: List[Dict[str, Any]], max_docs: int) -> str:
    parts: List[str] = []
    for idx, doc in enumerate(docs[:max_docs], 1):
        title = (doc.get("title") or (doc.get("metadata") or {}).get("title") or "").strip()
        content = doc.get("content") or ""
        snippet = truncate(redact(content), MAX_SNIPPET_CHARS)
        title = redact(title)
        summary = summarize_consult_snippet(snippet)
        if not title and not summary:
            continue
        parts.append(f"[Case {idx}]\nTitle: {title or 'N/A'}\nSummary: {summary or 'N/A'}")
    return "\n\n".join(parts).strip()


def build_messages(query: str, docs: List[Dict[str, Any]], consult_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    doc_block = build_doc_block(query, docs, MAX_DOCS)
    consult_block = build_consult_block(consult_docs, MAX_CONSULT_DOCS)

    system_prompt = (
        "당신은 카드사 콜센터 상담원을 돕는 내부 안내 스크립트를 작성하는 AI입니다. "
        "고객에게 바로 읽어줄 수 있는 '완성된 안내 문장'만 작성하세요.\n\n"
        "[작성 원칙]\n"
        "1. 반드시 제공된 Documents와 Consultation cases에 포함된 정보만 사용하세요.\n"
        "2. 문서에 없는 내용, 추측, 일반 상식, 약관 문장 그대로 인용은 절대 금지합니다.\n"
        "3. 법조문·약관 문장은 그대로 옮기지 말고, 상담원이 말하듯 쉽게 풀어서 설명하세요.\n"
        "4. 전화번호, URL, 이메일, 개인정보는 절대 포함하지 마세요.\n\n"
        "[출력 형식]\n"
        "- 전체는 최대 3문장\n"
        "- 문단, 번호, 불릿, 따옴표 사용 금지.\n\n"
        "[문장별 역할]\n"
        "첫 번째 문장: 고객 상황을 한 줄로 정리하며 공감 표현을 합니다.\n"
        "두 번째 문장: 지금 바로 안내해야 할 핵심 처리 방법 또는 절차를 명확하게 설명합니다.\n"
        "세 번째 문장: 안내를 마친 뒤 확인해야 할 핵심 한 가지를 질문합니다.\n\n"
        "[중요 제한 사항]\n"
        "- 이미 문서에 답이 충분한 경우, 불필요한 추가 질문을 하지 마세요.\n"
        "- '어떤 단계에서 막히셨는지', '확인 후 안내드리겠습니다' 같은 모호한 문장은 사용하지 마세요.\n"
        "- '손님:', '고객:', '상담사:' 같은 화자 표기는 절대 쓰지 마세요.\n"
        "- [날짜#], [금액#], [비율#], [카드사명#] 같은 대괄호 플레이스홀더는 절대 쓰지 마세요.\n"
        "- 문서 제목, 파일명, 조항 번호, 조문 표기는 고객에게 절대 말하지 마세요.\n"
        "- '잠시만 기다려 주세요', '확인 후 안내드리겠습니다', '기다려주셔서 감사합니다' 같은 관용구는 절대 쓰지 마세요.\n"
        "- 예방 수칙, 일반 주의사항, 배경 설명은 포함하지 마세요.\n"
        "- 답을 모를 경우에만 한 문장으로 정보 추가 요청을 하세요.\n\n"
        "[근거 사용]\n"
        "- 반드시 Documents 내용에 근거한 문장만 작성하세요.\n"
        "- Documents에 없는 절차/정책/요금/기간/조건은 절대 만들지 마세요.\n\n"
        "[필수 디테일]\n"
        "- Documents에 포함된 구체적 디테일을 최소 1개는 반드시 포함하세요.\n\n"
        "항상 상담원이 고객에게 바로 읽어주는 상황을 가정하고, 간결하고 단정하게 작성하세요."
    )

    user_prompt = (
        f"User query:\n{query}\n\n"
        f"Documents:\n{doc_block or 'NONE'}\n\n"
        f"Consultation cases:\n{consult_block or 'NONE'}"
    )

    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def sanitize_risky_sentence(sentence: str, intent: str) -> str:
    """
    '즉시 처리/정지해드리겠습니다' 같은 확정/권한 과장 표현은 삭제하지 않고 완화/중립화한다.
    => normalized가 빈 문자열이 되는 폴백 흔들림을 크게 줄임.
    """
    s = sentence.strip()
    if not s:
        return ""

    s = redact(s)
    s = re.sub(r"\b(바로|즉시)\b", "", s).strip()
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"(해\s*드리겠습니다|해드리겠습니다|처리해\s*드리겠습니다|처리해드리겠습니다)", "안내드릴 수 있습니다", s)
    s = re.sub(r"(하겠습니다|진행하겠습니다)", "진행할 수 있습니다", s)

    if intent == "loss":
        s = re.sub(r"(정지|차단)\s*해\s*드리", r"\1될 수 있", s)

    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def normalize_output(text: str, intent: str) -> str:
    """
    기존: 금지 패턴에 걸리면 문장을 DROP => filtered가 비어서 폴백 흔들림 발생
    변경: DROP 최소화, 치환/완화(sanitize) 후 남기는 쪽으로 수렴
    """
    if not text:
        return ""
    t = " ".join([ln.strip() for ln in text.splitlines() if ln and ln.strip()])
    t = _SPEAKER_PATTERN.sub("", t)
    t = _PLACEHOLDER_PATTERN.sub("", t)
    t = _FILLER_PATTERN.sub("", t)
    t = _CLAUSE_PATTERN.sub("", t)
    t = _CLAUSE_ITEM_PATTERN.sub("", t)
    t = _GIFT_TERMS_PATTERN.sub("", t)
    t = _GARBLED_PATTERN.sub("기재되어 있지 않아", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    if not t:
        return ""

    sents = [s.strip() for s in _SENT_SPLIT.split(t) if s and s.strip()]
    if not sents:
        return ""

    kept: List[str] = []
    for s in sents:
        if is_low_content_sentence(s):
            continue

        if _ACCOUNT_ASSERT_PATTERN.search(s) or _INSTANT_HANDLE_PATTERN.search(s) or _INSTANT_ASSERT_PATTERN.search(s):
            s = sanitize_risky_sentence(s, intent)
            if not s:
                continue

        s = _DOC_TITLE_PATTERN.sub("", s)
        s = _DOC_TRAIL_PATTERN.sub("", s)
        s = re.sub(r"\s{2,}", " ", s).strip()

        if not s:
            continue
        kept.append(s)
        if len(kept) >= 3:
            break

    if not kept:
        return ""

    normalized_sents: List[str] = []
    for s in kept[:3]:
        s = s.strip()
        if s and s[-1] not in ".!?":
            s = f"{s}."
        normalized_sents.append(s)
    return " ".join(normalized_sents).strip()


def question_allowed(sentence: str, intent: str) -> bool:
    if "?" not in sentence and not sentence.strip().endswith("요"):
        return False
    if any(token in sentence for token in ["카드 번호", "본인 확인", "인증"]):
        return False
    if "확인해 주시겠" in sentence or "확인해주시겠" in sentence:
        return False
    return True


def apply_question_policy(text: str, query: str) -> str:
    if not text:
        return ""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    if not sents:
        return ""
    intent = detect_intent(query)
    if len(sents) <= 2:
        return text
    last = sents[-1]
    if not question_allowed(last, intent):
        sents = sents[:-1]
    return " ".join(sents).strip()
