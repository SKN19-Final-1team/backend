from __future__ import annotations

from typing import Any, Dict, List
import os
import re

from app.guide.guide_client import generate_guide_text

MAX_DOCS = int(os.getenv("GUIDE_MAX_DOCS", "3"))
MAX_CONSULT_DOCS = int(os.getenv("GUIDE_MAX_CONSULT_DOCS", "2"))
MAX_SNIPPET_CHARS = int(os.getenv("GUIDE_MAX_SNIPPET_CHARS", "600"))

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
_UNIT_WITHOUT_NUMBER = re.compile(r"(?<![0-9가-힣])(원|일|월|시|분|개월|건|%)(?![0-9가-힣])")
_CLAUSE_PATTERN = re.compile(r"제\s*\d+\s*조")
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


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _redact(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = _URL_PATTERN.sub("", cleaned)
    cleaned = _EMAIL_PATTERN.sub("", cleaned)
    cleaned = _PHONE_PATTERN.sub("", cleaned)
    cleaned = _PLACEHOLDER_PATTERN.sub("", cleaned)
    cleaned = _SPEAKER_PATTERN.sub("", cleaned)
    cleaned = _FILLER_PATTERN.sub("", cleaned)
    cleaned = _UNIT_WITHOUT_NUMBER.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _summarize_consult_snippet(text: str, limit: int = 180) -> str:
    if not text:
        return ""
    t = re.sub(r"(?<=\.)\s+", " ", text)
    t = _SENT_SPLIT.sub(" ", t)
    t = _redact(t)
    if not t:
        return ""
    sents = [s.strip() for s in _SENT_SPLIT.split(t) if s and s.strip()]
    picked = sents[:2] if sents else [t]
    summary = " ".join(picked).strip()
    return summary[:limit].rstrip()


def _sort_docs_for_guide(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not docs:
        return docs
    q = (query or "").lower()
    loss_terms = ["분실", "도난", "잃어버"]
    query_terms = [t for t in loss_terms if t in q]
    if not query_terms:
        return docs

    def _score(doc: Dict[str, Any]) -> tuple[int, float]:
        title = str(doc.get("title") or "").lower()
        content = str(doc.get("content") or "").lower()
        text = f"{title} {content}"
        hit = sum(1 for t in query_terms if t in text)
        score = float(doc.get("score") or 0.0)
        return (hit, score)

    return sorted(docs, key=_score, reverse=True)


def _detect_intent(query: str) -> str:
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


def _filter_docs_by_intent(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not docs:
        return docs
    intent = _detect_intent(query)
    if intent == "general":
        return docs
    q_lower = (query or "").lower()
    if intent in {"loss", "reissue"} and not any(term in q_lower for term in ["gift", "기프트", "선불", "테디"]):
        filtered = []
        for doc in docs:
            title = str(doc.get("title") or "").lower()
            content = str(doc.get("content") or "").lower()
            text = f"{title} {content}"
            if any(term in text for term in ["gift", "기프트", "선불", "테디카드"]):
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
        content = str(doc.get("content") or "").lower()
        text = f"{title} {content}"
        if any(term in text for term in intent_terms):
            filtered.append(doc)
    return filtered or docs


def _filter_consult_by_intent(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _filter_docs_by_intent(query, docs)


def _is_low_content_sentence(sentence: str) -> bool:
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


def _build_doc_block(docs: List[Dict[str, Any]], max_docs: int) -> str:
    parts: List[str] = []
    for idx, doc in enumerate(docs[:max_docs], 1):
        title = (doc.get("title") or (doc.get("metadata") or {}).get("title") or "").strip()
        content = doc.get("content") or ""
        snippet = _truncate(_redact(content), MAX_SNIPPET_CHARS)
        title = _redact(title)
        if not title and not snippet:
            continue
        parts.append(
            f"[Doc {idx}]\nTitle: {title or 'N/A'}\nContent: {snippet or 'N/A'}"
        )
    return "\n\n".join(parts).strip()


def _build_consult_block(docs: List[Dict[str, Any]], max_docs: int) -> str:
    parts: List[str] = []
    for idx, doc in enumerate(docs[:max_docs], 1):
        title = (doc.get("title") or (doc.get("metadata") or {}).get("title") or "").strip()
        content = doc.get("content") or ""
        snippet = _truncate(_redact(content), MAX_SNIPPET_CHARS)
        title = _redact(title)
        summary = _summarize_consult_snippet(snippet)
        if not title and not summary:
            continue
        parts.append(
            f"[Case {idx}]\nTitle: {title or 'N/A'}\nSummary: {summary or 'N/A'}"
        )
    return "\n\n".join(parts).strip()


def _build_messages(query: str, docs: List[Dict[str, Any]], consult_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    docs = _filter_docs_by_intent(query, docs)
    docs = _sort_docs_for_guide(query, docs)
    doc_block = _build_doc_block(docs, MAX_DOCS)
    consult_docs = _filter_consult_by_intent(query, consult_docs)
    consult_block = _build_consult_block(consult_docs, MAX_CONSULT_DOCS)

    system_prompt = (
        "당신은 카드사 콜센터 상담원을 돕는 내부 안내 스크립트를 작성하는 AI입니다. "
        "고객에게 바로 읽어줄 수 있는 ‘완성된 안내 문장’만 작성하세요.\n\n"

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
        "- ‘어떤 단계에서 막히셨는지’, ‘확인 후 안내드리겠습니다’ 같은 모호한 문장은 사용하지 마세요.\n"
        "- '손님:', '고객:', '상담사:' 같은 화자 표기는 절대 쓰지 마세요.\n"
        "- [날짜#], [금액#], [비율#], [카드사명#] 같은 대괄호 플레이스홀더는 절대 쓰지 마세요.\n"
        "- 문서 제목, 파일명, 조항 번호, 조문 표기는 고객에게 절대 말하지 마세요.\n"
        "- ‘잠시만 기다려 주세요’, ‘확인 후 안내드리겠습니다’, ‘기다려주셔서 감사합니다’ 같은 관용구는 절대 쓰지 마세요.\n"
        "- 예방 수칙, 일반 주의사항, 배경 설명은 포함하지 마세요.\n"
        "- 답을 모를 경우에만 한 문장으로 정보 추가 요청을 하세요.\n\n"
        "[근거 사용]\n"
        "- 반드시 Documents 내용에 근거한 문장만 작성하세요.\n"
        "- Documents에 없는 절차/정책/요금/기간/조건은 절대 만들지 마세요.\n\n"

        "항상 상담원이 고객에게 바로 읽어주는 상황을 가정하고, 간결하고 단정하게 작성하세요."
    )

    user_prompt = (
        f"User query:\n{query}\n\n"
        f"Documents:\n{doc_block or 'NONE'}\n\n"
        f"Consultation cases:\n{consult_block or 'NONE'}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _normalize_output(text: str) -> str:
    if not text:
        return ""
    t = " ".join([ln.strip() for ln in text.splitlines() if ln and ln.strip()])
    t = _SPEAKER_PATTERN.sub("", t)
    t = _PLACEHOLDER_PATTERN.sub("", t)
    t = _FILLER_PATTERN.sub("", t)
    t = _CLAUSE_PATTERN.sub("", t)
    t = _DOC_TITLE_PATTERN.sub("", t)
    t = _DOC_TRAIL_PATTERN.sub("", t)
    t = _GIFT_TERMS_PATTERN.sub("", t)
    t = _UNIT_WITHOUT_NUMBER.sub("", t)
    t = _ACCOUNT_ASSERT_PATTERN.sub("확인이 필요합니다", t)
    t = _HARD_ASSERT_PATTERN.sub("진행하실 수 있습니다", t)
    t = _LOSS_HARD_PATTERN.sub("신고하실 수 있습니다", t)
    t = _INSTANT_BLOCK_PATTERN.sub("정지될 수 있습니다", t)
    t = _GARBLED_PATTERN.sub("기재되어 있지 않아", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    if not t:
        return ""
    sents = [s.strip() for s in _SENT_SPLIT.split(t) if s and s.strip()]
    if not sents:
        return ""
    filtered = [
        s
        for s in sents
        if not _is_low_content_sentence(s)
        and not _HARD_ASSERT_PATTERN.search(s)
        and not _LOSS_HARD_PATTERN.search(s)
    ]
    if not filtered:
        return ""
    normalized_sents: List[str] = []
    for s in filtered[:3]:
        s = s.strip()
        if s and s[-1] not in ".!?":
            s = f"{s}."
        normalized_sents.append(s)
    return " ".join(normalized_sents).strip()


def _question_allowed(sentence: str) -> bool:
    if "?" not in sentence and not sentence.strip().endswith("요"):
        return False
    if _QUESTION_ALLOWED_PATTERNS[0].search(sentence):
        return True
    if any(token in sentence for token in ["카드 번호", "본인 확인", "인증"]):
        return False
    if "확인해 주시겠" in sentence or "확인해주시겠" in sentence:
        return False
    if any(token in sentence for token in ["어느", "어떤", "어디", "몇", "경로"]):
        return False
    return any(pattern.search(sentence) for pattern in _QUESTION_ALLOWED_PATTERNS)


def _apply_question_policy(text: str, query: str) -> str:
    if not text:
        return ""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    if not sents:
        return ""
    kept: List[str] = []
    questions: List[str] = []
    for sent in sents:
        if _question_allowed(sent):
            questions.append(sent)
        elif "?" in sent:
            continue
        else:
            kept.append(sent)
    if questions:
        kept = kept[:2] + [questions[0]]
    else:
        intent = _detect_intent(query)
        if intent == "loss":
            kept = kept[:2] + ["분실인지 도난인지 확인해 주세요?"]
        else:
            kept = kept[:2] + ["안내를 진행해 드릴까요?"]
    # ensure there is a second sentence
    if len(kept) < 2:
        intent = _detect_intent(query)
        if intent == "loss":
            kept = kept[:1] + ["분실·도난 신고는 카드사 고객센터나 앱에서 진행하실 수 있습니다."] + kept[1:]
        elif intent == "loan":
            kept = kept[:1] + ["대출 신청은 카드사 앱이나 고객센터를 통해 진행하실 수 있습니다."] + kept[1:]
        else:
            kept = kept[:1] + ["필요한 절차는 카드사 안내에 따라 진행하실 수 있습니다."] + kept[1:]
    return " ".join(kept[:3]).strip()


def _fallback_message(query: str) -> str:
    intent = _detect_intent(query)
    if intent == "loss":
        return "카드 분실·도난은 즉시 카드사에 신고하셔야 합니다. 신고 후 재발급 절차를 진행하실 수 있습니다. 분실인지 도난인지 확인해 주세요?"
    if intent == "reissue":
        return "재발급은 카드사 고객센터 또는 앱에서 신청하실 수 있습니다. 재발급을 진행해 드릴까요?"
    if intent == "loan":
        return "카드대출은 카드사 앱이나 고객센터를 통해 신청하실 수 있습니다. 진행을 원하시면 말씀해 주세요?"
    return "해당 내용은 현재 안내 문서에 명시되어 있지 않아 카드사 고객센터에서 확인이 필요합니다."


def _has_doc_grounding(output: str, docs: List[Dict[str, Any]]) -> bool:
    if not output or not docs:
        return False
    out = output.lower()
    for doc in docs[:MAX_DOCS]:
        content = (doc.get("content") or "").strip()
        if content:
            # check a few key tokens
            tokens = [t for t in re.findall(r"[가-힣A-Za-z0-9]{2,}", content)][:6]
            if any(t.lower() in out for t in tokens):
                return True
    return False


def _docs_contain_terms(docs: List[Dict[str, Any]], terms: List[str]) -> bool:
    if not docs or not terms:
        return False
    for doc in docs:
        title = str(doc.get("title") or "").lower()
        content = str(doc.get("content") or "").lower()
        text = f"{title} {content}"
        if any(term in text for term in terms):
            return True
    return False


def generate_guide_message(
    query: str,
    docs: List[Dict[str, Any]],
    consult_docs: List[Dict[str, Any]],
) -> str:
    if not docs:
        return "해당 내용은 현재 안내 문서에 명시되어 있지 않아 카드사 고객센터에서 확인이 필요합니다."
    q_lower = (query or "").lower()
    if any(term in q_lower for term in ["dcc", "원화결제", "원화 결제"]) and not _docs_contain_terms(
        docs, ["dcc", "원화결제", "원화 결제"]
    ):
        return "해당 내용은 현재 안내 문서에 명시되어 있지 않아 카드사 고객센터에서 확인이 필요합니다."
    messages = _build_messages(query, docs, consult_docs)
    output = generate_guide_text(messages)
    normalized = _normalize_output(output)
    normalized = _apply_question_policy(normalized, query)
    if not _has_doc_grounding(normalized, docs):
        return "해당 내용은 현재 안내 문서에 명시되어 있지 않아 카드사 고객센터에서 확인이 필요합니다."
    if not normalized:
        return _fallback_message(query)
    return normalized


__all__ = ["generate_guide_message"]
