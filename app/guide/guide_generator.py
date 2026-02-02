"""
가이드 메시지 생성기
공통 유틸리티는 text_utils에서 import
"""
from __future__ import annotations

from typing import Any, Dict, List
import re

from app.guide.guide_client import generate_guide_text
from app.guide.text_utils import (
    MAX_DOCS,
    MAX_CONSULT_DOCS,
    MAX_SNIPPET_CHARS,
    _SENT_SPLIT,
    doc_text,
    truncate,
    redact,
    pick_doc_detail,
    detect_intent,
    filter_docs_by_intent,
    filter_consult_by_intent,
    sort_docs_for_guide,
    build_doc_block,
    build_consult_block,
    normalize_output,
    apply_question_policy,
)


def _build_messages(query: str, docs: List[Dict[str, Any]], consult_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """guide_generator 전용 _build_messages - 필터링/정렬 포함"""
    docs = filter_docs_by_intent(query, docs)
    docs = sort_docs_for_guide(query, docs)
    doc_block = build_doc_block(query, docs, MAX_DOCS)
    consult_docs = filter_consult_by_intent(query, consult_docs)
    if docs:
        consult_docs = []
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

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _fallback_message(query: str) -> str:
    intent = detect_intent(query)
    if intent == "loss":
        return "카드 분실·도난은 즉시 카드사에 신고하셔야 합니다. 신고 후 재발급 절차를 진행하실 수 있습니다. 분실 신고를 도와드릴까요?"
    if intent == "reissue":
        return "재발급은 카드사 고객센터 또는 앱에서 신청하실 수 있습니다. 재발급을 진행해 드릴까요?"
    if intent == "loan":
        return "카드대출 진행을 도와드릴까요?"
    return "해당 내용은 현재 안내 문서에 명시되어 있지 않아 카드사 고객센터에서 확인이 필요합니다."


def _has_doc_grounding(output: str, docs: List[Dict[str, Any]]) -> bool:
    if not output or not docs:
        return False
    return True


def _docs_contain_terms(docs: List[Dict[str, Any]], terms: List[str]) -> bool:
    if not docs or not terms:
        return False
    for doc in docs:
        title = str(doc.get("title") or "").lower()
        content = doc_text(doc).lower()
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
        return ""
    intent = detect_intent(query)
    messages = _build_messages(query, docs, consult_docs)
    output = generate_guide_text(messages)
    normalized = normalize_output(output, intent)
    normalized = apply_question_policy(normalized, query)
    if normalized:
        detail = pick_doc_detail(docs)
        if detail and detail not in normalized:
            sents = [s.strip() for s in _SENT_SPLIT.split(normalized) if s and s.strip()]
            if len(sents) == 1:
                sents = [sents[0], detail]
            elif len(sents) >= 2:
                sents[1] = detail
            normalized = " ".join(sents[:3]).strip()
        return normalized
    # 마지막 안전장치: 문서 기반 간단 템플릿으로 구성
    top = docs[0]
    content = redact(str(top.get("content") or ""))
    if content:
        snippet = truncate(content, 240)
        if intent == "loss":
            return f"카드 분실·도난으로 불편하시겠습니다. {snippet} 분실인지 도난인지 확인해 주세요"
        if intent == "reissue":
            return f"재발급 안내를 드리겠습니다. {snippet} 재발급을 진행해 드릴까요?"
        if intent == "loan":
            return f"대출 관련 안내를 드리겠습니다. {snippet} 진행을 원하시면 말씀해 주세요?"
        return f"안내 문서를 기준으로 설명드리겠습니다. {snippet} 안내를 진행해 드릴까요?"
    return ""


__all__ = ["generate_guide_message"]
