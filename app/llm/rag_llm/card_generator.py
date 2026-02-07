"""
카드 생성기 - guide_generator의 확장 버전
공통 유틸리티는 text_utils에서 import
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os
import re

from app.guide.guide_client import generate_guide_text
from app.guide.text_utils import (
    MAX_DOCS,
    MAX_CONSULT_DOCS,
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
    question_allowed,
)
from app.llm.base import get_openai_client


_CARD_FIELDS = (
    "id",
    "title",
    "keywords",
    "content",
    "systemPath",
    "requiredChecks",
    "exceptions",
    "regulation",
    "fullText",
    "time",
    "note",
    "documentType",
)


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v not in ("", None)]
    if isinstance(value, str):
        parts = [p.strip() for p in re.split(r"[\n;]", value) if p.strip()]
        return parts
    return [str(value)]


def _summarize_text(text: str, limit: int = 180) -> Optional[str]:
    if not text:
        return None
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    if not sents:
        return truncate(text, limit)
    summary = " ".join(sents[:2]).strip()
    return truncate(summary, limit) if summary else None


_TABLE_DOC_TYPE_MAP = {
    "card_tbl": "product-spec",
    "guide_tbl": "guide",
    "card_products": "product-spec",
    "service_guide_documents": "guide",
}


def _table_to_doc_type(table: Optional[str]) -> str:
    return _TABLE_DOC_TYPE_MAP.get(table or "", "general")


def _doc_to_card_base(doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not doc:
        return {k: None for k in _CARD_FIELDS}
    meta = doc.get("metadata") or {}
    # [v25] DB structured JSONB 데이터 활용 (전처리된 카드 표시용 데이터)
    structured = doc.get("structured") or {}
    raw_content = (
        structured.get("detailContent")
        or doc.get("detailContent")
        or meta.get("full_content")
        or doc.get("content")
        or ""
    )
    raw_content = str(raw_content)
    # structured.content가 있으면 이미 정리된 요약이므로 우선 사용
    summary = structured.get("content") or _summarize_text(raw_content)
    return {
        "id": str(doc.get("id") or meta.get("id") or ""),
        "title": structured.get("title") or doc.get("title") or meta.get("title") or "",
        "keywords": _ensure_list(doc.get("keywords") or meta.get("keywords")),
        "content": summary,
        "systemPath": structured.get("systemPath") or meta.get("systemPath") or meta.get("system_path") or "",
        "requiredChecks": _ensure_list(
            structured.get("requiredChecks") or meta.get("requiredChecks") or meta.get("required_checks")
        ),
        "exceptions": _ensure_list(structured.get("exceptions") or meta.get("exceptions")),
        "regulation": structured.get("regulation") or meta.get("regulation") or "",
        "fullText": raw_content.strip() or None,
        "time": structured.get("time") or meta.get("time") or "",
        "note": structured.get("note") or meta.get("note") or "",
        "documentType": meta.get("documentType") or _table_to_doc_type(doc.get("table")) or "general",
    }


def _merge_card(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key in _CARD_FIELDS:
        if key not in overlay:
            continue
        val = overlay.get(key)
        if key in ("requiredChecks", "exceptions", "keywords"):
            lst = _ensure_list(val)
            if lst:
                merged[key] = lst
            continue
        if val not in ("", None):
            merged[key] = val
    # enforce nullable fields for summary/full text
    for key in ("content", "fullText"):
        if merged.get(key) in ("", None):
            merged[key] = None
    # keep strings for others
    for key in ("title", "systemPath", "regulation", "time", "note", "documentType", "id"):
        if merged.get(key) is None:
            merged[key] = ""
    return merged


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text.strip()
    parts = re.split(r"```(?:json)?", text, flags=re.IGNORECASE)
    if len(parts) >= 2:
        rest = parts[1]
        rest = re.split(r"```", rest)[0]
        return rest.strip()
    return text.strip()


def _extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # try to slice to first JSON token
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                continue
    return None


def _build_card_messages(query: str, docs: List[Dict[str, Any]], max_cards: int) -> List[Dict[str, str]]:
    doc_block = build_doc_block(query, docs, max_cards)
    system_prompt = (
        "당신은 카드사 내부 상담 시나리오 카드를 작성하는 AI입니다.\n"
        "아래 Documents에 있는 내용만 사용하여 카드 JSON 배열을 생성하세요.\n\n"
        "[출력 형식]\n"
        "- 반드시 JSON만 출력하세요. (코드블록/설명 금지)\n"
        "- 최상위는 배열 또는 {\"cards\": [...]} 둘 중 하나로 출력합니다.\n\n"
        "[카드 필드]\n"
        "- id, title, keywords, content (1~2문장 요약)\n"
        "- requiredChecks/exceptions: 배열 (없으면 빈 배열)\n"
        "- fullText는 생성하지 마세요 (시스템이 자동 삽입)\n\n"
        "[제약]\n"
        "- 문서에 없는 절차/정책/기간/조건은 절대 만들지 마세요.\n"
        "- 요약은 문서에서 근거가 보이는 내용만 사용하세요.\n"
    )
    user_prompt = (
        f"User query:\n{query}\n\n"
        f"Documents:\n{doc_block or 'NONE'}\n\n"
        f"카드 개수는 최대 {max_cards}개까지만 생성하세요."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_rule_cards(query: str, docs: List[Dict[str, Any]], max_cards: int = 4) -> tuple[List[Dict[str, Any]], str]:
    cards = []
    for doc in docs[:max_cards]:
        base = _doc_to_card_base(doc)
        # ensure nullable fields are present
        if base.get("content") is None and base.get("fullText"):
            base["content"] = _summarize_text(str(base.get("fullText") or ""))
        cards.append(_merge_card(base, {}))
    return cards, ""


def generate_detail_cards(
    query: str,
    docs: List[Dict[str, Any]],
    model: str = "",
    temperature: float = 0.0,
    max_llm_cards: int = 4,
) -> tuple[List[Dict[str, Any]], str]:
    if not docs:
        return [], ""

    model = model or os.getenv("RAG_CARD_MODEL", "gpt-4.1-mini")
    messages = _build_card_messages(query, docs, max_llm_cards)

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=500,
            top_p=0.9,
        )
        output = (resp.choices[0].message.content or "").strip()
    except Exception:
        return build_rule_cards(query, docs, max_cards=max_llm_cards)

    parsed = _extract_json(output)
    if isinstance(parsed, dict):
        parsed = parsed.get("cards")
    if not isinstance(parsed, list):
        return build_rule_cards(query, docs, max_cards=max_llm_cards)

    # docs 순서대로 매칭 (LLM이 생성한 id는 무시하고 원본 문서 순서 사용)
    cards: List[Dict[str, Any]] = []
    for idx, item in enumerate(parsed[:max_llm_cards]):
        if not isinstance(item, dict):
            continue
        doc = docs[idx] if idx < len(docs) else None
        base = _doc_to_card_base(doc)
        merged = _merge_card(base, item)
        cards.append(merged)

    if not cards:
        return build_rule_cards(query, docs, max_cards=max_llm_cards)
    return cards, output


def _apply_question_policy(text: str, query: str) -> str:
    if not text:
        return ""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    if not sents:
        return ""
    kept: List[str] = []
    questions: List[str] = []
    intent = detect_intent(query)

    for sent in sents:
        if question_allowed(sent, intent):
            questions.append(sent)
        elif "?" in sent:
            continue
        else:
            kept.append(sent)

    if questions:
        selected = questions[0]
        q_lower = (query or "").lower()
        if intent == "loss" and "입대" in selected and not any(term in q_lower for term in ["입대", "재발급", "훈련소", "전역"]):
            selected = "분실인지 도난인지 확인해 주세요"
        kept = kept[:2] + [selected]
    else:
        if intent == "loss":
            kept = kept[:2] + ["분실인지 도난인지 확인해 주세요"]
        else:
            kept = kept[:2] + ["안내를 진행해 드릴까요?"]

    return " ".join(kept[:3]).strip()


def _fallback_message(query: str) -> str:
    intent = detect_intent(query)
    if intent == "loss":
        return "카드 분실·도난은 카드사에 신고하셔야 합니다. 신고 후 재발급 절차를 진행하실 수 있습니다. 분실인지 도난인지 확인해 주세요"
    if intent == "reissue":
        return "재발급은 카드사 앱 또는 고객센터에서 신청하실 수 있습니다. 재발급 사유를 확인해 주세요. 재발급을 진행해 드릴까요?"
    if intent == "loan":
        return "카드대출은 한도와 이자율 확인이 필요합니다. 이용 가능 여부를 확인해 주세요. 카드대출 진행을 도와드릴까요?"
    return "해당 내용은 안내 문서에 명시되어 있지 않아 확인이 필요합니다. 관련 정보를 알려주실 수 있을까요? 안내를 진행해 드릴까요?"


def _salvage_from_docs(query: str, docs: List[Dict[str, Any]]) -> str:
    """
    LLM 출력이 비거나 정규화로 소실된 경우에도,
    폴백 템플릿보다 문서 기반으로 최대한 수렴하도록 salvage 문장을 만든다.
    """
    if not docs:
        return ""
    intent = detect_intent(query)

    detail = pick_doc_detail(docs)
    top = docs[0]
    base = redact(str(top.get("content") or ""))
    snippet = truncate(base, 240) if base else ""

    # 1문장 공감/상황
    if intent == "loss":
        s1 = "카드를 잃어버리셔서 많이 당황스러우시겠습니다."
    elif intent == "reissue":
        s1 = "재발급이 필요하신 상황으로 보입니다."
    elif intent == "loan":
        s1 = "대출 관련 문의로 확인됩니다."
    elif intent == "overseas":
        s1 = "해외 이용 중 발생한 상황으로 확인됩니다."
    else:
        s1 = "문의하신 내용을 문서 기준으로 안내드리겠습니다."

    # 2문장: 문서 디테일 우선, 없으면 content snippet
    s2 = detail or snippet
    s2 = redact(s2)
    s2 = truncate(s2, 160)
    if s2 and s2[-1] not in ".!?":
        s2 += "."

    # 3문장 질문
    if intent == "loss":
        s3 = "분실인지 도난인지 확인해 주세요."
    else:
        s3 = "안내를 진행해 드릴까요?"

    # 문서 디테일이 너무 빈약하면 2문장을 최소 문구로
    if not s2.strip(". "):
        if intent == "loss":
            s2 = "분실·도난 신고 절차를 먼저 진행하셔야 합니다."
        elif intent == "reissue":
            s2 = "재발급 신청 절차를 안내드릴 수 있습니다."
        else:
            s2 = "관련 절차를 문서 기준으로 안내드릴 수 있습니다."
        if s2[-1] not in ".!?":
            s2 += "."

    return f"{s1} {s2} {s3}".strip()


def _build_messages(query: str, docs: List[Dict[str, Any]], consult_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """card_generator용 _build_messages - docs는 이미 전처리된 상태"""
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


def generate_guide_message(
    query: str,
    docs: List[Dict[str, Any]],
    consult_docs: List[Dict[str, Any]],
) -> str:
    if not docs:
        return ""

    # docs 전처리를 한 번만 하고, 이후 모든 단계에서 같은 docs를 사용 (입력 흔들림 감소)
    processed_docs = filter_docs_by_intent(query, docs)
    processed_docs = sort_docs_for_guide(query, processed_docs)

    processed_consult = filter_consult_by_intent(query, consult_docs)
    # docs가 있으면 consult_docs는 사용하지 않는 정책 유지
    if processed_docs:
        processed_consult = []

    intent = detect_intent(query)

    messages = _build_messages(query, processed_docs, processed_consult)

    # LLM 호출
    output = generate_guide_text(messages)

    # 정규화: DROP 최소화(치환/완화)
    normalized = normalize_output(output, intent=intent)

    # 질문 정책 적용(3번째 문장 보정)
    normalized = _apply_question_policy(normalized, query)

    # 문서 디테일 1개 강제(일관된 processed_docs 기준)
    if normalized:
        detail = pick_doc_detail(processed_docs)
        if detail and detail not in normalized:
            sents = [s.strip() for s in _SENT_SPLIT.split(normalized) if s and s.strip()]
            if len(sents) == 1:
                sents = [sents[0], detail]
            elif len(sents) >= 2:
                sents[1] = detail
            normalized = " ".join(sents[:3]).strip()
        return normalized

    # 여기로 떨어지는 경우(= LLM output이 비었거나 정규화로 소실):
    # 폴백 템플릿보다 '문서 기반 salvage'로 수렴시켜 흔들림을 줄임
    salvaged = _salvage_from_docs(query, processed_docs)
    if salvaged:
        return salvaged

    # 최후: 기존 템플릿
    return _fallback_message(query)


__all__ = ["generate_guide_message", "generate_detail_cards", "build_rule_cards"]
