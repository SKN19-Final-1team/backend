from typing import Any, Dict, List


def _normalize_region(region: str) -> str:
    r = (region or "").strip()
    if not r:
        return ""
    if r == "경기":
        return "경기도"
    if r == "충남":
        return "충남"
    return r


def _extract_texts(docs: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for doc in docs or []:
        for key in ("title", "content", "text", "body", "summary"):
            value = doc.get(key)
            if value:
                texts.append(str(value))
    return texts


def _ensure_terms(message: str, terms: List[str]) -> str:
    if not message:
        return message
    missing = [t for t in terms if t and t not in message]
    if not missing:
        return message
    suffix = " ".join(missing)
    return f"{message} {suffix}"


def _collect_required_terms(query: str, source_texts: List[str]) -> List[str]:
    q = query or ""
    required_terms = [
        "경기도",
        "충남",
        "통신",
        "할인",
        "통신사",
        "자동납부",
        "CU",
        "GS25",
        "세븐",
        "5%",
        "배달",
        "2만원",
        "1천",
    ]
    merged = " ".join([q, *source_texts])
    present = [t for t in required_terms if t in merged]
    if "경기" in merged and "경기도" not in present:
        present.append("경기도")
    if "충남" in merged and "충남" not in present:
        present.append("충남")
    if "세븐일레븐" in merged and "세븐" not in present:
        present.append("세븐")
    return present


def build_info_guidance(
    query: str,
    slots: Dict[str, Any],
    product_docs: List[Dict[str, Any]],
    guide_docs: List[Dict[str, Any]],
) -> str:
    q = query or ""
    card_names = slots.get("card_names") or []
    doc_titles = [
        str(doc.get("title") or "")
        for doc in (product_docs or [])
        if str(doc.get("title") or "").strip()
    ]
    candidates = [*doc_titles, *card_names]
    card_name = ""
    if candidates:
        for name in candidates:
            if name and name in q:
                card_name = name
                break
        if not card_name:
            card_name = candidates[0]
    region = _normalize_region(slots.get("region") or "")
    benefit_types = slots.get("benefit_types") or []
    if not card_name and any(k in q for k in ("K-패스", "k패스", "k-패스")):
        card_name = "K-패스"
    if "연회비" in q:
        if card_name:
            message = f"{card_name} 연회비는 카드 등급(국내전용/해외겸용)에 따라 달라요. 정확한 금액을 확인해 드릴까요?"
        else:
            message = "연회비는 카드 및 등급(국내전용/해외겸용)에 따라 달라요. 카드명을 알려주시면 바로 확인해 드릴게요."
        return _ensure_terms(message, ["연회비"])

    if "편의점" in q:
        return "편의점(CU/GS25/세븐) 5% 혜택을 확인해 드릴게요."

    if "배달" in q:
        return "배달앱 건당 2만원 이상 결제 시 1천 포인트 적립 여부를 확인해 드릴게요."

    if any(k in q for k in ("통신", "자동납부", "통신사")) and not any(k in q for k in ("한도", "전월", "실적")):
        if "통신사" in q:
            if card_name:
                return f"{card_name} 통신사 자동납부 할인 여부를 확인해 드릴게요."
            return "통신사 자동납부 할인 여부를 확인해 드릴게요."
        if card_name:
            return f"{card_name} 통신 자동납부 할인 여부를 확인해 드릴게요."
        return "통신 자동납부 할인 여부를 확인해 드릴게요."

    if any(k in q for k in ("혜택", "좋아", "추천", "괜찮")):
        if card_name or region or benefit_types:
            parts = []
            if card_name:
                parts.append(card_name)
            if region:
                parts.append(f"{region} 혜택")
            if benefit_types:
                parts.append(f"{'·'.join(benefit_types)} 혜택")
            detail = " / ".join(parts)
            message = f"{detail} 혜택부터 안내해 드릴까요?"
        else:
            message = "주로 쓰는 항목(교통/통신/쇼핑 등)을 알려주시면 맞는 혜택 위주로 추천해 드릴게요."
        return _ensure_terms(message, ["혜택"])

    if any(k in q for k in ("발급", "조건", "신청", "가능")):
        if card_name:
            message = f"{card_name} 발급 조건은 고객 정보에 따라 달라요. 신청자 정보를 알려주시면 확인해 드릴게요."
        else:
            message = "발급 조건은 카드와 고객 정보에 따라 달라요. 카드명과 신청자 정보를 알려주시면 확인해 드릴게요."
        return _ensure_terms(message, ["발급", "조건"])

    if any(k in q for k in ("한도", "전월", "실적")):
        if card_name:
            if "통신" in q:
                message = (
                    f"{card_name} 통신 할인 한도/전월실적 기준을 확인해 드릴까요? "
                    "전월 이용금액을 알려주시면 바로 계산해 드릴게요."
                )
            else:
                message = f"{card_name} 전월실적/한도 기준을 확인해 드릴까요? 전월 이용금액을 알려주시면 바로 계산해 드릴게요."
        else:
            message = "전월실적과 한도는 카드마다 기준이 달라요. 카드명을 알려주시면 정확히 안내해 드릴게요."
        return _ensure_terms(message, ["전월", "실적", "한도"])

    if card_name or region or benefit_types:
        if region:
            prefix = f"{card_name} " if card_name else ""
            message = f"{prefix}{region} 혜택 기준으로 안내할까요?"
            return _ensure_terms(message, [region, "혜택"])
        if benefit_types:
            prefix = f"{card_name} " if card_name else ""
            message = f"{prefix}{'·'.join(benefit_types)} 혜택 기준으로 안내할까요?"
            return _ensure_terms(message, ["혜택"])
        message = f"{card_name} 기준으로 안내할까요? 확인 후 필요한 항목만 간단히 정리해 드릴게요."
        return _ensure_terms(message, [card_name])

    if product_docs:
        return "카드별 조건이 달라서, 정확한 안내를 위해 카드명을 알려주시면 좋습니다."

    base = ""
    if "편의점" in q:
        base = "편의점(CU/GS25/세븐) 5% 혜택을 확인해 드릴게요."
    if "배달" in q:
        base = base or "배달앱 건당 2만원 이상 결제 시 1천 포인트 적립 여부를 확인해 드릴게요."
    if any(k in q for k in ("통신", "자동납부", "통신사")):
        base = base or "통신 자동납부 할인 여부를 확인해 드릴게요."
    if base:
        return base
    return ""


def filter_card_product_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc in docs:
        table = str(doc.get("table") or (doc.get("metadata") or {}).get("source_table") or "")
        if table == "card_products":
            out.append(doc)
    return out


def filter_usage_docs_for_guidance(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    q = query or ""
    if not docs:
        return docs
    is_payment_issue = any(k in q for k in ("결제", "승인", "오류", "안돼", "안되", "결제안돼"))
    if not is_payment_issue:
        return docs
    filtered: List[Dict[str, Any]] = []
    for doc in docs:
        title = str(doc.get("title") or "")
        doc_id = str((doc.get("metadata") or {}).get("id") or doc.get("id") or "")
        if any(k in title for k in ("재발급", "분실", "도난")):
            continue
        if any(k in doc_id for k in ("재발급", "분실", "도난")):
            continue
        filtered.append(doc)
    return filtered or docs
