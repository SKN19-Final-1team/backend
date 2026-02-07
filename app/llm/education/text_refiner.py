"""
sLLM 응답 마스킹 텍스트 정제 모듈

sLLM이 생성한 고객 응답에 포함된 마스킹 토큰([날짜], [금액] 등)을
자연스러운 한국어로 교체합니다.
"""
import re
from typing import Optional

# 날짜 교체값 (순환)
_DATE_REPLACEMENTS = ["1일", "15일", "3일", "20일", "7일", "25일"]

# 고정 교체값
_STATIC_REPLACEMENTS = {
    "금액": "5만원",
    "카드사명": "테디카드",
    "전화번호": "5916",
    "카드번호": "4927",
}

# 마스킹 패턴 정규식: [태그] 또는 [태그#N]
_MASK_PATTERN = re.compile(
    r"\[("
    r"날짜|금액|카드사명|고객명|전화번호|카드번호"
    r")(?:#?\d+)?\]"
)


def unmask_text(text: str, customer_name: Optional[str] = None) -> str:
    """
    sLLM 응답의 마스킹 토큰을 자연스러운 텍스트로 교체

    Args:
        text: sLLM 응답 텍스트 (예: "[날짜#1]부터 [날짜#2]까지요.")
        customer_name: 고객명 (세션의 customer_profile에서 가져옴)

    Returns:
        정제된 텍스트 (예: "1일부터 15일까지요.")
    """
    if not text:
        return text

    date_counter = 0

    def _replace(match: re.Match) -> str:
        nonlocal date_counter
        tag = match.group(1)

        if tag == "날짜":
            replacement = _DATE_REPLACEMENTS[date_counter % len(_DATE_REPLACEMENTS)]
            date_counter += 1
            return replacement

        if tag == "고객명":
            return customer_name or "고객"

        return _STATIC_REPLACEMENTS.get(tag, match.group(0))

    return _MASK_PATTERN.sub(_replace, text)
