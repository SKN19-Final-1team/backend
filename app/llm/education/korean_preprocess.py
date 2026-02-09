"""
한국어 TTS 전처리 모듈

숫자, 기호, 약어 등을 한글 발음으로 변환하여
TTS 엔진(특히 VibeVoice, edge-tts)이 정확하게 발음하도록 합니다.

사용법:
    from app.llm.education.korean_preprocess import preprocess_for_tts
    text = preprocess_for_tts("1,500원을 결제하시겠습니까?")
    # → "천오백원을 결제하시겠습니까?"
"""
import re

# ── 숫자 → 한글 ──────────────────────────────────────────────

_DIGITS_KO = {
    0: "", 1: "일", 2: "이", 3: "삼", 4: "사", 5: "오",
    6: "육", 7: "칠", 8: "팔", 9: "구",
}
_UNITS_KO = ["", "십", "백", "천"]
_BIG_UNITS_KO = ["", "만", "억", "조", "경"]


def _number_chunk_to_korean(n: int) -> str:
    """4자리 이하 숫자를 한글로 (0~9999)"""
    if n == 0:
        return ""
    result = []
    s = str(n).zfill(4)
    for i, ch in enumerate(s):
        d = int(ch)
        if d == 0:
            continue
        unit = _UNITS_KO[3 - i]
        if d == 1 and unit:
            result.append(unit)
        else:
            result.append(_DIGITS_KO[d] + unit)
    return "".join(result)


def number_to_korean(n: int) -> str:
    """정수를 한글 읽기로 변환 (최대 경 단위)"""
    if n == 0:
        return "영"
    if n < 0:
        return "마이너스 " + number_to_korean(-n)

    parts = []
    chunk_idx = 0
    while n > 0:
        chunk = n % 10000
        n //= 10000
        chunk_str = _number_chunk_to_korean(chunk)
        if chunk_str:
            big_unit = _BIG_UNITS_KO[chunk_idx] if chunk_idx < len(_BIG_UNITS_KO) else ""
            parts.append(chunk_str + big_unit)
        chunk_idx += 1

    parts.reverse()
    return "".join(parts)


def _float_to_korean(f: float) -> str:
    """소수점 포함 숫자를 한글로"""
    s = str(f)
    if "." not in s:
        return number_to_korean(int(f))
    integer_part, decimal_part = s.split(".", 1)
    result = number_to_korean(int(integer_part)) + " 점 "
    for d in decimal_part:
        if d == "0":
            result += "영 "
        else:
            result += _DIGITS_KO[int(d)] + " "
    return result.strip()


# ── 단위 매핑 ─────────────────────────────────────────────────

_UNIT_MAP = {
    "원": "원", "달러": "달러", "엔": "엔", "유로": "유로",
    "%": "퍼센트", "kg": "킬로그램", "km": "킬로미터",
    "cm": "센티미터", "mm": "밀리미터", "m": "미터",
    "g": "그램", "mg": "밀리그램", "ml": "밀리리터",
    "L": "리터", "℃": "도", "°C": "도",
    "건": "건", "개": "개", "명": "명", "회": "회",
    "일": "일", "주": "주", "월": "월", "년": "년",
    "시": "시", "분": "분", "초": "초",
}

# ── 기호 매핑 ─────────────────────────────────────────────────

_SYMBOL_MAP = {
    "&": " 앤드 ",
    "@": " 앳 ",
    "#": " 샵 ",
    "~": "에서 ",
    "+": " 플러스 ",
    "×": " 곱하기 ",
    "÷": " 나누기 ",
    "=": " 는 ",
    "→": " ",
    "←": " ",
    "↑": " ",
    "↓": " ",
    "※": " 참고, ",
    "·": " ",
    "•": " ",
    "…": " ",
}


# ── 전처리 함수 ───────────────────────────────────────────────

def _replace_number_with_unit(match: re.Match) -> str:
    """'1,500원', '3.5%' 등 숫자+단위 패턴 변환"""
    raw_num = match.group("num").replace(",", "")
    unit_str = match.group("unit") or ""

    # 숫자 변환
    if "." in raw_num:
        korean_num = _float_to_korean(float(raw_num))
    else:
        korean_num = number_to_korean(int(raw_num))

    # 단위 변환
    korean_unit = _UNIT_MAP.get(unit_str, unit_str)
    return korean_num + korean_unit


def _replace_plain_number(match: re.Match) -> str:
    """단위 없는 순수 숫자 변환"""
    raw_num = match.group(0).replace(",", "")
    if "." in raw_num:
        return _float_to_korean(float(raw_num))
    return number_to_korean(int(raw_num))


def _replace_phone_number(match: re.Match) -> str:
    """전화번호 패턴 (010-1234-5678 등)"""
    parts = match.group(0).split("-")
    result = []
    for part in parts:
        for d in part:
            if d == "0":
                result.append("공")
            else:
                result.append(_DIGITS_KO.get(int(d), d))
        result.append(" ")
    return " ".join("".join(result).split())


def _replace_date(match: re.Match) -> str:
    """날짜 패턴 (2024.01.15 또는 2024-01-15)"""
    year = match.group("year")
    month = match.group("month")
    day = match.group("day")
    return f"{number_to_korean(int(year))}년 {number_to_korean(int(month))}월 {number_to_korean(int(day))}일"


def _replace_time(match: re.Match) -> str:
    """시간 패턴 (14:30)"""
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    result = f"{number_to_korean(hour)}시"
    if minute > 0:
        result += f" {number_to_korean(minute)}분"
    return result


# 단위 패턴 (숫자 + 단위 조합)
_UNIT_KEYS = "|".join(re.escape(k) for k in sorted(_UNIT_MAP.keys(), key=len, reverse=True))
_RE_NUM_UNIT = re.compile(
    r'(?P<num>[\d,]+(?:\.\d+)?)(?P<unit>' + _UNIT_KEYS + r')'
)
# 전화번호
_RE_PHONE = re.compile(r'\b\d{2,4}-\d{3,4}-\d{4}\b')
# 날짜 (YYYY.MM.DD 또는 YYYY-MM-DD)
_RE_DATE = re.compile(r'(?P<year>\d{4})[.\-/](?P<month>\d{1,2})[.\-/](?P<day>\d{1,2})')
# 시간 (HH:MM) - \b 대신 (?!\d)로 한국어 호환
_RE_TIME = re.compile(r'(?P<hour>\d{1,2}):(?P<minute>\d{2})(?!\d)')
# 순수 숫자 (콤마 포함, 소수점 포함)
_RE_PLAIN_NUM = re.compile(r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d+\b|\b\d+\b')


def preprocess_for_tts(text: str) -> str:
    """
    TTS를 위한 한국어 텍스트 전처리

    변환 순서:
    1. 전화번호 → 한글 숫자
    2. 날짜 → 한글
    3. 시간 → 한글
    4. 숫자+단위 → 한글
    5. 순수 숫자 → 한글
    6. 기호 → 한글
    7. 공백 정리
    """
    if not text:
        return text

    # 1. 전화번호
    text = _RE_PHONE.sub(_replace_phone_number, text)
    # 2. 날짜
    text = _RE_DATE.sub(_replace_date, text)
    # 3. 시간
    text = _RE_TIME.sub(_replace_time, text)
    # 4. 숫자 + 단위
    text = _RE_NUM_UNIT.sub(_replace_number_with_unit, text)
    # 5. 남은 순수 숫자
    text = _RE_PLAIN_NUM.sub(_replace_plain_number, text)
    # 6. 기호
    for sym, replacement in _SYMBOL_MAP.items():
        text = text.replace(sym, replacement)
    # 7. 따옴표 정리
    text = text.replace("'", "'").replace('"', '"').replace('"', '"')
    # 8. 다중 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()

    return text
