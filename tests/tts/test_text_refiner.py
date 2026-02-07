"""text_refiner 단위 테스트"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.llm.education.text_refiner import unmask_text


def test_single_date():
    assert unmask_text("[날짜]부터요.") == "1일부터요."


def test_multiple_dates():
    result = unmask_text("[날짜#1]부터 [날짜#2]까지요.")
    assert result == "1일부터 15일까지요."


def test_three_dates():
    result = unmask_text("[날짜#1], [날짜#2], [날짜#3]")
    assert result == "1일, 15일, 3일"


def test_amount():
    assert unmask_text("[금액] 결제한 건 맞는데요") == "5만원 결제한 건 맞는데요"


def test_card_company():
    assert unmask_text("[카드사명] 카드인데요") == "테디카드 카드인데요"


def test_customer_name():
    result = unmask_text("[고객명]입니다.", customer_name="김민수")
    assert result == "김민수입니다."


def test_customer_name_fallback():
    result = unmask_text("[고객명]입니다.", customer_name=None)
    assert result == "고객입니다."


def test_phone_number():
    assert unmask_text("번호가 [전화번호]인데요") == "번호가 5916인데요"


def test_card_number():
    assert unmask_text("카드번호 [카드번호]입니다") == "카드번호 4927입니다"


def test_no_mask():
    text = "안녕하세요, 무엇을 도와드릴까요?"
    assert unmask_text(text) == text


def test_mixed():
    text = "[고객명]님, [날짜#1]에 [금액] 결제하신 [카드사명] 카드 건이요."
    result = unmask_text(text, customer_name="박지영")
    assert result == "박지영님, 1일에 5만원 결제하신 테디카드 카드 건이요."


def test_numbered_variant():
    assert unmask_text("[금액#1] 말씀하시는 거죠?") == "5만원 말씀하시는 거죠?"


def test_empty_string():
    assert unmask_text("") == ""


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
