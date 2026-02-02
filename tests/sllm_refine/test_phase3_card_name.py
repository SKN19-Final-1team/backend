"""
Phase 3: CARD_NAME 추출 테스트 (Test Report.md 기반)
"""

import sys
import os
from pathlib import Path

# UTF-8 출력 설정
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.llm.delivery.keyword_extractor import extract_keywords


def test_card_name_extraction():
    """CARD_NAME 추출 테스트 (Test Report.md CARD-001~006)"""
    print("\n" + "=" * 60)
    print("CARD_NAME 추출 테스트")
    print("=" * 60)

    test_cases = [
        {
            "id": "CARD-001",
            "input": "그... 나라 뭐시기 카드요, 군인들 쓰는 거",
            "expected_cards": ["나라사랑카드"],
        },
        {
            "id": "CARD-002",
            "input": "저 그 초록색 있잖아요",
            "expected_cards": ["그린카드"],
        },
        {
            "id": "CARD-003",
            "input": "교통카드 기능 있는 카드 추천해주세요",
            "expected_cards": ["알뜰교통카드"],
        },
        {
            "id": "CARD-004",
            "input": "국민행복카드 신청하려구요",
            "expected_cards": ["국민행복카드"],
        },
        {
            "id": "CARD-005",
            "input": "테디 있잖아요 그거",
            "expected_cards": ["테디카드"],
        },
        {
            "id": "CARD-006",
            "input": "나사카로 신청할게요",
            "expected_cards": ["나라사랑카드"],
        },
    ]

    passed = 0
    for case in test_cases:
        result = extract_keywords(case["input"])

        # expected_cards 중 하나라도 있으면 통과
        found = any(card in result.card_names for card in case["expected_cards"])

        if found:
            print(f"[PASS] {case['id']}")
            print(f"   입력: {case['input']}")
            print(f"   추출: {result.card_names}")
            passed += 1
        else:
            print(f"[FAIL] {case['id']}")
            print(f"   입력: {case['input']}")
            print(f"   기대: {case['expected_cards']}")
            print(f"   실제: {result.card_names}")

    print(f"\nCARD_NAME 통과: {passed}/{len(test_cases)}")
    return passed, len(test_cases)


def test_compound_card_name():
    """COMPOUND 테스트 (카드명 포함)"""
    print("\n" + "=" * 60)
    print("COMPOUND 테스트 (카드명 의존)")
    print("=" * 60)

    test_cases = [
        {
            "id": "COMP-001",
            "input": "나라사람카드 연예비가 얼마예요",
            "expected_cards": ["나라사랑카드"],
        },
        {
            "id": "COMP-005",
            "input": "테디카드 분실신고하고 재발급 받고 싶어요",
            "expected_cards": ["테디카드"],
        },
    ]

    passed = 0
    for case in test_cases:
        result = extract_keywords(case["input"])

        # expected_cards 중 하나라도 있으면 통과
        found = any(card in result.card_names for card in case["expected_cards"])

        if found:
            print(f"[PASS] {case['id']}")
            print(f"   입력: {case['input']}")
            print(f"   추출: {result.card_names}")
            passed += 1
        else:
            print(f"[FAIL] {case['id']}")
            print(f"   입력: {case['input']}")
            print(f"   기대: {case['expected_cards']}")
            print(f"   실제: {result.card_names}")

    print(f"\nCOMPOUND 통과: {passed}/{len(test_cases)}")
    return passed, len(test_cases)


def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 60)
    print("Phase 3: CARD_NAME 추출 개선 검증 테스트")
    print("=" * 60)
    print("목표: CARD_NAME 0% -> 100% (0/6 -> 6/6)")
    print()

    # CARD_NAME 테스트
    card_passed, card_total = test_card_name_extraction()

    # COMPOUND 테스트 (샘플)
    comp_passed, comp_total = test_compound_card_name()

    # 최종 결과
    print("\n" + "=" * 60)
    print("Phase 3 검증 결과")
    print("=" * 60)

    total_passed = card_passed + comp_passed
    total_count = card_total + comp_total

    print(f"CARD_NAME: {card_passed}/{card_total} ({card_passed/card_total*100:.1f}%)")
    print(f"COMPOUND:  {comp_passed}/{comp_total} ({comp_passed/comp_total*100:.1f}%)")
    print(f"\n총 통과율: {total_passed}/{total_count} ({total_passed/total_count*100:.1f}%)")

    if card_passed == card_total:
        print("\n[SUCCESS] Phase 3 목표 달성! CARD_NAME 100% 통과")
        print("예상 전체 통과율: 60.9% -> 73.9% (+6개)")
    elif card_passed >= card_total * 0.8:
        print(f"\n[PROGRESS] Phase 3 진행 중: {card_passed}/{card_total} 통과")
        print(f"남은 개선: {card_total - card_passed}개")
    else:
        print(f"\n[IN PROGRESS] 현재 {card_passed}/{card_total} 통과")

    return card_passed == card_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
