"""
Phase 2 개선사항 검증 테스트

Task 2.1: 구어체-전문용어 매핑 확장 (액션 15개, 의도 10개)
Task 2.2: 보정 사전 정리
Task 2.3: 결제수단 동의어 확장
"""

import sys
import os
from pathlib import Path

# UTF-8 출력 설정 (Windows 콘솔 인코딩 문제 해결)
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.llm.delivery.keyword_extractor import extract_keywords


def test_task_2_1_action_patterns_expanded():
    """Task 2.1: 액션 패턴 확장 검증 (15개)"""
    print("\n" + "=" * 60)
    print("Task 2.1: 액션 패턴 확장 검증 (5개 → 15개)")
    print("=" * 60)

    test_cases = [
        {
            "id": "ACT-002",
            "input": "새 카드 하나 더 만들려고 하는데요",
            "expected_actions": ["발급", "신청"]  # 둘 중 하나라도 있으면 통과
        },
        {
            "id": "ACT-008",
            "input": "카드 기간 다 됐는데 어떻게 해요",
            "expected_actions": ["갱신", "만료"]  # 둘 중 하나라도 있으면 통과
        },
        {
            "id": "ACT-EXTRA-2",
            "input": "카드 변경하고 싶어요",
            "expected_actions": ["변경"]
        },
        {
            "id": "ACT-EXTRA-3",
            "input": "이용내역 확인하려고요",
            "expected_actions": ["확인", "조회"]  # 둘 중 하나
        },
        {
            "id": "ACT-EXTRA-4",
            "input": "청구서 정산 문의드립니다",
            "expected_actions": ["정산"]
        },
    ]

    passed = 0
    for case in test_cases:
        result = extract_keywords(case["input"])

        # expected_actions 중 하나라도 있으면 통과
        found = any(action in result.actions for action in case["expected_actions"])

        if found:
            print(f"[PASS] {case['id']}: PASS")
            print(f"   입력: {case['input']}")
            print(f"   추출: {result.actions}")
            passed += 1
        else:
            print(f"[FAIL] {case['id']}: FAIL")
            print(f"   입력: {case['input']}")
            print(f"   기대: {case['expected_actions']} 중 하나")
            print(f"   실제: {result.actions}")

    print(f"\n통과: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_task_2_1_intent_patterns_added():
    """Task 2.1: 의도 패턴 추가 검증 (10개)"""
    print("\n" + "=" * 60)
    print("Task 2.1: 의도 패턴 추가 검증 (0개 → 10개)")
    print("=" * 60)

    test_cases = [
        {
            "id": "INT-001",
            "input": "이 카드 뭐가 좋아요?",
            "expected_intents": ["혜택"]
        },
        {
            "id": "INT-002",
            "input": "1년에 얼마 내야돼요",
            "expected_intents": ["연회비"]
        },
        {
            "id": "INT-003",
            "input": "여기 쓰면 얼마나 깎아줘요",
            "expected_intents": ["할인"]
        },
        {
            "id": "INT-004",
            "input": "돈 모으려면 어떤 카드가 좋아요",
            "expected_intents": ["적립"]
        },
        {
            "id": "INT-005",
            "input": "마일리지 쌓이는 카드 뭐 있어요",
            "expected_intents": ["적립", "마일리지"]  # 둘 중 하나
        },
        {
            "id": "INT-EXTRA-1",
            "input": "이 카드랑 저 카드 비교해주세요",
            "expected_intents": ["비교"]
        },
        {
            "id": "INT-EXTRA-2",
            "input": "발급 조건이 어떻게 돼요",
            "expected_intents": ["조건"]
        },
        {
            "id": "INT-EXTRA-3",
            "input": "캐시백 받으려면요",
            "expected_intents": ["캐시백"]
        },
    ]

    passed = 0
    for case in test_cases:
        result = extract_keywords(case["input"])

        # expected_intents 중 하나라도 있으면 통과
        found = any(intent in result.intents for intent in case["expected_intents"])

        if found:
            print(f"[PASS] {case['id']}: PASS")
            print(f"   입력: {case['input']}")
            print(f"   추출: {result.intents}")
            passed += 1
        else:
            print(f"[FAIL] {case['id']}: FAIL")
            print(f"   입력: {case['input']}")
            print(f"   기대: {case['expected_intents']} 중 하나")
            print(f"   실제: {result.intents}")

    print(f"\n통과: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_task_2_2_correction_dict_cleanup():
    """Task 2.2: 보정 사전 정리 검증 (자기 자신 매핑 삭제 확인)"""
    print("\n" + "=" * 60)
    print("Task 2.2: 보정 사전 정리 검증")
    print("=" * 60)

    test_cases = [
        {
            "id": "CARD-004",
            "input": "국민행복카드 신청하려구요",
            "expected_card": "국민행복카드",
            "should_not_contain": "국민행복카드카드"  # 중복 없어야 함
        },
        {
            "id": "CARD-EXTRA-1",
            "input": "나라사랑카드 연회비 얼마예요",
            "expected_card": "나라사랑카드",
            "should_not_contain": "나라사랑카드카드"
        },
    ]

    passed = 0
    for case in test_cases:
        result = extract_keywords(case["input"])

        # 중복 보정 체크
        has_duplicate = case["should_not_contain"] in result.corrected_text

        if not has_duplicate:
            print(f"[PASS] {case['id']}: PASS")
            print(f"   입력: {case['input']}")
            print(f"   보정: {result.corrected_text}")
            print(f"   중복 없음: '{case['should_not_contain']}' 발견 안 됨")
            passed += 1
        else:
            print(f"[FAIL] {case['id']}: FAIL")
            print(f"   입력: {case['input']}")
            print(f"   보정: {result.corrected_text}")
            print(f"   중복 발견: '{case['should_not_contain']}'")

    print(f"\n통과: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_task_2_3_payment_synonyms_expanded():
    """Task 2.3: 결제수단 동의어 확장 검증"""
    print("\n" + "=" * 60)
    print("Task 2.3: 결제수단 동의어 확장 검증")
    print("=" * 60)

    test_cases = [
        {
            "id": "PAY-001",
            "input": "삼성페이에 카드 넣으려는데요",
            "expected_payment": "삼성페이"
        },
        {
            "id": "PAY-EXTRA-1",
            "input": "네이버로 결제하고 싶어요",
            "expected_payment": "네이버페이"
        },
        {
            "id": "PAY-EXTRA-2",
            "input": "카카오 쓸 때 할인 돼요?",
            "expected_payment": "카카오페이"
        },
    ]

    passed = 0
    for case in test_cases:
        result = extract_keywords(case["input"])

        if case["expected_payment"] in result.payments:
            print(f"[PASS] {case['id']}: PASS")
            print(f"   입력: {case['input']}")
            print(f"   추출: {result.payments}")
            passed += 1
        else:
            print(f"[FAIL] {case['id']}: FAIL")
            print(f"   입력: {case['input']}")
            print(f"   기대: [{case['expected_payment']}]")
            print(f"   실제: {result.payments}")

    print(f"\n통과: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def main():
    """Phase 2 전체 검증"""
    print("\n" + "=" * 60)
    print("Phase 2 검증 테스트 시작")
    print("=" * 60)
    print("목표: 56.5% → 76.1% (26/46 → 35/46)")
    print()

    results = {}

    # Task 2.1 - 액션 패턴
    results['task_2_1_actions'] = test_task_2_1_action_patterns_expanded()

    # Task 2.1 - 의도 패턴
    results['task_2_1_intents'] = test_task_2_1_intent_patterns_added()

    # Task 2.2
    results['task_2_2'] = test_task_2_2_correction_dict_cleanup()

    # Task 2.3
    results['task_2_3'] = test_task_2_3_payment_synonyms_expanded()

    # 최종 결과
    print("\n" + "=" * 60)
    print("Phase 2 검증 결과")
    print("=" * 60)

    passed_tasks = sum(1 for v in results.values() if v)
    total_tasks = len(results)

    for task, passed in results.items():
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        print(f"{task}: {status}")

    print(f"\n총 통과: {passed_tasks}/{total_tasks} Task")

    if passed_tasks == total_tasks:
        print("\n[SUCCESS] Phase 2 검증 완료! 모든 개선사항이 정상 작동합니다.")
        print("다음 단계: 전체 46개 테스트 실행하여 통과율 확인")
    else:
        print(f"\n[WARNING] {total_tasks - passed_tasks}개 Task 실패. 코드 재검토 필요.")

    return passed_tasks == total_tasks


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
