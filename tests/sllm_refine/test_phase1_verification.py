"""
Phase 1 개선사항 검증 테스트

Task 1.1: 중복 보정 버그 수정
Task 1.2: 결제수단 약칭 확장
Task 1.3: 구어체 매핑 5개 패턴
"""

import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.llm.delivery.keyword_extractor import extract_keywords


def test_task_1_1_duplicate_correction_bug():
    """Task 1.1: 중복 보정 버그 수정 검증"""
    print("\n" + "=" * 60)
    print("Task 1.1: 중복 보정 버그 수정 검증")
    print("=" * 60)

    test_cases = [
        {
            "id": "CARD-004",
            "input": "국민행복카드 신청하려구요",
            "expected_card": "국민행복카드",
            "should_not_contain": "국민행복카드카드"
        },
        {
            "id": "COMP-001",
            "input": "나라사람카드 연예비가 얼마예요",
            "expected_correction": "나라사랑카드",
            "should_not_contain": "나라사랑카드카드"
        }
    ]

    passed = 0
    for case in test_cases:
        result = extract_keywords(case["input"])

        # 중복 보정 체크
        has_duplicate = case["should_not_contain"] in result.corrected_text

        if not has_duplicate:
            print(f"✅ {case['id']}: PASS")
            print(f"   입력: {case['input']}")
            print(f"   보정: {result.corrected_text}")
            print(f"   중복 없음: '{case['should_not_contain']}' 발견 안 됨")
            passed += 1
        else:
            print(f"❌ {case['id']}: FAIL")
            print(f"   입력: {case['input']}")
            print(f"   보정: {result.corrected_text}")
            print(f"   중복 발견: '{case['should_not_contain']}'")

    print(f"\n통과: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_task_1_2_payment_abbreviation():
    """Task 1.2: 결제수단 약칭 확장 검증"""
    print("\n" + "=" * 60)
    print("Task 1.2: 결제수단 약칭 확장 검증")
    print("=" * 60)

    test_cases = [
        {
            "id": "PAY-003",
            "input": "네이버 쓸 때 포인트 쌓여요?",
            "expected_payment": "네이버페이"
        },
        {
            "id": "PAY-005",
            "input": "카카오에서 결제했는데 취소하고싶어요",
            "expected_payment": "카카오페이"
        },
        {
            "id": "PAY-EXTRA-1",
            "input": "삼성페이 등록하려고요",
            "expected_payment": "삼성페이"
        },
        {
            "id": "PAY-EXTRA-2",
            "input": "애플 결제 안돼요",
            "expected_payment": "애플페이"
        }
    ]

    passed = 0
    for case in test_cases:
        result = extract_keywords(case["input"])

        if case["expected_payment"] in result.payments:
            print(f"✅ {case['id']}: PASS")
            print(f"   입력: {case['input']}")
            print(f"   추출: {result.payments}")
            passed += 1
        else:
            print(f"❌ {case['id']}: FAIL")
            print(f"   입력: {case['input']}")
            print(f"   기대: [{case['expected_payment']}]")
            print(f"   실제: {result.payments}")

    print(f"\n통과: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_task_1_3_colloquial_mapping():
    """Task 1.3: 구어체 매핑 검증"""
    print("\n" + "=" * 60)
    print("Task 1.3: 구어체 매핑 검증")
    print("=" * 60)

    test_cases = [
        {
            "id": "ACT-001",
            "input": "카드 잃어버렸어요 빨리 막아주세요",
            "expected_actions": ["분실", "분실신고"]  # 둘 중 하나라도 있으면 통과
        },
        {
            "id": "ACT-003",
            "input": "그냥 안 쓸래요 없애주세요",
            "expected_actions": ["해지"]
        },
        {
            "id": "ACT-005",
            "input": "카드 결제가 왜 자꾸 튕겨요",
            "expected_actions": ["오류"]
        },
        {
            "id": "ACT-EXTRA-1",
            "input": "새 카드 넣으려고 하는데요",
            "expected_actions": ["등록"]
        }
    ]

    passed = 0
    for case in test_cases:
        result = extract_keywords(case["input"])

        # expected_actions 중 하나라도 있으면 통과
        found = any(action in result.actions for action in case["expected_actions"])

        if found:
            print(f"✅ {case['id']}: PASS")
            print(f"   입력: {case['input']}")
            print(f"   추출: {result.actions}")
            passed += 1
        else:
            print(f"❌ {case['id']}: FAIL")
            print(f"   입력: {case['input']}")
            print(f"   기대: {case['expected_actions']} 중 하나")
            print(f"   실제: {result.actions}")

    print(f"\n통과: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def main():
    """Phase 1 전체 검증"""
    print("\n" + "=" * 60)
    print("Phase 1 검증 테스트 시작")
    print("=" * 60)
    print("목표: 39.1% → 56.5% (18/46 → 26/46)")
    print()

    results = {}

    # Task 1.1
    results['task_1_1'] = test_task_1_1_duplicate_correction_bug()

    # Task 1.2
    results['task_1_2'] = test_task_1_2_payment_abbreviation()

    # Task 1.3
    results['task_1_3'] = test_task_1_3_colloquial_mapping()

    # 최종 결과
    print("\n" + "=" * 60)
    print("Phase 1 검증 결과")
    print("=" * 60)

    passed_tasks = sum(1 for v in results.values() if v)
    total_tasks = len(results)

    for task, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{task}: {status}")

    print(f"\n총 통과: {passed_tasks}/{total_tasks} Task")

    if passed_tasks == total_tasks:
        print("\n🎉 Phase 1 검증 완료! 모든 개선사항이 정상 작동합니다.")
        print("다음 단계: 전체 46개 테스트 실행하여 통과율 확인")
    else:
        print(f"\n⚠️ {total_tasks - passed_tasks}개 Task 실패. 코드 재검토 필요.")

    return passed_tasks == total_tasks


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
