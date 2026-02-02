"""
Phase 3: 카드명 정규화 테스트
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

from app.llm.delivery.vocabulary_matcher import normalize_card_name, load_card_products


def test_normalize_card_name():
    """카드명 정규화 함수 테스트"""
    print("\n" + "=" * 60)
    print("카드명 정규화 테스트")
    print("=" * 60)

    test_cases = [
        ("#Pay 테디카드", "테디카드"),
        ("나라사랑 카드", "나라사랑카드"),
        ("AK PLAZA 테디카드 Plus", "테디카드"),
        ("국민행복카드 신청", "국민행복카드"),
        ("그린 카드", "그린카드"),
        ("알뜰교통카드 체크", "알뜰교통카드"),
        ("11번가 팝 테디카드 체크", "테디카드"),
        ("테디카드 나라사랑카드 체크", "나라사랑카드"),  # 나라사랑카드가 우선
    ]

    passed = 0
    for input_name, expected in test_cases:
        result = normalize_card_name(input_name)
        if result == expected:
            print(f"[PASS] '{input_name}' -> '{result}'")
            passed += 1
        else:
            print(f"[FAIL] '{input_name}' -> '{result}' (기대: '{expected}')")

    print(f"\n통과: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_load_card_products():
    """card_products 테이블 로드 테스트"""
    print("\n" + "=" * 60)
    print("card_products 테이블 로드 테스트")
    print("=" * 60)

    products = load_card_products(force_reload=True)

    if not products:
        print("[FAIL] 카드상품 로드 실패")
        return False

    print(f"[PASS] 카드상품 {len(products)}개 로드 성공")

    # 샘플 출력 (처음 5개)
    print("\n샘플 카드명 (처음 5개):")
    for i, product in enumerate(products[:5], 1):
        print(f"{i}. {product['name']} -> {product['normalized_name']}")

    # 특정 카드명 확인
    print("\n특정 카드명 확인:")
    target_names = ["테디카드", "나라사랑카드", "그린카드", "알뜰교통카드", "국민행복카드"]
    found_count = 0

    for target in target_names:
        found = any(
            product["normalized_name"] == target
            for product in products
        )
        if found:
            print(f"[PASS] '{target}' 발견")
            found_count += 1
        else:
            print(f"[FAIL] '{target}' 미발견")

    print(f"\n통과: {found_count}/{len(target_names)}")
    return found_count == len(target_names)


def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 60)
    print("Phase 3: 카드명 정규화 검증 테스트")
    print("=" * 60)

    results = {}

    # 테스트 1: 정규화 함수
    results['normalize'] = test_normalize_card_name()

    # 테스트 2: 카드상품 로드
    results['load'] = test_load_card_products()

    # 최종 결과
    print("\n" + "=" * 60)
    print("Phase 3 검증 결과")
    print("=" * 60)

    passed_tests = sum(1 for v in results.values() if v)
    total_tests = len(results)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name}: {status}")

    print(f"\n총 통과: {passed_tests}/{total_tests} Test")

    if passed_tests == total_tests:
        print("\n[SUCCESS] Phase 3 카드명 정규화 검증 완료!")
        print("다음 단계: CARD_NAME 테스트 실행")
    else:
        print(f"\n[WARNING] {total_tests - passed_tests}개 Test 실패. 코드 재검토 필요.")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
