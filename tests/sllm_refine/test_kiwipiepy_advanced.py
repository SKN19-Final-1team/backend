"""
Kiwipiepy 고급 기능 테스트

테스트 항목:
1. 텍스트 레벨 교정
2. 통합 파이프라인
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.llm.delivery.morphology_analyzer import (
    analyze_morphemes,
    extract_nouns,
    apply_text_corrections,
    get_correction_map
)


def test_text_corrections():
    """텍스트 레벨 교정 테스트"""
    print("=" * 70)
    print("텍스트 레벨 교정 테스트")
    print("=" * 70)
    
    test_cases = [
        "하나낸 계좌에서 출금",
        "연예비 납부와 그 바우저 한개 선택",
        "결채 금액이 얼마인가요",
        "이길 영업일에 처리됩니다",
        "채크카드로 할부 가능한가요",
    ]
    
    for text in test_cases:
        print(f"\n입력: {text}")
        
        # 텍스트 레벨 교정
        corrected = apply_text_corrections(text)
        print(f"교정: {corrected}")
        
        # 변경 여부
        if text != corrected:
            print("✅ 교정됨")
        else:
            print("⚠️ 변경 없음")
        
        print("-" * 70)


def test_integrated_pipeline():
    """통합 파이프라인 테스트"""
    print("\n" + "=" * 70)
    print("통합 파이프라인 테스트 (입력 → 교정 → 명사)")
    print("=" * 70)
    
    test_cases = [
        "하나낸 계좌에서 먼저 출금할까요",
        "연예비 납부와 그 바우저 한개 선택",
        "이길 영업일에 발송소리가 될것같아요",
        "잠시만 기다려 주시겠습니까",
        "결채 금액 할부로 바꿔주세요",
    ]
    
    for text in test_cases:
        print(f"\n[입력] {text}")
        
        # 텍스트 레벨 교정
        corrected = apply_text_corrections(text)
        print(f"[교정] {corrected}")
        
        # 명사 추출
        nouns = extract_nouns(text)
        print(f"[명사] {nouns}")
        
        # 변경된 부분 강조
        if text != corrected:
            print("✅ 교정 성공")
        else:
            print("⚠️ 교정 없음")
        
        print("-" * 70)


def test_correction_map_stats():
    """교정 맵 통계"""
    print("\n" + "=" * 70)
    print("교정 맵 통계")
    print("=" * 70)
    
    correction_map = get_correction_map()
    
    total = len(correction_map)
    same_count = sum(1 for k, v in correction_map.items() if k == v)
    diff_count = total - same_count
    
    print(f"전체 항목: {total}개")
    print(f"동일 항목 (교정 불필요): {same_count}개")
    print(f"교정 필요 항목: {diff_count}개")
    
    print("\n[교정 패턴 예시 (처음 10개)]")
    count = 0
    for k, v in correction_map.items():
        if k != v:
            print(f"  {k} → {v}")
            count += 1
            if count >= 10:
                break
    
    print("-" * 70)


def main():
    """메인 함수"""
    print("\n🚀 STT 오류 교정 테스트 시작\n")
    
    # 1. 교정 맵 통계
    test_correction_map_stats()
    
    # 2. 텍스트 레벨 교정 테스트
    test_text_corrections()
    
    # 3. 통합 파이프라인 테스트
    test_integrated_pipeline()
    
    print("\n" + "=" * 70)
    print("✅ 모든 테스트 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
