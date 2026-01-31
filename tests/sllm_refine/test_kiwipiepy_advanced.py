"""
Kiwipiepy 고급 기능 테스트

테스트 항목:
1. 오타 교정 기능
2. 기분석 형태 등록
3. 통합 파이프라인
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.llm.delivery.morphology_analyzer import (
    analyze_morphemes,
    extract_nouns,
    get_kiwi
)


def test_typo_correction():
    """오타 교정 기능 테스트"""
    print("=" * 70)
    print("오타 교정 기능 테스트")
    print("=" * 70)
    
    test_cases = [
        ("하나낸 계좌에서", ["하나은행", "계좌"]),
        ("연예비 납부", ["연회비", "납부"]),
        ("바우저 신청", ["바우처", "신청"]),
        ("발송소리가 될까요", ["발송", "처리"]),
        ("이길영업일날", ["익일", "영업일", "날"]),
    ]
    
    for text, expected_keywords in test_cases:
        print(f"\n입력: {text}")
        
        # 형태소 분석
        morphemes = analyze_morphemes(text)
        print(f"형태소: {morphemes[:5]}...")
        
        # 명사 추출
        nouns = extract_nouns(text)
        print(f"명사: {nouns}")
        
        # 검증
        success = all(keyword in nouns for keyword in expected_keywords)
        if success:
            print("✅ 교정 성공")
        else:
            print(f"⚠️ 교정 실패 - 기대값: {expected_keywords}")
        
        print("-" * 70)


def test_pre_analyzed_words():
    """기분석 형태 등록 테스트"""
    print("\n" + "=" * 70)
    print("기분석 형태 등록 테스트")
    print("=" * 70)
    
    kiwi = get_kiwi()
    if kiwi is None:
        print("❌ Kiwi 초기화 실패")
        return
    
    # 등록된 패턴 테스트
    test_cases = [
        "하나낸",
        "연예비",
        "바우저",
        "발송소리가",
        "이길영업일"
    ]
    
    for text in test_cases:
        print(f"\n입력: {text}")
        tokens = kiwi.tokenize(text)
        print(f"분석 결과: {[(t.form, t.tag) for t in tokens]}")
        print("-" * 70)


def test_integrated_pipeline():
    """통합 파이프라인 테스트"""
    print("\n" + "=" * 70)
    print("통합 파이프라인 테스트")
    print("=" * 70)
    
    test_cases = [
        "하나낸 계좌에서 먼저 출금할까요",
        "연예비 납부와 그 바우저 한개 선택",
        "이길영업일날 문자로 발송소리가 될것같아요",
        "잠시만 기다려 주시겠습니까",
    ]
    
    for text in test_cases:
        print(f"\n입력: {text}")
        
        # 형태소 분석
        morphemes = analyze_morphemes(text)
        print(f"형태소 수: {len(morphemes)}")
        
        # 명사 추출
        nouns = extract_nouns(text)
        print(f"명사: {nouns}")
        
        # 주요 교정 확인
        corrections = []
        if "하나은행" in nouns:
            corrections.append("하나낸 → 하나은행")
        if "연회비" in nouns:
            corrections.append("연예비 → 연회비")
        if "바우처" in nouns:
            corrections.append("바우저 → 바우처")
        if "익일" in nouns:
            corrections.append("이길 → 익일")
        
        if corrections:
            print(f"교정: {', '.join(corrections)}")
            print("✅ 교정 성공")
        else:
            print("⚠️ 교정 없음")
        
        print("-" * 70)


def main():
    """메인 함수"""
    print("\n🚀 Kiwipiepy 고급 기능 테스트 시작\n")
    
    # 1. 오타 교정 테스트
    test_typo_correction()
    
    # 2. 기분석 형태 테스트
    test_pre_analyzed_words()
    
    # 3. 통합 파이프라인 테스트
    test_integrated_pipeline()
    
    print("\n" + "=" * 70)
    print("✅ 모든 테스트 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
