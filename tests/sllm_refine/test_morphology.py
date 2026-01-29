"""
형태소 분석기 테스트 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.llm.delivery.morphology_analyzer import (
    analyze_morphemes,
    extract_nouns,
    extract_card_product_candidates,
    normalize_with_morphology,
    get_user_dict_stats
)


def test_user_dictionary():
    """사용자사전 생성 테스트"""
    print("=" * 70)
    print("사용자사전 생성 테스트")
    print("=" * 70)
    
    stats = get_user_dict_stats()
    
    if stats["exists"]:
        print(f"✓ 사용자사전 생성 완료")
        print(f"  경로: {stats['path']}")
        print(f"  등록 단어: {stats['entries']}개")
        print(f"  KOMORAN 로드: {stats['komoran_loaded']}")
    else:
        print("✗ 사용자사전 없음")
    print()


def test_morpheme_analysis():
    """형태소 분석 테스트"""
    print("=" * 70)
    print("형태소 분석 테스트")
    print("=" * 70)
    
    test_cases = [
        "테디카드로 결제해줘",
        "나라사랑카드 혜택이 뭐야",
        "테디카드 서울시 청년인생 설계학교 신청하고 싶어",
        "청년주택담보대출 금리가 궁금해",
    ]
    
    for text in test_cases:
        print(f"입력: {text}")
        morphemes = analyze_morphemes(text)
        print(f"형태소: {morphemes}")
        print()


def test_noun_extraction():
    """명사 추출 테스트"""
    print("=" * 70)
    print("명사 추출 테스트")
    print("=" * 70)
    
    test_cases = [
        "테디카드로 결제해줘",
        "나라사랑카드 혜택이 뭐야",
        "서울 청년 주택 대출 신청",
    ]
    
    for text in test_cases:
        nouns = extract_nouns(text)
        print(f"입력: {text}")
        print(f"명사: {nouns}")
        print()


def test_card_product_extraction():
    """카드상품명 후보 추출 테스트"""
    print("=" * 70)
    print("카드상품명 후보 추출 테스트")
    print("=" * 70)
    
    test_cases = [
        "테니카드 서울 청년라이프",
        "나라 사랑 카드 혜택",
        "테디카드로 결제",
        "청년주택담보대출 신청",
    ]
    
    for text in test_cases:
        candidates = extract_card_product_candidates(text)
        print(f"입력: {text}")
        print(f"후보: {candidates}")
        print()


def test_normalization():
    """형태소 기반 정규화 테스트"""
    print("=" * 70)
    print("형태소 기반 정규화 테스트")
    print("=" * 70)
    
    test_cases = [
        "테디카드로 결제해줘",
        "나라사랑카드의 혜택이 뭐야",
        "서울시 청년인생 설계학교에 신청하고 싶어",
    ]
    
    for text in test_cases:
        normalized = normalize_with_morphology(text)
        print(f"입력: {text}")
        print(f"정규화: {normalized}")
        print()


def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "형태소 분석기 테스트" + " " * 31 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        test_user_dictionary()
        test_morpheme_analysis()
        test_noun_extraction()
        test_card_product_extraction()
        test_normalization()
        
        print("=" * 70)
        print("✓ 모든 테스트 완료")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
