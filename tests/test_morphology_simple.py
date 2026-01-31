"""
Kiwipiepy + PyKoSpacing 기본 동작 테스트

RunPod 없이 형태소 분석 및 띄어쓰기 교정만 테스트
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.delivery.morphology_analyzer import (
    analyze_morphemes,
    extract_nouns,
    extract_card_product_candidates,
    get_user_dict_stats
)


def test_morphology():
    """형태소 분석 기본 테스트"""
    
    print("=" * 70)
    print("Kiwipiepy + PyKoSpacing 기본 동작 테스트")
    print("=" * 70)
    
    # 사용자 사전 통계
    print("\n[사용자 사전 통계]")
    stats = get_user_dict_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    
    test_cases = [
        "나라사람카드바우처신청",
        "연예비 납부와 그 바우저 한개 선택",
        "이길영업일날문자로발송소리가될것같고요",
        "신세계상품권 등록 좀 해주세요",
        "상담원 안수희입니다 나라사람카드 바우처 때문에"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}]")
        print(f"입력: {text}")
        
        try:
            # 형태소 분석
            morphemes = analyze_morphemes(text)
            print(f"형태소 분석 ({len(morphemes)}개):")
            print(f"  {morphemes[:10]}...")  # 처음 10개만
            
            # 명사 추출
            nouns = extract_nouns(text)
            print(f"명사 추출 ({len(nouns)}개): {nouns}")
            
            # 카드상품명 후보
            candidates = extract_card_product_candidates(text)
            print(f"카드상품명 후보: {candidates}")
            
            print("  ✅ 성공")
            
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ 기본 동작 테스트 완료")
    print("=" * 70)


if __name__ == "__main__":
    test_morphology()
