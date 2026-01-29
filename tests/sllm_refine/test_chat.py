"""
대화형 카드상품명 매칭 테스트

사용자 입력 → 형태소 분석 → 명사 추출 → 단어사전 매칭
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
    extract_card_product_candidates
)
from app.llm.delivery.vocabulary_matcher import (
    find_candidates,
    get_best_match
)


def print_header():
    """헤더 출력"""
    print("\n" + "=" * 70)
    print("카드상품명 매칭 테스트 (대화형)")
    print("=" * 70)
    print("사용자 입력 → 형태소 분석 → 명사 추출 → 단어사전 매칭")
    print("종료: 'exit', 'quit', 'q' 입력")
    print("=" * 70 + "\n")


def process_input(user_input: str):
    """사용자 입력 처리"""
    print(f"\n📝 입력: {user_input}")
    print("-" * 70)
    
    # 1. 형태소 분석
    print("\n[1단계] 형태소 분석")
    morphemes = analyze_morphemes(user_input)
    print(f"  결과: {morphemes[:5]}{'...' if len(morphemes) > 5 else ''}")  # 처음 5개만 표시
    
    # 2. 명사 추출
    print("\n[2단계] 명사 추출")
    nouns = extract_nouns(user_input)
    print(f"  명사: {nouns}")
    
    # 3. 카드상품명 후보 추출 (형태소 분석 기반)
    print("\n[3단계] 카드상품명 후보 추출 (형태소 분석)")
    card_candidates = extract_card_product_candidates(user_input)
    if card_candidates:
        print(f"  후보: {card_candidates}")
    else:
        print("  후보: (없음)")
    
    # 4. 단어사전 매칭 (발음 유사도 기반)
    print("\n[4단계] 단어사전 매칭 (발음 유사도 Top-3)")
    candidates = find_candidates(user_input, top_k=3, threshold=0.5)
    
    if candidates:
        for i, (name, score) in enumerate(candidates, 1):
            print(f"  {i}. {name:50} (유사도: {score:.3f})")
    else:
        print("  매칭 결과: (없음)")
    
    print("-" * 70)


def main():
    """메인 루프"""
    print_header()
    
    # 시스템 초기화 (사용자사전 로드)
    print("시스템 초기화 중...")
    analyze_morphemes("초기화")  # 내부적으로 KOMORAN 및 사용자사전 로드
    print("✓ 초기화 완료\n")
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input("\n💬 입력 > ").strip()
            
            # 종료 명령 확인
            if user_input.lower() in ['exit', 'quit', 'q', '종료']:
                print("\n프로그램을 종료합니다.")
                break
            
            # 빈 입력 무시
            if not user_input:
                continue
            
            # 입력 처리
            process_input(user_input)
            
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n[ERROR] 오류 발생: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
