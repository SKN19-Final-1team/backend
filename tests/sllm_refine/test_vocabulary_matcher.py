"""
Vocabulary Matcher 테스트 스크립트
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트 경로 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.llm.delivery.vocabulary_matcher import (
    load_card_products,
    find_candidates,
    get_best_match,
    phonetic_similarity
)


def test_phonetic_similarity():
    """발음 유사도 테스트"""
    print("=" * 70)
    print("발음 유사도 테스트")
    print("=" * 70)
    
    test_cases = [
        ("테디카드", "테니카드", "높음"),
        ("서울시", "서울", "높음"),
        ("청년인생", "청년라이프", "중간"),
        ("담보대출", "단보대출", "높음"),
        ("나라사랑카드", "나라 사랑 카드", "높음"),
    ]
    
    for text1, text2, expected in test_cases:
        score = phonetic_similarity(text1, text2)
        print(f"{text1:15} vs {text2:15} → {score:.3f} (예상: {expected})")
    print()


def test_candidate_finding():
    """후보 추출 테스트"""
    print("=" * 70)
    print("후보 추출 테스트")
    print("=" * 70)
    
    # DB 로드
    products = load_card_products()
    print(f"로드된 카드상품: {len(products)}개\n")
    
    test_queries = [
        "테니카드 서울 청년라이프",
        "테디카드 청렴 주택 단보 대출",
        "나라 사랑 카드",
        "테디 카드",
    ]
    
    for query in test_queries:
        print(f"쿼리: {query}")
        start = time.time()
        candidates = find_candidates(query, top_k=3, threshold=0.5)
        elapsed = (time.time() - start) * 1000
        
        if candidates:
            for i, (name, score) in enumerate(candidates, 1):
                print(f"  {i}. {name:40} (유사도: {score:.3f})")
        else:
            print("  매칭 결과 없음")
        
        print(f"  처리시간: {elapsed:.1f}ms")
        print()


def test_best_match():
    """최적 매칭 테스트 (고확신도)"""
    print("=" * 70)
    print("최적 매칭 테스트 (확신도 >= 0.85)")
    print("=" * 70)
    
    test_queries = [
        "테니카드 서울 청년라이프",
        "나라사랑카드",
        "테디카드",
    ]
    
    for query in test_queries:
        print(f"쿼리: {query}")
        start = time.time()
        result = get_best_match(query, confidence_threshold=0.85)
        elapsed = (time.time() - start) * 1000
        
        if result:
            print(f"  ✓ 즉시 매칭: {result}")
            print(f"  → sLLM 호출 불필요 (latency 최소화)")
        else:
            print(f"  ✗ 확신도 낮음 → sLLM 검증 필요")
        
        print(f"  처리시간: {elapsed:.1f}ms")
        print()


def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Vocabulary Matcher 테스트" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        test_phonetic_similarity()
        test_candidate_finding()
        test_best_match()
        
        print("=" * 70)
        print("✓ 모든 테스트 완료")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
