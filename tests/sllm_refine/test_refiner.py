"""
LLM 기반 텍스트 정제 및 RAG 검색 테스트 스크립트
"""

import os
import time
import asyncio
import sys
from pathlib import Path

# 테스트 데이터 import
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from test_data.noisy_utterances import get_test_dataset

from app.llm.sllm_refiner import refine_text
from app.rag.pipeline import run_rag, RAGConfig


def run_automated_test():
    """자동화된 테스트 실행"""
    print("=" * 70)
    print("LLM 기반 텍스트 정제 자동화 테스트")
    print("=" * 70)
    
    # 테스트 데이터 로드
    test_data = get_test_dataset()
    print(f"[테스트] 총 {len(test_data)}개 케이스 실행\n")
    print("=" * 70)
    
    # 통계
    total_cases = len(test_data)
    passed_cases = 0
    failed_cases = 0
    total_time = 0
    
    results = []
    
    for i, (original, noisy, expected_keywords) in enumerate(test_data, 1):
        print(f"\n[{i}/{total_cases}] 테스트 중...")
        print(f"원본:   {original}")
        print(f"입력:   {noisy}")
        
        # 텍스트 정제
        start = time.time()
        result = refine_text(noisy)
        elapsed = time.time() - start
        total_time += elapsed
        
        refined = result['text']
        keywords = result['keywords']
        
        print(f"정제:   {refined}")
        print(f"키워드: {', '.join(keywords) if keywords else '(없음)'}")
        print(f"기대:   {', '.join(['#' + k for k in expected_keywords]) if expected_keywords else '(없음)'}")
        print(f"시간:   {elapsed*1000:.1f}ms")
        
        # 검증 (키워드 매칭)
        if expected_keywords:
            # 기대 키워드가 추출된 키워드에 포함되는지 확인
            extracted_kw_set = set(kw.lstrip('#') for kw in keywords)
            expected_kw_set = set(expected_keywords)
            
            # 최소 1개 이상 매칭되면 통과
            matched = len(extracted_kw_set & expected_kw_set)
            if matched > 0:
                print(f"✅ PASS (매칭: {matched}/{len(expected_keywords)})")
                passed_cases += 1
            else:
                print(f"❌ FAIL (매칭: 0/{len(expected_keywords)})")
                failed_cases += 1
        else:
            # 기대 키워드가 없으면 무조건 통과
            print(f"⚪ SKIP (기대 키워드 없음)")
            passed_cases += 1
        
        results.append({
            'original': original,
            'noisy': noisy,
            'refined': refined,
            'keywords': keywords,
            'expected': expected_keywords,
            'time': elapsed
        })
        
        print("-" * 70)
    
    # 최종 통계
    print("\n" + "=" * 70)
    print("테스트 결과 요약 (LLM 기반)")
    print("=" * 70)
    print(f"총 케이스:     {total_cases}개")
    print(f"통과:          {passed_cases}개 ({passed_cases/total_cases*100:.1f}%)")
    print(f"실패:          {failed_cases}개 ({failed_cases/total_cases*100:.1f}%)")
    print(f"평균 처리시간: {total_time/total_cases*1000:.1f}ms")
    print(f"총 소요시간:   {total_time:.2f}s")
    print("=" * 70)
    
    # 실패 케이스 상세
    if failed_cases > 0:
        print("\n[실패 케이스 상세]")
        for i, r in enumerate(results, 1):
            if r['expected']:
                extracted = set(kw.lstrip('#') for kw in r['keywords'])
                expected = set(r['expected'])
                if len(extracted & expected) == 0:
                    print(f"\n{i}. {r['noisy']}")
                    print(f"   정제: {r['refined']}")
                    print(f"   추출: {r['keywords']}")
                    print(f"   기대: {['#' + k for k in r['expected']]}")


def run_interactive_test():
    """대화형 테스트"""
    print("=" * 70)
    print("LLM 기반 텍스트 정제 및 RAG 검색 테스트 (대화형)")
    print("=" * 70)
    
    while True:
        user_input = input("\n입력 (종료: 종료/q): ").strip()
        
        # 종료 조건
        if user_input.lower() in ['종료', 'q', 'quit']:
            break
            
        if not user_input:
            continue
        
        s = time.time()
        
        # 1. 텍스트 교정 및 키워드 추출
        result = refine_text(user_input)
        refined_text = result['text']
        keywords = result['keywords']
        
        # # 2. RAG 검색 쿼리 결정
        # # 키워드가 있으면 키워드로 검색, 없으면 교정된 문장으로 검색
        # if keywords:
        #     search_query = " ".join(kw.lstrip("#") for kw in keywords)
        # else:
        #     search_query = refined_text
        
        # # 3. RAG 검색 실행
        # config = RAGConfig(top_k=5)
        # rag_result = asyncio.run(run_rag(search_query, config))
        
        e = time.time()
        
        # 4. 결과 출력
        print(f"원본: {user_input}")
        print(f"교정: {refined_text}")
        print(f"키워드: {', '.join(keywords) if keywords else '(없음)'}")
        # print(f"검색쿼리: {search_query}")
        
        # RAG 결과 출력 (주석 처리됨)
        # print(f"\n[RAG 검색 결과]")
        # ...
        
        print(f"\n소요시간: {e-s:.2f}s")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        run_automated_test()
    else:
        run_interactive_test()


if __name__ == "__main__":
    main()
