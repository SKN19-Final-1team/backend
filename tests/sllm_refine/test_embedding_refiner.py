"""
임베딩 기반 텍스트 정제 자동화 테스트 및 모델 비교
"""

import time
import sys
from pathlib import Path
import numpy as np

# 테스트 데이터 import
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from test_data.noisy_utterances import get_test_dataset

from app.llm.sllm_refiner_embed import refine_text_with_embedding

# 비교할 모델 리스트
CANDIDATE_MODELS = [
    "jhgan/ko-sroberta-multitask",       # 빠름, 기본
    "BM-K/KoSimCSE-roberta-multitask",   # 성능 우수
    "jhgan/ko-sbert-nli",                # NLI 특화
]


def run_automated_test(model_name: str = "jhgan/ko-sbert-nli", show_details: bool = True):
    """자동화된 테스트 실행"""
    if show_details:
        print("=" * 70)
        print(f"임베딩 기반 텍스트 정제 테스트 (모델: {model_name})")
        print("=" * 70)
    
    # 초기화
    if show_details:
        print(f"\n[초기화] 모델 로딩 및 임베딩 생성 중... ({model_name})")
    
    start_init = time.time()
    # 웜업 (모델 로드 및 캐시 init)
    _ = refine_text_with_embedding("테스트", threshold=0.65, model_name=model_name)
    init_time = time.time() - start_init
    
    if show_details:
        print(f"✅ 초기화 완료! (소요시간: {init_time:.2f}s)\n")
    
    # 테스트 데이터 로드
    test_data = get_test_dataset()
    if show_details:
        print(f"[테스트] 총 {len(test_data)}개 케이스 실행\n")
        print("=" * 70)
    
    # 통계
    total_cases = len(test_data)
    passed_cases = 0
    failed_cases = 0
    total_time = 0
    
    results = []
    
    filter_conf = 0.65  # 임계값 통일
    
    for i, (original, noisy, expected_keywords) in enumerate(test_data, 1):
        if show_details:
            print(f"\n[{i}/{total_cases}] 테스트 중...")
            print(f"원본:   {original}")
            print(f"입력:   {noisy}")
        
        # 텍스트 정제
        start = time.time()
        # 모델명을 명시적으로 전달
        result = refine_text_with_embedding(noisy, threshold=filter_conf, model_name=model_name)
        elapsed = time.time() - start
        total_time += elapsed
        
        refined = result['text']
        keywords = result['keywords']
        
        passed = False
        
        # 검증 (키워드 매칭)
        if expected_keywords:
            extracted_kw_set = set(kw.lstrip('#') for kw in keywords)
            expected_kw_set = set(expected_keywords)
            
            matched = len(extracted_kw_set & expected_kw_set)
            if matched > 0:
                if show_details:
                    print(f"✅ PASS (매칭: {matched}/{len(expected_keywords)})")
                passed = True
            else:
                if show_details:
                    print(f"❌ FAIL (매칭: 0/{len(expected_keywords)})")
                passed = False
        else:
            if show_details:
                print(f"⚪ SKIP (기대 키워드 없음)")
            passed = True
        
        if passed:
            passed_cases += 1
        else:
            failed_cases += 1
            
        if show_details:
            print(f"정제:   {refined}")
            print(f"키워드: {', '.join(keywords) if keywords else '(없음)'}")
            print(f"시간:   {elapsed*1000:.1f}ms")
            print("-" * 70)
            
        results.append({
            'original': original,
            'noisy': noisy,
            'refined': refined,
            'keywords': keywords,
            'expected': expected_keywords,
            'passed': passed,
            'time': elapsed
        })
    
    # 통계 계산
    accuracy = (passed_cases / total_cases) * 100
    avg_time = (total_time / total_cases) * 1000
    
    if show_details:
        print("\n" + "=" * 70)
        print(f"테스트 결과 요약 ({model_name})")
        print("=" * 70)
        print(f"총 케이스:     {total_cases}개")
        print(f"통과:          {passed_cases}개 ({accuracy:.1f}%)")
        print(f"실패:          {failed_cases}개 ({failed_cases/total_cases*100:.1f}%)")
        print(f"평균 처리시간: {avg_time:.1f}ms")
        print(f"총 소요시간:   {total_time:.2f}s")
        print("=" * 70)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'avg_time': avg_time,
        'passed': passed_cases,
        'failed': failed_cases,
        'total_time': total_time,
        'init_time': init_time,
        'results': results
    }


def run_model_comparison():
    """여러 모델 비교 실행"""
    print("\n" + "=" * 80)
    print(f"🚀 임베딩 모델 성능 비교 ({len(CANDIDATE_MODELS)}개 모델)")
    print("=" * 80)
    print(f"대상 모델: {', '.join(CANDIDATE_MODELS)}")
    
    comparison_results = []
    
    for i, model in enumerate(CANDIDATE_MODELS, 1):
        print(f"\n\n[{i}/{len(CANDIDATE_MODELS)}] 모델 평가 중: {model}")
        print("-" * 40)
        # 상세 로그는 끄고 결과만 수집
        result = run_automated_test(model_name=model, show_details=True)
        comparison_results.append(result)
    
    # 최종 비교 테이블 출력
    print("\n\n" + "=" * 100)
    print(f"{'Rank':<5} {'Model Name':<40} {'Accuracy':<10} {'Avg Time':<10} {'Init Time':<10}")
    print("-" * 100)
    
    # 정확도 내림차순 정렬
    comparison_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    for rank, res in enumerate(comparison_results, 1):
        print(f"{rank:<5} {res['model']:<40} {res['accuracy']:.1f}%     {res['avg_time']:.1f}ms    {res['init_time']:.1f}s")
    print("=" * 100)
    
    # 최고 모델 추천
    best_model = comparison_results[0]
    print(f"\n🏆 최고 성능 모델: {best_model['model']} (정확도: {best_model['accuracy']:.1f}%)")
    
    return comparison_results


def run_interactive_test():
    """대화형 테스트"""
    print("=" * 70)
    print("대화형 테스트 모드")
    print("=" * 70)
    
    # 모델 선택
    print("사용할 모델을 선택하세요:")
    for i, model in enumerate(CANDIDATE_MODELS, 1):
        print(f"{i}. {model}")
    
    try:
        choice = int(input("\n선택 (1~3, 기본 1): ") or 1)
        model_name = CANDIDATE_MODELS[choice-1]
    except:
        model_name = CANDIDATE_MODELS[0]
    
    print(f"\n선택된 모델: {model_name}")
    print("초기화 중...")
    _ = refine_text_with_embedding("테스트", threshold=0.65, model_name=model_name)
    print("✅ 초기화 완료!\n")
    
    while True:
        try:
            user_input = input("\n입력 (종료: q): ").strip()
            
            if user_input.lower() in ['q', 'quit', '종료']:
                break
            
            if not user_input:
                continue
            
            start = time.time()
            result = refine_text_with_embedding(user_input, threshold=0.65, model_name=model_name)
            elapsed = time.time() - start
            
            print(f"\n{'='*70}")
            print(f"원본:   {user_input}")
            print(f"정제:   {result['text']}")
            print(f"키워드: {', '.join(result['keywords']) if result['keywords'] else '(없음)'}")
            print(f"시간:   {elapsed*1000:.1f}ms")
            print(f"{'='*70}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n오류 발생: {e}")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--compare':
            run_model_comparison()
        elif sys.argv[1] == '--auto':
            run_automated_test()
        else:
            run_interactive_test()
    else:
        # 인자 없으면 메뉴 표시
        print("1. 모델 비교 실행 (--compare)")
        print("2. 단일 모델 자동 테스트 (--auto)")
        print("3. 대화형 테스트 (기본)")
        
        choice = input("\n선택 (1~3): ").strip()
        
        if choice == '1':
            run_model_comparison()
        elif choice == '2':
            run_automated_test()
        else:
            run_interactive_test()


if __name__ == "__main__":
    main()
