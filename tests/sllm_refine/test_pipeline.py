"""
통합 파이프라인 테스트

correction_map + sLLM 전체 파이프라인 테스트

사용법:
    C:\\Users\\bsjun\\anaconda3\\envs\\final_env\\python.exe tests/sllm_refine/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.llm.delivery.deliverer import pipeline


TEST_CASES = [
    {
        "input": "하나낸 계좌에서 먼저 출금할까요",
        "expected": "하나은행 계좌에서 먼저 출금할까요",
    },
    {
        "input": "연예비 납부와 그 바우저 한개 선택",
        "expected": "연회비 납부와 그 바우처 한개 선택",
    },
    {
        "input": "결채 금액이 얼마인가요",
        "expected": "결제 금액이 얼마인가요",
    },
    {
        "input": "발송소리가 될것같아요",
        "expected": "발송처리가 될것같아요",
    },
    {
        "input": "이길 영업일에 처리됩니다",
        "expected": "익일 영업일에 처리됩니다",
    },
]


def test_pipeline_without_sllm():
    """correction_map만 테스트 (sLLM 제외)"""
    print("=" * 70)
    print("correction_map 단독 테스트 (sLLM 미사용)")
    print("=" * 70)
    
    for case in TEST_CASES:
        input_text = case["input"]
        expected = case["expected"]
        
        result = pipeline(input_text, use_sllm=False)
        
        print(f"\n[입력] {input_text}")
        print(f"[교정] {result['step1_corrected']}")
        print(f"[기대] {expected}")
        
        if result['step1_corrected'] == expected:
            print("✅ 완전 일치")
        elif result['step1_corrected'] != input_text:
            print("⚠️ 부분 교정")
        else:
            print("❌ 교정 없음")
        
        print("-" * 70)


def test_pipeline_with_sllm():
    """correction_map + sLLM 전체 파이프라인 테스트"""
    print("\n" + "=" * 70)
    print("전체 파이프라인 테스트 (correction_map + sLLM)")
    print("=" * 70)
    
    for case in TEST_CASES:
        input_text = case["input"]
        expected = case["expected"]
        
        result = pipeline(input_text, use_sllm=True)
        
        print(f"\n[입력]        {input_text}")
        print(f"[Step1 교정]  {result['step1_corrected']}")
        print(f"[최종 결과]   {result['refined']}")
        print(f"[기대값]      {expected}")
        
        if result['refined'] == expected:
            print("✅ 완전 일치")
        elif result['refined'] != input_text:
            print("⚠️ 부분 교정")
        else:
            print("❌ 교정 없음")
        
        print("-" * 70)


def main():
    print("\n🚀 통합 파이프라인 테스트 시작\n")
    
    # sLLM 포함 여부
    include_sllm = input("sLLM 포함 테스트? (y/n): ").strip().lower() == 'y'
    
    if include_sllm:
        test_pipeline_with_sllm()
    else:
        test_pipeline_without_sllm()
    
    print("\n✅ 테스트 완료")


if __name__ == "__main__":
    main()
