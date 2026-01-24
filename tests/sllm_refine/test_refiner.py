"""
LLM 기반 텍스트 정제 및 RAG 검색 테스트 스크립트
"""

import os
import time
import asyncio
import sys
from pathlib import Path

from tests.test_data.noisy_utterances import get_test_dataset

from app.llm.sllm_refiner import refine_text
# from app.rag.pipeline import run_rag, RAGConfig


def run_test():
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
        
        # 텍스트 정제
        start = time.time()
        result = refine_text(noisy)
        elapsed = time.time() - start
        total_time += elapsed
        
        print(result)
        print(f"Time     : {elapsed*1000:.0f}ms")

def main():
    run_test()

if __name__ == "__main__":
    main()
