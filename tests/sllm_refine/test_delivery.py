"""
LLM 기반 텍스트 정제 및 마스킹 통합 테스트 스크립트
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트 경로 설정 (절대 경로 사용)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.llm.delivery.deliverer import deliver

def run_test():
    print("=" * 70)
    print("LLM 기반 텍스트 정제 및 마스킹 통합 테스트")
    print("=" * 70)
    
    test_cases = [
        "제 이름은 홍길동이구요, 전화번호는 010",
        "1234에, 5678이요.",
        "카드번호 1234에",
        "5678, 9012, 3456입니다.",
        "계좌번호는 신한 110-123-456789 입니다.",
        "안녕 하세요 반갑 습니다."
    ]
    
    print(f"[테스트] 총 {len(test_cases)}개 케이스 실행\n")
    print("=" * 70)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] 테스트 중...")
        print(f"입력: {text}")
        
        start = time.time()
        result = deliver(text)
        elapsed = time.time() - start
        
        print(f"교정: {result['refined']}")
        print(f"마스킹: {result['masked']}")
        print(f"감지됨: {result['detected_info']}")
        print(f"Time: {elapsed*1000:.0f}ms")

if __name__ == "__main__":
    run_test()
