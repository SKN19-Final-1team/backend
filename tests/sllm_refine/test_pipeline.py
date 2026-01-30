"""
통합 텍스트 교정 파이프라인 대화형 테스트

전체 흐름:
1. STT 전사 (사용자 입력)
2. 형태소 분석 (Targeting)
3. 단어 매칭 및 교정 (Vocabulary Matching)
4. 최종 문장 생성 (sLLM Refining)
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.llm.delivery.deliverer import pipeline


def print_header():
    """헤더 출력"""
    print("\n" + "=" * 70)
    print("텍스트 교정 파이프라인 테스트")
    print("=" * 70)
    print("종료: 'exit', 'quit', 'q' 입력")
    print("=" * 70 + "\n")


def process_text(user_input: str, use_sllm: bool = True):
    """텍스트 처리 및 결과 출력"""
    import json
    import time
    
    print(f"\n📝 입력: {user_input}")
    print("-" * 70)
    
    # 파이프라인 실행 (시간 측정)
    start_time = time.time()
    result = pipeline(user_input, use_sllm=use_sllm)
    elapsed_time = time.time() - start_time
    
    # JSON 형식 출력
    print("\n결과 (JSON):")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 응답시간 출력
    print(f"\n⏱️  응답시간: {elapsed_time:.2f}초 ({elapsed_time*1000:.0f}ms)")
    print("-" * 70)


def main():
    """메인 루프"""
    print_header()
    
    # 시스템 초기화
    print("시스템 초기화 중...")
    try:
        # 더미 호출로 모듈 로드
        pipeline("초기화", use_sllm=False)
        print("✓ 초기화 완료\n")
    except Exception as e:
        print(f"⚠️  초기화 경고: {e}\n")
    
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
            
            # 텍스트 처리
            process_text(user_input, use_sllm=True)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[ERROR] 오류 발생: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
