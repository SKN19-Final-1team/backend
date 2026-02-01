"""
sLLM 텍스트 교정 대화형 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils.runpod_connector import call_runpod
from app.llm.delivery.sllm_refiner import refinement_payload, parse_refinement_result


def print_header():
    """헤더 출력"""
    print("\n" + "=" * 70)
    print("sLLM 텍스트 교정 테스트")
    print("=" * 70)
    print("종료: 'exit', 'quit', 'q' 입력")
    print("=" * 70 + "\n")


def test_refine(text: str):
    """텍스트 교정 테스트"""
    import time
    import json
    
    print(f"\n📝 입력: {text}")
    print("-" * 70)
    
    # 1. Payload 생성
    payload = refinement_payload(text)
    print(f"Temperature: {payload['temperature']}, Max Tokens: {payload['max_tokens']}")
    
    # 2. LLM 호출
    print("\n⏳ LLM 호출 중...")
    start_time = time.time()
    llm_output = call_runpod(payload)
    elapsed_time = time.time() - start_time
    
    # 3. LLM 원본 응답 출력
    print("\n[LLM 원본 응답]")
    print(llm_output)
    print()
    
    # 4. 파싱
    print("[파싱 결과]")
    result = parse_refinement_result(llm_output, text)
    print(f"교정된 텍스트: {result['text']}")
    
    # 5. 응답시간
    print(f"\n⏱️  응답시간: {elapsed_time:.2f}초 ({elapsed_time*1000:.0f}ms)")
    print("-" * 70)


def main():
    """메인 루프"""
    print_header()
    
    print("입력 방법:")
    print("  1. 한 줄 입력: 바로 텍스트 입력")
    print("  2. 여러 줄 입력: 'multi' 입력 후 여러 줄 작성 (빈 줄로 종료)")
    print("  3. 파일 입력: 'file:경로' 형식 (예: file:test.txt)")
    print()
    
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
            
            # 여러 줄 입력 모드
            if user_input.lower() == 'multi':
                print("여러 줄 입력 모드 (빈 줄 입력 시 종료):")
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
                text = '\n'.join(lines)
                
                if not text:
                    print("입력이 없습니다.")
                    continue
                    
                test_refine(text)
            
            # 파일 입력 모드
            elif user_input.lower().startswith('file:'):
                file_path = user_input[5:].strip()
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    print(f"파일 읽기 완료: {file_path} ({len(text)} 글자)")
                    test_refine(text)
                    
                except FileNotFoundError:
                    print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path}")
                except Exception as e:
                    print(f"[ERROR] 파일 읽기 실패: {e}")
            
            # 일반 한 줄 입력
            else:
                test_refine(user_input)
            
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n[ERROR] 오류 발생: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
