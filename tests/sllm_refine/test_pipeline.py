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

from app.llm.delivery.deliverer import refine_text_pipeline


def print_header():
    """헤더 출력"""
    print("\n" + "=" * 70)
    print("텍스트 교정 파이프라인 테스트")
    print("=" * 70)
    print("종료: 'exit', 'quit', 'q' 입력")
    print("=" * 70 + "\n")


def process_text(user_input: str, use_sllm: bool = True):
    """텍스트 처리 및 결과 출력"""
    print(f"\n📝 입력: {user_input}")
    print("-" * 70)
    
    # 파이프라인 실행
    result = refine_text_pipeline(user_input, use_sllm=use_sllm)
    
    # Step 2: 형태소 분석 - 카드상품명 후보만 출력
    card_candidates = result['step2_morphology'].get('card_candidates', [])
    print(f"\n[Step 2] 카드상품명 후보: {card_candidates if card_candidates else '(없음)'}")
    
    # Step 3: 단어 매칭 - Top 3 매칭 결과만 출력
    matches = result['step3_matching'].get('matches', [])
    print(f"\n[Step 3] 단어 매칭 (Top-3):")
    if matches:
        for i, (name, score) in enumerate(matches[:3], 1):
            print(f"  {i}. {name:50} (유사도: {score:.3f})")
    else:
        print("  (매칭 결과 없음)")
    
    # Step 4: sLLM 최종 교정 - 최종 교정 텍스트만 출력
    if use_sllm:
        final_text = result['step4_refined']
        print(f"\n[Step 4] 최종 교정: {final_text}")
    else:
        print(f"\n[Step 4] sLLM 사용 안 함")
    
    print("-" * 70)


def main():
    """메인 루프"""
    print_header()
    
    # 시스템 초기화
    print("시스템 초기화 중...")
    try:
        # 더미 호출로 모듈 로드
        refine_text_pipeline("초기화", use_sllm=False)
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
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n[ERROR] 오류 발생: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
