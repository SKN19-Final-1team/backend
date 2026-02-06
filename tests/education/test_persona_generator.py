"""
Persona Generator 테스트
"""
import json
import os
import sys

# 프로젝트 루트 경로 추가 (모듈 import 문제 해결)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.llm.education.persona_generator import create_system_prompt
from app.llm.education.client import generate_text, SIM_RUNPOD_URL

def test_model_connection():
    """모델 연결 확인"""
    print("=" * 60)
    print("RunPod 모델 연결 확인")
    print("=" * 60)
    
    if not SIM_RUNPOD_URL:
        print("❌ SIM_RUNPOD_URL 환경변수가 설정되지 않았습니다.")
        print("   .env 파일을 확인하거나 환경변수를 설정해주세요.")
        return False
        
    print(f"URL: {SIM_RUNPOD_URL}")
    
    try:
        # 간단한 테스트 요청
        print("연결 테스트 중...", end=" ", flush=True)
        response = generate_text("테스트 연결", max_tokens=10)
        
        if response:
             print("✅ 성공")
             print(f"응답 샘플: {response}")
             return True
        else:
             print("❌ 실패 (응답 없음)")
             return False
    except Exception as e:
        print(f"\n❌ 연결 실패: {e}")
        return False

def test_persona_generation_with_llm():
    """하드코딩된 고객 데이터로 페르소나 생성 및 LLM 응답 테스트"""
    
    print("\n" + "=" * 60)
    print("Persona Generator & LLM 연동 테스트")
    print("=" * 60)

    # 1. 하드코딩된 고객 정보
    # customer.json 형식을 참조하여 테스트용 데이터 생성
    customer_profile = {
        "name": "박철수",
        "age_group": "50대",
        "grade": "VIP",
        "gender": "male",
        # 리스트 형태로 입력 (내부에서 처리됨)
        "personality_tags": ["impatient", "direct", "demanding", "angry"], 
        "communication_style": {"tone": "direct", "speed": "fast"},
        "category": "분실/도난",
        "persona_name": "화난 VIP 고객",
        "persona_description": "매우 급한 성격이며, 자신의 지위를 이용하여 빨리 처리해주기를 원합니다. 작은 실수에도 화를 냅니다."
    }
    
    print(f"\n[고객 정보] {customer_profile['name']} ({customer_profile['age_group']}, {customer_profile['grade']})")
    print(f"성격: {', '.join(customer_profile['personality_tags'])}")
    print(f"설명: {customer_profile['persona_description']}")
    print("-" * 60)

    # 2. 시스템 프롬프트 생성
    system_prompt = create_system_prompt(customer_profile, difficulty="beginner")
    print("\n[생성된 시스템 프롬프트]")
    print(system_prompt)
    print("-" * 60)
    
    # 3. LLM 페르소나 테스트
    print("\n[LLM 페르소나 응답 테스트]")
    
    # 시나리오: 상담원이 인사를 건넴
    user_inputs = [
        "안녕하십니까, 테디카드 고객센터 상담원 이영희입니다. 무엇을 도와드릴까요?",
        "고객님, 잠시만 기다려주시면 확인해드리겠습니다.",
        "죄송하지만 규정상 어렵습니다."
    ]
    
    current_system_prompt = system_prompt
    
    for user_input in user_inputs:
        print(f"\n🗣️ 상담원: {user_input}")
        
        response = generate_text(
            prompt=user_input,
            system_prompt=current_system_prompt,
            temperature=0.7,
            max_tokens=200
        )
        
        print(f"👤 고객(페르소나): {response}")
        
        if not response:
            print("❌ 응답 생성 실패")
            break

    print("\n✅ 테스트 완료")

if __name__ == "__main__":
    if test_model_connection():
        test_persona_generation_with_llm()
    else:
        print("\n⛔ 모델 연결에 실패하여 페르소나 테스트를 중단합니다.")
