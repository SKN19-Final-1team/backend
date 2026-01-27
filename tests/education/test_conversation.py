"""
print("DEBUG: Script started")
tts_speaker.py 테스트 스크립트
"""
from app.llm.education.persona_generator import create_system_prompt
from app.llm.education.tts_speaker import (
    initialize_conversation,
    process_agent_input,
    get_conversation_history,
    end_conversation
)


def test_conversation_flow():
    """대화 흐름 테스트"""
    print("=" * 60)
    print("[테스트] 대화 시뮬레이션 흐름")
    print("=" * 60)
    
    # 1. 고객 프로필 생성
    customer_profile = {
        "name": "김영희",
        "age_group": "40대",
        "personality_tags": ["patient", "polite"],
        "communication_style": {
            "tone": "neutral",
            "speed": "moderate"
        },
        "llm_guidance": "친절하고 명확하게 안내해주세요."
    }
    
    # 2. 시스템 프롬프트 생성
    system_prompt = create_system_prompt(customer_profile, difficulty="beginner")
    
    # 3. 대화 세션 초기화
    session_id = "test_session_001"
    session = initialize_conversation(session_id, system_prompt, customer_profile)
    
    print(f"\n✅ 세션 초기화 완료: {session.session_id}")
    print(f"고객: {session.customer_profile['name']}")
    
    # 4. 대화 시뮬레이션
    test_messages = [
        "안녕하세요, 상담원 홍길동입니다.",
        "우선 본인확인 도와드리겠습니다. 성함과 생년월일 알려주시겠어요?",
        "네, 도와드리겠습니다. 카드 번호를 알려주시겠어요?",
    ]
    
    for i, agent_msg in enumerate(test_messages, 1):
        print(f"\n--- 턴 {i} ---")
        print(f"상담원: {agent_msg}")
        
        response = process_agent_input(session_id, agent_msg)
        
        print(f"고객: {response['customer_response']}")
        print(f"(턴 번호: {response['turn_number']})")
    
    # 5. 대화 히스토리 조회
    print("\n" + "=" * 60)
    print("[대화 히스토리]")
    print("=" * 60)
    
    history = get_conversation_history(session_id)
    for msg in history:
        role = "상담원" if msg["role"] == "agent" else "고객"
        print(f"{role}: {msg['content']}")
    
    # 6. 세션 종료
    print("\n" + "=" * 60)
    print("[세션 종료]")
    print("=" * 60)
    
    summary = end_conversation(session_id)
    print(f"총 턴 수: {summary['turn_count']}")
    print(f"대화 시간: {summary['duration_seconds']:.1f}초")
    
    print("\n✅ 테스트 완료")


if __name__ == "__main__":
    test_conversation_flow()
