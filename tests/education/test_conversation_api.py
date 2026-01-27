"""
대화 API 엔드포인트 통합 테스트
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1/education"


def test_full_conversation_flow():
    """전체 대화 흐름 테스트"""
    print("=" * 70)
    print("[통합 테스트] 교육 시뮬레이션 대화 흐름")
    print("=" * 70)
    
    # 1. 시뮬레이션 시작
    print("\n[1단계] 시뮬레이션 시작")
    print("-" * 70)
    
    start_payload = {
        "category": "도난/분실 신청/해제",
        "difficulty": "beginner"
    }
    
    response = requests.post(f"{BASE_URL}/simulation/start", json=start_payload)
    
    if response.status_code != 200:
        print(f"❌ 시뮬레이션 시작 실패: {response.status_code}")
        print(response.text)
        return
    
    session_data = response.json()
    session_id = session_data["session_id"]
    
    print(f"✅ 세션 생성: {session_id}")
    print(f"고객명: {session_data['customer_name']}")
    print(f"시스템 프롬프트 미리보기: {session_data['system_prompt'][:100]}...")
    
    # 2. 대화 진행
    print("\n[2단계] 대화 진행")
    print("-" * 70)
    
    turn = 1
    while True:
        try:
            agent_msg = input(f"\n턴 {turn} 상담원(당신): ").strip()
        except EOFError:
            break
            
        if agent_msg in ["종료", "quit", "exit"]:
            print("대화를 종료합니다.")
            break
            
        if not agent_msg:
            continue
        
        msg_payload = {
            "message": agent_msg,
            "mode": "text"
        }
        
        response = requests.post(
            f"{BASE_URL}/simulation/{session_id}/message",
            json=msg_payload
        )
        
        if response.status_code != 200:
            print(f"  ❌ 메시지 전송 실패: {response.status_code}")
            print(f"  {response.text}")
            continue
        
        msg_data = response.json()
        print(f"  고객: {msg_data['customer_response']}")
        print(f"  (턴 번호: {msg_data['turn_number']})")
        
        if msg_data.get('audio_url'):
            print(f"  🔊 TTS 오디오: {msg_data['audio_url']}")
            
        turn += 1
    
    # 3. 대화 히스토리 조회
    print("\n[3단계] 대화 히스토리 조회")
    print("-" * 70)
    
    response = requests.get(f"{BASE_URL}/simulation/{session_id}/history")
    
    if response.status_code == 200:
        history_data = response.json()
        print(f"✅ 총 {history_data['turn_count']}턴 진행됨")
        print(f"\n전체 대화:")
        for msg in history_data['conversation_history']:
            role = "상담원" if msg["role"] == "agent" else "고객"
            print(f"  {role}: {msg['content']}")
    else:
        print(f"❌ 히스토리 조회 실패: {response.status_code}")
    
    # 4. 세션 상태 조회
    print("\n[4단계] 세션 상태 조회")
    print("-" * 70)
    
    response = requests.get(f"{BASE_URL}/simulation/{session_id}/status")
    
    if response.status_code == 200:
        status_data = response.json()
        print(f"✅ 세션 상태: {status_data['status']}")
        print(f"   턴 수: {status_data['turn_count']}")
    else:
        print(f"❌ 상태 조회 실패: {response.status_code}")
    
    # 5. 시뮬레이션 종료
    print("\n[5단계] 시뮬레이션 종료")
    print("-" * 70)
    
    response = requests.post(f"{BASE_URL}/simulation/{session_id}/end")
    
    if response.status_code == 200:
        end_data = response.json()
        print(f"✅ 세션 종료")
        print(f"   총 턴 수: {end_data['turn_count']}")
        print(f"   대화 시간: {end_data['duration_seconds']:.1f}초")
    else:
        print(f"❌ 세션 종료 실패: {response.status_code}")
    
    print("\n" + "=" * 70)
    print("✅ 통합 테스트 완료")
    print("=" * 70)


if __name__ == "__main__":
    print("교육 시뮬레이션 대화 API 통합 테스트")
    print("서버가 http://localhost:8000 에서 실행 중이어야 합니다.\n")
    
    test_full_conversation_flow()
