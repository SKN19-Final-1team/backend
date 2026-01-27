"""
Education API 엔드포인트 테스트
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1/education"


def test_get_scenarios():
    """시나리오 목록 조회 테스트"""
    print("=" * 60)
    print("[테스트 1] GET /scenarios - 시나리오 목록 조회")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/scenarios")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 성공: {len(data['scenarios'])}개 시나리오 조회됨")
            print("\n시나리오 목록:")
            for scenario in data['scenarios'][:5]:  # 처음 5개만 출력
                print(f"  - {scenario['description']} (카운트: {scenario['count']})")
        else:
            print(f"❌ 실패: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ 오류: {e}")


def test_start_simulation():
    """시뮬레이션 시작 테스트"""
    print("\n" + "=" * 60)
    print("[테스트 2] POST /simulation/start - 시뮬레이션 시작")
    print("=" * 60)
    
    payload = {
        "category": "도난/분실 신청/해제",
        "difficulty": "beginner"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/simulation/start", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 성공: 세션 ID = {data['session_id']}")
            print(f"고객명: {data['customer_name']}")
            print(f"\n시스템 프롬프트 미리보기:")
            print(data['system_prompt'][:300] + "...")
            
            if data.get('scenario_script'):
                print(f"\n각본: {data['scenario_script']}")
        else:
            print(f"❌ 실패: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ 오류: {e}")


def test_advanced_simulation():
    """상급 난이도 시뮬레이션 테스트"""
    print("\n" + "=" * 60)
    print("[테스트 3] POST /simulation/start - 상급 난이도")
    print("=" * 60)
    
    payload = {
        "category": "도난/분실 신청/해제",
        "difficulty": "advanced"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/simulation/start", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 성공: 세션 ID = {data['session_id']}")
            print(f"고객명: {data['customer_name']}")
            
            if data.get('scenario_script'):
                print(f"\n각본 생성됨:")
                print(f"  - 예상 흐름: {len(data['scenario_script'].get('expected_flow', []))}단계")
                print(f"  - 핵심 포인트: {len(data['scenario_script'].get('key_points', []))}개")
        else:
            print(f"❌ 실패: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ 오류: {e}")


if __name__ == "__main__":
    print("Education API 테스트 시작")
    print("서버가 http://localhost:8000 에서 실행 중이어야 합니다.\n")
    
    test_get_scenarios()
    test_start_simulation()
    test_advanced_simulation()
    
    print("\n" + "=" * 60)
    print("✅ 모든 테스트 완료")
    print("=" * 60)
