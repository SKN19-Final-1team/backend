import os
import requests
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# RunPod API 설정
RUNPOD_IP = os.getenv("RUNPOD_IP")
RUNPOD_PORT = os.getenv("RUNPOD_PORT")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

RUNPOD_API_URL = f"http://{RUNPOD_IP}:{RUNPOD_PORT}/v1/chat/completions"
_session = requests.Session()


def call_runpod(
    payload: Dict,
    headers: Optional[Dict] = None,
    timeout: int = 30
) -> Optional[str]:
    """
    Runpod API에 요청을 보냅니다.
    
    Args:
        payload: 요청 본문 데이터 (model, messages, params 등)
        headers: 추가 헤더 (기본적으로 Authorization 헤더는 자동 추가됨)
        timeout: 요청 타임아웃 (기본값 30초)
    
    Returns:
        응답 텍스트 (content) 또는 None
    """
    try:
        # 기본 헤더 설정
        default_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RUNPOD_API_KEY}"
        }
        
        # 사용자 정의 헤더가 있다면 병합 (사용자 정의가 우선)
        if headers:
            default_headers.update(headers)
            
        response = _session.post(
            RUNPOD_API_URL, 
            json=payload, 
            headers=default_headers, 
            timeout=timeout
        )
        
        if response.status_code != 200:
            print(f"[RunPod] API 오류 ({response.status_code}): {response.text}")
            return None
        
        result = response.json()
        
        try:
            output = result['choices'][0]['message']['content'].strip()
            return output
        except (KeyError, IndexError):
            print(f"[RunPod] 응답 구조가 예상과 다릅니다: {result}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"[RunPod] 네트워크 오류: {e}")
        return None
    except Exception as e:
        print(f"[RunPod] 처리 중 문제 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_runpod_status() -> Dict[str, any]:
    """
    Runpod 연결 상태를 반환합니다.
    
    Returns:
        연결 상태 정보 딕셔너리
    """
    return {
        "api_url": RUNPOD_API_URL,
        "configured": all([RUNPOD_IP, RUNPOD_PORT, RUNPOD_API_KEY])
    }