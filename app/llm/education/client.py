import json
import requests
from typing import Optional, Dict, Any

# RunPod API 설정 (sllm_refiner.py 참고)
RUNPOD_IP = "213.192.2.88"
RUNPOD_PORT = "40070"
RUNPOD_API_KEY = "0211"
RUNPOD_MODEL_NAME = "kanana-nano-2.1b-instruct"

RUNPOD_API_URL = f"http://{RUNPOD_IP}:{RUNPOD_PORT}/v1/chat/completions"
_session = requests.Session()


def generate_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 500,
    json_output: bool = False
) -> str:
    """
    RunPod API를 통해 kanana-nano-2.1b-instruct 모델로 텍스트 생성
    
    Args:
        prompt: 사용자 입력 프롬프트
        system_prompt: 시스템 프롬프트 (선택사항)
        temperature: 샘플링 온도 (0.0 ~ 1.0)
        max_tokens: 생성할 최대 토큰 수
        json_output: JSON 형식 출력 여부
        
    Returns:
        생성된 텍스트 (str)
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": RUNPOD_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    try:
        response = _session.post(RUNPOD_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"[LLM Client] API 오류 ({response.status_code}): {response.text}")
            return ""
        
        result = response.json()
        output = result['choices'][0]['message']['content'].strip()
        
        return output
        
    except requests.exceptions.RequestException as e:
        print(f"[LLM Client] 네트워크 오류: {e}")
        return ""
    except (KeyError, IndexError) as e:
        print(f"[LLM Client] 응답 구조 오류: {e}")
        return ""
    except Exception as e:
        print(f"[LLM Client] 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return ""


def generate_json(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 500
) -> Dict[str, Any]:
    """
    LLM으로 JSON 형식 응답 생성
    
    Args:
        prompt: 사용자 입력 프롬프트
        system_prompt: 시스템 프롬프트 (선택사항)
        temperature: 샘플링 온도
        max_tokens: 생성할 최대 토큰 수
        
    Returns:
        파싱된 JSON dict, 실패 시 빈 dict
    """
    output = generate_text(prompt, system_prompt, temperature, max_tokens, json_output=True)
    
    try:
        # JSON 코드 블록 제거 (```json ... ``` 형태)
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0].strip()
        elif "```" in output:
            output = output.split("```")[1].split("```")[0].strip()
        
        return json.loads(output)
    except json.JSONDecodeError as e:
        print(f"[LLM Client] JSON 파싱 실패: {e}")
        print(f"원본 출력: {output}")
        return {}
