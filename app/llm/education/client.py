import os
import json
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

SIM_RUNPOD_URL = os.getenv("SIM_RUNPOD_URL")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_MODEL_NAME = "WindyAle/kanana-nano-2.1B-customer-emotional"

_session = requests.Session()


def generate_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 500,
    json_output: bool = False
) -> str:
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
        response = _session.post(SIM_RUNPOD_URL, json=payload, headers=headers, timeout=30)
        print(f"[LLM Client] 응답 수신: {response.status_code}")
        
        if response.status_code != 200:
            print(f"[LLM Client] API 오류 ({response.status_code}): {response.text}")
            return ""
        
        result = response.json()
        output = result['choices'][0]['message']['content'].strip()
        
        return output
        
    except requests.exceptions.RequestException as e:
        print(f"[Edu Client] 네트워크 오류 발생: {e}")
        return ""
    except (KeyError, IndexError) as e:
        print(f"[Edu Client] 응답 구조 오류: {e}")
        return ""
    except Exception as e:
        print(f"[Edu Client] 알 수 없는 오류: {e}")
        import traceback
        traceback.print_exc()
        return ""


def generate_json(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 500
) -> Dict[str, Any]:
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
