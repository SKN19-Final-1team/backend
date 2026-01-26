import os
import json
import re
import requests
from typing import Optional, Dict, List
from dotenv import load_dotenv
# from llama_cpp import Llama

# _sllm_model: Optional[Llama] = None
_model_loaded = False
_model_load_failed = False

# RunPod API 설정
RUNPOD_IP = os.getenv("RUNPOD_IP")
RUNPOD_PORT = os.getenv("RUNPOD_PORT")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_MODEL_NAME = "kanana-nano-2.1b-instruct"

RUNPOD_API_URL = f"http://{RUNPOD_IP}:{RUNPOD_PORT}/v1/chat/completions"
_session = requests.Session()

# 로컬 모델 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# MODEL_PATH = os.path.join(BASE_DIR, "tests", "models", "kanana-nano-2.1b-instruct.Q4_K_M.gguf")
# MODEL_PATH = os.path.join(BASE_DIR, "tests", "models", "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf")

def refine_text(text: str) -> Dict[str, any]:
    # 텍스트가 비어있다면 원본 반환
    if not text or not text.strip():
        return {"text": text, "keywords": []}

    try:
        system_prompt = """금융 텍스트 교정. 발음 뭉개짐, 띄어쓰기 오류, 맞춤법 오류, 오타 수정. 아래 출력 형식 외 부연설명 절대 금지
[출력 형식]
json
{
    "original": "입력 텍스트",
    "refined" : "교정 텍스트"
}
"""

        payload = {
            "model": RUNPOD_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"입력: {text}"}
            ],
            "temperature": 0.1,
            "max_tokens": 50,
            "top_p": 0.9,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RUNPOD_API_KEY}"
        }
        
        response = _session.post(RUNPOD_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"[RunPod] API 오류 ({response.status_code}): {response.text}")
            return {"text": text, "keywords": []}
        
        result = response.json()
        
        try:
            output = result['choices'][0]['message']['content'].strip()
            return output

        except (KeyError, IndexError):
            print(f"[RunPod] 응답 구조가 예상과 다릅니다: {result}")
            return {"text": text, "keywords": []}
            
    except requests.exceptions.RequestException as e:
        print(f"[RunPod] 네트워크 오류: {e}")
        return {"text": text, "keywords": []}
    except Exception as e:
        print(f"[RunPod] 처리 중 문제 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"text": text, "keywords": []}

def get_model_status() -> dict:
    return {
        "model_loaded": _model_loaded,
        "model_load_failed": _model_load_failed,
        "model_path": MODEL_PATH,
    }