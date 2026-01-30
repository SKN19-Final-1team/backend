import os
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

RUNPOD_IP = os.getenv("RUNPOD_IP")
RUNPOD_PORT = os.getenv("RUNPOD_PORT")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_GUIDE_MODEL_NAME = os.getenv("RUNPOD_GUIDE_MODEL_NAME", "sllm")

RUNPOD_API_URL = f"http://{RUNPOD_IP}:{RUNPOD_PORT}/v1/chat/completions"
_session = requests.Session()


def generate_guide_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 500,
    json_output: bool = False,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": RUNPOD_GUIDE_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "stream": False,
    }
    # print(f"[Guide LLM] model={RUNPOD_GUIDE_MODEL_NAME} url={RUNPOD_API_URL}")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }

    try:
        response = _session.post(RUNPOD_API_URL, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            # print(f"[Guide LLM] API 오류 ({response.status_code}): {response.text}")
            return ""
        result = response.json()
        output = result["choices"][0]["message"]["content"].strip()
        return output
    except requests.exceptions.RequestException as e:
        # print(f"[Guide LLM] 네트워크 오류 발생: {e}")
        return ""
    except (KeyError, IndexError) as e:
        # print(f"[Guide LLM] 응답 구조 오류: {e}")
        return ""
    except Exception as e:
        # print(f"[Guide LLM] 알 수 없는 오류: {e}")
        import traceback
        # traceback.print_exc()
        return ""
