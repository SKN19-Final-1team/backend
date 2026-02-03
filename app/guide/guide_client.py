import os
from typing import List, Dict, Optional

import requests
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


def _load_env() -> None:
    env_path = os.getenv("ENV_PATH")
    if env_path:
        load_dotenv(env_path, override=False)
        return
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate = os.path.join(base_dir, ".env")
    if os.path.exists(candidate):
        load_dotenv(candidate, override=False)
        return
    load_dotenv(find_dotenv(), override=False)


_load_env()

RUNPOD_GUIDE_BASE_URL = (os.getenv("RUNPOD_GUIDE_BASE_URL") or "").strip()
RUNPOD_IP = os.getenv("RUNPOD_IP")
RUNPOD_PORT = os.getenv("RUNPOD_PORT")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_GUIDE_MODEL_NAME = os.getenv("RUNPOD_GUIDE_MODEL_NAME", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_session = requests.Session()
_openai_client = None


def _resolve_api_url() -> str:
    if RUNPOD_GUIDE_BASE_URL:
        base = RUNPOD_GUIDE_BASE_URL.rstrip("/")
        if base.endswith("/v1/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"
    if RUNPOD_IP and RUNPOD_PORT:
        return f"http://{RUNPOD_IP}:{RUNPOD_PORT}/v1/chat/completions"
    return ""


def _should_use_openai(model: str) -> bool:
    if os.getenv("GUIDE_PROVIDER") == "openai":
        return True
    return model.startswith("gpt-")


def _get_openai_client() -> OpenAI | None:
    global _openai_client
    if not OPENAI_API_KEY:
        return None
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY.strip())
    return _openai_client


def get_guide_model_name() -> str:
    return RUNPOD_GUIDE_MODEL_NAME


def generate_guide_text(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 320,
    top_p: float = 0.9,
    timeout_sec: int = 30,
) -> str:
    if _should_use_openai(RUNPOD_GUIDE_MODEL_NAME):
        client = _get_openai_client()
        if not client:
            return ""
        try:
            resp = client.chat.completions.create(
                model=RUNPOD_GUIDE_MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=["손님:", "상담사:", "고객:"],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            return ""

    api_url = _resolve_api_url()
    if not api_url:
        return ""

    payload = {
        "model": RUNPOD_GUIDE_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": ["손님:", "상담사:", "\n손님", "\n상담사", "고객:"],
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if RUNPOD_API_KEY:
        headers["Authorization"] = f"Bearer {RUNPOD_API_KEY}"

    try:
        response = _session.post(api_url, json=payload, headers=headers, timeout=timeout_sec)
        if response.status_code != 200:
            return ""
        result = response.json()
        output = result["choices"][0]["message"]["content"].strip()
        return output
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        return ""
