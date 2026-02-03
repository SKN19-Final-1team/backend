import os
from typing import Dict, List

from dotenv import find_dotenv, load_dotenv
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

GUIDE_MODEL_NAME = "gpt-4.1-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_openai_client = None


def _get_openai_client() -> OpenAI | None:
    global _openai_client
    if not OPENAI_API_KEY:
        return None
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY.strip())
    return _openai_client


def get_guide_model_name() -> str:
    return GUIDE_MODEL_NAME


def generate_guide_text(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 320,
    top_p: float = 0.9,
    timeout_sec: int = 30,
) -> str:
    client = _get_openai_client()
    if not client:
        return ""
    try:
        resp = client.chat.completions.create(
            model=GUIDE_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=["손님:", "상담사:", "고객:"],
            timeout=timeout_sec,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        return ""
