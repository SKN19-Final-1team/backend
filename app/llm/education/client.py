import os
import json
import asyncio
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

SIM_RUNPOD_URL = os.getenv("SIM_RUNPOD_URL")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_MODEL_NAME = "WindyAle/kanana-nano-2.1B-customer-emotional"

# AsyncOpenAI 클라이언트
client = AsyncOpenAI(
    base_url=SIM_RUNPOD_URL,
    api_key=RUNPOD_API_KEY
)


async def generate_text_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 500,
    json_output: bool = False
) -> str:
    """비동기 텍스트 생성"""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.completions.create(
            model=RUNPOD_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )

        output = response.choices[0].message.content.strip()
        print(f"[LLM Client] 응답 수신 완료")

        return output

    except Exception as e:
        print(f"[Edu Client] API 호출 오류: {e}")
        import traceback
        traceback.print_exc()
        return ""


def generate_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 500,
    json_output: bool = False
) -> str:
    """동기 래퍼 (기존 호환성 유지)"""
    try:
        # 이벤트 루프가 이미 실행 중인지 확인
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 실행 중인 루프 없음 - 새로 실행
        return asyncio.run(generate_text_async(
            prompt, system_prompt, temperature, max_tokens, json_output
        ))

    # 이미 실행 중인 루프가 있으면 새 스레드에서 실행
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            asyncio.run,
            generate_text_async(prompt, system_prompt, temperature, max_tokens, json_output)
        )
        return future.result(timeout=60)


async def generate_json_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 500
) -> Dict[str, Any]:
    """비동기 JSON 생성"""
    output = await generate_text_async(prompt, system_prompt, temperature, max_tokens, json_output=True)

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


def generate_json(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 500
) -> Dict[str, Any]:
    """동기 래퍼 (기존 호환성 유지)"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(generate_json_async(
            prompt, system_prompt, temperature, max_tokens
        ))

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            asyncio.run,
            generate_json_async(prompt, system_prompt, temperature, max_tokens)
        )
        return future.result(timeout=60)
