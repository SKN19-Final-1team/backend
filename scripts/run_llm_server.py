"""
llama-cpp-python 기반 LLM 서버

GGUF 모델을 로드하여 OpenAI 호환 API로 제공합니다.
기존 client.py의 AsyncOpenAI 클라이언트와 호환됩니다.

사용법:
    conda activate final_env
    pip install llama-cpp-python[server]
    python scripts/run_llm_server.py --model ./model_cache/gguf/kanana-nano-2.1B.gguf --port 8000

API 엔드포인트:
    POST /v1/chat/completions - OpenAI 호환 채팅 API
    GET /v1/models - 모델 목록
"""
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# 기본 설정
DEFAULT_MODEL = "./model_cache/gguf/kanana-nano-2.1B-customer-emotional-Q4_K_M.gguf"
DEFAULT_PORT = 8080
DEFAULT_HOST = "0.0.0.0"

# GPU 레이어 수 (0 = CPU only, -1 = 모든 레이어 GPU)
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))

# 컨텍스트 크기
N_CTX = int(os.getenv("N_CTX", "2048"))


def check_dependencies():
    """의존성 확인"""
    try:
        import llama_cpp
        print(f"[OK] llama-cpp-python: {llama_cpp.__version__}")
        return True
    except ImportError:
        print("[ERROR] llama-cpp-python이 설치되어 있지 않습니다.")
        print("\nGPU 지원 설치 (CUDA):")
        print("  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python[server]")
        print("\nCPU 전용 설치:")
        print("  pip install llama-cpp-python[server]")
        return False


def run_server_builtin(model_path: str, host: str, port: int):
    """
    llama-cpp-python 내장 서버 실행
    """
    print(f"\n[Server] 내장 서버 시작")
    print(f"  모델: {model_path}")
    print(f"  주소: http://{host}:{port}")
    print(f"  GPU 레이어: {N_GPU_LAYERS}")
    print(f"  컨텍스트: {N_CTX}")

    # llama-cpp-python 서버 모듈 실행
    import subprocess

    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--n_gpu_layers", str(N_GPU_LAYERS),
        "--n_ctx", str(N_CTX),
        "--chat_format", "chatml",  # 채팅 포맷
    ]

    print(f"\n실행 명령: {' '.join(cmd)}\n")
    print("=" * 60)

    subprocess.run(cmd)


def run_server_custom(model_path: str, host: str, port: int):
    """
    커스텀 FastAPI 서버 (더 많은 제어 가능)
    """
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    import time

    from llama_cpp import Llama

    print(f"\n[Server] 모델 로드 중: {model_path}")

    # 모델 로드
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=False
    )

    print(f"[Server] 모델 로드 완료!")

    # FastAPI 앱
    app = FastAPI(title="LLM Server", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "local-model"
        messages: List[Message]
        temperature: float = 0.7
        max_tokens: int = 256
        top_p: float = 0.9

    class ChatChoice(BaseModel):
        index: int
        message: Message
        finish_reason: str

    class Usage(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    class ChatResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[ChatChoice]
        usage: Usage

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "local-model",
                    "object": "model",
                    "owned_by": "local"
                }
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        try:
            # 메시지를 프롬프트로 변환
            prompt = ""
            for msg in request.messages:
                if msg.role == "system":
                    prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
                elif msg.role == "user":
                    prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
                elif msg.role == "assistant":
                    prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"

            prompt += "<|im_start|>assistant\n"

            # 생성
            output = llm(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False
            )

            response_text = output["choices"][0]["text"].strip()

            return ChatResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=Message(role="assistant", content=response_text),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=output["usage"]["prompt_tokens"],
                    completion_tokens=output["usage"]["completion_tokens"],
                    total_tokens=output["usage"]["total_tokens"]
                )
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_path}

    print(f"\n[Server] 서버 시작: http://{host}:{port}")
    print("=" * 60)

    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="llama.cpp LLM 서버")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="GGUF 모델 경로")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="서버 포트")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="서버 호스트")
    parser.add_argument("--mode", type=str, choices=["builtin", "custom"], default="builtin",
                        help="서버 모드 (builtin: llama.cpp 내장, custom: FastAPI)")

    args = parser.parse_args()

    print("=" * 60)
    print("   llama.cpp LLM 서버")
    print("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    # 모델 파일 확인
    if not os.path.exists(args.model):
        print(f"\n[ERROR] 모델 파일을 찾을 수 없습니다: {args.model}")
        print("\n모델 다운로드/변환 스크립트를 먼저 실행하세요:")
        print("  python scripts/convert_to_gguf.py")
        sys.exit(1)

    if args.mode == "builtin":
        run_server_builtin(args.model, args.host, args.port)
    else:
        run_server_custom(args.model, args.host, args.port)


if __name__ == "__main__":
    main()
