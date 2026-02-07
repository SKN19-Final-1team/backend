"""
Qwen3 TTS 로컬 서버

Qwen3-TTS 모델을 로드하여 REST API로 제공합니다.
기존 RunPod TTS 서버와 동일한 API 구조를 유지합니다.

사용법:
    conda activate final_env
    python scripts/run_tts_server.py --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --port 8001

API 엔드포인트:
    POST /tts - 텍스트를 음성으로 변환
    GET /health - 서버 상태 확인
    GET /speakers - 사용 가능한 스피커 목록
"""
import os
import sys
import io
import argparse
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")


# ============================================================
# 설정
# ============================================================

# 모델 경로 (환경변수 또는 기본값)
DEFAULT_MODEL = os.getenv("LOCAL_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")

# 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 전역 모델 인스턴스
_model = None


# ============================================================
# FastAPI 앱
# ============================================================

app = FastAPI(
    title="Qwen3 TTS Server",
    description="로컬 TTS 서버 (Qwen3-TTS-12Hz-0.6B-CustomVoice)",
    version="1.0.0"
)


class TTSRequest(BaseModel):
    text: str
    language: str = "Korean"
    speaker: str = "default"
    instruct: str = ""


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


class SpeakerInfo(BaseModel):
    name: str
    language: str
    description: str


# ============================================================
# 모델 로드
# ============================================================

def load_model(model_path: str):
    """TTS 모델 로드"""
    global _model

    print(f"[TTS Server] 모델 로드 중: {model_path}")
    print(f"[TTS Server] 디바이스: {DEVICE}")

    try:
        from qwen_tts import Qwen3TTSModel

        # 모델 로드 (qwen_tts API 사용)
        _model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=DEVICE,
            dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            attn_implementation="eager"  # FlashAttention 비활성화 (경고 방지)
        )

        print(f"[TTS Server] 모델 로드 완료!")
        return True

    except Exception as e:
        print(f"[TTS Server] 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_audio(text: str, language: str = "Korean", speaker: str = "default", instruct: str = "") -> bytes:
    """텍스트를 음성으로 변환"""
    global _model

    if _model is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")

    print(f"[TTS Server] 음성 생성 중: {text[:50]}...")
    print(f"[TTS Server] 설정 - language={language}, speaker={speaker}, instruct={instruct}")

    try:
        # qwen_tts API 사용
        wavs, sr = _model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker if speaker != "default" else "Sohee",  # default를 실제 스피커 이름으로 매핑
            instruct=instruct
        )

        # WAV 형식으로 변환
        import scipy.io.wavfile as wavfile

        # numpy array를 WAV 바이트로 변환
        buffer = io.BytesIO()
        wavfile.write(buffer, sr, wavs[0])
        buffer.seek(0)

        print(f"[TTS Server] 음성 생성 완료 (sample_rate={sr})")
        return buffer.read()

    except Exception as e:
        print(f"[TTS Server] 음성 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================
# API 엔드포인트
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인"""
    return HealthResponse(
        status="ok" if _model is not None else "model_not_loaded",
        model=DEFAULT_MODEL,
        device=DEVICE
    )


@app.get("/speakers")
async def get_speakers():
    """사용 가능한 스피커 목록"""
    return {
        "speakers": [
            {"name": "default", "language": "Korean", "description": "기본 한국어 음성"},
            {"name": "Sohee", "language": "Korean", "description": "한국어 여성"},
            {"name": "Eric", "language": "Korean", "description": "한국어 남성"},
        ]
    }


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """텍스트를 음성으로 변환"""
    if _model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="텍스트가 비어있습니다.")

    try:
        audio_bytes = generate_audio(
            text=request.text,
            language=request.language,
            speaker=request.speaker,
            instruct=request.instruct
        )

        return Response(
            content=audio_bytes,
            media_type="audio/wav"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3 TTS 로컬 서버")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="TTS 모델 경로 또는 HuggingFace ID")
    parser.add_argument("--port", type=int, default=8001, help="서버 포트")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트")

    args = parser.parse_args()

    print("=" * 60)
    print("   Qwen3 TTS 로컬 서버")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"포트: {args.port}")
    print(f"디바이스: {DEVICE}")
    print()

    # 모델 로드
    if not load_model(args.model):
        print("[ERROR] 모델 로드에 실패했습니다.")
        sys.exit(1)

    print()
    print(f"서버 시작: http://{args.host}:{args.port}")
    print("=" * 60)

    # 서버 실행
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
