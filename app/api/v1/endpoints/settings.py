"""
개발자 설정 API - STT/TTS 엔진 런타임 전환

프론트엔드 사이드바 버튼으로 엔진을 전환할 수 있습니다.
STT는 다음 웹소켓 연결부터, TTS는 즉시 적용됩니다.
"""
import asyncio
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from app.audio.stt_engine import (
    get_current_stt_config,
    switch_stt_engine,
)
from app.llm.education.tts_engine import (
    get_current_config as get_tts_config,
    switch_engine as switch_tts_engine,
    preload_engine as preload_tts_engine,
)

router = APIRouter()


class STTSwitchRequest(BaseModel):
    engine: str  # "openai-whisper" | "qwen3-asr" | "faster-whisper"


class TTSSwitchRequest(BaseModel):
    engine: str  # "qwen3-tts" | "edge-tts" | "supertonic" | "vibevoice"
    speaker: Optional[str] = None  # "Sohee" | "Aiden" | "InJoon" | "SunHi"


@router.get("/engines")
async def get_engines():
    """현재 STT/TTS 엔진 설정 조회"""
    return {
        "stt": get_current_stt_config(),
        "tts": get_tts_config(),
    }


@router.post("/stt-engine")
async def set_stt_engine(req: STTSwitchRequest):
    """STT 엔진 전환 (다음 웹소켓 연결부터 적용)"""
    return switch_stt_engine(req.engine)


@router.post("/tts-engine")
async def set_tts_engine(req: TTSSwitchRequest, background_tasks: BackgroundTasks):
    """TTS 엔진 전환 + 백그라운드 프리로드"""
    result = switch_tts_engine(req.engine, req.speaker)
    if result.get("success"):
        # 백그라운드에서 모델 프리로드 (GPU 로컬 엔진만 해당)
        background_tasks.add_task(preload_tts_engine)
    return result
