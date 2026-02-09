"""
플러거블 STT 엔진 모듈

.env의 STT_ENGINE 설정으로 엔진을 전환할 수 있습니다:
  - openai-whisper  : OpenAI Whisper API (기본값, 클라우드)
  - qwen3-asr       : Qwen3-ASR 로컬 서버 (port 8104)
  - faster-whisper   : faster-whisper 로컬 (CPU/GPU)

사용법:
  from app.audio.stt_engine import create_stt_service
  stt = create_stt_service()
  stt.start(callback, loop)
  stt.add_audio(wav_bytes)
  stt.stop()
"""
import os
import threading
import queue
import io
import asyncio
import time
import re
import logging
from typing import Callable, Optional
from abc import ABC, abstractmethod

import requests
from openai import OpenAI
from dotenv import load_dotenv

from app.core.prompt import WHISPER_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# ── 할루시네이션 필터 ──────────────────────────────────────────
HALLUCINATION_KEYWORDS = [
    # YouTube / 방송 패턴
    "시청해주셔서", "시청해 주셔서", "구독과 좋아요", "구독 좋아요",
    "재택 플러스", "MBC", "뉴스", "투데이", "먹방",
    "영상편집", "영상", "편집", "진심으로",
    # 일반적 할루시네이션
    "오늘도 맛있게", "잘 먹었습니다", "감사합니다 여러분",
    "다음 시간에", "그럼 다음", "시간에 만나요",
    "자막 제공", "자막 by", "한국어 자막",
]

# 너무 짧은 응답 (1-2자) 필터
MIN_TEXT_LENGTH = 2

# 반복 패턴 필터 (같은 단어/문장이 3회 이상)
REPEAT_PATTERN = re.compile(r'(.{2,}?)\1{2,}')


def is_hallucination(text: str) -> bool:
    """Whisper 할루시네이션 여부 판정"""
    if not text or not text.strip():
        return True
    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return True
    if any(kw in text for kw in HALLUCINATION_KEYWORDS):
        return True
    if REPEAT_PATTERN.search(text):
        return True
    return False


# ── 기본 인터페이스 ────────────────────────────────────────────
class STTService(ABC):
    """STT 서비스 공통 인터페이스 (스트리밍 방식)"""

    @abstractmethod
    def start(self, callback: Callable, loop: asyncio.AbstractEventLoop):
        """콜백 + 이벤트루프를 받아 백그라운드 워커 시작"""

    @abstractmethod
    def add_audio(self, audio_data: bytes):
        """WAV 바이트를 큐에 추가"""

    @abstractmethod
    def stop(self):
        """워커 종료 및 자원 정리"""


# ── OpenAI Whisper (기본 엔진) ─────────────────────────────────
class OpenAIWhisperService(STTService):
    """OpenAI Whisper API 기반 STT"""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key)
        self.queue: queue.Queue = queue.Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.callback: Optional[Callable] = None

    def start(self, callback: Callable, loop: asyncio.AbstractEventLoop):
        self.callback = callback
        self.loop = loop
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        logger.info("[STT:whisper-1] 워커 종료")
        self.running = False
        self.queue.put(None)
        if self.thread:
            self.thread.join(timeout=5)

    def add_audio(self, audio_data: bytes):
        self.queue.put(audio_data)

    def _worker(self):
        logger.info("[STT:whisper-1] 워커 시작")
        while self.running:
            try:
                audio_data = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            if audio_data is None:
                break
            try:
                audio_file = io.BytesIO(audio_data)
                audio_file.name = "audio.wav"

                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko",
                    prompt=WHISPER_PROMPT,
                )
                text = transcript.text.strip()

                if is_hallucination(text):
                    logger.debug(f"[STT:whisper-1] 할루시네이션 필터: {text}")
                    continue

                if text and self.callback:
                    asyncio.run_coroutine_threadsafe(
                        self.callback(text), self.loop
                    )
            except Exception as e:
                logger.error(f"[STT:whisper-1] 오류: {e}")
            finally:
                self.queue.task_done()
        logger.info("[STT:whisper-1] 워커 종료")


# ── Qwen3-ASR (로컬 서버) ──────────────────────────────────────
class Qwen3ASRService(STTService):
    """Qwen3-ASR 로컬 서버 기반 STT (HTTP API)"""

    def __init__(self, server_url: str = None, model_size: str = "1.7B"):
        self.server_url = server_url or os.getenv("STT_QWEN3_URL", "http://localhost:8104")
        self.model_size = model_size
        self.queue: queue.Queue = queue.Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.callback: Optional[Callable] = None
        self._session = requests.Session()

    def start(self, callback: Callable, loop: asyncio.AbstractEventLoop):
        self.callback = callback
        self.loop = loop
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        logger.info("[STT:qwen3-asr] 워커 종료")
        self.running = False
        self.queue.put(None)
        if self.thread:
            self.thread.join(timeout=5)
        self._session.close()

    def add_audio(self, audio_data: bytes):
        self.queue.put(audio_data)

    def _worker(self):
        logger.info(f"[STT:qwen3-asr] 워커 시작 (서버: {self.server_url})")
        while self.running:
            try:
                audio_data = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            if audio_data is None:
                break
            try:
                start = time.perf_counter()
                files = {"file": ("audio.wav", io.BytesIO(audio_data), "audio/wav")}
                resp = self._session.post(
                    f"{self.server_url}/transcribe",
                    files=files,
                    timeout=30,
                )
                elapsed = time.perf_counter() - start

                if resp.status_code == 200:
                    result = resp.json()
                    text = result.get("text", "").strip()
                    logger.info(f"[STT:qwen3-asr] 인식 완료 ({elapsed*1000:.0f}ms): {text}")

                    if is_hallucination(text):
                        logger.debug(f"[STT:qwen3-asr] 할루시네이션 필터: {text}")
                        continue

                    if text and self.callback:
                        asyncio.run_coroutine_threadsafe(
                            self.callback(text), self.loop
                        )
                else:
                    logger.error(f"[STT:qwen3-asr] 서버 오류 ({resp.status_code})")
            except requests.exceptions.ConnectionError:
                logger.error(f"[STT:qwen3-asr] 서버 연결 실패: {self.server_url}")
            except Exception as e:
                logger.error(f"[STT:qwen3-asr] 오류: {e}")
            finally:
                self.queue.task_done()
        logger.info("[STT:qwen3-asr] 워커 종료")


# ── faster-whisper (로컬) ──────────────────────────────────────
class FasterWhisperService(STTService):
    """faster-whisper 로컬 모델 STT"""

    def __init__(self, model_size: str = None):
        self.model_size = model_size or os.getenv("STT_FASTER_WHISPER_SIZE", "base")
        self.queue: queue.Queue = queue.Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.callback: Optional[Callable] = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(f"[STT:faster-whisper] 모델 로딩: {self.model_size}")
            device = "cuda" if os.getenv("STT_DEVICE", "cpu") == "cuda" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
            )
            logger.info(f"[STT:faster-whisper] 모델 로드 완료 ({device})")
        return self._model

    def start(self, callback: Callable, loop: asyncio.AbstractEventLoop):
        self.callback = callback
        self.loop = loop
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        logger.info("[STT:faster-whisper] 워커 종료")
        self.running = False
        self.queue.put(None)
        if self.thread:
            self.thread.join(timeout=5)

    def add_audio(self, audio_data: bytes):
        self.queue.put(audio_data)

    def _worker(self):
        logger.info(f"[STT:faster-whisper] 워커 시작 (model={self.model_size})")
        model = self._load_model()

        while self.running:
            try:
                audio_data = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            if audio_data is None:
                break
            try:
                import numpy as np
                import soundfile as sf

                # WAV bytes → numpy
                audio_file = io.BytesIO(audio_data)
                data, sr = sf.read(audio_file)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = data.astype(np.float32)

                start = time.perf_counter()
                segments, info = model.transcribe(
                    data,
                    beam_size=5,
                    language="ko",
                    initial_prompt=WHISPER_PROMPT,
                )
                text = " ".join(seg.text for seg in segments).strip()
                elapsed = time.perf_counter() - start
                logger.info(f"[STT:faster-whisper] 인식 ({elapsed*1000:.0f}ms): {text}")

                if is_hallucination(text):
                    logger.debug(f"[STT:faster-whisper] 할루시네이션 필터: {text}")
                    continue

                if text and self.callback:
                    asyncio.run_coroutine_threadsafe(
                        self.callback(text), self.loop
                    )
            except Exception as e:
                logger.error(f"[STT:faster-whisper] 오류: {e}")
            finally:
                self.queue.task_done()
        logger.info("[STT:faster-whisper] 워커 종료")


# ── 설정 ─────────────────────────────────────────────────────
_current_stt_engine: str = os.getenv("STT_ENGINE", "openai-whisper").strip().lower()

# 사용 가능한 STT 엔진 정의
STT_ENGINE_OPTIONS = {
    "openai-whisper": {"label": "Whisper (OpenAI)", "description": "팀 프로덕션 기본"},
    "qwen3-asr": {"label": "Qwen3-ASR 1.7B", "description": "로컬 GPU STT"},
    "faster-whisper": {"label": "Faster-Whisper", "description": "로컬 경량 STT"},
}


# ── 팩토리 ─────────────────────────────────────────────────────
def create_stt_service(engine: str = None) -> STTService:
    """
    .env의 STT_ENGINE 또는 인자로 지정된 엔진 타입으로 STT 서비스 생성

    Args:
        engine: "openai-whisper", "qwen3-asr", "faster-whisper"

    Returns:
        STTService 인스턴스
    """
    engine = engine or _current_stt_engine
    engine = engine.strip().lower()

    logger.info(f"[STT] 엔진 생성: {engine}")

    if engine == "openai-whisper":
        return OpenAIWhisperService()
    elif engine == "qwen3-asr":
        return Qwen3ASRService()
    elif engine == "faster-whisper":
        return FasterWhisperService()
    else:
        logger.warning(f"[STT] 알 수 없는 엔진 '{engine}', openai-whisper로 폴백")
        return OpenAIWhisperService()


def switch_stt_engine(engine_type: str) -> dict:
    """런타임 STT 엔진 전환 (다음 웹소켓 연결부터 적용)"""
    global _current_stt_engine

    engine_type = engine_type.strip().lower()
    if engine_type not in STT_ENGINE_OPTIONS:
        return {"success": False, "error": f"Unknown engine: {engine_type}"}

    old = _current_stt_engine
    _current_stt_engine = engine_type
    logger.info(f"[STT] 엔진 전환: {old} → {engine_type}")
    return {
        "success": True,
        "engine": engine_type,
        "label": STT_ENGINE_OPTIONS[engine_type]["label"],
    }


def get_current_stt_config() -> dict:
    """현재 STT 설정 반환"""
    return {
        "engine": _current_stt_engine,
        "label": STT_ENGINE_OPTIONS.get(_current_stt_engine, {}).get("label", _current_stt_engine),
        "options": STT_ENGINE_OPTIONS,
    }
