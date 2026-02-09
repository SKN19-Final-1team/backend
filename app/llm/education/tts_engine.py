"""
플러거블 TTS 엔진 모듈

.env의 TTS_ENGINE 설정으로 엔진을 전환할 수 있습니다:
  - qwen3-tts    : Qwen3-TTS 로컬 서버 (기본값, port 8103)
  - edge-tts     : Microsoft Edge TTS (무료 클라우드, 빠름)
  - supertonic   : Supertonic TTS (초고속 로컬, RTF 0.06x)
  - vibevoice    : VibeVoice-Realtime (한국어 고품질, 로컬 GPU)

공통 기능:
  - 한국어 전처리 자동 적용 (숫자/기호 → 한글)
  - generate_speech() 함수 시그니처 유지 (하위 호환)

사용법:
  from app.llm.education.tts_engine import generate_speech
  success = generate_speech("안녕하세요", voice_config, output_path)
"""
import os
import sys
import asyncio
import copy
import logging
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv

from app.llm.education.korean_preprocess import preprocess_for_tts

load_dotenv()
logger = logging.getLogger(__name__)

# ── 설정 ────────────────────────────────────────────────────
TTS_ENGINE_TYPE = os.getenv("TTS_ENGINE", "qwen3-tts")
TTS_RUNPOD_URL = os.getenv("TTS_RUNPOD_URL", "http://localhost:8103")
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "30"))
TTS_PREPROCESS = os.getenv("TTS_PREPROCESS", "1") == "1"  # 한국어 전처리 ON/OFF

# HTTP 세션 재사용
_session = requests.Session()


# ── 기본 인터페이스 ────────────────────────────────────────────
class TTSEngine(ABC):
    """TTS 엔진 공통 인터페이스"""

    @abstractmethod
    def synthesize(
        self, text: str, voice_config: Dict[str, Any], output_path: str
    ) -> bool:
        """텍스트 → 음성 파일 생성. 성공 시 True."""

    def health_check(self) -> bool:
        return True


# ── Qwen3-TTS (로컬 서버, 기본) ────────────────────────────────
class Qwen3TTSEngine(TTSEngine):
    """Qwen3-TTS HTTP 서버 기반 엔진 (port 8103)"""

    def __init__(self, server_url: str = None, timeout: int = None):
        self.server_url = server_url or TTS_RUNPOD_URL
        self.timeout = timeout or TTS_TIMEOUT

    def synthesize(
        self, text: str, voice_config: Dict[str, Any], output_path: str
    ) -> bool:
        speaker = voice_config.get("speaker", "Sohee")
        language = voice_config.get("language", "Korean")
        instruct = voice_config.get("instruct", "")
        speed = voice_config.get("speed", "moderate")
        if speed == "slow" and not instruct:
            instruct = "천천히 말해"
        elif speed == "fast" and not instruct:
            instruct = "빠르게 말해"

        logger.info(f"[TTS:qwen3] 생성 중: {text[:50]}... (speaker={speaker})")

        payload = {
            "text": text,
            "language": language,
            "speaker": speaker,
            "instruct": instruct,
        }
        try:
            resp = _session.post(
                f"{self.server_url}/tts", json=payload, timeout=self.timeout
            )
            if resp.status_code != 200:
                logger.error(f"[TTS:qwen3] API 오류 ({resp.status_code})")
                return False

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.content)
            logger.info(f"[TTS:qwen3] 저장 완료: {output_path} ({len(resp.content)} bytes)")
            return True
        except requests.exceptions.Timeout:
            logger.error(f"[TTS:qwen3] 타임아웃 ({self.timeout}초)")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"[TTS:qwen3] 서버 연결 실패: {self.server_url}")
            return False
        except Exception as e:
            logger.error(f"[TTS:qwen3] 실패: {e}")
            return False

    def health_check(self) -> bool:
        try:
            resp = _session.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ── Edge-TTS (무료 클라우드, 빠름) ──────────────────────────────
class EdgeTTSEngine(TTSEngine):
    """Microsoft Edge TTS (edge-tts 패키지)"""

    # 한국어 음성 매핑
    VOICE_MAP = {
        "Sohee": "ko-KR-SunHiNeural",       # 여성 (기본)
        "SunHi": "ko-KR-SunHiNeural",       # 여성
        "InJoon": "ko-KR-InJoonNeural",     # 남성
    }
    DEFAULT_VOICE = "ko-KR-SunHiNeural"

    def synthesize(
        self, text: str, voice_config: Dict[str, Any], output_path: str
    ) -> bool:
        try:
            import edge_tts
        except ImportError:
            logger.error("[TTS:edge] edge-tts 패키지 미설치: pip install edge-tts")
            return False

        speaker = voice_config.get("speaker", "Sohee")
        voice = self.VOICE_MAP.get(speaker, self.DEFAULT_VOICE)

        logger.info(f"[TTS:edge] 생성 중: {text[:50]}... (voice={voice})")
        start = time.perf_counter()

        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            # edge-tts는 비동기이므로 이벤트루프에서 실행
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop and loop.is_running():
                # 이미 이벤트루프 안에 있으면 새 스레드에서 실행
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._async_synthesize(text, voice, output_path)
                    )
                    future.result(timeout=TTS_TIMEOUT)
            else:
                asyncio.run(self._async_synthesize(text, voice, output_path))

            elapsed = time.perf_counter() - start
            size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            logger.info(f"[TTS:edge] 완료: {elapsed:.1f}s, {size} bytes")
            return True
        except Exception as e:
            logger.error(f"[TTS:edge] 실패: {e}")
            return False

    async def _async_synthesize(self, text: str, voice: str, output_path: str):
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)


# ── Supertonic (초고속 로컬) ────────────────────────────────────
class SupertonicTTSEngine(TTSEngine):
    """Supertonic TTS (RTF 0.06x, MIT 라이센스)"""

    VOICE_MAP = {
        "Sohee": "F1",    # 여성 기본
        "InJoon": "M1",   # 남성 기본
    }
    _tts = None

    def _load(self):
        if self._tts is None:
            import supertonic
            logger.info("[TTS:supertonic] 모델 로딩...")
            self.__class__._tts = supertonic.TTS("supertonic-2")
            logger.info(f"[TTS:supertonic] 로드 완료 (sr={self._tts.sample_rate})")
        return self._tts

    def synthesize(
        self, text: str, voice_config: Dict[str, Any], output_path: str
    ) -> bool:
        try:
            tts = self._load()
        except ImportError:
            logger.error("[TTS:supertonic] supertonic 패키지 미설치: pip install supertonic")
            return False

        speaker = voice_config.get("speaker", "Sohee")
        voice_id = self.VOICE_MAP.get(speaker, "F1")

        logger.info(f"[TTS:supertonic] 생성 중: {text[:50]}... (voice={voice_id})")
        start = time.perf_counter()

        try:
            style = tts.get_voice_style(voice_id)
            audio, duration = tts.synthesize(text, voice_style=style, lang="ko")
            elapsed = time.perf_counter() - start

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            tts.save_audio(audio, output_path)

            rtf = elapsed / float(duration[0]) if duration[0] > 0 else 0
            logger.info(f"[TTS:supertonic] 완료: {elapsed*1000:.0f}ms, RTF={rtf:.3f}")
            return True
        except Exception as e:
            logger.error(f"[TTS:supertonic] 실패: {e}")
            return False


# ── VibeVoice-Realtime (한국어 고품질 로컬) ──────────────────────
class VibeVoiceTTSEngine(TTSEngine):
    """
    VibeVoice-Realtime 0.5B TTS (Microsoft)

    한국어 음성 품질이 좋음. GPU 필요 (~2GB VRAM).
    음성 프리셋: kr-spk0 (여성), kr-spk1 (남성)

    필요:
      - VibeVoice 패키지 (backend_dev/local_servers/VibeVoice)
      - 모델: microsoft/VibeVoice-Realtime-0.5B (HuggingFace 자동 다운로드)
    """

    VOICE_MAP = {
        "Sohee": "kr-spk0",    # 여성 기본
        "InJoon": "kr-spk1",   # 남성
    }
    _engine = None  # 클래스 레벨 싱글톤

    @classmethod
    def _load(cls):
        """VibeVoice 모델 + 프로세서 로드 (최초 1회)"""
        if cls._engine is not None:
            return cls._engine

        import torch

        # VibeVoice 패키지 경로 (backend_dev/local_servers/VibeVoice)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        vibevoice_paths = [
            project_root / "local_servers" / "VibeVoice",                     # backend/
            project_root.parent / "backend_dev" / "local_servers" / "VibeVoice",  # backend_dev/
        ]
        vibevoice_path = None
        for p in vibevoice_paths:
            if p.exists():
                vibevoice_path = p
                break

        if vibevoice_path is None:
            raise FileNotFoundError(
                "VibeVoice 패키지를 찾을 수 없습니다. "
                "backend_dev/local_servers/VibeVoice 또는 backend/local_servers/VibeVoice 경로를 확인하세요."
            )

        if str(vibevoice_path) not in sys.path:
            sys.path.insert(0, str(vibevoice_path))

        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor
        )

        model_path = "microsoft/VibeVoice-Realtime-0.5B"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"[TTS:vibevoice] 모델 로딩 중: {model_path} ({device})...")
        load_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Processor
        processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

        # Model (flash_attention_2 → sdpa 폴백)
        try:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map=device,
                attn_implementation="flash_attention_2",
            )
        except Exception as e:
            logger.warning(f"[TTS:vibevoice] FlashAttention 실패, SDPA 사용: {e}")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map=device,
                attn_implementation="sdpa",
            )

        model.eval()
        model.set_ddpm_inference_steps(num_steps=5)

        # 한국어 음성 프리셋 경로
        voices_dir = vibevoice_path / "demo" / "voices" / "streaming_model"
        kr_voices = {
            "kr-spk0": voices_dir / "kr-Spk0_woman.pt",
            "kr-spk1": voices_dir / "kr-Spk1_man.pt",
        }

        cls._engine = {
            "model": model,
            "processor": processor,
            "device": device,
            "torch_dtype": load_dtype,
            "voices": kr_voices,
        }
        logger.info(f"[TTS:vibevoice] 로드 완료 ({device})")
        return cls._engine

    @staticmethod
    def _convert_dynamic_cache(old_cache):
        """transformers >= 4.57 DynamicCache 호환성 변환"""
        import torch
        from transformers.cache_utils import DynamicCache

        if not hasattr(old_cache, 'key_cache'):
            return old_cache

        new_cache = DynamicCache()
        for i in range(len(old_cache.key_cache)):
            new_cache.update(
                old_cache.key_cache[i],
                old_cache.value_cache[i],
                layer_idx=i,
            )
        return new_cache

    @classmethod
    def _convert_prefilled_outputs(cls, all_prefilled_outputs):
        """all_prefilled_outputs 내의 모든 DynamicCache를 새 포맷으로 변환"""
        for key in ["lm", "tts_lm", "neg_lm", "neg_tts_lm"]:
            if key in all_prefilled_outputs and "past_key_values" in all_prefilled_outputs[key]:
                cache = all_prefilled_outputs[key]["past_key_values"]
                if hasattr(cache, 'key_cache') and not hasattr(cache, 'layers'):
                    all_prefilled_outputs[key]["past_key_values"] = cls._convert_dynamic_cache(cache)
        return all_prefilled_outputs

    def synthesize(
        self, text: str, voice_config: Dict[str, Any], output_path: str
    ) -> bool:
        import torch

        start = time.perf_counter()

        try:
            engine = self._load()
            model = engine["model"]
            processor = engine["processor"]
            device = engine["device"]
            voices = engine["voices"]

            # 음성 프리셋 선택
            speaker = voice_config.get("speaker", "Sohee")
            voice_key = self.VOICE_MAP.get(speaker, "kr-spk0")

            voice_path = voices.get(voice_key)
            if voice_path is None or not voice_path.exists():
                voice_path = voices.get("kr-spk0")
                if voice_path is None or not voice_path.exists():
                    logger.error(f"[TTS:vibevoice] 음성 프리셋 없음: {voice_key}")
                    return False

            logger.info(f"[TTS:vibevoice] 생성 중: {text[:50]}... (voice={voice_key})")

            # 캐시된 프롬프트 로드
            all_prefilled_outputs = torch.load(
                str(voice_path), map_location=device, weights_only=False
            )
            all_prefilled_outputs = self._convert_prefilled_outputs(all_prefilled_outputs)

            # 텍스트 정리 (따옴표 정규화)
            clean_text = text.replace("\u2018", "'").replace("\u2019", "'")
            clean_text = clean_text.replace("\u201c", '"').replace("\u201d", '"')

            # 입력 처리
            inputs = processor.process_input_with_cached_prompt(
                text=clean_text,
                cached_prompt=all_prefilled_outputs,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            # 오디오 생성
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.5,
                tokenizer=processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
            )

            elapsed = time.perf_counter() - start

            # 오디오 저장
            if outputs.speech_outputs and len(outputs.speech_outputs) > 0 and outputs.speech_outputs[0] is not None:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                processor.save_audio(outputs.speech_outputs[0], output_path=output_path)

                sample_rate = 24000
                audio_samples = outputs.speech_outputs[0].shape[-1]
                audio_duration = audio_samples / sample_rate
                rtf = elapsed / audio_duration if audio_duration > 0 else 0

                size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                logger.info(
                    f"[TTS:vibevoice] 완료: {elapsed:.1f}s, "
                    f"오디오 {audio_duration:.1f}s, RTF={rtf:.2f}, {size} bytes"
                )
                return True
            else:
                logger.warning("[TTS:vibevoice] 오디오 출력 없음")
                return False

        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"[TTS:vibevoice] 실패 ({elapsed:.1f}s): {e}")
            import traceback
            traceback.print_exc()
            return False


# ── 팩토리 ─────────────────────────────────────────────────────
_engine_instance: Optional[TTSEngine] = None
_current_engine_type: str = TTS_ENGINE_TYPE.strip().lower()
_current_speaker: str = "Sohee"

# 사용 가능한 엔진/보이스 정의
TTS_ENGINE_OPTIONS = {
    "qwen3-tts": {
        "label": "Qwen3-TTS",
        "voices": [
            {"id": "Sohee", "label": "Sohee (여)", "gender": "F"},
            {"id": "Aiden", "label": "Aiden (남)", "gender": "M"},
        ],
    },
    "edge-tts": {
        "label": "Edge-TTS",
        "voices": [
            {"id": "SunHi", "label": "SunHi (여)", "gender": "F"},
            {"id": "InJoon", "label": "InJoon (남)", "gender": "M"},
        ],
    },
    "supertonic": {
        "label": "Supertonic",
        "voices": [
            {"id": "Sohee", "label": "여성 (F1)", "gender": "F"},
            {"id": "InJoon", "label": "남성 (M1)", "gender": "M"},
        ],
    },
    "vibevoice": {
        "label": "VibeVoice",
        "voices": [
            {"id": "Sohee", "label": "여성 (kr-spk0)", "gender": "F"},
            {"id": "InJoon", "label": "남성 (kr-spk1)", "gender": "M"},
        ],
    },
}


def _create_engine(engine_type: str) -> TTSEngine:
    """엔진 타입으로 인스턴스 생성"""
    engine_type = engine_type.strip().lower()
    if engine_type == "qwen3-tts":
        return Qwen3TTSEngine()
    elif engine_type == "edge-tts":
        return EdgeTTSEngine()
    elif engine_type == "supertonic":
        return SupertonicTTSEngine()
    elif engine_type == "vibevoice":
        return VibeVoiceTTSEngine()
    else:
        logger.warning(f"[TTS] 알 수 없는 엔진 '{engine_type}', qwen3-tts로 폴백")
        return Qwen3TTSEngine()


def _get_engine() -> TTSEngine:
    """현재 설정된 TTS 엔진 인스턴스 반환 (싱글톤)"""
    global _engine_instance, _current_engine_type

    if _engine_instance is not None:
        return _engine_instance

    logger.info(f"[TTS] 엔진 초기화: {_current_engine_type}")
    _engine_instance = _create_engine(_current_engine_type)
    return _engine_instance


def switch_engine(engine_type: str, speaker: str = None) -> Dict[str, Any]:
    """런타임 TTS 엔진 전환"""
    global _engine_instance, _current_engine_type, _current_speaker

    engine_type = engine_type.strip().lower()
    if engine_type not in TTS_ENGINE_OPTIONS:
        return {"success": False, "error": f"Unknown engine: {engine_type}"}

    old_engine = _current_engine_type
    _current_engine_type = engine_type
    _engine_instance = None  # 싱글톤 리셋 → 다음 호출 시 새 엔진 생성

    if speaker:
        _current_speaker = speaker

    logger.info(f"[TTS] 엔진 전환: {old_engine} → {engine_type} (speaker={_current_speaker})")
    return {
        "success": True,
        "engine": engine_type,
        "speaker": _current_speaker,
        "label": TTS_ENGINE_OPTIONS[engine_type]["label"],
    }


def get_current_config() -> Dict[str, Any]:
    """현재 TTS 설정 반환"""
    return {
        "engine": _current_engine_type,
        "speaker": _current_speaker,
        "label": TTS_ENGINE_OPTIONS.get(_current_engine_type, {}).get("label", _current_engine_type),
        "options": TTS_ENGINE_OPTIONS,
    }


def preload_engine():
    """
    현재 선택된 TTS 엔진을 프리로드 (백그라운드 태스크용)

    Supertonic, VibeVoice 등 GPU 모델은 최초 로드에 수~수십 초 소요.
    사용자가 버튼 클릭 시 미리 로드해두면 실제 TTS 요청 시 즉시 응답 가능.
    """
    try:
        logger.info(f"[TTS] 프리로드 시작: {_current_engine_type}")
        engine = _get_engine()

        # Supertonic: 모델 로드 트리거
        if isinstance(engine, SupertonicTTSEngine):
            engine._load()
            logger.info("[TTS] 프리로드 완료: Supertonic 모델 로드됨")

        # VibeVoice: 모델 + 프로세서 로드 트리거
        elif isinstance(engine, VibeVoiceTTSEngine):
            engine._load()
            logger.info("[TTS] 프리로드 완료: VibeVoice 모델 로드됨")

        # Qwen3-TTS: 서버 기반이므로 health check만
        elif isinstance(engine, Qwen3TTSEngine):
            healthy = engine.health_check()
            logger.info(f"[TTS] 프리로드: Qwen3-TTS 서버 상태 = {'정상' if healthy else '비정상'}")

        # Edge-TTS: 클라우드 기반이므로 로드 불필요
        else:
            logger.info(f"[TTS] 프리로드: {_current_engine_type} (로드 불필요)")

    except Exception as e:
        logger.error(f"[TTS] 프리로드 실패: {e}")


# ── 공개 API (하위 호환) ────────────────────────────────────────

def generate_speech(
    text: str,
    voice_config: Dict[str, Any],
    output_path: str,
    speaker_wav: Optional[str] = None,  # 하위 호환 (미사용)
) -> bool:
    """
    텍스트를 음성으로 변환 (플러거블 엔진)

    기존 호출 코드와 100% 호환됩니다.
    한국어 전처리가 자동으로 적용됩니다 (TTS_PREPROCESS=1).
    """
    if not text or not text.strip():
        logger.info("[TTS] 빈 텍스트, 건너뜀")
        return False

    # 한국어 전처리
    if TTS_PREPROCESS:
        original = text
        text = preprocess_for_tts(text)
        if text != original:
            logger.info(f"[TTS] 전처리: '{original[:40]}' → '{text[:40]}'")

    engine = _get_engine()
    return engine.synthesize(text, voice_config, output_path)


def check_server_health() -> bool:
    """TTS 엔진 상태 확인"""
    return _get_engine().health_check()


def get_model_status() -> Dict[str, Any]:
    """TTS 모델 상태 조회"""
    engine = _get_engine()
    healthy = engine.health_check()
    return {
        "status": "ok" if healthy else "unavailable",
        "engine": TTS_ENGINE_TYPE,
        "server_url": TTS_RUNPOD_URL if isinstance(engine, Qwen3TTSEngine) else "local",
    }


def get_available_speakers() -> list:
    """사용 가능한 스피커 목록"""
    try:
        resp = _session.get(f"{TTS_RUNPOD_URL}/speakers", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("speakers", [])
    except Exception:
        pass
    return [{"name": "Sohee", "language": "Korean", "description": "한국어 여성"}]


def save_audio_file(audio_data: bytes, file_path: str) -> bool:
    """오디오 데이터를 파일로 저장"""
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(audio_data)
        return True
    except Exception as e:
        logger.error(f"[TTS] 파일 저장 실패: {e}")
        return False
