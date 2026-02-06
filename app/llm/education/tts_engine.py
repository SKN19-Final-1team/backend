"""
TTS 엔진 래퍼 모듈

RunPod에 배포된 Qwen3 TTS 서버를 호출하여 텍스트를 음성으로 변환합니다.
"""
import os
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# RunPod TTS 서버 설정
TTS_RUNPOD_URL = os.getenv("TTS_RUNPOD_URL", "http://localhost:8000")
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "30"))

# 세션 재사용
_session = requests.Session()

# 상태 캐시
_server_available = None


def check_server_health() -> bool:
    """TTS 서버 상태 확인"""
    global _server_available

    try:
        response = _session.get(f"{TTS_RUNPOD_URL}/health", timeout=5)
        _server_available = response.status_code == 200
        return _server_available
    except Exception as e:
        print(f"[TTS Engine] 서버 연결 실패: {e}")
        _server_available = False
        return False


def generate_speech(
    text: str,
    voice_config: Dict[str, Any],
    output_path: str,
    speaker_wav: Optional[str] = None  # 하위 호환성 (미사용)
) -> bool:
    """
    텍스트를 음성으로 변환 (RunPod 서버 호출)

    Args:
        text: 변환할 텍스트
        voice_config: 음성 설정 (speaker, language, instruct)
        output_path: 출력 파일 경로 (.wav)
        speaker_wav: (미사용) 하위 호환성을 위해 유지

    Returns:
        성공 여부 (bool)
    """
    if not text or not text.strip():
        print("[TTS Engine] 빈 텍스트, 건너뜀")
        return False

    try:
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 음성 설정 매핑
        speaker = voice_config.get("speaker", "Sohee")
        language = voice_config.get("language", "Korean")
        instruct = voice_config.get("instruct", "")

        # 속도를 instruct로 변환 (호환성)
        speed = voice_config.get("speed", "moderate")
        if speed == "slow" and not instruct:
            instruct = "천천히 말해"
        elif speed == "fast" and not instruct:
            instruct = "빠르게 말해"

        print(f"[TTS Engine] 음성 생성 중: {text[:50]}...")
        print(f"[TTS Engine] 설정: speaker={speaker}, language={language}")

        # RunPod TTS API 호출
        payload = {
            "text": text,
            "language": language,
            "speaker": speaker,
            "instruct": instruct
        }

        response = _session.post(
            f"{TTS_RUNPOD_URL}/tts",
            json=payload,
            timeout=TTS_TIMEOUT
        )

        if response.status_code != 200:
            print(f"[TTS Engine] API 오류 ({response.status_code}): {response.text[:200]}")
            return False

        # WAV 파일 저장
        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"[TTS Engine] 음성 파일 저장 완료: {output_path}")
        return True

    except requests.exceptions.Timeout:
        print(f"[TTS Engine] 서버 타임아웃 ({TTS_TIMEOUT}초)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"[TTS Engine] 서버 연결 실패: {TTS_RUNPOD_URL}")
        return False
    except Exception as e:
        print(f"[TTS Engine] 음성 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_model_status() -> Dict[str, Any]:
    """
    TTS 모델 상태 조회
    """
    try:
        response = _session.get(f"{TTS_RUNPOD_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    return {
        "status": "unavailable",
        "server_url": TTS_RUNPOD_URL
    }


def get_available_speakers() -> list:
    """
    사용 가능한 스피커 목록 조회
    """
    try:
        response = _session.get(f"{TTS_RUNPOD_URL}/speakers", timeout=5)
        if response.status_code == 200:
            return response.json().get("speakers", [])
    except Exception:
        pass

    # 기본 스피커 목록 (오프라인 시)
    return [
        {"name": "Sohee", "language": "Korean", "description": "한국어 여성"},
    ]


def save_audio_file(audio_data: bytes, file_path: str) -> bool:
    """
    오디오 데이터를 파일로 저장

    Args:
        audio_data: 오디오 바이트 데이터
        file_path: 저장 경로

    Returns:
        성공 여부
    """
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(file_path, 'wb') as f:
            f.write(audio_data)

        print(f"[TTS Engine] 오디오 파일 저장: {file_path}")
        return True

    except Exception as e:
        print(f"[TTS Engine] 파일 저장 실패: {e}")
        return False
