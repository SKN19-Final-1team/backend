"""
TTS 엔진 래퍼 모듈

XTTS-v2 모델을 사용하여 텍스트를 음성으로 변환합니다.
"""
import os
from typing import Dict, Any, Optional
from pathlib import Path
import torch

# TTS 모델 전역 변수
_tts_model = None
_model_loaded = False


def load_tts_model():
    """
    XTTS-v2 모델 로드
    
    Returns:
        TTS 모델 객체
    """
    global _tts_model, _model_loaded
    
    if _model_loaded:
        return _tts_model
    
    try:
        from TTS.api import TTS
        
        print("[TTS Engine] XTTS-v2 모델 로딩 중...")
        
        # GPU 사용 가능 여부 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTS Engine] 디바이스: {device}")
        
        # XTTS-v2 모델 초기화
        # trust_remote_code=True로 설정하여 가중치 로딩 허용
        try:
            _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device=="cuda"))
        except Exception as e:
            print(f"[TTS Engine] XTTS-v2 로딩 실패, 대체 모델 시도: {e}")
            # 대체 모델: 한국어 지원 TTS
            try:
                _tts_model = TTS("tts_models/ko/cv/vits", gpu=(device=="cuda"))
                print("[TTS Engine] 대체 모델(Korean VITS) 로딩 성공")
            except Exception as e2:
                print(f"[TTS Engine] 대체 모델도 실패: {e2}")
                print("[TTS Engine] TTS 기능을 사용할 수 없습니다. 텍스트 응답만 제공됩니다.")
                _model_loaded = False
                return None
        
        _model_loaded = True
        print("[TTS Engine] 모델 로딩 완료")
        
        return _tts_model
        
    except ImportError as e:
        print(f"[TTS Engine] TTS 라이브러리가 설치되지 않았습니다: {e}")
        print("[TTS Engine] 설치: pip install TTS")
        _model_loaded = False
        return None
    except Exception as e:
        print(f"[TTS Engine] 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        _model_loaded = False
        return None


def generate_speech(
    text: str,
    voice_config: Dict[str, Any],
    output_path: str,
    speaker_wav: Optional[str] = None
) -> bool:
    """
    텍스트를 음성으로 변환
    
    Args:
        text: 변환할 텍스트
        voice_config: 음성 설정 (age_group, speed, tone)
        output_path: 출력 파일 경로 (.wav)
        speaker_wav: 참조 음성 파일 경로 (voice cloning용, 선택사항)
        
    Returns:
        성공 여부 (bool)
    """
    model = load_tts_model()
    
    if model is None:
        print("[TTS Engine] 모델이 로드되지 않아 TTS 생성을 건너뜁니다.")
        return False
    
    try:
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 음성 특성 매핑
        # speed: slow(0.8), moderate(1.0), fast(1.2)
        speed_map = {"slow": 0.8, "moderate": 1.0, "fast": 1.2}
        speed = speed_map.get(voice_config.get("speed", "moderate"), 1.0)
        
        print(f"[TTS Engine] 음성 생성 중: {text[:50]}...")
        print(f"[TTS Engine] 설정: speed={speed}, tone={voice_config.get('tone', 'neutral')}")
        
        # 기본 한국어 음성 사용 (voice cloning 없이)
        if speaker_wav and os.path.exists(speaker_wav):
            # Voice cloning 사용
            model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language="ko",
                speed=speed
            )
        else:
            # 기본 음성 사용
            model.tts_to_file(
                text=text,
                file_path=output_path,
                language="ko",
                speed=speed
            )
        
        print(f"[TTS Engine] 음성 파일 저장 완료: {output_path}")
        return True
        
    except Exception as e:
        print(f"[TTS Engine] 음성 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_model_status() -> Dict[str, Any]:
    """
    TTS 모델 상태 조회
    
    Returns:
        모델 상태 정보
    """
    return {
        "model_loaded": _model_loaded,
        "model_name": "XTTS-v2" if _model_loaded else None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


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
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(audio_data)
        
        print(f"[TTS Engine] 오디오 파일 저장: {file_path}")
        return True
        
    except Exception as e:
        print(f"[TTS Engine] 파일 저장 실패: {e}")
        return False
