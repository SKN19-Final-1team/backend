"""
vLLM용 모델 다운로드 스크립트

HuggingFace에서 모델을 다운로드하여 로컬에 저장합니다.
conda activate final_env 환경에서 실행하세요.

사용법:
    cd c:\\SKN19\\callact\\backend
    conda activate final_env
    python scripts/download_models.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# backend 폴더 기준으로 .env 로드
backend_dir = Path(__file__).parent.parent
load_dotenv(backend_dir / ".env")

# 모델 저장 경로 (환경변수 또는 기본값)
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")

# 다운로드할 모델 목록
MODELS = {
    "llm": "WindyAle/kanana-nano-2.1B-customer-emotional",
    "tts": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}


def check_dependencies():
    """필요한 패키지 확인"""
    try:
        from huggingface_hub import snapshot_download
        return True
    except ImportError:
        print("[Error] huggingface_hub 패키지가 설치되어 있지 않습니다.")
        print("  pip install huggingface_hub")
        return False


def download_model(model_id: str, cache_dir: str) -> str:
    """
    HuggingFace에서 모델 다운로드

    Args:
        model_id: HuggingFace 모델 ID (예: "WindyAle/kanana-nano-2.1B-customer-emotional")
        cache_dir: 로컬 저장 경로

    Returns:
        다운로드된 모델 경로
    """
    from huggingface_hub import snapshot_download

    print(f"[Download] 모델 다운로드 시작: {model_id}")
    print(f"[Download] 저장 경로: {cache_dir}")

    # HuggingFace 토큰 (private 모델용)
    hf_token = os.getenv("HF_TOKEN")

    # 모델 저장 경로 생성
    local_dir = os.path.join(cache_dir, model_id.replace("/", "--"))

    # 모델 다운로드
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        token=hf_token,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

    print(f"[Download] 다운로드 완료: {local_path}")
    return local_path


def main():
    # 의존성 확인
    if not check_dependencies():
        sys.exit(1)

    # 캐시 디렉토리 생성 (절대 경로로 변환)
    if not os.path.isabs(MODEL_CACHE_DIR):
        cache_path = Path(backend_dir) / MODEL_CACHE_DIR
    else:
        cache_path = Path(MODEL_CACHE_DIR)

    cache_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("vLLM용 모델 다운로드")
    print("=" * 60)
    print(f"저장 경로: {cache_path.absolute()}")
    print()

    downloaded_paths = {}
    for name, model_id in MODELS.items():
        print(f"\n[{name.upper()}] {model_id}")
        print("-" * 40)
        try:
            local_path = download_model(model_id, str(cache_path))
            downloaded_paths[name] = local_path
            print(f"  -> 성공: {local_path}")
        except Exception as e:
            print(f"  -> 실패: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("다운로드 완료!")
    print("=" * 60)

    if "llm" in downloaded_paths:
        print("\n[1. vLLM 서버 실행 명령 (sLLM)]")
        print(f"  vllm serve {downloaded_paths['llm']} --host 0.0.0.0 --port 8000")
        print("\n  또는 HuggingFace 모델 ID로 실행:")
        print(f"  vllm serve {MODELS['llm']} --host 0.0.0.0 --port 8000")

    if "tts" in downloaded_paths:
        print("\n[2. TTS 서버 실행 명령]")
        print(f"  python scripts/run_tts_server.py --model {downloaded_paths['tts']} --port 8001")
        print("\n  또는 HuggingFace 모델 ID로 실행:")
        print(f"  python scripts/run_tts_server.py --model {MODELS['tts']} --port 8001")


if __name__ == "__main__":
    main()
