"""
HuggingFace 모델을 GGUF로 변환하는 스크립트

사용법:
    conda activate final_env
    pip install llama-cpp-python huggingface_hub transformers torch
    python scripts/convert_to_gguf.py
"""
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# 설정
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")
GGUF_OUTPUT_DIR = os.getenv("GGUF_OUTPUT_DIR", "./model_cache/gguf")

# 변환할 모델 (GGUF 버전이 이미 있으면 직접 다운로드)
MODELS = {
    "llm": {
        "hf_id": "WindyAle/kanana-nano-2.1B-customer-emotional-gguf",
        "gguf_name": "kanana-nano-2.1B-customer-emotional-Q4_K_M.gguf",
        "quantization": "q4_k_m"
    },
}


def check_llama_cpp():
    """llama.cpp 설치 확인"""
    try:
        import llama_cpp
        print(f"[OK] llama-cpp-python 버전: {llama_cpp.__version__}")
        return True
    except ImportError:
        print("[ERROR] llama-cpp-python이 설치되어 있지 않습니다.")
        print("  pip install llama-cpp-python")
        return False


def download_model(model_id: str, cache_dir: str) -> str:
    """HuggingFace에서 모델 다운로드"""
    from huggingface_hub import snapshot_download

    print(f"[Download] 모델 다운로드: {model_id}")

    hf_token = os.getenv("HF_TOKEN")
    local_dir = os.path.join(cache_dir, model_id.replace("/", "--"))

    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        token=hf_token,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

    print(f"[Download] 완료: {local_path}")
    return local_path


def convert_to_gguf(model_path: str, output_path: str, quantization: str = "q4_k_m"):
    """
    모델을 GGUF 형식으로 변환

    llama.cpp의 convert.py 스크립트 사용
    """
    print(f"\n[Convert] GGUF 변환 시작")
    print(f"  입력: {model_path}")
    print(f"  출력: {output_path}")
    print(f"  양자화: {quantization}")

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # llama.cpp convert 스크립트 경로 찾기
    # 방법 1: llama-cpp-python에 포함된 변환 도구 사용
    try:
        from llama_cpp import llama_cpp

        # llama-cpp-python은 직접 변환을 지원하지 않으므로
        # huggingface에서 이미 변환된 GGUF를 다운로드하거나
        # llama.cpp 저장소의 convert.py를 사용해야 함

        print("\n[INFO] llama-cpp-python은 직접 변환을 지원하지 않습니다.")
        print("       다음 방법 중 하나를 사용하세요:\n")

        print("방법 1: 이미 변환된 GGUF 모델 다운로드 (권장)")
        print("  - HuggingFace에서 *-GGUF 버전 검색")
        print("  - 예: TheBloke/모델명-GGUF\n")

        print("방법 2: llama.cpp 저장소에서 직접 변환")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp")
        print(f"  python convert_hf_to_gguf.py {model_path} --outfile {output_path}")
        print(f"  ./llama-quantize {output_path} {output_path.replace('.gguf', f'-{quantization}.gguf')} {quantization}\n")

        return False

    except Exception as e:
        print(f"[ERROR] 변환 실패: {e}")
        return False


def download_gguf_if_available(model_id: str, output_dir: str) -> str:
    """
    HuggingFace에서 GGUF 버전이 있으면 직접 다운로드
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    # GGUF 파일 검색 패턴
    gguf_patterns = [
        f"{model_id}-GGUF",
        f"{model_id.split('/')[-1]}-GGUF",
    ]

    # 원본 저장소에서 GGUF 파일 확인
    try:
        files = list_repo_files(model_id)
        gguf_files = [f for f in files if f.endswith('.gguf')]

        if gguf_files:
            print(f"[Found] 원본 저장소에 GGUF 파일 발견: {gguf_files}")

            # Q4_K_M 또는 가장 작은 파일 선택
            selected = None
            for f in gguf_files:
                if 'q4_k_m' in f.lower() or 'q4' in f.lower():
                    selected = f
                    break
            if not selected:
                selected = gguf_files[0]

            print(f"[Download] GGUF 다운로드: {selected}")
            local_path = hf_hub_download(
                repo_id=model_id,
                filename=selected,
                cache_dir=output_dir,
                local_dir=output_dir,
                token=os.getenv("HF_TOKEN")
            )
            return local_path

    except Exception as e:
        print(f"[INFO] 원본 저장소에서 GGUF 검색 실패: {e}")

    return None


def main():
    print("=" * 60)
    print("   GGUF 모델 다운로드/변환")
    print("=" * 60)

    if not check_llama_cpp():
        sys.exit(1)

    # 캐시 디렉토리 생성
    cache_path = Path(MODEL_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)

    gguf_path = Path(GGUF_OUTPUT_DIR)
    gguf_path.mkdir(parents=True, exist_ok=True)

    print(f"\n모델 캐시: {cache_path.absolute()}")
    print(f"GGUF 출력: {gguf_path.absolute()}")

    results = {}

    for name, config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"[{name.upper()}] {config['hf_id']}")
        print("=" * 60)

        output_file = gguf_path / config['gguf_name']

        # 1. 이미 GGUF가 있는지 확인
        if output_file.exists():
            print(f"[SKIP] 이미 존재: {output_file}")
            results[name] = str(output_file)
            continue

        # 2. HuggingFace에서 GGUF 버전 다운로드 시도
        gguf_result = download_gguf_if_available(config['hf_id'], str(gguf_path))
        if gguf_result:
            results[name] = gguf_result
            continue

        # 3. 원본 모델 다운로드 후 변환 안내
        print(f"\n[INFO] GGUF 버전이 없습니다. 원본 모델을 다운로드합니다.")
        model_path = download_model(config['hf_id'], str(cache_path))

        # 변환 안내
        convert_to_gguf(model_path, str(output_file), config['quantization'])

    # 결과 출력
    print("\n" + "=" * 60)
    print("   결과")
    print("=" * 60)

    if results:
        for name, path in results.items():
            print(f"  [{name}] {path}")

        print("\n[llama.cpp 서버 실행 명령]")
        if "llm" in results:
            print(f"  python scripts/run_llm_server.py --model \"{results['llm']}\" --port 8000")
    else:
        print("  다운로드된 GGUF 모델이 없습니다.")
        print("  수동으로 GGUF 변환이 필요합니다.")


if __name__ == "__main__":
    main()
