#!/bin/bash
set -e
export PIP_ROOT_USER_ACTION=ignore

cd /workspace
echo "Starting setup..."

# 필수 패키지 설치
apt-get update
apt-get install -y zstd curl

# Python 환경 설정
pip install --upgrade pip
pip install huggingface-hub

# 모델 다운로드
MODEL_DIR="/workspace/models"
REPO_ID="DevQuasar/kakaocorp.kanana-nano-2.1b-instruct-GGUF"
QUANTIZATION_PATTERN="*.Q4_K_M.gguf"

mkdir -p $MODEL_DIR

echo "Downloading model..."
python3 -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        repo_id='$REPO_ID',
        allow_patterns='$QUANTIZATION_PATTERN',
        local_dir='$MODEL_DIR',
        local_dir_use_symlinks=False
    )
    print('Download completed successfully via Python script.')
except Exception as e:
    print(f'Download failed: {e}')
    exit(1)
"

# 다운로드된 파일 찾기
MODEL_PATH=$(find $MODEL_DIR -name "*.gguf" | head -n 1)

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Could not find any model $QUANTIZATION_PATTERN in $MODEL_DIR"
    exit 1
fi
echo "Successfully downloaded: $MODEL_PATH"

# llama-cpp-python 설치 (CUDA 지원 명시적 재설치)
echo "Installing llama-cpp-python with CUDA..."
CMAKE_ARGS="-DGGML_CUDA=on" pip install 'llama-cpp-python[server]'

# 서버 실행
echo "Starting llama-cpp-python server..."

# 모델 경로 변수 사용
nohup python3 -m llama_cpp.server \
    --model "$MODEL_PATH" \
    --n_gpu_layers -1 \
    --n_ctx 4096 \
    --host 0.0.0.0 \
    --port 8000 \
    --api_key "0211" > /workspace/server.log 2>&1 &

# 서버 헬스 체크
echo "Waiting for Server to start..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    # 프로세스가 살아있는지 확인
    if ! pgrep -f "llama_cpp.server" > /dev/null; then
        echo ""
        echo "Server process died! Printing logs:"
        echo "----------------------------------------"
        cat /workspace/server.log
        echo "----------------------------------------"
        exit 1
    fi
    
    sleep 2
    COUNT=$((COUNT+1))
    echo -n "."
    
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo ""
        echo "Timeout! Server took too long."
        tail -n 20 /workspace/server.log
        exit 1
    fi
done

# 완료 메시지
PUBLIC_IP=$(curl -s ifconfig.me)

echo "--------------------------------------------------------"
echo "Setup Complete. Server is running."
echo ""
echo "   Address: http://${PUBLIC_IP}:<EXTERNAL_PORT>/v1"
echo "   API KEY: 0211"
echo "--------------------------------------------------------"

# 컨테이너 종료 방지
sleep infinity