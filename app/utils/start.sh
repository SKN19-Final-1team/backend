#!/bin/bash
set -e

cd /workspace
echo "Starting setup"

# 필수 패키지 설치
apt-get update
apt-get install -y zstd curl

# HuggingFace 설치
pip install --upgrade pip
pip install --user huggingface-hub
export PATH="$HOME/.local/bin:$PATH"

# PATH 영구 등록 (중복 방지)
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# llama-cpp-python 설치
echo "Installing llama-cpp-python with CUDA..."
CMAKE_ARGS="-DGGML_CUDA=on" pip install 'llama-cpp-python[server]'

# 모델 다운로드
# 지정된 리포지토리에서 4bit 양자화가 적용된 GGUF 파일 추적
MODEL_DIR="/workspace/models"
REPO_ID="DevQuasar/kakaocorp.kanana-nano-2.1b-instruct-GGUF"
QUANTIZATION_PATTERN="*.Q4_K_M.gguf"

mkdir -p $MODEL_DIR

echo "Downloading '${QUANTIZATION_PATTERN}'..."
python3 -m huggingface_hub.cli download $REPO_ID \
    --include "$QUANTIZATION_PATTERN" \
    --local-dir $MODEL_DIR \
    --local-dir-use-symlinks False

# 다운로드 성공 체크
# 해당 디렉토리에서 패턴에 맞는 파일 하나를 탐색
MODEL_PATH=$(find $MODEL_DIR -name "$QUANTIZATION_PATTERN" | head -n 1)

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Could not find any model $QUANTIZATION_PATTERN in $MODEL_DIR"
    exit 1
fi
echo "Successfully downloaded: $MODEL_PATH"

# 서버 실행 (OpenAI 호환)
echo "Starting llama-cpp-python server..."
nohup python3 -m llama_cpp.server \
    --model "$MODEL_DIR/$MODEL_FILE" \
    --n_gpu_layers -1 \
    --n_ctx 4096 \
    --n_parallel 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --api_key "0211" > /workspace/server.log 2>&1 &

# 서버가 뜰 때까지 대기
echo "Waiting for Server to start..."
until curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 2
    echo "Loading..."
done

# 서버 정상 실행
echo "--------------------------------------------------------"
echo "Setup Complete. Server is running."
echo "--------------------------------------------------------"

PUBLIC_IP=$(curl -s ifconfig.me)

echo ""
echo "   http://${PUBLIC_IP}:<EXTERNAL_PORT>/v1"
echo ""

# 컨테이너 종료 방지
sleep infinity