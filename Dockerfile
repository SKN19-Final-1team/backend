# syntax=docker/dockerfile:1

# 1단계: 베이스 이미지
FROM python:3.11-slim as base

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    portaudio19-dev \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 2단계: 의존성 설치
COPY requirements.txt .

# BuildKit 캐시 마운트로 pip 다운로드 캐시 재사용
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# 3단계: 애플리케이션 코드 복사
COPY . .

# 모델 캐시 디렉토리 생성
RUN mkdir -p /app/model_cache

# 4단계: 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# 포트 노출
EXPOSE 8000

# 5단계: 실행 명령어
# 프로덕션: gunicorn + uvicorn worker (더 안정적)
# 개발: uvicorn만 사용
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# 참고: gunicorn을 사용하려면 requirements.txt에 gunicorn 추가 후 아래 명령어 사용
# CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]
