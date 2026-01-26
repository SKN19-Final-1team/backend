## 환경 변수 설정

**⚠️ 중요**: 실제 비밀번호는 `.env` 파일에 저장하고, Git에 커밋하지 마세요.
`.env.example`은 템플릿만 제공합니다.

### 1. .env 파일 생성

프로젝트 루트(`backend/`)에서 `.env.example` 파일을 복사하여 `.env` 파일을 생성하세요:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# 또는 수동으로 생성
```

### 2. .env 파일 내용 수정

`.env` 파일을 열어서 실제 값을 입력하세요:

```env
# .env 파일 (Git에 커밋되지 않음)
# OPENAI
OPENAI_API_KEY=your_openai_api_key_here

# HUGGINGFACE
HF_TOKEN=your_hf_token_here

# RUNPOD
RUNPOD_API_KEY=your_runpod_api_key_here

# R-DB (PostgreSQL)
# 로컬 개발 시: localhost
# 팀원 Docker 서버: 팀원 IP 주소 또는 Tailscale IP (예: 192.168.0.100)
DB_HOST=localhost
DB_PORT=5555  # ⚠️ 중요: 호스트 포트는 5555 사용 (Windows 동적 포트 예약 범위 회피)
DB_USER=callact_admin
DB_PASSWORD=callact_pwd1  # 실제 비밀번호로 변경
DB_NAME=callact_db

# ⚠️ 포트 변경 사항 (2026-01-23):
# - 이전: 호스트 포트 5432 사용
# - 현재: 호스트 포트 5555 사용
# - 이유: Windows 동적 포트 예약 범위(5416-5515)와 충돌 방지
# - 컨테이너 내부 포트: 여전히 5432 사용 (변경 없음)
# - 호스트 포트: 5555 사용 (Docker 포트 매핑: 5555:5432)

# 모델 파일을 저장할 로컬 폴더 경로
MODEL_CACHE_DIR=./model_cache
```

### 3. .env.example 참고

`.env.example` 파일은 템플릿이며, 모든 팀원이 동일한 형식으로 `.env` 파일을 만들 수 있도록 제공됩니다.
실제 비밀번호는 팀원 간에 안전한 방법으로 공유하세요.

**팀원 Docker 서버 사용 시**:
- `DB_HOST`: 팀원의 Docker 서버 IP 주소
- 나머지 정보는 팀원이 공유한 정보 사용

## 실행 방법
```
# 0. 저장소 클론
git clone <repository-url>

# 1. 새 conda 가상환경 생성
conda env create -f environment.yml

# 2. 가상환경 활성화
conda activate final_env

# 3. 라이브러리 설치
pip install -r requirements.txt
```

## 기타
```
# 가상환경 활성화
conda activate final_env

# requirements 최신화
pip freeze > requirements.txt

# conda 가상환경 삭제
conda remove -n 삭제할가상환경 --all

# fastapi 실행
cd app
uvicorn main:app --reload
```
