# 하나카드 데이터 DB 적재 스크립트

## 개요

하나카드 전처리 데이터를 PostgreSQL + pgvector 데이터베이스에 적재하는 스크립트 모음입니다.

## 파일 구조

```
backend/app/db/scripts/
├── 01_setup_db.py                  # DB 스키마 생성
├── 02_generate_embeddings_hana.py  # 임베딩 생성 (선택)
├── 03_load_hana_to_db.py           # DB 적재
├── 04_verify_db_load.py            # 검증
├── db_setup.sql                    # DB 스키마 SQL 파일
└── README.md                       # 이 파일
```

## 사전 준비

### 1. 환경 설정

**상세 가이드**: `docs/04_dev/02_db/01_개발환경_설정_가이드.md` 참고

**요약**:
1. Conda 환경 생성 (`conda env create -f ../environment.yml`)
2. 패키지 설치 (`pip install -r ../requirements.txt`)
3. Docker로 PostgreSQL + pgvector 실행
4. `.env` 파일 설정

### 2. DB 스키마 생성

```bash
# DBeaver에서 실행 또는
psql -h localhost -U callact_admin -d callact_db -f backend/app/db/scripts/db_setup.sql
```

## 실행 순서

**⚠️ 참고**: 이미 임베딩 완료된 파일(`hana_vectordb_with_embeddings.json`)이 있는 경우, 1단계(임베딩 생성)는 건너뛰고 2단계(DB 적재)부터 진행하세요.

### 1단계: 임베딩 생성 (선택 사항)

**이미 임베딩 파일이 있는 경우**: 이 단계는 건너뛰세요.

임베딩 파일이 없는 경우에만 실행:

**테스트용 (100-200개)**:
```bash
conda activate final_env
cd backend/app/db/scripts
python 02_generate_embeddings_hana.py --limit 200
```

**전체 데이터**:
```bash
python 02_generate_embeddings_hana.py
```

**재시작 (중단된 경우)**:
```bash
python 02_generate_embeddings_hana.py --resume
```

**출력 파일**: `data-preprocessing/data/hana/hana_vectordb_with_embeddings.json`

### 2단계: DB 적재

**테스트용 (100-200개)**:
```bash
python 03_load_hana_to_db.py --limit 200
```

**전체 데이터**:
```bash
python 03_load_hana_to_db.py
```

**옵션**:
- `--skip-consultations`: consultations 테이블 적재 건너뛰기
- `--skip-documents`: consultation_documents 테이블 적재 건너뛰기

### 3단계: 검증

```bash
python 04_verify_db_load.py
```

**유사도 검색 테스트 포함**:
```bash
python 04_verify_db_load.py --test-search "카드 분실 신고"
```

## 실행 예시

### 전체 파이프라인 (테스트용)

**이미 임베딩 파일이 있는 경우**:
```bash
# 1. DB 적재 (200개)
python 03_load_hana_to_db.py --limit 200

# 2. 검증
python 04_verify_db_load.py
```

**임베딩 파일이 없는 경우**:
```bash
# 1. 임베딩 생성 (200개)
python 02_generate_embeddings_hana.py --limit 200

# 2. DB 적재 (200개)
python 03_load_hana_to_db.py --limit 200

# 3. 검증
python 04_verify_db_load.py
```

### 전체 데이터 처리

**이미 임베딩 파일이 있는 경우**:
```bash
# 1. DB 적재 (전체, 약 10분 소요)
python 03_load_hana_to_db.py

# 2. 검증
python 04_verify_db_load.py
```

**임베딩 파일이 없는 경우**:
```bash
# 1. 임베딩 생성 (전체, 약 40분 소요)
python 02_generate_embeddings_hana.py

# 2. DB 적재 (전체, 약 10분 소요)
python 03_load_hana_to_db.py

# 3. 검증
python 04_verify_db_load.py
```

## 환경 변수

**⚠️ 중요**: 
- **팀원**: 프로젝트 루트에 `.env` 파일 생성 (`call-act/.env`)
- **개인 개발**: `backend/app/db/scripts/.env` 파일 생성 가능 (우선 사용, 개인 파일)
- 스크립트는 다음 순서로 `.env` 파일을 읽습니다:
  1. `backend/app/db/scripts/.env` (있으면 우선 사용)
  2. 프로젝트 루트 `.env` (팀원 공용)

**참고**: 개인 개발 환경(`backend`)에서는 `backend/app/db/scripts/.env`가 우선 사용됩니다.

### .env 파일 생성 (팀원용)

**프로젝트 루트에 `.env` 파일 생성**:

```bash
# Windows PowerShell (프로젝트 루트에서)
cd C:\Users\AI-WS01\projects\call-act
Copy-Item backend\.env.example .env

# 또는 수동으로 생성
```

### .env 파일 내용 (프로젝트 루트)

```env
# PostgreSQL Database Configuration
# 로컬 개발 시: localhost
# 팀원 Docker 서버: 팀원 IP 주소 또는 Tailscale IP
DB_HOST=localhost
DB_PORT=5432
DB_USER=callact_admin
DB_PASSWORD=callact_pwd1
DB_NAME=callact_db

# OpenAI API Key (임베딩 생성 시 필요)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSION=1536

# 배치 처리 설정 (선택사항)
DB_LOAD_BATCH_SIZE=100
DB_COMMIT_INTERVAL=500
EMBEDDING_BATCH_SIZE=100
```

**⚠️ 참고**: 
- `DB_HOST_IP`는 사용하지 않습니다 (팀원의 Tailscale IP는 무시)
- 실제 비밀번호는 `.env` 파일에 저장하고 Git에 커밋하지 마세요
- `backend/app/db/scripts/.env`는 로컬 설정 파일이므로 필요시 생성

## 트러블슈팅

### 임베딩 생성 중 API 에러

**증상**: `openai.RateLimitError`

**해결**:
- 재시도 로직이 자동으로 처리
- `EMBEDDING_REQUEST_DELAY` 증가 (기본값: 0.5초)

### DB 연결 실패

**증상**: `psycopg2.OperationalError`

**해결**:
1. Docker 컨테이너 실행 확인: `docker ps`
2. `.env` 파일의 DB 정보 확인
3. 포트 충돌 확인: `5432` 포트 사용 중인지 확인

### Foreign Key 제약 조건 위반

**증상**: `psycopg2.IntegrityError: insert or update on table "consultation_documents" violates foreign key constraint`

**해결**:
- `consultations` 테이블을 먼저 적재해야 함
- `--skip-consultations` 옵션 사용하지 않기

## 참고 문서

- 개발 환경 설정: `docs/04_dev/02_db/01_개발환경_설정_가이드.md`
- DB 적재 설계: `docs/04_dev/02_db/00_hana_db_loading_설계.md`
- ERD 스키마: `data-preprocessing/docs/erd_diagram/CALL_ACT_ERD_Schema_설명.md`


