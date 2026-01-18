# DB 초기화 및 재적재 가이드

**작성일**: 2026-01-13  
**작성자**: CALL:ACT Team  
**버전**: v1.0  
**관련 문서**: [통합 DB 설정 가이드](./11_통합_DB_설정_가이드.md)

---

## 개요

이 가이드는 기존 DB를 완전히 초기화하고 새로운 대분류별 상담사 배분 로직이 적용된 데이터로 재적재하는 방법을 안내합니다.

**이 가이드를 사용하는 경우**:
- 새로운 상담사 배분 로직 적용 시
- DB 데이터를 완전히 새로 시작하고 싶을 때
- 테스트 환경을 깨끗하게 리셋하고 싶을 때

**주의사항**: 
- ⚠️ 이 가이드는 **모든 기존 데이터를 삭제**합니다
- 🛡️ 중요한 데이터가 있다면 **반드시 백업**하세요

---

## 1. DB 초기화 방법

### 방법 1: Docker 컨테이너 재시작 (권장) ⭐

가장 간단하고 안전한 방법입니다.

#### 1.1 현재 실행 중인 컨테이너 확인

```bash
# 현재 위치: backend/docker
docker ps

# 출력 예시:
# CONTAINER ID   IMAGE                    COMMAND                   CREATED      STATUS                PORTS                                         NAMES
# 60fbf6b22328   pgvector/pgvector:pg17   "docker-entrypoint.s…"   7 days ago   Up 4 days (healthy)   0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp   callact_db_container
```

#### 1.2 컨테이너 중지 및 삭제 (데이터 완전 삭제)

```bash
# 현재 위치: backend/docker
docker-compose down -v
```

**옵션 설명**:
- `down`: 컨테이너 중지 및 삭제
- `-v`: 볼륨(데이터)도 함께 삭제 → **완전 초기화**

**출력 예시**:
```
Stopping callact_db_container ... done
Removing callact_db_container ... done
Removing network docker_default
Removing volume docker_postgres_data
```

#### 1.3 컨테이너 재시작 (새로운 빈 DB 생성)

```bash
# 현재 위치: backend/docker
docker-compose up -d
```

**옵션 설명**:
- `up`: 컨테이너 시작
- `-d`: 백그라운드 실행 (detached mode)

#### 1.4 실행 확인

```bash
docker ps

# 출력 예시:
# CONTAINER ID   IMAGE                    COMMAND                   CREATED         STATUS          PORTS                                         NAMES
# abc123def456   pgvector/pgvector:pg17   "docker-entrypoint.s…"   5 seconds ago   Up 4 seconds    0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp   callact_db_container
```

**확인 사항**:
- `callact_db_container`가 목록에 보이면 정상
- `STATUS`가 `Up`이면 정상 실행 중

---

### 방법 2: SQL로 테이블만 삭제 (컨테이너 유지)

Docker 컨테이너는 유지하고 테이블과 데이터만 삭제하는 방법입니다.

#### 2.1 DBeaver 또는 psql 접속

**DBeaver**:
- Host: `localhost`
- Port: `5432`
- Database: `callact_db`
- Username: `callact_admin`
- Password: `callact_pwd1`

**psql**:
```bash
psql -h localhost -U callact_admin -d callact_db
```

#### 2.2 모든 테이블 삭제

```sql
-- 외래키 제약조건을 포함하여 모든 테이블 삭제 (CASCADE)
DROP TABLE IF EXISTS consultation_documents CASCADE;
DROP TABLE IF EXISTS consultations CASCADE;
DROP TABLE IF EXISTS service_guide_documents CASCADE;
DROP TABLE IF EXISTS card_products CASCADE;
DROP TABLE IF EXISTS notices CASCADE;
DROP TABLE IF EXISTS keyword_dictionary CASCADE;
DROP TABLE IF EXISTS keyword_synonyms CASCADE;
DROP TABLE IF EXISTS employees CASCADE;

-- 삭제 확인
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
-- 결과가 없으면 모든 테이블이 삭제된 것입니다.
```

#### 2.3 pgvector 확장 확인

```sql
-- pgvector 확장 확인
SELECT extname FROM pg_extension WHERE extname = 'vector';

-- 결과가 없으면 확장 재설치
CREATE EXTENSION IF NOT EXISTS vector;
```

---

### 방법 3: DB 자체 삭제 및 재생성 (완전 초기화)

가장 강력한 방법이지만, 다른 연결이 있으면 실행할 수 없습니다.

#### 3.1 다른 DB에 연결 (postgres)

```bash
psql -h localhost -U callact_admin -d postgres
```

#### 3.2 DB 삭제 및 재생성

```sql
-- 기존 DB 삭제 (활성 연결이 있으면 실패할 수 있음)
DROP DATABASE IF EXISTS callact_db;

-- 새 DB 생성
CREATE DATABASE callact_db;

-- 새 DB로 전환
\c callact_db

-- pgvector 확장 설치
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 2. 재적재 실행 순서

### 2.1 환경 확인

```bash
# 1. Conda 환경 활성화
conda activate final_env

# 2. 작업 디렉토리 이동
cd backend/app/db/scripts

# 3. Python 버전 확인 (3.11 이상)
python --version

# 4. .env 파일 확인
# 파일 위치: backend/app/db/scripts/.env
# 필수 변수: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
```

### 2.2 통합 스크립트 실행

```bash
# 전체 설정 및 데이터 적재 (한 번에 실행)
python 01_setup_callact_db.py
```

**실행 순서** (자동으로 진행됨):

1. **[1/9] 기본 스키마 생성**: `employees`, `consultations`, `consultation_documents` 테이블 생성
2. **[2/9] 테디카드 테이블 생성**: `service_guide_documents`, `card_products`, `notices` 테이블 생성
3. **[3/9] 키워드 사전 테이블 생성**: `keyword_dictionary`, `keyword_synonyms` 테이블 생성
4. **[4/9] 상담사 데이터 적재**: `employeesData.json` → `employees` 테이블 (EMP-001~EMP-070, 70명)
   - 성과 지표 컬럼 포함 (`consultations`, `fcr`, `avgTime`, `rank`)
   - 초기값은 `employeesData.json`에서 로딩
5. **[5/9] 하나카드 데이터 적재**: 대분류별 상담사 배분 적용 (6,533건)
   - 대분류별 풀 생성
   - 하이브리드 배분 (90% 순차 + 10% 랜덤)
6. **[6/9] 상담사 성과 지표 업데이트**: DB 실제 데이터 기반 성과 지표 계산 및 업데이트
   - `consultations`: agent_id별 실제 상담 건수
   - `fcr`: First Call Resolution 비율 (0-100)
   - `avgTime`: 평균 상담 시간 ("MM:SS" 형식)
   - `rank`: 성과 순위 (consultations DESC, fcr DESC, avgTime ASC 기준)
7. **[7/9] 키워드 사전 데이터 적재**: `keywords_dict_*.json` → `keyword_dictionary` 테이블
8. **[8/9] 테디카드 데이터 적재**: `teddycard_*.json` → 각 테이블
9. **[9/9] 데이터 적재 검증**: 통합 검증 (상담사별/대분류별 분포 확인 포함)

### 2.3 실행 결과 확인

스크립트가 자동으로 다음을 검증합니다:

#### 2.3.1 테이블 존재 확인

```
[1/4] 테이블 존재 확인
  ✅ 모든 테이블 존재: 8개
```

#### 2.3.2 데이터 개수 확인

```
[2/4] 데이터 개수 확인 및 검증
  ✅ employees: 70건 (EMP-001 ~ EMP-070)
  ✅ consultations: 6,533건
  ✅ consultation_documents: 6,533건
  ✅ service_guide_documents: XXX건
  ✅ card_products: XXX건
  ✅ notices: XXX건
  ✅ keyword_dictionary: XXX건
  ✅ keyword_synonyms: XXX건
```

**참고**: `employees` 테이블의 성과 지표(`consultations`, `fcr`, `avgTime`, `rank`)는 하나카드 데이터 적재 후 자동으로 업데이트됩니다.

#### 2.3.3 스키마 확인

```
[3/4] 스키마 확인
  ✅ pgvector 확장: 설치됨
  ✅ idx_consultation_documents_embedding_hnsw: 존재
  ✅ idx_service_guide_documents_embedding_hnsw: 존재
  ✅ idx_notices_embedding_hnsw: 존재
```

#### 2.3.4 상담사별/대분류별 분포 확인 (신규)

```
[4/4] 상담사별/대분류별 분포 확인

  상담사별 상담 건수 (총 70명):
    평균: XXX건/명, 표준편차: XXX
    상위 5명:
      - EMP-XXX: XXX건 (FCR: XX%, 평균 시간: XX:XX)
      - EMP-XXX: XXX건 (FCR: XX%, 평균 시간: XX:XX)
      - EMP-XXX: XXX건 (FCR: XX%, 평균 시간: XX:XX)
      - EMP-XXX: XXX건 (FCR: XX%, 평균 시간: XX:XX)
      - EMP-XXX: XXX건 (FCR: XX%, 평균 시간: XX:XX)
    하위 5명:
      - EMP-XXX: XXX건
      - EMP-XXX: XXX건
      - EMP-XXX: XXX건
      - EMP-XXX: XXX건
      - EMP-XXX: XXX건
    ✅ 상담사별 배분이 비교적 균등합니다. (변동계수: X.XX)
    
  참고: 성과 지표(consultations, fcr, avgTime, rank)는 DB 실제 데이터 기반으로 자동 계산되었습니다.

  세부 카테고리별 상담 건수 (총 57개):
    ...

  대분류별 상담 건수:
    - 결제/승인: 1,559건
    - 이용내역: 919건
    - 한도: 564건
    - 분실/도난: 398건
    - 포인트/혜택: 223건
    - 정부지원: 167건
    - 수수료/연체: 163건
    - 기타: XXX건

  대분류별 상담사 분포:
    - 결제/승인: 10명 담당 (예상: 10명)
      ✅ 풀 크기가 예상과 일치합니다.
    - 이용내역: 7명 담당 (예상: 7명)
      ✅ 풀 크기가 예상과 일치합니다.
    - 한도: 6명 담당 (예상: 6명)
      ✅ 풀 크기가 예상과 일치합니다.
    ...
```

---

## 3. 옵션: 단계별 실행

전체가 아닌 특정 단계만 실행하려면:

```bash
# 스키마만 생성
python 01_setup_callact_db.py --skip-employees --skip-hana --skip-keywords --skip-teddycard

# 상담사 + 하나카드만 적재
python 01_setup_callact_db.py --skip-schema --skip-keywords --skip-teddycard

# 검증만 실행
python 01_setup_callact_db.py --verify-only
```

**옵션 설명**:
- `--skip-schema`: 스키마 생성 건너뛰기
- `--skip-employees`: 상담사 데이터 적재 건너뛰기
- `--skip-hana`: 하나카드 데이터 적재 건너뛰기
- `--skip-keywords`: 키워드 사전 적재 건너뛰기
- `--skip-teddycard`: 테디카드 데이터 적재 건너뛰기
- `--verify-only`: 검증만 실행

---

## 4. 실행 시간 예상

| 단계 | 예상 시간 | 비고 |
|------|-----------|------|
| 스키마 생성 | 약 1분 | 테이블 및 인덱스 생성 |
| 상담사 데이터 적재 | 즉시 | 70건, 매우 빠름 |
| 하나카드 데이터 적재 | 약 2-3분 | 6,533건, 상담사 배분 포함 |
| 상담사 성과 지표 업데이트 | 약 10초 | DB 실제 데이터 기반 계산 |
| 키워드 사전 적재 | 약 1분 | 키워드 및 동의어 적재 |
| 테디카드 데이터 적재 | 약 1-2분 | 서비스 가이드, 카드 상품, 공지사항 |
| 검증 | 약 10초 | 통계 계산 및 확인 |
| **총 예상 시간** | **약 5-7분** | 전체 프로세스 |

---

## 5. 주의사항

### 5.1 데이터 백업

기존 데이터가 중요한 경우 반드시 백업하세요:

```bash
# 방법 1: pg_dump 사용
pg_dump -h localhost -U callact_admin -d callact_db > backup_$(date +%Y%m%d_%H%M%S).sql

# 방법 2: DBeaver Export Data
# DBeaver → Table → Export Data → CSV/JSON 선택
```

### 5.2 EMP-TEDDY-DEFAULT 처리

- 기존 환경에 `EMP-TEDDY-DEFAULT`가 있을 수 있습니다
- 새 로직에서는 사용하지 않으므로 삭제해도 됩니다
- 단, 기존 데이터와의 호환성을 위해 그대로 두어도 문제 없습니다

### 5.3 .env 파일 확인

반드시 다음 변수가 올바르게 설정되어 있어야 합니다:

```env
DB_HOST=localhost
DB_PORT=5432
DB_USER=callact_admin
DB_PASSWORD=callact_pwd1
DB_NAME=callact_db
```

---

## 6. 문제 해결

### 6.1 오류: "상담사 데이터가 없습니다"

**원인**: 하나카드 데이터 적재는 상담사 데이터가 먼저 필요합니다.

**해결**:
```bash
# 상담사 데이터를 먼저 적재
python 01_setup_callact_db.py --skip-schema --skip-hana --skip-keywords --skip-teddycard

# 그 다음 하나카드 데이터 적재
python 01_setup_callact_db.py --skip-schema --skip-employees --skip-keywords --skip-teddycard
```

### 6.2 오류: "파일을 찾을 수 없습니다"

**원인**: 데이터 파일 경로가 잘못되었습니다.

**해결**:
- `config.py`의 경로 설정 확인
- `data-preprocessing/data/` 하위 파일 존재 확인
- `backend/app/db/data/employeesData.json` 존재 확인

### 6.3 경고: "상담사별 배분이 불균등합니다"

**원인**: 랜덤 배분(10%)으로 인한 자연스러운 변동입니다.

**해결**:
- 변동계수(표준편차/평균)가 0.5 이하면 정상 범위입니다
- 0.5 이상인 경우에만 경고가 표시됩니다

### 6.4 오류: "컨테이너를 찾을 수 없습니다"

**원인**: Docker 컨테이너가 실행되지 않았습니다.

**해결**:
```bash
# 컨테이너 시작
cd backend/docker
docker-compose up -d

# 실행 확인
docker ps
```

### 6.5 오류: "pgvector 확장이 설치되지 않음"

**원인**: DB 초기화 시 pgvector 확장이 누락되었습니다.

**해결**:
```sql
-- DBeaver 또는 psql에서 실행
CREATE EXTENSION IF NOT EXISTS vector;

-- 확인
SELECT extname FROM pg_extension WHERE extname = 'vector';
```

---

## 7. 추천 실행 순서 (Quick Start)

```bash
# ===== 1단계: DB 초기화 =====
cd backend/docker

# 컨테이너 중지 및 삭제 (데이터 완전 삭제)
docker-compose down -v

# 컨테이너 재시작 (새로운 빈 DB 생성)
docker-compose up -d

# 실행 확인
docker ps


# ===== 2단계: 환경 확인 =====
# Conda 환경 활성화
conda activate final_env

# 작업 디렉토리 이동
cd ../../app/db/scripts

# .env 파일 확인
# (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)


# ===== 3단계: 통합 스크립트 실행 =====
python 01_setup_callact_db.py


# ===== 4단계: 결과 확인 =====
# 스크립트가 자동으로 모든 검증을 수행합니다
# 마지막에 "[4/4] 상담사별/대분류별 분포 확인" 섹션에서 결과 확인
```

---

## 8. 관련 문서

- [통합 DB 설정 가이드](./11_통합_DB_설정_가이드.md): 전체 DB 설정 및 데이터 적재 가이드
- [Git 커밋 가이드](./12_Git_커밋_가이드.md): 변경사항을 팀 레포에 커밋하는 방법
- [개발환경 설정 가이드](../02_db/01_개발환경_설정_가이드.md): Docker 및 PostgreSQL 설치 가이드

---

**문서 버전**: v1.0  
**최종 업데이트**: 2026-01-13
