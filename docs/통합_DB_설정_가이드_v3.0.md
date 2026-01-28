# 통합 DB 설정 가이드 v3.0

## 메타데이터
- **작성일**: 2026-01-26
- **작성자**: CALL:ACT Team
- **버전**: v3.0
- **상태**: 완료
- **관련 문서**: [CALLACT DB 종합 명세서](./CALLACT_DB_종합_명세서.md), [SETUP.md](./SETUP.md)

---

## 1. 개요

이 가이드는 `01_setup_callact_db.py` 스크립트를 사용하여 CALL:ACT 프로젝트의 전체 데이터베이스를 설정하고 데이터를 적재하는 방법을 안내합니다.

**v3.0 주요 변경사항**:

| 항목 | v1.0 | v3.0 |
|------|------|------|
| 스크립트 구조 | 단일 파일 | 오케스트레이터 + 9개 모듈 |
| 실행 단계 | 9단계 | 12단계 |
| 테이블 수 | 8개 | 16개 |
| employees | 50명 | 70명 (상담팀 60 + IT/관리/교육 10) |
| consultations | ~2,500건 | 6,533건 |
| customers | - | 2,500명 (신규) |
| persona_types | - | 12개 (신규) |
| keyword_dictionary | ~1,456건 | 2,881건 (카드상품명 398건 포함) |
| simulation_results | - | 148건 (Mock, 신규) |
| employee_learning_analytics | - | 70건 (Mock, 신규) |
| recording_download_logs | - | 100건 (Mock, 신규) |
| audit_logs | - | 50건 (Mock, 신규) |
| 재현성 | 미보장 | 보장 (random.seed + hashlib.md5 + 고정 기준일) |

---

## 2. 사전 준비

### 2.1 Docker Desktop 설치 및 실행

Docker Desktop이 설치되어 있어야 합니다.

**Windows**:
1. [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) 다운로드 및 설치
2. 설치 후 Docker Desktop 실행
3. 시스템 트레이에서 Docker 아이콘 확인

**Mac**:
1. [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/) 다운로드 및 설치
2. 설치 후 Docker Desktop 실행
3. 메뉴바에서 Docker 아이콘 확인

**설치 확인**:
```bash
docker --version
docker-compose --version
```

### 2.2 PostgreSQL + pgvector 실행

```bash
# backend/docker 디렉토리로 이동 후 실행
cd backend/docker
docker-compose up -d

# 실행 확인
docker ps
# PORTS 컬럼에 0.0.0.0:5555->5432/tcp가 표시되어야 함
```

**포트 안내**:
- 호스트 포트: `5555` (Windows 동적 포트 예약 범위 5416-5515 회피)
- 컨테이너 내부 포트: `5432` (PostgreSQL 기본)
- 포트 매핑: `호스트:5555 -> 컨테이너:5432`

### 2.3 Conda 환경 설정

```bash
# Conda 환경 활성화
conda activate final_env

# Python 버전 확인 (3.11 이상)
python --version

# 필수 패키지 설치 확인
pip install psycopg2-binary python-dotenv tqdm
```

### 2.4 .env 파일 설정

**위치**: `backend/app/db/scripts/.env`

`.env` 파일은 Git에 포함되지 않으므로 각 개발자가 직접 생성해야 합니다.

```env
DB_HOST=localhost
DB_PORT=5555
DB_USER=callact_admin
DB_PASSWORD=your_password_here
DB_NAME=callact_db
```

**참고**: `.env.example` 파일을 복사하여 사용할 수 있습니다.

### 2.5 Git LFS 설정

큰 데이터 파일(임베딩 JSON 등)을 다운로드하기 위해 Git LFS가 필요합니다.

```bash
# Git LFS 설치 확인
git lfs version

# Git LFS 초기화 (처음 한 번만)
git lfs install

# data-preprocessing 서브모듈에서 LFS 파일 다운로드
cd data-preprocessing
git lfs pull
cd ..
```

---

## 3. 실행 방법

### 3.1 Windows 사용자

```powershell
# 1. Conda 환경 활성화
conda activate final_env

# 2. 스크립트 디렉토리로 이동
cd backend\app\db\scripts

# 3. 통합 스크립트 실행
python 01_setup_callact_db.py
```

**PowerShell 주의사항**:
- 경로 구분자는 `\` (백슬래시) 사용
- `conda activate`가 안 되면 Anaconda Prompt 사용

### 3.2 Mac 사용자

```bash
# 1. Conda 환경 활성화
conda activate final_env

# 2. 스크립트 디렉토리로 이동
cd backend/app/db/scripts

# 3. 통합 스크립트 실행
python 01_setup_callact_db.py
```

### 3.3 완전 초기화 (DB 밀고 새로 시작)

기존 데이터를 모두 삭제하고 처음부터 다시 적재하려면:

```bash
# Docker 컨테이너 중지 및 볼륨 삭제
cd backend/docker
docker-compose down -v

# 컨테이너 재시작
docker-compose up -d

# 잠시 대기 (DB 초기화 시간)
# Windows:
timeout 5
# Mac:
sleep 5

# 스크립트 재실행
cd ../app/db/scripts
python 01_setup_callact_db.py
```

### 3.4 검증만 실행

```bash
cd backend/app/db/scripts
python 01_setup_callact_db.py --verify-only
```

---

## 4. 실행 단계 (12단계)

스크립트는 다음 순서로 실행됩니다:

| 단계 | 설명 | 건수 |
|------|------|------|
| 체크리스트 | 필수 파일, 데이터 파일, 환경변수 확인 | - |
| [1/12] | 기본 DB 스키마 생성 (employees, consultations, consultation_documents) | - |
| [2/12] | 테디카드 테이블 생성 (service_guide_documents, card_products, notices) | - |
| [3/12] | 키워드 사전 테이블 생성 (keyword_dictionary, keyword_synonyms) | - |
| [3-1/12] | 페르소나 유형 테이블 생성 (persona_types) | 12개 유형 |
| [3-2/12] | 고객 테이블 생성 (customers) | - |
| [3-3/12] | 시뮬레이션 교육 테이블 생성 | 5개 시나리오 |
| [3-4/12] | 감사 로그 테이블 생성 | - |
| [3-5/12] | 상담 이력 유의미성 함수/뷰 생성 | 함수 3개, 뷰 4개 |
| [3-6/12] | 카테고리 매핑 데이터 적재 | 55건 |
| [4/12] | 상담사 데이터 적재 (employeesData.json) | 70명 |
| [4-1/12] | 고객 데이터 적재 (customersData.json) | 2,500명 |
| [5/12] | 하나카드 상담 데이터 적재 | 6,533건 |
| [6/12] | 상담사 성과 지표 업데이트 (DB 실데이터 기반) | 60명 업데이트 |
| [6-1/12] | 고객별 상담 통계 업데이트 | 2,500명 업데이트 |
| [7/12] | 키워드 사전 데이터 적재 | 2,881건 |
| [8/12] | 테디카드 데이터 적재 | 1,723건 |
| [9/12] | 시뮬레이션 교육 Mock 데이터 생성 | 148건 + 70건 |
| [10/12] | 감사 로그 Mock 데이터 생성 | 100건 + 50건 |
| [12/12] | 데이터 적재 검증 | - |

---

## 5. 실행 결과 확인

### 5.1 예상 테이블 건수

실행 완료 후 아래 건수와 일치해야 합니다:

| 테이블 | 예상 건수 | 설명 |
|--------|----------|------|
| employees | 70 | 상담팀 60 + IT 5 + 관리 3 + 교육 2 |
| consultations | 6,533 | 하나카드 상담 데이터 |
| consultation_documents | 6,533 | 상담 문서 + 임베딩 |
| category_mappings | 55 | 카테고리 매핑 (57개 중 55개 등록) |
| customers | 2,500 | 고객 데이터 |
| persona_types | 12 | N1-N4 + S1-S8 |
| service_guide_documents | 1,273 | 서비스 가이드 |
| card_products | 398 | 카드 상품 |
| notices | 52 | 공지사항 |
| keyword_dictionary | 2,881 | 키워드 2,483 + 카드상품명 398 |
| keyword_synonyms | 450 | 동의어 |
| simulation_scenarios | 5 | 시나리오 템플릿 |
| simulation_results | 148 | Mock 데이터 |
| employee_learning_analytics | 70 | Mock 데이터 (전원) |
| recording_download_logs | 100 | Mock 데이터 |
| audit_logs | 50 | Mock 데이터 |

### 5.2 DBeaver 검증 쿼리

DBeaver에서 아래 쿼리를 실행하여 검증합니다:

```sql
-- 전체 테이블 건수 확인
SELECT
    schemaname,
    relname AS table_name,
    n_live_tup AS row_count
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY relname;

-- 상담사 부서별 현황
SELECT department, COUNT(*) as cnt
FROM employees
WHERE status = 'active'
GROUP BY department
ORDER BY cnt DESC;

-- 상담 배분 검증 (상담사별)
SELECT agent_id, COUNT(*) as cnt
FROM consultations
GROUP BY agent_id
ORDER BY cnt DESC
LIMIT 10;

-- 요일별 상담 건수 (균등해야 함)
SELECT
    EXTRACT(DOW FROM call_date) as day_of_week,
    to_char(call_date, 'Day') as day_name,
    COUNT(*) as cnt
FROM consultations
GROUP BY day_of_week, day_name
ORDER BY day_of_week;

-- FCR 비율 확인
SELECT
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE fcr = true) as fcr_true,
    ROUND(100.0 * COUNT(*) FILTER (WHERE fcr = true) / COUNT(*), 1) as fcr_rate
FROM consultations;

-- 비상담팀 상담건수 (0이어야 함)
SELECT e.department, COUNT(c.id) as cnt
FROM employees e
LEFT JOIN consultations c ON e.id = c.agent_id
WHERE e.department NOT LIKE '상담%'
GROUP BY e.department;
```

---

## 6. 옵션 사용법

| 옵션 | 설명 |
|------|------|
| `--skip-schema` | 스키마 생성 건너뛰기 (이미 생성된 경우) |
| `--skip-employees` | 상담사 데이터 적재 건너뛰기 |
| `--skip-hana` | 하나카드 데이터 적재 건너뛰기 |
| `--skip-keywords` | 키워드 사전 적재 건너뛰기 |
| `--skip-teddycard` | 테디카드 데이터 적재 건너뛰기 |
| `--verify-only` | 검증만 실행 (데이터 적재 없음) |

**예시**:
```bash
# 스키마만 생성
python 01_setup_callact_db.py --skip-employees --skip-hana --skip-keywords --skip-teddycard

# 테디카드만 적재 (스키마, 상담사, 하나카드 건너뛰기)
python 01_setup_callact_db.py --skip-schema --skip-employees --skip-hana --skip-keywords

# 검증만 실행
python 01_setup_callact_db.py --verify-only
```

---

## 7. 문제 해결 (트러블슈팅)

### 7.1 `.env` 파일 없음

```
[ERROR] 필수 환경 변수가 설정되지 않았습니다
```

**해결**: `backend/app/db/scripts/.env` 파일을 생성합니다. `.env.example`을 복사하여 비밀번호를 설정하세요.

### 7.2 SQL 파일 없음

```
[ERROR] SQL file not found: .../db_setup.sql
```

**해결**: `backend/app/db/scripts/` 디렉토리에 SQL 파일이 있는지 확인합니다. Git pull 후 파일이 누락되었을 수 있습니다.

### 7.3 Docker 연결 실패

```
[ERROR] Failed to connect to database
```

**해결**:
```bash
# Docker 컨테이너 실행 확인
docker ps

# 컨테이너가 없으면 실행
cd backend/docker
docker-compose up -d

# .env의 DB_PORT=5555 확인
```

### 7.4 포트 충돌 (Windows)

```
bind: An attempt was made to access a socket in a way forbidden by its access permissions
```

**해결**:
```cmd
# 예약된 포트 확인
netsh interface ipv4 show excludedportrange protocol=tcp

# 5555가 예약 범위에 있으면 docker-compose.yml에서 다른 포트 사용
```

### 7.5 Git LFS 파일 없음

```
[ERROR] Data file not found: teddycard_service_guides_with_embeddings.json
```

**해결**:
```bash
cd data-preprocessing
git lfs pull
cd ..
```

서브모듈이 아직 초기화되지 않은 경우:
```bash
git submodule update --init --recursive
cd data-preprocessing
git lfs pull
cd ..
```

### 7.6 psycopg2 설치 오류

```
ModuleNotFoundError: No module named 'psycopg2'
```

**해결**:
```bash
conda activate final_env
pip install psycopg2-binary
```

### 7.7 데이터 중복 적재

스크립트는 중복 실행에 안전합니다:
- 테이블 생성: `CREATE TABLE IF NOT EXISTS` 사용
- 데이터 적재: `ON CONFLICT DO UPDATE` 또는 테이블에 데이터가 있으면 스킵
- 완전 초기화가 필요하면 섹션 3.3 참조

---

## 8. 전체 테이블 목록 (16개)

| # | 테이블 | 카테고리 | 건수 |
|---|--------|---------|------|
| 1 | employees | 기본 | 70 |
| 2 | consultations | 기본 | 6,533 |
| 3 | consultation_documents | 기본 | 6,533 |
| 4 | category_mappings | 기본 | 55 |
| 5 | customers | 고객 | 2,500 |
| 6 | persona_types | 고객 | 12 |
| 7 | service_guide_documents | 테디카드 | 1,273 |
| 8 | card_products | 테디카드 | 398 |
| 9 | notices | 테디카드 | 52 |
| 10 | keyword_dictionary | 키워드 | 2,881 |
| 11 | keyword_synonyms | 키워드 | 450 |
| 12 | simulation_scenarios | 시뮬레이션 | 5 |
| 13 | simulation_results | 시뮬레이션 | 148 |
| 14 | employee_learning_analytics | 시뮬레이션 | 70 |
| 15 | recording_download_logs | 감사 로그 | 100 |
| 16 | audit_logs | 감사 로그 | 50 |

---

## 9. 스크립트 구조 (모듈화)

v3.0부터 스크립트가 모듈화되었습니다:

```
backend/app/db/scripts/
  01_setup_callact_db.py     # 오케스트레이터 (유일한 실행 진입점)
  config.py                   # 경로 설정 (dev/prod 전환)
  modules/                    # 모듈 디렉토리
    __init__.py               # 공통 유틸리티, DB 연결, 카테고리 매핑
    schema_runner.py          # SQL 스키마 실행
    load_employees.py         # 상담사 데이터 적재
    load_customers.py         # 고객 데이터 적재
    load_consultations.py     # 상담 데이터 적재 + 배분 + FCR
    update_stats.py           # 성과 지표 + 통계 업데이트
    load_keywords.py          # 키워드 사전 적재
    load_teddycard.py         # 테디카드 데이터 적재
    generate_mock.py          # Mock 데이터 생성
    verify.py                 # 체크리스트 + 검증
  db_setup.sql                # 기본 스키마
  02_setup_tedicard_tables.sql
  03_setup_keyword_dictionary.sql
  07_setup_customers_table.sql
  07a_setup_persona_types_table.sql
  10_setup_simulation_tables.sql
  11_setup_audit_tables.sql
  12_setup_consultation_relevance.sql
  99_verify_db_schema.sql     # DBeaver 수동 검증용
  .env.example
```

**실행 진입점은 `01_setup_callact_db.py` 하나**입니다. 개별 모듈은 직접 실행하지 않습니다.

---

## 10. 재현성 보장

모든 팀원이 동일한 결과를 얻을 수 있도록 다음 기법을 적용했습니다:

| 기법 | 설명 |
|------|------|
| `random.seed(42)` | 모든 모듈에서 랜덤 시드 고정 |
| `hashlib.md5()` | Python 내장 `hash()` 대신 결정적 해시 함수 사용 |
| 고정 기준일 | `datetime.now()` 미사용. `datetime(2026, 1, 19)` 등 고정 날짜 사용 |
| 고정 영업일 | 2026-01-19(월) ~ 2026-01-23(금), 5일 균등 배분 |

**결과**: 누가 어디서 실행해도 동일한 6,533건의 상담 데이터, 동일한 상담사 배분, 동일한 FCR 비율이 생성됩니다.

---

## 11. 체크리스트 (신규 팀원용)

- [ ] Docker Desktop 설치 및 실행
- [ ] `docker-compose up -d` 로 PostgreSQL 컨테이너 실행
- [ ] Conda 환경 활성화 (`final_env`)
- [ ] 필수 패키지 설치 (`psycopg2-binary`, `python-dotenv`, `tqdm`)
- [ ] `.env` 파일 생성 (`backend/app/db/scripts/.env`)
- [ ] Git LFS 초기화 및 `data-preprocessing` LFS 파일 다운로드
- [ ] 서브모듈 초기화 (`git submodule update --init --recursive`)
- [ ] `python 01_setup_callact_db.py` 실행
- [ ] 검증 결과에서 모든 테이블 데이터 확인
- [ ] DBeaver 접속 테스트 (포트 5555)

---

## 12. 참고 문서

- [CALLACT DB 종합 명세서](./CALLACT_DB_종합_명세서.md) - 전체 DB 구조 상세 명세
- [SETUP.md](./SETUP.md) - Backend 설정 가이드
- [통합 DB 설정 가이드 v1.0](./통합_DB_설정_가이드.md) - 이전 버전

---

**마지막 업데이트**: 2026-01-26
