# CALL:ACT DB 종합 명세서

## 메타데이터
- **작성일**: 2026-01-26
- **작성자**: CALL:ACT Team
- **버전**: v1.0
- **상태**: 완료
- **관련 문서**: [통합 DB 설정 가이드 v3.0](./통합_DB_설정_가이드_v3.0.md), [SETUP.md](./SETUP.md)

---

## 목적

이 문서는 CALL:ACT 프로젝트의 전체 데이터베이스 구조를 종합적으로 명세합니다. 테이블 정의, 함수/뷰, 데이터 품질 설계, 스크립트 구조, ERD를 포함하여 DB의 전체 아키텍처를 이해할 수 있도록 합니다.

---

## 배경

CALL:ACT는 카드사 콜센터 상담사를 위한 실시간 상담 지원 + 시뮬레이션 교육 플랫폼입니다. 하나카드 상담 데이터(6,533건)를 기반으로 테디카드(가상 카드사)의 서비스 가이드, 카드 상품, 공지사항, 키워드 사전 데이터를 결합하여 RAG 기반의 상담 지원 시스템을 제공합니다.

**기술 스택**: PostgreSQL 17 + pgvector (벡터 유사도 검색)

---

## 1. DB 아키텍처

| 항목 | 값 |
|------|-----|
| DBMS | PostgreSQL 17 |
| 확장 | pgvector (벡터 유사도 검색) |
| 호스트 | Docker 컨테이너 (`callact_db_container`) |
| 포트 | 호스트 5555 → 컨테이너 5432 |
| DB명 | callact_db |
| 스키마 | public |
| 테이블 수 | 16개 (15 + category_mappings) |
| 함수 | 3개 |
| 뷰 | 4개 |

---

## 2. 테이블 구조

### 2.1 기본 테이블

#### employees (직원/상담사)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | VARCHAR(50) PK | 직원 ID (예: `EMP001`) |
| name | VARCHAR(100) | 이름 |
| email | VARCHAR(100) UNIQUE | 이메일 |
| role | VARCHAR(50) | 직급 (사원, 대리, 과장 등) |
| department | VARCHAR(100) | 부서 (상담1팀~상담6팀, IT팀, 관리팀, 교육팀) |
| hire_date | DATE | 입사일 |
| status | status_type | 상태 (active, inactive, suspended, vacation) |
| consultations | INTEGER | 상담 건수 (DB 실데이터 기반 업데이트) |
| fcr | INTEGER | FCR 비율 0-100 (DB 실데이터 기반) |
| "avgTime" | VARCHAR(20) | 평균 상담 시간 "MM:SS" (DB 실데이터 기반) |
| rank | INTEGER | 성과 순위 (DB 실데이터 기반) |
| created_at | TIMESTAMP | 생성일 |
| updated_at | TIMESTAMP | 수정일 |

- **데이터 건수**: 70명 (상담팀 60명 + IT팀 5명 + 관리팀 3명 + 교육팀 2명)
- **인덱스**: role, department, status
- **비고**: IT/관리/교육팀은 상담 배정 대상에서 제외. 성과 지표(consultations, fcr, avgTime, rank)는 consultations 테이블 실데이터 기반으로 자동 계산됨.

#### consultations (상담 이력)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | VARCHAR(100) PK | 상담 ID (예: `CS-EMP001-202501191432`) |
| agent_id | VARCHAR(50) FK | 상담사 ID → employees(id) |
| customer_id | VARCHAR(50) FK | 고객 ID → customers(id) |
| call_date | DATE | 상담 날짜 |
| call_time | VARCHAR(20) | 상담 시작 시간 "HH:MM" |
| call_duration | VARCHAR(20) | 상담 시간 "MM:SS" |
| category_raw | VARCHAR(200) | 원본 카테고리 (57개) |
| category_main | VARCHAR(50) | 대분류 (8개) |
| category_sub | VARCHAR(50) | 중분류 (15개) |
| customer_name | VARCHAR(100) | 고객명 |
| customer_phone | VARCHAR(20) | 고객 전화번호 |
| status | consultation_status | 상태 (completed, in_progress, incomplete) |
| emotion | emotion_type | 감정 (positive, neutral, negative) |
| keywords | TEXT[] | 키워드 배열 |
| summary | TEXT | 상담 요약 |
| satisfaction | INTEGER | 만족도 (1-5) |
| quality | quality_rating | 품질 (high, medium, low) |
| fcr | BOOLEAN | FCR 여부 (7일 이내 동일 카테고리 재문의 = FALSE) |
| processing_timeline | JSONB | 처리 타임라인 |
| transcript | TEXT | 전체 녹취록 (마스킹 처리 완료) |
| created_at | TIMESTAMP | 생성일 |
| updated_at | TIMESTAMP | 수정일 |

- **데이터 건수**: 6,533건
- **인덱스**: agent_id, customer_id, call_date, category_raw, category_main, emotion, status
- **비고**: 하나카드 상담 데이터를 테디카드 맥락으로 변환하여 적재. 마스킹 태그(`[카드사명#1]` 등)를 `테디카드`, `테디페이`로 치환.

#### consultation_documents (상담 문서 + 임베딩)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | VARCHAR(100) PK | 문서 ID (상담 ID와 동일) |
| consultation_id | VARCHAR(100) FK | 상담 ID → consultations(id) |
| title | VARCHAR(500) | 제목 |
| content | TEXT | 전체 내용 |
| category | VARCHAR(200) | 카테고리 |
| keywords | TEXT[] | 키워드 배열 |
| summary | TEXT | 요약 |
| embedding | vector(1536) | OpenAI 임베딩 벡터 |
| metadata | JSONB | 메타데이터 |
| quality | quality_rating | 품질 |
| structured | JSONB | 구조화 데이터 |
| created_at | TIMESTAMP | 생성일 |
| updated_at | TIMESTAMP | 수정일 |

- **데이터 건수**: 6,533건 (consultations와 1:1 매핑)
- **인덱스**: consultation_id, category, HNSW 임베딩 인덱스 (cosine)
- **비고**: pgvector의 HNSW 인덱스를 사용하여 RAG 유사도 검색 지원.

#### category_mappings (카테고리 매핑)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | SERIAL PK | 자동 증가 ID |
| category_raw | VARCHAR(200) UNIQUE | 원본 카테고리 (57개) |
| category_main | VARCHAR(50) | 대분류 (8개) |
| category_sub | VARCHAR(50) | 중분류 (15개) |
| keywords | TEXT[] | 관련 키워드 |
| created_at | TIMESTAMP | 생성일 |

- **데이터 건수**: 55건
- **비고**: 57개 원본 카테고리를 8개 대분류 + 15개 중분류로 매핑. `__init__.py`의 `CATEGORY_MAPPINGS` 딕셔너리와 동기화.

**대분류 8개**:

| 대분류 | 건수(약) | 담당 풀(명) |
|--------|---------|------------|
| 결제/승인 | 1,559 | 12 |
| 기타 | 1,538 | 전체(60) |
| 이용내역 | 919 | 9 |
| 한도 | 564 | 7 |
| 분실/도난 | 400 | 6 |
| 포인트/혜택 | 223 | 4 |
| 정부지원 | 167 | 3 |
| 수수료/연체 | 163 | 4 |

**중분류 15개**: 조회/안내, 신청/등록, 변경, 취소/해지, 처리/실행, 발급, 확인서, 배송, 즉시출금, 상향/증액, 이체/전환, 환급/반환, 정지/해제, 결제일, 기타

---

### 2.2 고객 테이블

#### customers (고객)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | VARCHAR(50) PK | 고객 ID (예: `CUST-00001`) |
| name | VARCHAR(100) | 이름 |
| phone | VARCHAR(20) | 전화번호 |
| gender | VARCHAR(10) | 성별 (male, female, unknown) |
| age_group | VARCHAR(10) | 연령대 (20대, 30대, ...) |
| birth_date | DATE | 생년월일 |
| address | VARCHAR(200) | 주소 |
| grade | VARCHAR(20) | 등급 (GENERAL, SILVER, GOLD, VIP, VVIP) |
| card_type | VARCHAR(50) | 카드 유형 |
| card_number_last4 | VARCHAR(4) | 카드번호 뒷4자리 |
| card_brand | VARCHAR(30) | 카드 브랜드 |
| card_issue_date | DATE | 카드 발급일 |
| card_expiry_date | DATE | 카드 만료일 |
| current_type_code | VARCHAR(10) | 현재 페르소나 코드 (N1~N4, S1~S8) |
| type_history | JSONB | 페르소나 이력 |
| personality_tags | TEXT[] | 성격 태그 배열 |
| communication_style | JSONB | 커뮤니케이션 스타일 |
| customer_type_codes | TEXT[] | 고객 유형 코드 배열 |
| llm_guidance | TEXT | LLM 상담 가이던스 |
| total_consultations | INTEGER | 총 상담 건수 |
| resolved_first_call | INTEGER | FCR 해결 건수 |
| last_consultation_date | DATE | 마지막 상담일 |
| created_at | TIMESTAMP | 생성일 |
| updated_at | TIMESTAMP | 수정일 |

- **데이터 건수**: 2,500명
- **인덱스**: name, phone, grade, current_type_code, age_group
- **비고**: 페르소나 기반 고객 프로필. `total_consultations`, `resolved_first_call`, `last_consultation_date`는 상담 데이터 적재 후 자동 업데이트.

#### persona_types (페르소나 유형)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| type_code | VARCHAR(10) PK | 유형 코드 (N1~N4, S1~S8) |
| type_name | VARCHAR(50) | 유형명 |
| description | TEXT | 설명 |
| communication_tips | TEXT | 커뮤니케이션 팁 |
| is_active | BOOLEAN | 활성 여부 |
| created_at | TIMESTAMP | 생성일 |
| updated_at | TIMESTAMP | 수정일 |

- **데이터 건수**: 12개 유형
- **유형 목록**:

| 코드 | 유형명 | 설명 |
|------|--------|------|
| N1 | 일반-빠른해결형 | 빠른 문제 해결 선호 |
| N2 | 일반-꼼꼼한형 | 상세 설명 선호 |
| N3 | 일반-친근한형 | 친근한 대화 선호 |
| N4 | 일반-무관심형 | 최소한의 대화 선호 |
| S1 | 특수-VIP고객 | VIP 등급 고객 |
| S2 | 특수-고령자 | 고령 고객 (느린 설명) |
| S3 | 특수-외국인 | 외국인 고객 |
| S4 | 특수-장애인 | 장애인 고객 |
| S5 | 특수-법인고객 | 법인/기업 고객 |
| S6 | 특수-민원고객 | 불만/민원 고객 |
| S7 | 특수-금융취약계층 | 금융 이해도 낮은 고객 |
| S8 | 특수-긴급상황 | 분실/도난 등 긴급 상황 |

---

### 2.3 테디카드 테이블

#### service_guide_documents (서비스 가이드 문서)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | VARCHAR(200) PK | 문서 ID |
| document_type | VARCHAR(50) | 문서 유형 (service_guide 등) |
| category | VARCHAR(100) | 카테고리 |
| title | VARCHAR(500) | 제목 |
| content | TEXT | 내용 |
| keywords | TEXT[] | 키워드 배열 |
| embedding | vector(1536) | 임베딩 벡터 |
| metadata | JSONB | 메타데이터 |
| document_source | VARCHAR(200) | 문서 출처 |
| priority | VARCHAR(20) | 우선순위 (normal, high, urgent) |
| usage_count | INTEGER | 사용 횟수 |
| last_used | TIMESTAMP | 마지막 사용일 |
| structured | JSONB | 구조화 데이터 |
| created_at | TIMESTAMP | 생성일 |
| updated_at | TIMESTAMP | 수정일 |

- **데이터 건수**: 1,273건
- **인덱스**: category, document_type, priority, HNSW 임베딩 인덱스

#### card_products (카드 상품)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | VARCHAR(200) PK | 카드 상품 ID |
| name | VARCHAR(200) | 카드 상품명 |
| card_type | VARCHAR(50) | 카드 유형 (credit, debit, prepaid) |
| brand | VARCHAR(50) | 브랜드 (local, visa, mastercard 등) |
| annual_fee_domestic | INTEGER | 국내 연회비 |
| annual_fee_global | INTEGER | 해외 연회비 |
| performance_condition | TEXT | 실적 조건 |
| main_benefits | TEXT | 주요 혜택 |
| status | VARCHAR(20) | 상태 (active, discontinued) |
| metadata | JSONB | 메타데이터 |
| structured | JSONB | 구조화 데이터 |
| created_at | TIMESTAMP | 생성일 |
| updated_at | TIMESTAMP | 수정일 |

- **데이터 건수**: 398건
- **인덱스**: card_type, brand, status

#### notices (공지사항)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | VARCHAR(200) PK | 공지사항 ID |
| title | VARCHAR(500) | 제목 |
| content | TEXT | 내용 |
| category | VARCHAR(50) | 카테고리 (system, service, event, policy) |
| priority | VARCHAR(20) | 우선순위 |
| is_pinned | BOOLEAN | 상단 고정 여부 |
| start_date | DATE | 게시 시작일 |
| end_date | DATE | 게시 종료일 |
| status | VARCHAR(20) | 상태 |
| created_by | VARCHAR(100) | 작성자 |
| keywords | TEXT[] | 키워드 배열 |
| embedding | vector(1536) | 임베딩 벡터 |
| metadata | JSONB | 메타데이터 |
| created_at | TIMESTAMP | 생성일 |
| updated_at | TIMESTAMP | 수정일 |

- **데이터 건수**: 52건
- **인덱스**: category, priority, is_pinned, status, start_date, HNSW 임베딩 인덱스

---

### 2.4 키워드 테이블

#### keyword_dictionary (키워드 사전)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | SERIAL PK | 자동 증가 ID |
| keyword | VARCHAR(200) | 키워드 |
| category | VARCHAR(100) | 카테고리 |
| priority | INTEGER | 우선순위 (1-10) |
| urgency | VARCHAR(20) | 긴급도 (low, medium, high) |
| context_hints | TEXT[] | 문맥 힌트 배열 |
| weight | FLOAT | 가중치 (기본 1.0, 카드상품명 1.5) |
| synonyms | TEXT[] | 동의어 배열 |
| variations | TEXT[] | 변형어 배열 |
| compound_patterns | JSONB | 복합어 패턴 |
| ambiguity_rules | JSONB | 모호성 규칙 |
| created_at | TIMESTAMP | 생성일 |
| updated_at | TIMESTAMP | 수정일 |

- **데이터 건수**: 2,881건 (키워드 사전 2,483건 + 카드상품명 398건)
- **유니크 제약**: (keyword, category) 조합
- **비고**: 카드 상품명은 weight=1.5로 높은 가중치 부여.

#### keyword_synonyms (키워드 동의어)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | SERIAL PK | 자동 증가 ID |
| synonym | VARCHAR(200) | 동의어 |
| canonical_keyword | VARCHAR(200) | 정규 키워드 |
| category | VARCHAR(100) | 카테고리 |
| created_at | TIMESTAMP | 생성일 |

- **데이터 건수**: 450건
- **유니크 제약**: (synonym, canonical_keyword, category) 조합

---

### 2.5 시뮬레이션 테이블

#### simulation_scenarios (시나리오 템플릿)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | VARCHAR(50) PK | 시나리오 ID (예: `SIM-001`) |
| title | VARCHAR(200) | 제목 |
| difficulty | VARCHAR(20) | 난이도 (초급, 중급, 고급) |
| category | VARCHAR(100) | 카테고리 |
| description | TEXT | 설명 |
| duration_estimate | INTEGER | 예상 소요 시간 (초) |
| scenario_objectives | JSONB | 학습 목표 |
| scenario_steps | JSONB | 단계별 가이드 |
| required_documents | TEXT[] | 필수 참조 문서 ID |
| required_keywords | TEXT[] | 필수 키워드 |
| ai_customer_persona | VARCHAR(50) | AI 고객 성향 |
| ai_customer_name | VARCHAR(100) | AI 고객명 |
| ai_customer_background | TEXT | 고객 배경 정보 |
| ai_conversation_flow | JSONB | AI 대화 흐름 |
| evaluation_criteria | JSONB | 평가 기준 |
| passing_score | INTEGER | 합격 점수 (기본 70) |
| locked | BOOLEAN | 잠금 여부 |
| unlock_condition | VARCHAR(500) | 잠금 해제 조건 |
| tags | TEXT[] | 태그 배열 |

- **데이터 건수**: 5건 (SIM-001 ~ SIM-005)

#### simulation_results (시뮬레이션 결과)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | SERIAL PK | 자동 증가 ID |
| employee_id | VARCHAR(50) FK | 상담사 ID → employees(id) |
| simulation_type | VARCHAR(20) | 시뮬레이션 유형 (scenario, best_practice) |
| scenario_id | VARCHAR(50) FK | 시나리오 ID → simulation_scenarios(id) |
| simulation_consultation_id | VARCHAR(100) UNIQUE | 시뮬레이션 상담 ID |
| overall_score | INTEGER | 종합 점수 |
| passed | BOOLEAN | 합격 여부 |
| objective_completion_rate | INTEGER | 목표 달성률 |
| document_usage_score | INTEGER | 문서 활용 점수 |
| keyword_coverage_score | INTEGER | 키워드 커버리지 점수 |
| sequence_correctness_score | INTEGER | 순서 정확도 점수 |
| customer_satisfaction_score | INTEGER | 고객 만족도 점수 |
| feedback_data | JSONB | 피드백 데이터 |
| call_duration | INTEGER | 통화 시간 (초) |
| call_started_at | TIMESTAMP | 시작 시간 |
| call_ended_at | TIMESTAMP | 종료 시간 |
| created_at | TIMESTAMP | 생성일 |

- **데이터 건수**: 148건 (Mock)
- **인덱스**: employee_id, scenario_id, simulation_type, passed

#### employee_learning_analytics (상담사 학습 분석)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | SERIAL PK | 자동 증가 ID |
| employee_id | VARCHAR(50) UNIQUE FK | 상담사 ID → employees(id) |
| total_simulations | INTEGER | 총 시뮬레이션 수 |
| total_best_practice_simulations | INTEGER | 우수사례 시뮬레이션 수 |
| total_scenario_simulations | INTEGER | 시나리오 시뮬레이션 수 |
| average_score | NUMERIC(5,2) | 평균 점수 |
| average_similarity_score | NUMERIC(5,2) | 평균 유사도 점수 |
| average_objective_completion | NUMERIC(5,2) | 평균 목표 달성률 |
| pass_rate | NUMERIC(5,2) | 합격률 |
| best_practice_pass_rate | NUMERIC(5,2) | 우수사례 합격률 |
| scenario_pass_rate | NUMERIC(5,2) | 시나리오 합격률 |
| improvement_rate | NUMERIC(5,2) | 성장률 |
| strengths | JSONB | 강점 |
| weaknesses | JSONB | 약점 |
| category_performance | JSONB | 카테고리별 성과 |
| completed_scenarios | TEXT[] | 완료한 시나리오 ID |
| unlocked_scenarios | TEXT[] | 해제된 시나리오 ID |
| total_learning_time_seconds | INTEGER | 총 학습 시간 (초) |
| last_simulation_at | TIMESTAMP | 마지막 시뮬레이션 일시 |
| last_simulation_score | INTEGER | 마지막 시뮬레이션 점수 |
| created_at | TIMESTAMP | 생성일 |
| updated_at | TIMESTAMP | 수정일 |

- **데이터 건수**: 70건 (상담사 전원, Mock)

---

### 2.6 감사 로그 테이블

#### recording_download_logs (녹취 다운로드 이력)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | SERIAL PK | 자동 증가 ID |
| consultation_id | VARCHAR(100) FK | 상담 ID → consultations(id) |
| downloaded_by | VARCHAR(50) FK | 다운로드한 직원 ID → employees(id) |
| download_type | VARCHAR(10) | 다운로드 유형 (txt, wav, mp3) |
| download_ip | VARCHAR(45) | 다운로드 IP |
| download_user_agent | TEXT | User Agent |
| file_name | VARCHAR(255) | 파일명 |
| file_path | VARCHAR(500) | 파일 경로 |
| file_size | BIGINT | 파일 크기 (bytes) |
| downloaded_at | TIMESTAMP | 다운로드 일시 |

- **데이터 건수**: 100건 (Mock)
- **인덱스**: consultation_id, downloaded_by, downloaded_at, download_type

#### audit_logs (감사 로그)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | SERIAL PK | 자동 증가 ID |
| user_id | VARCHAR(50) | 사용자 ID |
| action | VARCHAR(50) | 액션 (RECORDING_DOWNLOAD, CUSTOMER_VIEW 등) |
| resource_type | VARCHAR(50) | 리소스 유형 |
| resource_id | VARCHAR(100) | 리소스 ID |
| ip_address | VARCHAR(45) | IP 주소 |
| user_agent | TEXT | User Agent |
| details | JSONB | 상세 정보 |
| created_at | TIMESTAMP | 생성일 |

- **데이터 건수**: 50건 (Mock)
- **인덱스**: user_id, action, resource_type, created_at

---

## 3. 함수 및 뷰

### 3.1 함수

#### fn_get_consultation_relevance(p_customer_id, p_days_threshold)

- **목적**: 특정 고객의 최근 상담 이력에 대한 유의미성 판단
- **입력**: 고객 ID, 기준 일수 (기본 30일)
- **반환**: consultation_id, consultation_date, category_sub, days_ago, relevance_level, relevance_score
- **유의미성 기준**:
  - 7일 이내: `high` (점수 65~100)
  - 30일 이내: `medium` (점수 19~65)
  - 90일 이내: `low` (점수 1~19)
  - 90일 초과: `none` (점수 0)
- **제한**: 최근 10건만 반환

#### fn_get_customer_persona_relevance(p_customer_id, p_category)

- **목적**: 고객의 페르소나 정보가 현재 상담에 유효한지 판단
- **입력**: 고객 ID, 상담 카테고리
- **반환**: customer_id, persona_code, persona_name, persona_relevance, recent_consultation_count, guidance_text

#### fn_find_similar_recent_consultations(p_customer_id, p_category, p_days)

- **목적**: 동일 고객의 최근 유사 상담 검색 (동일/유사 카테고리)
- **입력**: 고객 ID, 카테고리, 일수
- **반환**: consultation_id, call_date, category_main, category_sub, days_ago, is_same_category, similarity_reason

### 3.2 뷰

#### v_customer_guidance_info

- **목적**: 상담 가이던스용 고객 종합 정보 (페르소나 + 최근 이력 통합)
- **주요 필드**: 고객 ID, 이름, 등급, 페르소나 정보, 최근 상담 건수, LLM 가이던스

#### v_customer_recent_history

- **목적**: 고객별 최근 상담 이력 요약
- **주요 필드**: 고객 ID, 최근 상담 목록 (30일 이내)

#### v_suspicious_downloads

- **목적**: 이상 다운로드 감지 (동일 상담사가 1시간 내 5건 이상 다운로드)
- **주요 필드**: 다운로드 직원 ID, 다운로드 건수, 시간대

#### v_daily_download_stats

- **목적**: 일별 다운로드 통계
- **주요 필드**: 날짜, 다운로드 건수, 다운로드 유형별 건수

---

## 4. 데이터 품질 설계

### 4.1 상담사 배분 로직

6,533건의 상담 데이터를 60명의 상담 업무 담당자에게 현실적으로 배분합니다.

**하이브리드 배분 방식**:
- **90% 순차 배분**: rank(성과 순위) 기반 목표 건수에 따라 순차적으로 배분
- **10% 랜덤 배분**: 나머지 10%는 해당 대분류 전문 풀 내에서 랜덤 배분

**rank 기반 목표 건수**:
- 1위: 132건 (최대)
- 60위: 75건 (최소)
- 공식: `base_count + int((max_extra - min_extra) * (total_agents - rank) / (total_agents - 1))`

**대분류별 전문 풀**:
- 각 대분류마다 지정된 수의 전문 상담사가 담당 (예: 결제/승인 12명, 이용내역 9명)
- "기타" 대분류는 전체 60명이 순차적으로 담당
- 완료율 기반 배분: 목표 대비 현재 건수가 적은 상담사에게 우선 배분

**IT/관리/교육팀 제외**:
- IT팀(5명), 관리팀(3명), 교육팀(2명)은 상담 배정 대상에서 완전 제외
- 해당 팀의 consultations, fcr, avgTime 등 성과 지표는 0으로 유지

### 4.2 고객-상담 매칭

2,500명의 고객에게 6,533건의 상담을 현실적 분포로 매칭합니다.

**분포 설계**:

| 상담 건수 | 비율 | 고객 수(약) |
|-----------|------|------------|
| 1건 | 42.7% | 1,068명 |
| 2~3건 | 35.0% | 875명 |
| 4~6건 | 18.0% | 450명 |
| 7건 이상 | 4.3% | 107명 |

**재상담 시나리오**:
- 전체 상담의 15%는 동일 카테고리 재상담 (FCR 실패 시나리오)
- 같은 고객이 동일 대분류 카테고리로 7일 이내 재문의하는 패턴

### 4.3 FCR (First Call Resolution)

- **정의**: 7일 이내 동일 대분류 카테고리로 재문의가 없으면 FCR = TRUE
- **전체 FCR**: 약 85.3%
- **계산 방식**: 상담 적재 후 `load_consultations.py`에서 자동 계산
- **고객별 통계**: `update_customer_consultation_stats()`에서 customers 테이블 자동 업데이트

### 4.4 처리 타임라인 (processing_timeline JSONB)

각 상담의 처리 과정을 시간 순서로 기록한 JSONB 필드입니다.

**카테고리별 전형적 처리 단계**:
- 분실/도난: 본인확인 → 카드정지 → 재발급안내 → 완료
- 결제/승인: 본인확인 → 결제내역조회 → 처리결과안내 → 완료
- 한도: 본인확인 → 한도조회 → 한도변경 → 완료
- 기타: 본인확인 → 문의접수 → 처리결과안내 → 완료

각 단계에는 시작 시간, 소요 시간, 상태가 기록됩니다.

### 4.5 요일별 균등 배분

상담 날짜는 5일 영업일(월~금)에 균등 배분됩니다.

**기준 영업일**:
- 2026-01-19 (월) ~ 2026-01-23 (금)
- 각 영업일에 약 1,306건씩 배분 (6,533 / 5 = ~1,306.6)

### 4.6 재현성 보장

모든 데이터 생성은 결정적(deterministic)으로 동작하여, 누가 어디서 실행해도 동일한 결과를 보장합니다.

| 기법 | 적용 위치 | 설명 |
|------|----------|------|
| `random.seed(42)` | 전 모듈 | 랜덤 시드 전역 고정 |
| `hashlib.md5()` | load_consultations.py | `hash()` 대신 결정적 해시 사용 |
| 고정 기준일 | 전 모듈 | `datetime.now()` 미사용, 고정 날짜 사용 |
| `random.seed(43)` | generate_mock.py | 감사 로그는 별도 시드 |

---

## 5. 스크립트 구조

### 5.1 오케스트레이터

**파일**: `01_setup_callact_db.py` (144줄)

```
python 01_setup_callact_db.py [옵션]
```

| 옵션 | 설명 |
|------|------|
| `--skip-schema` | 스키마 생성 건너뛰기 |
| `--skip-employees` | 상담사 데이터 적재 건너뛰기 |
| `--skip-hana` | 하나카드 데이터 적재 건너뛰기 |
| `--skip-keywords` | 키워드 사전 적재 건너뛰기 |
| `--skip-teddycard` | 테디카드 데이터 적재 건너뛰기 |
| `--verify-only` | 검증만 실행 |

### 5.2 모듈 구조

```
modules/
  __init__.py           # 공통 유틸리티, 상수, 카테고리 매핑 (326줄)
  schema_runner.py      # 스키마 생성 + 카테고리 매핑 적재 (193줄)
  load_employees.py     # 상담사 데이터 적재 (142줄)
  load_customers.py     # 고객 데이터 적재 (144줄)
  load_consultations.py # 상담 데이터 적재 + 배분 + FCR 계산 (가장 큰 모듈)
  update_stats.py       # 상담사 성과 지표 + 고객 통계 업데이트 (215줄)
  load_keywords.py      # 키워드 사전 + 동의어 + 카드상품명 적재 (251줄)
  load_teddycard.py     # 테디카드 3종 데이터 적재 (264줄)
  generate_mock.py      # 시뮬레이션 + 감사 로그 Mock 데이터 (303줄)
  verify.py             # 체크리스트 + 검증 (443줄)
```

### 5.3 SQL 스키마 파일

| 파일 | 설명 |
|------|------|
| `db_setup.sql` | 기본 스키마 (employees, consultations, consultation_documents, category_mappings, Enum, 인덱스) |
| `02_setup_tedicard_tables.sql` | 테디카드 테이블 (service_guide_documents, card_products, notices) |
| `03_setup_keyword_dictionary.sql` | 키워드 사전 테이블 (keyword_dictionary, keyword_synonyms) |
| `07_setup_customers_table.sql` | 고객 테이블 (customers) |
| `07a_setup_persona_types_table.sql` | 페르소나 유형 테이블 (persona_types) + 12개 초기 데이터 |
| `10_setup_simulation_tables.sql` | 시뮬레이션 테이블 + 5개 시나리오 초기 데이터 |
| `11_setup_audit_tables.sql` | 감사 로그 테이블 + 뷰 |
| `12_setup_consultation_relevance.sql` | 함수 3개 + 뷰 2개 |
| `99_verify_db_schema.sql` | DBeaver 수동 검증용 (실행 대상 아님) |

### 5.4 config.py (경로 설정)

- **ENV_TYPE**: `'dev'` 또는 `'prod'` (환경변수 `DB_SCRIPTS_ENV`로 제어)
- **dev 모드**: `data-preprocessing_dev/` 경로 우선 탐색, `data-preprocessing/` 폴백
- **prod 모드**: `data-preprocessing/` 경로만 사용

### 5.5 실행 순서 (12단계)

```
[1/12] 기본 DB 스키마 생성
[2/12] 테디카드 테이블 생성
[3/12] 키워드 사전 테이블 생성
[3-1/12] 페르소나 유형 테이블 생성
[3-2/12] 고객 테이블 생성
[3-3/12] 시뮬레이션 교육 테이블 생성
[3-4/12] 감사 로그 테이블 생성
[3-5/12] 상담 이력 유의미성 함수/뷰 생성
[3-6/12] 카테고리 매핑 데이터 적재
[4/12] 상담사 데이터 적재 (70명)
[4-1/12] 고객 데이터 적재 (2,500명)
[5/12] 하나카드 상담 데이터 적재 (6,533건)
[6/12] 상담사 성과 지표 업데이트
[6-1/12] 고객별 상담 통계 업데이트
[7/12] 키워드 사전 데이터 적재 (2,881건)
[8/12] 테디카드 데이터 적재 (1,723건)
[9/12] 시뮬레이션 교육 Mock 데이터 (148건)
[10/12] 감사 로그 Mock 데이터 (150건)
[12/12] 데이터 적재 검증
```

---

## 6. ERD (텍스트 기반)

```
employees (70)
  │
  ├──< consultations (6,533) >── customers (2,500)
  │        │                         │
  │        ├──< consultation_documents (6,533)   persona_types (12)
  │        │
  │        ├──< recording_download_logs (100)
  │        │
  │        └── category_mappings (55)
  │
  ├──< simulation_results (148)
  │        │
  │        └── simulation_scenarios (5)
  │
  ├──< employee_learning_analytics (70)
  │
  └──< audit_logs (50)

service_guide_documents (1,273)   [독립]
card_products (398)               [독립]
notices (52)                      [독립]
keyword_dictionary (2,881)        [독립]
keyword_synonyms (450)            [독립]
```

**관계 요약**:
- `employees` 1:N `consultations` (agent_id)
- `customers` 1:N `consultations` (customer_id)
- `consultations` 1:1 `consultation_documents` (consultation_id)
- `consultations` 1:N `recording_download_logs` (consultation_id)
- `employees` 1:N `simulation_results` (employee_id)
- `simulation_scenarios` 1:N `simulation_results` (scenario_id)
- `employees` 1:1 `employee_learning_analytics` (employee_id)
- `employees` 1:N `audit_logs` (user_id)
- 테디카드 3종, 키워드 2종은 독립 테이블 (FK 없음)

---

## 결론

CALL:ACT DB는 16개 테이블, 3개 함수, 4개 뷰로 구성되며, PostgreSQL 17 + pgvector를 기반으로 합니다. 모든 데이터는 `01_setup_callact_db.py` 오케스트레이터를 통해 12단계로 자동 적재되며, `random.seed(42)` + `hashlib.md5()` + 고정 기준일을 사용하여 재현성을 보장합니다.

실행 방법은 [통합 DB 설정 가이드 v3.0](./통합_DB_설정_가이드_v3.0.md)을 참조하세요.
