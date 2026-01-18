-- ==================================================
-- CALL:ACT 테디카드 데이터 적재용 DB 설정 스크립트
-- ==================================================
-- Description: PostgreSQL + pgvector 확장 설치 및 필요한 테이블 생성
-- Author: CALL:ACT Team
-- Date: 2026-01-08
-- ==================================================

-- 1. pgvector 확장 설치
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Enum 타입 생성 (하나카드 데이터에 필요한 것만)
-- 참고: 전체 ERD에는 더 많은 Enum이 있지만, 하나카드 데이터 적재에는 최소한만 필요

-- 상담 상태
DO $$ BEGIN
    CREATE TYPE consultation_status AS ENUM ('completed', 'in_progress', 'incomplete');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 상담 카테고리 (하나카드 데이터의 실제 카테고리를 VARCHAR로 저장하기 위해 Enum 대신 사용)
-- 참고: 하나카드는 57개의 카테고리가 있으므로 Enum 대신 VARCHAR 사용
-- 추후 카테고리 정규화가 필요할 수 있음

-- 감정 타입
DO $$ BEGIN
    CREATE TYPE emotion_type AS ENUM ('positive', 'neutral', 'negative');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 품질 평가
DO $$ BEGIN
    CREATE TYPE quality_rating AS ENUM ('high', 'medium', 'low');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 화자 타입
DO $$ BEGIN
    CREATE TYPE speaker_type AS ENUM ('customer', 'agent');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 난이도 레벨
DO $$ BEGIN
    CREATE TYPE difficulty_level AS ENUM ('beginner', 'intermediate', 'advanced');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 시나리오 타입
DO $$ BEGIN
    CREATE TYPE scenario_type AS ENUM ('real_case', 'llm_generated');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 상태 타입
DO $$ BEGIN
    CREATE TYPE status_type AS ENUM ('active', 'inactive', 'suspended', 'vacation');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 3. employees 테이블 생성
CREATE TABLE IF NOT EXISTS employees (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    role VARCHAR(50),
    department VARCHAR(100),
    hire_date DATE,
    status status_type DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_employees_role ON employees(role);
CREATE INDEX IF NOT EXISTS idx_employees_department ON employees(department);
CREATE INDEX IF NOT EXISTS idx_employees_status ON employees(status);

COMMENT ON TABLE employees IS '직원(상담사) 정보 테이블';

-- employees 테이블에 성과 지표 컬럼 추가 (기존 테이블 업그레이드용)
DO $$ 
BEGIN
    -- consultations: 상담 건수
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'employees' AND column_name = 'consultations'
    ) THEN
        ALTER TABLE employees ADD COLUMN consultations INTEGER DEFAULT 0;
    END IF;
    
    -- fcr: First Call Resolution 비율 (0-100)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'employees' AND column_name = 'fcr'
    ) THEN
        ALTER TABLE employees ADD COLUMN fcr INTEGER DEFAULT 0;
    END IF;
    
    -- avgTime: 평균 상담 시간 (형식: "MM:SS")
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'employees' AND column_name = 'avgTime'
    ) THEN
        ALTER TABLE employees ADD COLUMN "avgTime" VARCHAR(10) DEFAULT '0:00';
    END IF;
    
    -- rank: 성과 순위 (1부터 시작)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'employees' AND column_name = 'rank'
    ) THEN
        ALTER TABLE employees ADD COLUMN rank INTEGER DEFAULT 0;
    END IF;
END $$;

-- 4. consultations 테이블 생성
CREATE TABLE IF NOT EXISTS consultations (
    id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    agent_id VARCHAR(50) NOT NULL REFERENCES employees(id),
    status consultation_status DEFAULT 'in_progress',
    category VARCHAR(50) NOT NULL,  -- 하나카드는 57개 카테고리이므로 VARCHAR 사용
    title TEXT,
    call_date DATE,
    call_time TIME,
    call_duration VARCHAR(20),  -- 형식: "MM:SS" 또는 "HH:MM:SS"
    fcr BOOLEAN,  -- First Call Resolution
    is_best_practice BOOLEAN DEFAULT false,
    quality_score INT,  -- 0-100
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_consultations_customer_id ON consultations(customer_id);
CREATE INDEX IF NOT EXISTS idx_consultations_agent_id ON consultations(agent_id);
CREATE INDEX IF NOT EXISTS idx_consultations_status ON consultations(status);
CREATE INDEX IF NOT EXISTS idx_consultations_category ON consultations(category);
CREATE INDEX IF NOT EXISTS idx_consultations_call_date ON consultations(call_date);
CREATE INDEX IF NOT EXISTS idx_consultations_fcr ON consultations(fcr);
CREATE INDEX IF NOT EXISTS idx_consultations_is_best_practice ON consultations(is_best_practice);

COMMENT ON TABLE consultations IS '상담 마스터 테이블';

-- 5. consultation_documents 테이블 생성 (VectorDB 포함)
CREATE TABLE IF NOT EXISTS consultation_documents (
    id VARCHAR(50) PRIMARY KEY,
    consultation_id VARCHAR(50) REFERENCES consultations(id),
    document_type VARCHAR(50) DEFAULT 'consultation_transcript',
    category VARCHAR(50) NOT NULL,  -- 하나카드는 57개 카테고리이므로 VARCHAR 사용
    title VARCHAR(300) NOT NULL,
    content TEXT NOT NULL,
    keywords TEXT[],  -- PostgreSQL 배열 타입
    embedding vector(1536),  -- pgvector 확장 타입
    metadata JSONB,  -- 추가 메타데이터 (slot_types, scenario_tags 등)
    usage_count INT DEFAULT 0,
    effectiveness_score DECIMAL(3,2),  -- 0.00-1.00
    last_used TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_consultation_documents_consultation_id ON consultation_documents(consultation_id);
CREATE INDEX IF NOT EXISTS idx_consultation_documents_document_type ON consultation_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_consultation_documents_category ON consultation_documents(category);
CREATE INDEX IF NOT EXISTS idx_consultation_documents_usage_count ON consultation_documents(usage_count);

COMMENT ON TABLE consultation_documents IS '상담 사례 문서 + RAG 검색용 VectorDB 메타데이터';

-- 6. 벡터 인덱스 생성 (HNSW - 대규모 데이터용)
-- 주의: 데이터 삽입 후 인덱스를 생성하는 것이 성능상 유리함
-- 하지만 여기서 미리 생성해도 됨 (빈 테이블에서 생성하면 빠름)
CREATE INDEX IF NOT EXISTS idx_consultation_documents_embedding_hnsw
ON consultation_documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

COMMENT ON INDEX idx_consultation_documents_embedding_hnsw IS 'consultation_documents 임베딩 벡터 인덱스 (HNSW)';

-- 7. 성공 메시지 출력
DO $$
BEGIN
    RAISE NOTICE 'DB 설정이 완료되었습니다.';
    RAISE NOTICE '- pgvector 확장 설치됨';
    RAISE NOTICE '- employees 테이블 생성됨';
    RAISE NOTICE '- consultations 테이블 생성됨';
    RAISE NOTICE '- consultation_documents 테이블 생성됨';
    RAISE NOTICE '- 벡터 인덱스 생성됨';
END $$;


