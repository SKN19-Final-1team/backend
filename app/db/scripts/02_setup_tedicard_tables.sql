-- ==================================================
-- CALL:ACT 테디카드 데이터 적재용 테이블 생성 스크립트
-- ==================================================
-- Description: service_guide_documents, card_products, notices 테이블 생성
-- Author: CALL:ACT Team
-- Date: 2026-01-13
-- ==================================================

-- 1. pgvector 확장 설치 (이미 설치되어 있어도 안전)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Enum 타입 생성

-- 카드 타입
DO $$ BEGIN
    CREATE TYPE card_type AS ENUM ('credit', 'debit');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 브랜드 타입
DO $$ BEGIN
    CREATE TYPE brand_type AS ENUM ('visa', 'mastercard', 'amex', 'unionpay', 'local');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 공지사항 카테고리
DO $$ BEGIN
    CREATE TYPE notice_category AS ENUM ('system', 'service', 'emergency');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 공지사항 우선순위
DO $$ BEGIN
    CREATE TYPE notice_priority AS ENUM ('normal', 'important', 'urgent');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 상태 타입 (이미 있으면 무시)
DO $$ BEGIN
    CREATE TYPE status_type AS ENUM ('active', 'inactive', 'suspended');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 3. service_guide_documents 테이블 생성
CREATE TABLE IF NOT EXISTS service_guide_documents (
    id VARCHAR(100) PRIMARY KEY,
    document_type VARCHAR(50),
    category VARCHAR(100),
    title VARCHAR(300) NOT NULL,
    content TEXT NOT NULL,
    keywords TEXT[],  -- PostgreSQL 배열 타입
    embedding vector(1536),  -- pgvector 확장 타입
    metadata JSONB,  -- 추가 메타데이터 (original_source, document_number 등)
    document_source VARCHAR(200),
    priority VARCHAR(20),
    usage_count INT DEFAULT 0,
    last_used TIMESTAMP,
    structured JSONB,  -- RAG 검색용 구조화 데이터
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_service_guide_documents_document_type ON service_guide_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_service_guide_documents_category ON service_guide_documents(category);
CREATE INDEX IF NOT EXISTS idx_service_guide_documents_usage_count ON service_guide_documents(usage_count);
CREATE INDEX IF NOT EXISTS idx_service_guide_documents_embedding_hnsw ON service_guide_documents USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

COMMENT ON TABLE service_guide_documents IS '카드사 이용 안내 문서 + RAG 검색용 VectorDB 메타데이터';

-- 4. card_products 테이블 생성
CREATE TABLE IF NOT EXISTS card_products (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    card_type card_type NOT NULL,
    brand brand_type,
    annual_fee_domestic INT,
    annual_fee_global INT,
    performance_condition TEXT,
    main_benefits TEXT,
    status status_type DEFAULT 'active',
    metadata JSONB,  -- 추가 메타데이터
    structured JSONB,  -- RAG 검색용 구조화 데이터
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_card_products_card_type ON card_products(card_type);
CREATE INDEX IF NOT EXISTS idx_card_products_brand ON card_products(brand);
CREATE INDEX IF NOT EXISTS idx_card_products_status ON card_products(status);

COMMENT ON TABLE card_products IS '카드 상품 마스터 테이블';

-- 5. notices 테이블 생성 (RAG 검색을 위해 keywords, embedding 추가)
CREATE TABLE IF NOT EXISTS notices (
    id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(300) NOT NULL,
    content TEXT NOT NULL,
    category notice_category,
    priority notice_priority,
    is_pinned BOOLEAN DEFAULT false,
    start_date DATE NOT NULL,
    end_date DATE,
    status status_type DEFAULT 'active',
    created_by VARCHAR(50),
    keywords TEXT[],  -- PostgreSQL 배열 타입 (RAG 검색용)
    embedding vector(1536),  -- pgvector 확장 타입 (RAG 검색용)
    metadata JSONB,  -- 추가 메타데이터
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_notices_category ON notices(category);
CREATE INDEX IF NOT EXISTS idx_notices_priority ON notices(priority);
CREATE INDEX IF NOT EXISTS idx_notices_is_pinned ON notices(is_pinned);
CREATE INDEX IF NOT EXISTS idx_notices_dates ON notices(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_notices_embedding_hnsw ON notices USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

COMMENT ON TABLE notices IS '시스템/서비스 공지사항 관리';

-- 6. 테이블 수정 (컬럼 추가 및 ID 길이 수정)
-- 6-1. brand_type ENUM에 'local' 추가 (이미 있으면 무시)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum 
        WHERE enumlabel = 'local' 
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'brand_type')
    ) THEN
        ALTER TYPE brand_type ADD VALUE 'local';
    END IF;
END $$;

-- 6-2. service_guide_documents 테이블에 structured 컬럼 추가 (이미 있으면 무시)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'service_guide_documents' 
        AND column_name = 'structured'
    ) THEN
        ALTER TABLE service_guide_documents 
        ADD COLUMN structured JSONB;
        
        RAISE NOTICE 'service_guide_documents.structured 컬럼 추가됨';
    ELSE
        RAISE NOTICE 'service_guide_documents.structured 컬럼이 이미 존재합니다.';
    END IF;
END $$;

-- 6-3. card_products 테이블에 metadata, structured 컬럼 추가 (이미 있으면 무시)
DO $$ 
BEGIN
    -- metadata 컬럼 추가
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'card_products' 
        AND column_name = 'metadata'
    ) THEN
        ALTER TABLE card_products 
        ADD COLUMN metadata JSONB;
        
        RAISE NOTICE 'card_products.metadata 컬럼 추가됨';
    ELSE
        RAISE NOTICE 'card_products.metadata 컬럼이 이미 존재합니다.';
    END IF;
    
    -- structured 컬럼 추가
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'card_products' 
        AND column_name = 'structured'
    ) THEN
        ALTER TABLE card_products 
        ADD COLUMN structured JSONB;
        
        RAISE NOTICE 'card_products.structured 컬럼 추가됨';
    ELSE
        RAISE NOTICE 'card_products.structured 컬럼이 이미 존재합니다.';
    END IF;
END $$;

-- 6-4. notices 테이블에 keywords, embedding, metadata 컬럼 추가 (이미 있으면 무시)
DO $$ 
BEGIN
    -- keywords 컬럼 추가
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'notices' 
        AND column_name = 'keywords'
    ) THEN
        ALTER TABLE notices 
        ADD COLUMN keywords TEXT[];
        
        RAISE NOTICE 'notices.keywords 컬럼 추가됨';
    ELSE
        RAISE NOTICE 'notices.keywords 컬럼이 이미 존재합니다.';
    END IF;
    
    -- embedding 컬럼 추가
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'notices' 
        AND column_name = 'embedding'
    ) THEN
        ALTER TABLE notices 
        ADD COLUMN embedding vector(1536);
        
        RAISE NOTICE 'notices.embedding 컬럼 추가됨';
    ELSE
        RAISE NOTICE 'notices.embedding 컬럼이 이미 존재합니다.';
    END IF;
    
    -- metadata 컬럼 추가
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'notices' 
        AND column_name = 'metadata'
    ) THEN
        ALTER TABLE notices 
        ADD COLUMN metadata JSONB;
        
        RAISE NOTICE 'notices.metadata 컬럼 추가됨';
    ELSE
        RAISE NOTICE 'notices.metadata 컬럼이 이미 존재합니다.';
    END IF;
END $$;

-- 6-5. notices 테이블에 embedding 인덱스 추가 (이미 있으면 무시)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE indexname = 'idx_notices_embedding_hnsw'
    ) THEN
        CREATE INDEX idx_notices_embedding_hnsw 
        ON notices USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64);
        
        RAISE NOTICE 'notices.embedding 인덱스 생성됨';
    ELSE
        RAISE NOTICE 'notices.embedding 인덱스가 이미 존재합니다.';
    END IF;
END $$;

-- 6-6. service_guide_documents 테이블의 id 컬럼 길이 확장 (VARCHAR(100) 확인 및 수정)
DO $$ 
BEGIN
    -- id 컬럼 타입 확인 및 변경 (VARCHAR(50) -> VARCHAR(100))
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'service_guide_documents' 
        AND column_name = 'id'
        AND character_maximum_length = 50
    ) THEN
        -- 기존 PRIMARY KEY 제약조건 삭제
        IF EXISTS (
            SELECT 1 FROM pg_constraint 
            WHERE conname = 'service_guide_documents_pkey'
        ) THEN
            ALTER TABLE service_guide_documents DROP CONSTRAINT service_guide_documents_pkey;
            RAISE NOTICE '기존 PRIMARY KEY 제약조건 삭제됨';
        END IF;
        
        -- id 컬럼 타입 변경
        ALTER TABLE service_guide_documents 
        ALTER COLUMN id TYPE VARCHAR(100);
        
        RAISE NOTICE 'service_guide_documents.id 컬럼을 VARCHAR(100)으로 변경됨';
        
        -- PRIMARY KEY 제약조건 재생성
        ALTER TABLE service_guide_documents 
        ADD CONSTRAINT service_guide_documents_pkey PRIMARY KEY (id);
        
        RAISE NOTICE 'PRIMARY KEY 제약조건 재생성됨';
    ELSE
        RAISE NOTICE 'service_guide_documents.id 컬럼이 이미 VARCHAR(100)입니다.';
    END IF;
END $$;

-- 7. 성공 메시지 출력
DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE '테디카드 데이터 테이블 생성 및 수정이 완료되었습니다.';
    RAISE NOTICE '- service_guide_documents 테이블 생성됨';
    RAISE NOTICE '- card_products 테이블 생성됨';
    RAISE NOTICE '- notices 테이블 생성됨';
    RAISE NOTICE '- 컬럼 추가 및 수정 완료';
    RAISE NOTICE '- 벡터 인덱스 생성됨';
    RAISE NOTICE '============================================================';
END $$;
