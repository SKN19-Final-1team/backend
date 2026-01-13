-- ==================================================
-- CALL:ACT 테디카드 데이터 적재용 테이블 수정 스크립트
-- ==================================================
-- Description: 기존 테이블에 RAG 검색용 컬럼 추가
-- Author: CALL:ACT Team
-- Date: 2026-01-13
-- ==================================================
-- 
-- 주의: 이 스크립트는 이미 생성된 테이블에 컬럼을 추가합니다.
-- 테이블이 이미 존재하는 경우에만 실행하세요.
-- ==================================================

-- 1. brand_type ENUM에 'local' 추가 (이미 있으면 무시)
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

-- 2. service_guide_documents 테이블에 structured 컬럼 추가
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

-- 3. card_products 테이블에 metadata, structured 컬럼 추가
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

-- 4. notices 테이블에 keywords, embedding, metadata 컬럼 추가
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

-- 5. notices 테이블에 embedding 인덱스 추가
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

-- 6. 완료 메시지
DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE '테디카드 데이터 테이블 수정이 완료되었습니다.';
    RAISE NOTICE '- service_guide_documents.structured 컬럼 추가됨';
    RAISE NOTICE '- card_products.metadata, structured 컬럼 추가됨';
    RAISE NOTICE '- notices.keywords, embedding, metadata 컬럼 추가됨';
    RAISE NOTICE '- notices.embedding 인덱스 생성됨';
    RAISE NOTICE '============================================================';
END $$;
