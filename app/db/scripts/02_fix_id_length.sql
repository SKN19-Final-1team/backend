-- ==================================================
-- CALL:ACT 테디카드 데이터 적재용 ID 컬럼 길이 수정 스크립트
-- ==================================================
-- Description: service_guide_documents.id 컬럼 길이를 50에서 100으로 확장
-- Author: CALL:ACT Team
-- Date: 2026-01-13
-- ==================================================
-- 
-- 문제: 일부 service_guide_documents의 id가 50자를 초과함
-- 해결: id 컬럼을 VARCHAR(50)에서 VARCHAR(100)으로 변경
-- ==================================================

-- service_guide_documents 테이블의 id 컬럼 길이 확장
DO $$ 
BEGIN
    -- 기존 제약조건 확인 및 삭제
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
END $$;

-- 완료 메시지
DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'service_guide_documents.id 컬럼 길이 수정 완료';
    RAISE NOTICE '- VARCHAR(50) -> VARCHAR(100)';
    RAISE NOTICE '============================================================';
END $$;
