-- ==================================================
-- CALL:ACT 키워드 사전 DB 스키마 생성 스크립트
-- ==================================================
-- Description: 키워드 사전 테이블 생성 (keyword_dictionary, keyword_synonyms)
-- Author: CALL:ACT Team
-- Date: 2026-01-11
-- ==================================================
-- 주의: 이 스크립트는 db_setup.sql 실행 후 실행해야 합니다.
-- ==================================================

-- 1. pg_trgm 확장 설치 (키워드 텍스트 검색용)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 2. keyword_dictionary 테이블 생성 (키워드 사전 메인 테이블)

CREATE TABLE IF NOT EXISTS keyword_dictionary (
    id SERIAL PRIMARY KEY,
    keyword VARCHAR(100) NOT NULL,  -- 표준 키워드 (canonical)
    category VARCHAR(100) NOT NULL,
    priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    urgency VARCHAR(20) DEFAULT 'medium' CHECK (urgency IN ('high', 'medium', 'low')),
    context_hints TEXT[],  -- 맥락 힌트 배열
    weight DECIMAL(3,2) DEFAULT 1.0,  -- 카테고리 내 가중치
    synonyms TEXT[],  -- 동의어 배열
    variations TEXT[],  -- 변형 표현 배열
    compound_patterns JSONB,  -- 복합 키워드 패턴
    ambiguity_rules JSONB,  -- 중의성 해소 규칙
    usage_count INTEGER DEFAULT 0,  -- 사용 빈도
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(keyword, category)
);

COMMENT ON TABLE keyword_dictionary IS '키워드 사전 메인 테이블';
COMMENT ON COLUMN keyword_dictionary.keyword IS '표준 키워드 (canonical form)';
COMMENT ON COLUMN keyword_dictionary.category IS '키워드 카테고리';
COMMENT ON COLUMN keyword_dictionary.priority IS '우선순위 (1-10, 높을수록 중요)';
COMMENT ON COLUMN keyword_dictionary.urgency IS '긴급성 (high, medium, low)';
COMMENT ON COLUMN keyword_dictionary.context_hints IS '맥락 힌트 배열 (컨텍스트 기반 해석에 사용)';
COMMENT ON COLUMN keyword_dictionary.weight IS '카테고리 내 가중치 (0.0-1.0)';
COMMENT ON COLUMN keyword_dictionary.synonyms IS '동의어 배열';
COMMENT ON COLUMN keyword_dictionary.variations IS '변형 표현 배열';
COMMENT ON COLUMN keyword_dictionary.compound_patterns IS '복합 키워드 패턴 (JSONB)';
COMMENT ON COLUMN keyword_dictionary.ambiguity_rules IS '중의성 해소 규칙 (JSONB)';
COMMENT ON COLUMN keyword_dictionary.usage_count IS '사용 빈도 (RAG 검색 시 증가)';

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_keyword_priority ON keyword_dictionary(priority DESC, urgency);
CREATE INDEX IF NOT EXISTS idx_keyword_text ON keyword_dictionary USING GIN(keyword gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_keyword_synonyms ON keyword_dictionary USING GIN(synonyms);
CREATE INDEX IF NOT EXISTS idx_keyword_category ON keyword_dictionary(category);
CREATE INDEX IF NOT EXISTS idx_keyword_updated_at ON keyword_dictionary(updated_at DESC);

COMMENT ON INDEX idx_keyword_priority IS '우선순위 및 긴급성 기반 검색용 인덱스';
COMMENT ON INDEX idx_keyword_text IS '키워드 텍스트 유사도 검색용 인덱스 (pg_trgm)';
COMMENT ON INDEX idx_keyword_synonyms IS '동의어 배열 검색용 인덱스';
COMMENT ON INDEX idx_keyword_category IS '카테고리별 검색용 인덱스';

-- 3. keyword_synonyms 테이블 생성 (동의어 매핑 테이블 - 빠른 검색용)

CREATE TABLE IF NOT EXISTS keyword_synonyms (
    id SERIAL PRIMARY KEY,
    synonym VARCHAR(100) NOT NULL,  -- 동의어
    canonical_keyword VARCHAR(100) NOT NULL,  -- 표준 키워드
    category VARCHAR(100) NOT NULL,
    FOREIGN KEY (canonical_keyword, category) 
        REFERENCES keyword_dictionary(keyword, category)
        ON DELETE CASCADE,
    UNIQUE(synonym, canonical_keyword, category),
    created_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE keyword_synonyms IS '동의어 매핑 테이블 (빠른 검색용)';
COMMENT ON COLUMN keyword_synonyms.synonym IS '동의어 (입력 키워드)';
COMMENT ON COLUMN keyword_synonyms.canonical_keyword IS '표준 키워드 (keyword_dictionary.keyword)';
COMMENT ON COLUMN keyword_synonyms.category IS '카테고리 (keyword_dictionary.category)';

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_synonym_lookup ON keyword_synonyms(synonym);
CREATE INDEX IF NOT EXISTS idx_synonym_canonical ON keyword_synonyms(canonical_keyword, category);

COMMENT ON INDEX idx_synonym_lookup IS '동의어 검색용 인덱스';
COMMENT ON INDEX idx_synonym_canonical IS '표준 키워드 기반 조회용 인덱스';

-- 4. 성공 메시지 출력
DO $$
BEGIN
    RAISE NOTICE '키워드 사전 DB 스키마 생성이 완료되었습니다.';
    RAISE NOTICE '- keyword_dictionary 테이블 생성됨';
    RAISE NOTICE '- keyword_synonyms 테이블 생성됨';
    RAISE NOTICE '- 인덱스 생성됨 (priority, text, synonyms, category)';
END $$;
