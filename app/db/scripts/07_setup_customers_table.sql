-- ==================================================
-- Customers 테이블 생성 스크립트
-- ==================================================
-- Description: 고객 정보 테이블 + LLM 상담 가이던스용 페르소나 정보
-- Author: CALL:ACT Team
-- Date: 2026-01-21
-- ==================================================

-- 1. 고객 등급 Enum 타입 생성
DO $$ BEGIN
    CREATE TYPE customer_grade AS ENUM ('VIP', 'GOLD', 'SILVER', 'GENERAL');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 2. customers 테이블 생성
CREATE TABLE IF NOT EXISTS customers (
    -- 기본 식별자
    id VARCHAR(50) PRIMARY KEY,               -- 'CUST-TEDDY-00001'

    -- 기본 정보 (실제 데이터 저장, 마스킹은 Frontend에서 처리)
    name VARCHAR(100) NOT NULL,               -- '김민수' (실제 이름)
    phone VARCHAR(20) NOT NULL,               -- '010-1234-5678' (실제 번호)
    gender VARCHAR(10) DEFAULT 'unknown',     -- 'male', 'female', 'unknown'
    age_group VARCHAR(10),                    -- '20대', '30대', ...
    birth_date DATE,                          -- 생년월일 (1985-03-15)
    address VARCHAR(300),                     -- 주소 (서울시 강남구 테헤란로 123)
    grade VARCHAR(20) DEFAULT 'GENERAL',      -- 'VIP', 'GOLD', 'SILVER', 'GENERAL'

    -- 카드 정보
    card_type VARCHAR(100),                   -- '테디카드 프리미엄'
    card_number_last4 VARCHAR(4),             -- '5678' (마지막 4자리만 저장)
    card_brand VARCHAR(20),                   -- 'visa', 'mastercard', 'local'
    card_issue_date DATE,                     -- 카드 발급일 (2023-01-15)
    card_expiry_date DATE,                    -- 카드 만료일 (2028-01-15) - 만료 임박 고객 안내용

    -- 고객 유형 (페르소나)
    current_type_code VARCHAR(10),            -- 'N1' - 현재 주요 유형 (persona_types.code 참조)
    type_history JSONB DEFAULT '[]'::jsonb,   -- 최근 3개 유형 이력 [{"type":"N1","date":"2024-12-10","consultation_id":"CS-001"},...]

    -- 고객 특성 태그 (LLM 가이던스용) - persona_types에서 조회하거나 커스텀 가능
    personality_tags TEXT[],                  -- ['impatient', 'detailed']
    communication_style JSONB,                -- {"speed": "fast", "tone": "formal"}
    customer_type_codes TEXT[],               -- ['N1', 'S2'] - 복합 유형 (여러 특성 보유 시)

    -- LLM 가이던스 메시지
    llm_guidance TEXT,                        -- LLM에게 전달할 상담 가이드 메시지

    -- 상담 통계
    total_consultations INT DEFAULT 0,
    resolved_first_call INT DEFAULT 0,
    last_consultation_date DATE,

    -- 타임스탬프
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP
);

-- 3. 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_customers_name ON customers(name);
CREATE INDEX IF NOT EXISTS idx_customers_phone ON customers(phone);
CREATE INDEX IF NOT EXISTS idx_customers_gender ON customers(gender);
CREATE INDEX IF NOT EXISTS idx_customers_age_group ON customers(age_group);
CREATE INDEX IF NOT EXISTS idx_customers_birth_date ON customers(birth_date);
CREATE INDEX IF NOT EXISTS idx_customers_grade ON customers(grade);
CREATE INDEX IF NOT EXISTS idx_customers_card_type ON customers(card_type);
CREATE INDEX IF NOT EXISTS idx_customers_total_consultations ON customers(total_consultations);

-- GIN 인덱스 for JSONB and TEXT[]
CREATE INDEX IF NOT EXISTS idx_customers_personality_tags ON customers USING GIN(personality_tags);
CREATE INDEX IF NOT EXISTS idx_customers_communication_style ON customers USING GIN(communication_style);

-- 4. 테이블 및 컬럼 코멘트
COMMENT ON TABLE customers IS '고객 정보 테이블 - LLM 상담 가이던스용 페르소나 정보 포함';
COMMENT ON COLUMN customers.id IS '고객 고유 ID (형식: CUST-TEDDY-00001)';
COMMENT ON COLUMN customers.name IS '고객 실명 (마스킹은 Frontend에서 처리)';
COMMENT ON COLUMN customers.phone IS '고객 전화번호 (형식: 010-1234-5678)';
COMMENT ON COLUMN customers.gender IS '성별 (male, female, unknown)';
COMMENT ON COLUMN customers.age_group IS '연령대 (10대, 20대, 30대, ...)';
COMMENT ON COLUMN customers.birth_date IS '생년월일 (YYYY-MM-DD 형식)';
COMMENT ON COLUMN customers.address IS '주소 (전체 주소 문자열)';
COMMENT ON COLUMN customers.grade IS '고객 등급 (VIP, GOLD, SILVER, GENERAL)';
COMMENT ON COLUMN customers.card_type IS '보유 카드 종류';
COMMENT ON COLUMN customers.card_number_last4 IS '카드번호 마지막 4자리';
COMMENT ON COLUMN customers.card_brand IS '카드 브랜드 (visa, mastercard, local)';
COMMENT ON COLUMN customers.card_issue_date IS '카드 발급일 (YYYY-MM-DD)';
COMMENT ON COLUMN customers.card_expiry_date IS '카드 만료일 (YYYY-MM-DD) - 만료 임박 고객 안내용';
COMMENT ON COLUMN customers.current_type_code IS '현재 주요 고객 유형 코드 (persona_types.code 참조)';
COMMENT ON COLUMN customers.type_history IS '최근 유형 이력 (JSONB 배열, 최대 3개)';
COMMENT ON COLUMN customers.personality_tags IS 'LLM 가이던스용 고객 특성 태그';
COMMENT ON COLUMN customers.customer_type_codes IS '고객 유형 코드 배열 (N1~N3: 일반, S1~S7: 특수)';
COMMENT ON COLUMN customers.communication_style IS 'LLM 가이던스용 의사소통 스타일 (JSON)';
COMMENT ON COLUMN customers.llm_guidance IS 'LLM에게 전달할 상담 가이드 메시지';
COMMENT ON COLUMN customers.total_consultations IS '총 상담 횟수';
COMMENT ON COLUMN customers.resolved_first_call IS 'FCR 성공 횟수';
COMMENT ON COLUMN customers.last_consultation_date IS '최근 상담 일자';

-- 5. 기존 테이블 업그레이드 (필드 추가)
-- 기존 customers 테이블에 birth_date, address 필드가 없으면 추가
DO $$
BEGIN
    -- birth_date 필드 추가
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'customers' AND column_name = 'birth_date'
    ) THEN
        ALTER TABLE customers ADD COLUMN birth_date DATE;
        CREATE INDEX IF NOT EXISTS idx_customers_birth_date ON customers(birth_date);
        RAISE NOTICE 'birth_date 필드 추가됨';
    ELSE
        RAISE NOTICE 'birth_date 필드 이미 존재함';
    END IF;
    
    -- address 필드 추가
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'customers' AND column_name = 'address'
    ) THEN
        ALTER TABLE customers ADD COLUMN address VARCHAR(300);
        RAISE NOTICE 'address 필드 추가됨';
    ELSE
        RAISE NOTICE 'address 필드 이미 존재함';
    END IF;

    -- card_issue_date 필드 추가
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'customers' AND column_name = 'card_issue_date'
    ) THEN
        ALTER TABLE customers ADD COLUMN card_issue_date DATE;
        RAISE NOTICE 'card_issue_date 필드 추가됨';
    ELSE
        RAISE NOTICE 'card_issue_date 필드 이미 존재함';
    END IF;

    -- card_expiry_date 필드 추가
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'customers' AND column_name = 'card_expiry_date'
    ) THEN
        ALTER TABLE customers ADD COLUMN card_expiry_date DATE;
        RAISE NOTICE 'card_expiry_date 필드 추가됨';
    ELSE
        RAISE NOTICE 'card_expiry_date 필드 이미 존재함';
    END IF;

    -- customer_type_codes 필드 추가
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'customers' AND column_name = 'customer_type_codes'
    ) THEN
        ALTER TABLE customers ADD COLUMN customer_type_codes TEXT[];
        CREATE INDEX IF NOT EXISTS idx_customers_type_codes ON customers USING GIN(customer_type_codes);
        RAISE NOTICE 'customer_type_codes 필드 추가됨';
    ELSE
        RAISE NOTICE 'customer_type_codes 필드 이미 존재함';
    END IF;

    -- current_type_code 필드 추가
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'customers' AND column_name = 'current_type_code'
    ) THEN
        ALTER TABLE customers ADD COLUMN current_type_code VARCHAR(10);
        CREATE INDEX IF NOT EXISTS idx_customers_current_type ON customers(current_type_code);
        RAISE NOTICE 'current_type_code 필드 추가됨';
    ELSE
        RAISE NOTICE 'current_type_code 필드 이미 존재함';
    END IF;

    -- type_history 필드 추가
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'customers' AND column_name = 'type_history'
    ) THEN
        ALTER TABLE customers ADD COLUMN type_history JSONB DEFAULT '[]'::jsonb;
        CREATE INDEX IF NOT EXISTS idx_customers_type_history ON customers USING GIN(type_history);
        RAISE NOTICE 'type_history 필드 추가됨';
    ELSE
        RAISE NOTICE 'type_history 필드 이미 존재함';
    END IF;
END $$;

-- 6. consultations 테이블에 customers FK 추가 (기존 테이블 업그레이드)
-- 주의: 기존 customer_id 컬럼이 VARCHAR로 존재하므로, FK 관계만 추가
DO $$
BEGIN
    -- FK 제약조건이 없으면 추가 (데이터 적재 후 실행 권장)
    -- ALTER TABLE consultations
    -- ADD CONSTRAINT fk_consultations_customer
    -- FOREIGN KEY (customer_id) REFERENCES customers(id);

    RAISE NOTICE 'FK 제약조건은 데이터 적재 후 별도로 추가하세요';
END $$;

-- 7. 성공 메시지 출력
DO $$
BEGIN
    RAISE NOTICE '======================================';
    RAISE NOTICE 'Customers 테이블 설정 완료:';
    RAISE NOTICE '- customers 테이블 생성됨';
    RAISE NOTICE '- customer_grade Enum 생성됨';
    RAISE NOTICE '- 인덱스 생성됨 (name, phone, gender, age_group, birth_date, grade, card_type)';
    RAISE NOTICE '- GIN 인덱스 생성됨 (personality_tags, communication_style)';
    RAISE NOTICE '======================================';
END $$;
