-- ==================================================
-- Persona Types 테이블 생성 스크립트
-- ==================================================
-- Description: 고객 페르소나 유형 마스터 테이블
-- Author: CALL:ACT Team
-- Date: 2026-01-23
-- ==================================================

-- 1. persona_types 테이블 생성
CREATE TABLE IF NOT EXISTS persona_types (
    -- 유형 코드 (Primary Key)
    code VARCHAR(10) PRIMARY KEY,             -- 'N1', 'S1' 등

    -- 기본 정보
    name VARCHAR(50) NOT NULL,                -- '실용주의형'
    description TEXT,                         -- '불필요한 말 없이 바로 문의사항을 말함'
    category VARCHAR(20) NOT NULL,            -- 'normal' 또는 'special'

    -- LLM 가이던스용 태그
    personality_tags TEXT[],                  -- ['practical', 'direct', 'efficient']
    communication_style JSONB,                -- {"speed": "fast", "tone": "direct"}
    llm_guidance TEXT,                        -- 'LLM에게 전달할 상담 가이드'

    -- 분포 비율 (데이터 생성 시 참고)
    distribution_ratio DECIMAL(4,2),          -- 0.25 (25%)

    -- 표시 순서
    display_order INT DEFAULT 0,

    -- 활성화 여부
    is_active BOOLEAN DEFAULT TRUE,

    -- 타임스탬프
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP
);

-- 2. 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_persona_types_category ON persona_types(category);
CREATE INDEX IF NOT EXISTS idx_persona_types_active ON persona_types(is_active);

-- 3. 초기 데이터 삽입 (12개 유형: N1-N4, S1-S8)
INSERT INTO persona_types (code, name, description, category, personality_tags, communication_style, llm_guidance, distribution_ratio, display_order)
VALUES
    -- Normal Types (일반 고객 유형)
    ('N1', '실용주의형', '불필요한 말 없이 바로 문의사항을 말함', 'normal',
     ARRAY['practical', 'direct', 'efficient'],
     '{"speed": "fast", "tone": "direct"}'::jsonb,
     '바로 본론으로 진행하세요. 간결하고 명확하게 답변해주세요.',
     0.20, 1),

    ('N2', '친화적수다형', '사적인 이야기나 본인 상황을 길게 설명함', 'normal',
     ARRAY['friendly', 'talkative', 'personal'],
     '{"speed": "moderate", "tone": "warm"}'::jsonb,
     '고객의 이야기를 경청하고 공감해주세요. 친근하게 응대하세요.',
     0.15, 2),

    ('N3', '신중/보안중시형', '신중하고 의심을 보임', 'normal',
     ARRAY['cautious', 'security_conscious', 'suspicious'],
     '{"speed": "slow", "tone": "formal"}'::jsonb,
     '본인 확인 절차를 상세히 안내하고, 보안 관련 우려를 해소해주세요.',
     0.10, 3),

    ('N4', '무관심/수동형', '관심 없이 최소한의 답변만 함, 설명을 듣지 않으려 함', 'normal',
     ARRAY['passive', 'disengaged', 'minimal_response'],
     '{"speed": "moderate", "tone": "engaging"}'::jsonb,
     '핵심만 간결하게 전달하되, 중요한 사항은 반드시 확인받으세요. 관심을 끌 수 있는 혜택 정보를 활용하세요.',
     0.05, 4),

    -- Special Types (특수 고객 유형)
    ('S1', '급한성격형', '빠른 처리를 선호함', 'special',
     ARRAY['impatient', 'urgent', 'busy'],
     '{"speed": "fast", "tone": "concise"}'::jsonb,
     '신속하게 처리해주세요. 핵심만 간결하게 전달하세요.',
     0.12, 5),

    ('S2', '꼼꼼상세형', '상세한 설명을 요구함', 'special',
     ARRAY['detailed', 'analytical', 'thorough'],
     '{"speed": "slow", "tone": "thorough"}'::jsonb,
     '단계별로 상세하게 설명해주세요. 모든 내용을 빠짐없이 안내하세요.',
     0.08, 6),

    ('S3', '이해부족형', '설명을 잘 이해하지 못하여 반복적으로 확인함', 'special',
     ARRAY['confused', 'needs_repetition', 'patient_required'],
     '{"speed": "slow", "tone": "patient"}'::jsonb,
     '쉬운 용어로 천천히 설명해주세요. 이해했는지 확인하고 필요시 반복 안내하세요.',
     0.06, 7),

    ('S4', '반복민원형', '과거의 상담 이력을 언급하며 해결되지 않은 불만을 제기함', 'special',
     ARRAY['repeat_caller', 'frustrated', 'unresolved'],
     '{"speed": "moderate", "tone": "solution_focused"}'::jsonb,
     '이전 상담 이력을 확인하고 근본적인 해결책을 제시하세요. 진심으로 사과하세요.',
     0.05, 8),

    ('S5', '불만형', '분노, 짜증을 드러냄', 'special',
     ARRAY['angry', 'frustrated', 'demanding'],
     '{"speed": "moderate", "tone": "calm_professional"}'::jsonb,
     '차분하게 경청하고 공감해주세요. 감정을 누그러뜨린 후 해결 방안을 제시하세요.',
     0.05, 9),

    ('S6', '고령/디지털취약형', '디지털 기기 사용이 어려워 전화 상담을 선호함', 'special',
     ARRAY['elderly', 'digital_vulnerable', 'phone_preferred'],
     '{"speed": "slow", "tone": "patient"}'::jsonb,
     '천천히 쉬운 말로 안내하세요. 앱/웹 대신 전화/방문 채널 위주로 안내하세요. 존댓말을 사용하고 경청하세요.',
     0.05, 10),

    ('S7', '다국어/외국인형', '한국어가 모국어가 아니어서 의사소통에 어려움이 있음', 'special',
     ARRAY['foreign', 'language_barrier', 'simple_korean'],
     '{"speed": "slow", "tone": "clear"}'::jsonb,
     '짧고 명확한 문장으로 안내하세요. 전문 용어 대신 쉬운 표현을 사용하고, 이해 여부를 자주 확인하세요.',
     0.04, 11),

    ('S8', 'VIP/특별관리형', '높은 등급의 고객으로 우대 서비스를 기대함', 'special',
     ARRAY['vip', 'premium', 'high_expectation'],
     '{"speed": "moderate", "tone": "premium_service"}'::jsonb,
     '최상의 서비스를 제공하세요. VIP 전용 혜택을 우선 안내하고, 특별한 관심과 배려를 보여주세요.',
     0.05, 12)

ON CONFLICT (code) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    category = EXCLUDED.category,
    personality_tags = EXCLUDED.personality_tags,
    communication_style = EXCLUDED.communication_style,
    llm_guidance = EXCLUDED.llm_guidance,
    distribution_ratio = EXCLUDED.distribution_ratio,
    display_order = EXCLUDED.display_order,
    updated_at = NOW();

-- 4. 테이블 및 컬럼 코멘트
COMMENT ON TABLE persona_types IS '고객 페르소나 유형 마스터 테이블 - LLM 상담 가이던스용';
COMMENT ON COLUMN persona_types.code IS '유형 코드 (N1~N4: 일반, S1~S8: 특수)';
COMMENT ON COLUMN persona_types.name IS '유형명 (예: 실용주의형)';
COMMENT ON COLUMN persona_types.description IS '유형 설명';
COMMENT ON COLUMN persona_types.category IS '유형 카테고리 (normal/special)';
COMMENT ON COLUMN persona_types.personality_tags IS 'LLM 가이던스용 특성 태그 배열';
COMMENT ON COLUMN persona_types.communication_style IS '의사소통 스타일 (JSON)';
COMMENT ON COLUMN persona_types.llm_guidance IS 'LLM에게 전달할 상담 가이드 메시지';
COMMENT ON COLUMN persona_types.distribution_ratio IS '데이터 생성 시 분포 비율';

-- 5. 성공 메시지
DO $$
BEGIN
    RAISE NOTICE '======================================';
    RAISE NOTICE 'Persona Types 테이블 설정 완료:';
    RAISE NOTICE '- persona_types 테이블 생성됨';
    RAISE NOTICE '- 12개 페르소나 유형 삽입됨';
    RAISE NOTICE '  - Normal: N1(실용주의형), N2(친화적수다형), N3(신중/보안중시형), N4(무관심/수동형)';
    RAISE NOTICE '  - Special: S1(급한성격형), S2(꼼꼼상세형), S3(이해부족형), S4(반복민원형), S5(불만형), S6(고령/디지털취약형), S7(다국어/외국인형), S8(VIP/특별관리형)';
    RAISE NOTICE '======================================';
END $$;
