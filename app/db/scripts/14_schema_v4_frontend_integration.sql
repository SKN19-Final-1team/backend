-- ==================================================
-- CALL:ACT DB 스키마 v4.0 - Frontend 통합 버전
-- ==================================================
-- Description: Frontend 요구사항 반영 + 페르소나 6타입 + 자동 통계 트리거
-- Author: CALL:ACT Team
-- Date: 2026-02-03
-- Version: 4.0
-- ==================================================
-- 변경 사항:
--   1. persona_types: 12타입 → 6타입 (팀원 LLM 분류 기준)
--   2. customers: llm_guidance, personality_tags, communication_style 삭제
--   3. customers: current_type_code → persona_types.code FK 연결
--   4. consultations: 피드백, 녹취, 후속조치 필드 추가
--   5. 자동 통계 업데이트 트리거 추가
--   6. 고객+페르소나 JOIN 뷰 생성
-- ==================================================

-- ============================================================
-- PART 1: persona_types 테이블 (6타입으로 교체)
-- ============================================================

-- 기존 데이터 삭제 (FK 제약 없으므로 안전)
DELETE FROM persona_types;

-- 5타입 삽입 (팀원 LLM 분류 기준 - 2026-02-03 확정)
INSERT INTO persona_types (code, name, description, category, personality_tags, communication_style, llm_guidance, distribution_ratio, display_order, is_active)
VALUES
    -- Normal Types (2개)
    ('N1', '일반형', '큰 특징이 없고 바로 문의사항을 말함', 'normal',
     ARRAY['normal', 'direct', 'straightforward'],
     '{"speed": "moderate", "tone": "neutral"}'::jsonb,
     '표준 응대 매뉴얼대로 진행하세요. 고객이 원하는 바를 파악하여 신속하게 처리하면 됩니다.',
     0.50, 1, TRUE),

    ('N2', '수다형', '사적인 이야기나 본인 상황을 길게 설명함', 'normal',
     ARRAY['talkative', 'personal', 'expressive'],
     '{"speed": "moderate", "tone": "warm"}'::jsonb,
     '고객의 이야기를 충분히 들어주세요. 공감 표현을 자주 사용하고, 적절한 타이밍에 본론으로 유도하세요. "네, 그러셨군요"와 같은 리액션이 효과적입니다.',
     0.20, 2, TRUE),

    -- Special Types (3개)
    ('S1', '급한성격형', '빠른 처리를 선호함', 'special',
     ARRAY['impatient', 'urgent', 'time_sensitive'],
     '{"speed": "fast", "tone": "concise"}'::jsonb,
     '시간을 최소화하세요. 인사말은 짧게, 즉시 해결 가능한 방법을 먼저 제시하세요. 불필요한 설명은 생략하고 핵심만 전달하세요.',
     0.10, 3, TRUE),

    ('S2', '디지털미아형', '기술적인 조작에 서툴고 앱 사용을 어려워함', 'special',
     ARRAY['tech_challenged', 'needs_guidance', 'patient_required'],
     '{"speed": "slow", "tone": "patient"}'::jsonb,
     '천천히 단계별로 안내하세요. 전문 용어를 피하고 "화면 왼쪽 위", "빨간색 버튼"처럼 시각적으로 설명하세요. 각 단계마다 "여기까지 되셨나요?"로 확인하세요.',
     0.10, 4, TRUE),

    ('S3', '불만형', '분노, 짜증을 드러냄', 'special',
     ARRAY['angry', 'frustrated', 'demanding'],
     '{"speed": "moderate", "tone": "calm_professional"}'::jsonb,
     '먼저 진심으로 사과하고 고객의 감정을 인정하세요. "불편을 드려 정말 죄송합니다"로 시작하고, 즉각적인 해결책을 제시하세요. 절대 감정적으로 대응하지 마세요.',
     0.10, 5, TRUE)

ON CONFLICT (code) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    category = EXCLUDED.category,
    personality_tags = EXCLUDED.personality_tags,
    communication_style = EXCLUDED.communication_style,
    llm_guidance = EXCLUDED.llm_guidance,
    distribution_ratio = EXCLUDED.distribution_ratio,
    display_order = EXCLUDED.display_order,
    is_active = EXCLUDED.is_active,
    updated_at = NOW();

DO $$
BEGIN
    RAISE NOTICE '[1/6] persona_types 5타입 설정 완료 (N1, N2, S1, S2, S3)';
END $$;

-- ============================================================
-- PART 2: customers 테이블 정리
-- ============================================================

-- 2.0 의존하는 뷰 먼저 삭제 (기존 v_customer_guidance_info가 llm_guidance에 의존)
DROP VIEW IF EXISTS v_customer_guidance_info CASCADE;

-- 2.1 불필요한 컬럼 삭제
ALTER TABLE customers DROP COLUMN IF EXISTS llm_guidance;
ALTER TABLE customers DROP COLUMN IF EXISTS personality_tags;
ALTER TABLE customers DROP COLUMN IF EXISTS communication_style;
ALTER TABLE customers DROP COLUMN IF EXISTS customer_type_codes;

-- 2.2 current_type_code 값 정리 (5타입에 없는 코드 → NULL)
UPDATE customers
SET current_type_code = NULL
WHERE current_type_code IS NOT NULL
  AND current_type_code NOT IN ('N1', 'N2', 'S1', 'S2', 'S3');

-- 2.3 FK 제약조건 추가 (이미 있으면 무시)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'fk_customers_persona_type'
    ) THEN
        ALTER TABLE customers
        ADD CONSTRAINT fk_customers_persona_type
        FOREIGN KEY (current_type_code) REFERENCES persona_types(code);
        RAISE NOTICE 'FK 제약조건 fk_customers_persona_type 추가됨';
    ELSE
        RAISE NOTICE 'FK 제약조건 fk_customers_persona_type 이미 존재함';
    END IF;
END $$;

-- 2.4 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_customers_current_type ON customers(current_type_code);

DO $$
BEGIN
    RAISE NOTICE '[2/6] customers 테이블 정리 완료';
END $$;

-- ============================================================
-- PART 3: consultations 테이블 확장
-- ============================================================

-- 3.1 시간 정보
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS call_end_time TIME;
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS acw_duration VARCHAR(20);

-- 3.2 상담 내용
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS transcript JSONB;
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS ai_summary TEXT;
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS agent_notes TEXT;

-- 3.3 후속 조치
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS follow_up_schedule TEXT;
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS transfer_department VARCHAR(100);
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS transfer_notes TEXT;

-- 3.4 참조 문서
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS referenced_documents JSONB;

-- 3.5 피드백 (신규)
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS feedback_text TEXT;
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS feedback_emotions JSONB;
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS emotion_score INT;

-- 3.6 감정/만족도
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS sentiment VARCHAR(20);
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS satisfaction_score INT;

-- 3.7 녹취 파일 (신규)
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS recording_file_path VARCHAR(500);
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS recording_duration VARCHAR(20);
ALTER TABLE consultations ADD COLUMN IF NOT EXISTS recording_file_size BIGINT;

-- 3.8 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_consultations_call_end_time ON consultations(call_end_time);
CREATE INDEX IF NOT EXISTS idx_consultations_sentiment ON consultations(sentiment);
CREATE INDEX IF NOT EXISTS idx_consultations_transfer_department ON consultations(transfer_department);
CREATE INDEX IF NOT EXISTS idx_consultations_emotion_score ON consultations(emotion_score);

-- GIN 인덱스 (JSONB 검색용)
CREATE INDEX IF NOT EXISTS idx_consultations_transcript ON consultations USING GIN(transcript);
CREATE INDEX IF NOT EXISTS idx_consultations_referenced_documents ON consultations USING GIN(referenced_documents);
CREATE INDEX IF NOT EXISTS idx_consultations_feedback_emotions ON consultations USING GIN(feedback_emotions);

-- 3.9 컬럼 코멘트
COMMENT ON COLUMN consultations.call_end_time IS '통화 종료 시간 (HH:MM:SS)';
COMMENT ON COLUMN consultations.acw_duration IS '후처리 시간 (형식: MM:SS)';
COMMENT ON COLUMN consultations.transcript IS '화자 분리된 상담 전문 (JSONB)';
COMMENT ON COLUMN consultations.ai_summary IS 'AI 생성 상담 요약';
COMMENT ON COLUMN consultations.agent_notes IS '상담사 메모';
COMMENT ON COLUMN consultations.follow_up_schedule IS '후속 일정 (추후 할 일)';
COMMENT ON COLUMN consultations.transfer_department IS '이관 부서';
COMMENT ON COLUMN consultations.transfer_notes IS '이관 부서 전달 사항';
COMMENT ON COLUMN consultations.referenced_documents IS '참조 문서 목록 (JSONB)';
COMMENT ON COLUMN consultations.feedback_text IS 'LLM 생성 상담 피드백 텍스트';
COMMENT ON COLUMN consultations.feedback_emotions IS '감정 분석 결과 (JSONB)';
COMMENT ON COLUMN consultations.emotion_score IS '감정 점수 (0-100)';
COMMENT ON COLUMN consultations.sentiment IS '전체 감정 분석 결과 (positive/neutral/negative)';
COMMENT ON COLUMN consultations.satisfaction_score IS '고객 만족도 (1-5)';
COMMENT ON COLUMN consultations.recording_file_path IS '녹취 파일 경로';
COMMENT ON COLUMN consultations.recording_duration IS '녹취 파일 길이 (HH:MM:SS)';
COMMENT ON COLUMN consultations.recording_file_size IS '녹취 파일 크기 (bytes)';

DO $$
BEGIN
    RAISE NOTICE '[3/6] consultations 테이블 확장 완료';
END $$;

-- ============================================================
-- PART 4: 자동 통계 업데이트 트리거
-- ============================================================

-- 4.1 상담 INSERT 시 고객 통계 자동 업데이트
CREATE OR REPLACE FUNCTION fn_update_customer_stats_on_consultation()
RETURNS TRIGGER AS $$
BEGIN
    -- total_consultations +1, last_consultation_date 업데이트
    UPDATE customers
    SET
        total_consultations = COALESCE(total_consultations, 0) + 1,
        last_consultation_date = NEW.call_date,
        updated_at = NOW()
    WHERE id = NEW.customer_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 기존 트리거 삭제 후 재생성
DROP TRIGGER IF EXISTS trg_consultation_insert_update_customer ON consultations;
CREATE TRIGGER trg_consultation_insert_update_customer
AFTER INSERT ON consultations
FOR EACH ROW
EXECUTE FUNCTION fn_update_customer_stats_on_consultation();

COMMENT ON FUNCTION fn_update_customer_stats_on_consultation IS '상담 INSERT 시 고객 통계 자동 업데이트';

-- 4.2 FCR(First Call Resolution) 자동 계산
CREATE OR REPLACE FUNCTION fn_update_fcr_on_consultation()
RETURNS TRIGGER AS $$
DECLARE
    v_rework_count INT;
BEGIN
    -- 같은 고객, 같은 카테고리로 7일 이내 이전 상담이 있으면 해당 상담의 FCR을 FALSE로
    UPDATE consultations
    SET fcr = FALSE, updated_at = NOW()
    WHERE customer_id = NEW.customer_id
      AND category_sub = NEW.category_sub
      AND call_date >= NEW.call_date - INTERVAL '7 days'
      AND call_date < NEW.call_date
      AND fcr = TRUE;

    GET DIAGNOSTICS v_rework_count = ROW_COUNT;

    -- 새 상담은 일단 FCR = TRUE로 설정
    UPDATE consultations
    SET fcr = TRUE
    WHERE id = NEW.id AND fcr IS NULL;

    -- resolved_first_call 재계산
    UPDATE customers
    SET
        resolved_first_call = (
            SELECT COUNT(*) FROM consultations
            WHERE customer_id = NEW.customer_id AND fcr = TRUE
        ),
        updated_at = NOW()
    WHERE id = NEW.customer_id;

    IF v_rework_count > 0 THEN
        RAISE NOTICE 'FCR 재계산: 고객 % - %건 재상담 감지', NEW.customer_id, v_rework_count;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_consultation_insert_check_fcr ON consultations;
CREATE TRIGGER trg_consultation_insert_check_fcr
AFTER INSERT ON consultations
FOR EACH ROW
EXECUTE FUNCTION fn_update_fcr_on_consultation();

COMMENT ON FUNCTION fn_update_fcr_on_consultation IS '상담 INSERT 시 FCR 자동 계산 (7일 이내 재상담 감지)';

DO $$
BEGIN
    RAISE NOTICE '[4/6] 자동 통계 업데이트 트리거 생성 완료';
END $$;

-- ============================================================
-- PART 5: 고객 + 페르소나 JOIN 뷰
-- ============================================================

CREATE OR REPLACE VIEW v_customer_with_persona AS
SELECT
    -- 고객 기본 정보
    c.id,
    c.name,
    c.phone,
    c.gender,
    c.age_group,
    c.birth_date,
    c.address,
    c.grade,

    -- 카드 정보
    c.card_type,
    c.card_number_last4,
    c.card_brand,
    c.card_issue_date,
    c.card_expiry_date,

    -- 페르소나 코드
    c.current_type_code,
    c.type_history,

    -- 상담 통계
    c.total_consultations,
    c.resolved_first_call,
    c.last_consultation_date,

    -- 계산된 값
    CASE
        WHEN c.birth_date IS NOT NULL
        THEN EXTRACT(YEAR FROM AGE(c.birth_date))::INT
        ELSE NULL
    END as age,

    CASE
        WHEN c.total_consultations > 0
        THEN ROUND((c.resolved_first_call::NUMERIC / c.total_consultations) * 100, 1)
        ELSE NULL
    END as fcr_rate,

    -- persona_types JOIN (핵심!)
    pt.name as persona_name,
    pt.description as persona_description,
    pt.category as persona_category,
    pt.personality_tags,
    pt.communication_style,
    pt.llm_guidance,

    -- 타임스탬프
    c.created_at,
    c.updated_at
FROM customers c
LEFT JOIN persona_types pt ON c.current_type_code = pt.code;

COMMENT ON VIEW v_customer_with_persona IS '고객 정보 + 페르소나 정보 통합 뷰 (API 조회용)';

DO $$
BEGIN
    RAISE NOTICE '[5/6] v_customer_with_persona 뷰 생성 완료';
END $$;

-- ============================================================
-- PART 6: 샘플 데이터 (빈 필드 형식 예시)
-- ============================================================

-- 6.1 샘플 상담 데이터 업데이트 (기존 데이터가 있다면)
DO $$
DECLARE
    v_sample_id VARCHAR(50);
BEGIN
    -- 첫 번째 상담 레코드 찾기
    SELECT id INTO v_sample_id FROM consultations LIMIT 1;

    IF v_sample_id IS NOT NULL THEN
        UPDATE consultations
        SET
            -- 시간 정보
            call_end_time = '14:37:30',
            acw_duration = '05:45',

            -- 상담 전문 (채팅 형식)
            transcript = '{
                "messages": [
                    {"speaker": "customer", "message": "안녕하세요, 카드를 분실했어요.", "timestamp": "14:32:00"},
                    {"speaker": "agent", "message": "안녕하세요, 고객님. 카드 분실 신고 도와드리겠습니다. 본인 확인을 위해 생년월일 말씀해주세요.", "timestamp": "14:32:15"},
                    {"speaker": "customer", "message": "1985년 3월 15일이요.", "timestamp": "14:32:30"},
                    {"speaker": "agent", "message": "확인되었습니다. 즉시 카드 사용을 정지하겠습니다.", "timestamp": "14:33:00"},
                    {"speaker": "customer", "message": "빨리 처리해주세요. 해외여행 가야해서요.", "timestamp": "14:33:15"},
                    {"speaker": "agent", "message": "카드 사용이 정지되었습니다. 재발급 카드는 등록된 주소로 3-5 영업일 내 배송됩니다. 긴급 배송 원하시면 추가 비용이 발생합니다.", "timestamp": "14:34:30"},
                    {"speaker": "customer", "message": "일반 배송으로 해주세요.", "timestamp": "14:35:00"},
                    {"speaker": "agent", "message": "네, 알겠습니다. 재발급 신청 완료되었습니다. 다른 문의사항 있으신가요?", "timestamp": "14:36:00"},
                    {"speaker": "customer", "message": "아니요, 감사합니다.", "timestamp": "14:37:00"},
                    {"speaker": "agent", "message": "감사합니다. 좋은 하루 되세요.", "timestamp": "14:37:15"}
                ],
                "total_turns": 10,
                "customer_turns": 5,
                "agent_turns": 5
            }'::jsonb,

            -- AI 요약
            ai_summary = '고객이 카드 분실을 신고하여 즉시 카드 사용을 정지 처리했습니다. 해외여행 일정이 있으나 일반 배송으로 재발급을 요청하여 3-5 영업일 내 배송 예정입니다.',

            -- 상담사 메모
            agent_notes = '해외여행 일정 확인 필요. 긴급 배송 거절함.',

            -- 후속 조치
            follow_up_schedule = '배송 완료 후 SMS 발송 예정 (3-5 영업일)',
            transfer_department = NULL,
            transfer_notes = NULL,

            -- 참조 문서
            referenced_documents = '[
                {"doc_id": "DOC-TERM-001", "doc_type": "terms", "title": "카드 분실/도난 처리 규정", "step": 1, "used": true, "view_count": 2},
                {"doc_id": "DOC-GUIDE-015", "doc_type": "guide", "title": "긴급 재발급 안내 가이드", "step": 2, "used": true, "view_count": 1}
            ]'::jsonb,

            -- 처리 타임라인
            processing_timeline = '[
                {"time": "14:32:30", "action": "본인 확인 완료", "category": "도난/분실 신청/해제"},
                {"time": "14:33:00", "action": "카드 사용 즉시 정지 처리", "category": "도난/분실 신청/해제"},
                {"time": "14:34:30", "action": "재발급 옵션 안내", "category": "긴급 배송 신청"},
                {"time": "14:36:00", "action": "일반 배송 재발급 신청 완료", "category": "도난/분실 신청/해제"}
            ]'::jsonb,

            -- 복합 처리 카테고리
            handled_categories = ARRAY['도난/분실 신청/해제', '긴급 배송 신청'],

            -- 피드백
            feedback_text = '전반적으로 안정적인 응대였습니다. 본인 확인 후 신속하게 카드 정지 처리한 점이 좋았습니다. 고객의 해외여행 일정을 인지하고 긴급 배송 옵션을 먼저 제안한 것도 적절했습니다. 개선점: 재발급 기간 안내 시 "영업일 기준"임을 더 명확히 강조하면 좋겠습니다.',
            feedback_emotions = '{
                "emotions": [
                    {"turn": 1, "speaker": "customer", "emotion": "anxious", "confidence": 0.85},
                    {"turn": 2, "speaker": "agent", "emotion": "professional", "confidence": 0.92},
                    {"turn": 3, "speaker": "customer", "emotion": "cooperative", "confidence": 0.88},
                    {"turn": 4, "speaker": "agent", "emotion": "reassuring", "confidence": 0.90},
                    {"turn": 5, "speaker": "customer", "emotion": "urgent", "confidence": 0.87},
                    {"turn": 6, "speaker": "agent", "emotion": "informative", "confidence": 0.91},
                    {"turn": 7, "speaker": "customer", "emotion": "accepting", "confidence": 0.85},
                    {"turn": 8, "speaker": "agent", "emotion": "helpful", "confidence": 0.93},
                    {"turn": 9, "speaker": "customer", "emotion": "satisfied", "confidence": 0.89},
                    {"turn": 10, "speaker": "agent", "emotion": "warm", "confidence": 0.90}
                ]
            }'::jsonb,
            emotion_score = 82,

            -- 감정/만족도
            sentiment = 'positive',
            satisfaction_score = 4,

            -- 녹취 (샘플 - 실제로는 파일 경로)
            recording_file_path = 'recordings/2026/01/28/' || v_sample_id || '.wav',
            recording_duration = '05:15',
            recording_file_size = 5242880,

            updated_at = NOW()
        WHERE id = v_sample_id;

        RAISE NOTICE '샘플 상담 데이터 업데이트 완료: %', v_sample_id;
    ELSE
        RAISE NOTICE '샘플 업데이트할 상담 데이터가 없습니다.';
    END IF;
END $$;

DO $$
BEGIN
    RAISE NOTICE '[6/6] 샘플 데이터 설정 완료';
END $$;

-- ============================================================
-- 완료 메시지
-- ============================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '══════════════════════════════════════════════════════════════';
    RAISE NOTICE 'CALL:ACT DB 스키마 v4.0 - Frontend 통합 버전 적용 완료';
    RAISE NOTICE '══════════════════════════════════════════════════════════════';
    RAISE NOTICE '';
    RAISE NOTICE '변경된 테이블:';
    RAISE NOTICE '  ├─ persona_types: 5타입 (N1, N2, S1, S2, S3)';
    RAISE NOTICE '  ├─ customers: 정리 완료 (llm_guidance 등 삭제, FK 추가)';
    RAISE NOTICE '  └─ consultations: 15개 컬럼 추가';
    RAISE NOTICE '';
    RAISE NOTICE '생성된 트리거:';
    RAISE NOTICE '  ├─ trg_consultation_insert_update_customer (통계 자동 업데이트)';
    RAISE NOTICE '  └─ trg_consultation_insert_check_fcr (FCR 자동 계산)';
    RAISE NOTICE '';
    RAISE NOTICE '생성된 뷰:';
    RAISE NOTICE '  └─ v_customer_with_persona (고객+페르소나 JOIN)';
    RAISE NOTICE '';
    RAISE NOTICE 'API 사용법:';
    RAISE NOTICE '  SELECT * FROM v_customer_with_persona WHERE id = ''CUST-TEDDY-00001'';';
    RAISE NOTICE '';
    RAISE NOTICE '페르소나 5타입 (팀원 LLM 분류 기준):';
    RAISE NOTICE '  N1: 일반형 - 표준 응대 매뉴얼대로 진행';
    RAISE NOTICE '  N2: 수다형 - 충분히 경청하고 공감 표현';
    RAISE NOTICE '  S1: 급한성격형 - 시간 최소화, 핵심만 전달';
    RAISE NOTICE '  S2: 디지털미아형 - 천천히 단계별로 시각적 안내';
    RAISE NOTICE '  S3: 불만형 - 먼저 사과, 즉각 해결책 제시';
    RAISE NOTICE '';
    RAISE NOTICE '══════════════════════════════════════════════════════════════';
END $$;
