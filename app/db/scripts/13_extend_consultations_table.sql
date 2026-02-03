-- ==================================================
-- Consultations 테이블 확장 스크립트
-- ==================================================
-- Description: Frontend 요구사항 반영을 위한 consultations 테이블 확장
-- Author: CALL:ACT Team
-- Date: 2026-02-03
-- Reference: docs/04_dev/04_Backend-Frontend_연동/03_Frontend_데이터_요구사항_명세서.md
-- ==================================================

-- 1. 시간 정보 확장
-- 통화 종료 시간
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consultations' AND column_name = 'call_end_time'
    ) THEN
        ALTER TABLE consultations ADD COLUMN call_end_time TIME;
        RAISE NOTICE 'call_end_time 필드 추가됨';
    ELSE
        RAISE NOTICE 'call_end_time 필드 이미 존재함';
    END IF;
END $$;

-- 후처리 시간 (ACW: After Call Work)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consultations' AND column_name = 'acw_duration'
    ) THEN
        ALTER TABLE consultations ADD COLUMN acw_duration VARCHAR(20);
        RAISE NOTICE 'acw_duration 필드 추가됨';
    ELSE
        RAISE NOTICE 'acw_duration 필드 이미 존재함';
    END IF;
END $$;

-- 2. 상담 내용 확장
-- 상담 전문 (화자 분리된 채팅 형식)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consultations' AND column_name = 'transcript'
    ) THEN
        ALTER TABLE consultations ADD COLUMN transcript JSONB;
        RAISE NOTICE 'transcript 필드 추가됨';
    ELSE
        RAISE NOTICE 'transcript 필드 이미 존재함';
    END IF;
END $$;

-- AI 상담 요약
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consultations' AND column_name = 'ai_summary'
    ) THEN
        ALTER TABLE consultations ADD COLUMN ai_summary TEXT;
        RAISE NOTICE 'ai_summary 필드 추가됨';
    ELSE
        RAISE NOTICE 'ai_summary 필드 이미 존재함';
    END IF;
END $$;

-- 상담사 메모
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consultations' AND column_name = 'agent_notes'
    ) THEN
        ALTER TABLE consultations ADD COLUMN agent_notes TEXT;
        RAISE NOTICE 'agent_notes 필드 추가됨';
    ELSE
        RAISE NOTICE 'agent_notes 필드 이미 존재함';
    END IF;
END $$;

-- 3. 후속 조치 필드
-- 후속 일정
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consultations' AND column_name = 'follow_up_schedule'
    ) THEN
        ALTER TABLE consultations ADD COLUMN follow_up_schedule TEXT;
        RAISE NOTICE 'follow_up_schedule 필드 추가됨';
    ELSE
        RAISE NOTICE 'follow_up_schedule 필드 이미 존재함';
    END IF;
END $$;

-- 이관 부서
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consultations' AND column_name = 'transfer_department'
    ) THEN
        ALTER TABLE consultations ADD COLUMN transfer_department VARCHAR(100);
        RAISE NOTICE 'transfer_department 필드 추가됨';
    ELSE
        RAISE NOTICE 'transfer_department 필드 이미 존재함';
    END IF;
END $$;

-- 이관 전달 사항
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consultations' AND column_name = 'transfer_notes'
    ) THEN
        ALTER TABLE consultations ADD COLUMN transfer_notes TEXT;
        RAISE NOTICE 'transfer_notes 필드 추가됨';
    ELSE
        RAISE NOTICE 'transfer_notes 필드 이미 존재함';
    END IF;
END $$;

-- 4. 참조 문서
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consultations' AND column_name = 'referenced_documents'
    ) THEN
        ALTER TABLE consultations ADD COLUMN referenced_documents JSONB;
        RAISE NOTICE 'referenced_documents 필드 추가됨';
    ELSE
        RAISE NOTICE 'referenced_documents 필드 이미 존재함';
    END IF;
END $$;

-- 5. 감정/피드백 분석 (선택적)
-- 감정 분석 결과
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consultations' AND column_name = 'sentiment'
    ) THEN
        ALTER TABLE consultations ADD COLUMN sentiment VARCHAR(20);
        RAISE NOTICE 'sentiment 필드 추가됨';
    ELSE
        RAISE NOTICE 'sentiment 필드 이미 존재함';
    END IF;
END $$;

-- 피드백 점수 (0-100)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consultations' AND column_name = 'feedback_score'
    ) THEN
        ALTER TABLE consultations ADD COLUMN feedback_score INT;
        RAISE NOTICE 'feedback_score 필드 추가됨';
    ELSE
        RAISE NOTICE 'feedback_score 필드 이미 존재함';
    END IF;
END $$;

-- 만족도 점수 (1-5)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consultations' AND column_name = 'satisfaction_score'
    ) THEN
        ALTER TABLE consultations ADD COLUMN satisfaction_score INT;
        RAISE NOTICE 'satisfaction_score 필드 추가됨';
    ELSE
        RAISE NOTICE 'satisfaction_score 필드 이미 존재함';
    END IF;
END $$;

-- 6. 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_consultations_call_end_time ON consultations(call_end_time);
CREATE INDEX IF NOT EXISTS idx_consultations_sentiment ON consultations(sentiment);
CREATE INDEX IF NOT EXISTS idx_consultations_transfer_department ON consultations(transfer_department);
CREATE INDEX IF NOT EXISTS idx_consultations_referenced_documents ON consultations USING GIN(referenced_documents);
CREATE INDEX IF NOT EXISTS idx_consultations_transcript ON consultations USING GIN(transcript);

-- 7. 컬럼 코멘트
COMMENT ON COLUMN consultations.call_end_time IS '통화 종료 시간 (HH:MM:SS)';
COMMENT ON COLUMN consultations.acw_duration IS '후처리 시간 (형식: MM:SS)';
COMMENT ON COLUMN consultations.transcript IS '화자 분리된 상담 전문 (JSONB: {messages: [{speaker, message, timestamp}]})';
COMMENT ON COLUMN consultations.ai_summary IS 'AI 생성 상담 요약';
COMMENT ON COLUMN consultations.agent_notes IS '상담사 메모';
COMMENT ON COLUMN consultations.follow_up_schedule IS '후속 일정 (추후 할 일)';
COMMENT ON COLUMN consultations.transfer_department IS '이관 부서';
COMMENT ON COLUMN consultations.transfer_notes IS '이관 부서 전달 사항';
COMMENT ON COLUMN consultations.referenced_documents IS '참조 문서 목록 (JSONB: [{step_number, doc_id, doc_type, title, used}])';
COMMENT ON COLUMN consultations.sentiment IS '감정 분석 결과 (positive, neutral, negative)';
COMMENT ON COLUMN consultations.feedback_score IS '피드백 점수 (0-100)';
COMMENT ON COLUMN consultations.satisfaction_score IS '고객 만족도 (1-5)';

-- 8. 성공 메시지
DO $$
BEGIN
    RAISE NOTICE '======================================';
    RAISE NOTICE 'Consultations 테이블 확장 완료:';
    RAISE NOTICE '추가된 필드:';
    RAISE NOTICE '  - 시간: call_end_time, acw_duration';
    RAISE NOTICE '  - 내용: transcript, ai_summary, agent_notes';
    RAISE NOTICE '  - 후속: follow_up_schedule, transfer_department, transfer_notes';
    RAISE NOTICE '  - 문서: referenced_documents';
    RAISE NOTICE '  - 분석: sentiment, feedback_score, satisfaction_score';
    RAISE NOTICE '======================================';
END $$;
