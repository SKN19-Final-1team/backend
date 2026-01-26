-- ============================================================
-- CALL:ACT 최근 상담 이력 유의미성 판단 로직
-- 작성일: 2026-01-24
-- 수정일: 2026-01-24 (컬럼명 수정: consultation_date → call_date)
-- ============================================================
-- 목적:
-- 최근 상담 이력이 현재 인입 케이스와 관련이 있는지 판단
-- - 1년 전 상담 → 유의미하지 않음
-- - 7일~30일 전 상담 → 유의미할 수 있음 (유사 케이스 가능성)
-- - 고객 성향 파악 시에도 최근 이력 기준 적용
-- ============================================================

-- ============================================================
-- 1. 상담 이력 유의미성 판단 함수
-- ============================================================
CREATE OR REPLACE FUNCTION fn_get_consultation_relevance(
    p_customer_id VARCHAR(50),
    p_days_threshold INT DEFAULT 30  -- 기본 30일
)
RETURNS TABLE (
    consultation_id VARCHAR(50),
    consultation_date DATE,
    "category_sub" VARCHAR(100),
    days_ago INT,
    relevance_level VARCHAR(20),  -- 'high', 'medium', 'low', 'none'
    relevance_score INT           -- 100 ~ 0
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id as consultation_id,
        c.call_date as consultation_date,
        c."category_sub",
        (CURRENT_DATE - c.call_date)::INT as days_ago,
        CASE
            WHEN (CURRENT_DATE - c.call_date) <= 7 THEN 'high'::VARCHAR(20)
            WHEN (CURRENT_DATE - c.call_date) <= 30 THEN 'medium'::VARCHAR(20)
            WHEN (CURRENT_DATE - c.call_date) <= 90 THEN 'low'::VARCHAR(20)
            ELSE 'none'::VARCHAR(20)
        END as relevance_level,
        CASE
            WHEN (CURRENT_DATE - c.call_date) <= 7 THEN 100 - ((CURRENT_DATE - c.call_date) * 5)
            WHEN (CURRENT_DATE - c.call_date) <= 30 THEN 65 - ((CURRENT_DATE - c.call_date - 7) * 2)
            WHEN (CURRENT_DATE - c.call_date) <= 90 THEN 20 - ((CURRENT_DATE - c.call_date - 30) / 6)
            ELSE 0
        END::INT as relevance_score
    FROM consultations c
    WHERE c.customer_id = p_customer_id
    ORDER BY c.call_date DESC
    LIMIT 10;  -- 최근 10건만
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION fn_get_consultation_relevance IS '고객의 최근 상담 이력과 유의미성 점수 반환';

-- ============================================================
-- 2. 고객 성향 유의미성 판단 함수
-- ============================================================
CREATE OR REPLACE FUNCTION fn_get_customer_persona_relevance(
    p_customer_id VARCHAR(50)
)
RETURNS TABLE (
    customer_id VARCHAR(50),
    current_type_code VARCHAR(10),
    last_consultation_date DATE,
    days_since_last_consultation INT,
    persona_relevance VARCHAR(20),  -- 'valid', 'stale', 'outdated'
    recent_consultation_count INT,  -- 최근 90일 상담 건수
    recommendation TEXT
) AS $$
DECLARE
    v_last_date DATE;
    v_days_ago INT;
    v_recent_count INT;
BEGIN
    -- 최근 상담일 조회
    SELECT c.last_consultation_date INTO v_last_date
    FROM customers c
    WHERE c.id = p_customer_id;

    -- 마지막 상담으로부터 경과 일수
    v_days_ago := COALESCE(CURRENT_DATE - v_last_date, 9999);

    -- 최근 90일 상담 건수
    SELECT COUNT(*) INTO v_recent_count
    FROM consultations con
    WHERE con.customer_id = p_customer_id
      AND con.call_date >= CURRENT_DATE - INTERVAL '90 days';

    RETURN QUERY
    SELECT
        p_customer_id,
        c.current_type_code,
        c.last_consultation_date,
        v_days_ago,
        CASE
            WHEN v_days_ago <= 30 THEN 'valid'::VARCHAR(20)
            WHEN v_days_ago <= 180 THEN 'stale'::VARCHAR(20)
            ELSE 'outdated'::VARCHAR(20)
        END,
        v_recent_count,
        CASE
            WHEN v_days_ago <= 30 THEN '최근 상담 이력이 있어 성향 정보가 유효합니다.'::TEXT
            WHEN v_days_ago <= 180 THEN '상담 이력이 다소 오래되었습니다. 성향 정보 재확인이 필요할 수 있습니다.'::TEXT
            ELSE '상담 이력이 6개월 이상 지났습니다. 성향 정보가 현재와 다를 수 있습니다.'::TEXT
        END
    FROM customers c
    WHERE c.id = p_customer_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION fn_get_customer_persona_relevance IS '고객 성향 정보의 유의미성 판단';

-- ============================================================
-- 3. 유사 케이스 탐지 함수 (동일 카테고리, 최근 상담)
-- ============================================================
CREATE OR REPLACE FUNCTION fn_find_similar_recent_consultations(
    p_customer_id VARCHAR(50),
    p_category VARCHAR(100),
    p_days_threshold INT DEFAULT 30
)
RETURNS TABLE (
    consultation_id VARCHAR(50),
    consultation_date DATE,
    "category_sub" VARCHAR(100),
    title TEXT,
    days_ago INT,
    is_fcr_failure BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id as consultation_id,
        c.call_date as consultation_date,
        c."category_sub",
        c.title,
        (CURRENT_DATE - c.call_date)::INT as days_ago,
        -- 동일 카테고리로 7일 이내 재문의가 있었다면 FCR 실패
        EXISTS (
            SELECT 1 FROM consultations c2
            WHERE c2.customer_id = c.customer_id
              AND c2."category_sub" = c."category_sub"
              AND c2.call_date > c.call_date
              AND c2.call_date <= c.call_date + INTERVAL '7 days'
        ) as is_fcr_failure
    FROM consultations c
    WHERE c.customer_id = p_customer_id
      AND c."category_sub" = p_category
      AND c.call_date >= CURRENT_DATE - (p_days_threshold || ' days')::INTERVAL
    ORDER BY c.call_date DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION fn_find_similar_recent_consultations IS '동일 고객의 동일 카테고리 최근 상담 이력 (FCR 실패 여부 포함)';

-- ============================================================
-- 4. 상담 가이던스용 고객 정보 뷰 (유의미성 포함)
-- ============================================================
CREATE OR REPLACE VIEW v_customer_guidance_info AS
SELECT
    c.id as customer_id,
    c.name,
    c.phone,
    c.grade,
    c.card_type,
    c.current_type_code,
    c.personality_tags,
    c.llm_guidance,
    c.total_consultations,
    c.resolved_first_call,
    c.last_consultation_date,
    -- 유의미성 판단
    (CURRENT_DATE - c.last_consultation_date)::INT as days_since_last,
    CASE
        WHEN c.last_consultation_date IS NULL THEN 'new_customer'
        WHEN (CURRENT_DATE - c.last_consultation_date) <= 7 THEN 'recent_active'
        WHEN (CURRENT_DATE - c.last_consultation_date) <= 30 THEN 'active'
        WHEN (CURRENT_DATE - c.last_consultation_date) <= 90 THEN 'moderate'
        WHEN (CURRENT_DATE - c.last_consultation_date) <= 365 THEN 'dormant'
        ELSE 'inactive'
    END as activity_status,
    -- FCR 비율
    CASE
        WHEN c.total_consultations > 0
        THEN ROUND((c.resolved_first_call::NUMERIC / c.total_consultations) * 100, 1)
        ELSE NULL
    END as fcr_rate,
    -- 가이던스 유효성
    CASE
        WHEN c.last_consultation_date IS NULL THEN '신규 고객입니다. 친절하게 안내해주세요.'
        WHEN (CURRENT_DATE - c.last_consultation_date) <= 30 THEN
            '최근 상담 이력이 있습니다. 이전 문의와 연관될 수 있으니 확인해주세요.'
        WHEN (CURRENT_DATE - c.last_consultation_date) <= 180 THEN
            '상담 이력이 있으나 다소 오래되었습니다. 현재 상황을 새로 파악해주세요.'
        ELSE
            '오랜만에 연락하신 고객입니다. 기본 정보부터 확인해주세요.'
    END as guidance_note
FROM customers c;

COMMENT ON VIEW v_customer_guidance_info IS '상담 시 사용할 고객 정보 및 유의미성 판단';

-- ============================================================
-- 5. 최근 상담 이력 요약 뷰 (고객별)
-- ============================================================
CREATE OR REPLACE VIEW v_customer_recent_history AS
SELECT
    c.customer_id,
    COUNT(*) as consultation_count,
    MAX(c.call_date) as last_consultation,
    MIN(c.call_date) as first_consultation,
    array_agg(DISTINCT c."category_sub") as categories,
    (CURRENT_DATE - MAX(c.call_date))::INT as days_since_last,
    CASE
        WHEN (CURRENT_DATE - MAX(c.call_date)) <= 30 THEN TRUE
        ELSE FALSE
    END as has_relevant_history
FROM consultations c
WHERE c.call_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY c.customer_id;

COMMENT ON VIEW v_customer_recent_history IS '고객별 최근 90일 상담 이력 요약';

-- ============================================================
-- 완료 메시지
-- ============================================================
DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE '상담 이력 유의미성 판단 로직 생성 완료';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '생성된 함수:';
    RAISE NOTICE '  - fn_get_consultation_relevance(customer_id, days)';
    RAISE NOTICE '  - fn_get_customer_persona_relevance(customer_id)';
    RAISE NOTICE '  - fn_find_similar_recent_consultations(customer_id, category, days)';
    RAISE NOTICE '생성된 뷰:';
    RAISE NOTICE '  - v_customer_guidance_info (상담 가이던스용 고객 정보)';
    RAISE NOTICE '  - v_customer_recent_history (최근 90일 상담 이력 요약)';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '사용 예시:';
    RAISE NOTICE '  SELECT * FROM fn_get_consultation_relevance(''CUST-TEDDY-00001'', 30);';
    RAISE NOTICE '  SELECT * FROM fn_get_customer_persona_relevance(''CUST-TEDDY-00001'');';
    RAISE NOTICE '  SELECT * FROM v_customer_guidance_info WHERE customer_id = ''CUST-TEDDY-00001'';';
    RAISE NOTICE '============================================================';
END $$;
