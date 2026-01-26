-- ==================================================
-- CALL:ACT DB 스키마 검증 스크립트
-- ==================================================
-- Description: 전체 테이블 구조 및 데이터 무결성 검증
-- Author: CALL:ACT Team
-- Date: 2026-01-23
-- ==================================================

-- 1. 모든 테이블 목록 확인
SELECT '=== 1. 테이블 목록 ===' as section;
SELECT table_name,
       (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
FROM information_schema.tables t
WHERE table_schema = 'public'
ORDER BY table_name;

-- 2. 필수 테이블 존재 확인
SELECT '=== 2. 필수 테이블 존재 확인 ===' as section;
SELECT
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'employees') THEN 'OK' ELSE 'MISSING' END as employees,
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'consultations') THEN 'OK' ELSE 'MISSING' END as consultations,
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'customers') THEN 'OK' ELSE 'MISSING' END as customers,
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'persona_types') THEN 'OK' ELSE 'MISSING' END as persona_types;

-- 3. customers 테이블 컬럼 확인
SELECT '=== 3. customers 테이블 컬럼 ===' as section;
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_name = 'customers'
ORDER BY ordinal_position;

-- 4. persona_types 테이블 데이터 확인
SELECT '=== 4. persona_types 데이터 ===' as section;
SELECT code, name, category, distribution_ratio
FROM persona_types
ORDER BY display_order;

-- 5. customers 테이블 신규 필드 확인
SELECT '=== 5. customers 신규 필드 존재 확인 ===' as section;
SELECT
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'customers' AND column_name = 'current_type_code') THEN 'OK' ELSE 'MISSING' END as current_type_code,
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'customers' AND column_name = 'type_history') THEN 'OK' ELSE 'MISSING' END as type_history,
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'customers' AND column_name = 'card_issue_date') THEN 'OK' ELSE 'MISSING' END as card_issue_date,
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'customers' AND column_name = 'card_expiry_date') THEN 'OK' ELSE 'MISSING' END as card_expiry_date;

-- 6. 데이터 건수 확인
SELECT '=== 6. 테이블별 데이터 건수 ===' as section;
SELECT 'employees' as table_name, COUNT(*) as row_count FROM employees
UNION ALL
SELECT 'consultations', COUNT(*) FROM consultations
UNION ALL
SELECT 'customers', COUNT(*) FROM customers
UNION ALL
SELECT 'persona_types', COUNT(*) FROM persona_types
UNION ALL
SELECT 'consultation_documents', COUNT(*) FROM consultation_documents;

-- 7. 고객별 최근 상담 이력 조회 (샘플)
SELECT '=== 7. 고객별 최근 상담 이력 샘플 (상위 3명) ===' as section;
SELECT
    c.customer_id,
    c."category_raw",
    c.call_date,
    c.status,
    ROW_NUMBER() OVER (PARTITION BY c.customer_id ORDER BY c.call_date DESC) as recent_rank
FROM consultations c
WHERE c.customer_id IN (
    SELECT DISTINCT customer_id
    FROM consultations
    LIMIT 3
)
ORDER BY c.customer_id, c.call_date DESC;

-- 8. 상담사별 배정 현황
SELECT '=== 8. 상담사별 배정 현황 (통계) ===' as section;
SELECT
    COUNT(DISTINCT agent_id) as total_agents,
    COUNT(*) as total_consultations,
    ROUND(AVG(consultation_count), 2) as avg_per_agent,
    MIN(consultation_count) as min_count,
    MAX(consultation_count) as max_count,
    MAX(consultation_count) - MIN(consultation_count) as range_count
FROM (
    SELECT agent_id, COUNT(*) as consultation_count
    FROM consultations
    WHERE agent_id IS NOT NULL
    GROUP BY agent_id
) sub;

-- 9. 고객 유형 분포 (customers 데이터가 있는 경우)
SELECT '=== 9. 고객 유형 분포 ===' as section;
SELECT
    current_type_code,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM customers), 0), 1) as percentage
FROM customers
WHERE current_type_code IS NOT NULL
GROUP BY current_type_code
ORDER BY current_type_code;

-- 10. 카드 만료 예정 고객 (30일 이내)
SELECT '=== 10. 카드 만료 예정 고객 (30일 이내) ===' as section;
SELECT
    COUNT(*) as expiring_soon_count
FROM customers
WHERE card_expiry_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days';

-- 검증 완료
SELECT '=== 검증 완료 ===' as section;
