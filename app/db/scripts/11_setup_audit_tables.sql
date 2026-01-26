-- ============================================================
-- CALL:ACT 감사 로그 테이블 생성
-- 작성일: 2026-01-24
-- Phase: 12-1
-- ============================================================
-- 목적:
-- 1. 녹취 파일 다운로드 이력 기록 (법적 준수)
-- 2. 시스템 전체 감사 로그
-- ============================================================

-- ============================================================
-- 1. recording_download_logs (녹취 파일 다운로드 이력)
-- 목적: 녹취 파일 다운로드 시 자동 기록 (법적 의무 준수, 감사 추적)
-- ============================================================
CREATE TABLE IF NOT EXISTS recording_download_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    consultation_id VARCHAR(50) NOT NULL REFERENCES consultations(id),
    downloaded_by VARCHAR(50) NOT NULL REFERENCES employees(id),
    download_type VARCHAR(20) NOT NULL,        -- 'txt', 'wav', 'mp3'
    download_ip VARCHAR(45) NOT NULL,          -- IPv4/IPv6
    download_user_agent TEXT,                  -- 브라우저 정보
    file_name VARCHAR(500) NOT NULL,           -- 다운로드 파일명
    file_path VARCHAR(500),                    -- 서버 파일 경로 (S3 등)
    file_size BIGINT,                          -- 바이트 단위
    downloaded_at TIMESTAMP DEFAULT NOW() NOT NULL,

    -- 체크 제약
    CONSTRAINT chk_download_type CHECK (download_type IN ('txt', 'wav', 'mp3'))
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_download_logs_consultation
    ON recording_download_logs(consultation_id, downloaded_at DESC);

CREATE INDEX IF NOT EXISTS idx_download_logs_employee
    ON recording_download_logs(downloaded_by, downloaded_at DESC);

CREATE INDEX IF NOT EXISTS idx_download_logs_date
    ON recording_download_logs(downloaded_at DESC);

-- 코멘트
COMMENT ON TABLE recording_download_logs IS '녹취 파일 다운로드 이력 (법적 의무 준수, 5년 보관)';
COMMENT ON COLUMN recording_download_logs.consultation_id IS '다운로드한 상담 ID';
COMMENT ON COLUMN recording_download_logs.downloaded_by IS '다운로드한 상담사 ID';
COMMENT ON COLUMN recording_download_logs.download_type IS '다운로드 파일 유형 (txt/wav/mp3)';
COMMENT ON COLUMN recording_download_logs.download_ip IS '다운로드 시점 IP 주소 (감사용)';
COMMENT ON COLUMN recording_download_logs.downloaded_at IS '다운로드 시각 (UTC)';

-- ============================================================
-- 2. audit_logs (시스템 전체 감사 로그)
-- 목적: 시스템 전반의 중요 액션 기록
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(50) NOT NULL REFERENCES employees(id),
    action VARCHAR(100) NOT NULL,              -- 'RECORDING_DOWNLOAD', 'CUSTOMER_VIEW', 'DATA_EXPORT' 등
    resource_type VARCHAR(50) NOT NULL,        -- 'consultation', 'recording', 'customer' 등
    resource_id VARCHAR(100) NOT NULL,         -- 리소스 식별자
    ip_address VARCHAR(45) NOT NULL,
    user_agent TEXT,
    details JSONB,                             -- 추가 정보 (파일명, 크기 등)
    created_at TIMESTAMP DEFAULT NOW() NOT NULL
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_audit_logs_user
    ON audit_logs(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_logs_action
    ON audit_logs(action, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_logs_resource
    ON audit_logs(resource_type, resource_id);

CREATE INDEX IF NOT EXISTS idx_audit_logs_date
    ON audit_logs(created_at DESC);

-- 코멘트
COMMENT ON TABLE audit_logs IS '시스템 전체 감사 로그';
COMMENT ON COLUMN audit_logs.action IS '수행한 액션 (RECORDING_DOWNLOAD, CUSTOMER_VIEW 등)';
COMMENT ON COLUMN audit_logs.resource_type IS '리소스 유형 (consultation, recording, customer 등)';
COMMENT ON COLUMN audit_logs.details IS '액션별 추가 정보 (JSON)';

-- ============================================================
-- 3. 감사 로그 액션 유형 정의 (참조용)
-- ============================================================
/*
액션 유형 정의:
- RECORDING_DOWNLOAD: 녹취 파일 다운로드
- RECORDING_PLAY: 녹취 파일 재생
- CUSTOMER_VIEW: 고객 정보 조회
- CUSTOMER_EDIT: 고객 정보 수정
- CONSULTATION_VIEW: 상담 상세 조회
- CONSULTATION_EDIT: 상담 정보 수정
- DATA_EXPORT: 데이터 내보내기
- REPORT_GENERATE: 리포트 생성
- LOGIN: 로그인
- LOGOUT: 로그아웃
- PASSWORD_CHANGE: 비밀번호 변경
- PERMISSION_CHANGE: 권한 변경
*/

-- ============================================================
-- 4. 이상 다운로드 감지용 뷰
-- ============================================================
CREATE OR REPLACE VIEW v_suspicious_downloads AS
SELECT
    downloaded_by,
    e.name as employee_name,
    COUNT(*) as download_count,
    MAX(downloaded_at) as last_download,
    array_agg(DISTINCT consultation_id) as consultation_ids
FROM recording_download_logs rdl
LEFT JOIN employees e ON rdl.downloaded_by = e.id
WHERE downloaded_at >= NOW() - INTERVAL '1 hour'
GROUP BY downloaded_by, e.name
HAVING COUNT(*) >= 10;

COMMENT ON VIEW v_suspicious_downloads IS '1시간 내 10건 이상 다운로드한 사용자 (이상 행동 감지)';

-- ============================================================
-- 5. 일일 다운로드 통계 뷰
-- ============================================================
CREATE OR REPLACE VIEW v_daily_download_stats AS
SELECT
    DATE(downloaded_at) as download_date,
    COUNT(*) as total_downloads,
    COUNT(DISTINCT downloaded_by) as unique_users,
    COUNT(DISTINCT consultation_id) as unique_consultations
FROM recording_download_logs
WHERE downloaded_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(downloaded_at)
ORDER BY download_date DESC;

COMMENT ON VIEW v_daily_download_stats IS '최근 30일 일별 다운로드 통계';

-- ============================================================
-- 완료 메시지
-- ============================================================
DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE '감사 로그 테이블 생성 완료';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '생성된 테이블:';
    RAISE NOTICE '  - recording_download_logs (녹취 파일 다운로드 이력)';
    RAISE NOTICE '  - audit_logs (시스템 전체 감사 로그)';
    RAISE NOTICE '생성된 뷰:';
    RAISE NOTICE '  - v_suspicious_downloads (이상 다운로드 감지)';
    RAISE NOTICE '  - v_daily_download_stats (일별 다운로드 통계)';
    RAISE NOTICE '============================================================';
END $$;
