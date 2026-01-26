-- ============================================================
-- CALL:ACT 시뮬레이션 교육 시스템 테이블 생성
-- 작성일: 2026-01-23
-- Phase: 12
-- ============================================================

-- ============================================================
-- 1. simulation_scenarios (시나리오 템플릿)
-- 목적: 기본 시뮬레이션 시나리오 정의 (SIM-001, SIM-002, ...)
-- ============================================================
CREATE TABLE IF NOT EXISTS simulation_scenarios (
    id VARCHAR(50) PRIMARY KEY,                    -- 'SIM-001'
    title VARCHAR(200) NOT NULL,                   -- '카드 분실 신고 및 재발급'
    difficulty VARCHAR(20) NOT NULL,               -- '초급', '중급', '고급'
    category VARCHAR(100),                         -- '카드분실'
    description TEXT,
    duration_estimate INT,                         -- 예상 소요 시간 (초)

    -- 시나리오 구조
    scenario_objectives JSONB,                     -- 학습 목표
    scenario_steps JSONB,                          -- 단계별 가이드
    required_documents TEXT[],                     -- 필수 참조 문서 ID
    required_keywords TEXT[],                      -- 필수 키워드

    -- AI 고객 설정
    ai_customer_persona VARCHAR(50),               -- '급한성향', '꼼꼼한성향', ...
    ai_customer_name VARCHAR(100),                 -- '김민수'
    ai_customer_background TEXT,                   -- 고객 배경 정보

    -- AI 인터랙티브 대화 흐름
    ai_conversation_flow JSONB,

    -- 평가 기준
    evaluation_criteria JSONB,
    passing_score INT DEFAULT 70,                  -- 합격 점수

    -- 잠금 설정
    locked BOOLEAN DEFAULT FALSE,
    unlock_condition VARCHAR(500),                 -- "SIM-001 완료 후 해제"

    -- 태그
    tags TEXT[],                                   -- ['카드분실', '재발급', '기본상담']

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_simulation_scenarios_difficulty ON simulation_scenarios(difficulty);
CREATE INDEX IF NOT EXISTS idx_simulation_scenarios_category ON simulation_scenarios(category);
CREATE INDEX IF NOT EXISTS idx_simulation_scenarios_locked ON simulation_scenarios(locked);

COMMENT ON TABLE simulation_scenarios IS '시뮬레이션 교육용 시나리오 템플릿';
COMMENT ON COLUMN simulation_scenarios.id IS '시나리오 ID (SIM-001 형식)';
COMMENT ON COLUMN simulation_scenarios.difficulty IS '난이도 (초급/중급/고급)';
COMMENT ON COLUMN simulation_scenarios.scenario_objectives IS '학습 목표 배열';
COMMENT ON COLUMN simulation_scenarios.scenario_steps IS '단계별 가이드 (step, action, documents)';
COMMENT ON COLUMN simulation_scenarios.ai_conversation_flow IS 'AI 고객 대화 흐름 (initial, responses, triggers)';
COMMENT ON COLUMN simulation_scenarios.evaluation_criteria IS '평가 기준 (document_usage, keyword_coverage 등)';

-- ============================================================
-- 2. simulation_results (시뮬레이션 결과)
-- 목적: 각 시뮬레이션 수행 결과 저장
-- ============================================================
CREATE TABLE IF NOT EXISTS simulation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    employee_id VARCHAR(50) REFERENCES employees(id),

    -- 시뮬레이션 유형
    simulation_type VARCHAR(20) NOT NULL,          -- 'best_practice' or 'scenario'

    -- 우수사례 시뮬레이션
    original_consultation_id VARCHAR(50) REFERENCES consultations(id),  -- is_best_practice=true

    -- 기본 시나리오 시뮬레이션
    scenario_id VARCHAR(50) REFERENCES simulation_scenarios(id),

    -- 생성된 시뮬레이션 상담 ID
    simulation_consultation_id VARCHAR(50) UNIQUE NOT NULL,

    -- 결과
    overall_score INT,                             -- 종합 점수
    passed BOOLEAN,                                -- 합격 여부

    -- 우수사례용 평가
    similarity_score INT,                          -- 우수사례 유사도 (0-100)
    keyword_match_score INT,
    document_match_score INT,
    sequence_match_score INT,
    time_comparison JSONB,                         -- 시간 비교

    -- 기본 시나리오용 평가
    objective_completion_rate INT,                 -- 목표 달성률 (0-100)
    document_usage_score INT,
    keyword_coverage_score INT,
    sequence_correctness_score INT,
    customer_satisfaction_score INT,

    -- 상세 피드백
    feedback_data JSONB,

    -- AI 고객 반응 로그 (기본 시나리오만)
    ai_customer_reactions JSONB,

    -- 통화 정보
    call_duration INT,                             -- 통화 시간 (초)
    call_started_at TIMESTAMP,
    call_ended_at TIMESTAMP,

    -- 녹음 파일
    recording_file_path VARCHAR(500),
    recording_transcript TEXT,                     -- 마스킹된 녹취록

    created_at TIMESTAMP DEFAULT NOW(),

    -- 체크 제약
    CONSTRAINT chk_simulation_type CHECK (simulation_type IN ('best_practice', 'scenario'))
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_simulation_results_employee ON simulation_results(employee_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_simulation_results_type ON simulation_results(simulation_type);
CREATE INDEX IF NOT EXISTS idx_simulation_results_passed ON simulation_results(passed);
CREATE INDEX IF NOT EXISTS idx_simulation_results_score ON simulation_results(overall_score DESC);

COMMENT ON TABLE simulation_results IS '시뮬레이션 수행 결과';
COMMENT ON COLUMN simulation_results.simulation_type IS '시뮬레이션 유형 (best_practice/scenario)';
COMMENT ON COLUMN simulation_results.original_consultation_id IS '우수사례 원본 상담 ID (best_practice 유형)';
COMMENT ON COLUMN simulation_results.scenario_id IS '시나리오 ID (scenario 유형)';
COMMENT ON COLUMN simulation_results.similarity_score IS '우수사례 유사도 점수 (0-100)';
COMMENT ON COLUMN simulation_results.time_comparison IS '시간 비교 (original_duration, simulation_duration 등)';
COMMENT ON COLUMN simulation_results.feedback_data IS '상세 피드백 (strengths, improvements, expert_tips)';
COMMENT ON COLUMN simulation_results.ai_customer_reactions IS 'AI 고객 반응 로그';

-- ============================================================
-- 3. employee_learning_analytics (학습 분석)
-- 목적: 개인별 학습 성과 및 분석 데이터
-- ============================================================
CREATE TABLE IF NOT EXISTS employee_learning_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    employee_id VARCHAR(50) REFERENCES employees(id) UNIQUE,

    -- 전체 통계
    total_simulations INT DEFAULT 0,
    total_best_practice_simulations INT DEFAULT 0,
    total_scenario_simulations INT DEFAULT 0,

    -- 평균 점수
    average_score DECIMAL(5,2),
    average_similarity_score DECIMAL(5,2),         -- 우수사례 유사도
    average_objective_completion DECIMAL(5,2),     -- 시나리오 목표 달성률

    -- 합격률
    pass_rate DECIMAL(5,2),
    best_practice_pass_rate DECIMAL(5,2),
    scenario_pass_rate DECIMAL(5,2),

    -- 개선율 (최근 vs 초기)
    improvement_rate DECIMAL(5,2),

    -- 강점/약점
    strengths JSONB,
    weaknesses JSONB,

    -- 카테고리별 성과
    category_performance JSONB,

    -- 완료한 시나리오
    completed_scenarios TEXT[],                    -- ['SIM-001', 'SIM-002']
    unlocked_scenarios TEXT[],                     -- ['SIM-001', 'SIM-002', 'SIM-003']

    -- 학습 시간
    total_learning_time_seconds INT DEFAULT 0,

    -- 최근 활동
    last_simulation_at TIMESTAMP,
    last_simulation_score INT,

    updated_at TIMESTAMP DEFAULT NOW()
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_learning_analytics_employee ON employee_learning_analytics(employee_id);
CREATE INDEX IF NOT EXISTS idx_learning_analytics_score ON employee_learning_analytics(average_score DESC);
CREATE INDEX IF NOT EXISTS idx_learning_analytics_improvement ON employee_learning_analytics(improvement_rate DESC);

COMMENT ON TABLE employee_learning_analytics IS '상담사별 학습 분석 데이터';
COMMENT ON COLUMN employee_learning_analytics.improvement_rate IS '개선율 (최근 5회 - 초기 5회 평균)';
COMMENT ON COLUMN employee_learning_analytics.strengths IS '강점 항목 [{skill, score}]';
COMMENT ON COLUMN employee_learning_analytics.weaknesses IS '약점 항목 [{skill, score}]';
COMMENT ON COLUMN employee_learning_analytics.category_performance IS '카테고리별 성과 {카테고리: {attempts, avg_score, pass_rate, best_score}}';
COMMENT ON COLUMN employee_learning_analytics.completed_scenarios IS '완료한 시나리오 ID 배열';
COMMENT ON COLUMN employee_learning_analytics.unlocked_scenarios IS '잠금 해제된 시나리오 ID 배열';

-- ============================================================
-- 샘플 데이터 삽입
-- ============================================================
INSERT INTO simulation_scenarios (
    id, title, difficulty, category, description, duration_estimate,
    scenario_objectives, scenario_steps, required_documents, required_keywords,
    ai_customer_persona, ai_customer_name, ai_customer_background,
    ai_conversation_flow, evaluation_criteria, passing_score, locked, tags
) VALUES
(
    'SIM-001',
    '카드 분실 신고 및 재발급',
    '초급',
    '카드분실',
    '고객의 카드 분실 신고를 접수하고 재발급 절차를 안내하는 시나리오',
    300,
    '["카드 즉시 정지 처리", "재발급 신청 완료", "배송 정보 안내"]'::jsonb,
    '[
        {"step": 1, "action": "고객 본인 확인", "documents": ["card-1-1-1"]},
        {"step": 2, "action": "카드 즉시 정지", "documents": ["card-1-1-2"]},
        {"step": 3, "action": "재발급 신청", "documents": ["card-1-1-3"]}
    ]'::jsonb,
    ARRAY['card-1-1-1', 'card-1-1-2', 'card-1-1-3'],
    ARRAY['본인확인', '즉시정지', '재발급', '배송'],
    '급한성향',
    '김민수',
    '어제 지갑을 분실하여 카드도 함께 없어짐. 부정 사용이 걱정되어 빠른 처리 원함.',
    '{
        "initial": {
            "message": "안녕하세요, 카드를 분실했어요! 빨리 정지시켜주세요!",
            "emotion": "urgent",
            "tts_settings": {"pitch": 1.1, "rate": 1.2}
        },
        "responses": {
            "empathy_high": {
                "trigger": ["걱정", "안심", "빠르게"],
                "message": "네, 빨리 처리해주셔서 감사합니다.",
                "emotion": "calm"
            },
            "empathy_low": {
                "trigger": ["기다리", "확인", "시간"],
                "message": "왜 이렇게 오래 걸리는 거예요?",
                "emotion": "angry"
            },
            "verification_request": {
                "trigger": ["본인확인", "이름", "생년월일"],
                "message": "네, 김민수입니다. 1990년 5월 15일생이에요.",
                "emotion": "neutral"
            }
        },
        "triggers": {
            "card_stopped": {
                "condition": "카드 정지 처리 완료",
                "message": "카드가 정지되었나요? 확인 부탁드려요.",
                "emotion": "anxious"
            },
            "reissue_applied": {
                "condition": "재발급 신청 완료",
                "message": "새 카드는 언제 받을 수 있나요?",
                "emotion": "curious"
            }
        }
    }'::jsonb,
    '{
        "document_usage": {
            "weight": 30,
            "required": ["card-1-1-1", "card-1-1-2", "card-1-1-3"],
            "description": "필수 문서 참조 여부"
        },
        "keyword_coverage": {
            "weight": 25,
            "required": ["본인확인", "즉시정지", "재발급", "배송"],
            "description": "필수 키워드 언급 여부"
        },
        "sequence_correctness": {
            "weight": 25,
            "expected_sequence": ["본인확인", "카드정지", "재발급신청"],
            "description": "처리 순서 정확성"
        },
        "customer_satisfaction": {
            "weight": 20,
            "factors": ["공감표현", "신속처리", "명확한안내"],
            "description": "고객 만족도"
        }
    }'::jsonb,
    70,
    FALSE,
    ARRAY['카드분실', '재발급', '기본상담']
),
(
    'SIM-002',
    '해외 결제 차단 해제 요청',
    '중급',
    '해외결제',
    '해외 여행 중 카드 결제가 차단된 고객의 문의를 처리하는 시나리오',
    420,
    '["해외 결제 차단 사유 확인", "차단 해제 처리", "사용 가능 국가 안내"]'::jsonb,
    '[
        {"step": 1, "action": "고객 본인 확인 및 현재 위치 확인", "documents": ["overseas-1-1"]},
        {"step": 2, "action": "해외 결제 차단 해제", "documents": ["overseas-1-2"]},
        {"step": 3, "action": "사용 가능 국가 및 주의사항 안내", "documents": ["overseas-1-3"]}
    ]'::jsonb,
    ARRAY['overseas-1-1', 'overseas-1-2', 'overseas-1-3'],
    ARRAY['본인확인', '현재위치', '차단해제', '사용가능국가'],
    '급한성향',
    '박철수',
    '일본 여행 중 카드가 결제되지 않음. 긴급히 해결 필요.',
    '{
        "initial": {
            "message": "지금 일본인데 카드가 안 돼요! 급해요!",
            "emotion": "urgent",
            "tts_settings": {"pitch": 1.1, "rate": 1.3}
        },
        "responses": {
            "empathy_high": {
                "trigger": ["도와드리", "해결", "바로"],
                "message": "네, 감사합니다. 빨리 부탁드려요.",
                "emotion": "hopeful"
            },
            "location_confirm": {
                "trigger": ["현재위치", "어디", "일본"],
                "message": "네, 지금 도쿄 시부야에 있어요.",
                "emotion": "neutral"
            }
        },
        "triggers": {
            "unblock_done": {
                "condition": "차단 해제 완료",
                "message": "해제됐어요? 바로 써볼게요!",
                "emotion": "relieved"
            }
        }
    }'::jsonb,
    '{
        "document_usage": {"weight": 30},
        "keyword_coverage": {"weight": 25},
        "sequence_correctness": {"weight": 25},
        "customer_satisfaction": {"weight": 20}
    }'::jsonb,
    70,
    FALSE,
    ARRAY['해외결제', '차단해제', '긴급처리']
),
(
    'SIM-003',
    '포인트 적립 누락 문의',
    '초급',
    '포인트',
    '결제 후 포인트가 적립되지 않은 고객의 문의를 처리하는 시나리오',
    240,
    '["결제 내역 확인", "포인트 적립 조건 안내", "수동 적립 처리"]'::jsonb,
    '[
        {"step": 1, "action": "고객 본인 확인", "documents": ["point-1-1"]},
        {"step": 2, "action": "결제 내역 및 포인트 확인", "documents": ["point-1-2"]},
        {"step": 3, "action": "적립 조건 안내 및 처리", "documents": ["point-1-3"]}
    ]'::jsonb,
    ARRAY['point-1-1', 'point-1-2', 'point-1-3'],
    ARRAY['본인확인', '결제내역', '포인트적립', '적립조건'],
    '꼼꼼한성향',
    '이영희',
    '어제 10만원 결제했는데 포인트가 안 쌓였음. 적립 조건 상세 확인 원함.',
    '{
        "initial": {
            "message": "안녕하세요, 어제 결제했는데 포인트가 안 쌓여 있어요. 확인 부탁드려요.",
            "emotion": "calm",
            "tts_settings": {"pitch": 1.0, "rate": 1.0}
        },
        "responses": {
            "detail_request": {
                "trigger": ["확인", "조회"],
                "message": "네, 어제 오후 3시쯤 마트에서 10만 3천원 결제했어요.",
                "emotion": "neutral"
            },
            "condition_question": {
                "trigger": ["적립", "조건", "제외"],
                "message": "그럼 어떤 결제가 포인트 적립 대상인지 자세히 알려주실 수 있나요?",
                "emotion": "curious"
            }
        }
    }'::jsonb,
    '{
        "document_usage": {"weight": 30},
        "keyword_coverage": {"weight": 25},
        "sequence_correctness": {"weight": 25},
        "customer_satisfaction": {"weight": 20}
    }'::jsonb,
    70,
    FALSE,
    ARRAY['포인트', '적립', '기본상담']
),
(
    'SIM-004',
    '분할결제 취소 요청',
    '중급',
    '결제취소',
    '할부 결제를 취소하려는 고객의 요청을 처리하는 시나리오',
    360,
    '["결제 내역 확인", "할부 취소 가능 여부 확인", "취소 처리 및 안내"]'::jsonb,
    '[
        {"step": 1, "action": "고객 본인 확인", "documents": ["cancel-1-1"]},
        {"step": 2, "action": "할부 결제 내역 확인", "documents": ["cancel-1-2"]},
        {"step": 3, "action": "취소 가능 여부 및 수수료 안내", "documents": ["cancel-1-3"]},
        {"step": 4, "action": "취소 처리", "documents": ["cancel-1-4"]}
    ]'::jsonb,
    ARRAY['cancel-1-1', 'cancel-1-2', 'cancel-1-3', 'cancel-1-4'],
    ARRAY['본인확인', '할부', '취소', '수수료', '환불'],
    '실용주의',
    '정대현',
    '3개월 할부로 50만원 결제했는데 다른 곳에서 더 저렴하게 구매 가능해서 취소하려 함.',
    '{
        "initial": {
            "message": "안녕하세요, 할부 결제한 거 취소하고 싶어요.",
            "emotion": "neutral",
            "tts_settings": {"pitch": 1.0, "rate": 1.1}
        },
        "responses": {
            "detail_provide": {
                "trigger": ["내역", "확인", "언제"],
                "message": "1월 15일에 온라인몰에서 50만원 3개월 할부로 결제했어요.",
                "emotion": "neutral"
            },
            "fee_concern": {
                "trigger": ["수수료", "비용"],
                "message": "취소 수수료가 있나요? 얼마인가요?",
                "emotion": "concerned"
            }
        }
    }'::jsonb,
    '{
        "document_usage": {"weight": 30},
        "keyword_coverage": {"weight": 25},
        "sequence_correctness": {"weight": 25},
        "customer_satisfaction": {"weight": 20}
    }'::jsonb,
    70,
    FALSE,
    ARRAY['결제취소', '할부', '환불']
),
(
    'SIM-005',
    '이중결제 환불 요청',
    '고급',
    '이중결제',
    '동일 결제가 중복 승인된 고객의 환불 요청을 처리하는 시나리오',
    480,
    '["이중결제 내역 확인", "가맹점 확인", "환불 처리 안내"]'::jsonb,
    '[
        {"step": 1, "action": "고객 본인 확인", "documents": ["refund-1-1"]},
        {"step": 2, "action": "이중결제 내역 상세 확인", "documents": ["refund-1-2"]},
        {"step": 3, "action": "가맹점 연락 및 환불 절차 안내", "documents": ["refund-1-3"]},
        {"step": 4, "action": "이의제기 접수 (필요시)", "documents": ["refund-1-4"]}
    ]'::jsonb,
    ARRAY['refund-1-1', 'refund-1-2', 'refund-1-3', 'refund-1-4'],
    ARRAY['본인확인', '이중결제', '환불', '가맹점', '이의제기'],
    '불만성향',
    '최민철',
    '같은 금액이 2번 결제되어 화가 난 상태. 빠른 환불 처리 요구.',
    '{
        "initial": {
            "message": "같은 금액이 두 번 빠져나갔어요! 왜 이런 일이 생기는 거예요?",
            "emotion": "angry",
            "tts_settings": {"pitch": 1.2, "rate": 1.3}
        },
        "responses": {
            "calm_down": {
                "trigger": ["죄송", "확인", "해결"],
                "message": "빨리 해결해주세요. 언제 환불받을 수 있어요?",
                "emotion": "impatient"
            },
            "escalate": {
                "trigger": ["기다리", "시간", "절차"],
                "message": "왜 이렇게 복잡해요? 그냥 바로 환불해주면 안 되나요?",
                "emotion": "frustrated"
            }
        }
    }'::jsonb,
    '{
        "document_usage": {"weight": 25},
        "keyword_coverage": {"weight": 20},
        "sequence_correctness": {"weight": 25},
        "customer_satisfaction": {"weight": 30}
    }'::jsonb,
    75,
    TRUE,
    ARRAY['이중결제', '환불', '민원', '고급']
)
ON CONFLICT (id) DO UPDATE SET
    title = EXCLUDED.title,
    difficulty = EXCLUDED.difficulty,
    category = EXCLUDED.category,
    description = EXCLUDED.description,
    duration_estimate = EXCLUDED.duration_estimate,
    scenario_objectives = EXCLUDED.scenario_objectives,
    scenario_steps = EXCLUDED.scenario_steps,
    required_documents = EXCLUDED.required_documents,
    required_keywords = EXCLUDED.required_keywords,
    ai_customer_persona = EXCLUDED.ai_customer_persona,
    ai_customer_name = EXCLUDED.ai_customer_name,
    ai_customer_background = EXCLUDED.ai_customer_background,
    ai_conversation_flow = EXCLUDED.ai_conversation_flow,
    evaluation_criteria = EXCLUDED.evaluation_criteria,
    passing_score = EXCLUDED.passing_score,
    locked = EXCLUDED.locked,
    tags = EXCLUDED.tags,
    updated_at = NOW();

-- ============================================================
-- 완료 메시지
-- ============================================================
DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE '시뮬레이션 테이블 생성 완료';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '생성된 테이블:';
    RAISE NOTICE '  - simulation_scenarios (시나리오 템플릿)';
    RAISE NOTICE '  - simulation_results (시뮬레이션 결과)';
    RAISE NOTICE '  - employee_learning_analytics (학습 분석)';
    RAISE NOTICE '============================================================';
END $$;
