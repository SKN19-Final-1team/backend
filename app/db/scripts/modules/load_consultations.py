"""
하나카드 상담 데이터 적재 모듈

상담 데이터 적재, 카테고리 매핑, 상담사 배분, FCR 계산,
고객-상담 매칭, 타임라인 생성, consultation_documents 적재
"""

import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, date, timedelta, time as time_type
from typing import Dict, List, Optional, Tuple

from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, Json as PsycopgJson

from . import (
    BATCH_SIZE, COMMIT_INTERVAL,
    HANA_RDB_METADATA_FILE, HANA_VECTORDB_FILE,
    CATEGORY_MAPPINGS, CATEGORY_TO_MAIN_CATEGORY,
    NORMALIZATION_RULES, MASKING_REPLACEMENTS,
    MAIN_CATEGORY_POOL_SIZES,
    check_table_has_data
)


# ==============================================================================
# 카테고리 매핑 함수
# ==============================================================================

def normalize_category_raw(category: str) -> str:
    """
    원본 카테고리 정규화: 띄어쓰기 통일, 순서 통일, 마스킹 태그 치환

    Args:
        category: 원본 카테고리 (예: "가상 계좌 안내", "도난/분실 신청/해제")

    Returns:
        정규화된 카테고리 (예: "가상계좌 안내", "분실/도난 신청/해제")
    """
    if not category:
        return ""

    result = category.strip()

    # 마스킹 태그 치환
    for pattern, replacement in MASKING_REPLACEMENTS.items():
        result = result.replace(pattern, replacement)

    # 띄어쓰기/순서 정규화
    for pattern, replacement in NORMALIZATION_RULES.items():
        result = result.replace(pattern, replacement)

    return result


def map_to_category(category: str) -> Tuple[str, str, str]:
    """
    원본 카테고리를 (대분류, 중분류, 정규화된 원본)으로 매핑

    Args:
        category: 원본 카테고리 (예: "선결제/즉시출금")

    Returns:
        (대분류, 중분류, 정규화된 원본) 튜플
        예: ("결제/승인", "즉시출금", "선결제/즉시출금")
    """
    # 1. 정규화
    normalized = normalize_category_raw(category)

    # 2. 정확한 매칭 시도
    if normalized in CATEGORY_MAPPINGS:
        main, sub = CATEGORY_MAPPINGS[normalized]
        return main, sub, normalized

    # 원본 그대로도 시도
    if category in CATEGORY_MAPPINGS:
        main, sub = CATEGORY_MAPPINGS[category]
        return main, sub, normalized

    # 3. 패턴 기반 매칭 (키워드)
    category_lower = normalized.lower()

    # 대분류 추론
    if any(kw in category_lower for kw in ["분실", "도난"]):
        main_cat = "분실/도난"
    elif any(kw in category_lower for kw in ["한도"]):
        main_cat = "한도"
    elif any(kw in category_lower for kw in ["결제", "승인", "매출", "즉시출금", "선결제", "가상계좌"]):
        main_cat = "결제/승인"
    elif any(kw in category_lower for kw in ["이용내역"]):
        main_cat = "이용내역"
    elif any(kw in category_lower for kw in ["연체", "수수료", "연회비"]):
        main_cat = "수수료/연체"
    elif any(kw in category_lower for kw in ["포인트", "혜택", "이벤트", "마일리지"]):
        main_cat = "포인트/혜택"
    elif any(kw in category_lower for kw in ["바우처", "정부지원"]):
        main_cat = "정부지원"
    else:
        main_cat = "기타"

    # 중분류 추론 (행위 기반)
    if any(kw in category_lower for kw in ["안내", "조회"]):
        sub_cat = "조회/안내"
    elif any(kw in category_lower for kw in ["신청", "등록"]):
        sub_cat = "신청/등록"
    elif any(kw in category_lower for kw in ["변경"]):
        sub_cat = "변경"
    elif any(kw in category_lower for kw in ["취소", "해지"]):
        sub_cat = "취소/해지"
    elif any(kw in category_lower for kw in ["처리", "실행"]):
        sub_cat = "처리/실행"
    elif any(kw in category_lower for kw in ["발급"]):
        sub_cat = "발급"
    elif any(kw in category_lower for kw in ["확인서", "증명서", "소득공제"]):
        sub_cat = "확인서"
    elif any(kw in category_lower for kw in ["배송", "재발송", "수령지"]):
        sub_cat = "배송"
    elif any(kw in category_lower for kw in ["즉시출금"]):
        sub_cat = "즉시출금"
    elif any(kw in category_lower for kw in ["상향", "증액", "임시한도", "특별한도"]):
        sub_cat = "상향/증액"
    elif any(kw in category_lower for kw in ["전환", "이체"]):
        sub_cat = "이체/전환"
    elif any(kw in category_lower for kw in ["환급", "반환"]):
        sub_cat = "환급/반환"
    elif any(kw in category_lower for kw in ["정지", "해제", "신고"]):
        sub_cat = "정지/해제"
    elif any(kw in category_lower for kw in ["결제일"]):
        sub_cat = "결제일"
    else:
        sub_cat = "기타"

    return main_cat, sub_cat, normalized


def map_to_main_category(category: str) -> str:
    """
    [호환성 유지] 세부 카테고리를 대분류로 매핑

    Note: 새 코드에서는 map_to_category() 사용 권장

    Args:
        category: 세부 카테고리 (예: "선결제/즉시출금")

    Returns:
        대분류 (예: "결제/승인")
    """
    main_cat, _, _ = map_to_category(category)
    return main_cat


# ==============================================================================
# 상담사 배분 함수
# ==============================================================================

def get_agent_pool_by_main_category(conn: psycopg2_connection, main_category: str,
                                     used_agents_by_category: Dict[str, set] = None) -> List[str]:
    """
    대분류별 상담사 풀 생성 (중복 최소화)

    Args:
        conn: DB 연결
        main_category: 대분류 (예: "결제/승인")
        used_agents_by_category: 다른 대분류에서 이미 사용된 상담사 집합 (중복 방지용)

    Returns:
        상담사 ID 리스트 (예: ["EMP-001", "EMP-002", ...])
    """
    cursor = conn.cursor()

    try:
        # EMP-TEDDY-DEFAULT 제외, active 상태만
        if main_category == "기타":
            # "기타"는 모든 상담사에게 순차 분배
            cursor.execute("""
                SELECT id FROM employees
                WHERE id != 'EMP-TEDDY-DEFAULT'
                AND status = 'active'
                AND department ~ '^상담'
                ORDER BY created_at ASC
            """)
        else:
            # 대분류별 풀 크기 결정
            pool_size = MAIN_CATEGORY_POOL_SIZES.get(main_category, 5)

            # 다른 대분류에서 사용된 상담사 제외 (중복 최소화)
            if used_agents_by_category:
                other_used_agents = set()
                for cat, agents in used_agents_by_category.items():
                    if cat != main_category and cat != "기타":
                        other_used_agents.update(agents)

                # 사용되지 않은 상담사 우선 선택
                if other_used_agents:
                    placeholders = ','.join(['%s'] * len(other_used_agents))
                    cursor.execute(f"""
                        SELECT id FROM employees
                        WHERE id != 'EMP-TEDDY-DEFAULT'
                        AND status = 'active'
                        AND department ~ '^상담'
                        AND id NOT IN ({placeholders})
                        ORDER BY created_at ASC
                        LIMIT %s
                    """, list(other_used_agents) + [pool_size])
                else:
                    cursor.execute("""
                        SELECT id FROM employees
                        WHERE id != 'EMP-TEDDY-DEFAULT'
                        AND status = 'active'
                        AND department ~ '^상담'
                        ORDER BY created_at ASC
                        LIMIT %s
                    """, (pool_size,))
            else:
                cursor.execute("""
                    SELECT id FROM employees
                    WHERE id != 'EMP-TEDDY-DEFAULT'
                    AND status = 'active'
                    AND department ~ '^상담'
                    ORDER BY created_at ASC
                    LIMIT %s
                """, (pool_size,))

        agent_ids = [row[0] for row in cursor.fetchall()]

        if not agent_ids:
            raise ValueError(f"[ERROR] 상담사 풀이 비어있습니다. (대분류: {main_category})")

        return agent_ids

    finally:
        cursor.close()


def convert_to_new_consultation_id(
    old_id: str,
    agent_id: str,
    call_start_time: Optional[str] = None,
    created_at: Optional[str] = None
) -> str:
    """
    기존 상담 ID를 새 형식으로 변환

    Args:
        old_id: 기존 ID (예: "hana_consultation_20593")
        agent_id: 상담사 ID (예: "EMP-001")
        call_start_time: 통화 시작 시간 (ISO 형식)
        created_at: 생성 시간 (ISO 형식, fallback)

    Returns:
        새 형식 ID (예: "CS-EMP001-202601211430")
    """
    # 1. agent_id에서 대시 제거
    clean_agent_id = agent_id.replace("-", "")

    # 2. 날짜/시간 추출
    dt = None
    if call_start_time:
        try:
            dt = datetime.fromisoformat(call_start_time.replace("Z", "+00:00"))
        except:
            pass

    # fallback: created_at 사용
    if not dt and created_at:
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except:
            pass

    # fallback: 고정 시간 사용 (재현성 보장)
    if not dt:
        dt = datetime(2026, 1, 20, 9, 0, 0)

    # 3. 새 형식 ID 생성 (YYYYMMDDHHMM)
    timestamp_str = dt.strftime("%Y%m%d%H%M")
    return f"CS-{clean_agent_id}-{timestamp_str}"


# ==============================================================================
# 평일 영업일 (주말 제외)
# ==============================================================================

BUSINESS_DAYS = [
    date(2026, 1, 19),  # 월
    date(2026, 1, 20),  # 화
    date(2026, 1, 21),  # 수
    date(2026, 1, 22),  # 목
    date(2026, 1, 23),  # 금
]

# ==============================================================================
# 카테고리별 전형적 처리 단계
# ==============================================================================

CATEGORY_TYPICAL_STEPS = {
    "분실/도난": [
        {"action": "고객 본인 확인", "category": None},
        {"action": "카드 분실 신고 접수", "category": "카드분실신고"},
        {"action": "카드 사용 즉시 정지 처리", "category": "카드정지/해제"},
        {"action": "재발급 카드 신청 안내", "category": "카드재발급"},
        {"action": "상담 종료 및 확인", "category": None},
    ],
    "결제/승인": [
        {"action": "고객 본인 확인", "category": None},
        {"action": "결제 내역 조회 및 확인", "category": "결제조회"},
        {"action": "요청 사항 처리", "category": None},
        {"action": "처리 결과 안내", "category": None},
        {"action": "상담 종료 및 확인", "category": None},
    ],
    "이용내역": [
        {"action": "고객 본인 확인", "category": None},
        {"action": "이용 내역 조회", "category": "이용내역조회"},
        {"action": "내역 상세 안내", "category": None},
        {"action": "상담 종료 및 확인", "category": None},
    ],
    "한도": [
        {"action": "고객 본인 확인", "category": None},
        {"action": "현재 한도 조회", "category": "한도조회"},
        {"action": "한도 변경 처리", "category": "한도변경"},
        {"action": "변경 결과 안내", "category": None},
        {"action": "상담 종료 및 확인", "category": None},
    ],
    "수수료/연체": [
        {"action": "고객 본인 확인", "category": None},
        {"action": "연체/수수료 내역 조회", "category": "연체조회"},
        {"action": "납부 방법 안내", "category": None},
        {"action": "상담 종료 및 확인", "category": None},
    ],
    "포인트/혜택": [
        {"action": "고객 본인 확인", "category": None},
        {"action": "포인트 잔액 조회", "category": "포인트조회"},
        {"action": "혜택 안내 및 처리", "category": None},
        {"action": "상담 종료 및 확인", "category": None},
    ],
    "정부지원": [
        {"action": "고객 본인 확인", "category": None},
        {"action": "지원 대상 확인", "category": "자격확인"},
        {"action": "신청 절차 안내", "category": None},
        {"action": "상담 종료 및 확인", "category": None},
    ],
    "기타": [
        {"action": "고객 본인 확인", "category": None},
        {"action": "문의 내용 확인", "category": None},
        {"action": "처리 및 안내", "category": None},
        {"action": "상담 종료 및 확인", "category": None},
    ],
}

# 관련 카테고리 매핑 (복합 업무 시 추가될 수 있는 관련 업무)
RELATED_CATEGORIES = {
    "분실/도난": ["카드재발급", "긴급 배송 신청", "이용내역 안내"],
    "결제/승인": ["이용내역 안내", "한도 안내", "결제일 안내/변경"],
    "이용내역": ["결제대금 안내", "포인트/마일리지 안내"],
    "한도": ["결제대금 안내", "한도상향 접수/처리"],
    "수수료/연체": ["결제대금 안내", "연체대금 즉시출금"],
    "포인트/혜택": ["이벤트 안내", "포인트/마일리지 전환등록"],
    "정부지원": ["이용방법 안내"],
    "기타": ["이용방법 안내", "서비스 이용방법 안내"],
}


# ==============================================================================
# 헬퍼 함수
# ==============================================================================

def distribute_consultations_to_customers(num_customers: int, num_consultations: int, rng) -> List[int]:
    """고객별 상담 건수 분포 생성

    Returns:
        각 고객의 상담 건수 리스트 (합계 = num_consultations)
    """
    # 분포: 1건 40%, 2-3건 35%, 4-6건 18%, 7+건 7%
    counts = []
    # 1건 고객: 40%
    n1 = int(num_customers * 0.40)
    counts.extend([1] * n1)
    # 2-3건 고객: 35%
    n2 = int(num_customers * 0.35)
    counts.extend([rng.randint(2, 3) for _ in range(n2)])
    # 4-6건 고객: 18%
    n3 = int(num_customers * 0.18)
    counts.extend([rng.randint(4, 6) for _ in range(n3)])
    # 7+건 고객: 나머지
    n4 = num_customers - n1 - n2 - n3
    counts.extend([rng.randint(7, 12) for _ in range(n4)])

    # 합계 조정
    current_total = sum(counts)
    diff = num_consultations - current_total

    # 부족하면 중간 고객들에게 추가
    idx = n1  # 2-3건 고객부터 시작
    while diff > 0 and idx < len(counts):
        add = min(diff, 2)
        counts[idx] += add
        diff -= add
        idx += 1
        if idx >= len(counts):
            idx = n1

    # 초과하면 큰 건수 고객에서 감소
    idx = len(counts) - 1
    while diff < 0 and idx >= 0:
        reduce = min(-diff, max(0, counts[idx] - 1))
        counts[idx] -= reduce
        diff += reduce
        idx -= 1

    rng.shuffle(counts)
    return counts


def generate_processing_timeline(call_time_obj, call_duration_str: Optional[str],
                                  main_category: str, category_raw: str) -> List[Dict]:
    """상담 처리 단계별 타임라인 합성 생성"""
    steps = CATEGORY_TYPICAL_STEPS.get(main_category, CATEGORY_TYPICAL_STEPS["기타"])

    # call_duration 파싱 (초)
    duration_seconds = 0
    if call_duration_str:
        parts = call_duration_str.split(':')
        try:
            if len(parts) == 2:
                duration_seconds = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                duration_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except (ValueError, IndexError):
            duration_seconds = 300  # 기본 5분

    if duration_seconds <= 0:
        duration_seconds = 300

    interval = duration_seconds / len(steps)
    timeline = []

    base_seconds = call_time_obj.hour * 3600 + call_time_obj.minute * 60 + call_time_obj.second

    for i, step in enumerate(steps):
        step_seconds = base_seconds + int(i * interval)
        h = (step_seconds // 3600) % 24
        m = (step_seconds % 3600) // 60
        s = step_seconds % 60
        step_category = step.get("category")
        # 첫 번째 실제 업무 단계에 category_raw 적용
        if step_category is None and i > 0 and i < len(steps) - 1:
            step_category = category_raw

        timeline.append({
            "time": f"{h:02d}:{m:02d}:{s:02d}",
            "action": step["action"],
            "category": step_category,
        })

    return timeline


# ==============================================================================
# 메인 적재 함수
# ==============================================================================

def load_hana_data(conn: psycopg2_connection):
    """하나카드 데이터 적재"""
    print("\n" + "=" * 60)
    print("[5/12] 하나카드 데이터 적재")
    print("=" * 60)

    # 이미 데이터가 있는지 확인
    has_data, count = check_table_has_data(conn, "consultations")
    if has_data:
        print(f"[INFO] consultations 테이블에 이미 데이터가 있습니다. (건수: {count}건) - 적재 스킵")
        return True

    # 03_load_hana_to_db.py의 로직 통합
    # consultations 적재
    if not HANA_RDB_METADATA_FILE.exists():
        print(f"[ERROR] 하나카드 RDB 메타데이터 파일을 찾을 수 없습니다: {HANA_RDB_METADATA_FILE}")
        return False

    print(f"[INFO] 하나카드 RDB 메타데이터 파일 로드: {HANA_RDB_METADATA_FILE}")

    with open(HANA_RDB_METADATA_FILE, 'r', encoding='utf-8') as f:
        consultations_data = json.load(f)

    print(f"[INFO] 총 상담 건수: {len(consultations_data)}건")

    cursor = conn.cursor()

    try:
        # consultations 테이블 적재
        print("[INFO] consultations 테이블 적재 중...")

        # 상담사 데이터 확인
        cursor.execute("""
            SELECT COUNT(*) FROM employees
            WHERE id != 'EMP-TEDDY-DEFAULT' AND status = 'active'
            AND department ~ '^상담'
        """)
        agent_count = cursor.fetchone()[0]

        if agent_count == 0:
            print("[ERROR] 상담사 데이터가 없습니다. 상담사 데이터를 먼저 적재해주세요.")
            cursor.close()
            return False

        print(f"[INFO] 사용 가능한 상담사 수: {agent_count}명")
        print(f"[INFO] 대분류별 상담사 배분 로직 적용 (하이브리드: 90% 순차 + 10% 랜덤)")

        insert_consultation = """
            INSERT INTO consultations (
                id, customer_id, agent_id, status,
                category_main, category_sub, category_raw, handled_categories,
                title, call_date, call_time, call_duration, fcr, is_best_practice, quality_score,
                processing_timeline,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW(), NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                customer_id = EXCLUDED.customer_id,
                agent_id = EXCLUDED.agent_id,
                status = EXCLUDED.status,
                category_main = EXCLUDED.category_main,
                category_sub = EXCLUDED.category_sub,
                category_raw = EXCLUDED.category_raw,
                handled_categories = EXCLUDED.handled_categories,
                title = EXCLUDED.title,
                call_date = EXCLUDED.call_date,
                call_time = EXCLUDED.call_time,
                call_duration = EXCLUDED.call_duration,
                fcr = EXCLUDED.fcr,
                processing_timeline = EXCLUDED.processing_timeline,
                updated_at = NOW()
        """

        # 랜덤 시드 고정 (재현 가능한 배분을 위해)
        RANDOM_SEED = 42
        random.seed(RANDOM_SEED)
        print(f"[INFO] 랜덤 시드 고정: {RANDOM_SEED} (재현 가능한 배분)")

        # ⭐ 고객-상담 매칭 재설계: 2,500명 고객에 6,533건 분배
        num_total_consultations = len(consultations_data)
        num_customers = 2500
        customer_consult_counts = distribute_consultations_to_customers(
            num_customers, num_total_consultations, random
        )
        # 고객별 상담 인덱스 배정 (순차 배정)
        consultation_to_customer = {}  # consultation index → CUST-TEDDY-xxxxx
        consult_idx = 0
        for cust_idx, count in enumerate(customer_consult_counts):
            cust_id = f"CUST-TEDDY-{cust_idx + 1:05d}"
            for _ in range(count):
                if consult_idx < num_total_consultations:
                    consultation_to_customer[consult_idx] = cust_id
                    consult_idx += 1

        # 남은 상담이 있으면 마지막 고객들에게 배정
        while consult_idx < num_total_consultations:
            cust_id = f"CUST-TEDDY-{(consult_idx % num_customers) + 1:05d}"
            consultation_to_customer[consult_idx] = cust_id
            consult_idx += 1

        print(f"[INFO] 고객-상담 매칭: {num_customers}명 고객 × 평균 {num_total_consultations / num_customers:.1f}건")

        # 같은 고객에게 같은 카테고리 재상담 배정 (~15%)
        category_indices = {}
        for idx, row in enumerate(consultations_data):
            cat = normalize_category_raw(row.get("consulting_category", ""))
            if cat not in category_indices:
                category_indices[cat] = []
            category_indices[cat].append(idx)

        customer_indices = {}  # customer_id → [indices]
        for idx, cust_id in consultation_to_customer.items():
            if cust_id not in customer_indices:
                customer_indices[cust_id] = []
            customer_indices[cust_id].append(idx)

        repeat_count = 0
        for cust_id, indices in customer_indices.items():
            if len(indices) >= 2 and random.random() < 0.15 * len(indices):
                first_cat = normalize_category_raw(
                    consultations_data[indices[0]].get("consulting_category", "")
                )
                same_cat_pool = [i for i in category_indices.get(first_cat, [])
                                 if i not in indices and i in consultation_to_customer]
                if same_cat_pool and len(indices) >= 2:
                    swap_src = indices[1]
                    swap_dst = random.choice(same_cat_pool)
                    consultation_to_customer[swap_src], consultation_to_customer[swap_dst] = \
                        consultation_to_customer[swap_dst], consultation_to_customer[swap_src]
                    repeat_count += 1

        print(f"[INFO] 같은 고객 같은 카테고리 재상담 교환: {repeat_count}건")

        def convert_status(status: str) -> str:
            """상태 변환: "완료" → "completed" """
            status_map = {
                "완료": "completed",
                "진행중": "in_progress",
                "미완료": "incomplete"
            }
            return status_map.get(status, "completed")

        def convert_duration(seconds: int) -> Optional[str]:
            """초를 "MM:SS" 형식으로 변환"""
            if seconds is None:
                return None
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:02d}:{secs:02d}"

        all_main_categories = set()
        for row in consultations_data:
            category = row.get("consulting_category", "")
            main_category = map_to_main_category(category)
            all_main_categories.add(main_category)

        # 대분류별 풀 미리 생성 (캐싱, 중복 최소화)
        agent_pools_cache = {}
        used_agents_by_category = {}

        print("[INFO] 대분류별 상담사 풀 생성 중... (중복 최소화)")

        category_consultation_counts = {}
        for row in consultations_data:
            category = row.get("consulting_category", "")
            main_cat = map_to_main_category(category)
            category_consultation_counts[main_cat] = category_consultation_counts.get(main_cat, 0) + 1

        categories_to_process = [cat for cat, _ in sorted(category_consultation_counts.items(), key=lambda x: x[1], reverse=True) if cat != "기타"]
        if "기타" in category_consultation_counts:
            categories_to_process.append("기타")

        for main_cat in categories_to_process:
            agent_pools_cache[main_cat] = get_agent_pool_by_main_category(conn, main_cat, used_agents_by_category)
            used_agents_by_category[main_cat] = set(agent_pools_cache[main_cat])
            print(f"[INFO] {main_cat} 풀 생성 완료: {len(agent_pools_cache[main_cat])}명")

        consultation_batch = []
        category_counters = {}
        agent_consultation_counts = {}
        agent_assigned_times = {}
        old_to_new_id_map = {}

        # 모든 상담사 초기화 (전체 상담사 기준 + rank 기반 목표 설정)
        cursor.execute("""
            SELECT id, rank FROM employees
            WHERE id != 'EMP-TEDDY-DEFAULT' AND status = 'active'
            AND department ~ '^상담'
            ORDER BY rank
        """)
        agent_rows = cursor.fetchall()
        all_agents = [row[0] for row in agent_rows]
        agent_ranks = {row[0]: row[1] for row in agent_rows}

        # rank 기반 목표 상담 건수 계산
        total_consultations = len(consultations_data)

        def calculate_target(rank):
            group_targets = [
                (1, 5, 132, 124),    # 에이스
                (6, 15, 122, 108),   # 고성과
                (16, 35, 106, 88),   # 중상위
                (36, 55, 87, 79),    # 중하위
                (56, 60, 78, 75),    # 저성과 (상담팀만, IT/관리/교육팀 제외)
            ]
            for start, end, high, low in group_targets:
                if start <= rank <= end:
                    count = end - start + 1
                    idx = rank - start
                    return round(high - (high - low) * idx / (count - 1)) if count > 1 else high
            return 93  # 기본값

        agent_targets = {}
        for agent_id in all_agents:
            rank = agent_ranks.get(agent_id, 35)
            agent_targets[agent_id] = calculate_target(rank)

        # 목표 합계 조정
        current_total = sum(agent_targets.values())
        diff = total_consultations - current_total
        mid_agents = [aid for aid, rank in agent_ranks.items() if 15 <= rank <= 55]
        idx = 0
        while diff != 0 and mid_agents:
            agent_id = mid_agents[idx % len(mid_agents)]
            agent_targets[agent_id] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            idx += 1

        for agent_id in all_agents:
            agent_consultation_counts[agent_id] = 0
            agent_assigned_times[agent_id] = set()

        print(f"[INFO] 전체 상담사 수: {len(all_agents)}명")
        print(f"[INFO] 총 상담 건수: {total_consultations}건")
        print(f"[INFO] 목표 평균: {total_consultations / len(all_agents):.2f}건/명")
        print(f"[INFO] 목표 범위: {min(agent_targets.values())}건 ~ {max(agent_targets.values())}건 (rank 기반)")

        for row_idx, row in enumerate(consultations_data):
            old_consultation_id = row.get("id", "")
            category = row.get("consulting_category", "")
            status = convert_status(row.get("status", "완료"))
            title = f"{category} 상담" if category else "상담"

            # 고객 ID: 사전 배정된 매핑 사용
            customer_id = consultation_to_customer.get(row_idx, f"CUST-TEDDY-{(row_idx % num_customers) + 1:05d}")

            # 현실적인 날짜/시간 생성 (평일 6일만, 주말 제외)
            call_date = None
            call_time = None
            timestamp_key = None
            call_start_time_str = row.get("call_start_time")

            if call_start_time_str:
                try:
                    dt = datetime.fromisoformat(call_start_time_str.replace("Z", "+00:00"))
                    call_date = dt.date()
                    call_time = dt.time()
                    if call_date.weekday() >= 5:
                        call_date = BUSINESS_DAYS[row_idx % len(BUSINESS_DAYS)]
                    timestamp_key = dt.strftime("%Y%m%d%H%M")
                except:
                    pass

            if not call_date or not call_time:
                source_id = row.get("source_id", old_consultation_id)
                try:
                    seed_num = int(re.findall(r'\d+', str(source_id))[-1])
                except:
                    seed_num = int(hashlib.md5(str(source_id).encode()).hexdigest(), 16) % 100000

                day_index = seed_num % len(BUSINESS_DAYS)
                call_date = BUSINESS_DAYS[day_index]

                hour = 9 + (seed_num // 100) % 9
                if hour == 12 and (seed_num % 100) < 30:
                    hour = 9 + (seed_num // 1000) % 8
                    if hour >= 12:
                        hour += 1
                minute = seed_num % 60
                call_time = time_type(hour, minute, 0)

                timestamp_key = f"{call_date.strftime('%Y%m%d')}{hour:02d}{minute:02d}"

            # 카테고리 매핑
            main_category, sub_category, category_raw = map_to_category(category)

            if main_category not in category_counters:
                category_counters[main_category] = 0

            # 상담사 배분 (rank 기반 목표 달성률 우선)
            agent_pool = agent_pools_cache[main_category]

            def get_completion_rate(aid):
                current = agent_consultation_counts[aid]
                target = agent_targets.get(aid, 93)
                return current / target if target > 0 else 1.0

            min_rate = min(get_completion_rate(aid) for aid in all_agents)
            tolerance = 0.05

            candidates = [
                aid for aid in all_agents
                if get_completion_rate(aid) <= min_rate + tolerance
            ]

            pool_candidates = [aid for aid in candidates if aid in agent_pool]
            if pool_candidates:
                candidates = pool_candidates

            # 시간 충돌 체크
            current_timestamp_key = timestamp_key
            current_dt = datetime.strptime(timestamp_key, "%Y%m%d%H%M") if timestamp_key else datetime(2026, 1, 20, 9, 0, 0)

            available_candidates = [
                aid for aid in candidates
                if current_timestamp_key not in agent_assigned_times.get(aid, set())
            ]

            if not available_candidates:
                time_available = [
                    aid for aid in all_agents
                    if current_timestamp_key not in agent_assigned_times.get(aid, set())
                ]
                if time_available:
                    min_rate = min(get_completion_rate(aid) for aid in time_available)
                    available_candidates = [
                        aid for aid in time_available
                        if get_completion_rate(aid) <= min_rate + tolerance
                    ]

            # 시간 shift
            shift_count = 0
            while not available_candidates and shift_count < 60:
                shift_count += 1
                current_dt = current_dt + timedelta(minutes=1)
                current_timestamp_key = current_dt.strftime("%Y%m%d%H%M")

                time_available = [
                    aid for aid in all_agents
                    if current_timestamp_key not in agent_assigned_times.get(aid, set())
                ]
                if time_available:
                    min_rate = min(get_completion_rate(aid) for aid in time_available)
                    available_candidates = [
                        aid for aid in time_available
                        if get_completion_rate(aid) <= min_rate + tolerance
                    ]

            if not available_candidates:
                available_candidates = candidates if candidates else all_agents

            # 하이브리드 배분 (95% 순차 + 5% 랜덤)
            if random.random() < 0.05:
                agent_id = random.choice(available_candidates)
            else:
                pool_index = category_counters[main_category] % len(available_candidates)
                agent_id = available_candidates[pool_index]

            agent_consultation_counts[agent_id] += 1
            agent_assigned_times[agent_id].add(current_timestamp_key)
            category_counters[main_category] += 1

            # 새 형식으로 상담 ID 변환
            if shift_count > 0:
                shifted_time_str = current_dt.isoformat()
                consultation_id = convert_to_new_consultation_id(
                    old_id=old_consultation_id,
                    agent_id=agent_id,
                    call_start_time=shifted_time_str,
                    created_at=row.get("created_at")
                )
                call_date = current_dt.date()
                call_time = current_dt.time()
            else:
                generated_time_str = datetime.combine(call_date, call_time).isoformat()
                consultation_id = convert_to_new_consultation_id(
                    old_id=old_consultation_id,
                    agent_id=agent_id,
                    call_start_time=generated_time_str,
                    created_at=row.get("created_at")
                )

            old_to_new_id_map[old_consultation_id] = consultation_id

            call_duration = convert_duration(row.get("call_duration"))

            # handled_categories 생성 (93% 단일, 7% 복합)
            handled_cats = [category_raw] if category_raw else []
            if random.random() < 0.07 and main_category in RELATED_CATEGORIES:
                related_list = RELATED_CATEGORIES[main_category]
                if related_list:
                    handled_cats.append(random.choice(related_list))

            # processing_timeline 합성 생성
            timeline = generate_processing_timeline(
                call_time, call_duration, main_category, category_raw
            )
            timeline_json = json.dumps(timeline, ensure_ascii=False) if timeline else None

            consultation_batch.append((
                consultation_id,
                customer_id,
                agent_id,
                status,
                main_category,
                sub_category,
                category_raw,
                handled_cats,
                title,
                call_date,
                call_time,
                call_duration,
                True,  # fcr - 기본값 True
                False,  # is_best_practice
                None,  # quality_score
                timeline_json
            ))

        # 대분류별 배분 통계 출력
        print(f"\n[INFO] 대분류별 상담 건수:")
        for main_cat, count in sorted(category_counters.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {main_cat}: {count}건")

        # 고객 ID 매핑 통계 출력
        unique_customers = len(set(consultation_to_customer.values()))
        print(f"\n[INFO] 고객 ID 매핑: {unique_customers}명 고객에 {num_total_consultations}건 배분")
        cust_counts = Counter(consultation_to_customer.values())
        count_dist = Counter(cust_counts.values())
        print(f"[INFO] 고객별 상담 건수 분포:")
        for n_consults in sorted(count_dist.keys()):
            pct = count_dist[n_consults] / unique_customers * 100
            print(f"  - {n_consults}건: {count_dist[n_consults]}명 ({pct:.1f}%)")

        if consultation_batch:
            execute_batch(cursor, insert_consultation, consultation_batch, page_size=BATCH_SIZE)
            conn.commit()
            print(f"[INFO] consultations 적재 완료: {len(consultation_batch)}건")

            # FCR 계산
            print("[INFO] FCR (First Call Resolution) 계산 중...")
            cursor.execute("""
                UPDATE consultations c1
                SET fcr = FALSE
                WHERE EXISTS (
                    SELECT 1 FROM consultations c2
                    WHERE c2.customer_id = c1.customer_id
                      AND c2.category_raw = c1.category_raw
                      AND c2.call_date > c1.call_date
                      AND c2.call_date <= c1.call_date + INTERVAL '7 days'
                      AND c2.id != c1.id
                )
            """)
            fcr_updated = cursor.rowcount
            conn.commit()
            print(f"[INFO] FCR FALSE 업데이트: {fcr_updated}건 (7일 이내 재문의)")

            # FCR 통계 출력
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE fcr = TRUE) as fcr_success,
                    COUNT(*) FILTER (WHERE fcr = FALSE) as fcr_fail
                FROM consultations
            """)
            stats = cursor.fetchone()
            if stats[0] > 0:
                fcr_rate = (stats[1] / stats[0]) * 100
                print(f"[INFO] FCR 통계: 성공 {stats[1]}건 / 실패 {stats[2]}건 = {fcr_rate:.1f}%")

        # consultation_documents 적재
        if HANA_VECTORDB_FILE.exists():
            print(f"[INFO] 하나카드 VectorDB 파일 로드: {HANA_VECTORDB_FILE}")

            with open(HANA_VECTORDB_FILE, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)

            print(f"[INFO] 총 문서 건수: {len(documents_data)}건")

            insert_document = """
                INSERT INTO consultation_documents (
                    id, consultation_id, document_type, category, title, content,
                    keywords, embedding, metadata, usage_count, effectiveness_score, last_used,
                    created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (id) DO UPDATE SET
                    consultation_id = EXCLUDED.consultation_id,
                    document_type = EXCLUDED.document_type,
                    category = EXCLUDED.category,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    keywords = EXCLUDED.keywords,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """

            document_batch = []
            for row in documents_data:
                doc_id = row.get("id", "")
                consultation_id_raw = row.get("consultation_id", "")

                old_id = None
                if consultation_id_raw.startswith("CS-HANA-"):
                    old_id = consultation_id_raw.replace("CS-HANA-", "hana_consultation_")
                elif consultation_id_raw.startswith("hana_consultation_"):
                    old_id = consultation_id_raw
                else:
                    old_id = consultation_id_raw if consultation_id_raw else doc_id

                consultation_id = old_to_new_id_map.get(old_id, old_id)

                document_type = row.get("document_type", "consultation_transcript")
                category = row.get("metadata", {}).get("category", "")
                title = row.get("title", "")
                content = row.get("content", "")
                keywords = row.get("metadata", {}).get("keywords", [])
                if isinstance(keywords, str):
                    keywords = [k.strip() for k in keywords.split(",")]
                embedding = row.get("embedding")
                embedding_str = "[" + ",".join(map(str, embedding)) + "]" if embedding else None
                metadata = {
                    "source_id": row.get("metadata", {}).get("source_id"),
                    "slot_types": row.get("metadata", {}).get("slot_types", []),
                    "scenario_tags": row.get("metadata", {}).get("scenario_tags", []),
                    "summary": row.get("metadata", {}).get("summary"),
                    "created_at": row.get("metadata", {}).get("created_at")
                }

                document_batch.append((
                    doc_id,
                    consultation_id,
                    document_type,
                    category,
                    title,
                    content,
                    keywords,
                    embedding_str,
                    PsycopgJson(metadata),
                    0,  # usage_count
                    None,  # effectiveness_score
                    None  # last_used
                ))

            if document_batch:
                for i in range(0, len(document_batch), BATCH_SIZE):
                    batch = document_batch[i:i+BATCH_SIZE]
                    execute_batch(cursor, insert_document, batch, page_size=len(batch))
                    if (i + len(batch)) % COMMIT_INTERVAL == 0:
                        conn.commit()
                        print(f"[INFO] Committed {i + len(batch)} documents")
                conn.commit()
                print(f"[INFO] consultation_documents 적재 완료: {len(document_batch)}건")
        else:
            print(f"[WARNING] 하나카드 VectorDB 파일을 찾을 수 없습니다: {HANA_VECTORDB_FILE}")

        cursor.close()
        return True

    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] 하나카드 데이터 적재 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
