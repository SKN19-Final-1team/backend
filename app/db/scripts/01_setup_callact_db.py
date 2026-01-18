"""
CALL:ACT 통합 DB 설정 및 데이터 적재 스크립트

기능:
- DB 스키마 생성 (기본 테이블: employees, consultations, consultation_documents)
- 테디카드 테이블 생성 (service_guide_documents, card_products, notices)
- 키워드 사전 테이블 생성 (keyword_dictionary, keyword_synonyms)
- 상담사 데이터 적재 (employeesData.json)
- 하나카드 상담 데이터 적재 (hana_rdb_metadata.json, hana_vectordb_with_embeddings.json)
- 상담사 성과 지표 업데이트 (DB 실제 데이터 기반: consultations, fcr, avgTime, rank)
- 키워드 사전 데이터 적재
- 테디카드 데이터 적재 (service_guides, card_products, notices)
- 검증

사용법:
    python 01_setup_callact_db.py [옵션]

옵션:
    --skip-schema: 스키마 생성 건너뛰기
    --skip-employees: 상담사 데이터 적재 건너뛰기
    --skip-hana: 하나카드 데이터 적재 건너뛰기
    --skip-keywords: 키워드 사전 적재 건너뛰기
    --skip-teddycard: 테디카드 데이터 적재 건너뛰기
    --verify-only: 검증만 실행
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, Json as PsycopgJson, RealDictCursor
from tqdm import tqdm

# 환경 변수 로드
load_dotenv(Path(__file__).parent / '.env', override=False)
load_dotenv(Path(__file__).parent.parent.parent.parent / '.env', override=False)

# config.py에서 경로 가져오기
from config import (
    PROJECT_ROOT,
    TEDDY_DATA_DIR_PROD, TEDDY_DATA_DIR_DEV,
    KEYWORDS_DICT_DIR_PROD, KEYWORDS_DICT_DIR_DEV,
    HANA_RDB_METADATA_FILE, HANA_VECTORDB_FILE,
    EMPLOYEES_DATA_FILE
)

# 상수 (기존 호환성 유지)
BASE_DIR = PROJECT_ROOT
SCRIPTS_DIR = Path(__file__).parent

# 데이터 파일 경로 (config.py에서 가져온 값 사용)
DATA_DIR_PROD = TEDDY_DATA_DIR_PROD
DATA_DIR_DEV = TEDDY_DATA_DIR_DEV

# 환경 변수
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432")) if os.getenv("DB_PORT") else None
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
BATCH_SIZE = int(os.getenv("DB_LOAD_BATCH_SIZE", "100"))
COMMIT_INTERVAL = int(os.getenv("DB_COMMIT_INTERVAL", "500"))

# 필수 환경 변수 확인
if not all([DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME]):
    print("[ERROR] 필수 환경 변수가 설정되지 않았습니다.")
    print("[ERROR] .env 파일에 다음 변수를 설정해주세요:")
    print("  - DB_HOST")
    print("  - DB_PORT")
    print("  - DB_USER")
    print("  - DB_PASSWORD")
    print("  - DB_NAME")
    sys.exit(1)


def connect_db() -> psycopg2_connection:
    """PostgreSQL 데이터베이스 연결"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print(f"[INFO] Connected to database: {DB_NAME}")
        return conn
    except Exception as e:
        print(f"[ERROR] Failed to connect to database: {e}")
        sys.exit(1)


def load_sql_file(file_path: Path) -> str:
    """SQL 파일 읽기"""
    if not file_path.exists():
        print(f"[ERROR] SQL file not found: {file_path}")
        sys.exit(1)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def execute_sql_script(conn: psycopg2_connection, sql_script: str, description: str = ""):
    """SQL 스크립트 실행"""
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql_script)
        conn.commit()
        if description:
            print(f"[INFO] {description} 완료!")
    except Exception as e:
        conn.rollback()
        # 일부 에러는 무시 가능 (이미 존재하는 객체 등)
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            if description:
                print(f"[WARNING] {description} - 일부 객체가 이미 존재합니다. (무시됨)")
        else:
            print(f"[ERROR] {description} 실패: {e}")
            raise
    finally:
        cursor.close()


def setup_basic_schema(conn: psycopg2_connection):
    """기본 DB 스키마 생성"""
    print("\n" + "=" * 60)
    print("[1/9] 기본 DB 스키마 생성")
    print("=" * 60)
    
    sql_file = SCRIPTS_DIR / "db_setup.sql"
    sql_script = load_sql_file(sql_file)
    execute_sql_script(conn, sql_script, "기본 DB 스키마 생성")
    
    # 테이블 목록 확인
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = cursor.fetchall()
    print(f"[INFO] 생성된 테이블: {len(tables)}개")
    cursor.close()


def setup_teddycard_tables(conn: psycopg2_connection):
    """테디카드 테이블 생성 (통합 SQL 파일 사용)"""
    print("\n" + "=" * 60)
    print("[2/9] 테디카드 테이블 생성 (통합본)")
    print("=" * 60)
    
    # 통합 SQL 파일 실행 (테이블 생성 + 컬럼 추가 + ID 길이 수정 모두 포함)
    sql_file = SCRIPTS_DIR / "02_setup_tedicard_tables.sql"
    sql_script = load_sql_file(sql_file)
    execute_sql_script(conn, sql_script, "테디카드 테이블 생성 및 수정")


def setup_keyword_dictionary_tables(conn: psycopg2_connection):
    """키워드 사전 테이블 생성"""
    print("\n" + "=" * 60)
    print("[3/9] 키워드 사전 테이블 생성")
    print("=" * 60)
    
    sql_file = SCRIPTS_DIR / "03_setup_keyword_dictionary.sql"
    sql_script = load_sql_file(sql_file)
    execute_sql_script(conn, sql_script, "키워드 사전 테이블 생성")


def find_keywords_dict_file() -> Optional[Path]:
    """키워드 사전 파일 찾기"""
    KEYWORDS_DICT_FILES = [
        "keywords_dict_v2_with_patterns.json",
        "keywords_dict_with_compound.json",
        "keywords_dict_with_synonyms.json",
        "keywords_dict_v2.json"
    ]
    
    # 프로덕션 경로 확인
    for filename in KEYWORDS_DICT_FILES:
        file_path = KEYWORDS_DICT_DIR_PROD / filename
        if file_path.exists():
            print(f"[INFO] Found keywords dictionary file (PROD): {file_path}")
            return file_path
    
    # 개발 경로 확인
    for filename in KEYWORDS_DICT_FILES:
        file_path = KEYWORDS_DICT_DIR_DEV / filename
        if file_path.exists():
            print(f"[INFO] Found keywords dictionary file (DEV): {file_path}")
            return file_path
    
    print(f"[ERROR] Keywords dictionary file not found")
    return None


def check_table_has_data(conn: psycopg2_connection, table_name: str) -> Tuple[bool, int]:
    """테이블에 데이터가 있는지 확인"""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        return count > 0, count
    finally:
        cursor.close()


def check_employees_has_meaningful_data(conn: psycopg2_connection, expected_min_count: int = 10) -> Tuple[bool, int, bool]:
    """employees 테이블에 의미있는 데이터가 있는지 확인
    
    Returns:
        (has_meaningful_data, count, has_default_only): 
        - has_meaningful_data: 예상 개수 이상의 데이터가 있는지
        - count: 현재 데이터 개수
        - has_default_only: 기본 상담사만 있는지
    """
    cursor = conn.cursor()
    try:
        # 전체 개수 확인
        cursor.execute("SELECT COUNT(*) FROM employees")
        count = cursor.fetchone()[0]
        
        if count == 0:
            return False, 0, False
        
        # 기본 상담사 패턴 확인 (EMP-TEDDY-DEFAULT만 확인 - 실제로 생성된 기본 상담사)
        cursor.execute("""
            SELECT COUNT(*) 
            FROM employees 
            WHERE id = 'EMP-TEDDY-DEFAULT'
               OR email LIKE '%default%teddycard%'
        """)
        default_count = cursor.fetchone()[0]
        has_default_only = (default_count == count)
        
        # 의미있는 데이터가 있는지 확인 (예상 개수 이상이거나, 기본 상담사가 아님)
        has_meaningful_data = (count >= expected_min_count) and not has_default_only
        
        return has_meaningful_data, count, has_default_only
    finally:
        cursor.close()


def load_employees_data(conn: psycopg2_connection):
    """상담사 데이터 적재"""
    print("\n" + "=" * 60)
    print("[4/9] 상담사 데이터 적재")
    print("=" * 60)
    
    # employeesData.json 파일 읽기 (예상 개수 확인용)
    if EMPLOYEES_DATA_FILE.exists():
        with open(EMPLOYEES_DATA_FILE, 'r', encoding='utf-8') as f:
            expected_employees = json.load(f)
        expected_count = len(expected_employees)
    else:
        expected_count = 50  # 기본 예상값
    
    # 의미있는 데이터가 이미 있는지 확인 (기본 상담사만 있는 경우 적재 진행)
    has_meaningful_data, current_count, has_default_only = check_employees_has_meaningful_data(conn, expected_min_count=10)
    
    if has_meaningful_data and current_count >= expected_count * 0.8:  # 80% 이상 있으면 스킵
        print(f"[INFO] employees 테이블에 이미 의미있는 데이터가 있습니다. (건수: {current_count}건, 예상: {expected_count}건) - 적재 스킵")
        if current_count < expected_count:
            print(f"[WARNING] 예상 개수보다 적습니다. (예상: {expected_count}건, 실제: {current_count}건)")
        return True
    elif has_default_only:
        print(f"[INFO] employees 테이블에 기본 상담사만 있습니다. (건수: {current_count}건, 기본 상담사: {has_default_only})")
        print(f"[INFO] 실제 상담사 데이터를 적재합니다. (예상: {expected_count}건)")
    elif current_count > 0:
        print(f"[INFO] employees 테이블에 데이터가 있지만 부족합니다. (건수: {current_count}건, 예상: {expected_count}건)")
        print(f"[INFO] 추가 데이터를 적재합니다.")
    
    # employeesData.json 파일 읽기 (이미 읽었으면 재사용)
    if not EMPLOYEES_DATA_FILE.exists():
        print(f"[ERROR] 상담사 데이터 파일을 찾을 수 없습니다: {EMPLOYEES_DATA_FILE}")
        return False
    
    # expected_employees가 아직 정의되지 않은 경우 다시 읽기
    if 'expected_employees' not in locals():
        with open(EMPLOYEES_DATA_FILE, 'r', encoding='utf-8') as f:
            expected_employees = json.load(f)
    
    employees_data = expected_employees
    print(f"[INFO] 상담사 데이터 파일 로드: {len(employees_data)}명")
    
    cursor = conn.cursor()
    
    try:
        # employees 테이블 적재
        print("[INFO] employees 테이블 적재 중...")
        
        insert_employee = """
            INSERT INTO employees (
                id, name, email, role, department, status, 
                consultations, fcr, "avgTime", rank, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                email = EXCLUDED.email,
                role = EXCLUDED.role,
                department = EXCLUDED.department,
                status = EXCLUDED.status,
                consultations = EXCLUDED.consultations,
                fcr = EXCLUDED.fcr,
                "avgTime" = EXCLUDED."avgTime",
                rank = EXCLUDED.rank,
                updated_at = NOW()
        """
        
        employee_batch = []
        for emp in employees_data:
            # DB 적재에 필요한 필드 추출
            emp_id = emp.get('id', '')
            name = emp.get('name', '')
            email = emp.get('email', '')
            team = emp.get('team', '')  # department로 사용
            position = emp.get('position', '')  # role로 사용
            status = emp.get('status', 'active')
            # 성과 지표 (초기값, 나중에 실제 DB 값으로 업데이트됨)
            consultations = emp.get('consultations', 0)
            fcr = emp.get('fcr', 0)
            avgTime = emp.get('avgTime', '0:00')
            rank = emp.get('rank', 0)
            
            employee_batch.append((
                emp_id,
                name,
                email,
                position,  # role
                team,  # department
                status,
                consultations,  # 초기값
                fcr,  # 초기값
                avgTime,  # 초기값
                rank  # 초기값
            ))
        
        # 상담사 적재
        if employee_batch:
            execute_batch(cursor, insert_employee, employee_batch, page_size=BATCH_SIZE)
            conn.commit()
            print(f"[INFO] 상담사 적재 완료: {len(employee_batch)}명")
        else:
            print("[WARNING] 적재할 상담사 데이터가 없습니다.")
        
        cursor.close()
        return True
        
    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] 상담사 데이터 적재 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 대분류 매핑 (57개 카테고리 → 8개 대분류) ====================

# 대분류별 담당 풀 크기 설정 (70명 기준)
MAIN_CATEGORY_POOL_SIZES = {
    "분실/도난": 5,      # ~400건
    "한도": 6,           # ~564건 (한도상향 + 한도 안내)
    "결제/승인": 10,     # ~1,559건 (선결제 + 결제대금 + 승인취소) - 최대
    "이용내역": 7,       # ~919건
    "수수료/연체": 3,    # ~163건
    "포인트/혜택": 4,    # ~223건
    "정부지원": 3,       # ~167건
    "기타": None,        # 순차 분배 (모든 상담사)
}

# 대분류 매핑 딕셔너리 (57개 카테고리 → 8개 대분류)
# 문서에서 확인한 상위 10개 카테고리 + 패턴 기반 매핑
CATEGORY_TO_MAIN_CATEGORY = {
    # 분실/도난 (긴급 처리 필요)
    "도난/분실 신청/해제": "분실/도난",
    "카드 분실 신고": "분실/도난",
    "카드 분실": "분실/도난",
    "도난 신고": "분실/도난",
    "분실 신청": "분실/도난",
    "분실 해제": "분실/도난",
    
    # 한도 (한도 관련)
    "한도상향 접수/처리": "한도",
    "한도 안내": "한도",
    "한도 조회": "한도",
    "한도 상향": "한도",
    "한도 증액": "한도",
    
    # 결제/승인 (결제 및 승인 관련)
    "선결제/즉시출금": "결제/승인",
    "결제대금 안내": "결제/승인",
    "승인취소/매출취소 안내": "결제/승인",
    "결제 안내": "결제/승인",
    "결제 확인": "결제/승인",
    "승인 취소": "결제/승인",
    "매출 취소": "결제/승인",
    "즉시 출금": "결제/승인",
    "선결제": "결제/승인",
    
    # 이용내역 (이용내역 관련)
    "이용내역 안내": "이용내역",
    "이용내역 조회": "이용내역",
    "이용 내역": "이용내역",
    
    # 수수료/연체 (수수료 및 연체 관련)
    "연체대금 즉시출금": "수수료/연체",
    "연체 수수료 안내": "수수료/연체",
    "연체대금 안내": "수수료/연체",
    "수수료 안내": "수수료/연체",
    "연체 안내": "수수료/연체",
    
    # 포인트/혜택 (포인트 및 혜택 관련)
    "이벤트 안내": "포인트/혜택",
    "포인트 적립": "포인트/혜택",
    "포인트 사용": "포인트/혜택",
    "포인트 안내": "포인트/혜택",
    "혜택 안내": "포인트/혜택",
    
    # 정부지원 (정부지원 관련)
    "정부지원 바우처 (등유, 임신 등)": "정부지원",
    "정부지원 바우처": "정부지원",
    "바우처 안내": "정부지원",
    
    # 기본값: "기타"로 처리 (위에 매핑되지 않은 모든 카테고리)
}


def map_to_main_category(category: str) -> str:
    """
    세부 카테고리를 대분류로 매핑
    
    Args:
        category: 세부 카테고리 (예: "선결제/즉시출금")
    
    Returns:
        대분류 (예: "결제/승인")
    """
    # 정확한 매칭 시도
    if category in CATEGORY_TO_MAIN_CATEGORY:
        return CATEGORY_TO_MAIN_CATEGORY[category]
    
    # 부분 매칭 (키워드 기반)
    category_lower = category.lower()
    
    # 분실/도난
    if any(keyword in category_lower for keyword in ["분실", "도난", "분실 신고"]):
        return "분실/도난"
    
    # 한도
    if any(keyword in category_lower for keyword in ["한도"]):
        return "한도"
    
    # 결제/승인
    if any(keyword in category_lower for keyword in ["결제", "승인", "매출", "즉시출금", "선결제"]):
        return "결제/승인"
    
    # 이용내역
    if any(keyword in category_lower for keyword in ["이용내역", "이용 내역"]):
        return "이용내역"
    
    # 수수료/연체
    if any(keyword in category_lower for keyword in ["연체", "수수료"]):
        return "수수료/연체"
    
    # 포인트/혜택
    if any(keyword in category_lower for keyword in ["포인트", "혜택", "이벤트"]):
        return "포인트/혜택"
    
    # 정부지원
    if any(keyword in category_lower for keyword in ["바우처", "정부지원"]):
        return "정부지원"
    
    # 기본값: "기타"
    return "기타"


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
                ORDER BY created_at ASC
            """)
        else:
            # 대분류별 풀 크기 결정
            pool_size = MAIN_CATEGORY_POOL_SIZES.get(main_category, 5)
            
            # 다른 대분류에서 사용된 상담사 제외 (중복 최소화)
            # 단, "기타"는 모든 상담사가 포함되므로 제외 대상에서 제외
            if used_agents_by_category:
                # 현재 대분류와 "기타"를 제외한 다른 대분류에서 사용된 상담사 집합
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
                        AND id NOT IN ({placeholders})
                        ORDER BY created_at ASC
                        LIMIT %s
                    """, list(other_used_agents) + [pool_size])
                else:
                    cursor.execute("""
                        SELECT id FROM employees 
                        WHERE id != 'EMP-TEDDY-DEFAULT' 
                        AND status = 'active'
                        ORDER BY created_at ASC
                        LIMIT %s
                    """, (pool_size,))
            else:
                cursor.execute("""
                    SELECT id FROM employees 
                    WHERE id != 'EMP-TEDDY-DEFAULT' 
                    AND status = 'active'
                    ORDER BY created_at ASC
                    LIMIT %s
                """, (pool_size,))
        
        agent_ids = [row[0] for row in cursor.fetchall()]
        
        if not agent_ids:
            raise ValueError(f"[ERROR] 상담사 풀이 비어있습니다. (대분류: {main_category})")
        
        return agent_ids
    
    finally:
        cursor.close()


def load_hana_data(conn: psycopg2_connection):
    """하나카드 데이터 적재"""
    print("\n" + "=" * 60)
    print("[5/9] 하나카드 데이터 적재")
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
                id, customer_id, agent_id, status, category, title,
                call_date, call_time, call_duration, fcr, is_best_practice, quality_score,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                customer_id = EXCLUDED.customer_id,
                agent_id = EXCLUDED.agent_id,
                status = EXCLUDED.status,
                category = EXCLUDED.category,
                title = EXCLUDED.title,
                call_date = EXCLUDED.call_date,
                call_time = EXCLUDED.call_time,
                call_duration = EXCLUDED.call_duration,
                updated_at = NOW()
        """
        
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
        
        # 풀 캐싱: 모든 대분류 수집 및 풀 미리 생성
        import random
        
        # 랜덤 시드 고정 (재현 가능한 배분을 위해)
        RANDOM_SEED = 42  # 고정 시드 값
        random.seed(RANDOM_SEED)
        print(f"[INFO] 랜덤 시드 고정: {RANDOM_SEED} (재현 가능한 배분)")
        
        all_main_categories = set()
        for row in consultations_data:
            category = row.get("consulting_category", "")
            main_category = map_to_main_category(category)
            all_main_categories.add(main_category)
        
        # 대분류별 풀 미리 생성 (캐싱, 중복 최소화)
        agent_pools_cache = {}
        used_agents_by_category = {}  # 대분류별 사용된 상담사 추적
        
        print("[INFO] 대분류별 상담사 풀 생성 중... (중복 최소화)")
        
        # 대분류별 상담 건수 순으로 정렬하여, 많은 상담 건수를 가진 대분류부터 풀 생성
        category_consultation_counts = {}
        for row in consultations_data:
            category = row.get("consulting_category", "")
            main_cat = map_to_main_category(category)
            category_consultation_counts[main_cat] = category_consultation_counts.get(main_cat, 0) + 1
        
        # 상담 건수가 많은 순으로 정렬 (단, "기타"는 마지막에 처리)
        categories_to_process = [cat for cat, _ in sorted(category_consultation_counts.items(), key=lambda x: x[1], reverse=True) if cat != "기타"]
        if "기타" in category_consultation_counts:
            categories_to_process.append("기타")  # "기타"는 마지막에 추가
        
        for main_cat in categories_to_process:
            agent_pools_cache[main_cat] = get_agent_pool_by_main_category(conn, main_cat, used_agents_by_category)
            used_agents_by_category[main_cat] = set(agent_pools_cache[main_cat])
            print(f"[INFO] {main_cat} 풀 생성 완료: {len(agent_pools_cache[main_cat])}명")
        
        consultation_batch = []
        category_counters = {}  # 대분류별 카운터 (순차 분배용)
        agent_consultation_counts = {}  # 상담사별 총 상담 건수 추적 (균등 배분용)
        
        # 모든 상담사 초기화 (전체 상담사 기준)
        cursor.execute("""
            SELECT id FROM employees 
            WHERE id != 'EMP-TEDDY-DEFAULT' AND status = 'active'
        """)
        all_agents = [row[0] for row in cursor.fetchall()]
        for agent_id in all_agents:
            agent_consultation_counts[agent_id] = 0
        
        print(f"[INFO] 전체 상담사 수: {len(all_agents)}명")
        print(f"[INFO] 총 상담 건수: {len(consultations_data)}건")
        print(f"[INFO] 목표 평균: {len(consultations_data) / len(all_agents):.2f}건/명")
        
        for row in consultations_data:
            consultation_id = row.get("id", "")
            customer_id = row.get("client_id", "")
            category = row.get("consulting_category", "")
            status = convert_status(row.get("status", "완료"))
            title = f"{category} 상담" if category else "상담"
            
            # 대분류 매핑 및 상담사 배분
            main_category = map_to_main_category(category)
            
            # 대분류별 카운터 초기화
            if main_category not in category_counters:
                category_counters[main_category] = 0
            
            # 상담사 배분 (하이브리드: 90% 순차 + 10% 랜덤, 전체 상담사 기준 최소 건수 우선 + 풀 제한 + 적절한 편차 허용)
            agent_pool = agent_pools_cache[main_category]
            
            # 1. 전체 상담사 중 최소 건수 확인 (균등 배분을 위해)
            all_min_count = min(agent_consultation_counts.values())
            all_max_count = max(agent_consultation_counts.values())
            
            # 2. 풀 내 상담사들의 건수 확인
            pool_counts = {aid: agent_consultation_counts.get(aid, 0) for aid in agent_pool}
            pool_min_count = min(pool_counts.values())
            
            # 3. 적절한 편차 허용 (전체 최소 건수와의 차이를 10건 이내로 허용)
            # 목표: 1등 110건, 꼴등 80건 정도의 편차 (평균 93건 기준 ±17건)
            # 하지만 풀 내에서는 더 엄격하게 제한 (5건 이내)
            tolerance = 10  # 전체 최소 건수와의 허용 편차
            
            # 4. 풀 내 상담사 중에서 전체 최소 건수에 가까운 상담사 선택
            # 전체 최소 건수와의 차이가 tolerance 이내인 상담사만 후보로 선택
            candidates = [
                aid for aid in agent_pool 
                if abs(pool_counts[aid] - all_min_count) <= tolerance
            ]
            
            # 5. 풀 내에 전체 최소 건수에 가까운 상담사가 없으면, 풀 내 최소 건수 상담사 선택 (fallback)
            if not candidates:
                candidates = [aid for aid in agent_pool if pool_counts[aid] == pool_min_count]
            
            # 6. 후보가 여러 명이면, 전체 최소 건수에 가장 가까운 상담사 우선 선택
            # 단, 차이가 3건 이내면 모두 후보로 유지 (약간의 다양성 허용)
            if len(candidates) > 1:
                min_diff = min(abs(pool_counts[aid] - all_min_count) for aid in candidates)
                # 차이가 3건 이내인 상담사들은 모두 후보로 유지
                candidates = [
                    aid for aid in candidates 
                    if abs(pool_counts[aid] - all_min_count) <= min_diff + 3
                ]
            
            # 7. 하이브리드 배분 (90% 순차 + 10% 랜덤)
            if random.random() < 0.1:  # 10% 랜덤 (후보 상담사 중에서)
                agent_id = random.choice(candidates)
            else:  # 90% 순차 (후보 상담사 중에서)
                pool_index = category_counters[main_category] % len(candidates)
                agent_id = candidates[pool_index]
            
            # 건수 증가
            agent_consultation_counts[agent_id] += 1
            category_counters[main_category] += 1
            
            # 날짜/시간 처리
            call_date = None
            call_time = None
            if row.get("call_start_time"):
                try:
                    dt = datetime.fromisoformat(row["call_start_time"].replace("Z", "+00:00"))
                    call_date = dt.date()
                    call_time = dt.time()
                except:
                    pass
            
            call_duration = convert_duration(row.get("call_duration"))
            
            consultation_batch.append((
                consultation_id,
                customer_id,
                agent_id,
                status,
                category,
                title,
                call_date,
                call_time,
                call_duration,
                None,  # fcr
                False,  # is_best_practice
                None  # quality_score
            ))
        
        # 대분류별 배분 통계 출력
        print(f"\n[INFO] 대분류별 상담 건수:")
        for main_cat, count in sorted(category_counters.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {main_cat}: {count}건")
        
        if consultation_batch:
            execute_batch(cursor, insert_consultation, consultation_batch, page_size=BATCH_SIZE)
            conn.commit()
            print(f"[INFO] consultations 적재 완료: {len(consultation_batch)}건")
        
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
                if consultation_id_raw.startswith("CS-HANA-"):
                    consultation_id = consultation_id_raw.replace("CS-HANA-", "hana_consultation_")
                else:
                    consultation_id = consultation_id_raw if consultation_id_raw else doc_id
                
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


def convert_time_to_seconds(time_str: str) -> int:
    """시간 문자열("MM:SS" 또는 "HH:MM:SS")을 초로 변환"""
    if not time_str or time_str == "0:00":
        return 0
    try:
        parts = time_str.split(':')
        if len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            return 0
    except (ValueError, AttributeError):
        return 0


def convert_seconds_to_time(seconds: int) -> str:
    """초를 시간 문자열("MM:SS")로 변환"""
    if seconds <= 0:
        return "0:00"
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def load_keyword_dictionary(conn: psycopg2_connection):
    """키워드 사전 데이터 적재"""
    print("\n" + "=" * 60)
    print("[6/9] 키워드 사전 데이터 적재")
    print("=" * 60)
    
    # 이미 데이터가 있는지 확인
    has_data, count = check_table_has_data(conn, "keyword_dictionary")
    if has_data:
        print(f"[INFO] keyword_dictionary 테이블에 이미 데이터가 있습니다. (건수: {count}건) - 적재 스킵")
        return True
    
    # 키워드 사전 파일 찾기
    KEYWORDS_DICT_FILES = [
        "keywords_dict_v2_with_patterns.json",
        "keywords_dict_with_compound.json",
        "keywords_dict_with_synonyms.json",
        "keywords_dict_v2.json"
    ]
    
    keywords_file = None
    # 프로덕션 경로 확인
    for filename in KEYWORDS_DICT_FILES:
        file_path = KEYWORDS_DICT_DIR_PROD / filename
        if file_path.exists():
            keywords_file = file_path
            break
    
    # 개발 경로 확인
    if not keywords_file:
        for filename in KEYWORDS_DICT_FILES:
            file_path = KEYWORDS_DICT_DIR_DEV / filename
            if file_path.exists():
                keywords_file = file_path
                break
    
    if not keywords_file:
        print("[ERROR] 키워드 사전 파일을 찾을 수 없습니다.")
        return False
    
    print(f"[INFO] 키워드 사전 파일: {keywords_file}")
    
    # JSON 파일 로드
    with open(keywords_file, 'r', encoding='utf-8') as f:
        keyword_dict = json.load(f)
    
    # 키워드 사전 구조 확인 (keywords_dict_v2 형식)
    if "keywords" in keyword_dict:
        keywords_data = keyword_dict["keywords"]
        print(f"[INFO] 키워드 사전 파일 로드: {len(keywords_data)}개 키워드")
    else:
        # 리스트 형식인 경우
        keywords_data = keyword_dict if isinstance(keyword_dict, list) else []
        print(f"[INFO] 키워드 사전 파일 로드: {len(keywords_data)}개 키워드")
    
    cursor = conn.cursor()
    
    try:
        # keyword_dictionary 테이블 적재
        print("[INFO] keyword_dictionary 테이블 적재 중...")
        
        keyword_dict_insert = """
            INSERT INTO keyword_dictionary (
                keyword, category, priority, urgency, context_hints,
                weight, synonyms, variations, compound_patterns, ambiguity_rules
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (keyword, category) 
            DO UPDATE SET
                priority = EXCLUDED.priority,
                urgency = EXCLUDED.urgency,
                context_hints = EXCLUDED.context_hints,
                weight = EXCLUDED.weight,
                synonyms = EXCLUDED.synonyms,
                variations = EXCLUDED.variations,
                compound_patterns = EXCLUDED.compound_patterns,
                ambiguity_rules = EXCLUDED.ambiguity_rules,
                updated_at = NOW()
        """
        
        keyword_batch = []
        
        # keywords_dict_v2 형식 처리
        if isinstance(keywords_data, dict):
            for keyword, data in tqdm(keywords_data.items(), desc="키워드 처리"):
                canonical = data.get("canonical", keyword)
                
                # 각 카테고리별로 키워드 추가
                for cat_info in data.get("categories", []):
                    category = cat_info.get("category", "")
                    priority = cat_info.get("priority", 5)
                    urgency = cat_info.get("urgency", "medium")
                    context_hints = cat_info.get("context_hints", [])
                    weight = float(cat_info.get("weight", 1.0))
                    synonyms = data.get("synonyms", [])
                    variations = data.get("variations", [])
                    compound_patterns = cat_info.get("compound_patterns")
                    ambiguity_rules = cat_info.get("ambiguity_rules")
                    
                    keyword_batch.append((
                        canonical, category, priority, urgency, context_hints,
                        weight, synonyms, variations,
                        PsycopgJson(compound_patterns) if compound_patterns else None,
                        PsycopgJson(ambiguity_rules) if ambiguity_rules else None
                    ))
        else:
            # 리스트 형식 처리
            for keyword_entry in tqdm(keywords_data, desc="키워드 처리"):
                keyword = keyword_entry.get('keyword', '')
                category = keyword_entry.get('category', '')
                priority = keyword_entry.get('priority', 5)
                urgency = keyword_entry.get('urgency', 'medium')
                context_hints = keyword_entry.get('context_hints', [])
                weight = float(keyword_entry.get('weight', 1.0))
                synonyms = keyword_entry.get('synonyms', [])
                variations = keyword_entry.get('variations', [])
                compound_patterns = keyword_entry.get('compound_patterns')
                ambiguity_rules = keyword_entry.get('ambiguity_rules')
                
                keyword_batch.append((
                    keyword, category, priority, urgency, context_hints,
                    weight, synonyms, variations,
                    PsycopgJson(compound_patterns) if compound_patterns else None,
                    PsycopgJson(ambiguity_rules) if ambiguity_rules else None
                ))
        
        # 키워드 적재
        if keyword_batch:
            execute_batch(cursor, keyword_dict_insert, keyword_batch, page_size=BATCH_SIZE)
            conn.commit()
            print(f"[INFO] 키워드 적재 완료: {len(keyword_batch)}개")
        else:
            print("[WARNING] 적재할 키워드가 없습니다.")
        
        # keyword_synonyms 테이블 적재
        print("[INFO] keyword_synonyms 테이블 적재 중...")
        
        keyword_synonyms_insert = """
            INSERT INTO keyword_synonyms (synonym, canonical_keyword, category)
            VALUES (%s, %s, %s)
            ON CONFLICT (synonym, canonical_keyword, category) DO NOTHING
        """
        
        synonym_batch = []
        
        # keywords_dict_v2 형식 처리
        if isinstance(keywords_data, dict):
            for keyword, data in keywords_data.items():
                canonical = data.get("canonical", keyword)
                synonyms = data.get("synonyms", [])
                
                # 각 카테고리별로 동의어 매핑 생성
                for cat_info in data.get("categories", []):
                    category = cat_info.get("category", "")
                    
                    for synonym in synonyms:
                        if synonym and synonym != canonical:
                            synonym_batch.append((synonym, canonical, category))
        else:
            # 리스트 형식 처리
            for keyword_entry in keywords_data:
                keyword = keyword_entry.get('keyword', '')
                category = keyword_entry.get('category', '')
                synonyms = keyword_entry.get('synonyms', [])
                
                for synonym in synonyms:
                    if synonym and synonym != keyword:
                        synonym_batch.append((synonym, keyword, category))
        
        # 동의어 적재
        if synonym_batch:
            execute_batch(cursor, keyword_synonyms_insert, synonym_batch, page_size=BATCH_SIZE)
            conn.commit()
            print(f"[INFO] 동의어 적재 완료: {len(synonym_batch)}개")
        else:
            print("[INFO] 적재할 동의어가 없습니다.")
        
        cursor.close()
        return True
        
    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] 키워드 사전 적재 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def update_employee_performance(conn: psycopg2_connection):
    """DB에서 실제 상담 데이터를 기반으로 employees 테이블의 성과 지표 업데이트
    
    - consultations: agent_id별 실제 상담 건수
    - fcr: First Call Resolution 비율 (0-100)
    - avgTime: 평균 상담 시간 ("MM:SS" 형식)
    - rank: 성과 순위 (consultations DESC, fcr DESC, avgTime ASC 기준)
    """
    print("\n" + "=" * 60)
    print("[9/9] 상담사 성과 지표 업데이트 (DB 실제 데이터 기반)")
    print("=" * 60)
    
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    try:
        # 1. agent_id별 실제 상담 건수 및 성과 지표 계산
        cursor.execute("""
            SELECT 
                agent_id,
                COUNT(*) as consultations,
                COUNT(*) FILTER (WHERE fcr = true) as fcr_count,
                AVG(CASE 
                    WHEN call_duration IS NOT NULL AND call_duration != '' THEN
                        CASE 
                            WHEN call_duration ~ '^[0-9]+:[0-9]{2}$' THEN
                                -- MM:SS 형식
                                EXTRACT(EPOCH FROM (
                                    INTERVAL '1 minute' * SPLIT_PART(call_duration, ':', 1)::INTEGER +
                                    INTERVAL '1 second' * SPLIT_PART(call_duration, ':', 2)::INTEGER
                                ))
                            WHEN call_duration ~ '^[0-9]+:[0-9]{2}:[0-9]{2}$' THEN
                                -- HH:MM:SS 형식
                                EXTRACT(EPOCH FROM (
                                    INTERVAL '1 hour' * SPLIT_PART(call_duration, ':', 1)::INTEGER +
                                    INTERVAL '1 minute' * SPLIT_PART(call_duration, ':', 2)::INTEGER +
                                    INTERVAL '1 second' * SPLIT_PART(call_duration, ':', 3)::INTEGER
                                ))
                            ELSE 0
                        END
                    ELSE 0
                END) as avg_duration_seconds
            FROM consultations
            WHERE agent_id IS NOT NULL
            GROUP BY agent_id
        """)
        
        performance_data = cursor.fetchall()
        
        if not performance_data:
            print("[WARNING] consultations 테이블에 상담사별 데이터가 없습니다. 성과 지표를 업데이트할 수 없습니다.")
            cursor.close()
            return False
        
        print(f"[INFO] {len(performance_data)}명의 상담사 성과 지표 계산 중...")
        
        # 2. 성과 지표 계산 및 정렬 (rank 계산용)
        employees_performance = []
        for row in performance_data:
            consultations = row['consultations'] or 0
            fcr_count = row['fcr_count'] or 0
            fcr_percentage = int((fcr_count / consultations * 100)) if consultations > 0 else 0
            avg_duration_seconds = int(row['avg_duration_seconds'] or 0)
            avg_time_str = convert_seconds_to_time(avg_duration_seconds)
            
            employees_performance.append({
                'agent_id': row['agent_id'],
                'consultations': consultations,
                'fcr': fcr_percentage,
                'avgTime': avg_time_str,
                'avg_duration_seconds': avg_duration_seconds  # 정렬용
            })
        
        # 3. rank 계산 (consultations DESC, fcr DESC, avgTime ASC)
        employees_performance.sort(
            key=lambda x: (
                -x['consultations'],  # 내림차순
                -x['fcr'],  # 내림차순
                x['avg_duration_seconds']  # 오름차순 (짧을수록 좋음)
            )
        )
        
        for rank, emp in enumerate(employees_performance, start=1):
            emp['rank'] = rank
        
        # 4. employees 테이블 업데이트
        update_query = """
            UPDATE employees
            SET 
                consultations = %s,
                fcr = %s,
                "avgTime" = %s,
                rank = %s,
                updated_at = NOW()
            WHERE id = %s
        """
        
        update_batch = []
        for emp in employees_performance:
            update_batch.append((
                emp['consultations'],
                emp['fcr'],
                emp['avgTime'],
                emp['rank'],
                emp['agent_id']
            ))
        
        if update_batch:
            execute_batch(cursor, update_query, update_batch, page_size=BATCH_SIZE)
            conn.commit()
            
            print(f"[INFO] 성과 지표 업데이트 완료: {len(update_batch)}명")
            print(f"[INFO] 상위 5명:")
            for emp in employees_performance[:5]:
                print(f"  {emp['rank']}위: {emp['agent_id']} - {emp['consultations']}건, FCR {emp['fcr']}%, {emp['avgTime']}")
        else:
            print("[WARNING] 업데이트할 성과 지표가 없습니다.")
        
        cursor.close()
        return True
        
    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] 성과 지표 업데이트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_data_file(filename: str) -> Optional[Path]:
    """데이터 파일 찾기"""
    file_path = DATA_DIR_PROD / filename
    if file_path.exists():
        print(f"[INFO] Found data file (PROD): {file_path}")
        return file_path
    
    file_path = DATA_DIR_DEV / filename
    if file_path.exists():
        print(f"[INFO] Found data file (DEV): {file_path}")
        return file_path
    
    print(f"[ERROR] Data file not found: {filename}")
    return None


def extract_document_number(title: str) -> Optional[str]:
    """제목에서 문서 번호 추출 (예: 제1조, 제2조)"""
    if not title:
        return None
    match = re.search(r'제\d+조', title)
    return match.group(0) if match else None


def prepare_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    """metadata 준비"""
    metadata = doc.get("metadata", {}).copy()
    title = doc.get("title", "")
    document_number = extract_document_number(title)
    if document_number:
        metadata["document_number"] = document_number
    return metadata


def map_service_guide_data(doc: Dict[str, Any]) -> Tuple:
    """service_guide_documents 데이터 매핑"""
    doc_id = doc.get("id", "")
    document_type = doc.get("document_type", "service_guide")
    category = doc.get("category", "")
    title = doc.get("title", "")
    content = doc.get("content", "") or doc.get("text", "")
    keywords = doc.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    embedding = doc.get("embedding")
    embedding_str = "[" + ",".join(map(str, embedding)) + "]" if embedding else None
    metadata = prepare_metadata(doc)
    document_source = doc.get("document_source", "") or metadata.get("original_source", "")
    priority = doc.get("priority", "normal")
    structured = doc.get("structured")
    usage_count = doc.get("usage_count", 0)
    last_used = None
    return (
        doc_id, document_type, category, title, content, keywords,
        embedding_str, PsycopgJson(metadata), document_source, priority,
        usage_count, last_used, PsycopgJson(structured) if structured else None
    )


def map_card_product_data(doc: Dict[str, Any]) -> Tuple:
    """card_products 데이터 매핑"""
    doc_id = doc.get("id", "")
    name = doc.get("name", "")
    card_type = doc.get("card_type", "credit")
    brand = doc.get("brand", "local")
    annual_fee_domestic = doc.get("annual_fee_domestic")
    annual_fee_global = doc.get("annual_fee_global")
    performance_condition = doc.get("performance_condition", "")
    main_benefits = doc.get("main_benefits", "")
    status = doc.get("status", "active")
    metadata = doc.get("metadata", {})
    structured = doc.get("structured")
    return (
        doc_id, name, card_type, brand, annual_fee_domestic, annual_fee_global,
        performance_condition, main_benefits, status,
        PsycopgJson(metadata) if metadata else None,
        PsycopgJson(structured) if structured else None
    )


def map_notice_data(doc: Dict[str, Any]) -> Tuple:
    """notices 데이터 매핑"""
    doc_id = doc.get("id", "")
    title = doc.get("title", "")
    content = doc.get("content", "") or doc.get("text", "")
    category = doc.get("category", "system")
    priority = doc.get("priority", "normal")
    is_pinned = doc.get("is_pinned", False)
    start_date = doc.get("start_date")
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = doc.get("end_date")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
    status = doc.get("status", "active")
    created_by = doc.get("created_by", "")
    keywords = doc.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    embedding = doc.get("embedding")
    embedding_str = "[" + ",".join(map(str, embedding)) + "]" if embedding else None
    metadata = doc.get("metadata", {})
    return (
        doc_id, title, content, category, priority, is_pinned,
        start_date, end_date, status, created_by, keywords,
        embedding_str, PsycopgJson(metadata) if metadata else None
    )


def load_teddycard_data(conn: psycopg2_connection):
    """테디카드 데이터 적재"""
    print("\n" + "=" * 60)
    print("[7/9] 테디카드 데이터 적재")
    print("=" * 60)
    
    # 데이터 파일 찾기
    service_guides_file = find_data_file("teddycard_service_guides_with_embeddings.json")
    card_products_file = find_data_file("teddycard_card_products_with_embeddings.json")
    notices_file = find_data_file("teddycard_notices_with_embeddings.json")
    
    if not all([service_guides_file, card_products_file, notices_file]):
        print("[ERROR] 일부 데이터 파일을 찾을 수 없습니다.")
        if not service_guides_file:
            print(f"  - teddycard_service_guides_with_embeddings.json")
            print(f"    찾은 경로: {DATA_DIR_PROD} 또는 {DATA_DIR_DEV}")
        if not card_products_file:
            print(f"  - teddycard_card_products_with_embeddings.json")
            print(f"    찾은 경로: {DATA_DIR_PROD} 또는 {DATA_DIR_DEV}")
        if not notices_file:
            print(f"  - teddycard_notices_with_embeddings.json")
            print(f"    찾은 경로: {DATA_DIR_PROD} 또는 {DATA_DIR_DEV}")
        print("[INFO] 데이터 파일이 없으면 --skip-teddycard 옵션을 사용하여 건너뛸 수 있습니다.")
        return False
    
    # 이미 데이터가 있는지 확인 (service_guide_documents로 확인)
    has_data, count = check_table_has_data(conn, "service_guide_documents")
    if has_data:
        print(f"[INFO] service_guide_documents 테이블에 이미 데이터가 있습니다. (건수: {count}건)")
        has_card_data, card_count = check_table_has_data(conn, "card_products")
        has_notice_data, notice_count = check_table_has_data(conn, "notices")
        print(f"[INFO] card_products: {card_count}건, notices: {notice_count}건")
        print(f"[INFO] 테디카드 데이터가 이미 적재되어 있습니다. - 적재 스킵")
        return True
    
    cursor = conn.cursor()
    
    # service_guide_documents 적재
    print("[INFO] service_guide_documents 적재 중...")
    with open(service_guides_file, 'r', encoding='utf-8') as f:
        service_guides = json.load(f)
    
    insert_service_guide = """
        INSERT INTO service_guide_documents (
            id, document_type, category, title, content, keywords,
            embedding, metadata, document_source, priority,
            usage_count, last_used, structured
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (id) DO UPDATE SET
            document_type = EXCLUDED.document_type,
            category = EXCLUDED.category,
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            keywords = EXCLUDED.keywords,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            document_source = EXCLUDED.document_source,
            priority = EXCLUDED.priority,
            structured = EXCLUDED.structured,
            updated_at = NOW()
    """
    
    service_guide_batch = [map_service_guide_data(doc) for doc in service_guides]
    execute_batch(cursor, insert_service_guide, service_guide_batch, page_size=BATCH_SIZE)
    conn.commit()
    print(f"[INFO] service_guide_documents 적재 완료: {len(service_guide_batch)}개")
    
    # card_products 적재
    print("[INFO] card_products 적재 중...")
    with open(card_products_file, 'r', encoding='utf-8') as f:
        card_products = json.load(f)
    
    insert_card_product = """
        INSERT INTO card_products (
            id, name, card_type, brand, annual_fee_domestic, annual_fee_global,
            performance_condition, main_benefits, status, metadata, structured
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            card_type = EXCLUDED.card_type,
            brand = EXCLUDED.brand,
            annual_fee_domestic = EXCLUDED.annual_fee_domestic,
            annual_fee_global = EXCLUDED.annual_fee_global,
            performance_condition = EXCLUDED.performance_condition,
            main_benefits = EXCLUDED.main_benefits,
            status = EXCLUDED.status,
            metadata = EXCLUDED.metadata,
            structured = EXCLUDED.structured,
            updated_at = NOW()
    """
    
    card_product_batch = [map_card_product_data(doc) for doc in card_products]
    execute_batch(cursor, insert_card_product, card_product_batch, page_size=BATCH_SIZE)
    conn.commit()
    print(f"[INFO] card_products 적재 완료: {len(card_product_batch)}개")
    
    # notices 적재
    print("[INFO] notices 적재 중...")
    with open(notices_file, 'r', encoding='utf-8') as f:
        notices = json.load(f)
    
    insert_notice = """
        INSERT INTO notices (
            id, title, content, category, priority, is_pinned,
            start_date, end_date, status, created_by, keywords, embedding, metadata
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            category = EXCLUDED.category,
            priority = EXCLUDED.priority,
            is_pinned = EXCLUDED.is_pinned,
            start_date = EXCLUDED.start_date,
            end_date = EXCLUDED.end_date,
            status = EXCLUDED.status,
            keywords = EXCLUDED.keywords,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
    """
    
    notice_batch = [map_notice_data(doc) for doc in notices]
    execute_batch(cursor, insert_notice, notice_batch, page_size=BATCH_SIZE)
    conn.commit()
    print(f"[INFO] notices 적재 완료: {len(notice_batch)}개")
    
    cursor.close()
    return True


def check_required_files() -> Tuple[bool, List[str]]:
    """필수 파일 존재 여부 확인"""
    required_files = [
        SCRIPTS_DIR / "db_setup.sql",
        SCRIPTS_DIR / "02_setup_tedicard_tables.sql",  # 통합본 사용
        SCRIPTS_DIR / "03_setup_keyword_dictionary.sql",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    return len(missing_files) == 0, missing_files


def check_data_files() -> Tuple[bool, Dict[str, Optional[Path]]]:
    """데이터 파일 존재 여부 확인"""
    data_files = {
        "employees": None,
        "hana_rdb": None,
        "hana_vectordb": None,
        "service_guides": None,
        "card_products": None,
        "notices": None,
        "keywords_dict": None
    }
    
    # 상담사 데이터 파일 확인
    if EMPLOYEES_DATA_FILE.exists():
        data_files["employees"] = EMPLOYEES_DATA_FILE
    
    # 하나카드 데이터 파일 확인
    if HANA_RDB_METADATA_FILE.exists():
        data_files["hana_rdb"] = HANA_RDB_METADATA_FILE
    if HANA_VECTORDB_FILE.exists():
        data_files["hana_vectordb"] = HANA_VECTORDB_FILE
    
    # 테디카드 데이터 파일 확인
    service_guides_file = find_data_file("teddycard_service_guides_with_embeddings.json")
    card_products_file = find_data_file("teddycard_card_products_with_embeddings.json")
    notices_file = find_data_file("teddycard_notices_with_embeddings.json")
    
    data_files["service_guides"] = service_guides_file
    data_files["card_products"] = card_products_file
    data_files["notices"] = notices_file
    
    # 키워드 사전 파일 확인
    keywords_file = find_keywords_dict_file()
    data_files["keywords_dict"] = keywords_file
    
    all_exist = all([
        data_files["employees"],
        data_files["hana_rdb"],
        service_guides_file,
        card_products_file,
        notices_file,
        keywords_file
    ])
    
    return all_exist, data_files


def print_checklist(skip_schema: bool, skip_keywords: bool, skip_teddycard: bool, skip_employees: bool = False, skip_hana: bool = False):
    """실행 전 체크리스트 출력"""
    print("\n" + "=" * 60)
    print("실행 전 체크리스트")
    print("=" * 60)
    
    # 필수 파일 확인
    files_ok, missing_files = check_required_files()
    if files_ok:
        print("✅ 필수 SQL 파일: 모두 존재")
    else:
        print("❌ 필수 SQL 파일 누락:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    # 데이터 파일 확인
    if not skip_employees or not skip_hana or not skip_keywords or not skip_teddycard:
        data_ok, data_files = check_data_files()
        if data_ok:
            print("✅ 데이터 파일: 모두 존재")
            if not skip_employees:
                print(f"   - Employees: {data_files['employees']}")
            if not skip_hana:
                print(f"   - Hana RDB: {data_files['hana_rdb']}")
                print(f"   - Hana VectorDB: {data_files['hana_vectordb']}")
            if not skip_teddycard:
                print(f"   - Service Guides: {data_files['service_guides']}")
                print(f"   - Card Products: {data_files['card_products']}")
                print(f"   - Notices: {data_files['notices']}")
            if not skip_keywords:
                print(f"   - Keywords Dict: {data_files['keywords_dict']}")
        else:
            print("⚠️ 데이터 파일 일부 누락:")
            if not skip_employees and not data_files['employees']:
                print("   - employeesData.json")
            if not skip_hana:
                if not data_files['hana_rdb']:
                    print("   - hana_rdb_metadata.json")
                if not data_files['hana_vectordb']:
                    print("   - hana_vectordb_with_embeddings.json")
            if not skip_teddycard:
                if not data_files['service_guides']:
                    print("   - teddycard_service_guides_with_embeddings.json")
                if not data_files['card_products']:
                    print("   - teddycard_card_products_with_embeddings.json")
                if not data_files['notices']:
                    print("   - teddycard_notices_with_embeddings.json")
            if not skip_keywords and not data_files['keywords_dict']:
                print("   - keywords_dict_*.json")
            if not skip_employees and not data_files['employees']:
                print("   ⚠️ 상담사 데이터 파일이 없으면 상담사 적재를 건너뜁니다.")
            if not skip_hana and not data_files['hana_rdb']:
                print("   ⚠️ 하나카드 데이터 파일이 없으면 하나카드 적재를 건너뜁니다.")
            if not skip_keywords and not data_files['keywords_dict']:
                print("   ⚠️ 키워드 사전 파일이 없으면 키워드 적재를 건너뜁니다.")
            if not skip_teddycard and not all([data_files['service_guides'], data_files['card_products'], data_files['notices']]):
                print("   ⚠️ 테디카드 데이터 파일이 없으면 테디카드 적재를 건너뜁니다.")
    
    # 환경 변수 확인
    env_vars_ok = all([DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME])
    if env_vars_ok:
        print("✅ 환경 변수: 모두 설정됨")
        print(f"   - DB: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    else:
        print("❌ 환경 변수: 일부 누락")
        return False
    
    print("=" * 60)
    return True


def verify_load(conn: psycopg2_connection):
    """데이터 적재 검증 (통합 검증)"""
    print("\n" + "=" * 60)
    print("[9/9] 데이터 적재 검증")
    print("=" * 60)
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # 1. 테이블 존재 확인
    print("\n[1/3] 테이블 존재 확인")
    expected_tables = [
        'employees', 'consultations', 'consultation_documents',
        'service_guide_documents', 'card_products', 'notices',
        'keyword_dictionary', 'keyword_synonyms'
    ]
    
    existing_tables = []
    missing_tables = []
    
    for table in expected_tables:
        try:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = %s
            """, (table,))
            if cursor.fetchone():
                existing_tables.append(table)
            else:
                missing_tables.append(table)
        except Exception as e:
            missing_tables.append(table)
            print(f"  ❌ {table}: 확인 실패 - {e}")
    
    if missing_tables:
        print(f"  ⚠️ 누락된 테이블: {', '.join(missing_tables)}")
    else:
        print(f"  ✅ 모든 테이블 존재: {len(existing_tables)}개")
    
    # 2. 데이터 개수 확인 및 검증
    print("\n[2/3] 데이터 개수 확인 및 검증")
    tables_with_data = []
    tables_without_data = []
    tables_with_warnings = []
    
    # 예상 최소 개수 (데이터가 적재되었는지 확인용)
    expected_min_counts = {
        'employees': 10,  # 기본 상담사만 있으면 1건이므로 10건 미만이면 경고
        'consultations': 1,  # 최소 1건은 있어야 함
        'consultation_documents': 1,
        'service_guide_documents': 1,
        'card_products': 1,
        'notices': 1,
        'keyword_dictionary': 1,
        'keyword_synonyms': 1
    }
    
    for table in existing_tables:
        try:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            result = cursor.fetchone()
            count = result['count'] if result else 0
            expected_min = expected_min_counts.get(table, 1)
            
            if count > 0:
                tables_with_data.append((table, count))
                # 예상 개수와 비교
                if table == 'employees' and count < expected_min:
                    print(f"  ⚠️ {table}: {count:,}건 (예상 최소: {expected_min}건) - 기본 상담사만 있을 수 있음")
                    tables_with_warnings.append((table, count, expected_min))
                    
                    # 기본 상담사만 있는지 확인 (EMP-TEDDY-DEFAULT만 확인)
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM employees 
                        WHERE id = 'EMP-TEDDY-DEFAULT'
                           OR email LIKE '%default%teddycard%'
                    """)
                    default_count = cursor.fetchone()[0]
                    if default_count == count:
                        print(f"    ⚠️ 경고: 기본 상담사만 있습니다. employeesData.json 데이터를 적재해야 합니다.")
                elif count < expected_min:
                    print(f"  ⚠️ {table}: {count:,}건 (예상 최소: {expected_min}건) - 데이터가 부족할 수 있음")
                    tables_with_warnings.append((table, count, expected_min))
                else:
                    print(f"  ✅ {table}: {count:,}건")
            else:
                tables_without_data.append(table)
                print(f"  ⚠️ {table}: 0건 (데이터 없음)")
        except Exception as e:
            print(f"  ❌ {table}: 오류 - {e}")
    
    # 3. 스키마 확인 (pgvector 확장, 주요 인덱스)
    print("\n[3/3] 스키마 확인")
    
    # pgvector 확장 확인
    try:
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        if cursor.fetchone():
            print("  ✅ pgvector 확장: 설치됨")
        else:
            print("  ❌ pgvector 확장: 설치되지 않음")
    except Exception as e:
        print(f"  ❌ pgvector 확장 확인 실패: {e}")
    
    # 주요 인덱스 확인 (임베딩 인덱스)
    vector_indexes = [
        ('consultation_documents', 'idx_consultation_documents_embedding_hnsw'),
        ('service_guide_documents', 'idx_service_guide_documents_embedding_hnsw'),
        ('notices', 'idx_notices_embedding_hnsw')
    ]
    
    for table_name, index_name in vector_indexes:
        if table_name in existing_tables:
            try:
                cursor.execute("""
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE schemaname = 'public' AND indexname = %s
                """, (index_name,))
                if cursor.fetchone():
                    print(f"  ✅ {index_name}: 존재")
                else:
                    print(f"  ⚠️ {index_name}: 없음 (성능 저하 가능)")
            except Exception as e:
                print(f"  ❌ {index_name} 확인 실패: {e}")
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("검증 요약")
    print("=" * 60)
    print(f"테이블: {len(existing_tables)}/{len(expected_tables)}개 존재")
    print(f"데이터 적재된 테이블: {len(tables_with_data)}개")
    if tables_without_data:
        print(f"⚠️ 데이터 없는 테이블: {', '.join(tables_without_data)}")
    if tables_with_warnings:
        print(f"⚠️ 데이터 부족한 테이블:")
        for table, count, expected_min in tables_with_warnings:
            print(f"   - {table}: {count}건 (예상 최소: {expected_min}건)")
    
    if missing_tables:
        print(f"\n⚠️ 주의: 누락된 테이블이 있습니다. 스키마 생성이 완료되지 않았을 수 있습니다.")
    
    # 검증 통과 여부
    verification_passed = (len(missing_tables) == 0 and len(tables_without_data) == 0 and len(tables_with_warnings) == 0)
    if verification_passed:
        print(f"\n✅ 검증 통과: 모든 테이블과 데이터가 정상적으로 적재되었습니다.")
    else:
        print(f"\n⚠️ 검증 경고: 일부 테이블에 데이터가 부족하거나 누락되었습니다. 위 내용을 확인해주세요.")
    
    cursor.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='CALL:ACT 통합 DB 설정 및 데이터 적재')
    parser.add_argument('--skip-schema', action='store_true', help='스키마 생성 건너뛰기')
    parser.add_argument('--skip-employees', action='store_true', help='상담사 데이터 적재 건너뛰기')
    parser.add_argument('--skip-hana', action='store_true', help='하나카드 데이터 적재 건너뛰기')
    parser.add_argument('--skip-keywords', action='store_true', help='키워드 사전 적재 건너뛰기')
    parser.add_argument('--skip-teddycard', action='store_true', help='테디카드 데이터 적재 건너뛰기')
    parser.add_argument('--verify-only', action='store_true', help='검증만 실행')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CALL:ACT 통합 DB 설정 및 데이터 적재 스크립트")
    print("=" * 60)
    print(f"[INFO] Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    # 실행 전 체크리스트 확인
    if not print_checklist(args.skip_schema, args.skip_keywords, args.skip_teddycard, args.skip_employees, args.skip_hana):
        print("\n[ERROR] 체크리스트 확인 실패. 위 내용을 확인해주세요.")
        sys.exit(1)
    
    print()
    
    # DB 연결
    conn = connect_db()
    
    try:
        if args.verify_only:
            verify_load(conn)
        else:
            # 1. 기본 스키마 생성
            if not args.skip_schema:
                setup_basic_schema(conn)
                setup_teddycard_tables(conn)
                setup_keyword_dictionary_tables(conn)
            
            # 2. 상담사 데이터 적재 (하나카드 데이터 적재 전 필요)
            if not args.skip_employees:
                load_employees_data(conn)
            
            # 3. 하나카드 데이터 적재 (상담사 데이터 적재 후)
            if not args.skip_hana:
                load_hana_data(conn)
                # 하나카드 데이터 적재 후 실제 성과 지표 업데이트
                update_employee_performance(conn)
            
            # 4. 키워드 사전 적재
            if not args.skip_keywords:
                load_keyword_dictionary(conn)
            
            # 5. 테디카드 데이터 적재
            if not args.skip_teddycard:
                load_teddycard_data(conn)
            
            # 6. 검증
            verify_load(conn)
        
        print("\n" + "=" * 60)
        print("[SUCCESS] 모든 작업이 완료되었습니다!")
        print("[9/9] 완료")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 작업 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
