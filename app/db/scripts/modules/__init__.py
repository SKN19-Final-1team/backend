"""
CALL:ACT DB 설정 모듈 패키지

공통 유틸리티, DB 연결, 상수 정의
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, Json as PsycopgJson, RealDictCursor
from tqdm import tqdm

# ==============================================================================
# 경로 설정
# ==============================================================================

SCRIPTS_DIR = Path(__file__).parent.parent  # scripts/
sys.path.insert(0, str(SCRIPTS_DIR))

from config import (
    PROJECT_ROOT,
    TEDDY_DATA_DIR_PROD, TEDDY_DATA_DIR_DEV,
    KEYWORDS_DICT_DIR_PROD, KEYWORDS_DICT_DIR_DEV,
    HANA_RDB_METADATA_FILE, HANA_VECTORDB_FILE,
    EMPLOYEES_DATA_FILE, CUSTOMERS_DATA_FILE
)

# 상수 (기존 호환성 유지)
BASE_DIR = PROJECT_ROOT

# 데이터 파일 경로 (config.py에서 가져온 값 사용)
DATA_DIR_PROD = TEDDY_DATA_DIR_PROD
DATA_DIR_DEV = TEDDY_DATA_DIR_DEV

# ==============================================================================
# 환경 변수
# ==============================================================================

load_dotenv(SCRIPTS_DIR / '.env', override=False)
load_dotenv(SCRIPTS_DIR.parent.parent.parent / '.env', override=False)

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432")) if os.getenv("DB_PORT") else None
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
BATCH_SIZE = int(os.getenv("DB_LOAD_BATCH_SIZE", "100"))
COMMIT_INTERVAL = int(os.getenv("DB_COMMIT_INTERVAL", "500"))


# ==============================================================================
# 공통 유틸리티 함수
# ==============================================================================

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


# ==============================================================================
# 카테고리 매핑 시스템 (57개 원본 → 8개 대분류 + 15개 중분류)
# ==============================================================================

# 대분류별 담당 풀 크기 설정 (60명 기준 - 상담 업무 담당자)
MAIN_CATEGORY_POOL_SIZES = {
    "결제/승인": 12,      # ~1,559건
    "이용내역": 9,        # ~919건
    "한도": 7,            # ~564건
    "분실/도난": 6,       # ~400건
    "수수료/연체": 4,     # ~163건
    "포인트/혜택": 4,     # ~223건
    "정부지원": 3,        # ~167건
    "기타": None,         # ~1,538건 (전체 상담사 순차)
}

# 대분류 8개 (인입 기준)
MAIN_CATEGORIES = [
    "결제/승인", "이용내역", "한도", "분실/도난",
    "수수료/연체", "포인트/혜택", "정부지원", "기타"
]

# 중분류 15개 (행위 기반)
SUB_CATEGORIES = [
    "조회/안내", "신청/등록", "변경", "취소/해지", "처리/실행",
    "발급", "확인서", "배송", "즉시출금", "상향/증액",
    "이체/전환", "환급/반환", "정지/해제", "결제일", "기타"
]

# 정규화 규칙: 띄어쓰기 통일, 순서 통일
NORMALIZATION_RULES = {
    # 띄어쓰기 통일
    "가상 계좌": "가상계좌",
    "이용 내역": "이용내역",
    "결제 계좌": "결제계좌",
    "결제 일": "결제일",
    # 순서 통일 (대분류명과 일치)
    "도난/분실": "분실/도난",
}

# 마스킹 태그 치환
MASKING_REPLACEMENTS = {
    "[카드사명#1]": "테디카드",
    "[카드사명#2]": "테디카드",
    "[서비스명#1]": "테디페이",
    "[서비스명#2]": "테디페이",
    "[페이서비스#1]": "테디페이",
}

# 57개 카테고리 → (대분류, 중분류) 매핑 딕셔너리
CATEGORY_MAPPINGS = {
    # 분실/도난 (4개)
    "도난/분실 신청/해제": ("분실/도난", "정지/해제"),
    "분실/도난 신청/해제": ("분실/도난", "정지/해제"),

    # 한도 (3개)
    "한도상향 접수/처리": ("한도", "상향/증액"),
    "한도 안내": ("한도", "조회/안내"),
    "특별한도/임시한도/일시한도 안내/신청": ("한도", "상향/증액"),

    # 결제/승인 (10개)
    "선결제/즉시출금": ("결제/승인", "즉시출금"),
    "가상계좌 안내": ("결제/승인", "조회/안내"),
    "결제일 안내/변경": ("결제/승인", "결제일"),
    "테디카드 결제계좌 변경": ("결제/승인", "변경"),
    "결제대금 안내": ("결제/승인", "조회/안내"),
    "결제계좌 변경": ("결제/승인", "변경"),
    "선결제/즉시출금 안내": ("결제/승인", "즉시출금"),
    "연체대금 즉시출금": ("결제/승인", "즉시출금"),
    "매출 취소": ("결제/승인", "취소/해지"),
    "매출 취소 문의/요청/확인": ("결제/승인", "취소/해지"),

    # 이용내역 (5개)
    "이용내역 안내": ("이용내역", "조회/안내"),
    "이용내역 확인": ("이용내역", "조회/안내"),
    "이용내역확인": ("이용내역", "조회/안내"),
    "이용내역 변경": ("이용내역", "변경"),
    "이용내역 조회": ("이용내역", "조회/안내"),

    # 수수료/연체 (5개)
    "연회비 안내": ("수수료/연체", "조회/안내"),
    "연체 안내": ("수수료/연체", "조회/안내"),
    "수수료 안내": ("수수료/연체", "조회/안내"),
    "연체대금 안내": ("수수료/연체", "조회/안내"),
    "이자/수수료 안내": ("수수료/연체", "조회/안내"),

    # 포인트/혜택 (5개)
    "포인트/마일리지 안내": ("포인트/혜택", "조회/안내"),
    "포인트/마일리지 전환등록": ("포인트/혜택", "이체/전환"),
    "이벤트 안내": ("포인트/혜택", "조회/안내"),
    "혜택 안내": ("포인트/혜택", "조회/안내"),
    "포인트/마일리지 전환/등록": ("포인트/혜택", "이체/전환"),

    # 정부지원 (3개)
    "바우처 발급/조회/변경": ("정부지원", "발급"),
    "정부지원 안내": ("정부지원", "조회/안내"),
    "정부지원 신청": ("정부지원", "신청/등록"),

    # 기타 (22개)
    "이용방법 안내": ("기타", "조회/안내"),
    "서비스 이용방법 안내": ("기타", "조회/안내"),
    "카드해지 신청": ("기타", "취소/해지"),
    "카드재발급": ("기타", "발급"),
    "카드 재발급": ("기타", "발급"),
    "카드정지/해제": ("기타", "정지/해제"),
    "카드 정지/해제": ("기타", "정지/해제"),
    "카드배송 안내/변경": ("기타", "배송"),
    "소득공제 안내/확인서": ("기타", "확인서"),
    "카드분실신고": ("기타", "정지/해제"),
    "카드 분실신고": ("기타", "정지/해제"),
    "테디페이 이용방법 안내": ("기타", "조회/안내"),
    "테디페이 결제계좌 변경": ("기타", "변경"),
    "테디페이 결제 안내": ("기타", "조회/안내"),
    "테디페이 결제계좌 등록": ("기타", "신청/등록"),
    "전자금융 가입/변경": ("기타", "신청/등록"),
    "긴급 배송 신청": ("기타", "배송"),
    "주민등록번호 변경": ("기타", "변경"),
    "비밀번호 변경/발급": ("기타", "변경"),
    "교육비": ("기타", "기타"),
    "도시가스": ("기타", "기타"),
    "전화요금": ("기타", "기타"),
}

# 기존 호환성을 위한 별칭 (deprecated, 향후 제거 예정)
CATEGORY_TO_MAIN_CATEGORY = {k: v[0] for k, v in CATEGORY_MAPPINGS.items()}
