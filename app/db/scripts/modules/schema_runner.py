"""
스키마 생성 모듈

기본 DB 스키마, 테디카드, 키워드, 고객, 시뮬레이션, 감사 로그 테이블 생성
카테고리 매핑 데이터 적재
"""

import re
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch

from . import (
    SCRIPTS_DIR, load_sql_file, execute_sql_script,
    check_table_has_data, CATEGORY_MAPPINGS
)


def setup_basic_schema(conn: psycopg2_connection):
    """기본 DB 스키마 생성"""
    print("\n" + "=" * 60)
    print("[1/12] 기본 DB 스키마 생성")
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
    print("[2/12] 테디카드 테이블 생성 (통합본)")
    print("=" * 60)

    sql_file = SCRIPTS_DIR / "02_setup_tedicard_tables.sql"
    sql_script = load_sql_file(sql_file)
    execute_sql_script(conn, sql_script, "테디카드 테이블 생성 및 수정")


def setup_keyword_dictionary_tables(conn: psycopg2_connection):
    """키워드 사전 테이블 생성"""
    print("\n" + "=" * 60)
    print("[3/12] 키워드 사전 테이블 생성")
    print("=" * 60)

    sql_file = SCRIPTS_DIR / "03_setup_keyword_dictionary.sql"
    sql_script = load_sql_file(sql_file)
    execute_sql_script(conn, sql_script, "키워드 사전 테이블 생성")


def setup_persona_types_table(conn: psycopg2_connection):
    """페르소나 유형 테이블 생성"""
    print("\n" + "=" * 60)
    print("[3-1/12] 페르소나 유형 테이블 생성")
    print("=" * 60)

    sql_file = SCRIPTS_DIR / "07a_setup_persona_types_table.sql"
    sql_script = load_sql_file(sql_file)
    execute_sql_script(conn, sql_script, "페르소나 유형 테이블 생성")


def setup_customers_table(conn: psycopg2_connection):
    """고객 테이블 생성"""
    print("\n" + "=" * 60)
    print("[3-2/12] 고객 테이블 생성")
    print("=" * 60)

    sql_file = SCRIPTS_DIR / "07_setup_customers_table.sql"
    sql_script = load_sql_file(sql_file)
    execute_sql_script(conn, sql_script, "고객 테이블 생성")


def setup_simulation_tables(conn: psycopg2_connection):
    """시뮬레이션 교육 테이블 생성"""
    print("\n" + "=" * 60)
    print("[3-3/12] 시뮬레이션 교육 테이블 생성")
    print("=" * 60)

    sql_file = SCRIPTS_DIR / "10_setup_simulation_tables.sql"
    if sql_file.exists():
        sql_script = load_sql_file(sql_file)
        execute_sql_script(conn, sql_script, "시뮬레이션 교육 테이블 생성")
    else:
        print(f"[WARNING] {sql_file} 파일이 없습니다. 건너뜁니다.")


def setup_audit_tables(conn: psycopg2_connection):
    """감사 로그 테이블 생성 (녹취 다운로드 이력 포함)"""
    print("\n" + "=" * 60)
    print("[3-4/12] 감사 로그 테이블 생성")
    print("=" * 60)

    sql_file = SCRIPTS_DIR / "11_setup_audit_tables.sql"
    if sql_file.exists():
        sql_script = load_sql_file(sql_file)
        execute_sql_script(conn, sql_script, "감사 로그 테이블 생성")
    else:
        print(f"[WARNING] {sql_file} 파일이 없습니다. 건너뜁니다.")


def setup_consultation_relevance(conn: psycopg2_connection):
    """상담 이력 유의미성 함수/뷰 생성"""
    print("\n" + "=" * 60)
    print("[3-5/12] 상담 이력 유의미성 함수/뷰 생성")
    print("=" * 60)

    sql_file = SCRIPTS_DIR / "12_setup_consultation_relevance.sql"
    if sql_file.exists():
        sql_script = load_sql_file(sql_file)
        execute_sql_script(conn, sql_script, "상담 이력 유의미성 함수/뷰 생성")
    else:
        print(f"[WARNING] {sql_file} 파일이 없습니다. 건너뜁니다.")


def load_category_mappings(conn: psycopg2_connection):
    """카테고리 매핑 테이블 데이터 적재 (57개 원본 → 8개 대분류 + 15개 중분류)"""
    print("\n" + "=" * 60)
    print("[3-6/12] 카테고리 매핑 데이터 적재")
    print("=" * 60)

    # 이미 데이터가 있는지 확인
    has_data, count = check_table_has_data(conn, "category_mappings")
    if has_data and count >= len(CATEGORY_MAPPINGS):
        print(f"[INFO] category_mappings 테이블에 이미 데이터가 있습니다. ({count}건) - 적재 스킵")
        return

    cursor = conn.cursor()

    try:
        insert_mapping = """
            INSERT INTO category_mappings (category_raw, category_main, category_sub, keywords)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (category_raw) DO UPDATE SET
                category_main = EXCLUDED.category_main,
                category_sub = EXCLUDED.category_sub,
                keywords = EXCLUDED.keywords
        """

        mapping_batch = []
        for raw_cat, (main_cat, sub_cat) in CATEGORY_MAPPINGS.items():
            # 키워드 추출 (원본 카테고리에서 '/', ' ' 로 분리)
            keywords = [kw.strip() for kw in re.split(r'[/\s]+', raw_cat) if kw.strip()]
            mapping_batch.append((raw_cat, main_cat, sub_cat, keywords))

        if mapping_batch:
            execute_batch(cursor, insert_mapping, mapping_batch, page_size=50)
            conn.commit()
            print(f"[INFO] category_mappings 적재 완료: {len(mapping_batch)}건")

            # 통계 출력
            cursor.execute("""
                SELECT category_main, COUNT(*)
                FROM category_mappings
                GROUP BY category_main
                ORDER BY COUNT(*) DESC
            """)
            stats = cursor.fetchall()
            print("[INFO] 대분류별 매핑 건수:")
            for main_cat, cnt in stats:
                print(f"  - {main_cat}: {cnt}건")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] 카테고리 매핑 적재 실패: {e}")
        raise
    finally:
        cursor.close()


def run_all_schemas(conn: psycopg2_connection):
    """모든 스키마 생성 + 카테고리 매핑 적재"""
    setup_basic_schema(conn)
    setup_teddycard_tables(conn)
    setup_keyword_dictionary_tables(conn)
    setup_persona_types_table(conn)
    setup_customers_table(conn)
    setup_simulation_tables(conn)
    setup_audit_tables(conn)
    setup_consultation_relevance(conn)
    load_category_mappings(conn)
