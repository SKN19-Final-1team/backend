"""
검증 모듈

데이터 적재 검증, 체크리스트, 필수 파일 확인
"""

import math
from typing import Dict, List, Optional, Tuple

from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import RealDictCursor

from . import (
    SCRIPTS_DIR,
    EMPLOYEES_DATA_FILE, CUSTOMERS_DATA_FILE,
    HANA_RDB_METADATA_FILE, HANA_VECTORDB_FILE,
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME,
    MAIN_CATEGORY_POOL_SIZES,
    check_table_has_data, find_data_file, find_keywords_dict_file
)
from .load_consultations import map_to_main_category


def check_required_files() -> Tuple[bool, List[str]]:
    """필수 파일 존재 여부 확인"""
    required_files = [
        SCRIPTS_DIR / "db_setup.sql",
        SCRIPTS_DIR / "02_setup_tedicard_tables.sql",
        SCRIPTS_DIR / "03_setup_keyword_dictionary.sql",
        SCRIPTS_DIR / "07a_setup_persona_types_table.sql",
        SCRIPTS_DIR / "07_setup_customers_table.sql",
    ]

    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))

    return len(missing_files) == 0, missing_files


def check_data_files() -> Tuple[bool, Dict[str, Optional[object]]]:
    """데이터 파일 존재 여부 확인"""
    data_files = {
        "employees": None,
        "customers": None,
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

    # 고객 데이터 파일 확인
    if CUSTOMERS_DATA_FILE.exists():
        data_files["customers"] = CUSTOMERS_DATA_FILE

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
        data_files["customers"],
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
                print(f"   - Customers: {data_files['customers']}")
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
            if not skip_employees and not data_files['customers']:
                print("   - customersData.json")
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
            if not skip_employees and not data_files['customers']:
                print("   ⚠️ 고객 데이터 파일이 없으면 고객 적재를 건너뜁니다.")
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
    print("[12/12] 데이터 적재 검증")
    print("=" * 60)

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # 1. 테이블 존재 확인
    print("\n[1/4] 테이블 존재 확인")
    expected_tables = [
        # 기본 테이블
        'employees', 'consultations', 'consultation_documents',
        # 테디카드 테이블
        'service_guide_documents', 'card_products', 'notices',
        # 키워드 사전
        'keyword_dictionary', 'keyword_synonyms',
        # 고객 및 페르소나
        'persona_types', 'customers',
        # 시뮬레이션 교육
        'simulation_scenarios', 'simulation_results', 'employee_learning_analytics',
        # 감사 로그
        'recording_download_logs', 'audit_logs'
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
    print("\n[2/4] 데이터 개수 확인 및 검증")
    tables_with_data = []
    tables_without_data = []
    tables_with_warnings = []

    expected_min_counts = {
        'employees': 10,
        'consultations': 1,
        'consultation_documents': 1,
        'service_guide_documents': 1,
        'card_products': 1,
        'notices': 1,
        'keyword_dictionary': 1,
        'keyword_synonyms': 1,
        'persona_types': 12,  # 12개 유형 (N1-N4, S1-S8)
        'customers': 0,
        'simulation_scenarios': 5,
        'simulation_results': 0,
        'employee_learning_analytics': 0,
        'recording_download_logs': 0,
        'audit_logs': 0
    }

    for table in existing_tables:
        try:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            result = cursor.fetchone()
            count = result['count'] if result else 0
            expected_min = expected_min_counts.get(table, 1)

            if count > 0:
                tables_with_data.append((table, count))
                if table == 'employees' and count < expected_min:
                    print(f"  ⚠️ {table}: {count:,}건 (예상 최소: {expected_min}건) - 기본 상담사만 있을 수 있음")
                    tables_with_warnings.append((table, count, expected_min))

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
    print("\n[3/4] 스키마 확인")

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

    # 4. 상담사별/대분류별 분포 확인
    print("\n[4/4] 상담사별/대분류별 분포 확인")

    if 'consultations' in existing_tables:
        try:
            # 상담사별 상담 건수 확인
            cursor.execute("""
                SELECT agent_id, COUNT(*) as count
                FROM consultations
                WHERE agent_id IS NOT NULL
                GROUP BY agent_id
                ORDER BY count DESC
            """)
            agent_counts = cursor.fetchall()

            if agent_counts:
                print(f"\n  상담사별 상담 건수 (총 {len(agent_counts)}명):")
                total_consultations = sum(row['count'] for row in agent_counts)
                avg_count = total_consultations / len(agent_counts) if agent_counts else 0

                # 표준편차 계산
                variance = sum((row['count'] - avg_count) ** 2 for row in agent_counts) / len(agent_counts) if agent_counts else 0
                std_dev = math.sqrt(variance)

                # 상위 5명, 하위 5명 출력
                top_5 = agent_counts[:5]
                bottom_5 = agent_counts[-5:] if len(agent_counts) >= 5 else agent_counts

                print(f"    평균: {avg_count:.1f}건/명, 표준편차: {std_dev:.1f}")
                print(f"    상위 5명:")
                for row in top_5:
                    print(f"      - {row['agent_id']}: {row['count']}건")
                if len(agent_counts) > 5:
                    print(f"    하위 5명:")
                    for row in bottom_5:
                        print(f"      - {row['agent_id']}: {row['count']}건")

                # 분산 경고
                if avg_count > 0 and std_dev / avg_count > 0.5:
                    print(f"    ⚠️ 경고: 상담사별 배분이 불균등합니다. (변동계수: {std_dev/avg_count:.2f})")
                else:
                    print(f"    ✅ 상담사별 배분이 비교적 균등합니다. (변동계수: {std_dev/avg_count:.2f})")
            else:
                print("  ⚠️ 상담사별 상담 데이터가 없습니다.")

            # 대분류별 상담 건수 확인
            cursor.execute("""
                SELECT category_raw, COUNT(*) as count
                FROM consultations
                WHERE category_raw IS NOT NULL
                GROUP BY category_raw
                ORDER BY count DESC
            """)
            category_counts = cursor.fetchall()

            if category_counts:
                print(f"\n  세부 카테고리별 상담 건수 (총 {len(category_counts)}개):")

                # 대분류별 집계
                main_category_counts = {}
                for row in category_counts:
                    main_cat = map_to_main_category(row['category_raw'])
                    if main_cat not in main_category_counts:
                        main_category_counts[main_cat] = 0
                    main_category_counts[main_cat] += row['count']

                print(f"\n  대분류별 상담 건수:")
                for main_cat in sorted(main_category_counts.keys(), key=lambda x: main_category_counts[x], reverse=True):
                    count = main_category_counts[main_cat]
                    print(f"    - {main_cat}: {count:,}건")

                # 대분류별 상담사 분포 확인
                print(f"\n  대분류별 상담사 분포:")
                for main_cat in sorted(main_category_counts.keys(), key=lambda x: main_category_counts[x], reverse=True):
                    related_categories = [row['category_raw'] for row in category_counts
                                        if map_to_main_category(row['category_raw']) == main_cat]

                    if related_categories:
                        placeholders = ','.join(['%s'] * len(related_categories))
                        cursor.execute(f"""
                            SELECT DISTINCT agent_id, COUNT(*) as count
                            FROM consultations
                            WHERE category_raw IN ({placeholders})
                            GROUP BY agent_id
                            ORDER BY count DESC
                        """, related_categories)
                        agent_dist = cursor.fetchall()

                        if agent_dist:
                            agent_ids = [row['agent_id'] for row in agent_dist]
                            expected_pool_size = MAIN_CATEGORY_POOL_SIZES.get(main_cat)
                            if expected_pool_size is None:
                                expected_pool_size = len(agent_ids)  # "기타"는 전체

                            print(f"    - {main_cat}: {len(agent_ids)}명 담당 (예상: {expected_pool_size}명)")
                            if abs(len(agent_ids) - expected_pool_size) > 2:
                                print(f"      ⚠️ 경고: 예상 풀 크기와 다릅니다.")
                            else:
                                print(f"      ✅ 풀 크기가 예상과 일치합니다.")
        except Exception as e:
            print(f"  ❌ 상담사별/대분류별 분포 확인 실패: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ⚠️ consultations 테이블이 없어 분포를 확인할 수 없습니다.")

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
    return verification_passed
