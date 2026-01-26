"""
CALL:ACT 통합 DB 설정 및 데이터 적재 스크립트 (오케스트레이터)

기능:
- DB 스키마 생성 (기본 테이블: employees, consultations, consultation_documents)
- 테디카드 테이블 생성 (service_guide_documents, card_products, notices)
- 키워드 사전 테이블 생성 (keyword_dictionary, keyword_synonyms)
- 고객 테이블 생성 (customers, persona_types)
- 시뮬레이션 교육 테이블 생성 (simulation_scenarios, simulation_results, employee_learning_analytics)
- 감사 로그 테이블 생성 (recording_download_logs, audit_logs)
- 상담 이력 유의미성 함수/뷰 생성
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

최종 수정일: 2026-01-26
"""

import argparse
import sys

from modules import connect_db, DB_HOST, DB_PORT, DB_NAME

from modules.schema_runner import run_all_schemas
from modules.load_employees import load_employees_data
from modules.load_customers import load_customers_data
from modules.load_consultations import load_hana_data
from modules.update_stats import update_employee_performance, update_customer_consultation_stats
from modules.load_keywords import load_keyword_dictionary
from modules.load_teddycard import load_teddycard_data
from modules.generate_mock import generate_mock_simulation_data, generate_mock_audit_data
from modules.verify import verify_load, print_checklist


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
                run_all_schemas(conn)

            # 2. 상담사 데이터 적재 (하나카드 데이터 적재 전 필요)
            if not args.skip_employees:
                load_employees_data(conn)

            # 2-1. 고객 데이터 적재
            load_customers_data(conn)

            # 3. 하나카드 데이터 적재 (상담사 데이터 적재 후)
            if not args.skip_hana:
                load_hana_data(conn)
                # 하나카드 데이터 적재 후 실제 성과 지표 업데이트
                update_employee_performance(conn)
                # 고객별 상담 통계 업데이트
                update_customer_consultation_stats(conn)

            # 4. 키워드 사전 적재
            if not args.skip_keywords:
                load_keyword_dictionary(conn)

            # 5. 테디카드 데이터 적재
            if not args.skip_teddycard:
                load_teddycard_data(conn)

            # 6. Mock 데이터 생성 (시뮬레이션, 감사 로그)
            generate_mock_simulation_data(conn)
            generate_mock_audit_data(conn)

            # 7. 검증
            verify_load(conn)

        print("\n" + "=" * 60)
        print("[SUCCESS] 모든 작업이 완료되었습니다!")
        print("[12/12] 완료")
        print("=" * 60)
        print("\n생성된 테이블:")
        print("  - 기본: employees, consultations, consultation_documents")
        print("  - 카테고리: category_mappings (57개 → 8대분류 + 15중분류)")
        print("  - 테디카드: service_guide_documents, card_products, notices")
        print("  - 키워드: keyword_dictionary, keyword_synonyms")
        print("  - 고객: customers, persona_types")
        print("  - 시뮬레이션: simulation_scenarios, simulation_results, employee_learning_analytics")
        print("  - 감사로그: recording_download_logs, audit_logs")
        print("\n생성된 함수/뷰:")
        print("  - fn_get_consultation_relevance(): 상담 이력 유의미성")
        print("  - fn_get_customer_persona_relevance(): 고객 성향 유효성")
        print("  - v_customer_guidance_info: 상담 가이던스용 고객 정보")
        print("  - v_suspicious_downloads: 이상 다운로드 감지")
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
