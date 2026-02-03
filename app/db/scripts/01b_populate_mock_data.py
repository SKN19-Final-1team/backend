"""
CALL:ACT Mock 데이터 생성 오케스트레이션 스크립트

01a_setup_callact_db.py 실행 후 바로 실행하여 모든 테이블의 확장 필드를 채웁니다.

실행 순서:
1. Customers 5타입 매핑 및 type_history 생성
2. Consultations 확장 필드 생성 (transcript, ai_summary 등)
3. Usage 통계 데이터 생성 (service_guide_documents, consultation_documents)
4. Simulation 데이터 생성 (점수, 녹취, AI 반응)
5. Keyword 동의어/변형어 데이터 적용

멱등성: 모든 모듈은 RANDOM_SEED=42를 사용하여 동일한 결과 보장
"""

import sys
import time
from datetime import datetime

# 모듈 임포트
from modules.connect_db import connect_db
from modules.populate_customers_types import populate_customers_types
from modules.populate_extended_fields import populate_extended_fields
from modules.populate_usage_stats import populate_usage_stats
from modules.populate_simulation_data import populate_simulation_data
from modules.populate_keyword_extensions import populate_keyword_extensions


def run_all_populate_scripts():
    """모든 데이터 생성 스크립트 실행"""
    print("=" * 70)
    print("CALL:ACT Mock 데이터 생성")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print("이 스크립트는 01a_setup_callact_db.py 실행 후 사용합니다.")
    print("모든 테이블의 확장 필드에 현실적인 Mock 데이터를 채웁니다.")
    print("RANDOM_SEED=42로 멱등성이 보장됩니다.")
    print()

    start_time = time.time()

    # DB 연결
    print("[DB] 데이터베이스 연결 중...")
    conn = connect_db()
    if not conn:
        print("[ERROR] 데이터베이스 연결 실패")
        return False

    print("[DB] 연결 성공")
    print()

    results = {}

    try:
        # 1. Customers 5타입 매핑
        print("=" * 70)
        print("[1/5] Customers 5타입 매핑 시작...")
        results['customers_types'] = populate_customers_types(conn)

        # 2. Consultations 확장 필드
        print()
        print("=" * 70)
        print("[2/5] Consultations 확장 필드 생성 시작...")
        results['extended_fields'] = populate_extended_fields(conn)

        # 3. Usage 통계
        print()
        print("=" * 70)
        print("[3/5] Usage 통계 데이터 생성 시작...")
        results['usage_stats'] = populate_usage_stats(conn)

        # 4. Simulation 데이터
        print()
        print("=" * 70)
        print("[4/5] Simulation 데이터 생성 시작...")
        results['simulation_data'] = populate_simulation_data(conn)

        # 5. Keyword 동의어/변형어
        print()
        print("=" * 70)
        print("[5/5] Keyword 동의어/변형어 적용 시작...")
        results['keyword_extensions'] = populate_keyword_extensions(conn)

    except Exception as e:
        print(f"\n[ERROR] 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        conn.close()
        return False

    finally:
        conn.close()
        print("\n[DB] 연결 종료")

    # 결과 요약
    elapsed_time = time.time() - start_time

    print()
    print("=" * 70)
    print("실행 결과 요약")
    print("=" * 70)

    all_success = True
    for name, success in results.items():
        status = "[OK] 성공" if success else "[FAIL] 실패"
        print(f"  {name}: {status}")
        if not success:
            all_success = False

    print()
    print(f"총 소요 시간: {elapsed_time:.2f}초")
    print()

    if all_success:
        print("=" * 70)
        print("모든 Mock 데이터 생성 완료!")
        print("=" * 70)
    else:
        print("=" * 70)
        print("[WARNING] 일부 작업이 실패했습니다. 로그를 확인하세요.")
        print("=" * 70)

    return all_success


if __name__ == "__main__":
    success = run_all_populate_scripts()
    sys.exit(0 if success else 1)
