"""
Usage 통계 데이터 생성 모듈

service_guide_documents, consultation_documents 테이블의 usage 데이터 생성:
- usage_count: 사용 횟수
- last_used: 마지막 사용 시간
- effectiveness_score: 효과성 점수 (consultation_documents)

멱등성: RANDOM_SEED=42로 동일한 결과 보장
"""

import random
from datetime import datetime, timedelta

from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch


RANDOM_SEED = 42

# 기준 날짜 (2026-01-23)
BASE_DATE = datetime(2026, 1, 23, 18, 0, 0)


def populate_service_guide_usage(conn: psycopg2_connection, batch_size: int = 500):
    """service_guide_documents 테이블의 usage 데이터 채우기"""
    print("\n" + "-" * 50)
    print("service_guide_documents usage 데이터 생성")
    print("-" * 50)

    cursor = conn.cursor()
    rng = random.Random(RANDOM_SEED)

    try:
        cursor.execute("SELECT id FROM service_guide_documents ORDER BY id")
        rows = cursor.fetchall()
        total = len(rows)
        print(f"[INFO] 총 문서 수: {total}")

        update_query = """
            UPDATE service_guide_documents SET
                usage_count = %s,
                last_used = %s,
                updated_at = NOW()
            WHERE id = %s
        """

        update_batch = []

        for row in rows:
            doc_id = row[0]

            # usage_count: 0-500 (대부분 0-50, 일부 인기 문서는 높음)
            if rng.random() < 0.1:  # 10%는 인기 문서
                usage_count = rng.randint(100, 500)
            elif rng.random() < 0.3:  # 20%는 중간
                usage_count = rng.randint(20, 99)
            else:  # 70%는 낮음
                usage_count = rng.randint(0, 19)

            # last_used: 최근 30일 내 (usage_count > 0인 경우만)
            if usage_count > 0:
                days_ago = rng.randint(0, 30)
                hours_ago = rng.randint(0, 23)
                last_used = BASE_DATE - timedelta(days=days_ago, hours=hours_ago)
            else:
                last_used = None

            update_batch.append((usage_count, last_used, doc_id))

        if update_batch:
            execute_batch(cursor, update_query, update_batch, page_size=batch_size)
            conn.commit()

        print(f"[INFO] {total}건 업데이트 완료")

        # 통계 확인
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(last_used) as has_last_used,
                AVG(usage_count)::int as avg_usage,
                MAX(usage_count) as max_usage
            FROM service_guide_documents
        """)
        stats = cursor.fetchone()
        print(f"[INFO] 통계: last_used={stats[1]}/{stats[0]}, avg_usage={stats[2]}, max_usage={stats[3]}")

        cursor.close()
        return True

    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] service_guide_documents usage 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def populate_consultation_doc_usage(conn: psycopg2_connection, batch_size: int = 500):
    """consultation_documents 테이블의 usage/effectiveness 데이터 채우기"""
    print("\n" + "-" * 50)
    print("consultation_documents usage/effectiveness 데이터 생성")
    print("-" * 50)

    cursor = conn.cursor()
    rng = random.Random(RANDOM_SEED + 1)  # 다른 시드 사용

    try:
        cursor.execute("SELECT id FROM consultation_documents ORDER BY id")
        rows = cursor.fetchall()
        total = len(rows)
        print(f"[INFO] 총 문서 수: {total}")

        update_query = """
            UPDATE consultation_documents SET
                usage_count = %s,
                effectiveness_score = %s,
                last_used = %s
            WHERE id = %s
        """

        update_batch = []

        for row in rows:
            doc_id = row[0]

            # usage_count: 0-100
            usage_count = rng.randint(0, 100)

            # effectiveness_score: 0.00-1.00 (DECIMAL(3,2) 범위)
            # 사용량과 어느정도 상관관계 (usage_count를 0-1로 정규화 후 노이즈 추가)
            base_ratio = usage_count / 100.0
            noise = rng.uniform(-0.2, 0.3)
            effectiveness_score = round(max(0.0, min(1.0, base_ratio + noise)), 2)

            # last_used: 최근 30일 내
            if usage_count > 0:
                days_ago = rng.randint(0, 30)
                hours_ago = rng.randint(0, 23)
                last_used = BASE_DATE - timedelta(days=days_ago, hours=hours_ago)
            else:
                last_used = None

            update_batch.append((usage_count, effectiveness_score, last_used, doc_id))

        if update_batch:
            execute_batch(cursor, update_query, update_batch, page_size=batch_size)
            conn.commit()

        print(f"[INFO] {total}건 업데이트 완료")

        # 통계 확인
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(last_used) as has_last_used,
                ROUND(AVG(effectiveness_score)::numeric, 2) as avg_score
            FROM consultation_documents
        """)
        stats = cursor.fetchone()
        print(f"[INFO] 통계: last_used={stats[1]}/{stats[0]}, avg_effectiveness={stats[2]}")

        cursor.close()
        return True

    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] consultation_documents usage 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def populate_usage_stats(conn: psycopg2_connection):
    """모든 usage 통계 데이터 채우기"""
    print("\n" + "=" * 60)
    print("[3/4] Usage 통계 데이터 생성")
    print("=" * 60)
    print(f"[INFO] 랜덤 시드: {RANDOM_SEED}")

    success = True
    success = success and populate_service_guide_usage(conn)
    success = success and populate_consultation_doc_usage(conn)

    return success


if __name__ == "__main__":
    from connect_db import connect_db

    conn = connect_db()
    if conn:
        populate_usage_stats(conn)
        conn.close()
