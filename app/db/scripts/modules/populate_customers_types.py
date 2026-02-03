"""
Customers 테이블 5타입 매핑 및 type_history 생성 모듈

- current_type_code: 5타입 (N1, N2, S1, S2, S3) 할당
- type_history: 최근 3건의 상담과 연결된 타입 이력 (JSONB)

멱등성: RANDOM_SEED=42로 동일한 결과 보장
"""

import random
from typing import List, Dict

from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, Json as PsycopgJson


RANDOM_SEED = 42

# 5타입 시스템
TYPE_CODES = ['N1', 'N2', 'S1', 'S2', 'S3']

# 타입별 분포 (N1: 50%, N2: 20%, S1: 10%, S2: 10%, S3: 10%)
TYPE_DISTRIBUTION = ['N1'] * 50 + ['N2'] * 20 + ['S1'] * 10 + ['S2'] * 10 + ['S3'] * 10


def populate_customers_types(conn: psycopg2_connection, batch_size: int = 500):
    """
    Customers 테이블의 current_type_code와 type_history 채우기

    type_history 구조 (최대 3건):
    [
        {"type_code": "N1", "consultation_id": "CS-...", "assigned_at": "2026-01-19"},
        {"type_code": "N2", "consultation_id": "CS-...", "assigned_at": "2026-01-20"},
        {"type_code": "N1", "consultation_id": "CS-...", "assigned_at": "2026-01-21"}
    ]
    """
    print("\n" + "=" * 60)
    print("[1/4] Customers 5타입 매핑 및 type_history 생성")
    print("=" * 60)

    cursor = conn.cursor()
    rng = random.Random(RANDOM_SEED)

    try:
        # 고객별 최근 3건의 상담 조회
        print("[INFO] 고객별 최근 상담 이력 조회 중...")
        cursor.execute("""
            WITH ranked_consultations AS (
                SELECT
                    customer_id,
                    id as consultation_id,
                    call_date,
                    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY call_date DESC, id DESC) as rn
                FROM consultations
            )
            SELECT customer_id, consultation_id, call_date
            FROM ranked_consultations
            WHERE rn <= 3
            ORDER BY customer_id, rn
        """)

        # 고객별 상담 이력 그룹화
        customer_consultations: Dict[str, List[Dict]] = {}
        for row in cursor.fetchall():
            customer_id = row[0]
            consultation_id = row[1]
            call_date = row[2]

            if customer_id not in customer_consultations:
                customer_consultations[customer_id] = []

            customer_consultations[customer_id].append({
                "consultation_id": consultation_id,
                "call_date": str(call_date) if call_date else None
            })

        print(f"[INFO] 상담 이력이 있는 고객: {len(customer_consultations)}명")

        # 고객 데이터 조회
        cursor.execute("""
            SELECT id, current_type_code, total_consultations
            FROM customers
            ORDER BY id
        """)
        rows = cursor.fetchall()
        total = len(rows)
        print(f"[INFO] 총 고객 수: {total}")
        print(f"[INFO] 랜덤 시드: {RANDOM_SEED}")

        update_query = """
            UPDATE customers SET
                current_type_code = %s,
                type_history = %s,
                updated_at = NOW()
            WHERE id = %s
        """

        update_batch = []
        null_count = 0

        for row in rows:
            customer_id = row[0]
            current_type = row[1]
            total_consultations = row[2] or 0

            # current_type_code가 NULL이거나 유효하지 않으면 새로 할당
            if current_type is None or current_type not in TYPE_CODES:
                current_type = rng.choice(TYPE_DISTRIBUTION)
                null_count += 1

            # type_history 생성 (최대 3건, 상담 ID와 연결)
            type_history = []
            consultations = customer_consultations.get(customer_id, [])

            if consultations:
                # 상담 이력이 있는 경우: 각 상담에 타입 할당
                prev_type = current_type
                for i, cons in enumerate(consultations):
                    # 가장 최근(첫 번째)은 현재 타입, 나머지는 확률적으로 변경
                    if i == 0:
                        assigned_type = current_type
                    else:
                        # 30% 확률로 다른 타입
                        if rng.random() < 0.3:
                            candidates = [t for t in TYPE_CODES if t != prev_type]
                            assigned_type = rng.choice(candidates)
                        else:
                            assigned_type = prev_type
                        prev_type = assigned_type

                    type_history.append({
                        "type_code": assigned_type,
                        "consultation_id": cons["consultation_id"],
                        "assigned_at": cons["call_date"]
                    })

            update_batch.append((
                current_type,
                PsycopgJson(type_history),
                customer_id
            ))

        # 배치 업데이트
        if update_batch:
            execute_batch(cursor, update_query, update_batch, page_size=batch_size)
            conn.commit()

        print(f"[INFO] NULL → 5타입 할당: {null_count}명")
        print(f"[INFO] type_history 생성: {total}명")

        # 결과 확인
        print("\n[INFO] 업데이트 후 current_type_code 분포:")
        cursor.execute("""
            SELECT current_type_code, COUNT(*)
            FROM customers
            GROUP BY current_type_code
            ORDER BY current_type_code
        """)
        for r in cursor.fetchall():
            pct = r[1] / total * 100
            print(f"  {r[0]}: {r[1]}명 ({pct:.1f}%)")

        # type_history 샘플 확인
        print("\n[INFO] type_history 샘플 (상담 많은 고객):")
        cursor.execute("""
            SELECT id, current_type_code, total_consultations, type_history
            FROM customers
            WHERE total_consultations > 5
            ORDER BY total_consultations DESC
            LIMIT 3
        """)
        for r in cursor.fetchall():
            history_preview = r[3][:2] if r[3] and len(r[3]) > 2 else r[3]
            print(f"  {r[0]}: type={r[1]}, consultations={r[2]}, history={history_preview}...")

        cursor.close()
        return True

    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] customers 타입 매핑 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    from connect_db import connect_db

    conn = connect_db()
    if conn:
        populate_customers_types(conn)
        conn.close()
