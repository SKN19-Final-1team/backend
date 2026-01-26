"""
통계 업데이트 모듈

상담사 성과 지표 업데이트, 고객별 상담 통계 업데이트
"""

import psycopg2
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, RealDictCursor

from . import BATCH_SIZE


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


def update_employee_performance(conn: psycopg2_connection):
    """DB에서 실제 상담 데이터를 기반으로 employees 테이블의 성과 지표 업데이트

    - consultations: agent_id별 실제 상담 건수
    - fcr: First Call Resolution 비율 (0-100)
    - avgTime: 평균 상담 시간 ("MM:SS" 형식)
    - rank: 성과 순위 (consultations DESC, fcr DESC, avgTime ASC 기준)
    """
    print("\n" + "=" * 60)
    print("[6/12] 상담사 성과 지표 업데이트 (DB 실제 데이터 기반)")
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


def update_customer_consultation_stats(conn: psycopg2_connection):
    """고객별 상담 통계 업데이트 (상담 데이터 적재 후 실행)"""
    print("\n" + "=" * 60)
    print("[6-1/12] 고객별 상담 통계 업데이트")
    print("=" * 60)

    cursor = conn.cursor()

    try:
        # 1. 고객별 상담 건수, FCR 건수, 마지막 상담일 업데이트
        print("[INFO] 고객별 상담 통계 계산 중 (total_consultations + resolved_first_call)...")

        cursor.execute("""
            UPDATE customers c
            SET
                total_consultations = stats.total_count,
                resolved_first_call = stats.fcr_count,
                last_consultation_date = stats.last_date,
                updated_at = NOW()
            FROM (
                SELECT
                    customer_id,
                    COUNT(*) as total_count,
                    COUNT(*) FILTER (WHERE fcr = TRUE) as fcr_count,
                    MAX(call_date) as last_date
                FROM consultations
                GROUP BY customer_id
            ) stats
            WHERE c.id = stats.customer_id
        """)

        updated_count = cursor.rowcount
        conn.commit()
        print(f"[INFO] 고객 상담 통계 업데이트 완료: {updated_count}명")

        cursor.close()
        return True

    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] 고객 상담 통계 업데이트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
