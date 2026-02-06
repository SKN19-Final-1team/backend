"""
Trend 계산 모듈

employees.trend와 frequent_inquiries.trend를 계산하는 함수
- 최초 DB 적재 시 호출
- 매일 12시 배치 Job에서 호출

계산 로직:
- 이번 주 일평균 vs 저번 주 일평균 비교 (영업일 수 보정)
- 5% 이상 증가: 'up', 5% 이상 감소: 'down', 그 외: 'same'
"""

from datetime import datetime, timedelta
from psycopg2.extensions import connection as psycopg2_connection


def calculate_employee_trends(conn: psycopg2_connection):
    """
    상담사별 성과 추이 계산

    최근 7일 일평균 vs 그 전 7일 일평균 비교하여 trend 업데이트
    - 이번 주 평균이 더 높으면: 'up'
    - 이번 주 평균이 더 낮으면: 'down'
    - 동일하면: 'same'
    """
    print("\n[INFO] 상담사 성과 추이(trend) 계산 중...")

    cursor = conn.cursor()

    try:
        # 영업일 수 계산 (주말 제외)
        # 일평균 = 상담 건수 / 영업일 수
        query = """
        WITH business_days AS (
            -- 이번 주 영업일 수 (최근 7일 중 주말 제외)
            SELECT
                COUNT(DISTINCT call_date) FILTER (
                    WHERE call_date >= CURRENT_DATE - INTERVAL '7 days'
                ) AS this_week_days,
                COUNT(DISTINCT call_date) FILTER (
                    WHERE call_date >= CURRENT_DATE - INTERVAL '14 days'
                      AND call_date < CURRENT_DATE - INTERVAL '7 days'
                ) AS last_week_days
            FROM consultations
        ),
        trend_calculation AS (
            SELECT
                emp.id,
                -- 이번 주 상담 건수 (최근 7일)
                COUNT(CASE
                    WHEN c.call_date >= CURRENT_DATE - INTERVAL '7 days' THEN 1
                END) AS this_week_cnt,
                -- 저번 주 상담 건수 (8~14일 전)
                COUNT(CASE
                    WHEN c.call_date >= CURRENT_DATE - INTERVAL '14 days'
                     AND c.call_date < CURRENT_DATE - INTERVAL '7 days' THEN 1
                END) AS last_week_cnt
            FROM employees emp
            LEFT JOIN consultations c ON c.agent_id = emp.id
            GROUP BY emp.id
        )
        UPDATE employees e
        SET
            trend = CASE
                -- 일평균 비교 (영업일 수로 나눔)
                WHEN bd.this_week_days > 0 AND bd.last_week_days > 0 THEN
                    CASE
                        WHEN (tc.this_week_cnt::float / bd.this_week_days) >
                             (tc.last_week_cnt::float / bd.last_week_days) * 1.05 THEN 'up'
                        WHEN (tc.this_week_cnt::float / bd.this_week_days) <
                             (tc.last_week_cnt::float / bd.last_week_days) * 0.95 THEN 'down'
                        ELSE 'same'
                    END
                ELSE 'same'
            END,
            updated_at = NOW()
        FROM trend_calculation tc, business_days bd
        WHERE e.id = tc.id;
        """

        cursor.execute(query)
        updated_count = cursor.rowcount
        conn.commit()

        print(f"[INFO] 상담사 trend 업데이트 완료: {updated_count}명")
        return updated_count

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] 상담사 trend 계산 실패: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        cursor.close()


def calculate_frequent_inquiry_trends(conn: psycopg2_connection):
    """
    자주 찾는 문의 추이 계산

    최근 7일 일평균 vs 그 전 7일 일평균 비교하여 trend 업데이트
    문의 건수는 해당 키워드가 포함된 상담 건수로 계산
    """
    print("\n[INFO] 자주 찾는 문의 추이(trend) 계산 중...")

    cursor = conn.cursor()

    try:
        # 영업일 수 계산 후 일평균 비교
        query = """
        WITH business_days AS (
            -- 이번 주/저번 주 영업일 수
            SELECT
                COUNT(DISTINCT call_date) FILTER (
                    WHERE call_date >= CURRENT_DATE - INTERVAL '7 days'
                ) AS this_week_days,
                COUNT(DISTINCT call_date) FILTER (
                    WHERE call_date >= CURRENT_DATE - INTERVAL '14 days'
                      AND call_date < CURRENT_DATE - INTERVAL '7 days'
                ) AS last_week_days
            FROM consultations
        ),
        inquiry_counts AS (
            SELECT
                fi.id,
                fi.keyword,
                -- 이번 주 관련 상담 건수 (최근 7일)
                (
                    SELECT COUNT(*)
                    FROM consultations c
                    WHERE c.call_date >= CURRENT_DATE - INTERVAL '7 days'
                    AND (
                        c.category_raw ILIKE '%%' || fi.keyword || '%%'
                        OR c.title ILIKE '%%' || fi.keyword || '%%'
                        OR c.category_main ILIKE '%%' || fi.keyword || '%%'
                        OR c.category_sub ILIKE '%%' || fi.keyword || '%%'
                    )
                ) AS this_week_cnt,
                -- 저번 주 관련 상담 건수 (8~14일 전)
                (
                    SELECT COUNT(*)
                    FROM consultations c
                    WHERE c.call_date >= CURRENT_DATE - INTERVAL '14 days'
                      AND c.call_date < CURRENT_DATE - INTERVAL '7 days'
                    AND (
                        c.category_raw ILIKE '%%' || fi.keyword || '%%'
                        OR c.title ILIKE '%%' || fi.keyword || '%%'
                        OR c.category_main ILIKE '%%' || fi.keyword || '%%'
                        OR c.category_sub ILIKE '%%' || fi.keyword || '%%'
                    )
                ) AS last_week_cnt
            FROM frequent_inquiries fi
            WHERE fi.is_active = true
        )
        UPDATE frequent_inquiries fi
        SET
            trend = CASE
                -- 일평균 비교 (5% 이상 차이 시 up/down, 그 외 same)
                WHEN bd.this_week_days > 0 AND bd.last_week_days > 0 THEN
                    CASE
                        WHEN (ic.this_week_cnt::float / bd.this_week_days) >
                             (ic.last_week_cnt::float / bd.last_week_days) * 1.05 THEN 'up'
                        WHEN (ic.this_week_cnt::float / bd.this_week_days) <
                             (ic.last_week_cnt::float / bd.last_week_days) * 0.95 THEN 'down'
                        ELSE 'same'
                    END
                ELSE 'same'
            END,
            count = ic.this_week_cnt + ic.last_week_cnt,  -- 최근 2주간 총 건수
            updated_at = NOW()
        FROM inquiry_counts ic, business_days bd
        WHERE fi.id = ic.id;
        """

        cursor.execute(query)
        updated_count = cursor.rowcount
        conn.commit()

        print(f"[INFO] 자주 찾는 문의 trend 업데이트 완료: {updated_count}건")
        return updated_count

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] 자주 찾는 문의 trend 계산 실패: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        cursor.close()


def calculate_all_trends(conn: psycopg2_connection):
    """
    모든 trend 계산 (employees + frequent_inquiries)

    최초 DB 적재 시 또는 배치 Job에서 호출
    """
    print("\n" + "=" * 60)
    print("[TREND] Trend 계산 시작")
    print("=" * 60)

    employee_count = calculate_employee_trends(conn)
    inquiry_count = calculate_frequent_inquiry_trends(conn)

    print("\n[INFO] Trend 계산 완료")
    print(f"  - 상담사: {employee_count}명")
    print(f"  - 자주 찾는 문의: {inquiry_count}건")

    return employee_count, inquiry_count


# 직접 실행 시 (배치 Job용)
if __name__ == "__main__":
    from . import connect_db

    print("=" * 60)
    print("Trend 계산 배치 Job")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    conn = connect_db()
    try:
        calculate_all_trends(conn)
    finally:
        conn.close()

    print("\n[SUCCESS] 배치 Job 완료")
