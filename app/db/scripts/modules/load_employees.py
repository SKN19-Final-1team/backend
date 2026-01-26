"""
상담사 데이터 적재 모듈

employeesData.json에서 상담사 데이터를 읽어 employees 테이블에 적재
hire_date 포함
"""

import json
from datetime import date
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch

from . import (
    EMPLOYEES_DATA_FILE, BATCH_SIZE,
    check_employees_has_meaningful_data
)


def load_employees_data(conn: psycopg2_connection):
    """상담사 데이터 적재"""
    print("\n" + "=" * 60)
    print("[4/12] 상담사 데이터 적재")
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
                id, name, email, role, department, hire_date, status,
                consultations, fcr, "avgTime", rank, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                email = EXCLUDED.email,
                role = EXCLUDED.role,
                department = EXCLUDED.department,
                hire_date = COALESCE(EXCLUDED.hire_date, employees.hire_date),
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
            hire_date_str = emp.get('hire_date')
            hire_date_val = None
            if hire_date_str:
                try:
                    hire_date_val = date.fromisoformat(hire_date_str)
                except (ValueError, TypeError):
                    hire_date_val = None
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
                hire_date_val,  # hire_date
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
