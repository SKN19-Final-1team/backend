"""
고객 데이터 적재 모듈

customersData.json에서 고객 데이터를 읽어 customers 테이블에 적재
"""

import json
from datetime import datetime, date
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, Json as PsycopgJson

from . import (
    CUSTOMERS_DATA_FILE, BATCH_SIZE,
    check_table_has_data
)


def load_customers_data(conn: psycopg2_connection):
    """고객 데이터 적재 (customersData.json)"""
    print("\n" + "=" * 60)
    print("[4-1/12] 고객 데이터 적재")
    print("=" * 60)

    # 이미 데이터가 있는지 확인
    has_data, count = check_table_has_data(conn, "customers")
    if has_data:
        print(f"[INFO] customers 테이블에 이미 데이터가 있습니다. ({count}건) - 적재 스킵")
        return True

    if not CUSTOMERS_DATA_FILE.exists():
        print(f"[ERROR] 고객 데이터 파일을 찾을 수 없습니다: {CUSTOMERS_DATA_FILE}")
        return False

    with open(CUSTOMERS_DATA_FILE, 'r', encoding='utf-8') as f:
        customers_data = json.load(f)

    print(f"[INFO] 고객 데이터 파일 로드: {len(customers_data)}명")

    cursor = conn.cursor()

    try:
        insert_customer = """
            INSERT INTO customers (
                id, name, phone, gender, age_group, birth_date, address, grade,
                card_type, card_number_last4, card_brand, card_issue_date, card_expiry_date,
                current_type_code, type_history, personality_tags, communication_style,
                customer_type_codes, llm_guidance, total_consultations, resolved_first_call,
                last_consultation_date, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                phone = EXCLUDED.phone,
                gender = EXCLUDED.gender,
                age_group = EXCLUDED.age_group,
                grade = EXCLUDED.grade,
                card_type = EXCLUDED.card_type,
                current_type_code = EXCLUDED.current_type_code,
                personality_tags = EXCLUDED.personality_tags,
                communication_style = EXCLUDED.communication_style,
                customer_type_codes = EXCLUDED.customer_type_codes,
                llm_guidance = EXCLUDED.llm_guidance,
                updated_at = NOW()
        """

        customer_batch = []
        for cust in customers_data:
            # 날짜 필드 처리
            birth_date = None
            if cust.get('birth_date'):
                try:
                    birth_date = datetime.fromisoformat(cust['birth_date'].replace('Z', '+00:00')).date() if 'T' in cust['birth_date'] else date.fromisoformat(cust['birth_date'])
                except:
                    birth_date = None

            card_issue_date = None
            if cust.get('card_issue_date'):
                try:
                    card_issue_date = date.fromisoformat(cust['card_issue_date'])
                except:
                    card_issue_date = None

            card_expiry_date = None
            if cust.get('card_expiry_date'):
                try:
                    card_expiry_date = date.fromisoformat(cust['card_expiry_date'])
                except:
                    card_expiry_date = None

            last_consultation_date = None
            if cust.get('last_consultation_date'):
                try:
                    last_consultation_date = date.fromisoformat(cust['last_consultation_date'])
                except:
                    last_consultation_date = None

            customer_batch.append((
                cust.get('id', ''),
                cust.get('name', ''),
                cust.get('phone', ''),
                cust.get('gender', 'unknown'),
                cust.get('age_group'),
                birth_date,
                cust.get('address'),
                cust.get('grade', 'GENERAL'),
                cust.get('card_type'),
                cust.get('card_number_last4'),
                cust.get('card_brand'),
                card_issue_date,
                card_expiry_date,
                cust.get('current_type_code'),
                PsycopgJson(cust.get('type_history', [])),
                cust.get('personality_tags', []),
                PsycopgJson(cust.get('communication_style', {})),
                cust.get('customer_type_codes', []),
                cust.get('llm_guidance'),
                cust.get('total_consultations', 0),
                cust.get('resolved_first_call', 0),
                last_consultation_date
            ))

        if customer_batch:
            execute_batch(cursor, insert_customer, customer_batch, page_size=BATCH_SIZE)
            conn.commit()
            print(f"[INFO] 고객 데이터 적재 완료: {len(customer_batch)}명")
        else:
            print("[WARNING] 적재할 고객 데이터가 없습니다.")

        cursor.close()
        return True

    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] 고객 데이터 적재 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
