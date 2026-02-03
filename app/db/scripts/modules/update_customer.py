import json

def get_personality_history(conn, customer_id: str):
    """
    특정 고객의 최신 성향 이력을 리스트로 반환
    """
    try:
        with conn.cursor() as cur:
            query = """
                SELECT type_history 
                FROM customers
                WHERE id = %s;
            """
            cur.execute(query, (customer_id,))
            
            row = cur.fetchone()
            
            if row:
                return row[0]
            
            return []

    except Exception as e:
        print(f"[ERROR] Failed to fetch history for customer {customer_id}: {e}")
        return []
      

def update_customer(conn, customer_id: str, current_type_code: str, type_history, fcr=None):
    """
    고객의 페르소나(성향) 정보 업데이트

    v4.0 변경사항:
    - total_consultations, last_consultation_date, resolved_first_call은
      DB 트리거(trg_consultation_insert_*)가 자동 처리
    - 이 함수는 페르소나 정보(current_type_code, type_history)만 업데이트

    Args:
        conn: DB 연결
        customer_id: 고객 ID
        current_type_code: LLM 분류 결과 (N1, N2, S1, S2, S3)
        type_history: 성향 이력 배열 (예: ["N1", "S2", "S2"])
        fcr: 더 이상 사용하지 않음 (트리거에서 자동 계산)
    """
    # v4.0: 5타입 유효성 검증
    VALID_TYPE_CODES = {'N1', 'N2', 'S1', 'S2', 'S3'}
    if current_type_code and current_type_code not in VALID_TYPE_CODES:
        print(f"[WARNING] Invalid type code '{current_type_code}'. Must be one of {VALID_TYPE_CODES}")
        return

    try:
        with conn.cursor() as cur:
            # v4.0: 페르소나 정보만 업데이트 (통계는 트리거가 처리)
            update_query = """
                UPDATE customers
                SET
                    type_history = %s,
                    current_type_code = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s;
            """

            cur.execute(update_query, (json.dumps(type_history), current_type_code, customer_id))

            conn.commit()

            if cur.rowcount == 0:
                print(f"[WARNING] No customer found with id {customer_id}. Nothing updated.")
            else:
                print(f"[INFO] Successfully updated customer {customer_id} persona to {current_type_code}")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to update customer {customer_id}: {e}")