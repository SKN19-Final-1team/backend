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
      

def update_customer(conn, customer_id: str, current_type_code: str, type_history, fcr):
    """
    기존 고객의 성향 정보 업데이트
    """
    try:
        with conn.cursor() as cur:
            update_query = """
                UPDATE customers 
                SET 
                    type_history = %s, 
                    current_type_code = %s, 
                    updated_at = CURRENT_TIMESTAMP,
                    last_consultation_date = CURRENT_DATE,
                    total_consultations = total_consultations + 1,
                    resolved_first_call = %s
                WHERE id = %s;
            """
            
            cur.execute(update_query, (json.dumps(type_history), current_type_code, fcr, customer_id))
            
            conn.commit()
            
            if cur.rowcount == 0:
                print(f"[WARNING] No customer found with id {customer_id}. Nothing updated.")
            else:
                print(f"[INFO] Successfully updated customer {customer_id}")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to update customer {customer_id}: {e}")