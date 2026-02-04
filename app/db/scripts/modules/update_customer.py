import json
from datetime import datetime

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



def save_consultation_to_db(conn, data, script, evaluation):
    try:
        with conn.cursor() as cur:
            # 상태값 Enum 매핑
            status_mapping = {
                "진행중": "in_progress",
                "완료": "completed",
                "미완료": "incomplete"
            }
            db_status = status_mapping.get(data.status, "in_progress")

            # 날짜/시간 처리
            dt_obj = datetime.strptime(data.datetime, '%Y-%m-%d %H:%M')
            
            # 품질 점수 추출
            mc = evaluation.get('manual_compliance', {})
            q_score_raw = mc.get('manual_score', 0)
            quality_score = int(''.join(filter(str.isdigit, str(q_score_raw)))) if any(c.isdigit() for c in str(q_score_raw)) else 0

            # SQL 실행 (category_sub 컬럼 추가)
            query = """
                INSERT INTO consultations (
                    id, customer_id, agent_id, status, 
                    category_main, category_sub, category_raw, title, 
                    call_date, call_time, call_duration, acw_duration,
                    transcript, ai_summary, agent_notes, 
                    follow_up_schedule, transfer_department, transfer_notes, 
                    referenced_documents, 
                    feedback_text, feedback_emotions, emotion_score,
                    quality_score, updated_at
                ) VALUES (
                    %s, %s, %s, %s, 
                    %s, %s, %s, %s, 
                    %s, %s, %s, %s,
                    %s, %s, %s, 
                    %s, %s, %s, 
                    %s, 
                    %s, %s, %s,
                    %s, NOW()
                )
            """

            cur.execute(query, (
                data.consultationId,
                data.customerId,
                data.employeeId,
                db_status,
                data.category,     
                data.subcategory,  
                data.category,     
                data.title,
                dt_obj.date(),
                dt_obj.time(),
                data.callTimeSeconds,
                data.acwTimeSeconds,
                json.dumps(script, ensure_ascii=False),
                data.aiSummary,
                data.memo,
                data.followUpTasks,
                data.handoffDepartment,
                data.handoffNotes,
                json.dumps(data.referencedDocuments, ensure_ascii=False),
                evaluation.get('feedback', ''),
                json.dumps(evaluation.get('emotions', {}), ensure_ascii=False),
                evaluation.get('emotion_score', 0),
                quality_score
            ))

            conn.commit()

    except Exception as e:
        if conn: conn.rollback()
        print(f"상담 DB 저장 실패 {e}")
        return False