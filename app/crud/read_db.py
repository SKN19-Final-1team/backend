import sys
import os

from app.db.base import get_connection
import psycopg2.extras 

# 스크립트로 직접 실행 시 backend 경로를 sys.path에 추가하여 app 패키지를 찾을 수 있게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(backend_dir)

TABLE_NAME = "guide_tbl"
LIMIT = 5

def read_table(table_name: str, limit: int = None, conditions: dict = None):
    conn = get_connection()
    data = []
    
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            query = f"SELECT * FROM {table_name}"
            params = []
            
            if conditions:
                clauses = []
                for key, value in conditions.items():
                    clauses.append(f"{key} = %s")
                    params.append(value)
                query += " WHERE " + " AND ".join(clauses)
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
                
            cur.execute(query, tuple(params))
            data = cur.fetchall()
            
    except Exception as e:
        print(f"❌ {table_name} 테이블 읽기 실패: {e}")
    finally:
        conn.close()
        print(">>> conn closed\n")
        
    return data

if __name__ == "__main__":
    try:
        results = read_table(TABLE_NAME, LIMIT)
        for row in results:
            print(row)
        print(f"{TABLE_NAME} 테이블에서 {len(results)}개 행 조회")
    except Exception as e:
        print(f"실행 실패: {e}")