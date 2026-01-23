import json
import psycopg2.extras

from app.db.base import get_connection

def test_fetch_persona():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM consultation_documents LIMIT 1")
            result = cur.fetchone()
            
            if result:
                print(json.dumps(result, indent=4, default=str, ensure_ascii=False))
            else:
                print(json.dumps({"message": "No data found"}, indent=4))
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    test_fetch_persona()