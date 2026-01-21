import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

# DB 정보
DB_CONFIG = {
    'host': os.getenv('DB_HOST'), 
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT'))   
}

def test_remote_connection():
    conn = None
    try:
        # 1. 연결 시도
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # 2. 간단한 쿼리 실행 (DB 버전 확인)
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        
        print("\n✅ 연결 성공")
        print(f"📡 서버 정보: {db_version[0]}")
        
        try:
            cur.execute("SELECT count(*) FROM guide_tbl;")
            count = cur.fetchone()[0]
            print(f"📚 현재 저장된 문서 개수: {count}개")
        except Exception:
            print("⚠️ 접속은 성공했지만 테이블을 찾을 수 없습니다.")

    except psycopg2.OperationalError as e:
        print("\n❌ 접속 실패 (네트워크 또는 설정 문제)")
        print(f"에러 메시지: {e}")
        
    except Exception as e:
        print(f"\n❌ 기타 오류 발생: {e}")
        
    finally:
        if conn:
            conn.close()
            print("\n연결 종료")

if __name__ == "__main__":
    test_remote_connection()