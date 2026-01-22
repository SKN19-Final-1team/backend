import psycopg2
from app.core import config

def get_connection():
    try:
        conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            dbname=config.DB_NAME
        )
        return conn
        
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        raise e
