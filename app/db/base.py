import os
import psycopg2
from dotenv import load_dotenv
load_dotenv()

def get_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dbname=os.getenv("DB_NAME")
        )
        return conn
        
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        raise e
