"""
DB 스키마 생성 스크립트 (Python)

기능:
- db_setup.sql 파일을 읽어서 PostgreSQL에 실행
- pgvector 확장 설치
- 테이블 생성
- 인덱스 생성
- 기본 데이터 생성

사용법:
    python setup_db.py
"""

import psycopg2
from psycopg2.extensions import connection as psycopg2_connection
from pathlib import Path
import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
# 1. 로컬 .env 파일 우선 (backend/app/db/scripts/.env)
# 2. 프로젝트 루트 .env 파일 (override=False로 이미 로드된 값은 유지)
load_dotenv(Path(__file__).parent / '.env', override=False)
load_dotenv(Path(__file__).parent.parent.parent.parent.parent / '.env', override=False)

# 상수
SQL_FILE = Path(__file__).parent / "db_setup.sql"

# 환경 변수
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "callact_admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "callact_pwd1")
DB_NAME = os.getenv("DB_NAME", "callact_db")


def connect_db() -> psycopg2_connection:
    """PostgreSQL 데이터베이스 연결"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        print(f"[INFO] Connected to database: {DB_NAME}")
        return conn
    except Exception as e:
        print(f"[ERROR] Failed to connect to database: {e}")
        print(f"[INFO] Please check:")
        print(f"  - Docker container is running: docker ps")
        print(f"  - Connection info in .env file: {DB_HOST}:{DB_PORT}/{DB_NAME}")
        sys.exit(1)


def load_sql_file(file_path: Path) -> str:
    """SQL 파일 읽기"""
    if not file_path.exists():
        print(f"[ERROR] SQL file not found: {file_path}")
        sys.exit(1)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def execute_sql_script(conn: psycopg2_connection, sql_script: str):
    """SQL 스크립트 실행"""
    cursor = conn.cursor()
    
    try:
        # SQL 스크립트 실행
        # 주의: psycopg2는 기본적으로 한 번에 하나의 명령만 실행
        # 하지만 DO $$ 블록 등이 있으므로 전체를 한 번에 실행
        cursor.execute(sql_script)
        conn.commit()
        print("[INFO] DB 스키마 생성 완료!")
        
        # 성공 메시지 확인 (SQL 스크립트의 RAISE NOTICE 메시지)
        # RAISE NOTICE는 psycopg2에서 자동으로 출력됨
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to execute SQL script: {e}")
        print(f"[INFO] Rollback completed")
        
        # 일부 에러는 무시 가능 (이미 존재하는 객체 등)
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            print(f"[WARNING] Some objects may already exist. This is usually safe to ignore.")
        
        raise


def verify_schema(conn: psycopg2_connection):
    """스키마 생성 확인"""
    cursor = conn.cursor()
    
    # 테이블 목록 확인
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    
    tables = cursor.fetchall()
    print(f"\n[INFO] 생성된 테이블:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # pgvector 확장 확인
    cursor.execute("""
        SELECT extname 
        FROM pg_extension 
        WHERE extname = 'vector';
    """)
    
    vector_ext = cursor.fetchone()
    if vector_ext:
        print(f"[INFO] pgvector 확장 설치됨: ✅")
    else:
        print(f"[WARNING] pgvector 확장이 설치되지 않았습니다.")
    
    cursor.close()


def main():
    """메인 함수"""
    print("=" * 60)
    print("DB 스키마 생성 스크립트")
    print("=" * 60)
    print(f"[INFO] SQL file: {SQL_FILE}")
    print(f"[INFO] Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print()
    
    # SQL 파일 로드
    sql_script = load_sql_file(SQL_FILE)
    print(f"[INFO] SQL file loaded ({len(sql_script)} characters)")
    
    # DB 연결
    conn = connect_db()
    
    try:
        # SQL 스크립트 실행
        execute_sql_script(conn, sql_script)
        
        # 스키마 확인
        verify_schema(conn)
        
        print("\n" + "=" * 60)
        print("[SUCCESS] DB 스키마 생성이 완료되었습니다!")
        print("=" * 60)
        print("\n다음 단계:")
        print("  python 03_load_hana_to_db.py")
        print()
        
    except Exception as e:
        print(f"\n[ERROR] 스키마 생성 실패: {e}")
        sys.exit(1)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()

