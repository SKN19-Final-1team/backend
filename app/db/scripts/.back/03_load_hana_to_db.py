"""
하나카드 데이터 DB 적재 스크립트

기능:
- consultations 테이블 적재 (hana_rdb_metadata.json)
- consultation_documents 테이블 적재 (hana_vectordb_with_embeddings.json)
- 데이터 매핑 및 변환
- Foreign Key 관계 설정
- 트랜잭션 처리 및 에러 핸들링
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, time as dt_time
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, Json
from psycopg2 import sql
from tqdm import tqdm

# 환경 변수 로드
# 1. 로컬 .env 파일 우선 (scripts/db_loading/.env)
# 2. 프로젝트 루트 .env 파일 (override=False로 이미 로드된 값은 유지)
load_dotenv(Path(__file__).parent / '.env', override=False)
load_dotenv(Path(__file__).parent.parent.parent / '.env', override=False)

# config.py에서 경로 가져오기
from config import HANA_RDB_METADATA_FILE, HANA_VECTORDB_FILE

# 상수
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
RDB_METADATA_FILE = HANA_RDB_METADATA_FILE
VECTORDB_FILE = HANA_VECTORDB_FILE

# 환경 변수
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "callact_admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "callact_pwd1")
DB_NAME = os.getenv("DB_NAME", "callact_db")
BATCH_SIZE = int(os.getenv("DB_LOAD_BATCH_SIZE", "100"))
COMMIT_INTERVAL = int(os.getenv("DB_COMMIT_INTERVAL", "500"))

# 기본 상담사 ID
DEFAULT_AGENT_ID = "EMP-TEDDY-DEFAULT"


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
        sys.exit(1)


def ensure_default_agent(conn: psycopg2_connection) -> str:
    """기본 상담사가 없으면 생성"""
    cursor = conn.cursor()
    
    # 기본 상담사 확인
    cursor.execute("SELECT id FROM employees WHERE id = %s", (DEFAULT_AGENT_ID,))
    if cursor.fetchone():
        print(f"[INFO] Default agent exists: {DEFAULT_AGENT_ID}")
        cursor.close()
        return DEFAULT_AGENT_ID
    
    # 기본 상담사 생성
    try:
        cursor.execute("""
            INSERT INTO employees (id, name, email, role, department, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """, (
            DEFAULT_AGENT_ID,
            "테디카드 기본 상담사",
            "default@teddycard.com",
            "상담사",
            "테디카드 상담팀",
            "active"
        ))
        conn.commit()
        print(f"[INFO] Created default agent: {DEFAULT_AGENT_ID}")
    except Exception as e:
        print(f"[WARNING] Failed to create default agent: {e}")
        conn.rollback()
    finally:
        cursor.close()
    
    return DEFAULT_AGENT_ID


def convert_status(status: str) -> str:
    """상태 변환: "완료" → "completed" """
    status_map = {
        "완료": "completed",
        "진행중": "in_progress",
        "미완료": "incomplete"
    }
    return status_map.get(status, "completed")


def convert_duration(seconds: int) -> str:
    """초를 "MM:SS" 형식으로 변환"""
    if seconds is None:
        return None
    
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def map_consultation_data(row: Dict, agent_id: str) -> Tuple:
    """consultations 테이블 데이터 매핑"""
    # ID
    consultation_id = row.get("id", "")
    
    # 고객 ID
    customer_id = row.get("client_id", "")
    
    # 카테고리 (VARCHAR로 저장, Enum 변환 없음)
    category = row.get("consulting_category", "")
    
    # 상태 변환
    status = convert_status(row.get("status", "완료"))
    
    # 제목 생성
    title = f"{category} 상담" if category else "상담"
    
    # 날짜/시간 (없으면 null)
    call_date = None
    call_time = None
    if row.get("call_start_time"):
        try:
            dt = datetime.fromisoformat(row["call_start_time"].replace("Z", "+00:00"))
            call_date = dt.date()
            call_time = dt.time()
        except:
            pass
    
    # 통화 시간 변환
    call_duration = convert_duration(row.get("call_duration"))
    
    # 기타 필드
    fcr = None  # First Call Resolution (데이터에 없음)
    is_best_practice = False
    quality_score = None
    
    return (
        consultation_id,
        customer_id,
        agent_id,
        status,
        category,
        title,
        call_date,
        call_time,
        call_duration,
        fcr,
        is_best_practice,
        quality_score
    )


def load_consultations(
    conn: psycopg2_connection,
    json_path: Path,
    limit: Optional[int] = None
) -> int:
    """consultations 테이블 적재"""
    
    print(f"\n[INFO] Loading consultations from {json_path}...")
    
    # 데이터 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
        print(f"[INFO] Limited to {limit} records")
    
    print(f"[INFO] Total records: {len(data)}")
    
    # 기본 상담사 확인/생성
    agent_id = ensure_default_agent(conn)
    
    # 데이터 매핑
    mapped_data = []
    for row in data:
        try:
            mapped = map_consultation_data(row, agent_id)
            mapped_data.append(mapped)
        except Exception as e:
            print(f"[WARNING] Failed to map consultation {row.get('id')}: {e}")
            continue
    
    # 배치 삽입
    insert_query = """
        INSERT INTO consultations (
            id, customer_id, agent_id, status, category, title,
            call_date, call_time, call_duration, fcr, is_best_practice, quality_score,
            created_at, updated_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
        )
        ON CONFLICT (id) DO UPDATE SET
            customer_id = EXCLUDED.customer_id,
            agent_id = EXCLUDED.agent_id,
            status = EXCLUDED.status,
            category = EXCLUDED.category,
            title = EXCLUDED.title,
            call_date = EXCLUDED.call_date,
            call_time = EXCLUDED.call_time,
            call_duration = EXCLUDED.call_duration,
            updated_at = NOW()
    """
    
    cursor = conn.cursor()
    
    try:
        print(f"[INFO] Inserting {len(mapped_data)} consultations...")
        execute_batch(cursor, insert_query, mapped_data, page_size=BATCH_SIZE)
        conn.commit()
        print(f"[INFO] Successfully inserted {len(mapped_data)} consultations")
    except Exception as e:
        print(f"[ERROR] Failed to insert consultations: {e}")
        conn.rollback()
        return 0
    finally:
        cursor.close()
    
    return len(mapped_data)


def map_document_data(row: Dict) -> Tuple:
    """consultation_documents 테이블 데이터 매핑"""
    # ID
    doc_id = row.get("id", "")
    
    # Consultation ID (FK)
    # consultation_id는 "CS-HANA-20593" 형식이지만, consultations.id는 "hana_consultation_20593" 형식
    consultation_id_raw = row.get("consultation_id", "")
    if consultation_id_raw.startswith("CS-HANA-"):
        consultation_id = consultation_id_raw.replace("CS-HANA-", "hana_consultation_")
    else:
        # 이미 "hana_consultation_" 형식이면 그대로 사용
        consultation_id = consultation_id_raw if consultation_id_raw else doc_id
    
    # Document type
    document_type = row.get("document_type", "consultation_transcript")
    
    # Category
    category = row.get("metadata", {}).get("category", "")
    
    # Title
    title = row.get("title", "")
    
    # Content
    content = row.get("content", "")
    
    # Keywords (배열 변환)
    keywords = row.get("metadata", {}).get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    
    # Embedding (벡터 변환)
    embedding = row.get("embedding")
    if embedding:
        # pgvector 형식: '[0.1,0.2,0.3,...]'
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    else:
        embedding_str = None
    
    # Metadata (JSONB)
    metadata = {
        "source_id": row.get("metadata", {}).get("source_id"),
        "slot_types": row.get("metadata", {}).get("slot_types", []),
        "scenario_tags": row.get("metadata", {}).get("scenario_tags", []),
        "summary": row.get("metadata", {}).get("summary"),
        "created_at": row.get("metadata", {}).get("created_at")
    }
    
    # 기타 필드
    usage_count = 0
    effectiveness_score = None
    last_used = None
    
    return (
        doc_id,
        consultation_id,
        document_type,
        category,
        title,
        content,
        keywords,
        embedding_str,
        Json(metadata),
        usage_count,
        effectiveness_score,
        last_used
    )


def load_consultation_documents(
    conn: psycopg2_connection,
    json_path: Path,
    limit: Optional[int] = None
) -> int:
    """consultation_documents 테이블 적재"""
    
    print(f"\n[INFO] Loading consultation_documents from {json_path}...")
    
    # 파일 존재 확인
    if not json_path.exists():
        print(f"[ERROR] File not found: {json_path}")
        print(f"[INFO] Please run generate_embeddings_hana.py first")
        return 0
    
    # 데이터 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
        print(f"[INFO] Limited to {limit} records")
    
    print(f"[INFO] Total records: {len(data)}")
    
    # 데이터 매핑
    mapped_data = []
    for row in data:
        try:
            mapped = map_document_data(row)
            mapped_data.append(mapped)
        except Exception as e:
            print(f"[WARNING] Failed to map document {row.get('id')}: {e}")
            continue
    
    # 배치 삽입
    insert_query = """
        INSERT INTO consultation_documents (
            id, consultation_id, document_type, category, title, content,
            keywords, embedding, metadata, usage_count, effectiveness_score, last_used,
            created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s, NOW()
        )
        ON CONFLICT (id) DO UPDATE SET
            consultation_id = EXCLUDED.consultation_id,
            document_type = EXCLUDED.document_type,
            category = EXCLUDED.category,
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            keywords = EXCLUDED.keywords,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata
    """
    
    cursor = conn.cursor()
    
    try:
        print(f"[INFO] Inserting {len(mapped_data)} consultation_documents...")
        pbar = tqdm(total=len(mapped_data), desc="Loading documents")
        
        for i in range(0, len(mapped_data), BATCH_SIZE):
            batch = mapped_data[i:i+BATCH_SIZE]
            execute_batch(cursor, insert_query, batch, page_size=len(batch))
            
            # 주기적 커밋
            if (i + len(batch)) % COMMIT_INTERVAL == 0:
                conn.commit()
                print(f"[INFO] Committed {i + len(batch)} documents")
            
            pbar.update(len(batch))
        
        # 최종 커밋
        conn.commit()
        pbar.close()
        print(f"[INFO] Successfully inserted {len(mapped_data)} consultation_documents")
    except Exception as e:
        print(f"[ERROR] Failed to insert consultation_documents: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
        return 0
    finally:
        cursor.close()
    
    return len(mapped_data)


def verify_load(conn: psycopg2_connection) -> Dict:
    """적재 데이터 검증"""
    cursor = conn.cursor()
    
    results = {}
    
    try:
        # consultations 개수
        cursor.execute("SELECT COUNT(*) FROM consultations")
        results["consultations_count"] = cursor.fetchone()[0]
        
        # consultation_documents 개수
        cursor.execute("SELECT COUNT(*) FROM consultation_documents")
        results["documents_count"] = cursor.fetchone()[0]
        
        # 임베딩이 있는 문서 개수
        cursor.execute("SELECT COUNT(*) FROM consultation_documents WHERE embedding IS NOT NULL")
        results["documents_with_embedding"] = cursor.fetchone()[0]
        
        # Foreign Key 관계 확인
        cursor.execute("""
            SELECT COUNT(*) 
            FROM consultation_documents cd
            LEFT JOIN consultations c ON cd.consultation_id = c.id
            WHERE c.id IS NULL
        """)
        results["orphaned_documents"] = cursor.fetchone()[0]
        
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
    finally:
        cursor.close()
    
    return results


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load Hana Card data to PostgreSQL")
    parser.add_argument("--limit", type=int, help="Limit number of records to load (for testing)")
    parser.add_argument("--skip-consultations", action="store_true", help="Skip loading consultations")
    parser.add_argument("--skip-documents", action="store_true", help="Skip loading consultation_documents")
    
    args = parser.parse_args()
    
    # 데이터베이스 연결
    conn = connect_db()
    
    try:
        total_consultations = 0
        total_documents = 0
        
        # consultations 적재
        if not args.skip_consultations:
            total_consultations = load_consultations(
                conn,
                RDB_METADATA_FILE,
                limit=args.limit
            )
        
        # consultation_documents 적재
        if not args.skip_documents:
            total_documents = load_consultation_documents(
                conn,
                VECTORDB_FILE,
                limit=args.limit
            )
        
        # 검증
        print("\n[INFO] Verifying loaded data...")
        verification = verify_load(conn)
        
        print("\n[SUMMARY]")
        print(f"  Consultations loaded: {total_consultations}")
        print(f"  Documents loaded: {total_documents}")
        print(f"  Total consultations in DB: {verification.get('consultations_count', 0)}")
        print(f"  Total documents in DB: {verification.get('documents_count', 0)}")
        print(f"  Documents with embedding: {verification.get('documents_with_embedding', 0)}")
        print(f"  Orphaned documents: {verification.get('orphaned_documents', 0)}")
        
        if verification.get('orphaned_documents', 0) > 0:
            print(f"\n[WARNING] Found {verification['orphaned_documents']} documents without matching consultation")
    
    finally:
        conn.close()
        print("\n[INFO] Database connection closed")


if __name__ == "__main__":
    main()
