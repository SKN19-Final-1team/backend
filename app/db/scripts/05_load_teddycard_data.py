"""
테디카드 전처리 데이터 DB 적재 스크립트

기능:
- service_guide_documents 테이블 적재
- card_products 테이블 적재
- notices 테이블 적재
- metadata에 document_number 추가 (약관 조 번호 추출)
- structured 필드 처리 (JSONB로 저장)
- 트랜잭션 처리 및 에러 핸들링
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, date
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, Json as PsycopgJson
from tqdm import tqdm

# 환경 변수 로드
load_dotenv(Path(__file__).parent / '.env', override=False)
load_dotenv(Path(__file__).parent.parent.parent.parent / '.env', override=False)

# config.py에서 경로 가져오기
from config import TEDDY_DATA_DIR_PROD, TEDDY_DATA_DIR_DEV

# 상수
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
# 데이터 파일 경로 (config.py에서 가져온 값 사용)
DATA_DIR_PROD = TEDDY_DATA_DIR_PROD
DATA_DIR_DEV = TEDDY_DATA_DIR_DEV

# 데이터 파일
SERVICE_GUIDES_FILE = "teddycard_service_guides_with_embeddings.json"
CARD_PRODUCTS_FILE = "teddycard_card_products_with_embeddings.json"
NOTICES_FILE = "teddycard_notices_with_embeddings.json"

# 환경 변수 (기본값 없음, .env 파일에서 필수)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432")) if os.getenv("DB_PORT") else None
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
BATCH_SIZE = int(os.getenv("DB_LOAD_BATCH_SIZE", "100"))
COMMIT_INTERVAL = int(os.getenv("DB_COMMIT_INTERVAL", "500"))

# 필수 환경 변수 확인
if not all([DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME]):
    print("[ERROR] 필수 환경 변수가 설정되지 않았습니다.")
    print("[ERROR] .env 파일에 다음 변수를 설정해주세요:")
    print("  - DB_HOST")
    print("  - DB_PORT")
    print("  - DB_USER")
    print("  - DB_PASSWORD")
    print("  - DB_NAME")
    sys.exit(1)


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


def find_data_file(filename: str) -> Optional[Path]:
    """데이터 파일 찾기 (프로덕션 우선, 개발 환경 대체)"""
    # 프로덕션 경로 확인
    file_path = DATA_DIR_PROD / filename
    if file_path.exists():
        print(f"[INFO] Found data file (PROD): {file_path}")
        return file_path
    
    # 개발 경로 확인
    file_path = DATA_DIR_DEV / filename
    if file_path.exists():
        print(f"[INFO] Found data file (DEV): {file_path}")
        return file_path
    
    print(f"[ERROR] Data file not found: {filename}")
    print(f"[ERROR] Checked paths:")
    print(f"  - PROD: {DATA_DIR_PROD}")
    print(f"  - DEV: {DATA_DIR_DEV}")
    return None


def extract_document_number(title: str) -> Optional[str]:
    """title에서 문서 번호 추출 (예: 제1조, 제2조)"""
    if not title:
        return None
    
    # 정규식으로 조 번호 추출: "제1조", "제2조" 등
    match = re.search(r'제\d+조', title)
    return match.group(0) if match else None


def prepare_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    """metadata 필드 준비 (document_number 추가)"""
    metadata = doc.get("metadata", {}).copy()
    
    # document_number 추가 (약관 조 번호)
    title = doc.get("title", "")
    document_number = extract_document_number(title)
    if document_number:
        metadata["document_number"] = document_number
    
    return metadata


def map_service_guide_data(doc: Dict[str, Any]) -> Tuple:
    """service_guide_documents 테이블 데이터 매핑"""
    # ID
    doc_id = doc.get("id", "")
    
    # Document type
    document_type = doc.get("document_type", "service_guide")
    
    # Category
    category = doc.get("category", "")
    
    # Title
    title = doc.get("title", "")
    
    # Content
    content = doc.get("content", "") or doc.get("text", "")
    
    # Keywords (배열)
    keywords = doc.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    
    # Embedding (벡터 변환)
    embedding = doc.get("embedding")
    if embedding:
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    else:
        embedding_str = None
    
    # Metadata (document_number 추가)
    metadata = prepare_metadata(doc)
    
    # Document source
    document_source = doc.get("document_source", "") or metadata.get("original_source", "")
    
    # Priority
    priority = doc.get("priority", "normal")
    
    # Structured (JSONB)
    structured = doc.get("structured")
    
    # 기타 필드
    usage_count = doc.get("usage_count", 0)
    last_used = None
    
    return (
        doc_id,
        document_type,
        category,
        title,
        content,
        keywords,
        embedding_str,
        PsycopgJson(metadata),
        document_source,
        priority,
        usage_count,
        last_used,
        PsycopgJson(structured) if structured else None
    )


def map_card_product_data(doc: Dict[str, Any]) -> Tuple:
    """card_products 테이블 데이터 매핑"""
    # ID
    card_id = doc.get("id", "")
    
    # Name
    name = doc.get("name", "")
    
    # Card type
    card_type = doc.get("card_type", "credit")  # credit, debit
    
    # Brand
    brand = doc.get("brand")
    if brand:
        # brand_type enum: visa, mastercard, amex, unionpay, local
        brand = brand.lower()
        if brand not in ["visa", "mastercard", "amex", "unionpay", "local"]:
            brand = "local"  # 기본값
    else:
        brand = None
    
    # Annual fees
    annual_fee_domestic = doc.get("annual_fee_domestic")
    if isinstance(annual_fee_domestic, str):
        # 문자열인 경우 숫자 추출 시도
        try:
            annual_fee_domestic = int(re.sub(r'[^\d]', '', annual_fee_domestic))
        except:
            annual_fee_domestic = None
    
    annual_fee_global = doc.get("annual_fee_global")
    if isinstance(annual_fee_global, str):
        try:
            annual_fee_global = int(re.sub(r'[^\d]', '', annual_fee_global))
        except:
            annual_fee_global = None
    
    # Performance condition
    performance_condition = doc.get("performance_condition")
    
    # Main benefits
    main_benefits = doc.get("main_benefits", "")
    
    # Full content (ERD에는 없지만 metadata에 저장)
    full_content = doc.get("full_content", "")
    
    # Status
    status = doc.get("status", "active")
    
    # Metadata (full_content 포함)
    metadata = doc.get("metadata", {}).copy()
    if full_content:
        metadata["full_content"] = full_content
    
    # Structured (JSONB)
    structured = doc.get("structured")
    
    return (
        card_id,
        name,
        card_type,
        brand,
        annual_fee_domestic,
        annual_fee_global,
        performance_condition,
        main_benefits,
        status,
        PsycopgJson(metadata),
        PsycopgJson(structured) if structured else None
    )


def map_notice_data(doc: Dict[str, Any]) -> Tuple:
    """notices 테이블 데이터 매핑"""
    # ID
    notice_id = doc.get("id", "")
    
    # Title
    title = doc.get("title", "")
    
    # Content
    content = doc.get("content", "")
    
    # Category (enum: system, service, emergency)
    category = doc.get("category", "service")
    if category not in ["system", "service", "emergency"]:
        category = "service"  # 기본값
    
    # Priority (enum: normal, important, urgent)
    priority = doc.get("priority", "normal")
    if priority not in ["normal", "important", "urgent"]:
        priority = "normal"  # 기본값
    
    # Is pinned
    is_pinned = doc.get("is_pinned", False)
    
    # Dates
    start_date_str = doc.get("start_date")
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        except:
            start_date = None
    else:
        start_date = None
    
    end_date_str = doc.get("end_date")
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        except:
            end_date = None
    else:
        end_date = None
    
    # Status
    status = doc.get("status", "active")
    
    # Created by
    created_by = doc.get("created_by")
    
    # Keywords (배열)
    keywords = doc.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    
    # Embedding (벡터 변환)
    embedding = doc.get("embedding")
    if embedding:
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    else:
        embedding_str = None
    
    # Metadata
    metadata = doc.get("metadata", {})
    
    return (
        notice_id,
        title,
        content,
        category,
        priority,
        is_pinned,
        start_date,
        end_date,
        status,
        created_by,
        keywords,
        embedding_str,
        PsycopgJson(metadata)
    )


def load_service_guides(
    conn: psycopg2_connection,
    json_path: Path,
    limit: Optional[int] = None
) -> int:
    """service_guide_documents 테이블 적재"""
    
    print(f"\n[INFO] Loading service_guide_documents from {json_path}...")
    
    # 데이터 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
        print(f"[INFO] Limited to {limit} records")
    
    print(f"[INFO] Total records: {len(data)}")
    
    # 데이터 매핑
    mapped_data = []
    for doc in tqdm(data, desc="Mapping data"):
        try:
            mapped = map_service_guide_data(doc)
            mapped_data.append(mapped)
        except Exception as e:
            print(f"[WARNING] Failed to map document {doc.get('id')}: {e}")
            continue
    
    # 배치 삽입
    insert_query = """
        INSERT INTO service_guide_documents (
            id, document_type, category, title, content, keywords,
            embedding, metadata, document_source, priority,
            usage_count, last_used, structured, created_at, updated_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, NOW(), NOW()
        )
        ON CONFLICT (id) DO UPDATE SET
            document_type = EXCLUDED.document_type,
            category = EXCLUDED.category,
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            keywords = EXCLUDED.keywords,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            document_source = EXCLUDED.document_source,
            priority = EXCLUDED.priority,
            structured = EXCLUDED.structured,
            updated_at = NOW()
    """
    
    cursor = conn.cursor()
    
    try:
        print(f"[INFO] Inserting {len(mapped_data)} service_guide_documents...")
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
        print(f"[SUCCESS] Inserted {len(mapped_data)} service_guide_documents")
    except Exception as e:
        print(f"[ERROR] Failed to insert service_guide_documents: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
        return 0
    finally:
        cursor.close()
    
    return len(mapped_data)


def load_card_products(
    conn: psycopg2_connection,
    json_path: Path,
    limit: Optional[int] = None
) -> int:
    """card_products 테이블 적재"""
    
    print(f"\n[INFO] Loading card_products from {json_path}...")
    
    # 데이터 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
        print(f"[INFO] Limited to {limit} records")
    
    print(f"[INFO] Total records: {len(data)}")
    
    # 데이터 매핑
    mapped_data = []
    for doc in tqdm(data, desc="Mapping data"):
        try:
            mapped = map_card_product_data(doc)
            mapped_data.append(mapped)
        except Exception as e:
            print(f"[WARNING] Failed to map card product {doc.get('id')}: {e}")
            continue
    
    # 배치 삽입 (full_content는 metadata에 포함)
    insert_query = """
        INSERT INTO card_products (
            id, name, card_type, brand, annual_fee_domestic, annual_fee_global,
            performance_condition, main_benefits, status,
            metadata, structured, created_at, updated_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
        )
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            card_type = EXCLUDED.card_type,
            brand = EXCLUDED.brand,
            annual_fee_domestic = EXCLUDED.annual_fee_domestic,
            annual_fee_global = EXCLUDED.annual_fee_global,
            performance_condition = EXCLUDED.performance_condition,
            main_benefits = EXCLUDED.main_benefits,
            status = EXCLUDED.status,
            metadata = EXCLUDED.metadata,
            structured = EXCLUDED.structured,
            updated_at = NOW()
    """
    
    cursor = conn.cursor()
    
    try:
        print(f"[INFO] Inserting {len(mapped_data)} card_products...")
        pbar = tqdm(total=len(mapped_data), desc="Loading cards")
        
        for i in range(0, len(mapped_data), BATCH_SIZE):
            batch = mapped_data[i:i+BATCH_SIZE]
            execute_batch(cursor, insert_query, batch, page_size=len(batch))
            
            # 주기적 커밋
            if (i + len(batch)) % COMMIT_INTERVAL == 0:
                conn.commit()
                print(f"[INFO] Committed {i + len(batch)} cards")
            
            pbar.update(len(batch))
        
        # 최종 커밋
        conn.commit()
        pbar.close()
        print(f"[SUCCESS] Inserted {len(mapped_data)} card_products")
    except Exception as e:
        print(f"[ERROR] Failed to insert card_products: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
        return 0
    finally:
        cursor.close()
    
    return len(mapped_data)


def load_notices(
    conn: psycopg2_connection,
    json_path: Path,
    limit: Optional[int] = None
) -> int:
    """notices 테이블 적재"""
    
    print(f"\n[INFO] Loading notices from {json_path}...")
    
    # 데이터 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
        print(f"[INFO] Limited to {limit} records")
    
    print(f"[INFO] Total records: {len(data)}")
    
    # 데이터 매핑
    mapped_data = []
    for doc in tqdm(data, desc="Mapping data"):
        try:
            mapped = map_notice_data(doc)
            mapped_data.append(mapped)
        except Exception as e:
            print(f"[WARNING] Failed to map notice {doc.get('id')}: {e}")
            continue
    
    # 배치 삽입
    insert_query = """
        INSERT INTO notices (
            id, title, content, category, priority, is_pinned,
            start_date, end_date, status, created_by,
            keywords, embedding, metadata, created_at, updated_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, NOW(), NOW()
        )
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            category = EXCLUDED.category,
            priority = EXCLUDED.priority,
            is_pinned = EXCLUDED.is_pinned,
            start_date = EXCLUDED.start_date,
            end_date = EXCLUDED.end_date,
            status = EXCLUDED.status,
            created_by = EXCLUDED.created_by,
            keywords = EXCLUDED.keywords,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
    """
    
    cursor = conn.cursor()
    
    try:
        print(f"[INFO] Inserting {len(mapped_data)} notices...")
        pbar = tqdm(total=len(mapped_data), desc="Loading notices")
        
        for i in range(0, len(mapped_data), BATCH_SIZE):
            batch = mapped_data[i:i+BATCH_SIZE]
            execute_batch(cursor, insert_query, batch, page_size=len(batch))
            
            # 주기적 커밋
            if (i + len(batch)) % COMMIT_INTERVAL == 0:
                conn.commit()
                print(f"[INFO] Committed {i + len(batch)} notices")
            
            pbar.update(len(batch))
        
        # 최종 커밋
        conn.commit()
        pbar.close()
        print(f"[SUCCESS] Inserted {len(mapped_data)} notices")
    except Exception as e:
        print(f"[ERROR] Failed to insert notices: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
        return 0
    finally:
        cursor.close()
    
    return len(mapped_data)


def verify_load(conn: psycopg2_connection) -> Dict[str, Any]:
    """적재 데이터 검증"""
    cursor = conn.cursor()
    
    results = {}
    
    try:
        # service_guide_documents 개수
        cursor.execute("SELECT COUNT(*) FROM service_guide_documents")
        results["service_guides_count"] = cursor.fetchone()[0]
        
        # document_number가 있는 문서 개수
        cursor.execute("""
            SELECT COUNT(*) 
            FROM service_guide_documents 
            WHERE metadata->>'document_number' IS NOT NULL
        """)
        results["service_guides_with_document_number"] = cursor.fetchone()[0]
        
        # card_products 개수
        cursor.execute("SELECT COUNT(*) FROM card_products")
        results["card_products_count"] = cursor.fetchone()[0]
        
        # notices 개수
        cursor.execute("SELECT COUNT(*) FROM notices")
        results["notices_count"] = cursor.fetchone()[0]
        
        # 임베딩이 있는 문서 개수
        cursor.execute("SELECT COUNT(*) FROM service_guide_documents WHERE embedding IS NOT NULL")
        results["service_guides_with_embedding"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM notices WHERE embedding IS NOT NULL")
        results["notices_with_embedding"] = cursor.fetchone()[0]
        
        # structured 필드가 있는 문서 개수
        cursor.execute("SELECT COUNT(*) FROM service_guide_documents WHERE structured IS NOT NULL")
        results["service_guides_with_structured"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM card_products WHERE structured IS NOT NULL")
        results["card_products_with_structured"] = cursor.fetchone()[0]
        
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
    
    return results


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load TeddyCard data to PostgreSQL")
    parser.add_argument("--limit", type=int, help="Limit number of records to load (for testing)")
    parser.add_argument("--skip-service-guides", action="store_true", help="Skip loading service_guide_documents")
    parser.add_argument("--skip-card-products", action="store_true", help="Skip loading card_products")
    parser.add_argument("--skip-notices", action="store_true", help="Skip loading notices")
    
    args = parser.parse_args()
    
    # 데이터베이스 연결
    conn = connect_db()
    
    try:
        total_loaded = 0
        
        # 1. service_guide_documents 적재
        if not args.skip_service_guides:
            service_guides_file = find_data_file(SERVICE_GUIDES_FILE)
            if service_guides_file:
                count = load_service_guides(conn, service_guides_file, args.limit)
                total_loaded += count
            else:
                print(f"[WARNING] Skipping service_guide_documents (file not found)")
        
        # 2. card_products 적재
        if not args.skip_card_products:
            card_products_file = find_data_file(CARD_PRODUCTS_FILE)
            if card_products_file:
                count = load_card_products(conn, card_products_file, args.limit)
                total_loaded += count
            else:
                print(f"[WARNING] Skipping card_products (file not found)")
        
        # 3. notices 적재
        if not args.skip_notices:
            notices_file = find_data_file(NOTICES_FILE)
            if notices_file:
                count = load_notices(conn, notices_file, args.limit)
                total_loaded += count
            else:
                print(f"[WARNING] Skipping notices (file not found)")
        
        # 4. 검증
        print("\n[INFO] Verifying loaded data...")
        verification_results = verify_load(conn)
        
        print("\n" + "=" * 60)
        print("적재 완료 및 검증 결과")
        print("=" * 60)
        print(f"service_guide_documents: {verification_results.get('service_guides_count', 0)}건")
        print(f"  - document_number 포함: {verification_results.get('service_guides_with_document_number', 0)}건")
        print(f"  - embedding 포함: {verification_results.get('service_guides_with_embedding', 0)}건")
        print(f"  - structured 포함: {verification_results.get('service_guides_with_structured', 0)}건")
        print(f"card_products: {verification_results.get('card_products_count', 0)}건")
        print(f"  - structured 포함: {verification_results.get('card_products_with_structured', 0)}건")
        print(f"notices: {verification_results.get('notices_count', 0)}건")
        print(f"  - embedding 포함: {verification_results.get('notices_with_embedding', 0)}건")
        print(f"\n[SUCCESS] 총 {total_loaded}건 적재 완료!")
        
    except Exception as e:
        print(f"\n[ERROR] 데이터 적재 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()
        print("\n[INFO] Database connection closed")


if __name__ == "__main__":
    main()
