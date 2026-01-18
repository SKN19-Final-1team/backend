"""
CALL:ACT 통합 DB 설정 및 데이터 적재 스크립트

기능:
- DB 스키마 생성 (기본 테이블)
- 테디카드 테이블 생성 및 수정
- 키워드 사전 테이블 생성
- 키워드 사전 데이터 적재
- 테디카드 데이터 적재
- 검증

사용법:
    python 01_setup_callact_db.py [옵션]

옵션:
    --skip-schema: 스키마 생성 건너뛰기
    --skip-keywords: 키워드 사전 적재 건너뛰기
    --skip-teddycard: 테디카드 데이터 적재 건너뛰기
    --verify-only: 검증만 실행
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, Json as PsycopgJson, RealDictCursor
from tqdm import tqdm

# 환경 변수 로드
load_dotenv(Path(__file__).parent / '.env', override=False)
load_dotenv(Path(__file__).parent.parent.parent.parent / '.env', override=False)

# 상수
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent

# 데이터 파일 경로
DATA_DIR_PROD = BASE_DIR / "data-preprocessing" / "data" / "teddycard"
DATA_DIR_DEV = BASE_DIR / "data-preprocessing_dev" / "preprocessing" / "output"
KEYWORDS_DICT_DIR_PROD = BASE_DIR / "data-preprocessing" / "data" / "teddycard"
KEYWORDS_DICT_DIR_DEV = BASE_DIR / "data-preprocessing_dev" / "preprocessing" / "teddycard"

# 환경 변수
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


def load_sql_file(file_path: Path) -> str:
    """SQL 파일 읽기"""
    if not file_path.exists():
        print(f"[ERROR] SQL file not found: {file_path}")
        sys.exit(1)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def execute_sql_script(conn: psycopg2_connection, sql_script: str, description: str = ""):
    """SQL 스크립트 실행"""
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql_script)
        conn.commit()
        if description:
            print(f"[INFO] {description} 완료!")
    except Exception as e:
        conn.rollback()
        # 일부 에러는 무시 가능 (이미 존재하는 객체 등)
        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
            if description:
                print(f"[WARNING] {description} - 일부 객체가 이미 존재합니다. (무시됨)")
        else:
            print(f"[ERROR] {description} 실패: {e}")
            raise
    finally:
        cursor.close()


def setup_basic_schema(conn: psycopg2_connection):
    """기본 DB 스키마 생성"""
    print("\n" + "=" * 60)
    print("[1/7] 기본 DB 스키마 생성")
    print("=" * 60)
    
    sql_file = SCRIPTS_DIR / "db_setup.sql"
    sql_script = load_sql_file(sql_file)
    execute_sql_script(conn, sql_script, "기본 DB 스키마 생성")
    
    # 테이블 목록 확인
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = cursor.fetchall()
    print(f"[INFO] 생성된 테이블: {len(tables)}개")
    cursor.close()


def setup_teddycard_tables(conn: psycopg2_connection):
    """테디카드 테이블 생성"""
    print("\n" + "=" * 60)
    print("[2/7] 테디카드 테이블 생성")
    print("=" * 60)
    
    # 테이블 생성
    sql_file = SCRIPTS_DIR / "02_setup_tedicard_tables.sql"
    sql_script = load_sql_file(sql_file)
    execute_sql_script(conn, sql_script, "테디카드 테이블 생성")
    
    # 테이블 수정 (기존 테이블에 컬럼 추가)
    sql_file = SCRIPTS_DIR / "02_alter_tedicard_tables.sql"
    sql_script = load_sql_file(sql_file)
    execute_sql_script(conn, sql_script, "테디카드 테이블 수정")
    
    # ID 길이 수정
    sql_file = SCRIPTS_DIR / "02_fix_id_length.sql"
    sql_script = load_sql_file(sql_file)
    execute_sql_script(conn, sql_script, "ID 컬럼 길이 수정")


def setup_keyword_dictionary_tables(conn: psycopg2_connection):
    """키워드 사전 테이블 생성"""
    print("\n" + "=" * 60)
    print("[3/7] 키워드 사전 테이블 생성")
    print("=" * 60)
    
    sql_file = SCRIPTS_DIR / "03_setup_keyword_dictionary.sql"
    sql_script = load_sql_file(sql_file)
    execute_sql_script(conn, sql_script, "키워드 사전 테이블 생성")


def find_keywords_dict_file() -> Optional[Path]:
    """키워드 사전 파일 찾기"""
    KEYWORDS_DICT_FILES = [
        "keywords_dict_v2_with_patterns.json",
        "keywords_dict_with_compound.json",
        "keywords_dict_with_synonyms.json",
        "keywords_dict_v2.json"
    ]
    
    # 프로덕션 경로 확인
    for filename in KEYWORDS_DICT_FILES:
        file_path = KEYWORDS_DICT_DIR_PROD / filename
        if file_path.exists():
            print(f"[INFO] Found keywords dictionary file (PROD): {file_path}")
            return file_path
    
    # 개발 경로 확인
    for filename in KEYWORDS_DICT_FILES:
        file_path = KEYWORDS_DICT_DIR_DEV / filename
        if file_path.exists():
            print(f"[INFO] Found keywords dictionary file (DEV): {file_path}")
            return file_path
    
    print(f"[ERROR] Keywords dictionary file not found")
    return None


def load_keyword_dictionary(conn: psycopg2_connection):
    """키워드 사전 데이터 적재"""
    print("\n" + "=" * 60)
    print("[4/7] 키워드 사전 데이터 적재")
    print("=" * 60)
    
    # 키워드 사전 파일 찾기
    KEYWORDS_DICT_FILES = [
        "keywords_dict_v2_with_patterns.json",
        "keywords_dict_with_compound.json",
        "keywords_dict_with_synonyms.json",
        "keywords_dict_v2.json"
    ]
    
    keywords_file = None
    # 프로덕션 경로 확인
    for filename in KEYWORDS_DICT_FILES:
        file_path = KEYWORDS_DICT_DIR_PROD / filename
        if file_path.exists():
            keywords_file = file_path
            break
    
    # 개발 경로 확인
    if not keywords_file:
        for filename in KEYWORDS_DICT_FILES:
            file_path = KEYWORDS_DICT_DIR_DEV / filename
            if file_path.exists():
                keywords_file = file_path
                break
    
    if not keywords_file:
        print("[ERROR] 키워드 사전 파일을 찾을 수 없습니다.")
        return False
    
    print(f"[INFO] 키워드 사전 파일: {keywords_file}")
    
    # JSON 파일 로드
    with open(keywords_file, 'r', encoding='utf-8') as f:
        keyword_dict = json.load(f)
    
    # 키워드 사전 구조 확인 (keywords_dict_v2 형식)
    if "keywords" in keyword_dict:
        keywords_data = keyword_dict["keywords"]
        print(f"[INFO] 키워드 사전 파일 로드: {len(keywords_data)}개 키워드")
    else:
        # 리스트 형식인 경우
        keywords_data = keyword_dict if isinstance(keyword_dict, list) else []
        print(f"[INFO] 키워드 사전 파일 로드: {len(keywords_data)}개 키워드")
    
    cursor = conn.cursor()
    
    try:
        # keyword_dictionary 테이블 적재
        print("[INFO] keyword_dictionary 테이블 적재 중...")
        
        keyword_dict_insert = """
            INSERT INTO keyword_dictionary (
                keyword, category, priority, urgency, context_hints,
                weight, synonyms, variations, compound_patterns, ambiguity_rules
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (keyword, category) 
            DO UPDATE SET
                priority = EXCLUDED.priority,
                urgency = EXCLUDED.urgency,
                context_hints = EXCLUDED.context_hints,
                weight = EXCLUDED.weight,
                synonyms = EXCLUDED.synonyms,
                variations = EXCLUDED.variations,
                compound_patterns = EXCLUDED.compound_patterns,
                ambiguity_rules = EXCLUDED.ambiguity_rules,
                updated_at = NOW()
        """
        
        keyword_batch = []
        
        # keywords_dict_v2 형식 처리
        if isinstance(keywords_data, dict):
            for keyword, data in tqdm(keywords_data.items(), desc="키워드 처리"):
                canonical = data.get("canonical", keyword)
                
                # 각 카테고리별로 키워드 추가
                for cat_info in data.get("categories", []):
                    category = cat_info.get("category", "")
                    priority = cat_info.get("priority", 5)
                    urgency = cat_info.get("urgency", "medium")
                    context_hints = cat_info.get("context_hints", [])
                    weight = float(cat_info.get("weight", 1.0))
                    synonyms = data.get("synonyms", [])
                    variations = data.get("variations", [])
                    compound_patterns = cat_info.get("compound_patterns")
                    ambiguity_rules = cat_info.get("ambiguity_rules")
                    
                    keyword_batch.append((
                        canonical, category, priority, urgency, context_hints,
                        weight, synonyms, variations,
                        PsycopgJson(compound_patterns) if compound_patterns else None,
                        PsycopgJson(ambiguity_rules) if ambiguity_rules else None
                    ))
        else:
            # 리스트 형식 처리
            for keyword_entry in tqdm(keywords_data, desc="키워드 처리"):
                keyword = keyword_entry.get('keyword', '')
                category = keyword_entry.get('category', '')
                priority = keyword_entry.get('priority', 5)
                urgency = keyword_entry.get('urgency', 'medium')
                context_hints = keyword_entry.get('context_hints', [])
                weight = float(keyword_entry.get('weight', 1.0))
                synonyms = keyword_entry.get('synonyms', [])
                variations = keyword_entry.get('variations', [])
                compound_patterns = keyword_entry.get('compound_patterns')
                ambiguity_rules = keyword_entry.get('ambiguity_rules')
                
                keyword_batch.append((
                    keyword, category, priority, urgency, context_hints,
                    weight, synonyms, variations,
                    PsycopgJson(compound_patterns) if compound_patterns else None,
                    PsycopgJson(ambiguity_rules) if ambiguity_rules else None
                ))
        
        # 키워드 적재
        if keyword_batch:
            execute_batch(cursor, keyword_dict_insert, keyword_batch, page_size=BATCH_SIZE)
            conn.commit()
            print(f"[INFO] 키워드 적재 완료: {len(keyword_batch)}개")
        else:
            print("[WARNING] 적재할 키워드가 없습니다.")
        
        # keyword_synonyms 테이블 적재
        print("[INFO] keyword_synonyms 테이블 적재 중...")
        
        keyword_synonyms_insert = """
            INSERT INTO keyword_synonyms (synonym, canonical_keyword, category)
            VALUES (%s, %s, %s)
            ON CONFLICT (synonym, canonical_keyword, category) DO NOTHING
        """
        
        synonym_batch = []
        
        # keywords_dict_v2 형식 처리
        if isinstance(keywords_data, dict):
            for keyword, data in keywords_data.items():
                canonical = data.get("canonical", keyword)
                synonyms = data.get("synonyms", [])
                
                # 각 카테고리별로 동의어 매핑 생성
                for cat_info in data.get("categories", []):
                    category = cat_info.get("category", "")
                    
                    for synonym in synonyms:
                        if synonym and synonym != canonical:
                            synonym_batch.append((synonym, canonical, category))
        else:
            # 리스트 형식 처리
            for keyword_entry in keywords_data:
                keyword = keyword_entry.get('keyword', '')
                category = keyword_entry.get('category', '')
                synonyms = keyword_entry.get('synonyms', [])
                
                for synonym in synonyms:
                    if synonym and synonym != keyword:
                        synonym_batch.append((synonym, keyword, category))
        
        # 동의어 적재
        if synonym_batch:
            execute_batch(cursor, keyword_synonyms_insert, synonym_batch, page_size=BATCH_SIZE)
            conn.commit()
            print(f"[INFO] 동의어 적재 완료: {len(synonym_batch)}개")
        else:
            print("[INFO] 적재할 동의어가 없습니다.")
        
        cursor.close()
        return True
        
    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] 키워드 사전 적재 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_data_file(filename: str) -> Optional[Path]:
    """데이터 파일 찾기"""
    file_path = DATA_DIR_PROD / filename
    if file_path.exists():
        print(f"[INFO] Found data file (PROD): {file_path}")
        return file_path
    
    file_path = DATA_DIR_DEV / filename
    if file_path.exists():
        print(f"[INFO] Found data file (DEV): {file_path}")
        return file_path
    
    print(f"[ERROR] Data file not found: {filename}")
    return None


def extract_document_number(title: str) -> Optional[str]:
    """제목에서 문서 번호 추출 (예: 제1조, 제2조)"""
    if not title:
        return None
    match = re.search(r'제\d+조', title)
    return match.group(0) if match else None


def prepare_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    """metadata 준비"""
    metadata = doc.get("metadata", {}).copy()
    title = doc.get("title", "")
    document_number = extract_document_number(title)
    if document_number:
        metadata["document_number"] = document_number
    return metadata


def map_service_guide_data(doc: Dict[str, Any]) -> Tuple:
    """service_guide_documents 데이터 매핑"""
    doc_id = doc.get("id", "")
    document_type = doc.get("document_type", "service_guide")
    category = doc.get("category", "")
    title = doc.get("title", "")
    content = doc.get("content", "") or doc.get("text", "")
    keywords = doc.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    embedding = doc.get("embedding")
    embedding_str = "[" + ",".join(map(str, embedding)) + "]" if embedding else None
    metadata = prepare_metadata(doc)
    document_source = doc.get("document_source", "") or metadata.get("original_source", "")
    priority = doc.get("priority", "normal")
    structured = doc.get("structured")
    usage_count = doc.get("usage_count", 0)
    last_used = None
    return (
        doc_id, document_type, category, title, content, keywords,
        embedding_str, PsycopgJson(metadata), document_source, priority,
        usage_count, last_used, PsycopgJson(structured) if structured else None
    )


def map_card_product_data(doc: Dict[str, Any]) -> Tuple:
    """card_products 데이터 매핑"""
    doc_id = doc.get("id", "")
    name = doc.get("name", "")
    card_type = doc.get("card_type", "credit")
    brand = doc.get("brand", "local")
    annual_fee_domestic = doc.get("annual_fee_domestic")
    annual_fee_global = doc.get("annual_fee_global")
    performance_condition = doc.get("performance_condition", "")
    main_benefits = doc.get("main_benefits", "")
    status = doc.get("status", "active")
    metadata = doc.get("metadata", {})
    structured = doc.get("structured")
    return (
        doc_id, name, card_type, brand, annual_fee_domestic, annual_fee_global,
        performance_condition, main_benefits, status,
        PsycopgJson(metadata) if metadata else None,
        PsycopgJson(structured) if structured else None
    )


def map_notice_data(doc: Dict[str, Any]) -> Tuple:
    """notices 데이터 매핑"""
    doc_id = doc.get("id", "")
    title = doc.get("title", "")
    content = doc.get("content", "") or doc.get("text", "")
    category = doc.get("category", "system")
    priority = doc.get("priority", "normal")
    is_pinned = doc.get("is_pinned", False)
    start_date = doc.get("start_date")
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = doc.get("end_date")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
    status = doc.get("status", "active")
    created_by = doc.get("created_by", "")
    keywords = doc.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    embedding = doc.get("embedding")
    embedding_str = "[" + ",".join(map(str, embedding)) + "]" if embedding else None
    metadata = doc.get("metadata", {})
    return (
        doc_id, title, content, category, priority, is_pinned,
        start_date, end_date, status, created_by, keywords,
        embedding_str, PsycopgJson(metadata) if metadata else None
    )


def load_teddycard_data(conn: psycopg2_connection):
    """테디카드 데이터 적재"""
    print("\n" + "=" * 60)
    print("[5/7] 테디카드 데이터 적재")
    print("=" * 60)
    
    # 데이터 파일 찾기
    service_guides_file = find_data_file("teddycard_service_guides_with_embeddings.json")
    card_products_file = find_data_file("teddycard_card_products_with_embeddings.json")
    notices_file = find_data_file("teddycard_notices_with_embeddings.json")
    
    if not all([service_guides_file, card_products_file, notices_file]):
        print("[ERROR] 일부 데이터 파일을 찾을 수 없습니다.")
        if not service_guides_file:
            print(f"  - teddycard_service_guides_with_embeddings.json")
            print(f"    찾은 경로: {DATA_DIR_PROD} 또는 {DATA_DIR_DEV}")
        if not card_products_file:
            print(f"  - teddycard_card_products_with_embeddings.json")
            print(f"    찾은 경로: {DATA_DIR_PROD} 또는 {DATA_DIR_DEV}")
        if not notices_file:
            print(f"  - teddycard_notices_with_embeddings.json")
            print(f"    찾은 경로: {DATA_DIR_PROD} 또는 {DATA_DIR_DEV}")
        print("[INFO] 데이터 파일이 없으면 --skip-teddycard 옵션을 사용하여 건너뛸 수 있습니다.")
        return False
    
    cursor = conn.cursor()
    
    # service_guide_documents 적재
    print("[INFO] service_guide_documents 적재 중...")
    with open(service_guides_file, 'r', encoding='utf-8') as f:
        service_guides = json.load(f)
    
    insert_service_guide = """
        INSERT INTO service_guide_documents (
            id, document_type, category, title, content, keywords,
            embedding, metadata, document_source, priority,
            usage_count, last_used, structured
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
    
    service_guide_batch = [map_service_guide_data(doc) for doc in service_guides]
    execute_batch(cursor, insert_service_guide, service_guide_batch, page_size=BATCH_SIZE)
    conn.commit()
    print(f"[INFO] service_guide_documents 적재 완료: {len(service_guide_batch)}개")
    
    # card_products 적재
    print("[INFO] card_products 적재 중...")
    with open(card_products_file, 'r', encoding='utf-8') as f:
        card_products = json.load(f)
    
    insert_card_product = """
        INSERT INTO card_products (
            id, name, card_type, brand, annual_fee_domestic, annual_fee_global,
            performance_condition, main_benefits, status, metadata, structured
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
    
    card_product_batch = [map_card_product_data(doc) for doc in card_products]
    execute_batch(cursor, insert_card_product, card_product_batch, page_size=BATCH_SIZE)
    conn.commit()
    print(f"[INFO] card_products 적재 완료: {len(card_product_batch)}개")
    
    # notices 적재
    print("[INFO] notices 적재 중...")
    with open(notices_file, 'r', encoding='utf-8') as f:
        notices = json.load(f)
    
    insert_notice = """
        INSERT INTO notices (
            id, title, content, category, priority, is_pinned,
            start_date, end_date, status, created_by, keywords, embedding, metadata
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
            keywords = EXCLUDED.keywords,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
    """
    
    notice_batch = [map_notice_data(doc) for doc in notices]
    execute_batch(cursor, insert_notice, notice_batch, page_size=BATCH_SIZE)
    conn.commit()
    print(f"[INFO] notices 적재 완료: {len(notice_batch)}개")
    
    cursor.close()
    return True


def check_required_files() -> Tuple[bool, List[str]]:
    """필수 파일 존재 여부 확인"""
    required_files = [
        SCRIPTS_DIR / "db_setup.sql",
        SCRIPTS_DIR / "02_setup_tedicard_tables.sql",
        SCRIPTS_DIR / "02_alter_tedicard_tables.sql",
        SCRIPTS_DIR / "02_fix_id_length.sql",
        SCRIPTS_DIR / "03_setup_keyword_dictionary.sql",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    return len(missing_files) == 0, missing_files


def check_data_files() -> Tuple[bool, Dict[str, Optional[Path]]]:
    """데이터 파일 존재 여부 확인"""
    data_files = {
        "service_guides": None,
        "card_products": None,
        "notices": None,
        "keywords_dict": None
    }
    
    # 테디카드 데이터 파일 확인
    service_guides_file = find_data_file("teddycard_service_guides_with_embeddings.json")
    card_products_file = find_data_file("teddycard_card_products_with_embeddings.json")
    notices_file = find_data_file("teddycard_notices_with_embeddings.json")
    
    data_files["service_guides"] = service_guides_file
    data_files["card_products"] = card_products_file
    data_files["notices"] = notices_file
    
    # 키워드 사전 파일 확인
    keywords_file = find_keywords_dict_file()
    data_files["keywords_dict"] = keywords_file
    
    all_exist = all([
        service_guides_file,
        card_products_file,
        notices_file,
        keywords_file
    ])
    
    return all_exist, data_files


def print_checklist(skip_schema: bool, skip_keywords: bool, skip_teddycard: bool):
    """실행 전 체크리스트 출력"""
    print("\n" + "=" * 60)
    print("실행 전 체크리스트")
    print("=" * 60)
    
    # 필수 파일 확인
    files_ok, missing_files = check_required_files()
    if files_ok:
        print("✅ 필수 SQL 파일: 모두 존재")
    else:
        print("❌ 필수 SQL 파일 누락:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    # 데이터 파일 확인
    if not skip_keywords or not skip_teddycard:
        data_ok, data_files = check_data_files()
        if data_ok:
            print("✅ 데이터 파일: 모두 존재")
            print(f"   - Service Guides: {data_files['service_guides']}")
            print(f"   - Card Products: {data_files['card_products']}")
            print(f"   - Notices: {data_files['notices']}")
            print(f"   - Keywords Dict: {data_files['keywords_dict']}")
        else:
            print("⚠️ 데이터 파일 일부 누락:")
            if not data_files['service_guides']:
                print("   - teddycard_service_guides_with_embeddings.json")
            if not data_files['card_products']:
                print("   - teddycard_card_products_with_embeddings.json")
            if not data_files['notices']:
                print("   - teddycard_notices_with_embeddings.json")
            if not data_files['keywords_dict']:
                print("   - keywords_dict_*.json")
            if not skip_keywords and not data_files['keywords_dict']:
                print("   ⚠️ 키워드 사전 파일이 없으면 키워드 적재를 건너뜁니다.")
            if not skip_teddycard and not all([data_files['service_guides'], data_files['card_products'], data_files['notices']]):
                print("   ⚠️ 테디카드 데이터 파일이 없으면 테디카드 적재를 건너뜁니다.")
    
    # 환경 변수 확인
    env_vars_ok = all([DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME])
    if env_vars_ok:
        print("✅ 환경 변수: 모두 설정됨")
        print(f"   - DB: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    else:
        print("❌ 환경 변수: 일부 누락")
        return False
    
    print("=" * 60)
    return True


def verify_load(conn: psycopg2_connection):
    """데이터 적재 검증"""
    print("\n" + "=" * 60)
    print("[6/7] 데이터 적재 검증")
    print("=" * 60)
    
    # 06_verify_teddycard_load.py의 함수들을 import하여 사용
    # 간단한 검증만 수행
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # 데이터 개수 확인
    tables = ['service_guide_documents', 'card_products', 'notices', 'keyword_dictionary', 'keyword_synonyms']
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            result = cursor.fetchone()
            count = result['count'] if result else 0
            print(f"  {table}: {count:,}건")
        except Exception as e:
            print(f"  {table}: ❌ 오류 - {e}")
    
    cursor.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='CALL:ACT 통합 DB 설정 및 데이터 적재')
    parser.add_argument('--skip-schema', action='store_true', help='스키마 생성 건너뛰기')
    parser.add_argument('--skip-keywords', action='store_true', help='키워드 사전 적재 건너뛰기')
    parser.add_argument('--skip-teddycard', action='store_true', help='테디카드 데이터 적재 건너뛰기')
    parser.add_argument('--verify-only', action='store_true', help='검증만 실행')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CALL:ACT 통합 DB 설정 및 데이터 적재 스크립트")
    print("=" * 60)
    print(f"[INFO] Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    # 실행 전 체크리스트 확인
    if not print_checklist(args.skip_schema, args.skip_keywords, args.skip_teddycard):
        print("\n[ERROR] 체크리스트 확인 실패. 위 내용을 확인해주세요.")
        sys.exit(1)
    
    print()
    
    # DB 연결
    conn = connect_db()
    
    try:
        if args.verify_only:
            verify_load(conn)
        else:
            # 1. 기본 스키마 생성
            if not args.skip_schema:
                setup_basic_schema(conn)
                setup_teddycard_tables(conn)
                setup_keyword_dictionary_tables(conn)
            
            # 2. 키워드 사전 적재
            if not args.skip_keywords:
                load_keyword_dictionary(conn)
            
            # 3. 테디카드 데이터 적재
            if not args.skip_teddycard:
                load_teddycard_data(conn)
            
            # 4. 검증
            verify_load(conn)
        
        print("\n" + "=" * 60)
        print("[SUCCESS] 모든 작업이 완료되었습니다!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 작업 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
