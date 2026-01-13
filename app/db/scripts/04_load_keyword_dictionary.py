"""
키워드 사전 DB 적재 스크립트

기능:
- keyword_dictionary 테이블 적재 (keywords_dict_v2.json 또는 keywords_dict_with_compound.json)
- keyword_synonyms 테이블 적재 (동의어 매핑)
- 데이터 매핑 및 변환
- 트랜잭션 처리 및 에러 핸들링
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch
from psycopg2.extras import Json as PsycopgJson
from tqdm import tqdm

# 환경 변수 로드
load_dotenv(Path(__file__).parent / '.env', override=False)
load_dotenv(Path(__file__).parent.parent.parent.parent / '.env', override=False)

# 상수
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
# 키워드 사전 파일 경로 (우선순위: with_patterns > with_compound > with_synonyms > v2)
# 개발 환경: data-preprocessing_dev 사용, 프로덕션: data-preprocessing 사용
KEYWORDS_DICT_DIR_DEV = BASE_DIR / "data-preprocessing_dev" / "preprocessing" / "teddycard"
KEYWORDS_DICT_DIR_PROD = BASE_DIR / "data-preprocessing" / "data" / "teddycard"
KEYWORDS_DICT_FILES = [
    "keywords_dict_v2_with_patterns.json",  # 최신 버전
    "keywords_dict_with_compound.json",
    "keywords_dict_with_synonyms.json",
    "keywords_dict_v2.json"
]

# 환경 변수 (기본값 없음, .env 파일에서 필수)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432")) if os.getenv("DB_PORT") else None
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
BATCH_SIZE = int(os.getenv("DB_LOAD_BATCH_SIZE", "100"))

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


def find_keywords_dict_file() -> Optional[Path]:
    """키워드 사전 파일 찾기 (우선순위 적용, dev/prod 모두 확인)"""
    # 먼저 프로덕션 경로 확인 (data-preprocessing/data/teddycard)
    for filename in KEYWORDS_DICT_FILES:
        file_path = KEYWORDS_DICT_DIR_PROD / filename
        if file_path.exists():
            print(f"[INFO] Found keywords dictionary file (PROD): {file_path}")
            return file_path
    
    # 프로덕션 경로에 없으면 개발 경로 확인 (data-preprocessing_dev)
    for filename in KEYWORDS_DICT_FILES:
        file_path = KEYWORDS_DICT_DIR_DEV / filename
        if file_path.exists():
            print(f"[INFO] Found keywords dictionary file (DEV): {file_path}")
            return file_path
    
    print(f"[ERROR] Keywords dictionary file not found")
    print(f"[ERROR] Checked paths:")
    print(f"  - PROD: {KEYWORDS_DICT_DIR_PROD}")
    print(f"  - DEV: {KEYWORDS_DICT_DIR_DEV}")
    print(f"[ERROR] Expected files: {', '.join(KEYWORDS_DICT_FILES)}")
    return None


def load_keyword_dictionary(json_path: Path) -> Dict[str, Any]:
    """키워드 사전 JSON 파일 로드"""
    print(f"[INFO] Loading keywords dictionary from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "keywords" not in data:
        raise ValueError("Invalid keywords dictionary format: 'keywords' key not found")
    
    total_keywords = len(data["keywords"])
    print(f"[INFO] Loaded {total_keywords} keywords")
    return data


def load_keyword_dictionary_table(
    conn: psycopg2_connection,
    keyword_dict: Dict[str, Any]
) -> int:
    """keyword_dictionary 테이블 적재"""
    cursor = conn.cursor()
    
    try:
        print("[INFO] Loading keyword_dictionary table...")
        
        # 배치 처리용 데이터 준비
        batch_data = []
        
        for keyword, data in tqdm(keyword_dict["keywords"].items(), desc="Preparing data"):
            canonical = data.get("canonical", keyword)
            synonyms = data.get("synonyms", [])
            variations = data.get("variations", [])
            compound_patterns = data.get("compound_patterns", [])
            ambiguity_resolution = data.get("ambiguity_resolution", {})
            
            # 각 카테고리별로 레코드 생성
            for cat_info in data.get("categories", []):
                category = cat_info.get("category", "")
                priority = cat_info.get("priority", 5)
                urgency = cat_info.get("urgency", "medium")
                context_hints = cat_info.get("context_hints", [])
                weight = cat_info.get("weight", 1.0)
                
                batch_data.append((
                    canonical,  # keyword
                    category,  # category
                    priority,  # priority
                    urgency,  # urgency
                    context_hints,  # context_hints (TEXT[])
                    float(weight),  # weight
                    synonyms,  # synonyms (TEXT[])
                    variations,  # variations (TEXT[])
                    PsycopgJson(compound_patterns),  # compound_patterns (JSONB)
                    PsycopgJson(ambiguity_resolution),  # ambiguity_rules (JSONB)
                ))
        
        # 배치 삽입
        insert_query = """
            INSERT INTO keyword_dictionary (
                keyword, category, priority, urgency,
                context_hints, weight, synonyms, variations,
                compound_patterns, ambiguity_rules
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        
        execute_batch(cursor, insert_query, batch_data, page_size=BATCH_SIZE)
        conn.commit()
        
        inserted_count = len(batch_data)
        print(f"[SUCCESS] Inserted/Updated {inserted_count} keyword_dictionary records")
        
        cursor.close()
        return inserted_count
        
    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] Failed to load keyword_dictionary table: {e}")
        raise


def load_keyword_synonyms_table(
    conn: psycopg2_connection,
    keyword_dict: Dict[str, Any]
) -> int:
    """keyword_synonyms 테이블 적재"""
    cursor = conn.cursor()
    
    try:
        print("[INFO] Loading keyword_synonyms table...")
        
        # 배치 처리용 데이터 준비
        batch_data = []
        
        for keyword, data in tqdm(keyword_dict["keywords"].items(), desc="Preparing synonyms"):
            canonical = data.get("canonical", keyword)
            synonyms = data.get("synonyms", [])
            
            # 각 카테고리별로 동의어 매핑 생성
            for cat_info in data.get("categories", []):
                category = cat_info.get("category", "")
                
                for synonym in synonyms:
                    batch_data.append((
                        synonym,  # synonym
                        canonical,  # canonical_keyword
                        category,  # category
                    ))
        
        if not batch_data:
            print("[INFO] No synonyms to load")
            cursor.close()
            return 0
        
        # 배치 삽입
        insert_query = """
            INSERT INTO keyword_synonyms (
                synonym, canonical_keyword, category
            ) VALUES (%s, %s, %s)
            ON CONFLICT (synonym, canonical_keyword, category) DO NOTHING
        """
        
        execute_batch(cursor, insert_query, batch_data, page_size=BATCH_SIZE)
        conn.commit()
        
        inserted_count = len(batch_data)
        print(f"[SUCCESS] Inserted {inserted_count} keyword_synonyms records")
        
        cursor.close()
        return inserted_count
        
    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] Failed to load keyword_synonyms table: {e}")
        raise


def verify_load(conn: psycopg2_connection) -> Dict[str, Any]:
    """적재 데이터 검증"""
    cursor = conn.cursor()
    
    results = {}
    
    try:
        # keyword_dictionary 개수 확인
        cursor.execute("SELECT COUNT(*) FROM keyword_dictionary")
        results["keyword_dictionary_count"] = cursor.fetchone()[0]
        
        # keyword_synonyms 개수 확인
        cursor.execute("SELECT COUNT(*) FROM keyword_synonyms")
        results["keyword_synonyms_count"] = cursor.fetchone()[0]
        
        # 카테고리별 키워드 수 확인
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM keyword_dictionary
            GROUP BY category
            ORDER BY count DESC
            LIMIT 10
        """)
        results["top_categories"] = cursor.fetchall()
        
        # 우선순위별 키워드 수 확인
        cursor.execute("""
            SELECT priority, COUNT(*) as count
            FROM keyword_dictionary
            GROUP BY priority
            ORDER BY priority DESC
        """)
        results["priority_distribution"] = cursor.fetchall()
        
        # 긴급성별 키워드 수 확인
        cursor.execute("""
            SELECT urgency, COUNT(*) as count
            FROM keyword_dictionary
            GROUP BY urgency
            ORDER BY urgency DESC
        """)
        results["urgency_distribution"] = cursor.fetchall()
        
        cursor.close()
        return results
        
    except Exception as e:
        cursor.close()
        print(f"[ERROR] Failed to verify load: {e}")
        raise


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("키워드 사전 DB 적재 시작")
    print("=" * 60)
    
    # 1. 키워드 사전 파일 찾기
    keywords_dict_file = find_keywords_dict_file()
    if not keywords_dict_file:
        sys.exit(1)
    
    # 2. 키워드 사전 로드
    try:
        keyword_dict = load_keyword_dictionary(keywords_dict_file)
    except Exception as e:
        print(f"[ERROR] Failed to load keywords dictionary: {e}")
        sys.exit(1)
    
    # 3. DB 연결
    conn = connect_db()
    
    try:
        # 4. keyword_dictionary 테이블 적재
        keyword_count = load_keyword_dictionary_table(conn, keyword_dict)
        
        # 5. keyword_synonyms 테이블 적재
        synonym_count = load_keyword_synonyms_table(conn, keyword_dict)
        
        # 6. 검증
        print("\n[INFO] Verifying loaded data...")
        verification_results = verify_load(conn)
        
        print("\n" + "=" * 60)
        print("적재 완료 및 검증 결과")
        print("=" * 60)
        print(f"keyword_dictionary: {verification_results['keyword_dictionary_count']}건")
        print(f"keyword_synonyms: {verification_results['keyword_synonyms_count']}건")
        print(f"\n상위 10개 카테고리:")
        for category, count in verification_results['top_categories']:
            print(f"  - {category}: {count}건")
        print(f"\n우선순위 분포:")
        for priority, count in verification_results['priority_distribution']:
            print(f"  - Priority {priority}: {count}건")
        print(f"\n긴급성 분포:")
        for urgency, count in verification_results['urgency_distribution']:
            print(f"  - {urgency}: {count}건")
        
        print("\n[SUCCESS] 키워드 사전 DB 적재 완료!")
        
    except Exception as e:
        print(f"\n[ERROR] 키워드 사전 DB 적재 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
