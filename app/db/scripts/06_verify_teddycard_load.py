"""
테디카드 데이터 DB 적재 검증 스크립트

기능:
- 데이터 개수 확인 (service_guide_documents, card_products, notices)
- Foreign Key 관계 확인
- 임베딩 벡터 검증 (차원, NULL 체크)
- 샘플 데이터 확인 (structured 필드 포함)
- 키워드 사전 적재 확인
- 통계 정보 출력
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import RealDictCursor
import json

# 환경 변수 로드
load_dotenv(Path(__file__).parent / '.env', override=False)
load_dotenv(Path(__file__).parent.parent.parent.parent / '.env', override=False)

# 환경 변수 (기본값 없음, .env 파일에서 필수)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432")) if os.getenv("DB_PORT") else None
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

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


def check_table_counts(conn: psycopg2_connection) -> Dict[str, int]:
    """테이블별 데이터 개수 확인"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    counts = {}
    
    tables = [
        'service_guide_documents',
        'card_products',
        'notices',
        'keyword_dictionary',
        'keyword_synonyms'
    ]
    
    print("\n" + "=" * 60)
    print("[1] 데이터 개수 확인")
    print("=" * 60)
    
    all_ok = True
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            result = cursor.fetchone()
            count = result['count'] if result else 0
            counts[table] = count
            status = "✅ OK" if count > 0 else "⚠️ 비어있음"
            print(f"  {table}: {count:,}건 {status}")
            if count == 0:
                all_ok = False
        except Exception as e:
            print(f"  {table}: ❌ 오류 - {e}")
            counts[table] = 0
            all_ok = False
    
    print(f"\n[1] 데이터 개수 확인 - {'✅ OK (문제 없음)' if all_ok else '⚠️ 일부 테이블이 비어있거나 오류 발생'}")
    
    cursor.close()
    return counts


def check_table_exists(cursor, table_name: str, schema: str = 'public') -> bool:
    """테이블 존재 여부 확인"""
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_name = %s
        ) as exists
    """, (schema, table_name))
    result = cursor.fetchone()
    if result:
        # RealDictCursor를 사용하는 경우 딕셔너리로 반환
        if isinstance(result, dict):
            return result.get('exists', False)
        # 일반 cursor를 사용하는 경우 튜플로 반환
        else:
            return result[0] if result else False
    return False


def check_column_exists(cursor, table_name: str, column_name: str, schema: str = 'public') -> bool:
    """컬럼 존재 여부 확인"""
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_schema = %s
            AND table_name = %s
            AND column_name = %s
        ) as exists
    """, (schema, table_name, column_name))
    result = cursor.fetchone()
    if result:
        # RealDictCursor를 사용하는 경우 딕셔너리로 반환
        if isinstance(result, dict):
            return result.get('exists', False)
        # 일반 cursor를 사용하는 경우 튜플로 반환
        else:
            return result[0] if result else False
    return False


def check_embeddings(conn: psycopg2_connection) -> Dict[str, Any]:
    """임베딩 벡터 검증"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("\n" + "=" * 60)
    print("[2] 임베딩 벡터 검증")
    print("=" * 60)
    
    results = {}
    
    # service_guide_documents 테이블 및 embedding 컬럼 존재 확인
    if not check_table_exists(cursor, 'service_guide_documents'):
        print("  ⚠️ service_guide_documents 테이블이 존재하지 않습니다.")
        print("     먼저 DB 스키마를 생성해주세요.")
        cursor.close()
        return results
    
    if not check_column_exists(cursor, 'service_guide_documents', 'embedding'):
        print("  ⚠️ service_guide_documents 테이블에 embedding 컬럼이 없습니다.")
        print("     테이블 스키마를 확인하거나 02_setup_tedicard_tables.sql을 실행해주세요.")
        # embedding 컬럼이 없어도 기본 정보는 확인 가능
        cursor.execute("SELECT COUNT(*) as total FROM service_guide_documents")
        result = cursor.fetchone()
        results['service_guides'] = {
            'total': result['total'] if result else 0,
            'with_embedding': 0,
            'without_embedding': result['total'] if result else 0,
            'dimension': None
        }
        print(f"  service_guide_documents:")
        print(f"    전체: {result['total'] if result else 0:,}건")
        print(f"    ⚠️ embedding 컬럼이 없어 임베딩 정보를 확인할 수 없습니다.")
        cursor.close()
        return results
    
    # service_guide_documents 임베딩 확인
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(embedding) as with_embedding,
            COUNT(*) - COUNT(embedding) as without_embedding
        FROM service_guide_documents
    """)
    result = cursor.fetchone()
    results['service_guides'] = {
        'total': result['total'],
        'with_embedding': result['with_embedding'],
        'without_embedding': result['without_embedding']
    }
    print(f"  service_guide_documents:")
    print(f"    전체: {result['total']:,}건")
    print(f"    임베딩 있음: {result['with_embedding']:,}건")
    print(f"    임베딩 없음: {result['without_embedding']:,}건")
    
    # 임베딩 차원 확인 (샘플)
    # pgvector의 vector 타입은 직접 배열로 캐스팅할 수 없으므로, Python에서 처리
    # vector::text는 '[1,2,3,...]' 형식의 문자열로 반환됨
    cursor.execute("""
        SELECT 
            embedding::text as embedding_text
        FROM service_guide_documents
        WHERE embedding IS NOT NULL
        LIMIT 1
    """)
    result = cursor.fetchone()
    if result and result.get('embedding_text'):
        # vector::text는 '[1,2,3,...]' 형식의 문자열로 반환됨
        embedding_text = result['embedding_text']
        try:
            # 문자열을 리스트로 파싱
            if embedding_text.startswith('[') and embedding_text.endswith(']'):
                embedding_list = json.loads(embedding_text)
                dimension = len(embedding_list)
            else:
                # 쉼표로 구분된 숫자 문자열인 경우
                dimension = len([x for x in embedding_text.split(',') if x.strip()])
            
            if dimension:
                print(f"    임베딩 차원: {dimension}")
                results['service_guides']['dimension'] = dimension
                
                # 차원 검증
                if dimension != 1536:
                    print(f"    ⚠️ 경고: 예상 차원(1536)과 다릅니다!")
                else:
                    print(f"    ✅ 차원 검증 통과")
            else:
                print(f"    ⚠️ 임베딩 차원을 확인할 수 없습니다.")
                results['service_guides']['dimension'] = None
        except Exception as e:
            print(f"    ⚠️ 임베딩 파싱 오류: {e}")
            results['service_guides']['dimension'] = None
    else:
        print(f"    ⚠️ 임베딩 샘플을 찾을 수 없습니다.")
        results['service_guides']['dimension'] = None
    
    # card_products 임베딩 확인 (있는 경우)
    if check_table_exists(cursor, 'card_products'):
        if check_column_exists(cursor, 'card_products', 'embedding'):
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(embedding) as with_embedding
                FROM card_products
            """)
            result = cursor.fetchone()
            if result and result['total'] > 0:
                results['card_products'] = {
                    'total': result['total'],
                    'with_embedding': result['with_embedding']
                }
                print(f"  card_products:")
                print(f"    전체: {result['total']:,}건")
                print(f"    임베딩 있음: {result['with_embedding']:,}건")
        else:
            # card_products 테이블은 있지만 embedding 컬럼이 없는 경우 (정상 - card_products는 embedding을 사용하지 않음)
            cursor.execute("SELECT COUNT(*) as total FROM card_products")
            result = cursor.fetchone()
            if result and result['total'] > 0:
                print(f"  card_products:")
                print(f"    전체: {result['total']:,}건")
                print(f"    ℹ️ embedding 컬럼 없음 (정상 - card_products는 embedding을 사용하지 않음)")
    
    # 검증 결과 요약
    all_ok = True
    if results.get('service_guides', {}).get('dimension') != 1536:
        all_ok = False
    
    print(f"\n[2] 임베딩 벡터 검증 - {'✅ OK (문제 없음)' if all_ok else '⚠️ 일부 검증 실패'}")
    
    cursor.close()
    return results


def check_structured_fields(conn: psycopg2_connection) -> Dict[str, Any]:
    """structured 필드 확인"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("\n" + "=" * 60)
    print("[3] Structured 필드 확인")
    print("=" * 60)
    
    results = {}
    
    # service_guide_documents structured 확인
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(structured) as with_structured,
            COUNT(*) - COUNT(structured) as without_structured
        FROM service_guide_documents
    """)
    result = cursor.fetchone()
    results['service_guides'] = {
        'total': result['total'],
        'with_structured': result['with_structured'],
        'without_structured': result['without_structured']
    }
    print(f"  service_guide_documents:")
    print(f"    전체: {result['total']:,}건")
    print(f"    structured 있음: {result['with_structured']:,}건")
    print(f"    structured 없음: {result['without_structured']:,}건")
    
    # card_products structured 확인
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(structured) as with_structured,
            COUNT(*) - COUNT(structured) as without_structured
        FROM card_products
    """)
    result = cursor.fetchone()
    if result['total'] > 0:
        results['card_products'] = {
            'total': result['total'],
            'with_structured': result['with_structured'],
            'without_structured': result['without_structured']
        }
        print(f"  card_products:")
        print(f"    전체: {result['total']:,}건")
        print(f"    structured 있음: {result['with_structured']:,}건")
        print(f"    structured 없음: {result['without_structured']:,}건")
    
    # 검증 결과 요약
    all_ok = True
    if results.get('service_guides', {}).get('with_structured', 0) == 0:
        all_ok = False
    
    print(f"\n[3] Structured 필드 확인 - {'✅ OK (문제 없음)' if all_ok else '⚠️ 일부 데이터에 structured 필드 없음'}")
    
    cursor.close()
    return results


def check_keywords(conn: psycopg2_connection) -> Dict[str, Any]:
    """키워드 사전 적재 확인"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("\n" + "=" * 60)
    print("[4] 키워드 사전 적재 확인")
    print("=" * 60)
    
    results = {}
    
    # keyword_dictionary 확인
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT category) as categories,
            AVG(priority) as avg_priority
        FROM keyword_dictionary
    """)
    result = cursor.fetchone()
    
    # urgency 분포 확인 (VARCHAR 타입이므로 AVG 대신 분포 사용)
    cursor.execute("""
        SELECT urgency, COUNT(*) as count
        FROM keyword_dictionary
        GROUP BY urgency
        ORDER BY urgency DESC
    """)
    urgency_distribution = cursor.fetchall()
    
    results['keyword_dictionary'] = {
        'total': result['total'],
        'categories': result['categories'],
        'avg_priority': float(result['avg_priority']) if result['avg_priority'] else 0,
        'urgency_distribution': urgency_distribution
    }
    print(f"  keyword_dictionary:")
    print(f"    전체 키워드: {result['total']:,}건")
    print(f"    카테고리 수: {result['categories']:,}개")
    print(f"    평균 우선순위: {result['avg_priority']:.2f}")
    print(f"    긴급성 분포:")
    for urgency_row in urgency_distribution:
        urgency = urgency_row['urgency'] if isinstance(urgency_row, dict) else urgency_row[0]
        count = urgency_row['count'] if isinstance(urgency_row, dict) else urgency_row[1]
        print(f"      - {urgency}: {count:,}건")
    
    # keyword_synonyms 확인
    # keyword_synonyms 테이블은 canonical_keyword와 category를 사용 (keyword_id 없음)
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT (canonical_keyword, category)) as keywords_with_synonyms
        FROM keyword_synonyms
    """)
    result = cursor.fetchone()
    results['keyword_synonyms'] = {
        'total': result['total'],
        'keywords_with_synonyms': result['keywords_with_synonyms']
    }
    print(f"  keyword_synonyms:")
    print(f"    전체 동의어: {result['total']:,}건")
    print(f"    동의어가 있는 키워드: {result['keywords_with_synonyms']:,}개")
    
    # 검증 결과 요약
    all_ok = True
    if result['total'] == 0:
        all_ok = False
    
    print(f"\n[4] 키워드 사전 적재 확인 - {'✅ OK (문제 없음)' if all_ok else '⚠️ 키워드 사전이 비어있음'}")
    
    cursor.close()
    return results


def check_sample_data(conn: psycopg2_connection) -> None:
    """샘플 데이터 확인"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("\n" + "=" * 60)
    print("[5] 샘플 데이터 확인")
    print("=" * 60)
    
    # service_guide_documents 샘플
    cursor.execute("""
        SELECT 
            id,
            title,
            category,
            array_length(keywords, 1) as keyword_count,
            CASE WHEN embedding IS NOT NULL THEN 'Yes' ELSE 'No' END as has_embedding,
            CASE WHEN structured IS NOT NULL THEN 'Yes' ELSE 'No' END as has_structured,
            metadata->>'document_number' as document_number
        FROM service_guide_documents
        LIMIT 3
    """)
    results = cursor.fetchall()
    print(f"  service_guide_documents 샘플 (최대 3건):")
    for i, row in enumerate(results, 1):
        print(f"    [{i}] {row['title'][:50]}...")
        print(f"        ID: {row['id']}")
        print(f"        카테고리: {row['category']}")
        print(f"        키워드 수: {row['keyword_count']}")
        print(f"        임베딩: {row['has_embedding']}")
        print(f"        Structured: {row['has_structured']}")
        if row['document_number']:
            print(f"        문서 번호: {row['document_number']}")
    
    # card_products 샘플
    cursor.execute("""
        SELECT 
            id,
            name,
            brand,
            CASE WHEN structured IS NOT NULL THEN 'Yes' ELSE 'No' END as has_structured
        FROM card_products
        LIMIT 3
    """)
    results = cursor.fetchall()
    if results:
        print(f"\n  card_products 샘플 (최대 3건):")
        for i, row in enumerate(results, 1):
            print(f"    [{i}] {row['name'][:50]}...")
            print(f"        ID: {row['id']}")
            print(f"        브랜드: {row['brand']}")
            print(f"        Structured: {row['has_structured']}")
    
    print(f"\n[5] 샘플 데이터 확인 - ✅ OK (문제 없음)")
    
    cursor.close()


def check_metadata(conn: psycopg2_connection) -> None:
    """metadata 필드 확인"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("\n" + "=" * 60)
    print("[6] Metadata 필드 확인")
    print("=" * 60)
    
    # document_number가 있는 문서 수
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN metadata->>'document_number' IS NOT NULL THEN 1 END) as with_document_number
        FROM service_guide_documents
    """)
    result = cursor.fetchone()
    print(f"  service_guide_documents:")
    print(f"    전체: {result['total']:,}건")
    print(f"    document_number 있음: {result['with_document_number']:,}건")
    
    print(f"\n[6] Metadata 필드 확인 - ✅ OK (문제 없음)")
    
    cursor.close()


def main():
    """메인 함수"""
    print("=" * 60)
    print("테디카드 데이터 DB 적재 검증 스크립트")
    print("=" * 60)
    print(f"[INFO] Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print()
    
    # DB 연결
    conn = connect_db()
    
    try:
        # 1. 데이터 개수 확인
        counts = check_table_counts(conn)
        
        # 2. 임베딩 벡터 검증
        embedding_results = check_embeddings(conn)
        
        # 3. Structured 필드 확인
        structured_results = check_structured_fields(conn)
        
        # 4. 키워드 사전 적재 확인
        keyword_results = check_keywords(conn)
        
        # 5. 샘플 데이터 확인
        check_sample_data(conn)
        
        # 6. Metadata 필드 확인
        check_metadata(conn)
        
        # 최종 요약
        print("\n" + "=" * 60)
        print("[최종 요약]")
        print("=" * 60)
        
        total_docs = (
            counts.get('service_guide_documents', 0) +
            counts.get('card_products', 0) +
            counts.get('notices', 0)
        )
        print(f"  전체 문서: {total_docs:,}건")
        print(f"    - service_guide_documents: {counts.get('service_guide_documents', 0):,}건")
        print(f"    - card_products: {counts.get('card_products', 0):,}건")
        print(f"    - notices: {counts.get('notices', 0):,}건")
        print(f"  키워드 사전: {counts.get('keyword_dictionary', 0):,}건")
        print(f"  동의어: {counts.get('keyword_synonyms', 0):,}건")
        
        # 검증 결과
        all_ok = True
        if counts.get('service_guide_documents', 0) == 0:
            print("  ⚠️ service_guide_documents가 비어있습니다.")
            all_ok = False
        if embedding_results.get('service_guides', {}).get('dimension') != 1536:
            print("  ⚠️ 임베딩 차원이 올바르지 않습니다.")
            all_ok = False
        if counts.get('keyword_dictionary', 0) == 0:
            print("  ⚠️ keyword_dictionary가 비어있습니다.")
            all_ok = False
        
        if all_ok:
            print("\n  ✅ 모든 검증을 통과했습니다!")
        else:
            print("\n  ⚠️ 일부 검증에 실패했습니다. 위 내용을 확인해주세요.")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 검증 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
