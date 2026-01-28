"""
하나카드 데이터 DB 적재 검증 스크립트

기능:
- 적재된 데이터 개수 확인
- 임베딩 벡터 검증
- Foreign Key 관계 확인
- 샘플 유사도 검색 테스트
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import RealDictCursor
from openai import OpenAI

# 환경 변수 로드
# 1. 로컬 .env 파일 우선 (backend/app/db/scripts/.env)
# 2. 프로젝트 루트 .env 파일 (override=False로 이미 로드된 값은 유지)
# 참고: 개인 개발 환경(backend_dev)에서는 backend_dev/app/db/scripts/.env가 우선 사용됨
load_dotenv(Path(__file__).parent / '.env', override=False)
load_dotenv(Path(__file__).parent.parent.parent.parent.parent / '.env', override=False)

# 환경 변수
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "callact_admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "callact_pwd1")
DB_NAME = os.getenv("DB_NAME", "callact_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


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
        return conn
    except Exception as e:
        print(f"[ERROR] Failed to connect to database: {e}")
        sys.exit(1)


def verify_data_counts(conn: psycopg2_connection) -> Dict:
    """데이터 개수 확인"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    results = {}
    
    try:
        # consultations 개수
        cursor.execute("SELECT COUNT(*) as count FROM consultations")
        results["consultations_count"] = cursor.fetchone()["count"]
        
        # consultation_documents 개수
        cursor.execute("SELECT COUNT(*) as count FROM consultation_documents")
        results["documents_count"] = cursor.fetchone()["count"]
        
        # 임베딩이 있는 문서 개수
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM consultation_documents 
            WHERE embedding IS NOT NULL
        """)
        results["documents_with_embedding"] = cursor.fetchone()["count"]
        
        # 임베딩이 없는 문서 개수
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM consultation_documents 
            WHERE embedding IS NULL
        """)
        results["documents_without_embedding"] = cursor.fetchone()["count"]
        
        # 카테고리별 분포
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM consultations
            GROUP BY category
            ORDER BY count DESC
            LIMIT 10
        """)
        results["top_categories"] = cursor.fetchall()
        
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
    finally:
        cursor.close()
    
    return results


def verify_foreign_keys(conn: psycopg2_connection) -> Dict:
    """Foreign Key 관계 확인"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    results = {}
    
    try:
        # consultation_documents의 consultation_id가 consultations에 없는 경우
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM consultation_documents cd
            LEFT JOIN consultations c ON cd.consultation_id = c.id
            WHERE c.id IS NULL
        """)
        results["orphaned_documents"] = cursor.fetchone()["count"]
        
        # consultations에 연결된 documents가 없는 경우
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM consultations c
            LEFT JOIN consultation_documents cd ON c.id = cd.consultation_id
            WHERE cd.id IS NULL
        """)
        results["consultations_without_documents"] = cursor.fetchone()["count"]
        
        # 정상적인 관계 수
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM consultation_documents cd
            INNER JOIN consultations c ON cd.consultation_id = c.id
        """)
        results["valid_relationships"] = cursor.fetchone()["count"]
        
    except Exception as e:
        print(f"[ERROR] Foreign key verification failed: {e}")
    finally:
        cursor.close()
    
    return results


def verify_embeddings(conn: psycopg2_connection) -> Dict:
    """임베딩 벡터 검증"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    results = {}
    
    try:
        # 임베딩 차원 확인 (1536차원)
        cursor.execute("""
            SELECT 
                id,
                array_length(string_to_array(embedding::text, ','), 1) as dimension
            FROM consultation_documents
            WHERE embedding IS NOT NULL
            LIMIT 10
        """)
        sample_embeddings = cursor.fetchall()
        
        # 차원 검증
        valid_dimensions = [e["dimension"] for e in sample_embeddings if e["dimension"] == 1536]
        results["valid_dimension_count"] = len(valid_dimensions)
        results["total_checked"] = len(sample_embeddings)
        
        # 임베딩이 null인 문서 ID 샘플
        cursor.execute("""
            SELECT id, title
            FROM consultation_documents
            WHERE embedding IS NULL
            LIMIT 5
        """)
        results["null_embedding_samples"] = cursor.fetchall()
        
    except Exception as e:
        print(f"[ERROR] Embedding verification failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
    
    return results


def test_similarity_search(conn: psycopg2_connection, test_query: str = "카드 분실 신고") -> List[Dict]:
    """유사도 검색 테스트"""
    if not OPENAI_API_KEY:
        print("[WARNING] OPENAI_API_KEY not found, skipping similarity search test")
        return []
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # 테스트 쿼리 임베딩 생성
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=test_query
        )
        query_embedding = response.data[0].embedding
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        # 유사도 검색 (Cosine Similarity)
        cursor.execute("""
            SELECT 
                id,
                title,
                category,
                1 - (embedding <=> %s::vector) as similarity
            FROM consultation_documents
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        """, (embedding_str, embedding_str))
        
        results = cursor.fetchall()
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Similarity search test failed: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        cursor.close()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Hana Card data load")
    parser.add_argument("--test-search", type=str, help="Test similarity search with query")
    
    args = parser.parse_args()
    
    # 데이터베이스 연결
    conn = connect_db()
    
    try:
        print("=" * 60)
        print("하나카드 데이터 DB 적재 검증")
        print("=" * 60)
        
        # 1. 데이터 개수 확인
        print("\n[1] 데이터 개수 확인")
        print("-" * 60)
        counts = verify_data_counts(conn)
        print(f"  Consultations: {counts.get('consultations_count', 0):,}건")
        print(f"  Consultation Documents: {counts.get('documents_count', 0):,}건")
        print(f"  Documents with Embedding: {counts.get('documents_with_embedding', 0):,}건")
        print(f"  Documents without Embedding: {counts.get('documents_without_embedding', 0):,}건")
        
        if counts.get('top_categories'):
            print(f"\n  상위 10개 카테고리:")
            for cat in counts['top_categories']:
                print(f"    - {cat['category']}: {cat['count']}건")
        
        # 2. Foreign Key 관계 확인
        print("\n[2] Foreign Key 관계 확인")
        print("-" * 60)
        fk_results = verify_foreign_keys(conn)
        print(f"  Valid Relationships: {fk_results.get('valid_relationships', 0):,}건")
        print(f"  Orphaned Documents: {fk_results.get('orphaned_documents', 0):,}건")
        print(f"  Consultations without Documents: {fk_results.get('consultations_without_documents', 0):,}건")
        
        if fk_results.get('orphaned_documents', 0) > 0:
            print(f"\n  [WARNING] {fk_results['orphaned_documents']} documents have no matching consultation")
        
        # 3. 임베딩 벡터 검증
        print("\n[3] 임베딩 벡터 검증")
        print("-" * 60)
        embedding_results = verify_embeddings(conn)
        print(f"  Valid Dimensions (1536): {embedding_results.get('valid_dimension_count', 0)}/{embedding_results.get('total_checked', 0)}")
        
        if embedding_results.get('null_embedding_samples'):
            print(f"\n  임베딩이 없는 문서 샘플:")
            for doc in embedding_results['null_embedding_samples']:
                print(f"    - {doc['id']}: {doc['title']}")
        
        # 4. 유사도 검색 테스트
        if args.test_search or True:  # 기본적으로 테스트 실행
            print("\n[4] 유사도 검색 테스트")
            print("-" * 60)
            test_query = args.test_search or "카드 분실 신고"
            print(f"  Test Query: '{test_query}'")
            
            search_results = test_similarity_search(conn, test_query)
            
            if search_results:
                print(f"\n  Top 5 Similar Documents:")
                for i, result in enumerate(search_results, 1):
                    similarity = result.get('similarity', 0)
                    print(f"    {i}. {result['title']} (카테고리: {result['category']}, 유사도: {similarity:.3f})")
            else:
                print("  [WARNING] No results found or test failed")
        
        # 최종 요약
        print("\n" + "=" * 60)
        print("검증 완료")
        print("=" * 60)
        
        # 경고 사항
        warnings = []
        if counts.get('documents_without_embedding', 0) > 0:
            warnings.append(f"{counts['documents_without_embedding']} documents without embedding")
        if fk_results.get('orphaned_documents', 0) > 0:
            warnings.append(f"{fk_results['orphaned_documents']} orphaned documents")
        
        if warnings:
            print("\n[WARNINGS]")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("\n[SUCCESS] All checks passed!")
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()


