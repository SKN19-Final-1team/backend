import sys
import os
import glob
import json
import psycopg2

from app.db.base import get_connection
from pgvector.psycopg2 import register_vector

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 스크립트로 직접 실행 시 backend 경로를 sys.path에 추가하여 app 패키지를 찾을 수 있게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(backend_dir)

# 임베딩할 파일들이 있는 경로
TARGET_DIR = "C:/SKN19/data-preprocessing/data/hyundai/" 
FILE_EXTENSIONS = '*.json'

# DB에서 마지막 ID 가져오기
def get_last_id(conn):
    cur = conn.cursor()
    try:
        # id 필드에서 guide_숫자 형식의 최댓값 찾기
        cur.execute("""
            SELECT metadata->>'id' 
            FROM guide_tbl 
            WHERE metadata->>'id' LIKE 'guide_%'
            ORDER BY CAST(SUBSTRING(metadata->>'id' FROM 'guide_([0-9]+)') AS INTEGER) DESC
            LIMIT 1;
        """)
        result = cur.fetchone()
        
        if result and result[0]:
            last_id = result[0]
            last_number = int(last_id.split('_')[1])
            return last_number + 1
        else:
            return 1
            
    except Exception as e:
        print(f"⚠️ 마지막 ID 조회 실패: {e}\nguide_1부터 시작")
        return 1
    finally:
        cur.close()

# 파일 읽기 및 청킹
def load_files(start_id=1):
    documents = []
    id_counter = start_id
    
    # 지정된 경로에서 파일 찾기
    files = []
    if isinstance(FILE_EXTENSIONS, str):
        files.extend(glob.glob(os.path.join(TARGET_DIR, FILE_EXTENSIONS)))
    else:
        for ext in FILE_EXTENSIONS:
            files.extend(glob.glob(os.path.join(TARGET_DIR, ext)))
    
    print(f"찾은 파일 {len(files)}개")

    # 파일 로드
    for file_path in files:
        # JSON 파일은 별도로 처리
        if file_path.endswith('.json'):
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    
                for item in json_data:
                    # 순차적 ID 생성
                    new_id = f"guide_{id_counter}"
                    id_counter += 1
                    
                    # JSON 객체를 텍스트로 변환
                    # title과 content를 주요 필드로 사용
                    title = item.get('title', 'N/A')
                    content_text = item.get('content', item.get('text', 'N/A'))
                    
                    # 메타데이터 추출
                    metadata_obj = item.get('metadata', {})
                    category1 = metadata_obj.get('category1', item.get('tag', 'N/A'))
                    category2 = metadata_obj.get('category2', 'N/A')
                    
                    # 임베딩용 텍스트 구성
                    page_content = f"제목: {title}\n\n내용: {content_text}"
                    
                    # Document 객체 생성
                    from langchain_core.documents import Document
                    doc = Document(
                        page_content=page_content,
                        metadata={
                            "id": new_id,
                            "title": title,
                            "category1": category1,
                            "category2": category2
                        }
                    )
                    documents.append(doc)
                
                print(f"✅ JSON 파일 로드 완료: {file_path}")

            except Exception as e:
                print(f"⚠️ JSON 파일 로드 실패 ({file_path}): {e}")

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 500자 단위로 자름
        chunk_overlap=50 # 50자는 겹치기 (맥락 유지)
    )
    split_docs = text_splitter.split_documents(documents)
    
    return split_docs

# 임베딩 생성 및 적재
def embed_and_save(chunks, conn):
    # pgvector 어댑터 등록 (이미 연결된 conn 사용)
    register_vector(conn)
    cur = conn.cursor()

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    try:
        # 배치 처리
        for i, chunk in enumerate(chunks):
            content = chunk.page_content
            meta = chunk.metadata
            
            vector = embeddings_model.embed_query(content)
            
            cur.execute(
                "INSERT INTO guide_tbl (content, metadata, embedding) VALUES (%s, %s, %s)",
                (content, json.dumps(meta), vector)
            )
            
            if (i + 1) % 10 == 0:
                print(f"   - {i + 1}개 저장 완료...")
        
        conn.commit()
        print(f"총 {len(chunks)}개 적재 완료")

    except Exception as e:
        print(f"❌ 적재 중 오류 발생: {e}")
        conn.rollback()
    finally:
        cur.close()


def main():
    conn = None
    try:
        conn = get_connection()
        print("✅ DB 연결 성공")
        
        start_id = get_last_id(conn)
        chunks = load_files(start_id=start_id)
        
        if chunks:
            embed_and_save(chunks, conn)
        else:
            print("적재할 데이터가 없습니다.")
            
    except Exception as e:
        print(f"❌ 작업 중 오류 발생: {e}")
    finally:
        if conn:
            conn.close()
            print(">>> conn closed")

if __name__ == "__main__":
    main()
