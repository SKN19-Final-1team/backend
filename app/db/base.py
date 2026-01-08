import psycopg2
import json
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# JSON 파일 로드 (데이터 추가 시 주석 해제)
# with open('data/samsung/notice.json', 'r', encoding='utf-8') as f:
#     data_list = json.load(f)

# DB 연결 설정
db_config = {
    'host': os.getenv('DB_HOST'),
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT'))
}

def main():
    try:
        # DB 연결
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        print("✅ DB 연결 성공")

        # 테이블 생성 (초기 설정 이후 필요없음)
        # JSON 키: id(PK), tag, title, content, date
        # create_table_query = """
        # CREATE TABLE IF NOT EXISTS notices (
        #     id VARCHAR(50) PRIMARY KEY,
        #     tag VARCHAR(50),
        #     title VARCHAR(200) NOT NULL,
        #     content TEXT NOT NULL,
        #     created_at DATE,
        #     modified_at DATE
        # );
        # """
        # cur.execute(create_table_query)
        # conn.commit()
        # print("✅ 테이블 생성(또는 확인) 완료")

        # 데이터 전처리 및 적재 (INSERT)
        insert_query = """
        INSERT INTO notices (id, tag, title, content, created_at, modified_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING; 
        """
        # ON CONFLICT: 이미 같은 ID가 있으면 무시

        # JSON 파일의 모든 데이터를 순회하며 적재
        success_count = 0
        skip_count = 0
        
        for data in data_list:
            # Date 타입('YYYY-MM-DD')에 맞게 변환
            created_at = data['date'].replace('.', '-')
            
            # modified_at 필드가 존재하지 않으면 None(NULL)으로 설정
            modified_at = None
            if 'modified_at' in data and data['modified_at']:
                modified_at = data['modified_at'].replace('.', '-')

            cur.execute(insert_query, (
                data['id'],
                data['tag'],
                data['title'],
                data['content'],
                created_at,
                modified_at
            ))
            
            if cur.rowcount > 0:
                success_count += 1
                print(f"✅ 데이터 적재 완료 (ID: {data['id']})")
            else:
                skip_count += 1
                print(f"⚠️ 데이터 중복 (ID: {data['id']})")
        
        conn.commit()
        print(f"\n적재 완료: 성공 {success_count}건, 스킵 {skip_count}건, 총 {len(data_list)}건")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    
    finally:
        # 연결 종료
        if cur:
            cur.close()
        if conn:
            conn.close()
        print("\n✅ DB 연결 종료")

if __name__ == "__main__":
    main()