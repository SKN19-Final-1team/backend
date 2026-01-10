import sys
import os
import json

from app.db.base import get_connection

# 스크립트로 직접 실행 시 backend 경로를 sys.path에 추가하여 app 패키지를 찾을 수 있게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(backend_dir)

# JSON 파일 로드 (데이터 추가 시 주석 해제 및 경로 수정)
# with open('data/samsung/notice.json', 'r', encoding='utf-8') as f:
#     data_list = json.load(f)
# 또는 외부에서 import 할 수 있도록 구조 변경 고려

def main():
    conn = None
    cur = None
    try:
        # DB 연결
        conn = get_connection()
        cur = conn.cursor()
        print("✅ DB 연결 성공")

        # 데이터 전처리 및 적재 (INSERT)
        # ON CONFLICT: 이미 같은 ID가 있으면 무시
        insert_query = """
        INSERT INTO notices (id, tag, title, content, created_at, modified_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING; 
        """

        # main 함수 내에서 data_list를 정의하거나 인자로 받아야 함.
        # 기존 코드가 data_list를 전역에서 기대했으므로, 여기서는 예시로 빈 리스트 혹은 주석 처리.
        # 실제 사용 시에는 이 부분을 수정해야 합니다.
        if 'data_list' in locals() or 'data_list' in globals():
             # JSON 파일의 모든 데이터를 순회하며 적재
            success_count = 0
            skip_count = 0
            
            # data_list가 정의되어 있다고 가정 (위 주석 해제 시)
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
        else:
             print("ℹ️ 적재할 데이터가 없습니다. (data_list 미정의)")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    
    finally:
        # 연결 종료
        if cur:
            cur.close()
        if conn:
            conn.close()
            print(">>> conn closed")
        print("\n✅ DB 연결 종료")

if __name__ == "__main__":
    main()
