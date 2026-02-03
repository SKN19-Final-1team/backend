"""
키워드 동의어/변형어 적용 모듈

keyword_dictionary 테이블의 synonyms, variations 필드에
수동으로 정의한 데이터를 채웁니다.

멱등성: 동일한 데이터로 반복 실행해도 결과 동일
"""

from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch

from .keyword_synonyms_data import KEYWORD_EXTENSIONS, TOTAL_KEYWORDS


def populate_keyword_extensions(conn: psycopg2_connection, batch_size: int = 100):
    """
    keyword_dictionary 테이블의 synonyms, variations 필드 채우기
    """
    print("\n" + "=" * 60)
    print("[5/5] 키워드 동의어/변형어 데이터 적용")
    print("=" * 60)

    cursor = conn.cursor()

    try:
        print(f"[INFO] 정의된 키워드 확장 데이터: {TOTAL_KEYWORDS}개")

        update_query = """
            UPDATE keyword_dictionary SET
                synonyms = %s,
                variations = %s,
                updated_at = NOW()
            WHERE keyword = %s
        """

        update_batch = []
        matched_count = 0
        updated_keywords = []

        for keyword, extensions in KEYWORD_EXTENSIONS.items():
            synonyms = extensions.get("synonyms", [])
            variations = extensions.get("variations", [])

            # 해당 키워드가 DB에 존재하는지 확인
            cursor.execute(
                "SELECT COUNT(*) FROM keyword_dictionary WHERE keyword = %s",
                (keyword,)
            )
            count = cursor.fetchone()[0]

            if count > 0:
                update_batch.append((synonyms, variations, keyword))
                matched_count += 1
                updated_keywords.append(keyword)

        # 배치 업데이트
        if update_batch:
            execute_batch(cursor, update_query, update_batch, page_size=batch_size)
            conn.commit()

        print(f"[INFO] DB에서 매칭된 키워드: {matched_count}개")
        print(f"[INFO] 업데이트 완료: {len(update_batch)}개")

        # 업데이트된 키워드 샘플 출력
        print("\n[INFO] 업데이트된 키워드 샘플:")
        for kw in updated_keywords[:10]:
            ext = KEYWORD_EXTENSIONS[kw]
            syn = ext.get("synonyms", [])[:3]
            var = ext.get("variations", [])[:3]
            print(f"  - {kw}: synonyms={syn}, variations={var}")

        if len(updated_keywords) > 10:
            print(f"  ... 외 {len(updated_keywords) - 10}개")

        # 통계 확인
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN synonyms IS NOT NULL AND array_length(synonyms, 1) > 0 THEN 1 END) as has_synonyms,
                COUNT(CASE WHEN variations IS NOT NULL AND array_length(variations, 1) > 0 THEN 1 END) as has_variations
            FROM keyword_dictionary
        """)
        stats = cursor.fetchone()
        print(f"\n[INFO] 최종 통계: total={stats[0]}, has_synonyms={stats[1]}, has_variations={stats[2]}")

        cursor.close()
        return True

    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] 키워드 확장 데이터 적용 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    from connect_db import connect_db

    conn = connect_db()
    if conn:
        populate_keyword_extensions(conn)
        conn.close()
