"""
교육 시뮬레이터 기능 테스트 스크립트
- DB 연결 및 조회 테스트
- 각 모듈별 기능 테스트
"""
import os
import sys
import json

# 프로젝트 경로 추가
sys.path.insert(0, r"c:\Projects\backend")
os.chdir(r"c:\Projects\backend")

from dotenv import load_dotenv
load_dotenv()

results = []

def log(msg):
    print(msg)
    results.append(msg)

log("=" * 60)
log("교육 시뮬레이터 기능 테스트")
log("=" * 60)

# ============================================================
# 1. 환경 변수 및 설정 확인
# ============================================================
log("\n[1] 환경 변수 확인")
log("-" * 40)

env_vars = {
    "DB_HOST": os.getenv("DB_HOST"),
    "DB_PORT": os.getenv("DB_PORT"),
    "DB_NAME": os.getenv("DB_NAME"),
    "DB_USER": os.getenv("DB_USER"),
    "RUNPOD_IP": os.getenv("RUNPOD_IP"),
    "RUNPOD_PORT": os.getenv("RUNPOD_PORT"),
}

for key, value in env_vars.items():
    status = "OK" if value else "MISSING"
    masked = value[:10] + "..." if value and len(value) > 10 else value
    log(f"  {key}: {masked} [{status}]")

# ============================================================
# 2. DB 연결 테스트
# ============================================================
log("\n[2] DB 연결 테스트")
log("-" * 40)

db_stats = {}
try:
    from app.db.base import get_connection
    import psycopg2.extras
    conn = get_connection()
    log("  DB 연결: SUCCESS")

    with conn.cursor() as cur:
        # 테이블 존재 확인
        tables = ["consultations", "consultation_documents", "customers", "persona_types"]
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            db_stats[table] = count
            log(f"  {table}: {count} rows")

    conn.close()
except Exception as e:
    log(f"  DB 연결: FAILED - {e}")

# ============================================================
# 3. client.py 모델명 확인
# ============================================================
log("\n[3] LLM Client 설정 확인")
log("-" * 40)

model_check = {"status": "FAIL", "model": ""}
try:
    from app.llm.education.client import RUNPOD_MODEL_NAME, RUNPOD_API_URL
    log(f"  모델명: {RUNPOD_MODEL_NAME}")
    log(f"  API URL: {RUNPOD_API_URL}")

    expected_model = "WindyAle/kanana-nano-2.1B-customer-emotional"
    model_check["model"] = RUNPOD_MODEL_NAME
    if RUNPOD_MODEL_NAME == expected_model:
        log("  모델명 검증: PASS (emotional 버전)")
        model_check["status"] = "PASS"
    else:
        log(f"  모델명 검증: FAIL (expected: {expected_model})")
except Exception as e:
    log(f"  LLM Client 로드 실패: {e}")

# ============================================================
# 4. DB 조회 로직 테스트 (education.py 시뮬레이션)
# ============================================================
log("\n[4] DB 조회 로직 테스트 (education.py)")
log("-" * 40)

db_query_results = {
    "consultations": "NOT_TESTED",
    "consultation_documents": "NOT_TESTED",
    "customers": "NOT_TESTED",
    "persona_types": "NOT_TESTED"
}

try:
    conn = get_connection()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # 사용 가능한 카테고리 먼저 확인
        cur.execute("SELECT DISTINCT category_main, COUNT(*) as cnt FROM consultations GROUP BY category_main ORDER BY cnt DESC")
        categories = cur.fetchall()
        log(f"  사용 가능한 카테고리:")
        for c in categories:
            log(f"    - {c['category_main']}: {c['cnt']}건")

        # 첫 번째 카테고리로 테스트
        if categories:
            test_category = categories[0]['category_main']
            log(f"\n  테스트 카테고리: '{test_category}'")

            # 4-1. consultations 테이블에서 category_main으로 조회
            cur.execute("""
                SELECT c.id, c.customer_id, c.category_main, c.category_sub
                FROM consultations c
                WHERE c.category_main = %s
                ORDER BY RANDOM()
                LIMIT 1
            """, (test_category,))

            consultation_row = cur.fetchone()

            if consultation_row:
                log(f"  [4-1] consultations 조회: PASS")
                log(f"        consultation_id: {consultation_row['id']}")
                log(f"        customer_id: {consultation_row['customer_id']}")
                db_query_results["consultations"] = "PASS"

                consultation_id = consultation_row["id"]
                customer_id = consultation_row["customer_id"]

                # 4-2. consultation_documents 조회
                cur.execute("""
                    SELECT title, keywords, LEFT(content, 100) as content_preview
                    FROM consultation_documents
                    WHERE consultation_id = %s
                    LIMIT 1
                """, (consultation_id,))

                doc_row = cur.fetchone()
                if doc_row:
                    log(f"  [4-2] consultation_documents 조회: PASS")
                    log(f"        title: {doc_row['title']}")
                    db_query_results["consultation_documents"] = "PASS"
                else:
                    log(f"  [4-2] consultation_documents 조회: NO DATA")
                    db_query_results["consultation_documents"] = "NO_DATA"

                # 4-3. customers 조회
                cur.execute("""
                    SELECT name, gender, age_group, grade, card_type,
                           personality_tags, customer_type_codes
                    FROM customers
                    WHERE id = %s
                """, (customer_id,))

                customer_row = cur.fetchone()
                if customer_row:
                    log(f"  [4-3] customers 조회: PASS")
                    log(f"        name: {customer_row['name']}")
                    log(f"        customer_type_codes: {customer_row.get('customer_type_codes')}")
                    db_query_results["customers"] = "PASS"

                    # 4-4. persona_types 조회
                    type_codes = customer_row.get("customer_type_codes")
                    if type_codes and len(type_codes) > 0:
                        cur.execute("""
                            SELECT code, name, description, communication_style
                            FROM persona_types
                            WHERE code = %s
                        """, (type_codes[0],))

                        persona_row = cur.fetchone()
                        if persona_row:
                            log(f"  [4-4] persona_types 조회: PASS")
                            log(f"        code: {persona_row['code']}")
                            log(f"        name: {persona_row['name']}")
                            db_query_results["persona_types"] = "PASS"
                        else:
                            log(f"  [4-4] persona_types 조회: NO DATA for code {type_codes[0]}")
                            db_query_results["persona_types"] = "NO_DATA"
                    else:
                        log(f"  [4-4] persona_types 조회: SKIP (no customer_type_codes)")
                        db_query_results["persona_types"] = "SKIP"
                else:
                    log(f"  [4-3] customers 조회: NO DATA for {customer_id}")
                    db_query_results["customers"] = "NO_DATA"
            else:
                log(f"  [4-1] consultations 조회: NO DATA")
                db_query_results["consultations"] = "NO_DATA"

    conn.close()
except Exception as e:
    log(f"  DB 조회 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 5. feature_analyzer.py 테스트
# ============================================================
log("\n[5] feature_analyzer.py 테스트")
log("-" * 40)

feature_analyzer_results = {"functions": [], "tests": []}
try:
    from app.llm.education.feature_analyzer import analyze_consultation, summarize_consultation_flow

    # 5-1. analyze_consultation 함수 존재 확인
    log(f"  analyze_consultation 함수: EXISTS")
    feature_analyzer_results["functions"].append("analyze_consultation")

    # 5-2. summarize_consultation_flow 함수 존재 확인
    log(f"  summarize_consultation_flow 함수: EXISTS")
    feature_analyzer_results["functions"].append("summarize_consultation_flow")

    # 5-3. 빈 입력에 대한 기본값 반환 테스트
    result = summarize_consultation_flow("")
    if "overall_flow" in result and "consultation_scenario" in result:
        log(f"  summarize_consultation_flow 기본값 반환: PASS")
        feature_analyzer_results["tests"].append(("default_value", "PASS"))
    else:
        log(f"  summarize_consultation_flow 기본값 반환: FAIL")
        feature_analyzer_results["tests"].append(("default_value", "FAIL"))

except Exception as e:
    log(f"  feature_analyzer 로드 실패: {e}")

# ============================================================
# 6. similarity_calculator.py 테스트
# ============================================================
log("\n[6] similarity_calculator.py 테스트")
log("-" * 40)

similarity_results = {"functions": [], "jaccard_test": None}
try:
    from app.llm.education.similarity_calculator import (
        calculate_consultation_similarity,
        _get_original_consultation_content,
        _keyword_similarity_fallback
    )

    log(f"  calculate_consultation_similarity 함수: EXISTS")
    similarity_results["functions"].append("calculate_consultation_similarity")
    log(f"  _get_original_consultation_content 함수: EXISTS")
    similarity_results["functions"].append("_get_original_consultation_content")
    log(f"  _keyword_similarity_fallback 함수: EXISTS")
    similarity_results["functions"].append("_keyword_similarity_fallback")

    # 키워드 기반 유사도 테스트
    test_sim = "카드 분실 신고합니다. 정지 처리 부탁드립니다."
    test_orig = "카드를 분실했습니다. 카드 정지 요청합니다. 재발급도 해주세요."

    result = _keyword_similarity_fallback(test_sim, test_orig)
    similarity_results["jaccard_test"] = result
    log(f"  Jaccard 유사도 테스트: {result['similarity_score']}점")
    log(f"    일치 키워드: {result['matching_elements'][:5]}")

except Exception as e:
    log(f"  similarity_calculator 로드 실패: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 7. followup.py import 테스트
# ============================================================
log("\n[7] followup.py import 테스트")
log("-" * 40)

followup_results = {"import": "FAIL", "call": "FAIL"}
try:
    # followup.py에서 similarity_calculator import 확인
    with open(r"c:\Projects\backend\app\api\v1\endpoints\followup.py", "r", encoding="utf-8") as f:
        content = f.read()

    if "from app.llm.education.similarity_calculator import calculate_consultation_similarity" in content:
        log(f"  similarity_calculator import: PASS")
        followup_results["import"] = "PASS"
    else:
        log(f"  similarity_calculator import: FAIL (import 문 없음)")

    if "await calculate_consultation_similarity" in content:
        log(f"  calculate_consultation_similarity 호출: PASS")
        followup_results["call"] = "PASS"
    else:
        log(f"  calculate_consultation_similarity 호출: FAIL")

except Exception as e:
    log(f"  followup.py 검사 실패: {e}")

# ============================================================
# 결과 저장
# ============================================================
log("\n" + "=" * 60)
log("테스트 완료")
log("=" * 60)

# 결과 파일 저장
output = {
    "db_stats": db_stats,
    "model_check": model_check,
    "db_query_results": db_query_results,
    "feature_analyzer_results": feature_analyzer_results,
    "similarity_results": similarity_results,
    "followup_results": followup_results,
    "log": results
}

with open(r"c:\Projects\backend\test_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

log("\n결과가 test_results.json에 저장되었습니다.")
