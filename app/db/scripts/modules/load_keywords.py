"""
키워드 사전 데이터 적재 모듈

키워드 사전 파일에서 keyword_dictionary, keyword_synonyms 테이블에 적재
카드 상품명 키워드 등록 포함
"""

import json
from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, Json as PsycopgJson
from tqdm import tqdm

from . import (
    BATCH_SIZE,
    DATA_DIR_PROD, DATA_DIR_DEV,
    KEYWORDS_DICT_DIR_PROD, KEYWORDS_DICT_DIR_DEV,
    check_table_has_data
)


def load_keyword_dictionary(conn: psycopg2_connection):
    """키워드 사전 데이터 적재"""
    print("\n" + "=" * 60)
    print("[7/12] 키워드 사전 데이터 적재")
    print("=" * 60)

    # 이미 데이터가 있는지 확인
    has_data, count = check_table_has_data(conn, "keyword_dictionary")
    if has_data:
        print(f"[INFO] keyword_dictionary 테이블에 이미 데이터가 있습니다. (건수: {count}건) - 적재 스킵")
        return True

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

        # 카드 상품명 키워드 등록 (weight: 1.5)
        print("[INFO] 카드 상품명 키워드 등록 중...")
        card_products_file = None
        card_product_filenames = ["teddycard_card_products_with_embeddings.json"]
        for fn in card_product_filenames:
            for search_dir in [DATA_DIR_PROD, DATA_DIR_DEV]:
                fp = search_dir / fn
                if fp.exists():
                    card_products_file = fp
                    break
            if card_products_file:
                break

        if card_products_file:
            with open(card_products_file, 'r', encoding='utf-8') as f:
                card_products_data = json.load(f)

            card_keyword_batch = []
            for card in card_products_data:
                card_name = card.get("name", "").strip()
                if not card_name:
                    continue
                card_keyword_batch.append((
                    card_name,
                    "카드상품",  # category
                    3,  # priority (중간)
                    "low",  # urgency
                    ["카드 상품 문의", "카드 추천"],  # context_hints
                    1.5,  # weight (약간 높은 가중치)
                    [],  # synonyms
                    [],  # variations
                    None,  # compound_patterns
                    None,  # ambiguity_rules
                ))

            if card_keyword_batch:
                execute_batch(cursor, keyword_dict_insert, card_keyword_batch, page_size=BATCH_SIZE)
                conn.commit()
                print(f"[INFO] 카드 상품명 키워드 등록 완료: {len(card_keyword_batch)}개 (weight=1.5)")
        else:
            print("[WARNING] 카드 상품 데이터 파일을 찾을 수 없어 카드명 키워드 등록을 건너뜁니다.")

        cursor.close()
        return True

    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] 키워드 사전 적재 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
