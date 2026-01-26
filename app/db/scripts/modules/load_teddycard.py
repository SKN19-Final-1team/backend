"""
테디카드 데이터 적재 모듈

service_guide_documents, card_products, notices 테이블에 데이터 적재
"""

import re
import json
import random as _rng
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, Tuple

from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, Json as PsycopgJson

from . import (
    BATCH_SIZE,
    DATA_DIR_PROD, DATA_DIR_DEV,
    check_table_has_data, find_data_file
)


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
    # created_by: 빈 값이면 "관리자" 설정
    created_by = doc.get("created_by", "") or "관리자"
    keywords = doc.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    embedding = doc.get("embedding")
    embedding_str = "[" + ",".join(map(str, embedding)) + "]" if embedding else None
    metadata = doc.get("metadata", {})
    # created_at: start_date - random(1~5일) (작성→검토→게시 프로세스)
    created_at = None
    if start_date and isinstance(start_date, date):
        days_before = _rng.randint(1, 5)
        created_at = datetime.combine(start_date - timedelta(days=days_before),
                                       datetime.min.time().replace(hour=_rng.randint(9, 17),
                                                                    minute=_rng.randint(0, 59)))
    return (
        doc_id, title, content, category, priority, is_pinned,
        start_date, end_date, status, created_by, keywords,
        embedding_str, PsycopgJson(metadata) if metadata else None,
        created_at
    )


def load_teddycard_data(conn: psycopg2_connection):
    """테디카드 데이터 적재"""
    print("\n" + "=" * 60)
    print("[8/12] 테디카드 데이터 적재")
    print("=" * 60)

    _rng.seed(42)

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

    # 이미 데이터가 있는지 확인 (service_guide_documents로 확인)
    has_data, count = check_table_has_data(conn, "service_guide_documents")
    if has_data:
        print(f"[INFO] service_guide_documents 테이블에 이미 데이터가 있습니다. (건수: {count}건)")
        has_card_data, card_count = check_table_has_data(conn, "card_products")
        has_notice_data, notice_count = check_table_has_data(conn, "notices")
        print(f"[INFO] card_products: {card_count}건, notices: {notice_count}건")
        print(f"[INFO] 테디카드 데이터가 이미 적재되어 있습니다. - 적재 스킵")
        return True

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
            start_date, end_date, status, created_by, keywords, embedding, metadata,
            created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            COALESCE(%s, NOW())
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
            created_by = EXCLUDED.created_by,
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
