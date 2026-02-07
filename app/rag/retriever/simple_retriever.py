"""
단순화된 검색 로직 - 복잡도 제거, 핵심만 유지

원칙:
1. Vector search 우선 (의미 기반)
2. 최소한의 필터링
3. 명확한 로직
"""
from typing import Dict, List, Optional
import re

from app.rag.retriever.db import vector_search, text_search
from app.rag.retriever.terms import _build_search_context


def simple_retrieve(
    query: str,
    routing: Dict[str, object],
    tables: List[str],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    """
    단순 검색: vector 우선, 복잡한 로직 제거

    Args:
        query: 검색 쿼리
        routing: 라우팅 정보
        tables: 검색 대상 테이블 리스트
        top_k: 반환할 문서 수

    Returns:
        검색된 문서 리스트
    """
    context = _build_search_context(query, routing)
    filters = routing.get("filters") or {}

    all_results = []

    for table in tables:
        # 테이블별 필터 조정
        table_filters = dict(filters)

        # card_products: intent 필터 제거 (테이블에 intent 필드 없음)
        if table == "card_products":
            table_filters.pop("intent", None)
            table_filters.pop("weak_intent", None)

        # 1. Vector search 시도 (의미 기반, 가장 유연함)
        rows = vector_search(
            query=context.query_text,
            table=table,
            limit=top_k * 3,  # 충분히 가져오기
            filters=table_filters
        )

        # 2. Vector search 실패 시 text search
        if not rows:
            # 카드명에서 타입 제거하여 검색어 확장
            search_terms = list(context.query_terms)
            if context.card_values and table == "service_guide_documents":
                _CARD_TYPE = re.compile(r'(카드|신용카드|체크카드|직불카드|선불카드)$')
                for card_name in context.card_values:
                    core = _CARD_TYPE.sub('', card_name).strip()
                    if core and len(core) >= 2:
                        search_terms.insert(0, core)

            rows = text_search(
                table=table,
                terms=search_terms,
                limit=top_k * 3,
                filters=table_filters
            )

        # 3. 결과 변환
        for row in rows:
            doc_id, content, metadata, structured, score = row
            meta = metadata if isinstance(metadata, dict) else {}

            # card_products: 점수 보정 (text_search는 0.0 반환, 벡터 검색보다 높게)
            original_score = score
            if table == "card_products" and (not score or score == 0.0):
                score = 0.9
                print(f"[DEBUG simple_retriever] Card product score boosted: {doc_id}, {original_score} → {score}")

            result = {
                "id": doc_id,
                "content": content or "",
                "metadata": meta,
                "structured": structured,
                "score": float(score) if score else 0.0,
                "table": table,
                "title": meta.get("title") or meta.get("name") or meta.get("card_name"),
            }
            all_results.append(result)
            print(f"[DEBUG simple_retriever] Added: table={table}, id={doc_id}, score={result['score']}, title={result['title']}")

    # 4. 점수 기준 정렬 및 중복 제거
    print(f"[DEBUG simple_retriever] Total results before sorting: {len(all_results)}")
    seen_ids = set()
    unique_results = []
    for doc in sorted(all_results, key=lambda x: x["score"], reverse=True):
        if doc["id"] not in seen_ids:
            seen_ids.add(doc["id"])
            unique_results.append(doc)

    print(f"[DEBUG simple_retriever] Final top {top_k} results:")
    for i, doc in enumerate(unique_results[:top_k], 1):
        print(f"  [{i}] {doc['id']} (table={doc['table']}, score={doc['score']:.4f}, title={doc.get('title')})")

    return unique_results[:top_k]
