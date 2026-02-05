"""
향상된 RAG 검색 파이프라인

기존 검색에 다음 개선사항 적용:
1. 시맨틱 캐시 (유사 쿼리 캐시 히트)
2. Cross-Encoder 리랭킹 (정밀 재순위화)
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from app.rag.pipeline.search import run_search as base_search, SearchResult
from app.rag.retriever.db import embed_query

# 개선 모듈 설정
_SEMANTIC_CACHE_ENABLED = os.getenv("RAG_SEMANTIC_CACHE", "1") != "0"
_RERANK_ENABLED = os.getenv("RAG_RERANK", "1") != "0"


@dataclass
class EnhancedSearchResult:
    """향상된 검색 결과"""
    docs: List[Dict[str, Any]]
    routing: Dict[str, Any]
    consult_docs: List[Dict[str, Any]]

    # 메타데이터
    original_query: str
    cache_hit: bool
    cache_similarity: float
    reranked: bool

    # 타이밍
    t_total: float
    t_cache_check: float
    t_search: float
    t_rerank: float


async def enhanced_search(
    query: str,
    top_k: int = 5,
    use_semantic_cache: bool = None,
    use_rerank: bool = None,
) -> EnhancedSearchResult:
    """
    향상된 검색 수행

    Args:
        query: 사용자 쿼리
        top_k: 반환할 문서 수
        use_semantic_cache: 시맨틱 캐시 사용 여부
        use_rerank: 리랭킹 사용 여부

    Returns:
        EnhancedSearchResult
    """
    t_start = time.time()

    # 기본값 설정
    use_semantic_cache = use_semantic_cache if use_semantic_cache is not None else _SEMANTIC_CACHE_ENABLED
    use_rerank = use_rerank if use_rerank is not None else _RERANK_ENABLED

    original_query = query
    cache_hit = False
    cache_similarity = 0.0
    reranked = False

    t_cache_check = 0.0
    t_search = 0.0
    t_rerank = 0.0

    # 1. 시맨틱 캐시 확인
    query_embedding = None
    cached_results = None

    if use_semantic_cache:
        t0 = time.time()
        try:
            # 임베딩 생성 (캐시 확인용)
            query_embedding = embed_query(query)

            from app.rag.cache.semantic_cache import get_semantic_cache
            cache = get_semantic_cache()
            cache_result = cache.get(query, query_embedding)

            if cache_result:
                cached_results, cache_similarity = cache_result
                cache_hit = True
        except Exception as e:
            print(f"[enhanced_search] Cache check failed: {e}")
        t_cache_check = time.time() - t0

    # 2. 검색 수행 (캐시 미스 시)
    if cache_hit and cached_results:
        docs = cached_results
        routing = {}
        consult_docs = []
    else:
        t0 = time.time()
        result = await base_search(query, top_k=top_k * 2 if use_rerank else top_k)
        docs = result.docs
        routing = result.routing
        consult_docs = result.consult_docs
        t_search = time.time() - t0

        # 캐시에 저장
        if use_semantic_cache and query_embedding and docs:
            try:
                from app.rag.cache.semantic_cache import get_semantic_cache
                cache = get_semantic_cache()
                cache.set(query, query_embedding, docs)
            except Exception as e:
                print(f"[enhanced_search] Cache set failed: {e}")

    # 3. 리랭킹
    if use_rerank and docs and not cache_hit:
        t0 = time.time()
        try:
            from app.rag.rerank.cross_encoder import rerank
            docs = rerank(query, docs, top_k=top_k)
            reranked = True
        except Exception as e:
            print(f"[enhanced_search] Rerank failed: {e}")
            docs = docs[:top_k]
        t_rerank = time.time() - t0
    else:
        docs = docs[:top_k]

    t_total = time.time() - t_start

    return EnhancedSearchResult(
        docs=docs,
        routing=routing,
        consult_docs=consult_docs,
        original_query=original_query,
        cache_hit=cache_hit,
        cache_similarity=cache_similarity,
        reranked=reranked,
        t_total=t_total,
        t_cache_check=t_cache_check,
        t_search=t_search,
        t_rerank=t_rerank,
    )


# 동기 래퍼
def enhanced_search_sync(
    query: str,
    top_k: int = 5,
    **kwargs,
) -> EnhancedSearchResult:
    """동기 버전"""
    import asyncio
    return asyncio.run(enhanced_search(query, top_k, **kwargs))
