"""
시맨틱 캐시 모듈

임베딩 유사도 기반 캐시 - 유사한 쿼리도 캐시 히트 가능
- 코사인 유사도 기반 매칭
- 임계값 조정 가능
"""

import os
import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

_SEMANTIC_CACHE_ENABLED = os.getenv("RAG_SEMANTIC_CACHE", "1") != "0"
_SEMANTIC_CACHE_THRESHOLD = float(os.getenv("RAG_SEMANTIC_CACHE_THRESHOLD", "0.85"))
_SEMANTIC_CACHE_TTL = float(os.getenv("RAG_SEMANTIC_CACHE_TTL", "300"))
_SEMANTIC_CACHE_MAX_SIZE = int(os.getenv("RAG_SEMANTIC_CACHE_MAX_SIZE", "200"))

# 캐시 저장소: {query_hash: (timestamp, embedding, results)}
_SEMANTIC_CACHE: Dict[str, Tuple[float, List[float], List[Dict[str, Any]]]] = {}
_SEMANTIC_CACHE_LOCK = threading.Lock()


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """코사인 유사도 계산"""
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _prune_cache(now: float) -> None:
    """만료된 캐시 제거"""
    if not _SEMANTIC_CACHE:
        return

    expired = [
        key for key, (ts, _, _) in _SEMANTIC_CACHE.items()
        if now - ts > _SEMANTIC_CACHE_TTL
    ]
    for key in expired:
        _SEMANTIC_CACHE.pop(key, None)

    # 크기 제한
    if len(_SEMANTIC_CACHE) > _SEMANTIC_CACHE_MAX_SIZE:
        sorted_items = sorted(_SEMANTIC_CACHE.items(), key=lambda x: x[1][0])
        for key, _ in sorted_items[:len(_SEMANTIC_CACHE) - _SEMANTIC_CACHE_MAX_SIZE]:
            _SEMANTIC_CACHE.pop(key, None)


def _find_similar_cache(
    query_embedding: List[float],
    threshold: float = None,
) -> Optional[Tuple[str, List[Dict[str, Any]], float]]:
    """유사한 캐시 항목 찾기"""
    threshold = threshold or _SEMANTIC_CACHE_THRESHOLD

    best_match = None
    best_score = 0.0

    for key, (ts, cached_embedding, results) in _SEMANTIC_CACHE.items():
        similarity = _cosine_similarity(query_embedding, cached_embedding)
        if similarity >= threshold and similarity > best_score:
            best_score = similarity
            best_match = (key, results, similarity)

    return best_match


def semantic_cache_get(
    query: str,
    query_embedding: List[float],
    threshold: float = None,
) -> Optional[Tuple[List[Dict[str, Any]], float]]:
    """
    시맨틱 캐시에서 결과 조회

    Returns:
        (results, similarity_score) 또는 None
    """
    if not _SEMANTIC_CACHE_ENABLED or not query_embedding:
        return None

    now = time.time()

    with _SEMANTIC_CACHE_LOCK:
        _prune_cache(now)

        # 정확히 일치하는 쿼리 먼저 확인
        exact_key = query.strip().lower()
        if exact_key in _SEMANTIC_CACHE:
            ts, _, results = _SEMANTIC_CACHE[exact_key]
            if now - ts <= _SEMANTIC_CACHE_TTL:
                return results, 1.0

        # 유사 쿼리 검색
        match = _find_similar_cache(query_embedding, threshold)
        if match:
            _, results, similarity = match
            return results, similarity

    return None


def semantic_cache_set(
    query: str,
    query_embedding: List[float],
    results: List[Dict[str, Any]],
) -> None:
    """시맨틱 캐시에 결과 저장"""
    if not _SEMANTIC_CACHE_ENABLED or not query_embedding or not results:
        return

    now = time.time()
    key = query.strip().lower()

    with _SEMANTIC_CACHE_LOCK:
        _prune_cache(now)
        _SEMANTIC_CACHE[key] = (now, query_embedding, results)


def semantic_cache_clear() -> int:
    """캐시 전체 삭제"""
    with _SEMANTIC_CACHE_LOCK:
        count = len(_SEMANTIC_CACHE)
        _SEMANTIC_CACHE.clear()
        return count


def semantic_cache_stats() -> Dict[str, Any]:
    """캐시 통계"""
    with _SEMANTIC_CACHE_LOCK:
        return {
            "enabled": _SEMANTIC_CACHE_ENABLED,
            "size": len(_SEMANTIC_CACHE),
            "max_size": _SEMANTIC_CACHE_MAX_SIZE,
            "threshold": _SEMANTIC_CACHE_THRESHOLD,
            "ttl": _SEMANTIC_CACHE_TTL,
        }


class SemanticCache:
    """시맨틱 캐시 클래스"""

    def __init__(
        self,
        threshold: float = None,
        ttl: float = None,
        max_size: int = None,
    ):
        self.threshold = threshold or _SEMANTIC_CACHE_THRESHOLD
        self.ttl = ttl or _SEMANTIC_CACHE_TTL
        self.max_size = max_size or _SEMANTIC_CACHE_MAX_SIZE
        self._cache: Dict[str, Tuple[float, List[float], List[Dict[str, Any]]]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _prune(self, now: float) -> None:
        expired = [k for k, (ts, _, _) in self._cache.items() if now - ts > self.ttl]
        for k in expired:
            self._cache.pop(k, None)
        if len(self._cache) > self.max_size:
            sorted_items = sorted(self._cache.items(), key=lambda x: x[1][0])
            for k, _ in sorted_items[:len(self._cache) - self.max_size]:
                self._cache.pop(k, None)

    def get(
        self,
        query: str,
        embedding: List[float],
    ) -> Optional[Tuple[List[Dict[str, Any]], float]]:
        """캐시 조회"""
        if not embedding:
            return None

        now = time.time()
        with self._lock:
            self._prune(now)

            # 정확 매칭
            key = query.strip().lower()
            if key in self._cache:
                ts, _, results = self._cache[key]
                if now - ts <= self.ttl:
                    self._hits += 1
                    return results, 1.0

            # 유사 매칭
            best_score = 0.0
            best_results = None
            for k, (ts, cached_emb, results) in self._cache.items():
                sim = _cosine_similarity(embedding, cached_emb)
                if sim >= self.threshold and sim > best_score:
                    best_score = sim
                    best_results = results

            if best_results:
                self._hits += 1
                return best_results, best_score

            self._misses += 1
            return None

    def set(
        self,
        query: str,
        embedding: List[float],
        results: List[Dict[str, Any]],
    ) -> None:
        """캐시 저장"""
        if not embedding or not results:
            return

        now = time.time()
        key = query.strip().lower()

        with self._lock:
            self._prune(now)
            self._cache[key] = (now, embedding, results)

    def stats(self) -> Dict[str, Any]:
        """통계"""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0,
            }

    def clear(self) -> int:
        """캐시 삭제"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            return count


# 글로벌 인스턴스
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache()
    return _semantic_cache
