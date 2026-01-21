from typing import Dict, List, Optional, Tuple

import json
import os
import time

try:
    import redis.asyncio as redis_async
except Exception:
    redis_async = None

RETRIEVE_CACHE_TTL_SEC = float(os.getenv("RAG_RETRIEVE_CACHE_TTL", "60"))
RETRIEVE_CACHE_ENABLED = (
    RETRIEVE_CACHE_TTL_SEC > 0 and os.getenv("RAG_RETRIEVE_CACHE", "1") != "0"
)
REDIS_URL = os.getenv("RAG_REDIS_URL")
REDIS_ENABLED = RETRIEVE_CACHE_ENABLED and bool(REDIS_URL) and redis_async is not None

_REDIS_CLIENT = None
_RETRIEVE_CACHE: Dict[tuple, tuple[float, List[Dict[str, object]]]] = {}


def _redis_client():
    global _REDIS_CLIENT
    if not REDIS_ENABLED:
        return None
    if _REDIS_CLIENT is None:
        _REDIS_CLIENT = redis_async.from_url(REDIS_URL, decode_responses=True)
    return _REDIS_CLIENT


def _prune_cache(now: float) -> None:
    if not _RETRIEVE_CACHE:
        return
    expired = [
        key for key, (ts, _) in _RETRIEVE_CACHE.items() if now - ts > RETRIEVE_CACHE_TTL_SEC
    ]
    for key in expired:
        _RETRIEVE_CACHE.pop(key, None)


def _normalize_filters(filters: Dict[str, object]) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    if not filters:
        return tuple()
    items: List[Tuple[str, Tuple[str, ...]]] = []
    for key in sorted(filters.keys()):
        value = filters.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values = tuple(sorted(str(v) for v in value if v is not None))
        else:
            values = (str(value),)
        items.append((str(key), values))
    return tuple(items)


def build_retrieval_cache_key(
    normalized_query: str,
    route: str,
    db_route: str,
    filters: Dict[str, object],
    top_k: int,
) -> Optional[tuple]:
    if not normalized_query:
        return None
    return (
        route or "",
        db_route or "",
        normalized_query,
        _normalize_filters(filters),
        int(top_k),
    )


def _cache_key_str(key: tuple) -> str:
    return "rag:retrieve:" + json.dumps(key, ensure_ascii=False, separators=(",", ":"))


async def retrieval_cache_get(
    key: Optional[tuple],
) -> Optional[tuple[List[Dict[str, object]], str]]:
    if not RETRIEVE_CACHE_ENABLED or not key:
        return None

    if REDIS_ENABLED:
        client = _redis_client()
        if client:
            try:
                payload = await client.get(_cache_key_str(key))
                if payload:
                    data = json.loads(payload)
                    entries = data.get("entries") or []
                    if entries:
                        return entries, "redis"
            except Exception as exc:
                print("[rag] redis retrieval cache get failed:", repr(exc))

    now = time.time()
    _prune_cache(now)
    entry = _RETRIEVE_CACHE.get(key)
    if not entry:
        return None
    ts, entries = entry
    if now - ts > RETRIEVE_CACHE_TTL_SEC:
        _RETRIEVE_CACHE.pop(key, None)
        return None
    return entries, "mem"


async def retrieval_cache_set(
    key: Optional[tuple],
    entries: List[Dict[str, object]],
) -> None:
    if not RETRIEVE_CACHE_ENABLED or not key or not entries:
        return

    if REDIS_ENABLED:
        client = _redis_client()
        if client:
            try:
                payload = json.dumps({"entries": entries}, ensure_ascii=False, separators=(",", ":"))
                ttl = max(1, int(RETRIEVE_CACHE_TTL_SEC))
                await client.setex(_cache_key_str(key), ttl, payload)
            except Exception as exc:
                print("[rag] redis retrieval cache set failed:", repr(exc))

    now = time.time()
    _prune_cache(now)
    _RETRIEVE_CACHE[key] = (now, entries)
