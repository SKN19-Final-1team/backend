from typing import Any, Dict, List, Optional, Tuple

import copy
import hashlib
import json
import os
import time

from app.llm.rag_llm.card_generator import CARD_PROMPT_VERSION

try:
    import redis.asyncio as redis_async
except Exception:
    redis_async = None

CARD_CACHE_TTL_SEC = float(os.getenv("RAG_CARD_CACHE_TTL", "120"))
CARD_CACHE_ENABLED = CARD_CACHE_TTL_SEC > 0 and os.getenv("RAG_CARD_CACHE", "1") != "0"
LOG_CACHE_KEYS = os.getenv("RAG_CACHE_LOG_KEYS", "0") == "1"
REDIS_URL = os.getenv("RAG_REDIS_URL")
REDIS_ENABLED = CARD_CACHE_ENABLED and bool(REDIS_URL) and redis_async is not None

_REDIS_CLIENT = None
_CARD_CACHE: Dict[tuple, tuple[float, Dict[str, Dict[str, Any]], str]] = {}


def _redis_client():
    global _REDIS_CLIENT
    if not REDIS_ENABLED:
        return None
    if _REDIS_CLIENT is None:
        _REDIS_CLIENT = redis_async.from_url(REDIS_URL, decode_responses=True)
    return _REDIS_CLIENT


def _prune_card_cache(now: float) -> None:
    if not _CARD_CACHE:
        return
    expired = [key for key, (ts, _, _) in _CARD_CACHE.items() if now - ts > CARD_CACHE_TTL_SEC]
    for key in expired:
        _CARD_CACHE.pop(key, None)


def doc_cache_id(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata") or {}
    return str(meta.get("id") or doc.get("id") or "")


def _cards_by_id(cards: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for card in cards:
        card_id = str(card.get("id") or "")
        if not card_id or card_id in mapping:
            continue
        mapping[card_id] = card
    return mapping


def _cards_from_cache(
    cards_by_id: Dict[str, Dict[str, Any]],
    ordered_doc_ids: List[str],
) -> Optional[List[Dict[str, Any]]]:
    if not cards_by_id or not ordered_doc_ids:
        return None
    out: List[Dict[str, Any]] = []
    for doc_id in ordered_doc_ids:
        card = cards_by_id.get(doc_id)
        if not card:
            return None
        out.append(copy.deepcopy(card))
    return out


def build_card_cache_key(
    route: str,
    model: str,
    llm_card_top_n: int,
    normalized_query_template: str,
    normalized_query: str,
    doc_ids: List[str],
) -> Optional[tuple]:
    if not doc_ids or any(not doc_id for doc_id in doc_ids):
        return None
    sorted_ids = tuple(sorted(doc_ids))
    return (
        model,
        llm_card_top_n,
        route or "",
        CARD_PROMPT_VERSION,
        normalized_query_template,
        normalized_query,
        sorted_ids,
    )


def _cache_key_str(key: tuple) -> str:
    return "rag:cards:" + json.dumps(key, ensure_ascii=False, separators=(",", ":"))


def _short_key(key: tuple) -> str:
    raw = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _log_cache_key(action: str, key: tuple, hit: Optional[str], doc_count: int) -> None:
    if not LOG_CACHE_KEYS:
        return
    raw = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    model, llm_card_top_n, route, prompt_version, query_template, query, doc_ids = key
    template_preview = (query_template or "")[:60]
    query_preview = (query or "")[:60]
    print(
        "[rag][cache] cards "
        f"{action} hit={hit or 'miss'} hash={digest} "
        f"route={route} model={model} top_n={llm_card_top_n} "
        f"prompt_ver={prompt_version} docs={doc_count} "
        f"template={template_preview} query={query_preview}"
    )


async def card_cache_get(
    key: Optional[tuple],
    ordered_doc_ids: List[str],
) -> Optional[tuple[List[Dict[str, Any]], str, str]]:
    if not CARD_CACHE_ENABLED:
        return None
    if not key:
        return None

    if REDIS_ENABLED:
        client = _redis_client()
        if client:
            try:
                payload = await client.get(_cache_key_str(key))
                if payload:
                    data = json.loads(payload)
                    cards_by_id = data.get("cards_by_id") or {}
                    guidance_script = data.get("guidance_script") or ""
                    cards = _cards_from_cache(cards_by_id, ordered_doc_ids)
                    if cards is not None:
                        _log_cache_key("get", key, "redis", len(ordered_doc_ids))
                        print(f"[card_cache] hit=1 layer=redis key={_short_key(key)}")
                        return cards, guidance_script, "redis"
            except Exception as exc:
                print("[rag] redis cache get failed:", repr(exc))

    now = time.time()
    _prune_card_cache(now)
    entry = _CARD_CACHE.get(key)
    if not entry:
        _log_cache_key("get", key, None, len(ordered_doc_ids))
        return None
    ts, cards_by_id, guidance_script = entry
    if now - ts > CARD_CACHE_TTL_SEC:
        _CARD_CACHE.pop(key, None)
        _log_cache_key("get", key, None, len(ordered_doc_ids))
        return None
    cards = _cards_from_cache(cards_by_id, ordered_doc_ids)
    if cards is None:
        _log_cache_key("get", key, None, len(ordered_doc_ids))
        return None
    _log_cache_key("get", key, "mem", len(ordered_doc_ids))
    print(f"[card_cache] hit=1 layer=mem key={_short_key(key)}")
    return cards, guidance_script, "mem"


async def card_cache_set(
    key: Optional[tuple],
    cards: List[Dict[str, Any]],
    guidance_script: str,
) -> None:
    if not CARD_CACHE_ENABLED:
        return
    if not key:
        return

    cards_by_id = _cards_by_id(cards)
    if not cards_by_id:
        return

    if REDIS_ENABLED:
        client = _redis_client()
        if client:
            try:
                payload = json.dumps(
                    {"cards_by_id": cards_by_id, "guidance_script": guidance_script},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                ttl = max(1, int(CARD_CACHE_TTL_SEC))
                await client.setex(_cache_key_str(key), ttl, payload)
                print(f"[card_cache] set layer=redis key={_short_key(key)} ttl={ttl}")
            except Exception as exc:
                print("[rag] redis cache set failed:", repr(exc))

    now = time.time()
    _prune_card_cache(now)
    _CARD_CACHE[key] = (now, copy.deepcopy(cards_by_id), guidance_script)
    _log_cache_key("set", key, "mem", len(cards))
    print(f"[card_cache] set layer=mem key={_short_key(key)} ttl={int(CARD_CACHE_TTL_SEC)}")
