from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Tuple

DOC_TITLE_CACHE_TTL_SEC = float(os.getenv("RAG_DOC_TITLE_CACHE_TTL", "600"))
DOC_TITLE_CACHE_ENABLED = DOC_TITLE_CACHE_TTL_SEC > 0 and os.getenv("RAG_DOC_TITLE_CACHE", "1") != "0"

_DOC_TITLE_CACHE: Dict[Tuple[str, str], Tuple[float, str]] = {}
_DOC_TITLE_CACHE_LOCK = threading.Lock()

_ALLOWED_TABLES = {"card_products", "service_guide_documents"}


def _doc_id(doc: Dict[str, Any]) -> str:
    for key in ("id", "db_id", "doc_id", "document_id", "source_id"):
        value = doc.get(key)
        if value:
            return str(value)
    return ""


def _doc_title(doc: Dict[str, Any]) -> str:
    title = doc.get("title")
    if title:
        return str(title)
    meta = doc.get("metadata") or {}
    for key in ("title", "name", "card_name"):
        value = meta.get(key)
        if value:
            return str(value)
    return ""


def _doc_table(doc: Dict[str, Any]) -> str:
    table = doc.get("table")
    if table:
        return str(table)
    meta = doc.get("metadata") or {}
    source_table = meta.get("source_table")
    if source_table:
        return str(source_table)
    return ""


def _prune(now: float) -> None:
    if not _DOC_TITLE_CACHE:
        return
    expired = [
        key for key, (ts, _) in _DOC_TITLE_CACHE.items()
        if DOC_TITLE_CACHE_TTL_SEC > 0 and (now - ts) > DOC_TITLE_CACHE_TTL_SEC
    ]
    for key in expired:
        _DOC_TITLE_CACHE.pop(key, None)


def record_doc_titles(docs: List[Dict[str, Any]]) -> None:
    if not DOC_TITLE_CACHE_ENABLED or not docs:
        return
    now = time.time()
    with _DOC_TITLE_CACHE_LOCK:
        _prune(now)
        for doc in docs:
            table = _doc_table(doc)
            if table not in _ALLOWED_TABLES:
                continue
            doc_id = _doc_id(doc)
            title = _doc_title(doc)
            if not doc_id or not title:
                continue
            _DOC_TITLE_CACHE[(table, doc_id)] = (now, title)


def get_doc_title_cache() -> List[Dict[str, Any]]:
    now = time.time()
    with _DOC_TITLE_CACHE_LOCK:
        _prune(now)
        return [
            {"table": table, "id": doc_id, "title": title, "cached_at": ts}
            for (table, doc_id), (ts, title) in _DOC_TITLE_CACHE.items()
        ]


def drain_doc_title_cache() -> List[Dict[str, Any]]:
    now = time.time()
    with _DOC_TITLE_CACHE_LOCK:
        _prune(now)
        entries = [
            {"table": table, "id": doc_id, "title": title, "cached_at": ts}
            for (table, doc_id), (ts, title) in _DOC_TITLE_CACHE.items()
        ]
        _DOC_TITLE_CACHE.clear()
        return entries
