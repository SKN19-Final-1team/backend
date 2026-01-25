from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

from app.rag.postprocess.keywords import (
    BENEFIT_FILTER_TOKENS,
    ISSUE_FILTER_TOKENS,
    normalize_text,
    text_has_any,
)
from app.rag.retriever.db import fetch_docs_by_ids

GUIDE_INTENT_TOKENS = ISSUE_FILTER_TOKENS + BENEFIT_FILTER_TOKENS + (
    "등록",
    "인증",
    "조건",
    "방법",
    "다자녀",
    "다둥이",
    "청년",
    "환급",
)

_SESSION_CONTEXT_ENABLED = os.getenv("RAG_SESSION_CONTEXT", "1") != "0"
_SESSION_TTL_SEC = int(os.getenv("RAG_SESSION_TTL_SEC", "180"))
_SESSION_TURN_TTL = int(os.getenv("RAG_SESSION_TURN_TTL", "3"))
_CONSULT_COOLDOWN_SEC = int(os.getenv("RAG_CONSULT_COOLDOWN_SEC", "12"))
_CONSULT_INTENT_THRESHOLD = float(os.getenv("RAG_CONSULT_INTENT_THRESHOLD", "0.6"))
_CONSULT_MIN_KEYWORD_HITS = int(os.getenv("RAG_CONSULT_MIN_KEYWORD_HITS", "2"))
_CONSULT_MIN_TURNS = int(os.getenv("RAG_CONSULT_MIN_TURNS", "2"))
_CONSULT_MIN_SENTENCES = int(os.getenv("RAG_CONSULT_MIN_SENTENCES", "2"))


def text_has_any_compact(text: str, tokens: tuple[str, ...]) -> bool:
    if text_has_any(text, tokens):
        return True
    compact = (text or "").lower().replace(" ", "")
    return any(token in compact for token in tokens)


def should_expand_card_info(
    query: str,
    routing: Dict[str, Any],
    filters: Dict[str, Any],
) -> bool:
    if not filters.get("card_name"):
        return False
    db_route = routing.get("db_route")
    if db_route in ("both", "guide_tbl"):
        return True
    return text_has_any_compact(query, GUIDE_INTENT_TOKENS)


def _as_list(value: object | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [item for item in value if item]


def _unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _session_is_fresh(
    updated_at: Optional[float],
    updated_turn: Optional[int],
    now: float,
    turn: int,
) -> bool:
    if not updated_at or updated_turn is None:
        return False
    if _SESSION_TTL_SEC > 0 and (now - updated_at) > _SESSION_TTL_SEC:
        return False
    if _SESSION_TURN_TTL > 0 and (turn - updated_turn) > _SESSION_TURN_TTL:
        return False
    return True


def _build_query_template(card_names: List[str], intent_terms: List[str]) -> Optional[str]:
    if not card_names and not intent_terms:
        return None
    if card_names and intent_terms:
        return f"{card_names[0]} {intent_terms[0]} 방법"
    if card_names:
        return card_names[0]
    return f"{intent_terms[0]} 방법"


def apply_session_context(
    query: str,
    routing: Dict[str, Any],
    session_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not session_state or not _SESSION_CONTEXT_ENABLED:
        return routing

    now = time.time()
    turn = int(session_state.get("turn") or 0) + 1
    session_state["turn"] = turn

    routing = dict(routing)
    filters = dict(routing.get("filters") or {})
    boost = dict(routing.get("boost") or {})
    matched = dict(routing.get("matched") or {})

    card_names = _unique_in_order(
        _as_list(matched.get("card_names") or filters.get("card_name") or boost.get("card_name"))
    )
    intent_terms = _unique_in_order(
        _as_list(matched.get("actions") or filters.get("intent") or boost.get("intent"))
    )
    weak_terms = _unique_in_order(
        _as_list(matched.get("weak_intents") or filters.get("weak_intent") or boost.get("weak_intent"))
    )

    if card_names:
        session_state["current_card_name"] = card_names
        session_state["card_updated_at"] = now
        session_state["card_updated_turn"] = turn
    else:
        session_card = _as_list(session_state.get("current_card_name"))
        if _session_is_fresh(
            session_state.get("card_updated_at"),
            session_state.get("card_updated_turn"),
            now,
            turn,
        ):
            if session_card:
                card_names = session_card
                filters.setdefault("card_name", card_names)
                boost.setdefault("card_name", card_names)
                matched["card_names"] = card_names
        else:
            session_state.pop("current_card_name", None)
            session_state.pop("card_updated_at", None)
            session_state.pop("card_updated_turn", None)

    intent_kind = None
    if intent_terms:
        intent_kind = "intent"
        session_state["current_intent"] = intent_terms
        session_state["intent_kind"] = intent_kind
        session_state["intent_updated_at"] = now
        session_state["intent_updated_turn"] = turn
    elif weak_terms:
        intent_kind = "weak_intent"
        session_state["current_intent"] = weak_terms
        session_state["intent_kind"] = intent_kind
        session_state["intent_updated_at"] = now
        session_state["intent_updated_turn"] = turn
        intent_terms = weak_terms
    else:
        session_intent = _as_list(session_state.get("current_intent"))
        session_kind = session_state.get("intent_kind") or "intent"
        if _session_is_fresh(
            session_state.get("intent_updated_at"),
            session_state.get("intent_updated_turn"),
            now,
            turn,
        ):
            if session_intent:
                intent_terms = session_intent
                intent_kind = session_kind
                if session_kind == "weak_intent":
                    filters.setdefault("weak_intent", intent_terms)
                    boost.setdefault("weak_intent", intent_terms)
                    matched["weak_intents"] = intent_terms
                else:
                    filters.setdefault("intent", intent_terms)
                    boost.setdefault("intent", intent_terms)
                    matched["actions"] = intent_terms
        else:
            session_state.pop("current_intent", None)
            session_state.pop("intent_kind", None)
            session_state.pop("intent_updated_at", None)
            session_state.pop("intent_updated_turn", None)

    if card_names and intent_terms:
        filters.setdefault("card_name", card_names)
        boost.setdefault("card_name", card_names)
        matched.setdefault("card_names", card_names)

    if intent_terms and intent_kind == "intent":
        filters.setdefault("intent", intent_terms)
        boost.setdefault("intent", intent_terms)
        matched.setdefault("actions", intent_terms)
    elif intent_terms and intent_kind == "weak_intent":
        filters.setdefault("weak_intent", intent_terms)
        boost.setdefault("weak_intent", intent_terms)
        matched.setdefault("weak_intents", intent_terms)

    if card_names:
        db_route = routing.get("db_route")
        if db_route in (None, "guide_tbl"):
            routing["db_route"] = "both" if intent_terms else "card_tbl"

    if not routing.get("query_template"):
        query_template = _build_query_template(card_names, intent_terms)
        if query_template:
            routing["query_template"] = query_template

    if card_names or intent_terms:
        routing["should_search"] = True
        routing["should_trigger"] = True
        routing["should_route"] = True

    routing["filters"] = filters
    routing["boost"] = boost
    routing["matched"] = matched
    return routing


def should_search_consult_cases(
    query: str,
    routing: Dict[str, Any],
    session_state: Optional[Dict[str, Any]],
) -> bool:
    base_flag = bool(routing.get("need_consult_case_search"))

    now = time.time()
    last = None
    if session_state:
        last = session_state.get("consult_last_search_at")
        if last and _CONSULT_COOLDOWN_SEC > 0 and (now - float(last)) < _CONSULT_COOLDOWN_SEC:
            return False

    keyword_hits = int(routing.get("consult_keyword_hits") or 0)
    keyword_trigger = keyword_hits >= _CONSULT_MIN_KEYWORD_HITS

    intent_confidence = None
    if session_state and session_state.get("intent_confidence") is not None:
        intent_confidence = float(session_state.get("intent_confidence"))
    elif routing.get("intent_confidence") is not None:
        intent_confidence = float(routing.get("intent_confidence"))
    intent_trigger = intent_confidence is not None and intent_confidence >= _CONSULT_INTENT_THRESHOLD

    turn_trigger = False
    if session_state:
        turn = int(session_state.get("turn") or 0)
        if turn >= _CONSULT_MIN_TURNS:
            turn_trigger = True
        sentence_count = session_state.get("stt_sentence_count")
        if sentence_count is not None and int(sentence_count) >= _CONSULT_MIN_SENTENCES:
            turn_trigger = True

    if not (base_flag or keyword_trigger or intent_trigger or turn_trigger):
        return False

    if session_state is not None:
        session_state["consult_last_search_at"] = now
        session_state["consult_last_query"] = query
    return True


def strict_guidance_script(script: str, docs: List[Dict[str, Any]]) -> str:
    if not script:
        return ""
    content = " ".join(doc.get("content") or "" for doc in docs).strip()
    if not content:
        return ""
    normalized_content = normalize_text(content)
    sentences = [s.strip() for s in re.split(r"[.!?\\n]+", script) if s.strip()]
    for sentence in sentences:
        if normalize_text(sentence) not in normalized_content:
            return ""
    return script


def format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def build_retrieve_cache_entries(docs: List[Dict[str, Any]]) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for doc in docs:
        table = doc.get("table")
        doc_id = doc.get("db_id") or doc.get("id")
        if not table or doc_id is None:
            continue
        entries.append(
            {
                "table": str(table),
                "id": str(doc_id),
                "score": float(doc.get("score", 0.0)),
            }
        )
    return entries


def docs_from_retrieve_cache(entries: List[Dict[str, object]]) -> List[Dict[str, Any]]:
    if not entries:
        return []
    ids_by_table: Dict[str, List[str]] = {}
    for entry in entries:
        table = str(entry.get("table") or "")
        doc_id = entry.get("id")
        if not table or doc_id is None:
            continue
        ids_by_table.setdefault(table, []).append(str(doc_id))

    docs_by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
    for table, ids in ids_by_table.items():
        fetched = fetch_docs_by_ids(table, ids)
        for doc in fetched:
            key = (table, str(doc.get("db_id") or doc.get("id") or ""))
            docs_by_key[key] = doc

    docs: List[Dict[str, Any]] = []
    for entry in entries:
        table = str(entry.get("table") or "")
        doc_id = str(entry.get("id") or "")
        doc = docs_by_key.get((table, doc_id))
        if not doc:
            return []
        doc = dict(doc)
        doc["score"] = float(entry.get("score", doc.get("score", 0.0)))
        docs.append(doc)
    return docs
