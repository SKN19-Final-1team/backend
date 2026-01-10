from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.llm.rag_answerer import generate_answer
from app.rag.retriever import retrieve_docs
from app.rag.router import route_query


@dataclass(frozen=True)
class RAGConfig:
    top_k: int = 5
    model: str = "gpt-4.1"
    temperature: float = 0.2
    no_route_answer: str = "카드명/상황을 조금 더 구체적으로 말씀해 주세요."
    include_docs: bool = True


def route(query: str) -> Dict[str, Any]:
    return route_query(query)


async def retrieve(
    query: str,
    routing: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    filters = routing.get("filters") or {}
    query_template = routing.get("query_template")

    merged: List[Dict[str, Any]] = []

    intent_terms = filters.get("intent")
    has_card = bool(filters.get("card_name"))
    card_filters: Dict[str, Any] = {}
    if filters.get("card_name"):
        card_filters["card_name"] = filters["card_name"]
    if filters.get("payment_method"):
        card_filters["payment_method"] = filters["payment_method"]
    if filters.get("weak_intent"):
        card_filters["weak_intent"] = filters["weak_intent"]

    if card_filters:
        card_routing = {"filters": card_filters, "query_template": query_template}
        card_docs = await retrieve_docs(
            query=query,
            routing=card_routing,
            top_k=top_k,
            table="card_tbl",
            allow_fallback=False,
        )
        merged.extend(card_docs)

    if intent_terms and (not has_card or not merged):
        guide_query_template = f"{intent_terms[0]} 방법"
        guide_routing = {"filters": {"intent": intent_terms}, "query_template": guide_query_template}
        guide_docs = await retrieve_docs(
            query=query,
            routing=guide_routing,
            top_k=top_k,
            table="guide_tbl",
            allow_fallback=False,
        )
        merged.extend(guide_docs)

    if not merged:
        merged = await retrieve_docs(query=query, routing=routing, top_k=top_k)

    for doc in merged:
        if has_card:
            doc["_table_priority"] = 1 if doc.get("table") == "card_tbl" else 0
        else:
            doc["_table_priority"] = 1 if intent_terms and doc.get("table") == "guide_tbl" else 0
    merged.sort(
        key=lambda doc: (doc.get("_table_priority", 0), doc.get("title_score", 0), doc.get("score", 0.0)),
        reverse=True,
    )
    out: List[Dict[str, Any]] = []
    seen_titles = set()
    for doc in merged:
        title = doc.get("title")
        if title and title in seen_titles:
            continue
        if title:
            seen_titles.add(title)
        out.append(doc)
        if len(out) >= top_k:
            break

    return out


def answer(
    query: str,
    docs: List[Dict[str, Any]],
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    return generate_answer(query=query, docs=docs, model=model, temperature=temperature)


async def run_rag(query: str, config: Optional[RAGConfig] = None) -> Dict[str, Any]:
    cfg = config or RAGConfig()
    routing = route(query)

    if not routing.get("should_route"):
        return {
            "answer": cfg.no_route_answer,
            "routing": routing,
            "docs": [],
            "meta": {"model": None, "doc_count": 0, "context_chars": 0},
        }

    docs = await retrieve(query=query, routing=routing, top_k=cfg.top_k)
    llm = answer(query=query, docs=docs, model=cfg.model, temperature=cfg.temperature)

    response = {
        "answer": llm["answer"],
        "routing": routing,
        "meta": {k: llm[k] for k in ("model", "doc_count", "context_chars")},
    }
    response["docs"] = docs if cfg.include_docs else []
    return response
