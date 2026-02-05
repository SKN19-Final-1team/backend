from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.rag.pipeline import RAGConfig, run_rag

router = APIRouter()


def _ensure_list(value):
    if isinstance(value, list):
        return [v for v in value if v is not None]
    if value is None:
        return []
    return [value]


def _to_front_card(card):
    if not isinstance(card, dict):
        return {}
    document_type = card.get("documentType") or "general"
    content = card.get("content")
    if content in ("", None):
        content = None
    full_text = card.get("fullText")
    if full_text in ("", None):
        full_text = card.get("detailContent")
    if full_text in ("", None):
        full_text = None
    return {
        "id": str(card.get("id") or ""),
        "title": card.get("title") or "",
        "keywords": _ensure_list(card.get("keywords")),
        "content": content,
        "systemPath": card.get("systemPath") or "",
        "requiredChecks": _ensure_list(card.get("requiredChecks")),
        "exceptions": _ensure_list(card.get("exceptions")),
        "regulation": card.get("regulation") or "",
        "fullText": full_text,
        "time": card.get("time") or "",
        "note": card.get("note") or "",
        "documentType": document_type,
    }


class RAGFrontendRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4
    enable_consult_search: Optional[bool] = None


@router.post("/")
async def rag_frontend(request: RAGFrontendRequest):
    try:
        cfg_kwargs = {"top_k": request.top_k or 4}
        if request.enable_consult_search is not None:
            cfg_kwargs["enable_consult_search"] = request.enable_consult_search
        result = await run_rag(request.query, config=RAGConfig(**cfg_kwargs))
        current_cards = [_to_front_card(card) for card in (result.get("currentSituation") or [])]
        next_cards = [_to_front_card(card) for card in (result.get("nextStep") or [])]
        return {
            "guidanceScript": result.get("guidanceScript", ""),
            "currentSituation": current_cards,
            "nextStep": next_cards,
            "currentSituationCards": current_cards,
            "nextStepCards": next_cards,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
