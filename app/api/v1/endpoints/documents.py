"""
Documents API 엔드포인트 (Phase B)

범용 문서 상세 조회 API
- FAQ 관련문서 클릭 → 실 가이드 문서 상세 표시
- 상담 이력 참조문서 클릭 → 문서 재조회
- (documentId, sourceTable) 조합으로 모든 문서를 유일하게 식별
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Any
import psycopg2.extras
import json

from app.db.base import get_connection

router = APIRouter()


# ==============================================================================
# Response Models
# ==============================================================================

class DocumentDetailResponse(BaseModel):
    """문서 상세 응답"""
    id: str
    title: str
    content: Optional[str] = None
    fullText: Optional[str] = None
    documentType: Optional[str] = None
    sourceTable: str
    keywords: List[str] = []
    category: Optional[str] = None
    regulation: Optional[str] = None
    systemPath: Optional[str] = None
    metadata: Optional[dict] = None


# ==============================================================================
# 테이블별 조회 로직
# ==============================================================================

# 허용된 테이블 목록 (SQL injection 방지)
ALLOWED_TABLES = {
    "card_products",
    "service_guide_documents",
    "notices",
    "consultation_documents",
}


def _fetch_card_product(cur, doc_id: str) -> Optional[dict]:
    cur.execute("""
        SELECT id, name, keywords, structured, metadata
        FROM card_products WHERE id = %s
    """, (doc_id,))
    row = cur.fetchone()
    if not row:
        return None
    structured = row.get("structured") or {}
    meta = row.get("metadata") or {}
    return {
        "id": row["id"],
        "title": structured.get("title") or row.get("name", ""),
        "content": structured.get("content"),
        "fullText": structured.get("detailContent") or meta.get("full_content"),
        "documentType": "product-spec",
        "sourceTable": "card_products",
        "keywords": row.get("keywords") or [],
        "category": meta.get("category") or row.get("card_type"),
        "regulation": structured.get("regulation"),
        "systemPath": structured.get("systemPath") or meta.get("system_path"),
        "metadata": meta,
    }


def _fetch_guide_document(cur, doc_id: str) -> Optional[dict]:
    cur.execute("""
        SELECT id, title, content, metadata, document_type, structured
        FROM service_guide_documents WHERE id = %s
    """, (doc_id,))
    row = cur.fetchone()
    if not row:
        return None
    structured = row.get("structured") or {}
    meta = row.get("metadata") or {}
    return {
        "id": row["id"],
        "title": structured.get("title") or row.get("title", ""),
        "content": structured.get("content") or _summarize(row.get("content")),
        "fullText": structured.get("detailContent") or meta.get("full_content") or row.get("content"),
        "documentType": row.get("document_type") or "guide",
        "sourceTable": "service_guide_documents",
        "keywords": meta.get("keywords") or [],
        "category": meta.get("category"),
        "regulation": structured.get("regulation") or meta.get("regulation"),
        "systemPath": structured.get("systemPath") or meta.get("system_path"),
        "metadata": meta,
    }


def _fetch_notice(cur, doc_id: str) -> Optional[dict]:
    cur.execute("""
        SELECT id, title, content, category, keywords, metadata
        FROM notices WHERE id = %s
    """, (doc_id,))
    row = cur.fetchone()
    if not row:
        return None
    meta = row.get("metadata") or {}
    return {
        "id": row["id"],
        "title": row.get("title", ""),
        "content": _summarize(row.get("content")),
        "fullText": row.get("content"),
        "documentType": "notice",
        "sourceTable": "notices",
        "keywords": row.get("keywords") or [],
        "category": row.get("category"),
        "regulation": None,
        "systemPath": None,
        "metadata": meta,
    }


def _fetch_consultation_document(cur, doc_id: str) -> Optional[dict]:
    cur.execute("""
        SELECT id, title, content, metadata, document_type
        FROM consultation_documents WHERE id = %s
    """, (doc_id,))
    row = cur.fetchone()
    if not row:
        return None
    meta = row.get("metadata") or {}
    return {
        "id": row["id"],
        "title": row.get("title", ""),
        "content": _summarize(row.get("content")),
        "fullText": row.get("content"),
        "documentType": row.get("document_type") or "consultation",
        "sourceTable": "consultation_documents",
        "keywords": meta.get("keywords") or [],
        "category": meta.get("category"),
        "regulation": None,
        "systemPath": None,
        "metadata": meta,
    }


_TABLE_FETCHERS = {
    "card_products": _fetch_card_product,
    "service_guide_documents": _fetch_guide_document,
    "notices": _fetch_notice,
    "consultation_documents": _fetch_consultation_document,
}


def _summarize(text: Optional[str], limit: int = 200) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


# ==============================================================================
# API Endpoints
# ==============================================================================

@router.get("/{document_id}")
async def get_document(
    document_id: str,
    source: Optional[str] = Query(
        default=None,
        description="원본 테이블명 (card_products, service_guide_documents, notices, consultation_documents)"
    ),
):
    """
    문서 상세 조회

    sourceTable이 지정되면 해당 테이블만 조회 (빠름).
    미지정이면 모든 테이블 순회하여 검색 (느림, fallback).
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # source가 지정된 경우: 해당 테이블만 조회
            if source and source in ALLOWED_TABLES:
                fetcher = _TABLE_FETCHERS[source]
                result = fetcher(cur, document_id)
                if result:
                    return {"success": True, "data": result, "message": "문서 조회 성공"}
                raise HTTPException(status_code=404, detail=f"문서를 찾을 수 없습니다: {document_id} (table={source})")

            # source 미지정: 모든 테이블 순회
            for table_name, fetcher in _TABLE_FETCHERS.items():
                result = fetcher(cur, document_id)
                if result:
                    return {"success": True, "data": result, "message": "문서 조회 성공"}

            raise HTTPException(status_code=404, detail=f"문서를 찾을 수 없습니다: {document_id}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 조회 중 오류: {str(e)}")
    finally:
        if conn:
            conn.close()
