"""
Frequent Inquiries API 엔드포인트

자주 찾는 문의 조회 API
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
import psycopg2.extras

from app.db.base import get_connection

router = APIRouter()


# ==============================================================================
# Response Models
# ==============================================================================

class RelatedDocumentResponse(BaseModel):
    """관련 문서 응답 모델"""
    document_id: str
    title: str
    regulation: Optional[str] = None
    summary: Optional[str] = None


class FrequentInquiryResponse(BaseModel):
    """자주 찾는 문의 응답 모델"""
    id: int
    keyword: str
    question: str
    count: int
    trend: str  # up/down/same
    content: Optional[str] = None
    relatedDocument: Optional[RelatedDocumentResponse] = None


class FrequentInquiryListResponse(BaseModel):
    """자주 찾는 문의 목록 응답"""
    success: bool
    data: List[FrequentInquiryResponse]
    total: int
    message: str


# ==============================================================================
# Helper Functions
# ==============================================================================

def row_to_response(row: dict) -> FrequentInquiryResponse:
    """DB 행을 FrequentInquiryResponse로 변환"""
    related_doc = None
    if row.get('related_document_id'):
        related_doc = RelatedDocumentResponse(
            document_id=row.get('related_document_id', ''),
            title=row.get('related_document_title', ''),
            regulation=row.get('related_document_regulation'),
            summary=row.get('related_document_summary')
        )

    return FrequentInquiryResponse(
        id=row.get('id', 0),
        keyword=row.get('keyword', ''),
        question=row.get('question', ''),
        count=row.get('count', 0) or 0,
        trend=row.get('trend', 'same') or 'same',
        content=row.get('content'),
        relatedDocument=related_doc
    )


# ==============================================================================
# API Endpoints
# ==============================================================================

@router.get("", response_model=FrequentInquiryListResponse)
async def get_frequent_inquiries(
    limit: int = Query(default=10, ge=1, le=50, description="조회 개수"),
    offset: int = Query(default=0, ge=0, description="오프셋")
):
    """
    자주 찾는 문의 목록 조회

    대시보드에서 자주 찾는 문의를 표시할 때 사용합니다.
    count(문의 건수) 기준 내림차순 정렬됩니다.

    Args:
        limit: 조회 개수 (기본 10, 최대 50)
        offset: 오프셋 (페이지네이션용)

    Returns:
        FrequentInquiryListResponse: 자주 찾는 문의 목록
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 전체 개수 조회
            cur.execute("SELECT COUNT(*) FROM frequent_inquiries WHERE is_active = true")
            total = cur.fetchone()['count']

            # 문의 목록 조회
            cur.execute("""
                SELECT
                    id, keyword, question, count, trend, content,
                    related_document_id, related_document_title,
                    related_document_regulation, related_document_summary
                FROM frequent_inquiries
                WHERE is_active = true
                ORDER BY count DESC, id ASC
                LIMIT %s OFFSET %s
            """, (limit, offset))
            rows = cur.fetchall()

            inquiries = [row_to_response(row) for row in rows]

            return FrequentInquiryListResponse(
                success=True,
                data=inquiries,
                total=total,
                message=f"자주 찾는 문의 {len(inquiries)}건 조회 완료"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"자주 찾는 문의 조회 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.get("/{inquiry_id}", response_model=dict)
async def get_frequent_inquiry_by_id(inquiry_id: int):
    """
    특정 자주 찾는 문의 조회

    Args:
        inquiry_id: 문의 ID

    Returns:
        문의 상세 정보
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    id, keyword, question, count, trend, content,
                    related_document_id, related_document_title,
                    related_document_regulation, related_document_summary
                FROM frequent_inquiries
                WHERE id = %s AND is_active = true
            """, (inquiry_id,))

            row = cur.fetchone()

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"문의를 찾을 수 없습니다: {inquiry_id}"
                )

            inquiry = row_to_response(row)

            return {
                "success": True,
                "data": inquiry.model_dump(),
                "message": "문의 조회 성공"
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"문의 조회 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()
