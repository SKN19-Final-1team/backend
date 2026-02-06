"""
Customers API 엔드포인트

DirectCall 시 실제 DB에서 랜덤 고객 정보 조회
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import date
import psycopg2.extras

from app.db.base import get_connection

router = APIRouter()


# ==============================================================================
# Response Models
# ==============================================================================

class CustomerResponse(BaseModel):
    """고객 정보 응답 모델 (Frontend 호환)"""
    id: str
    name: str
    phone: str
    birthDate: Optional[str] = None
    address: Optional[str] = None
    grade: Optional[str] = None
    cardName: Optional[str] = None
    cardNumber: Optional[str] = None  # 마지막 4자리만
    cardIssueDate: Optional[str] = None
    cardExpiryDate: Optional[str] = None
    cardBrand: Optional[str] = None
    # 페르소나 정보 (LLM 가이던스용)
    currentTypeCode: Optional[str] = None
    personalityTags: Optional[List[str]] = None
    llmGuidance: Optional[str] = None


class ApiResponse(BaseModel):
    """공통 API 응답 포맷"""
    success: bool
    data: Optional[CustomerResponse] = None
    message: str
    timestamp: Optional[str] = None


class ConsultationHistoryItem(BaseModel):
    """최근 상담 내역 항목"""
    id: str
    title: str
    date: str
    category: str
    status: str


class ConsultationHistoryResponse(BaseModel):
    """최근 상담 내역 응답"""
    success: bool
    data: List[ConsultationHistoryItem]
    message: str


# ==============================================================================
# Helper Functions
# ==============================================================================

def format_date(d: date) -> Optional[str]:
    """date 객체를 문자열로 변환"""
    if d is None:
        return None
    return d.strftime("%Y-%m-%d")


def format_card_number(last4: str) -> Optional[str]:
    """카드번호 마지막 4자리를 마스킹된 형식으로 변환"""
    if not last4:
        return None
    return f"****-****-****-{last4}"


def row_to_customer_response(row: dict) -> CustomerResponse:
    """DB 행을 CustomerResponse로 변환"""
    return CustomerResponse(
        id=row.get('id', ''),
        name=row.get('name', ''),
        phone=row.get('phone', ''),
        birthDate=format_date(row.get('birth_date')),
        address=row.get('address'),
        grade=row.get('grade'),
        cardName=row.get('card_type'),
        cardNumber=format_card_number(row.get('card_number_last4')),
        cardIssueDate=format_date(row.get('card_issue_date')),
        cardExpiryDate=format_date(row.get('card_expiry_date')),
        cardBrand=row.get('card_brand'),
        currentTypeCode=row.get('current_type_code'),
        personalityTags=row.get('personality_tags'),
        llmGuidance=row.get('llm_guidance')
    )


# ==============================================================================
# API Endpoints
# ==============================================================================

@router.get("/random", response_model=ApiResponse)
async def get_random_customer():
    """
    랜덤 고객 정보 조회

    DirectCall 인입 시 실제 DB에서 랜덤으로 고객 1명을 조회합니다.
    Frontend의 defaultCustomerInfo를 대체하는 API입니다.

    Returns:
        ApiResponse: 랜덤 고객 정보
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 랜덤으로 고객 1명 조회 (PostgreSQL RANDOM() 사용)
            # persona_types 테이블과 JOIN하여 personality_tags, llm_guidance 가져옴
            cur.execute("""
                SELECT
                    c.id, c.name, c.phone, c.birth_date, c.address, c.grade,
                    c.card_type, c.card_number_last4, c.card_brand,
                    c.card_issue_date, c.card_expiry_date,
                    c.current_type_code,
                    p.personality_tags,
                    p.llm_guidance
                FROM customers c
                LEFT JOIN persona_types p ON c.current_type_code = p.code
                ORDER BY RANDOM()
                LIMIT 1
            """)
            row = cur.fetchone()

            if not row:
                # 고객이 없는 경우 기본값 반환
                return ApiResponse(
                    success=True,
                    data=CustomerResponse(
                        id='CUST-TEDDY-DEFAULT',
                        name='테스트 고객',
                        phone='010-0000-0000',
                        birthDate=None,
                        address=None,
                        grade='GENERAL'
                    ),
                    message="등록된 고객이 없어 기본값을 반환합니다."
                )

            customer = row_to_customer_response(row)
            return ApiResponse(
                success=True,
                data=customer,
                message="랜덤 고객 정보 조회 성공"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"고객 정보 조회 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.get("/{customer_id}", response_model=ApiResponse)
async def get_customer_by_id(customer_id: str):
    """
    특정 고객 정보 조회

    Args:
        customer_id: 고객 ID (예: CUST-TEDDY-00001)

    Returns:
        ApiResponse: 고객 정보
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # persona_types 테이블과 JOIN하여 personality_tags, llm_guidance 가져옴
            cur.execute("""
                SELECT
                    c.id, c.name, c.phone, c.birth_date, c.address, c.grade,
                    c.card_type, c.card_number_last4, c.card_brand,
                    c.card_issue_date, c.card_expiry_date,
                    c.current_type_code,
                    p.personality_tags,
                    p.llm_guidance
                FROM customers c
                LEFT JOIN persona_types p ON c.current_type_code = p.code
                WHERE c.id = %s
            """, (customer_id,))
            row = cur.fetchone()

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"고객을 찾을 수 없습니다: {customer_id}"
                )

            customer = row_to_customer_response(row)
            return ApiResponse(
                success=True,
                data=customer,
                message="고객 정보 조회 성공"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"고객 정보 조회 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


# 카테고리 매핑 (DB ENUM → 한글)
CATEGORY_MAP = {
    'card_loss': '카드분실',
    'overseas_payment': '해외결제',
    'fee_inquiry': '수수료문의',
    'points': '포인트',
    'limit_inquiry': '한도조회',
    'other': '기타',
}

# 상태 매핑 (DB ENUM → 한글)
STATUS_MAP = {
    'completed': '완료',
    'in_progress': '진행중',
    'incomplete': '미완료',
}


@router.get("/{customer_id}/consultations", response_model=ConsultationHistoryResponse)
async def get_customer_consultations(customer_id: str, limit: int = 3):
    """
    고객의 최근 상담 내역 조회

    Args:
        customer_id: 고객 ID (예: CUST-TEDDY-00001)
        limit: 조회할 최대 건수 (기본값: 3)

    Returns:
        ConsultationHistoryResponse: 최근 상담 내역 리스트
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 해당 고객의 최근 상담 내역 조회
            # ⭐ [v24] category → category_main 수정 (DB 스키마와 일치)
            cur.execute("""
                SELECT
                    id,
                    title,
                    category_main,
                    category_raw,
                    status,
                    call_date,
                    call_time
                FROM consultations
                WHERE customer_id = %s
                ORDER BY call_date DESC, call_time DESC
                LIMIT %s
            """, (customer_id, limit))
            rows = cur.fetchall()

            # 결과 변환
            consultations = []
            for row in rows:
                # 날짜/시간 포맷팅
                call_date = row.get('call_date')
                call_time = row.get('call_time')
                if call_date and call_time:
                    date_str = f"{call_date.strftime('%Y-%m-%d')} {str(call_time)[:5]}"
                elif call_date:
                    date_str = call_date.strftime('%Y-%m-%d')
                else:
                    date_str = ''

                # 카테고리/상태 한글 변환 (category_main 또는 category_raw 사용)
                raw_category = row.get('category_main') or row.get('category_raw') or '기타'
                category = CATEGORY_MAP.get(raw_category, raw_category)
                status = STATUS_MAP.get(row.get('status'), row.get('status') or '완료')

                consultations.append(ConsultationHistoryItem(
                    id=row.get('id', ''),
                    title=row.get('title') or '상담 내역',
                    date=date_str,
                    category=category,
                    status=status,
                ))

            return ConsultationHistoryResponse(
                success=True,
                data=consultations,
                message=f"최근 상담 내역 {len(consultations)}건 조회 성공"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"상담 내역 조회 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()
