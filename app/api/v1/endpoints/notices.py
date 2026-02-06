"""
Notices API 엔드포인트

대시보드 공지사항 조회
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import date, datetime
import psycopg2.extras

from app.db.base import get_connection

router = APIRouter()


# ==============================================================================
# Response Models
# ==============================================================================

class NoticeResponse(BaseModel):
    """공지사항 응답 모델 (Frontend 호환)"""
    id: int
    tag: str
    title: str
    date: str
    author: str
    views: int
    pinned: bool
    content: str


class NoticeListResponse(BaseModel):
    """공지사항 목록 응답"""
    success: bool
    data: List[NoticeResponse]
    total: int
    message: str


class NoticeCreateRequest(BaseModel):
    """공지사항 생성 요청"""
    tag: str
    title: str
    content: str
    author: Optional[str] = "관리자"
    pinned: bool = False


class NoticeUpdateRequest(BaseModel):
    """공지사항 수정 요청"""
    tag: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    pinned: Optional[bool] = None


# ==============================================================================
# Helper Functions
# ==============================================================================

def map_category_to_tag(category: str, priority: str) -> str:
    """DB category/priority를 Frontend tag로 매핑"""
    # priority가 urgent이면 '긴급'
    if priority == 'urgent':
        return '긴급'

    # category 기반 매핑
    category_map = {
        'emergency': '긴급',
        'system': '시스템',
        'service': '서비스',
        'event': '이벤트',
        'policy': '정책',
        'education': '교육',
        'work': '근무',
        'welfare': '복지',
    }
    return category_map.get(category, '공지')


def map_tag_to_category(tag: str) -> tuple:
    """Frontend tag를 DB category/priority로 매핑"""
    tag_map = {
        '긴급': ('emergency', 'urgent'),
        '시스템': ('system', 'normal'),
        '서비스': ('service', 'normal'),
        '이벤트': ('event', 'normal'),
        '정책': ('policy', 'important'),
        '교육': ('education', 'normal'),
        '근무': ('work', 'normal'),
        '복지': ('welfare', 'normal'),
        '공지': ('system', 'normal'),
    }
    return tag_map.get(tag, ('system', 'normal'))


def generate_notice_id(cur) -> str:
    """새 공지사항 ID 생성"""
    cur.execute("SELECT id FROM notices ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    if row and row.get('id', '').startswith('notice_'):
        try:
            last_num = int(row['id'].replace('notice_', ''))
            return f"notice_{str(last_num + 1).zfill(2)}"
        except:
            pass
    return "notice_53"  # 기본값 (52개 이후)


def row_to_notice_response(row: dict, index: int) -> NoticeResponse:
    """DB 행을 NoticeResponse로 변환"""
    # id를 숫자로 변환 (notice_01 -> 1)
    notice_id = index + 1
    try:
        if row.get('id', '').startswith('notice_'):
            notice_id = int(row['id'].replace('notice_', ''))
    except:
        pass

    # 날짜 포맷
    created_at = row.get('created_at')
    if isinstance(created_at, datetime):
        date_str = created_at.strftime('%Y-%m-%d')
    elif isinstance(created_at, date):
        date_str = created_at.strftime('%Y-%m-%d')
    else:
        date_str = str(created_at)[:10] if created_at else ''

    return NoticeResponse(
        id=notice_id,
        tag=map_category_to_tag(row.get('category', ''), row.get('priority', '')),
        title=row.get('title', ''),
        date=date_str,
        author=row.get('created_by', '관리자') or '관리자',
        views=row.get('view_count', 0) or 0,
        pinned=row.get('is_pinned', False) or False,
        content=row.get('content', '')
    )


# ==============================================================================
# API Endpoints
# ==============================================================================

@router.get("", response_model=NoticeListResponse)
async def get_notices(
    limit: int = Query(default=10, ge=1, le=100, description="조회 개수"),
    offset: int = Query(default=0, ge=0, description="오프셋"),
    pinned_only: bool = Query(default=False, description="고정 공지만 조회"),
    sort_by: str = Query(default="pinned_first", description="정렬 방식: pinned_first(고정우선+최신순), date_only(최신순만)")
):
    """
    공지사항 목록 조회

    대시보드에서 사용하는 공지사항 목록을 조회합니다.
    is_pinned=true인 공지사항이 먼저 표시되고, 이후 최신순으로 정렬됩니다.

    Args:
        limit: 조회 개수 (기본 10, 최대 100)
        offset: 오프셋 (페이지네이션용)
        pinned_only: true이면 고정 공지만 조회

    Returns:
        NoticeListResponse: 공지사항 목록
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 전체 개수 조회
            count_query = "SELECT COUNT(*) FROM notices WHERE status = 'active'"
            if pinned_only:
                count_query += " AND is_pinned = true"
            cur.execute(count_query)
            total = cur.fetchone()['count']

            # 공지사항 목록 조회
            query = """
                SELECT
                    id, title, content, category, priority,
                    is_pinned, created_by, created_at, view_count
                FROM notices
                WHERE status = 'active'
            """
            if pinned_only:
                query += " AND is_pinned = true"

            # 정렬 방식 선택
            if sort_by == "date_only":
                query += """
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            else:  # pinned_first (기본값)
                query += """
                ORDER BY is_pinned DESC, created_at DESC
                LIMIT %s OFFSET %s
            """

            cur.execute(query, (limit, offset))
            rows = cur.fetchall()

            notices = [row_to_notice_response(row, i + offset) for i, row in enumerate(rows)]

            return NoticeListResponse(
                success=True,
                data=notices,
                total=total,
                message=f"공지사항 {len(notices)}건 조회 완료"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"공지사항 조회 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.get("/{notice_id}", response_model=dict)
async def get_notice_by_id(notice_id: str):
    """
    특정 공지사항 조회

    Args:
        notice_id: 공지사항 ID (예: notice_01 또는 1)

    Returns:
        공지사항 상세 정보
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # ID 형식 처리 (숫자면 notice_XX로 변환)
            if notice_id.isdigit():
                db_id = f"notice_{notice_id.zfill(2)}"
            else:
                db_id = notice_id

            cur.execute("""
                SELECT
                    id, title, content, category, priority,
                    is_pinned, created_by, created_at, view_count
                FROM notices
                WHERE id = %s AND status = 'active'
            """, (db_id,))

            row = cur.fetchone()

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"공지사항을 찾을 수 없습니다: {notice_id}"
                )

            notice = row_to_notice_response(row, 0)

            return {
                "success": True,
                "data": notice.model_dump(),
                "message": "공지사항 조회 성공"
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"공지사항 조회 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.post("", response_model=dict)
async def create_notice(request: NoticeCreateRequest):
    """
    공지사항 생성

    Args:
        request: 공지사항 생성 요청 (tag, title, content, author, pinned)

    Returns:
        생성된 공지사항 정보
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 새 ID 생성
            new_id = generate_notice_id(cur)

            # tag를 category/priority로 변환
            category, priority = map_tag_to_category(request.tag)

            # 오늘 날짜
            today = date.today()

            cur.execute("""
                INSERT INTO notices (
                    id, title, content, category, priority,
                    is_pinned, start_date, created_by, status, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, 'active', NOW()
                )
                RETURNING id, title, content, category, priority, is_pinned, created_by, created_at
            """, (
                new_id, request.title, request.content, category, priority,
                request.pinned, today, request.author
            ))

            row = cur.fetchone()
            conn.commit()

            notice = row_to_notice_response(row, 0)

            return {
                "success": True,
                "data": notice.model_dump(),
                "message": "공지사항 생성 완료"
            }

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"공지사항 생성 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.put("/{notice_id}", response_model=dict)
async def update_notice(notice_id: str, request: NoticeUpdateRequest):
    """
    공지사항 수정

    Args:
        notice_id: 공지사항 ID
        request: 수정할 필드들

    Returns:
        수정된 공지사항 정보
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # ID 형식 처리
            if notice_id.isdigit():
                db_id = f"notice_{notice_id.zfill(2)}"
            else:
                db_id = notice_id

            # 기존 공지사항 확인
            cur.execute("SELECT * FROM notices WHERE id = %s", (db_id,))
            existing = cur.fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail=f"공지사항을 찾을 수 없습니다: {notice_id}")

            # 업데이트할 필드 구성
            updates = []
            params = []

            if request.title is not None:
                updates.append("title = %s")
                params.append(request.title)

            if request.content is not None:
                updates.append("content = %s")
                params.append(request.content)

            if request.tag is not None:
                category, priority = map_tag_to_category(request.tag)
                updates.append("category = %s")
                updates.append("priority = %s")
                params.append(category)
                params.append(priority)

            if request.pinned is not None:
                updates.append("is_pinned = %s")
                params.append(request.pinned)

            if not updates:
                raise HTTPException(status_code=400, detail="수정할 내용이 없습니다")

            updates.append("updated_at = NOW()")
            params.append(db_id)

            cur.execute(f"""
                UPDATE notices SET {', '.join(updates)}
                WHERE id = %s
                RETURNING id, title, content, category, priority, is_pinned, created_by, created_at
            """, params)

            row = cur.fetchone()
            conn.commit()

            notice = row_to_notice_response(row, 0)

            return {
                "success": True,
                "data": notice.model_dump(),
                "message": "공지사항 수정 완료"
            }

    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"공지사항 수정 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.delete("/{notice_id}", response_model=dict)
async def delete_notice(notice_id: str):
    """
    공지사항 삭제 (soft delete - status를 'inactive'로 변경)

    Args:
        notice_id: 공지사항 ID

    Returns:
        삭제 결과
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # ID 형식 처리
            if notice_id.isdigit():
                db_id = f"notice_{notice_id.zfill(2)}"
            else:
                db_id = notice_id

            # soft delete
            cur.execute("""
                UPDATE notices SET status = 'inactive', updated_at = NOW()
                WHERE id = %s AND status = 'active'
                RETURNING id
            """, (db_id,))

            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"공지사항을 찾을 수 없습니다: {notice_id}")

            conn.commit()

            return {
                "success": True,
                "message": f"공지사항 삭제 완료: {notice_id}"
            }

    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"공지사항 삭제 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.patch("/{notice_id}/pin", response_model=dict)
async def toggle_pin_notice(notice_id: str):
    """
    공지사항 핀 토글

    Args:
        notice_id: 공지사항 ID

    Returns:
        토글 결과
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # ID 형식 처리
            if notice_id.isdigit():
                db_id = f"notice_{notice_id.zfill(2)}"
            else:
                db_id = notice_id

            # 현재 상태 확인 후 토글
            cur.execute("""
                UPDATE notices SET is_pinned = NOT is_pinned, updated_at = NOW()
                WHERE id = %s AND status = 'active'
                RETURNING id, is_pinned
            """, (db_id,))

            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"공지사항을 찾을 수 없습니다: {notice_id}")

            conn.commit()

            return {
                "success": True,
                "data": {"id": notice_id, "pinned": row['is_pinned']},
                "message": f"공지사항 {'고정' if row['is_pinned'] else '고정 해제'} 완료"
            }

    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"공지사항 핀 토글 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.patch("/{notice_id}/view", response_model=dict)
async def increment_view_count(notice_id: str):
    """
    공지사항 조회수 증가

    공지사항 상세 조회 시 호출하여 조회수를 1 증가시킵니다.

    Args:
        notice_id: 공지사항 ID

    Returns:
        증가된 조회수
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # ID 형식 처리
            if notice_id.isdigit():
                db_id = f"notice_{notice_id.zfill(2)}"
            else:
                db_id = notice_id

            # 조회수 증가
            cur.execute("""
                UPDATE notices SET view_count = COALESCE(view_count, 0) + 1
                WHERE id = %s AND status = 'active'
                RETURNING id, view_count
            """, (db_id,))

            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"공지사항을 찾을 수 없습니다: {notice_id}")

            conn.commit()

            return {
                "success": True,
                "data": {"id": notice_id, "views": row['view_count']},
                "message": f"조회수 증가: {row['view_count']}"
            }

    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"조회수 증가 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()
