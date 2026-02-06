"""
Consultations API 엔드포인트

후처리(ACW) 상담 저장 API
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import psycopg2.extras
import json

from app.db.base import get_connection

router = APIRouter()


# ==============================================================================
# Request/Response Models
# ==============================================================================

class ReferencedDocument(BaseModel):
    """참조 문서 (Frontend 스키마 호환)"""
    stepNumber: Optional[int] = None       # RAG 조회 순서
    documentId: str                        # DOC-123
    title: Optional[str] = ""              # 제목 (없으면 빈 문자열)
    used: Optional[bool] = False           # 클릭 여부
    viewCount: Optional[int] = 0           # 조회 횟수


class ProcessingTimelineItem(BaseModel):
    """처리 타임라인 항목"""
    time: str
    action: str
    category: Optional[str] = None


class SaveConsultationRequest(BaseModel):
    """상담 저장 요청 모델 (Frontend 호환)"""
    consultationId: str
    employeeId: str
    customerId: str
    customerName: str
    category: str
    title: Optional[str] = ""
    status: str
    datetime: str
    callTimeSeconds: int
    acwTimeSeconds: int
    aiSummary: str
    memo: str
    transcript: Optional[str] = None
    followUpTasks: str
    handoffDepartment: Optional[str] = ""
    handoffNotes: Optional[str] = ""
    referencedDocuments: List[ReferencedDocument] = []
    referencedDocumentIds: List[str] = []
    processingTimeline: List[ProcessingTimelineItem] = []  # ⭐ 처리 타임라인
    sentiment: Optional[str] = None
    feedbackScore: Optional[int] = None
    satisfactionScore: Optional[int] = None


class ApiResponse(BaseModel):
    """공통 API 응답"""
    success: bool
    data: Optional[dict] = None
    message: str


# ==============================================================================
# Helper Functions
# ==============================================================================

def parse_datetime(dt_str: str) -> tuple:
    """datetime 문자열을 date와 time으로 분리"""
    try:
        # "2026-02-05 09:06" 형식
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        return dt.date(), dt.time()
    except:
        # 파싱 실패 시 현재 시간
        now = datetime.now()
        return now.date(), now.time()


def seconds_to_duration(seconds: int) -> str:
    """초를 MM:SS 형식으로 변환"""
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def parse_category(category: str) -> tuple:
    """카테고리 문자열 파싱 (대분류 > 소분류)"""
    if '>' in category:
        parts = category.split('>')
        return parts[0].strip(), parts[1].strip() if len(parts) > 1 else parts[0].strip()
    return category, category


def map_status_to_enum(status: str) -> str:
    """한글 상태를 DB enum 값으로 변환"""
    status_map = {
        "진행중": "in_progress",
        "완료": "completed",
        "미완료": "incomplete",
        # 영문 값은 그대로 통과
        "in_progress": "in_progress",
        "completed": "completed",
        "incomplete": "incomplete",
    }
    return status_map.get(status, "completed")  # 기본값: completed


# ==============================================================================
# API Endpoints
# ==============================================================================

class ConsultationListItem(BaseModel):
    """상담 목록 아이템"""
    id: str
    agent: str
    agentId: Optional[str] = None
    customer: str
    customerId: Optional[str] = None
    category: str
    categoryMain: Optional[str] = None
    categorySub: Optional[str] = None
    status: str
    content: str
    datetime: str
    duration: str
    isBestPractice: bool = False
    isSimulation: bool = False
    fcr: bool = True
    memo: Optional[str] = None
    team: Optional[str] = None


class ConsultationListResponse(BaseModel):
    """상담 목록 응답"""
    success: bool
    data: List[ConsultationListItem]
    total: int
    message: str


# 상태 매핑 (영문 enum → 한글)
STATUS_TO_KOREAN = {
    'completed': '완료',
    'in_progress': '진행중',
    'incomplete': '미완료',
}

# 팀 매핑 (agent_id 기반)
def get_team_by_agent(agent_id: str) -> str:
    """상담사 ID로 팀 반환"""
    team_map = {
        'EMP001': '1팀', 'EMP002': '1팀', 'EMP003': '1팀',
        'EMP004': '2팀', 'EMP005': '2팀', 'EMP006': '2팀',
        'EMP007': '3팀', 'EMP008': '3팀', 'EMP009': '3팀',
    }
    return team_map.get(agent_id, '1팀')


@router.get("", response_model=ConsultationListResponse)
async def get_consultations(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    category: Optional[str] = None,
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    """
    상담 목록 조회

    Args:
        limit: 조회 개수 (기본 100)
        offset: 오프셋
        status: 상태 필터 (완료/진행중/미완료)
        category: 카테고리 필터
        agent_id: 상담사 ID 필터
        date_from: 시작일 (YYYY-MM-DD)
        date_to: 종료일 (YYYY-MM-DD)

    Returns:
        ConsultationListResponse: 상담 목록
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 기본 쿼리
            query = """
                SELECT
                    c.id,
                    c.agent_id,
                    c.customer_id,
                    c.category_main,
                    c.category_sub,
                    c.status,
                    c.title,
                    c.ai_summary,
                    c.agent_notes,
                    c.call_date,
                    c.call_time,
                    c.call_duration,
                    c.is_best_practice,
                    c.fcr,
                    COALESCE(cust.name, '알 수 없음') as customer_name,
                    COALESCE(emp.name, '알 수 없음') as agent_name
                FROM consultations c
                LEFT JOIN customers cust ON c.customer_id = cust.id
                LEFT JOIN employees emp ON c.agent_id = emp.id
                WHERE 1=1
            """
            params = []

            # 필터 적용
            if status:
                # 한글 상태를 영문으로 변환
                db_status = map_status_to_enum(status)
                query += " AND c.status = %s"
                params.append(db_status)

            if category:
                query += " AND (c.category_main = %s OR c.category_sub = %s)"
                params.append(category)
                params.append(category)

            if agent_id:
                query += " AND c.agent_id = %s"
                params.append(agent_id)

            if date_from:
                query += " AND c.call_date >= %s"
                params.append(date_from)

            if date_to:
                query += " AND c.call_date <= %s"
                params.append(date_to)

            # 전체 개수
            count_query = f"SELECT COUNT(*) FROM ({query}) as subq"
            cur.execute(count_query, params)
            total = cur.fetchone()['count']

            # 정렬 및 페이지네이션
            query += " ORDER BY c.call_date DESC, c.call_time DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cur.execute(query, params)
            rows = cur.fetchall()

            # 응답 데이터 변환
            consultations = []
            for row in rows:
                # 카테고리 조합
                cat_main = row.get('category_main') or '기타'
                cat_sub = row.get('category_sub') or ''
                category_str = f"{cat_main} > {cat_sub}" if cat_sub and cat_sub != cat_main else cat_main

                # 날짜/시간 조합
                call_date = row.get('call_date')
                call_time = row.get('call_time')
                if call_date and call_time:
                    datetime_str = f"{call_date} {str(call_time)[:5]}"
                elif call_date:
                    datetime_str = str(call_date)
                else:
                    datetime_str = ''

                # 상태 한글 변환
                status_kr = STATUS_TO_KOREAN.get(row.get('status'), '완료')

                # 내용 (AI 요약 또는 메모)
                content = row.get('ai_summary') or row.get('agent_notes') or row.get('title') or ''

                consultations.append(ConsultationListItem(
                    id=row.get('id', ''),
                    agent=row.get('agent_name', '알 수 없음'),
                    agentId=row.get('agent_id'),
                    customer=row.get('customer_name', '알 수 없음'),
                    customerId=row.get('customer_id'),
                    category=category_str,
                    categoryMain=cat_main,
                    categorySub=cat_sub,
                    status=status_kr,
                    content=content[:200] if content else '',  # 200자 제한
                    datetime=datetime_str,
                    duration=row.get('call_duration') or '0:00',
                    isBestPractice=row.get('is_best_practice') or False,
                    isSimulation=row.get('is_simulation') or False,
                    fcr=row.get('fcr') if row.get('fcr') is not None else True,
                    memo=row.get('agent_notes'),
                    team=get_team_by_agent(row.get('agent_id') or ''),
                ))

            return ConsultationListResponse(
                success=True,
                data=consultations,
                total=total,
                message=f"상담 목록 {len(consultations)}건 조회 완료"
            )

    except Exception as e:
        print(f"[ERROR] 상담 목록 조회 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"상담 목록 조회 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.post("", response_model=ApiResponse)
async def save_consultation(request: SaveConsultationRequest):
    """
    상담 데이터 저장 (후처리 완료 시 호출)

    기존 상담 데이터가 있으면 UPDATE, 없으면 INSERT합니다.

    Args:
        request: 상담 저장 요청 데이터

    Returns:
        ApiResponse: 저장 결과
    """
    # ⭐ [v24] 디버그 로깅
    print(f"[DEBUG] 상담 저장 요청 수신:")
    print(f"  - consultationId: {request.consultationId}")
    print(f"  - employeeId: {request.employeeId}")
    print(f"  - customerId: {request.customerId}")
    print(f"  - category: {request.category}")
    print(f"  - referencedDocuments: {len(request.referencedDocuments)}개")
    print(f"  - processingTimeline: {len(request.processingTimeline)}개")

    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # datetime 파싱
            call_date, call_time = parse_datetime(request.datetime)

            # 카테고리 파싱
            category_main, category_sub = parse_category(request.category)

            # 통화 시간 변환
            call_duration = seconds_to_duration(request.callTimeSeconds)
            acw_duration = seconds_to_duration(request.acwTimeSeconds)

            # 참조 문서 JSON
            referenced_docs = json.dumps([doc.model_dump() for doc in request.referencedDocuments])

            # 처리 타임라인 JSON
            processing_timeline_json = json.dumps([item.model_dump() for item in request.processingTimeline]) if request.processingTimeline else None

            # ⭐ [v24] transcript 처리: 문자열이면 JSON 배열로 변환
            transcript_json = None
            if request.transcript:
                # 이미 JSON 배열 형식인지 확인
                if request.transcript.strip().startswith('['):
                    try:
                        # 유효한 JSON인지 확인
                        json.loads(request.transcript)
                        transcript_json = request.transcript
                    except json.JSONDecodeError:
                        # JSON 파싱 실패 시 문자열로 래핑
                        transcript_json = json.dumps([{"speaker": "all", "text": request.transcript}])
                else:
                    # 문자열을 JSON 배열로 변환
                    transcript_json = json.dumps([{"speaker": "all", "text": request.transcript}])

            # 상태 매핑 (한글 → 영문 enum)
            db_status = map_status_to_enum(request.status)

            # 기존 상담 확인
            cur.execute("SELECT id FROM consultations WHERE id = %s", (request.consultationId,))
            existing = cur.fetchone()

            if existing:
                # UPDATE
                cur.execute("""
                    UPDATE consultations SET
                        customer_id = %s,
                        agent_id = %s,
                        status = %s,
                        category_main = %s,
                        category_sub = %s,
                        title = %s,
                        call_date = %s,
                        call_time = %s,
                        call_duration = %s,
                        acw_duration = %s,
                        ai_summary = %s,
                        agent_notes = %s,
                        transcript = %s,
                        follow_up_schedule = %s,
                        transfer_department = %s,
                        transfer_notes = %s,
                        referenced_documents = %s,
                        processing_timeline = %s,
                        sentiment = %s,
                        emotion_score = %s,
                        satisfaction_score = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id
                """, (
                    request.customerId,
                    request.employeeId,
                    db_status,
                    category_main,
                    category_sub,
                    request.title or f"{category_main} 상담",
                    call_date,
                    call_time,
                    call_duration,
                    acw_duration,
                    request.aiSummary,
                    request.memo,
                    transcript_json,  # ⭐ [v24] JSON 변환된 transcript 사용
                    request.followUpTasks,
                    request.handoffDepartment,
                    request.handoffNotes,
                    referenced_docs,
                    processing_timeline_json,
                    request.sentiment,
                    request.feedbackScore,
                    request.satisfactionScore,
                    request.consultationId
                ))
                action = "updated"
            else:
                # INSERT
                cur.execute("""
                    INSERT INTO consultations (
                        id, customer_id, agent_id, status,
                        category_main, category_sub, title,
                        call_date, call_time, call_duration, acw_duration,
                        ai_summary, agent_notes, transcript,
                        follow_up_schedule, transfer_department, transfer_notes,
                        referenced_documents, processing_timeline, sentiment, emotion_score, satisfaction_score,
                        created_at
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        NOW()
                    )
                    RETURNING id
                """, (
                    request.consultationId,
                    request.customerId,
                    request.employeeId,
                    db_status,
                    category_main,
                    category_sub,
                    request.title or f"{category_main} 상담",
                    call_date,
                    call_time,
                    call_duration,
                    acw_duration,
                    request.aiSummary,
                    request.memo,
                    transcript_json,  # ⭐ [v24] JSON 변환된 transcript 사용
                    request.followUpTasks,
                    request.handoffDepartment,
                    request.handoffNotes,
                    referenced_docs,
                    processing_timeline_json,
                    request.sentiment,
                    request.feedbackScore,
                    request.satisfactionScore
                ))
                action = "created"

            result = cur.fetchone()
            conn.commit()

            return ApiResponse(
                success=True,
                data={"consultationId": result['id'], "action": action},
                message=f"상담 저장 완료 ({action})"
            )

    except Exception as e:
        if conn:
            conn.rollback()
        import traceback
        print(f"[ERROR] 상담 저장 실패: {str(e)}")
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"상담 저장 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.get("/{consultation_id}", response_model=ApiResponse)
async def get_consultation(consultation_id: str):
    """
    특정 상담 조회

    Args:
        consultation_id: 상담 ID

    Returns:
        ApiResponse: 상담 상세 정보
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    c.*,
                    cust.name as customer_name,
                    cust.phone as customer_phone,
                    cust.birth_date as customer_birth_date,
                    cust.address as customer_address,
                    cust.current_type_code as customer_type,
                    emp.name as agent_name,
                    emp.department as agent_team
                FROM consultations c
                LEFT JOIN customers cust ON c.customer_id = cust.id
                LEFT JOIN employees emp ON c.agent_id = emp.id
                WHERE c.id = %s
            """, (consultation_id,))

            row = cur.fetchone()

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"상담을 찾을 수 없습니다: {consultation_id}"
                )

            # 날짜/시간 변환
            data = dict(row)
            if data.get('call_date'):
                data['call_date'] = str(data['call_date'])
            if data.get('call_time'):
                data['call_time'] = str(data['call_time'])
            if data.get('call_end_time'):
                data['call_end_time'] = str(data['call_end_time'])
            if data.get('created_at'):
                data['created_at'] = str(data['created_at'])
            if data.get('updated_at'):
                data['updated_at'] = str(data['updated_at'])
            if data.get('customer_birth_date'):
                data['customer_birth_date'] = str(data['customer_birth_date'])

            return ApiResponse(
                success=True,
                data=data,
                message="상담 조회 성공"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"상담 조회 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()
