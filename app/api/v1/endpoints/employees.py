"""
Employees API 엔드포인트

상담사 정보 및 성과 조회
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import date
import psycopg2.extras

from app.db.base import get_connection

router = APIRouter()


# ==============================================================================
# Response Models
# ==============================================================================

class EmployeeResponse(BaseModel):
    """상담사 응답 모델"""
    id: str
    name: str
    email: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    hireDate: Optional[str] = None
    status: Optional[str] = None
    consultations: int = 0
    fcr: int = 0
    avgTime: str = "0:00"
    rank: int = 0
    # Frontend 호환 필드
    team: Optional[str] = None      # = department
    position: Optional[str] = None  # = role
    joinDate: Optional[str] = None  # = hireDate
    phone: Optional[str] = None     # 선택적 (DB에 없음)
    trend: Optional[str] = "same"   # 선택적 (DB에 없음)


class TopEmployeeResponse(BaseModel):
    """우수 상담사 응답 모델 (대시보드용)"""
    id: str
    name: str
    title: str
    team: str
    rank: int


class EmployeeListResponse(BaseModel):
    """상담사 목록 응답"""
    success: bool
    data: List[EmployeeResponse]
    total: int
    message: str


class TopEmployeesResponse(BaseModel):
    """우수 상담사 목록 응답"""
    success: bool
    data: List[TopEmployeeResponse]
    message: str


# ==============================================================================
# Helper Functions
# ==============================================================================

def row_to_employee_response(row: dict) -> EmployeeResponse:
    """DB 행을 EmployeeResponse로 변환"""
    hire_date = row.get('hire_date')
    if isinstance(hire_date, date):
        hire_date_str = hire_date.strftime('%Y-%m-%d')
    else:
        hire_date_str = str(hire_date) if hire_date else None

    department = row.get('department')
    role = row.get('role')

    return EmployeeResponse(
        id=row.get('id', ''),
        name=row.get('name', ''),
        email=row.get('email'),
        role=role,
        department=department,
        hireDate=hire_date_str,
        status=row.get('status'),
        consultations=row.get('consultations', 0) or 0,
        fcr=row.get('fcr', 0) or 0,
        avgTime=row.get('avgTime', '0:00') or '0:00',
        rank=row.get('rank', 0) or 0,
        # Frontend 호환 필드
        team=department,
        position=role,
        joinDate=hire_date_str,
        phone=row.get('phone'),  # DB에서 로드
        trend=row.get('trend', 'same') or 'same'  # DB에서 로드
    )


# ==============================================================================
# API Endpoints
# ==============================================================================

@router.get("", response_model=EmployeeListResponse)
async def get_employees(
    limit: int = Query(default=50, ge=1, le=200, description="조회 개수"),
    offset: int = Query(default=0, ge=0, description="오프셋"),
    department: Optional[str] = Query(default=None, description="부서 필터"),
    status: Optional[str] = Query(default=None, description="상태 필터 (active/inactive/vacation, 미지정시 전체)")
):
    """
    상담사 목록 조회

    Args:
        limit: 조회 개수 (기본 50, 최대 200)
        offset: 오프셋 (페이지네이션용)
        department: 부서 필터 (예: '상담1팀')
        status: 상태 필터 (active/inactive/vacation, 미지정시 전체 조회)

    Returns:
        EmployeeListResponse: 상담사 목록
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 전체 개수 조회
            count_query = "SELECT COUNT(*) FROM employees WHERE 1=1"
            params = []

            if status:
                count_query += " AND status = %s"
                params.append(status)

            if department:
                count_query += " AND department = %s"
                params.append(department)

            cur.execute(count_query, params)
            total = cur.fetchone()['count']

            # 상담사 목록 조회
            query = """
                SELECT
                    id, name, email, phone, role, department, hire_date, status,
                    consultations, fcr, "avgTime", rank, trend
                FROM employees
                WHERE 1=1
            """
            query_params = []

            if status:
                query += " AND status = %s"
                query_params.append(status)

            if department:
                query += " AND department = %s"
                query_params.append(department)

            query += " ORDER BY rank ASC, consultations DESC LIMIT %s OFFSET %s"
            query_params.extend([limit, offset])

            cur.execute(query, query_params)
            rows = cur.fetchall()

            employees = [row_to_employee_response(row) for row in rows]

            return EmployeeListResponse(
                success=True,
                data=employees,
                total=total,
                message=f"상담사 {len(employees)}명 조회 완료"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"상담사 조회 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.get("/top", response_model=TopEmployeesResponse)
async def get_top_employees(
    limit: int = Query(default=3, ge=1, le=10, description="조회 개수")
):
    """
    우수 상담사 조회 (대시보드용)

    rank가 1, 2, 3인 상담사를 조회하여 대시보드에 표시합니다.

    Args:
        limit: 조회 개수 (기본 3)

    Returns:
        TopEmployeesResponse: 우수 상담사 목록
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    id, name, department, rank, consultations, fcr, "avgTime"
                FROM employees
                WHERE status = 'active' AND rank > 0
                ORDER BY rank ASC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()

            # 순위별 타이틀 생성
            top_employees = []
            for i, row in enumerate(rows):
                titles = [
                    f"FCR {row.get('fcr', 0)}% 달성",
                    f"평균 {row.get('avgTime', '0:00')} 처리 시간",
                    f"월간 {row.get('consultations', 0)}건 상담"
                ]
                top_employees.append(TopEmployeeResponse(
                    id=row.get('id', ''),
                    name=row.get('name', ''),
                    title=titles[i] if i < len(titles) else f"FCR {row.get('fcr', 0)}% 달성",
                    team=row.get('department', ''),
                    rank=row.get('rank', 0)
                ))

            return TopEmployeesResponse(
                success=True,
                data=top_employees,
                message=f"우수 상담사 {len(top_employees)}명 조회 완료"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"우수 상담사 조회 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.get("/{employee_id}", response_model=dict)
async def get_employee_by_id(employee_id: str):
    """
    특정 상담사 조회

    Args:
        employee_id: 상담사 ID (예: EMP-001)

    Returns:
        상담사 상세 정보
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    id, name, email, phone, role, department, hire_date, status,
                    consultations, fcr, "avgTime", rank, trend
                FROM employees
                WHERE id = %s
            """, (employee_id,))

            row = cur.fetchone()

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"상담사를 찾을 수 없습니다: {employee_id}"
                )

            employee = row_to_employee_response(row)

            return {
                "success": True,
                "data": employee.model_dump(),
                "message": "상담사 조회 성공"
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"상담사 조회 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


# ==============================================================================
# CRUD Endpoints (Create, Update, Delete)
# ==============================================================================

class CreateEmployeeRequest(BaseModel):
    """사원 생성 요청 모델"""
    id: Optional[str] = None  # 자동 생성 가능
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    hireDate: Optional[str] = None
    status: Optional[str] = "active"


class UpdateEmployeeRequest(BaseModel):
    """사원 수정 요청 모델"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    hireDate: Optional[str] = None
    status: Optional[str] = None


def generate_employee_id(conn) -> str:
    """새 사원 ID 생성 (EMP-XXX 형식)"""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM employees WHERE id LIKE 'EMP-%' ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    cursor.close()

    if row:
        last_num = int(row[0].replace('EMP-', ''))
        return f"EMP-{last_num + 1:03d}"
    return "EMP-001"


@router.post("", response_model=dict)
async def create_employee(request: CreateEmployeeRequest):
    """
    사원 생성

    Args:
        request: 사원 생성 요청 데이터

    Returns:
        생성된 사원 정보
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # ID 생성 또는 사용
            employee_id = request.id if request.id else generate_employee_id(conn)

            # 중복 ID 확인
            cur.execute("SELECT id FROM employees WHERE id = %s", (employee_id,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail=f"이미 존재하는 사원 ID입니다: {employee_id}")

            # hire_date 파싱
            hire_date = None
            if request.hireDate:
                try:
                    hire_date = request.hireDate
                except:
                    pass

            # INSERT
            cur.execute("""
                INSERT INTO employees (id, name, email, phone, role, department, hire_date, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING *
            """, (
                employee_id,
                request.name,
                request.email,
                request.phone,
                request.role,
                request.department,
                hire_date,
                request.status or 'active'
            ))

            row = cur.fetchone()
            conn.commit()

            return {
                "success": True,
                "data": row_to_employee_response(row).model_dump(),
                "message": f"사원 생성 완료: {employee_id}"
            }

    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"사원 생성 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.put("/{employee_id}", response_model=dict)
async def update_employee(employee_id: str, request: UpdateEmployeeRequest):
    """
    사원 정보 수정

    Args:
        employee_id: 사원 ID
        request: 수정할 데이터

    Returns:
        수정된 사원 정보
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 존재 확인
            cur.execute("SELECT id FROM employees WHERE id = %s", (employee_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail=f"사원을 찾을 수 없습니다: {employee_id}")

            # 동적 UPDATE 쿼리 생성
            update_fields = []
            params = []

            if request.name is not None:
                update_fields.append("name = %s")
                params.append(request.name)
            if request.email is not None:
                update_fields.append("email = %s")
                params.append(request.email)
            if request.phone is not None:
                update_fields.append("phone = %s")
                params.append(request.phone)
            if request.role is not None:
                update_fields.append("role = %s")
                params.append(request.role)
            if request.department is not None:
                update_fields.append("department = %s")
                params.append(request.department)
            if request.hireDate is not None:
                update_fields.append("hire_date = %s")
                params.append(request.hireDate)
            if request.status is not None:
                update_fields.append("status = %s")
                params.append(request.status)

            if not update_fields:
                raise HTTPException(status_code=400, detail="수정할 필드가 없습니다")

            update_fields.append("updated_at = NOW()")
            params.append(employee_id)

            query = f"UPDATE employees SET {', '.join(update_fields)} WHERE id = %s RETURNING *"
            cur.execute(query, params)

            row = cur.fetchone()
            conn.commit()

            return {
                "success": True,
                "data": row_to_employee_response(row).model_dump(),
                "message": f"사원 정보 수정 완료: {employee_id}"
            }

    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"사원 수정 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@router.delete("/{employee_id}", response_model=dict)
async def delete_employee(employee_id: str):
    """
    사원 삭제 (soft delete - status를 inactive로 변경)

    Args:
        employee_id: 사원 ID

    Returns:
        삭제 결과
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # 존재 확인
            cur.execute("SELECT id FROM employees WHERE id = %s", (employee_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail=f"사원을 찾을 수 없습니다: {employee_id}")

            # Soft delete (status를 inactive로 변경)
            cur.execute("""
                UPDATE employees
                SET status = 'inactive', updated_at = NOW()
                WHERE id = %s
            """, (employee_id,))

            conn.commit()

            return {
                "success": True,
                "data": {"id": employee_id},
                "message": f"사원 삭제 완료 (비활성화): {employee_id}"
            }

    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"사원 삭제 중 오류 발생: {str(e)}"
        )
    finally:
        if conn:
            conn.close()
