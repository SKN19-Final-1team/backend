"""
교육 시뮬레이션 엔드포인트
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import random

from app.db.base import get_connection
import psycopg2.extras

from app.llm.education.feature_analyzer import analyze_consultation, format_analysis_for_db
from app.llm.education.persona_generator import create_system_prompt, create_scenario_script


router = APIRouter()


class ScenarioListResponse(BaseModel):
    """시나리오 목록 응답"""
    scenarios: List[Dict[str, Any]]


class SimulationStartRequest(BaseModel):
    """시뮬레이션 시작 요청"""
    category: str  # 문의 유형 (예: "도난/분실 신청/해제")
    difficulty: str  # "beginner" 또는 "advanced"


class SimulationStartResponse(BaseModel):
    """시뮬레이션 시작 응답"""
    session_id: str
    system_prompt: str
    customer_name: str
    customer_profile: Dict[str, Any]
    scenario_script: Optional[Dict[str, Any]] = None


@router.get("/scenarios", response_model=ScenarioListResponse)
async def get_scenarios():
    """
    사용 가능한 시나리오 목록 조회
    
    Returns:
        시나리오 목록 (카테고리별, 난이도별)
    """
    conn = get_connection()
    
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # consultation_documents 테이블에서 고유한 카테고리 목록 조회
            cur.execute("""
                SELECT DISTINCT category, COUNT(*) as count
                FROM consultation_documents
                GROUP BY category
                ORDER BY category
            """)
            
            categories = cur.fetchall()
            
            scenarios = []
            for cat in categories:
                scenarios.append({
                    "category": cat["category"],
                    "difficulty": "beginner",
                    "count": cat["count"],
                    "description": f"{cat['category']} 상담 (초급)"
                })
                scenarios.append({
                    "category": cat["category"],
                    "difficulty": "advanced",
                    "count": cat["count"],
                    "description": f"{cat['category']} 상담 (상급)"
                })
            
            return ScenarioListResponse(scenarios=scenarios)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시나리오 조회 실패: {str(e)}")
    finally:
        conn.close()


@router.post("/simulation/start", response_model=SimulationStartResponse)
async def start_simulation(request: SimulationStartRequest):
    """
    교육 시뮬레이션 시작
    
    Args:
        request: 시뮬레이션 시작 요청 (카테고리, 난이도)
        
    Returns:
        시뮬레이션 세션 정보 (페르소나 프롬프트, 고객 프로필 등)
    """
    conn = get_connection()
    
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 해당 카테고리의 상담 사례 중 1개를 랜덤으로 선택
            cur.execute("""
                SELECT * FROM consultation_documents
                WHERE category = %s
                ORDER BY RANDOM()
                LIMIT 1
            """, (request.category,))
            
            consultation = cur.fetchone()
            
            if not consultation:
                raise HTTPException(
                    status_code=404,
                    detail=f"카테고리 '{request.category}'에 해당하는 상담 사례가 없습니다."
                )
            
            # 상담 내용 분석
            analysis = analyze_consultation(consultation["content"])
            
            # 고객 프로필 생성
            customer_profile = {
                "name": f"고객_{random.randint(1000, 9999)}",
                "age_group": analysis.get("age_group_inferred", "40대"),
                "personality_tags": analysis["personality_tags"],
                "communication_style": analysis["communication_style"],
                "llm_guidance": analysis["llm_guidance"]
            }
            
            # 시스템 프롬프트 생성
            system_prompt = create_system_prompt(customer_profile, difficulty=request.difficulty)
            
            # 상급 난이도의 경우 각본 생성
            scenario_script = None
            if request.difficulty == "advanced":
                scenario_script = create_scenario_script(
                    consultation["content"],
                    difficulty="advanced"
                )
            
            # 세션 ID 생성
            session_id = f"sim_{consultation['id']}_{random.randint(10000, 99999)}"
            
            return SimulationStartResponse(
                session_id=session_id,
                system_prompt=system_prompt,
                customer_name=customer_profile["name"],
                customer_profile=customer_profile,
                scenario_script=scenario_script
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시뮬레이션 시작 실패: {str(e)}")
    finally:
        conn.close()


@router.get("/simulation/{session_id}/status")
async def get_simulation_status(session_id: str):
    """
    시뮬레이션 상태 조회 (미구현)
    
    향후 구현 예정:
    - 진행 중인 시뮬레이션의 상태 조회
    - 대화 히스토리 조회
    """
    return {"session_id": session_id, "status": "not_implemented"}


@router.post("/simulation/{session_id}/complete")
async def complete_simulation(session_id: str):
    """
    시뮬레이션 완료 및 평가 (미구현)
    
    향후 구현 예정:
    - 상담 내용 평가
    - 피드백 생성
    - 점수 산정
    """
    return {"session_id": session_id, "message": "평가 기능은 향후 구현 예정입니다."}