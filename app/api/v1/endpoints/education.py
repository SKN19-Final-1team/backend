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
from app.llm.education.tts_speaker import (
    initialize_conversation,
    process_agent_input,
    get_conversation_history,
    end_conversation,
    get_session_info
)


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


class ConversationMessageRequest(BaseModel):
    """대화 메시지 요청"""
    message: str
    mode: str = "text"  # "text" 또는 "voice"


class ConversationMessageResponse(BaseModel):
    """대화 메시지 응답"""
    customer_response: str
    turn_number: int
    audio_url: Optional[str] = None


class ConversationHistoryResponse(BaseModel):
    """대화 히스토리 응답"""
    session_id: str
    customer_name: str
    conversation_history: List[Dict[str, str]]
    turn_count: int


class ConversationEndResponse(BaseModel):
    """대화 종료 응답"""
    session_id: str
    customer_name: str
    turn_count: int
    duration_seconds: float
    conversation_history: List[Dict[str, str]]


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
            
            # 대화 세션 초기화
            initialize_conversation(session_id, system_prompt, customer_profile)
            
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


@router.post("/simulation/{session_id}/message", response_model=ConversationMessageResponse)
async def send_message(session_id: str, request: ConversationMessageRequest):
    """
    대화 메시지 전송 및 AI 고객 응답 받기
    
    Args:
        session_id: 세션 ID
        request: 메시지 요청 (상담원 메시지)
        
    Returns:
        AI 고객 응답
    """
    try:
        response = process_agent_input(
            session_id=session_id,
            agent_message=request.message,
            input_mode=request.mode
        )
        
        return ConversationMessageResponse(**response)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메시지 처리 실패: {str(e)}")


@router.get("/simulation/{session_id}/history", response_model=ConversationHistoryResponse)
async def get_history(session_id: str):
    """
    대화 히스토리 조회
    
    Args:
        session_id: 세션 ID
        
    Returns:
        대화 히스토리
    """
    try:
        session_info = get_session_info(session_id)
        
        if not session_info:
            raise HTTPException(status_code=404, detail=f"세션을 찾을 수 없습니다: {session_id}")
        
        return ConversationHistoryResponse(
            session_id=session_info["session_id"],
            customer_name=session_info["customer_name"],
            conversation_history=session_info["conversation_history"],
            turn_count=session_info["turn_count"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"히스토리 조회 실패: {str(e)}")


@router.post("/simulation/{session_id}/end", response_model=ConversationEndResponse)
async def end_simulation(session_id: str):
    """
    시뮬레이션 종료
    
    Args:
        session_id: 세션 ID
        
    Returns:
        대화 요약 정보
    """
    try:
        summary = end_conversation(session_id)
        
        return ConversationEndResponse(**summary)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세션 종료 실패: {str(e)}")


@router.get("/simulation/{session_id}/status")
async def get_simulation_status(session_id: str):
    """
    시뮬레이션 상태 조회
    
    Args:
        session_id: 세션 ID
        
    Returns:
        세션 상태 정보
    """
    session_info = get_session_info(session_id)
    
    if not session_info:
        raise HTTPException(status_code=404, detail=f"세션을 찾을 수 없습니다: {session_id}")
    
    return {
        "session_id": session_info["session_id"],
        "customer_name": session_info["customer_name"],
        "turn_count": session_info["turn_count"],
        "status": "active"
    }
