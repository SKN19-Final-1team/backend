"""
교육 시뮬레이션 엔드포인트
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import random
import json

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
from app.llm.education.evaluation import evaluate_simulation_result
from app.llm.education.result_manager import (
    save_simulation_result,
    get_employee_analytics,
    get_simulation_history
)


# 시뮬레이션 메타데이터 저장 (세션별)
_simulation_metadata: Dict[str, Dict[str, Any]] = {}


# 카테고리 정규화 매핑 (프론트엔드 → DB category_main)
CATEGORY_MAPPING = {
    '카드분실': '분실/도난',
    '분실/도난': '분실/도난',
    '해외결제': '결제/승인',
    '결제/승인': '결제/승인',
    '수수료문의': '수수료/연체',
    '수수료/연체': '수수료/연체',
    '포인트/혜택': '포인트/혜택',
    '한도증액': '한도',
    '한도': '한도',
    '이용내역': '이용내역',
    '정부지원': '정부지원',
    '기타': '기타',
    '기타문의': '기타',
}


def normalize_category(category: str) -> str:
    """프론트엔드 카테고리를 DB category_main으로 변환"""
    return CATEGORY_MAPPING.get(category, category)


router = APIRouter()


class ScenarioListResponse(BaseModel):
    """시나리오 목록 응답"""
    scenarios: List[Dict[str, Any]]


class SimulationStartRequest(BaseModel):
    """시뮬레이션 시작 요청"""
    category: str  # 문의 유형 (예: "분실/도난")
    difficulty: str  # "beginner" 또는 "advanced"
    employee_id: Optional[str] = None  # 직원 ID (결과 저장용)
    scenario_id: Optional[str] = None  # 기본 시나리오 ID (초급용)
    consultation_id: Optional[str] = None  # 우수사례 상담 ID (상급용)


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
    evaluation: Optional[Dict[str, Any]] = None
    result_id: Optional[str] = None


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

    # 카테고리 정규화
    normalized_category = normalize_category(request.category)

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            consultation_row = None

            # 난이도별 분기
            if request.difficulty == "advanced" and request.consultation_id:
                # 상급 (우수사례): 특정 consultation_id로 조회
                cur.execute("""
                    SELECT c.id, c.customer_id
                    FROM consultations c
                    WHERE c.id = %s
                """, (request.consultation_id,))
                consultation_row = cur.fetchone()

                if not consultation_row:
                    raise HTTPException(
                        status_code=404,
                        detail=f"상담 ID '{request.consultation_id}'를 찾을 수 없습니다."
                    )
            else:
                # 초급 (기본 시나리오): 카테고리로 랜덤 선택
                cur.execute("""
                    SELECT c.id, c.customer_id
                    FROM consultations c
                    WHERE c.category_main = %s
                    ORDER BY RANDOM()
                    LIMIT 1
                """, (normalized_category,))
                consultation_row = cur.fetchone()

            if not consultation_row:
                raise HTTPException(
                    status_code=404,
                    detail=f"카테고리 '{normalized_category}'에 해당하는 상담 사례가 없습니다."
                )

            consultation_id = consultation_row["id"]
            customer_id = consultation_row["customer_id"]

            # 2. consultation_documents에서 상담 문서 조회
            consultation_content = None
            consultation_title = None
            consultation_keywords = None

            cur.execute("""
                SELECT title, keywords, content
                FROM consultation_documents
                WHERE consultation_id = %s
                LIMIT 1
            """, (consultation_id,))

            doc_row = cur.fetchone()
            if doc_row:
                consultation_title = doc_row.get("title")
                consultation_keywords = doc_row.get("keywords")
                consultation_content = doc_row.get("content")

            # 3. customers 테이블에서 고객 정보 조회
            cur.execute("""
                SELECT name, gender, age_group, grade, card_type,
                       current_type_code
                FROM customers
                WHERE id = %s
            """, (customer_id,))

            customer_row = cur.fetchone()

            # 4. persona_types 테이블에서 페르소나 정보 조회
            persona_info = None
            if customer_row and customer_row.get("current_type_code"):
                type_codes = customer_row["current_type_code"]
                if type_codes and len(type_codes) > 0:
                    cur.execute("""
                        SELECT name, description, communication_style, personality_tags
                        FROM persona_types
                        WHERE code = %s
                    """, (type_codes[0],))
                    persona_info = cur.fetchone()

            # 고객 프로필 생성 (DB 정보 + 분석 결과)
            if customer_row:
                # DB에서 가져온 실제 고객 정보 사용
                personality_tags = customer_row.get("personality_tags") or ["normal", "polite"]
                communication_style = {
                    "tone": "neutral",
                    "speed": "moderate"
                }
                llm_guidance = "일반적인 응대로 친절하게 안내해주세요."

                # persona_types 정보가 있으면 반영
                if persona_info:
                    if persona_info.get("communication_style"):
                        comm_style = persona_info["communication_style"]
                        if isinstance(comm_style, dict):
                            communication_style = comm_style
                    llm_guidance = persona_info.get("description") or llm_guidance

                customer_profile = {
                    "name": customer_row.get("name") or f"고객_{random.randint(1000, 9999)}",
                    "gender": customer_row.get("gender"),
                    "age_group": customer_row.get("age_group") or "40대",
                    "grade": customer_row.get("grade"),
                    "card_type": customer_row.get("card_type"),
                    "personality_tags": personality_tags,
                    "communication_style": communication_style,
                    "llm_guidance": llm_guidance,
                    "persona_name": persona_info.get("name") if persona_info else None,
                    "persona_description": persona_info.get("description") if persona_info else None
                }
            else:
                # 고객 정보가 없는 경우 상담 내용 분석으로 대체
                analysis = analyze_consultation(consultation_content or "")
                customer_profile = {
                    "name": f"고객_{random.randint(1000, 9999)}",
                    "age_group": analysis.get("age_group_inferred", "40대"),
                    "personality_tags": analysis["personality_tags"],
                    "communication_style": analysis["communication_style"],
                    "llm_guidance": analysis["llm_guidance"]
                }

            # 시스템 프롬프트 생성
            system_prompt = create_system_prompt(customer_profile, difficulty=request.difficulty)

            # 상급 난이도의 경우 각본 생성 (상담 전문 필요)
            scenario_script = None
            if request.difficulty == "advanced" and consultation_content:
                scenario_script = create_scenario_script(
                    consultation_content,
                    difficulty="advanced"
                )
                # 우수사례(상급) 시뮬레이션에서만 시나리오 생성 성공 메세지 표시
                print("\n" + "="*60)
                print("[Education] ✅ 시나리오 생성 성공 (우수사례 시뮬레이션)")
                print("="*60)

            # === 터미널에 JSON 출력 ===
            print("\n" + "="*60)
            print(f"[Education] 시뮬레이션 시작 - 난이도: {request.difficulty}")
            print("="*60)

            # 1. DB에서 가져온 데이터 출력
            db_data = {
                "consultation": {
                    "id": consultation_id,
                    "customer_id": customer_id,
                    "category": normalized_category
                },
                "consultation_document": {
                    "title": consultation_title,
                    "keywords": consultation_keywords,
                    "content_preview": consultation_content[:200] + "..." if consultation_content and len(consultation_content) > 200 else consultation_content
                } if doc_row else None,
                "customer": dict(customer_row) if customer_row else None,
                "persona_type": dict(persona_info) if persona_info else None
            }
            print("\n[Education] 📊 DB 데이터:")
            print(json.dumps(db_data, ensure_ascii=False, indent=2, default=str))

            # 2. 시나리오 출력 (우수사례의 경우만)
            if request.difficulty == "advanced" and scenario_script:
                print("\n[Education] 📋 시나리오 (우수사례):")
                print(json.dumps(scenario_script, ensure_ascii=False, indent=2))

            # 3. 고객 페르소나 출력
            print("\n[Education] 👤 고객 페르소나:")
            print(json.dumps(customer_profile, ensure_ascii=False, indent=2, default=str))
            print("="*60 + "\n")

            # 세션 ID 생성
            session_id = f"sim_{consultation_id}_{random.randint(10000, 99999)}"

            # 대화 세션 초기화
            initialize_conversation(session_id, system_prompt, customer_profile)

            # 시뮬레이션 메타데이터 저장 (종료 시 평가에 사용)
            _simulation_metadata[session_id] = {
                "employee_id": request.employee_id,
                "category": normalized_category,
                "difficulty": request.difficulty,
                "consultation_id": consultation_id,
                "consultation_content": consultation_content,
                "consultation_title": consultation_title,
                "consultation_keywords": consultation_keywords,
                "scenario_script": scenario_script,
                "simulation_type": "best_practice" if request.difficulty == "advanced" else "scenario",
                "scenario_id": request.scenario_id,
            }

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
    시뮬레이션 종료 및 평가

    Args:
        session_id: 세션 ID

    Returns:
        대화 요약 정보 및 평가 결과
    """
    try:
        summary = end_conversation(session_id)

        # 메타데이터 조회
        metadata = _simulation_metadata.pop(session_id, None)

        evaluation = None
        result_id = None

        if metadata:
            # 대화 전문 생성
            transcript = "\n".join([
                f"{'상담원' if msg['role'] == 'agent' else '고객'}: {msg['content']}"
                for msg in summary.get("conversation_history", [])
            ])

            # 시나리오 각본 (상급 난이도일 때만)
            scenario_script = metadata.get("scenario_script") or {
                "expected_flow": ["상담 시작", "문의 파악", "해결책 제시", "상담 종료"],
                "key_points": ["고객 니즈 파악", "정확한 정보 제공"],
                "evaluation_criteria": {}
            }

            # 평가 수행
            evaluation = evaluate_simulation_result(
                simulation_transcript=transcript,
                original_transcript=metadata.get("consultation_content"),
                scenario_script=scenario_script,
                simulation_type=metadata.get("simulation_type", "scenario")
            )

            # 결과 저장 (employee_id가 있는 경우만)
            employee_id = metadata.get("employee_id")
            if employee_id:
                try:
                    result_id = save_simulation_result(
                        employee_id=employee_id,
                        simulation_type=metadata.get("simulation_type", "scenario"),
                        original_consultation_id=metadata.get("consultation_id"),
                        scenario_id=None,
                        scores=evaluation,
                        feedback_data=evaluation.get("feedback", {}),
                        call_duration=int(summary.get("duration_seconds", 0)),
                        recording_transcript=transcript
                    )
                except Exception as save_error:
                    print(f"[Education] 결과 저장 실패: {save_error}")

        return ConversationEndResponse(
            session_id=summary["session_id"],
            customer_name=summary["customer_name"],
            turn_count=summary["turn_count"],
            duration_seconds=summary["duration_seconds"],
            conversation_history=summary["conversation_history"],
            evaluation=evaluation,
            result_id=result_id
        )

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


@router.get("/analytics/{employee_id}")
async def get_analytics(employee_id: str):
    """
    직원 학습 분석 데이터 조회

    Args:
        employee_id: 직원 ID

    Returns:
        학습 분석 데이터
    """
    analytics = get_employee_analytics(employee_id)

    if not analytics:
        return {
            "employee_id": employee_id,
            "total_simulations": 0,
            "average_score": 0,
            "pass_rate": 0,
            "message": "아직 시뮬레이션 기록이 없습니다."
        }

    return analytics


@router.get("/history/{employee_id}")
async def get_history_endpoint(employee_id: str, limit: int = 10, offset: int = 0):
    """
    직원 시뮬레이션 히스토리 조회

    Args:
        employee_id: 직원 ID
        limit: 조회 개수 (기본: 10)
        offset: 시작 위치 (기본: 0)

    Returns:
        시뮬레이션 결과 리스트
    """
    history = get_simulation_history(employee_id, limit, offset)

    return {
        "employee_id": employee_id,
        "results": history,
        "count": len(history)
    }
