from app.utils.evaluate_call import evaluate_call
from app.llm.follow_up.feedback_generator import generate_feedback
from app.llm.follow_up.summarize_generator import get_summarize
from app.llm.follow_up.personality_generator import get_personality, determine_personality
from app.db.scripts.modules.connect_db import connect_db
from app.db.scripts.modules.update_customer import get_personality_history, update_customer
from app.utils.get_dialogue import get_dialogue, refine_script
from fastapi import APIRouter, HTTPException
from app.core.prompt import FEEDBACK_SYSTEM_PROMPT, EDU_FEEDBACK_SYSTEM_PROMPT
import time
import asyncio
from pydantic import BaseModel

router = APIRouter()

class SummaryRequest(BaseModel):
    consultation_id: str
    is_simulation: bool

@router.post("/")
async def create_summary(request: SummaryRequest):
    try:
        # redis에서 전문 가져오기
        script, json_script = await get_dialogue(request.consultation_id)
        print(f'전문 : {script}')
        
        start_parallel = time.time()
        
        # 두 함수 동시에 실행
        summarize_task = get_summarize(script)

        if request.is_simulation:
            feedback_task = generate_feedback(script, EDU_FEEDBACK_SYSTEM_PROMPT)
        else:
            feedback_task = generate_feedback(script, FEEDBACK_SYSTEM_PROMPT)
            
        summarize_result, feedback = await asyncio.gather(summarize_task, feedback_task)
        
        end_parallel = time.time()
        parallel_time = end_parallel - start_parallel

        # 오류 체크
        if "error" in summarize_result:
            raise HTTPException(status_code=500, detail=summarize_result["error"])
        if isinstance(feedback, str):
            raise HTTPException(status_code=500, detail=feedback)

        if request.is_simulation:
            # ---상준님 우수 사례랑 비교해서 유사한지 평가하는 함수 추가 ---
            score = ''
            feedback['similarity_score'] = score.get("similarity_score", 0)
        else:
            # 감정 점수 계산
            score = evaluate_call(feedback['emotions'])
            feedback["emotion_score"] = score.get("emotion_score", 0)

        print(f"병렬 처리 시간(요약+피드백): {parallel_time:.2f}초")

        return {
            "isSuccess": True,
            "code": 200,
            "message": "후처리 문서가 생성되었습니다.",
            "summary": summarize_result,
            "evaluation": feedback,
            "script": json_script,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SaveConsultationRequest(BaseModel):
    customer_id: str      # 고객 ID
    consultation_id: str  # 상담 ID
    fcr: int              # 수정할 fcr
    summary: dict         # 수정된 요약본
    evaluation: dict      # 피드백/감정


@router.post("/save")
async def save_consultation(data: SaveConsultationRequest):
    try:
        conn = connect_db()
        
        # redis에서 전문 가져오기
        script, _ = await get_dialogue(data.consultation_id)
        print(f'전문 : {script}')
        
        customer_script = refine_script(script)
        print(f'고객전문 : {customer_script}')
        
        # DB에서 최근 성향 이력 3개 조회
        past_history = get_personality_history(conn, data.customer_id)
        print(past_history)
        
        # 현재 상담에서의 고객 성향 분석
        current_personality = await get_personality(customer_script)
        
        # 최종 성향 확정 (과거 3개 + 현재 1개)
        total_history = (past_history + [current_personality])
        print(total_history)
        current_type_code, type_history = determine_personality(total_history)
        
        # 고객 정보 업데이트
        print(f"최종 성향: {current_type_code}, 최종 히스토리: {type_history}")
        update_customer(conn, data.customer_id, current_type_code, type_history, data.fcr)
        
        # 상담 내역 저장 코드 추가하기
        
        conn.close()

        return {
            "isSuccess": True,
            "code": 200,
            "message": "상담 내역 및 고객 성향이 저장되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"저장 중 오류 발생: {str(e)}")
