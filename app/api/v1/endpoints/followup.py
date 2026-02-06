from app.utils.evaluate_call import evaluate_call
from app.llm.follow_up.feedback_generator import generate_feedback
from app.llm.follow_up.summarize_generator import get_summarize
from app.llm.follow_up.personality_generator import get_personality, determine_personality
from app.llm.education.similarity_calculator import calculate_consultation_similarity
from app.db.scripts.modules.connect_db import connect_db
from app.db.scripts.modules.update_customer import get_personality_history, update_customer, save_consultation_to_db
from app.utils.get_dialogue import get_dialogue, refine_script
from fastapi import APIRouter, HTTPException
from app.core.prompt import FEEDBACK_SYSTEM_PROMPT, EDU_FEEDBACK_SYSTEM_PROMPT
import time
import asyncio
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class SummaryRequest(BaseModel):
    consultation_id: str
    is_simulation: bool
    simulation_type: str = None  # 시뮬레이션 난이도
    original_consultation_id: str = None  # 우수사례 원본 ID

@router.post("")
async def create_summary(request: SummaryRequest):
    try:
        script = None
        json_script = None

        # 데이터 확보를 위한 재시도 루프 (LLM 처리 시간 고려하여 30회로 증가)
        max_retries = 30
        for i in range(max_retries):
            script, json_script = await get_dialogue(request.consultation_id)
            if script and len(script.strip()) > 0:
                print(f"[{request.consultation_id}] {i+1}차 시도만에 데이터 확보 성공")
                break

            print(f"[{request.consultation_id}] 데이터 대기 중 ({i+1}/{max_retries})")
            await asyncio.sleep(1)

        # max_retries초가 지나도 데이터가 없으면 에러 반환
        if not script or len(script.strip()) == 0:
            raise HTTPException(status_code=404, detail="상담 데이터를 찾을 수 없습니다. (처리 지연)")

        start_parallel = time.time()

        # 비동기 태스크 생성
        summarize_task = get_summarize(script)
        similarity_result = None

        if request.is_simulation:
            feedback_task = generate_feedback(script, EDU_FEEDBACK_SYSTEM_PROMPT)

            # 우수사례 시뮬레이션인 경우 유사도 계산
            if request.simulation_type == "best_practice" and request.original_consultation_id:
                similarity_result = await calculate_consultation_similarity(
                    script, request.original_consultation_id
                )
        else:
            feedback_task = generate_feedback(script, FEEDBACK_SYSTEM_PROMPT)

        # 병렬 실행
        summarize_result, feedback = await asyncio.gather(summarize_task, feedback_task)

        # 결과 검증
        if isinstance(summarize_result, dict) and "error" in summarize_result:
            raise HTTPException(status_code=500, detail=summarize_result["error"])
        if isinstance(feedback, str):
            raise HTTPException(status_code=500, detail=feedback)

        if request.is_simulation:
            # 우수사례인 경우 유사도 피드백 추가
            if request.simulation_type == "best_practice" and similarity_result:
                similarity_score = similarity_result.get("similarity_score", 0)
                feedback["similarity_score"] = similarity_score

                # 피드백 텍스트에 유사도 관련 코멘트 추가
                existing_feedback = feedback.get("feedback", "")
                if similarity_score >= 80:
                    feedback["feedback"] = existing_feedback + f" 우수사례 유사도 {similarity_score}%로 우수합니다."
                elif similarity_score >= 60:
                    feedback["feedback"] = existing_feedback + f" 우수사례 유사도 {similarity_score}%. 핵심 응대를 더 참고하세요."
                else:
                    feedback["feedback"] = existing_feedback + f" 우수사례 유사도 {similarity_score}%. 모범 응대를 학습하세요."
        else:
            # 감정 점수 계산 (일반 피드백만)
            emotions = feedback.get('emotions', [])
            score = evaluate_call(emotions)
            feedback["emotion_score"] = score.get("emotion_score", 0)

        parallel_time = time.time() - start_parallel
        print(f"병렬 처리 시간(요약+피드백): {parallel_time:.2f}초")

        return {
            "isSuccess": True,
            "code": 200,
            "message": "후처리 문서가 생성되었습니다.",
            "summary": summarize_result,
            "evaluation": feedback,
            "script": json_script,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"최종 처리 중 예외 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class SaveConsultationRequest(BaseModel):
    consultationId: str
    employeeId: str
    customerId: str
    customerName: str
    title: str
    status: str
    category: str
    subcategory: str
    aiSummary: str
    memo: str
    followUpTasks: str
    handoffDepartment: str
    handoffNotes: str
    callTimeSeconds: int
    acwTimeSeconds: int
    datetime: str
    referencedDocumentIds: list
    referencedDocuments: list
    evaluation: dict


@router.post("/save")
async def save_consultation(data: SaveConsultationRequest):
    try:
        conn = connect_db()

        # 기존 데이터 및 상담 스크립트 확보
        script, _ = await get_dialogue(data.consultationId)
        customer_script = refine_script(script)
        print(f'고객전문 : {customer_script}')

        # DB에서 기존 히스토리 조회
        past_history = get_personality_history(conn, data.customerId)
        print(past_history)

        # 현재 상담 성향 분석
        current_type_code = await get_personality(customer_script)

        # 현재 상담 정보를 DB 예시 포맷에 맞게 객체화
        current_entry = {
            "type_code": current_type_code,
            "assigned_at": datetime.now().strftime("%Y-%m-%d"),
            "consultation_id": data.consultationId
        }

        # 히스토리 합치기 및 최종 성향 결정
        total_history = past_history + [current_entry]
        print(total_history)
        final_type_code, updated_history = determine_personality(total_history)

        # DB 업데이트
        update_customer(conn, data.customerId, final_type_code, updated_history, False)

        # 상담 내역 저장 코드 추가하기
        save_consultation_to_db(conn, data, script, data.evaluation)
        conn.close()

        return {
            "isSuccess": True,
            "code": 200,
            "message": "상담 내역 및 고객 성향이 저장되었습니다."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상담 내역 및 고객 성향 저장 실패: {str(e)}")
