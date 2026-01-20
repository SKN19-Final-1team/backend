from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List
from utils.evaluate_call import evaluate_call
from llm.follow_up.feedback_generator import generate_feedback

router = APIRouter()

# 요약 요청용
class SummaryRequest(BaseModel):
    script: str

# 평가 점수 요청용
class EvaluationRequest(BaseModel):
    script: str
    work_time: int
    emotions: List[str]


# /api/v1/followup/summary
@router.post("/summary")
async def create_summary(request: Request, body: SummaryRequest):
    summarizer = getattr(request.app.state, "summarizer", None)
    if not summarizer:
        raise HTTPException(status_code=500, detail="AI 모델 초기화 실패")
    
    return summarizer.summarize(body.script)


# /api/v1/followup/evaluate
@router.post("/evaluate")
async def create_evaluation(body: EvaluationRequest):
    
    try:
        # OpenAI를 통한 피드백 생성
        feedback = await generate_feedback(body.script)
        
        # 로컬 로직을 통한 정량 점수 계산
        score = evaluate_call(body.work_time, body.emotions)
        
        final_report = {
            "feedback" : feedback,
            "score": score
            }
        
        return final_report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))