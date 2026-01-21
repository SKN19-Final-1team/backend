from app.utils.evaluate_call import evaluate_call
from app.llm.follow_up.feedback_generator import generate_feedback
from app.llm.follow_up.summarize_generator import get_summarize    
from fastapi import APIRouter, UploadFile, File, HTTPException
import time
import asyncio

router = APIRouter()

# 화자 분리 STT
async def transcribe_audio():
    # 화자 분리 STT 
    # 파일로 분리해서 다른 함수처럼 모듈로 불러오기
    # 현재는 그냥 mockdata 넣은 함수
    script = "상담사: 상담원 ▲▲▲입니다.\n손님: 네, 어 금요일 날인가 ▲▲카드에서 아 제가 카드를 해지했거든요. 다른 카드로 바꿨는데\n상담사: 네.\n손님: 그게 무슨 분쟁 뭐에서 연락이 왔어요.\n상담사: 네.\n손님: 내가 못 받았는데 그게 문자가 왔더라고요, ▲▲▲ 씨라고요.\n손님: 뭐 분쟁 뭐 뭐라고 그러면서요.\n상담사: 네.\n손님: 그래서 저 저 통화를 하고 싶어서요.\n상담사: 아 그러셨을까요, 손님? 많이 궁금하셨을 텐데요, 제가 본인 확인 후 안내해 드리겠습니다. 휴대폰 번호와 생년월일 말씀 부탁드립니다.\n손님: 네.\n손님: ▲▲▲▲▲▲▲▲▲\n손님: ▲▲▲▲▲▲\n상담사: 네, 전화주신 손님 본인에 성함 말씀해 주시겠습니까?\n손님: 네.\n손님: ▲▲▲입니다.\n상담사: 네, 손님 마지막으로 카드와 연결되어 있는 결제 계좌 은행 말씀 부탁드립니다.\n손님: ▲▲은행이요.\n상담사: 네, 손님 본인 확인 감사합니다. 네, 손님 확인해 보니 문자받으셨던 내용 중에 담당자분 연락처 직통 번호 있으신 걸로 확인되는데요.\n손님: 네.\n손님: 네, 그래서 왜냐하면 전화를 요즘 함부로 받지 않고 내가 인제 그런 문제로 제가 해지됐기 때문에 그 대표전화를 일단 한 거예요.\n상담사: 네.\n상담사: 아 그러셨을까요? 그러면 제가 이 부분에 대해서 그럼 담당부서로 전달을 해 드릴까요.\n손님: 네.\n손님: 아 좀 연결해 줄 수는 없어요?\n상담사: 네, 바로 연결은 어렵습니다.\n손님: 아 그러면 그 전화가 맞는 거죠? 네, 알겠습니다. 네 그럼 전화할게요.\n상담사: 네. 그렇습니다.\n상담사: 네.\n상담사: 네, 알겠습니다. 상담사 ▲▲▲이었습니다\n손님: 네."
    return script

@router.post("/")
async def create_summary(file: UploadFile = File(...)):
    try:
        # 화자 분리 STT
        script = await transcribe_audio()
        
        start_parallel = time.time()
        
        # 두 함수 동시에 실행
        summarize_task = get_summarize(script)
        feedback_task = generate_feedback(script)
        
        summarize_result, feedback = await asyncio.gather(summarize_task, feedback_task)
        
        end_parallel = time.time()
        parallel_time = end_parallel - start_parallel

        # 오류 체크
        if "error" in summarize_result:
            raise HTTPException(status_code=500, detail=summarize_result["error"])
        if isinstance(feedback, str):
            raise HTTPException(status_code=500, detail=feedback)

        # 감정 점수 계산
        score = evaluate_call(feedback['emotions'])
        feedback["emotion_score"] = score.get("emotion_score", 0)

        print(f"병렬 처리 시간(요약+피드백): {parallel_time:.2f}초")

        return {
            "status": "success",
            "summary": summarize_result,
            "evaluation": feedback
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))