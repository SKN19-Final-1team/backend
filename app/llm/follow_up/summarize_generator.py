from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import os

system_prompt = """
상담 스크립트를 바탕으로 아래 JSON 형식에 맞춰 응답하세요

### 제약 사항
1. 출력은 JSON 데이터만 허용합니다
2. JSON 외의 서론, 결론, 마크다운 기호(```), ```json\n은 절대 포함하지 마세요
3. ▲는 포함하지 마세요

### 출력 형식 (JSON)
{{
    "title": "상담의 핵심 주제를 나타내는 간결한 제목 (예: 결제 오류 문의 및 해결)",
    "status": "'진행중', '완료' 중 택일",
    "inquiry": "고객이 문의한 핵심 내용을 1줄로 요약",
    "process": ["상담 과정 요약", "1단계", "2단계", ...],
    "result": "상담 결과 요약",
    "next_step": "상담 종료 후 상담원이 추가로 할 일 (없으면 '')",
    "transfer_dep": "이관이 필요한 경우 부서명 (없으면 '')",
    "transfer_note": "이관 부서에 전달할 내용 (없으면 '')"
}}
"""

load_dotenv()

client = AsyncOpenAI(
    base_url=os.getenv("RUNPOD_URL"),
    api_key=os.getenv("API_KEY")
)

async def get_summarize(script):
    try:
        response = await client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"상담 전문:\n{script}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        # 답변 반환
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {"error": f"런팟 서버 통신 중 오류 발생: {str(e)}"}