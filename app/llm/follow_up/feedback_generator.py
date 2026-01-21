import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

SYSTEM_PROMPT = """
상담 스크립트를 평가 기준에 따라 객관적으로 평가하세요

### 제약 사항
1. 피드백에는 점수에 대한 근거를 반드시 제시한다
2. 고객의 감사 표현과 고객 감정 변화 지표는 고객 발화에서만 평가한다
3. 추측, 설명 문장, 자연어 해설 금지 - JSON만 출력한다

### 평가 기준
1. 매뉴얼 준수 (50점에서 감점하는 방식)
intro
- 인사말
0점: 첫인사 + 마무리 멘트 모두 수행
-5점: 첫인사 또는 마무리 멘트 누락
- 고객확인
0점: 고객정보를 고객에게 직접 질문
-5점: 상담원이 고객정보를 먼저 발화하여 정보 누출

response
- 호응어
0점: 공감/감성 호응
-5점: 기운 없음, 짜증 섞인 표현
- 대기 표현
0점: 대기 표현 모두 수행
-5점: 대기 표현 누락

explanation
- 커뮤니케이션
0점: 핵심 요약 + 이해 쉬운 설명
-5점: 일방적 설명, 단답형
- 알기 쉬운 설명
0점: 고객 눈높이 설명 + 부연
-5점: 복잡한 설명/상담자 관점 설명

proactivity
- 적극성
0점: 적극적 대응
-5점: 수동적 대응, 대안 없음
- 언어표현
0점: 정중/경어체/긍정 표현
-5점: 전문용어, 줄임말, 명령조, 무시 표현

accuracy
- 정확한 업무처리
0점: 오류 없음
-10점: 임의 판단으로 업무 오류 발생

2. 고객 감사 표현 (10점)
- 고객 발화 중 감사/칭찬 키워드 포함 시 1회 카운트
- 0회: 0점 / 1회: 5점 / 2회 이상: 10점

3. 고객 감정 변화 지표 (점수 없음)
고객 발화 중 구간을 3파트로 나눠 감정을 3가지 중 하나 선택 - 부정 | 중립 | 긍정
- 부정: 불만, 분노, 짜증, 불안, 항의, 문제 제기
- 중립: 사실 전달, 질문, 감정 표현 거의 없음
- 긍정: 만족, 안도, 감사, 동의, 긍정적 반응


### 출력 형식 (JSON)
{{
"manual_compliance": {{
    "intro_score": 0,
    "response_score": 0,
    "explanation_score": 0,
    "proactivity_score": 0,
    "accuracy_score": 0,
    "manual_score": "0~50점"}},
"customer_thanks": {{
    "count": 0,
    "thanks_score": "0~10점"}},
"feedback": "존댓말을 사용하고 5문장 이내로 피드백 요약",
"emotions": {{
    "early": "부정|중립|긍정",
    "mid": "부정|중립|긍정",
    "late": "부정|중립|긍정"}}
}}
"""

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

async def generate_feedback(script, model_name="gpt-4.1-mini"):
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"상담 스크립트:\n{script}"}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        result_json = json.loads(content)

        return result_json

    except Exception as e:
        return f"{model_name} 호출 중 오류 발생 : {e}"