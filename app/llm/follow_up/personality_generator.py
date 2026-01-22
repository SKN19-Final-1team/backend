from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

SYSTEM_PROMPT = """
상담 스크립트에서 고객의 성향을 분류하세요.
전체적인 맥락을 참고하되 판단의 근거는 고객의 발화에 한정합니다.

### 분류 규칙
1. 제시된 성향 키워드 중 가장 적절한 한가지만 선택한다
2. 오직 키워드만 추출 (예: S5)

### 성향 키워드 목록
- N1 (조용한내성형): 간결한 답변을 원함
- N2 (실용주의형): 불필요한 말 없이 목적 달성에만 집중함
- N3 (친화적수다형): 사적인 이야기나 본인 상황을 길게 설명함
- N4 (신중/보안 중시형): 신중하고 의심을 보임
- S1 (급한성격형): 빠른 결론과 처리를 선호함
- S2 (감정호소형): 본인의 사정을 감정적으로 호소하며 공감을 구함
- S3 (시니어친화형): 정보 기기 사용이 서툴고 설명을 잘 이해하지 못함
- S4 (디지털네이티브): 앱 사용에 능숙하며 전문 용어나 신조어 사용에 거부감이 없음
- S5 (VIP고객형): 본인의 등급이나 기여도를 언급하며 특별한 대우를 기대함
- S6 (반복민원형): 과거의 상담 이력을 언급하며 해결되지 않은 불만을 반복 제기함
- S7 (불만항의형): 강한 어조, 분노, 짜증을 드러내며 서비스나 규정에 대해 항의함
"""

load_dotenv()

client = AsyncOpenAI(
    base_url=os.getenv("RUNPOD_URL2"),
    api_key=os.getenv("API_KEY")
)

async def get_personality(script):
    try:
        response = await client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"상담 전문:\n{script}"}
            ],
            temperature=0.0,
        )

        # 답변 반환
        return response.choices[0].message.content

    except Exception as e:
        return {"error": f"런팟 서버 통신 중 오류 발생: {str(e)}"}