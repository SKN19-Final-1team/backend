from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from collections import Counter
import time

SYSTEM_PROMPT = """
상담 스크립트에서 고객의 성향을 분류하세요
전체적인 맥락을 참고하되 판단의 근거는 고객의 발화에 한정합니다

### 분류 규칙
1. 제시된 성향 키워드 중 가장 적절한 한가지만 선택
2. 설명이나 판단 근거는 절대 출력하지말고 오직 하나의 키워드만 출력한다
3. 여러 개의 성향을 가질 경우 S3 > S2 > S1 > N3 > N2 > N1 의 순서로 우선 순위를 가진다

### 성향 키워드 목록
- N1: 실용주의형. 불필요한 말 없이 바로 문의사항을 말함
- N2: 수다형. 사적인 이야기나 본인 상황을 길게 설명함
- N3: 신중형. 신중하고 의심을 보임
- S1: 급한성격형. 빠른 처리를 선호함
- S2: 이해부족형. 설명을 잘 이해하지 못하여 반복적으로 확인함
- S3: 불만형. 분노, 짜증을 드러냄
"""

load_dotenv()

client = AsyncOpenAI(
    base_url=os.getenv("RUNPOD_URL"),
    api_key=os.getenv("API_KEY")
)

async def get_personality(script):
    try:
        start = time.perf_counter()
        
        response = await client.chat.completions.create(
            model="ansui/customer-analysis-merged",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": script}
            ],
            temperature=0.0,
            max_tokens=10,
            # [| 문자가 나타나면 바로 멈추도록 stop 추가
            stop=["[|", "[|end|]", "[|user|]", "\n"] 
        )
        
        result = response.choices[0].message.content.strip()
        
        # 특수 토큰 파편 제거
        if "[" in result:
            result = result.split("[")[0].strip()
            
        # 혹시나 포함되어 있을 수 있는 태그 추가 정제
        result = result.replace("[|assistant|]", "").replace("[|end|]", "").strip()

        end = time.perf_counter()
        latency = end - start

        print(f"Latency: {latency:.4f}s | Result: {result}")
        
        return result

    except Exception as e:
        print(f"런팟 통신 실패: {str(e)}")
        return "N1"
    
    
def determine_personality(personality_history):
    """
    personality_history: 최근 상담 성향 리스트 (예: ['N1', 'S2', 'S2'])
    index 0이 가장 오래된 상담, 마지막 index가 가장 최근 상담이라고 가정
    """
    if not personality_history:
        return "데이터 없음"
    
    # 빈도수 계산
    counts = Counter(personality_history)
    most_common = counts.most_common()
    
    # 다수결 확인 (가장 많이 나온 성향이 2번 이상인 경우)
    if most_common[0][1] >= 2:
        return most_common[0][0]
    
    # 모두 다를 경우 -> 가장 최근 상담 성향 반영
    # S5, S4, S1 같은 특정 키워드가 포함되어 있다면 우선 순위 부여
    priority_codes = ['S5', 'S4', 'S1']
    for code in priority_codes:
        if code in personality_history:
            return f"{code}"
            
    return personality_history[-1]