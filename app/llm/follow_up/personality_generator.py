from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from collections import Counter
import time
from app.core.prompt import PERSONALITY_SYSTEM_PROMPT

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
                {"role": "system", "content": PERSONALITY_SYSTEM_PROMPT},
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


def determine_personality(total_history):
    """
    total_history: 최근 상담 성향 리스트 (예: ['N1', 'S2', 'S2', 'N2'])
    index 0이 가장 오래된 상담, 마지막 index가 가장 최근 상담이라고 가정
    
    반환값: (최종 결정된 성향, 업데이트된 3개의 히스토리)
    """
    # ['N1', 'S2', 'S2', 'S3'] -> ['S2', 'S2', 'S3']
    updated_history = (total_history)[-3:]
        
    # 업데이트된 3개 내에서 빈도수 계산
    counts = Counter(updated_history)
    most_common = counts.most_common()
    
    # 3개 중 2개 이상 일치하는 성향이 있으면 그것으로 결정
    if most_common[0][1] >= 2:
        representative = most_common[0][0]
    
    # 3개가 모두 다를 경우 (예: ['N1', 'S2', 'S3'])
    else:
        # 특정 성향에 우선 순위 부여
        priority_codes = ['S3', 'S2', 'S1'] 
        representative = None
        
        for code in priority_codes:
            if code in updated_history:
                representative = code
                break
        
        # 우선 순위 코드도 없다면 가장 최신 결과를 선택
        if not representative:
            representative = total_history[-1]

    return representative, updated_history
