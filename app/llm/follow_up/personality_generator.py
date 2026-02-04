from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from collections import Counter
import time
from app.core.prompt import PERSONALITY_SYSTEM_PROMPT

load_dotenv()

client = AsyncOpenAI(
    base_url=os.getenv("ACW_TYPE_RUNPOD_URL"),
    api_key=os.getenv("RUNPOD_API_KEY")
)

async def get_personality(script):
    try:
        start = time.perf_counter()
        
        response = await client.chat.completions.create(
            model="ansui/kanana-customer-analysis-merged",
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


def determine_personality(total_history_objs):
    """
    total_history_objs: [{type_code, assigned_at, consultation_id}, ...] 형태의 리스트
    반환값: (최종 결정된 성향 코드, 업데이트된 3개의 히스토리 객체 리스트)
    """
    # 최근 3개의 히스토리 객체만 유지
    updated_history_objs = total_history_objs[-3:]
    
    # type_code만 추출
    type_codes = [obj['type_code'] for obj in updated_history_objs]
    
    # 빈도수 계산
    counts = Counter(type_codes)
    most_common = counts.most_common()
    
    # 3개 중 2개 이상 일치하는 성향이 있으면 그것으로 결정
    if most_common[0][1] >= 2:
        representative = most_common[0][0]
    
    # 3개가 모두 다를 경우 (우선순위 로직)
    else:
        priority_codes = ['S3', 'S2', 'S1'] 
        representative = None
        
        for code in priority_codes:
            if code in type_codes:
                representative = code
                break
        
        # 우선순위 코드도 없다면 가장 최신(마지막) 결과의 코드를 선택
        if not representative:
            representative = type_codes[-1]

    return representative, updated_history_objs