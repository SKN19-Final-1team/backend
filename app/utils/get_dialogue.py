import json
import redis.asyncio as redis
from app.core.config import DIALOGUE_REDIS_URL
import re

redis_client = redis.from_url(DIALOGUE_REDIS_URL, decode_responses=True)

async def get_dialogue(session_id: str):
    key = f"stt:{session_id}"
    
    exists = await redis_client.exists(key)
    if not exists:
        return ""

    raw_data = await redis_client.get(key)
    if not raw_data:
        return ""

    try:
        data = json.loads(raw_data)
        
        # 화자 매핑 딕셔너리 생성
        speaker_map = {
            "agent": "상담원",
            "customer": "고객"
        }
        
        # 매핑 정보를 사용하여 텍스트 변환
        formatted_text = "\n".join([
            f"{speaker_map.get(i['speaker'], i['speaker'])}: {i['message']}" 
            for i in data
        ])
        
        return formatted_text, data
    
    except Exception as e:
        print(f"--- Debug: JSON Parsing Error: {e} ---")
        return ""


def refine_script(script):
    noise_patterns = [
        "안녕하세요?", "예.", "네.", "알겠습니다.", "수고하십니다.", "감사합니다."
    ]
    
    lines = script.split('\n')
    refined_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith("고객:"):
            # "손님:" 태그 제거
            content = line.replace("고객:", "").strip()
            
            # 문장 안에 포함된 노이즈 패턴들을 하나씩 찾아서 ""(빈칸)으로 변경
            for pattern in noise_patterns:
                content = content.replace(pattern, "")
            
            # 양끝 공백 정리
            content = content.strip()
            
            # 만약 노이즈를 다 지웠더니 남은 내용이 너무 짧으면(5자 이하) 빈칸 처리
            if len(content) <= 4:
                continue
            
            refined_lines.append(content)
            
    # 리스트를 공백 하나를 사이에 두고 합침
    result = " ".join(refined_lines)
    
    # 연속된 공백(빈 문자열 때문에 생긴 것들)을 하나로 줄임
    return re.sub(r'\s+', ' ', result).strip()