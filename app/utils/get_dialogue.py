import json
import redis.asyncio as redis
from app.core.config import DIALOGUE_REDIS_URL

redis_client = redis.from_url(DIALOGUE_REDIS_URL, decode_responses=True)

async def get_dialogue(session_id: str):
    key = f"stt:{session_id}"
    
    # 키가 존재하는지 확인
    exists = await redis_client.exists(key)
    print(f"--- Debug: Key {key} exists? {exists} ---")
    
    if not exists:
        return ""

    # 데이터 가져오기 (문자열로 저장했으므로 get)
    raw_data = await redis_client.get(key)
    print(f"--- Debug: Raw data from Redis: {raw_data} ---")

    if not raw_data:
        return ""

    try:
        data = json.loads(raw_data)
        # 리스트 형태를 "speaker: message" 문자열로 변환
        formatted_text = "\n".join([f"{i['speaker']}: {i['message']}" for i in data])
        return formatted_text
    
    except Exception as e:
        print(f"--- Debug: JSON Parsing Error: {e} ---")
        return ""