import redis.asyncio as redis
import json
from app.audio.diarizer import call_diarizer_fulltext, merge_batches
from app.core.config import DIALOGUE_REDIS_URL

class DiarizationManager:
    def __init__(self, session_id, client):
        self.session_id = session_id
        self.client = client
        self.redis_url = DIALOGUE_REDIS_URL
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        self.buffer = []              # 아직 화자 분리가 안 된 STT 파편들
        self.global_items = []        # 화자 분리가 완료된 최종 대화
        self.batch_threshold = 5      # 몇 개의 파편이 모이면 LLM을 호출할지

    async def add_fragment(self, text, system_prompt):
        self.buffer.append(text)
        
        # 버퍼가 일정 수준 쌓이면 화자 분리 실행
        if len(self.buffer) >= self.batch_threshold:
            await self.process_diarization(system_prompt)

    async def process_diarization(self, system_prompt):
        batch_text = " ".join(self.buffer)
        
        # LLM 호출 (화자 분리)
        new_items, _, _ = await call_diarizer_fulltext(
            client=self.client,
            model="ft:gpt-4o-mini-2024-07-18:callact:diar-v1:D0pqahvO",
            system_prompt=system_prompt,
            raw_stream_batch=batch_text
        )
        
        print(new_items)

        # 기존 대화와 병합
        self.global_items = merge_batches(self.global_items, new_items)

        # Redis에 최종 결과 업데이트
        await self.redis.set(f"stt:{self.session_id}", json.dumps(self.global_items, ensure_ascii=False))
        print(f"redis 저장 완료 : {self.global_items}")
        
        # 버퍼 비우기
        self.buffer = self.buffer[-3:] 

    async def get_final_script(self, system_prompt: str):
        # 남은 버퍼가 있다면 마저 처리
        if self.buffer:
            await self.process_diarization(system_prompt)
            self.buffer = [] # 처리 완료 후 비우기

        # 최종 결과 반환
        return self.global_items