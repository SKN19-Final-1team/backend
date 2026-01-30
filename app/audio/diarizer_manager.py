import redis.asyncio as redis
import json
from app.core.config import DIALOGUE_REDIS_URL
from app.audio.diarizer import (
    call_diarizer_fulltext, 
    merge_batches, 
    merge_same_speaker, 
    dedupe_near_duplicates
)
from app.core.config import DIALOGUE_REDIS_URL

class DiarizationManager:
    def __init__(self, session_id, client):
        self.session_id = session_id
        self.client = client
        self.redis_url = DIALOGUE_REDIS_URL
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        
        self.buffer = []           # 실시간 STT 파편
        self.global_items = []     # 최종적으로 누적된 화자분리 결과물
        self.batch_threshold = 3 

    async def add_fragment(self, text, system_prompt):
        """텍스트를 버퍼에 넣고 LLM 처리 후 비움"""
        if text.strip():
            self.buffer.append(text)
            
        if len(self.buffer) >= self.batch_threshold:
            # 특정 개수만큼 쌓였을 때 즉시 처리
            await self.process_diarization(system_prompt)

    async def process_diarization(self, system_prompt):
        """병합"""
        if not self.buffer:
            return

        # 현재 버퍼를 텍스트로 합치고 즉시 비움
        batch_text = " ".join(self.buffer)
        self.buffer = [] 
                
        try:
            new_items, _, _ = await call_diarizer_fulltext(
                client=self.client,
                model="ft:gpt-4o-mini-2024-07-18:callact:diar-v1:D0pqahvO",
                system_prompt=system_prompt,
                raw_stream_batch=batch_text
            )

            if new_items:
                # 유사도 및 부분 겹침 트리밍 적용
                self.global_items = merge_batches(
                    self.global_items,
                    new_items,
                    max_overlap_utts=8,            # 이전 대화 8개까지 참조하여 겹침 확인
                    min_partial_overlap_chars=12   # 12자 이상 겹치면 트리밍
                )

        except Exception as e:
            print(f"[{self.session_id}] 배치 처리 중 에러: {e}")
            # 에러 시 데이터 유실 방지를 위해 원문 보관
            self.global_items.append({"speaker": "unknown", "message": batch_text})

    async def save_to_redis(self):
        """최종 결과를 Redis에 저장"""
        if self.global_items:
            # 저장 전 최종적으로 동일 화자 병합 및 근사 중복 제거
            self.global_items = merge_same_speaker(self.global_items)
            self.global_items = dedupe_near_duplicates(self.global_items, ratio=0.95)
            
            await self.redis.set(
                f"stt:{self.session_id}", 
                json.dumps(self.global_items, ensure_ascii=False)
            )
            print(f"===[{self.session_id}] Redis 최종 저장 완료===")
            print(f"[{self.session_id}] 처리 완료 / 현재 총 {len(self.global_items)}개 발화")

        return self.global_items

    async def get_final_script(self, system_prompt: str):
        """종료 시 남은 파편 처리 후 최종 반환"""
        if self.buffer:
            await self.process_diarization(system_prompt)
        
        return await self.save_to_redis()