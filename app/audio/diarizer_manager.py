import redis.asyncio as redis
import json
import asyncio
from app.core.config import DIALOGUE_REDIS_URL
from app.audio.diarizer import (
    call_diarizer_fulltext, 
    merge_batches, 
    merge_same_speaker, 
    dedupe_near_duplicates
)
from app.llm.delivery.sllm_refiner import refine_diarized_batch

class DiarizationManager:
    def __init__(self, session_id, client):
        self.session_id = session_id
        self.client = client
        self.redis_url = DIALOGUE_REDIS_URL
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        
        self.buffer = []               # 실시간 STT 파편
        self.global_items = []         # 최종적으로 누적된 화자분리 결과물
        self.batch_threshold = 3
        self.active_tasks = set()

    async def add_fragment(self, text, system_prompt):
        """텍스트를 버퍼에 넣고 LLM 처리 후 비움"""
        if text.strip():
            self.buffer.append(text)
            
        if len(self.buffer) >= self.batch_threshold:
            batch_text = " ".join(self.buffer)
            self.buffer = []
            
            task = asyncio.create_task(self.process_diarization(batch_text, system_prompt))
            self.active_tasks.add(task)
            task.add_done_callback(self.active_tasks.discard)

    async def process_diarization(self, batch_text, system_prompt):
        """병합"""
        try:
            new_items, _, _ = await call_diarizer_fulltext(
                client=self.client,
                model="gpt-4o",
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
        if self.buffer:
            combined = " ".join(self.buffer)
            self.global_items.append({"speaker": "agent", "message": combined})
            self.buffer = []

        if self.global_items:
            # 저장 전 최종적으로 동일 화자 병합 및 근사 중복 제거
            self.global_items = merge_same_speaker(self.global_items)
            self.global_items = dedupe_near_duplicates(self.global_items, ratio=0.95)
            
            # 대화 전문 보정
            # 블로킹 방지를 위해 run_in_executor로 호출 
            loop = asyncio.get_event_loop()
            self.global_items = await loop.run_in_executor(
                None, refine_diarized_batch, self.global_items
            )
            
            await self.redis.set(
                f"stt:{self.session_id}", 
                json.dumps(self.global_items, ensure_ascii=False)
            )
            print(f"===[{self.session_id}] Redis 최종 저장 완료===")
            print(f"[{self.session_id}] 처리 완료 / 현재 총 {len(self.global_items)}개 발화")

        return self.global_items

    async def get_final_script(self, system_prompt: str):
        """종료 시 호출: 진행 중인 태스크 완료 대기 후 남은 버퍼 처리"""
        if self.active_tasks:
            print(f"[{self.session_id}] 실행 중인 작업 {len(self.active_tasks)}개 마무리 중...")
            await asyncio.gather(*self.active_tasks)
        
        # 마지막으로 버퍼에 남은 작업 처리
        if self.buffer:
            buffer_size = len(self.buffer)
            batch_text = " ".join(self.buffer)
            
            if buffer_size >= 2:
                # 2개 이상이면 LLM 호출
                print(f"[{self.session_id}] 잔여 파편 {buffer_size}개 감지: LLM으로 화자 분리 진행")
                await self.process_diarization(batch_text, system_prompt)
            else:
                # 1개이면 상담원으로 즉시 할당
                print(f"[{self.session_id}] 잔여 파편 1개 감지: 상담원(agent)으로 자동 할당")
                self.global_items.append({"speaker": "agent", "message": batch_text})
            
            self.buffer = []

        return await self.save_to_redis()