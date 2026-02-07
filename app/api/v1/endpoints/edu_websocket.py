import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import asyncio
import uuid
from openai import AsyncOpenAI
from app.audio.whisper import WhisperService
from app.rag.pipeline import RAGConfig, run_rag
from app.audio.diarizer_manager import DiarizationManager
from app.core.prompt import DIAR_SYSTEM_PROMPT
import time

from app.llm.education.edu_handler import (
    handle_json_message,
    handle_agent_message,
    is_simulation_active,
    cleanup_session,
)

router = APIRouter()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.websocket("/ws/edu")
async def call_websocket_endpoint(websocket: WebSocket, consultation_id: str = None):
    os.environ.setdefault("RAG_LOG_TIMING", "1")
    await websocket.accept()
    session_id = consultation_id if consultation_id else str(uuid.uuid4())[:8]
    
    print(f"[{session_id}] 교육 웹소켓 연결 완료")
    await websocket.send_json({
        "type": "connected",
        "ws_session_id": session_id
    })
    
    whisper_service = WhisperService()
    diarizer_manager = DiarizationManager(session_id, client)
    session_state = {}

    async def on_transcription_result(text: str):
        if not text.strip():
            return

        print(f"[{session_id}] STT 원문 : {text}")

        # ⭐ [v25] STT 결과를 프론트엔드로 전송 (call_websocket과 동일)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "stt", "text": text})

        # --- STT 텍스트 적재 ---
        await diarizer_manager.add_fragment(text, DIAR_SYSTEM_PROMPT)

        # --- 시뮬레이션 모드: sLLM 응답 생성 ---
        if is_simulation_active(session_id):
            await handle_agent_message(session_id, text, "voice", websocket, session_state)

        try:
            # --- RAG 실행 ---
            result = await run_rag(
                text,
                config=RAGConfig(top_k=6, normalize_keywords=True, llm_card_top_n=4),
                session_state=session_state,
            )
                
            if result and websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "rag", "data": result})
        
        except Exception as e:
            print(f"[{session_id}] 처리 중 에러 : {e}")

    loop = asyncio.get_running_loop()
    whisper_service.start(callback=on_transcription_result, loop=loop)

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                whisper_service.add_audio(message["bytes"])
            elif "text" in message:
                await handle_json_message(session_id, message["text"], websocket, session_state)

    except (WebSocketDisconnect, RuntimeError):
        pass
    
    finally:
        whisper_service.stop()
        cleanup_session(session_id)

        # 즉시 처리 중 상태 마커를 Redis에 저장 (followup API가 대기하도록)
        await diarizer_manager.mark_processing_started()

        await asyncio.sleep(2)

        final_start = time.perf_counter()
        final_script = await diarizer_manager.get_final_script(DIAR_SYSTEM_PROMPT)
        final_end = time.perf_counter()
        test_time = final_end - final_start

        print(f"화자 분리 전문 : {final_script}")
        print(f"시간 : {test_time:.4f}s")