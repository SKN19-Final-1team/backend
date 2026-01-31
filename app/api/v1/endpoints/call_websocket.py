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

router = APIRouter()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.websocket("/ws/call")
async def call_websocket_endpoint(websocket: WebSocket):
    os.environ.setdefault("RAG_LOG_TIMING", "1")
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    
    print(f"[{session_id}] 웹소켓 연결 완료")
    await websocket.send_json(session_id)
    
    whisper_service = WhisperService()
    diarizer_manager = DiarizationManager(session_id, client)
    session_state = {}

    async def on_transcription_result(text: str):
        if not text.strip():
            return
        
        print(f"[{session_id}] STT 원문 : {text}")

        # --- STT 텍스트 적재 ---
        await diarizer_manager.add_fragment(text, DIAR_SYSTEM_PROMPT)
               
        try:
            # --- RAG 실행 ---
            result = await run_rag(
                text,
                config=RAGConfig(top_k=4, normalize_keywords=True),
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
            data = await websocket.receive_bytes()
            whisper_service.add_audio(data)
            
    except WebSocketDisconnect:
        pass
    
    finally:    
        whisper_service.stop()
        await asyncio.sleep(2)
        
        final_start = time.perf_counter()
        final_script = await diarizer_manager.get_final_script(DIAR_SYSTEM_PROMPT)
        final_end = time.perf_counter()
        test_time = final_end - final_start
        
        print(f"화자 분리 전문 : {final_script}")
        print(f"시간 : {test_time:.4f}s")