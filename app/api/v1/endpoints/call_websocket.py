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
from fastapi.encoders import jsonable_encoder

router = APIRouter()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.websocket("/ws/call")
async def call_websocket_endpoint(websocket: WebSocket, consultation_id: str = None):
    os.environ.setdefault("RAG_LOG_TIMING", "1")
    await websocket.accept()
    session_id = consultation_id if consultation_id else str(uuid.uuid4())[:8]
    
    print(f"[{session_id}] 웹소켓 연결 완료")
    await websocket.send_json(session_id)
    
    whisper_service = WhisperService()
    diarizer_manager = DiarizationManager(session_id, client)
    session_state = {}

    # --- 공통 처리 로직 (Whisper 이후 혹은 직접 입력된 텍스트) ---
    async def process_text_payload(text: str, is_stt: bool = False):
        if not text.strip():
            return

        print(f"[{session_id}] 처리할 텍스트 : {text}")

        # ⭐ [v24] STT 결과를 프론트엔드로 먼저 전송
        if is_stt and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "stt", "text": text})

        # Diarizer 적재
        await diarizer_manager.add_fragment(text, DIAR_SYSTEM_PROMPT)

        try:
            # RAG 실행 (top_k 증가 및 llm_card_top_n 명시적 설정)
            result = await run_rag(
                text,
                config=RAGConfig(top_k=6, normalize_keywords=True, llm_card_top_n=4),
                session_state=session_state,
            )

            if result and websocket.client_state == WebSocketState.CONNECTED:
                safe_result = jsonable_encoder(result)
                await websocket.send_json({"type": "rag", "data": safe_result})

        except Exception as e:
            print(f"[{session_id}] 처리 중 에러 : {e}")

    # Whisper 콜백 함수도 분리된 로직을 사용하도록 수정
    async def on_transcription_result(text: str):
        await process_text_payload(text, is_stt=True)  # ⭐ STT 결과임을 표시

    loop = asyncio.get_running_loop()
    whisper_service.start(callback=on_transcription_result, loop=loop)

    try:
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                print(f"[{session_id}] 클라이언트가 연결을 끊었습니다.")
                break

            # 바이너리(음성) 데이터 처리
            if "bytes" in message:
                whisper_service.add_audio(message["bytes"])
            
            # 일반 텍스트 처리
            elif "text" in message:
                text_data = message["text"]
                await process_text_payload(text_data)
            
    except WebSocketDisconnect:
        print(f"[{session_id}] 연결 종료 (Exception)")
    
    finally:
        whisper_service.stop()

        # 즉시 처리 중 상태 마커를 Redis에 저장 (followup API가 대기하도록)
        await diarizer_manager.mark_processing_started()

        await asyncio.sleep(2)

        final_script = await diarizer_manager.get_final_script(DIAR_SYSTEM_PROMPT)
        print(f"화자 분리 전문 : {final_script}")