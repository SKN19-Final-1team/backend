import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import uuid
from openai import AsyncOpenAI
from app.audio.whisper import WhisperService
from app.rag.pipeline import RAGConfig, run_rag
from app.audio.diarizer_manager import DiarizationManager
from app.llm.delivery.deliverer import refine_text

router = APIRouter()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    os.environ.setdefault("RAG_LOG_TIMING", "1")
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    
    print(f"[{session_id}] 웹소켓 연결 완료")
    await websocket.send_json(session_id)
    
    whisper_service = WhisperService()
    diarizer_manager = DiarizationManager(session_id, client)
    session_state = {}

    DIAR_SYSTEM_PROMPT= """
    You are an expert transcript diarizer for Hana Card.
    The user input is a raw, noisy, fragmented text stream from STT (Speech-to-Text).

    ### Your task
    1) Understand the context despite broken spacing and missing punctuation.
    2) Identify who is speaking ('agent' or 'customer').
    3) Output ONLY a valid JSON array of objects like
    [{"speaker":"agent","message":"..."}, {"speaker":"customer","message":"..."}]

    ### Rules
    - Use speaker values exactly: 'agent' or 'customer'.
    - Keep the semantic content; do not invent facts.
    """

    async def on_transcription_result(text: str):
        print(f"[{session_id}] STT 원문 : {text}")
               
        try:
            # --- STT 1차 보정 ---
            refined_result = refine_text(text)
            print(f"정제된 텍스트 : {refined_result}")
            
            if isinstance(refined_result, dict):
                final_text = refined_result.get('text', '')
            else:
                final_text = refined_result
            
            # --- RAG 실행 ---
            # result = await run_rag(
            #     final_text,
            #     config=RAGConfig(top_k=4, normalize_keywords=True),
            #     session_state=session_state,
            # )
            
            # if result:
            #     await websocket.send_json({"type": "rag", "data": result})

            # --- 실시간 화자분리 및 Redis 저장 ---
            if final_text:
                asyncio.create_task(diarizer_manager.add_fragment(final_text, DIAR_SYSTEM_PROMPT))

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
        final_script = await diarizer_manager.get_final_script(DIAR_SYSTEM_PROMPT)
        print(f"[{session_id}] 최종 Redis 저장 완료 (총 {len(final_script)}개 발화)")
        print(f"[{session_id}] 연결 종료 및 리소스 정리")