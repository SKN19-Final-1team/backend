import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import asyncio
import uuid
from openai import AsyncOpenAI
from app.audio.whisper import WhisperService
from app.rag.pipeline import RAGConfig, run_rag
from app.audio.diarizer_manager import DiarizationManager

router = APIRouter()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.websocket("/ws/edu")
async def edu_websocket_endpoint(websocket: WebSocket):
    os.environ.setdefault("RAG_LOG_TIMING", "1")
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    
    print(f"[{session_id}] 웹소켓 연결 완료")
    await websocket.send_json(session_id)
    
    whisper_service = WhisperService()
    diarizer_manager = DiarizationManager(session_id, client)
    session_state = {}

    async def on_transcription_result(agent_text: str):
        if not agent_text.strip():
            return
        
        print(f"[{session_id}] 상담원(STT) : {agent_text}")

        # --- STT 텍스트(상담원) 적재 ---
        diarizer_manager.global_items.append({'speaker': 'agent', 'message': agent_text})
               
        try:
            # --- 상담원 텍스트 고객 llm에 전달 ---
            # 상준님 여기 추가해주세요
            # customer_text = llm전달함수(agent_text)
            customer_text = ''
        
            if customer_text:
                # --- 받은 응답(고객) 적재 ---
                diarizer_manager.global_items.append({'speaker': 'customer', 'message': customer_text})

                # --- RAG 실행 ---
                result = await run_rag(
                    customer_text,
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
        await asyncio.sleep(1)
        final_script = await diarizer_manager.save_to_redis()

        print(f"화자 분리 전문 : {final_script}")