from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import uuid
from audio.whisper import WhisperService 
from app.rag.pipeline import RAGConfig, run_rag

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    print(f"[{session_id}] 웹소켓 연결 완료")

    whisper_service = WhisperService()

    # 결과 처리 콜백
    async def on_transcription_result(text: str):
        print(f"[{session_id}] STT 결과 : {text}")
        # await websocket.send_json(text)

        try:
            result = await run_rag(text, config=RAGConfig(top_k=4, normalize_keywords=True))
            
            if result:
                print(f"[{session_id}] RAG 검색 결과 : {result}")
                await websocket.send_json(result)  # RAG 검색 결과 프론트로 전송

                
        except Exception as e:
            print(f"[{session_id}] RAG 처리 중 에러 : {e}")

    # 서비스 시작
    loop = asyncio.get_running_loop()
    whisper_service.start(callback=on_transcription_result, loop=loop)

    try:
        while True:
            data = await websocket.receive_bytes()
            whisper_service.add_audio(data)
            
    except WebSocketDisconnect:
        print(f"[{session_id}] 연결 종료")

    except Exception as e:
        print(f"[{session_id}] 오류 발생: {e}")

    finally:
        whisper_service.stop()
        print(f"[{session_id}] 연결 종료")
