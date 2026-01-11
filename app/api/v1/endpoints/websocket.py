from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import uuid
from audio.whisper import WhisperService 

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    print(f"[{session_id}] 웹소켓 연결 완료")

    whisper_service = WhisperService()

    # 결과 처리 콜백
    async def on_transcription_result(text: str, keyword_data: dict):
        print(f"[{session_id}] STT 결과: {text}")

        if keyword_data:
            print(f"[{session_id}] 키워드 감지 : {keyword_data}")

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