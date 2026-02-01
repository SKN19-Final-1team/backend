import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import asyncio
import uuid
from openai import AsyncOpenAI
from app.audio.whisper import WhisperService
from app.rag.pipeline import RAGConfig, run_rag
from app.audio.diarizer_manager import DiarizationManager
from app.llm.education.tts_speaker import (
    initialize_conversation,
    process_agent_input,
    get_session_info,
    end_conversation,
)

router = APIRouter()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 활성 시뮬레이션 세션 저장 (WebSocket session_id -> simulation session_id)
_active_simulation_sessions = {}

@router.websocket("/ws/edu")
async def edu_websocket_endpoint(websocket: WebSocket):
    os.environ.setdefault("RAG_LOG_TIMING", "1")
    await websocket.accept()
    ws_session_id = str(uuid.uuid4())[:8]
    simulation_session_id = None

    print(f"[{ws_session_id}] 웹소켓 연결 완료")
    await websocket.send_json({"type": "connected", "ws_session_id": ws_session_id})

    whisper_service = WhisperService()
    diarizer_manager = DiarizationManager(ws_session_id, client)
    session_state = {}

    async def on_transcription_result(agent_text: str):
        nonlocal simulation_session_id

        if not agent_text.strip():
            return

        print(f"[{ws_session_id}] 상담원(STT) : {agent_text}")

        # --- STT 텍스트(상담원) 적재 ---
        diarizer_manager.global_items.append({'speaker': 'agent', 'message': agent_text})

        try:
            customer_text = ''
            result = None

            # --- 상담원 텍스트 고객 LLM에 전달 ---
            if simulation_session_id and get_session_info(simulation_session_id):
                try:
                    response = process_agent_input(
                        session_id=simulation_session_id,
                        agent_message=agent_text,
                        input_mode="voice"
                    )
                    customer_text = response.get("customer_response", "")
                    turn_number = response.get("turn_number", 0)
                    audio_url = response.get("audio_url")

                    print(f"[{ws_session_id}] AI고객 응답 (턴 {turn_number}): {customer_text}")

                    # AI 고객 응답을 WebSocket으로 전송
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json({
                            "type": "customer_response",
                            "data": {
                                "text": customer_text,
                                "turn_number": turn_number,
                                "audio_url": audio_url
                            }
                        })

                except Exception as llm_error:
                    print(f"[{ws_session_id}] LLM 호출 오류: {llm_error}")
                    customer_text = ""
            else:
                print(f"[{ws_session_id}] 시뮬레이션 세션이 초기화되지 않음")

            if customer_text:
                # --- 받은 응답(고객) 적재 ---
                diarizer_manager.global_items.append({'speaker': 'customer', 'message': customer_text})

                # --- RAG 실행 (상담원 가이드 제공) ---
                result = await run_rag(
                    customer_text,
                    config=RAGConfig(top_k=4, normalize_keywords=True),
                    session_state=session_state,
                )

            if result and websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "rag", "data": result})

        except Exception as e:
            print(f"[{ws_session_id}] 처리 중 에러 : {e}")

    loop = asyncio.get_running_loop()
    whisper_service.start(callback=on_transcription_result, loop=loop)

    try:
        while True:
            message = await websocket.receive()

            # 바이너리 데이터 (오디오)
            if "bytes" in message:
                whisper_service.add_audio(message["bytes"])

            # JSON 메시지 처리
            elif "text" in message:
                import json
                try:
                    json_data = json.loads(message["text"])
                    msg_type = json_data.get("type")

                    # 시뮬레이션 세션 초기화
                    if msg_type == "init_simulation":
                        simulation_session_id = json_data.get("simulation_session_id")
                        _active_simulation_sessions[ws_session_id] = simulation_session_id
                        print(f"[{ws_session_id}] 시뮬레이션 세션 연결: {simulation_session_id}")

                        if websocket.client_state == WebSocketState.CONNECTED:
                            await websocket.send_json({
                                "type": "simulation_initialized",
                                "simulation_session_id": simulation_session_id
                            })

                    # 텍스트 메시지 직접 전송 (음성 없이)
                    elif msg_type == "text_message":
                        text = json_data.get("text", "")
                        if text:
                            await on_transcription_result(text)

                except json.JSONDecodeError:
                    print(f"[{ws_session_id}] JSON 파싱 실패")

    except WebSocketDisconnect:
        pass

    finally:
        whisper_service.stop()
        await asyncio.sleep(2)

        # 시뮬레이션 세션 정리
        if ws_session_id in _active_simulation_sessions:
            del _active_simulation_sessions[ws_session_id]

        final_script = await diarizer_manager.save_to_redis()

        print(f"[{ws_session_id}] 화자 분리 전문 : {final_script}")