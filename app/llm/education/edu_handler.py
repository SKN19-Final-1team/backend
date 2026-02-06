"""
교육 시뮬레이션 WebSocket 핸들러

edu_websocket.py의 수정을 최소화하기 위해
시뮬레이션 관련 로직을 분리한 모듈입니다.
"""
import json
import asyncio
from typing import Dict, Any, Optional
from starlette.websockets import WebSocket, WebSocketState

from app.llm.education.tts_speaker import (
    process_agent_input,
    get_session_info,
    end_conversation,
)
from app.rag.pipeline import RAGConfig, run_rag

# 활성 시뮬레이션 세션 (ws_session_id -> simulation_session_id)
_active_sessions: Dict[str, str] = {}


async def handle_init_simulation(
    ws_session_id: str,
    simulation_session_id: str,
    websocket: WebSocket
) -> bool:
    """
    시뮬레이션 세션 초기화

    Args:
        ws_session_id: WebSocket 세션 ID
        simulation_session_id: 시뮬레이션 세션 ID (education.py에서 생성)
        websocket: WebSocket 연결

    Returns:
        초기화 성공 여부
    """
    session_info = get_session_info(simulation_session_id)

    if not session_info:
        await websocket.send_json({
            "type": "error",
            "error": f"시뮬레이션 세션을 찾을 수 없습니다: {simulation_session_id}"
        })
        return False

    _active_sessions[ws_session_id] = simulation_session_id

    await websocket.send_json({
        "type": "simulation_initialized",
        "simulation_session_id": simulation_session_id,
        "customer_name": session_info.get("customer_name")
    })

    print(f"[{ws_session_id}] 시뮬레이션 초기화 완료: {simulation_session_id}")
    return True


async def handle_agent_message(
    ws_session_id: str,
    text: str,
    input_mode: str,
    websocket: WebSocket,
    session_state: dict
) -> None:
    """
    상담원 메시지 처리 (STT 또는 텍스트)

    Args:
        ws_session_id: WebSocket 세션 ID
        text: 상담원 메시지 텍스트
        input_mode: 입력 모드 ("voice" 또는 "text")
        websocket: WebSocket 연결
        session_state: RAG 세션 상태
    """
    simulation_session_id = _active_sessions.get(ws_session_id)

    if not simulation_session_id:
        return  # 시뮬레이션 미초기화 시 무시

    try:
        # ⭐ [v25] sLLM 응답 + TTS 생성 (동기 함수를 스레드 풀에서 실행하여 event loop 차단 방지)
        result = await asyncio.to_thread(
            process_agent_input,
            session_id=simulation_session_id,
            agent_message=text,
            input_mode=input_mode
        )

        # customer_response 전송
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "customer_response",
                "data": {
                    "text": result["customer_response"],
                    "turn_number": result["turn_number"],
                    "audio_url": result.get("audio_url")
                }
            })

    except Exception as e:
        print(f"[{ws_session_id}] sLLM 처리 중 에러: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "error": f"AI 고객 응답 생성 실패: {str(e)}"
            })


async def handle_json_message(
    ws_session_id: str,
    message_text: str,
    websocket: WebSocket,
    session_state: dict
) -> None:
    """
    JSON 메시지 처리

    Args:
        ws_session_id: WebSocket 세션 ID
        message_text: JSON 문자열
        websocket: WebSocket 연결
        session_state: RAG 세션 상태
    """
    try:
        data = json.loads(message_text)
    except json.JSONDecodeError:
        await websocket.send_json({
            "type": "error",
            "error": "잘못된 JSON 형식입니다."
        })
        return

    msg_type = data.get("type")

    if msg_type == "init_simulation":
        await handle_init_simulation(
            ws_session_id,
            data.get("simulation_session_id"),
            websocket
        )

    elif msg_type == "text_message":
        text = data.get("text", "").strip()
        if text:
            await handle_agent_message(
                ws_session_id, text, "text", websocket, session_state
            )
            # RAG도 실행
            try:
                rag_result = await run_rag(
                    text,
                    config=RAGConfig(top_k=4, normalize_keywords=True),
                    session_state=session_state,
                )
                if rag_result and websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"type": "rag", "data": rag_result})
            except Exception as e:
                print(f"[{ws_session_id}] RAG 처리 중 에러: {e}")


def is_simulation_active(ws_session_id: str) -> bool:
    """
    시뮬레이션 활성 여부 확인

    Args:
        ws_session_id: WebSocket 세션 ID

    Returns:
        활성 여부
    """
    return ws_session_id in _active_sessions


def cleanup_session(ws_session_id: str) -> None:
    """
    세션 정리

    Args:
        ws_session_id: WebSocket 세션 ID
    """
    simulation_session_id = _active_sessions.pop(ws_session_id, None)
    if simulation_session_id:
        try:
            end_conversation(simulation_session_id)
            print(f"[{ws_session_id}] 시뮬레이션 세션 정리 완료: {simulation_session_id}")
        except Exception as e:
            print(f"[{ws_session_id}] 시뮬레이션 세션 정리 중 오류: {e}")
