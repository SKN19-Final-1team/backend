"""
교육 시뮬레이션 대화 관리 모듈

가상 고객 페르소나와의 실시간 대화를 관리합니다.
- 대화 세션 초기화 및 상태 관리
- 상담원 입력 처리 및 AI 고객 응답 생성
- 대화 히스토리 추적
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from app.llm.education.client import generate_text


# 전역 세션 저장소 (추후 Redis 등으로 대체 가능)
_conversation_sessions: Dict[str, Dict[str, Any]] = {}


class ConversationSession:
    """대화 세션 클래스"""
    
    def __init__(self, session_id: str, system_prompt: str, customer_profile: Dict[str, Any]):
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.customer_profile = customer_profile
        self.conversation_history: List[Dict[str, str]] = []
        self.created_at = datetime.now()
        self.turn_count = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """세션 정보를 딕셔너리로 변환"""
        return {
            "session_id": self.session_id,
            "customer_name": self.customer_profile.get("name", "고객"),
            "customer_profile": self.customer_profile,
            "conversation_history": self.conversation_history,
            "turn_count": self.turn_count,
            "created_at": self.created_at.isoformat(),
        }


def initialize_conversation(
    session_id: str,
    system_prompt: str,
    customer_profile: Dict[str, Any]
) -> ConversationSession:
    """
    대화 세션 초기화
    
    Args:
        session_id: 세션 ID (simulation/start에서 생성된 ID)
        system_prompt: 페르소나 시스템 프롬프트
        customer_profile: 고객 프로필 정보
        
    Returns:
        ConversationSession 객체
    """
    session = ConversationSession(session_id, system_prompt, customer_profile)
    _conversation_sessions[session_id] = session
    
    print(f"[Conversation] 세션 초기화: {session_id}")
    print(f"[Conversation] 고객: {customer_profile.get('name', '고객')}")
    
    return session


def process_agent_input(
    session_id: str,
    agent_message: str,
    input_mode: str = "text"
) -> Dict[str, Any]:
    """
    상담원 입력 처리 및 AI 고객 응답 생성
    
    Args:
        session_id: 세션 ID
        agent_message: 상담원 메시지 (텍스트 또는 STT 변환된 텍스트)
        input_mode: 입력 모드 ("text" 또는 "voice")
        
    Returns:
        {
            "customer_response": "AI 고객 응답 텍스트",
            "turn_number": 대화 턴 번호,
            "audio_url": TTS 오디오 파일 경로
        }
    """
    session = _conversation_sessions.get(session_id)
    
    if not session:
        raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
    
    # 대화 히스토리에 상담원 메시지 추가
    session.conversation_history.append({
        "role": "agent",
        "content": agent_message,
        "timestamp": datetime.now().isoformat()
    })
    
    # LLM에 전달할 메시지 구성
    # 시스템 프롬프트 + 대화 히스토리
    conversation_context = _build_conversation_context(session)
    
    # AI 고객 응답 생성
    print(f"[Conversation] LLM 요청 시작 (세션: {session_id})")
    customer_response = generate_text(
        prompt=agent_message,
        system_prompt=conversation_context,
        temperature=0.3,  # 페르소나 일관성을 위해 낮은 temperature
        max_tokens=200
    )
    
    if not customer_response:
        customer_response = "죄송합니다, 잘 이해하지 못했습니다. 다시 말씀해주시겠어요?"
    
    # 대화 히스토리에 고객 응답 추가
    session.conversation_history.append({
        "role": "customer",
        "content": customer_response,
        "timestamp": datetime.now().isoformat()
    })
    
    session.turn_count += 1
    
    # TTS 음성 생성
    audio_url = None
    try:
        from app.llm.education.tts_engine import generate_speech
        
        # 오디오 파일 경로 생성
        output_dir = f"app/llm/education/tts_output/{session_id}"
        audio_filename = f"response_{session.turn_count:03d}.wav"
        audio_path = f"{output_dir}/{audio_filename}"
        
        # 음성 설정
        voice_config = session.customer_profile.get("communication_style", {})
        
        # TTS 생성
        success = generate_speech(
            text=customer_response,
            voice_config=voice_config,
            output_path=audio_path
        )
        
        if success:
            audio_url = f"/static/tts_output/{session_id}/{audio_filename}"
            print(f"[Conversation] TTS 생성 완료: {audio_url}")
        else:
            print(f"[Conversation] TTS 생성 실패 (텍스트 응답만 반환)")
            
    except ImportError:
        print(f"[Conversation] TTS 엔진을 사용할 수 없습니다 (텍스트 응답만 반환)")
    except Exception as e:
        print(f"[Conversation] TTS 생성 중 오류: {e}")
    
    print(f"[Conversation] 고객 응답 생성 완료 (턴: {session.turn_count})")
    
    return {
        "customer_response": customer_response,
        "turn_number": session.turn_count,
        "audio_url": audio_url
    }


def _build_conversation_context(session: ConversationSession) -> str:
    """
    대화 컨텍스트 구성
    
    시스템 프롬프트 + 최근 대화 히스토리를 결합하여
    LLM에 전달할 컨텍스트를 생성합니다.
    """
    context = session.system_prompt + "\n\n## 현재까지의 대화\n\n"
    
    # 최근 5턴만 포함 (토큰 제한 고려)
    recent_history = session.conversation_history[-10:]  # 5턴 = 10개 메시지
    
    for msg in recent_history:
        role_label = "상담원" if msg["role"] == "agent" else "고객(당신)"
        context += f"{role_label}: {msg['content']}\n"
    
    context += "\n상담원의 마지막 말에 고객으로서 자연스럽게 응답하세요."
    
    return context


def get_conversation_history(session_id: str) -> List[Dict[str, str]]:
    """
    대화 히스토리 조회
    
    Args:
        session_id: 세션 ID
        
    Returns:
        대화 히스토리 리스트
    """
    session = _conversation_sessions.get(session_id)
    
    if not session:
        raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
    
    return session.conversation_history


def end_conversation(session_id: str) -> Dict[str, Any]:
    """
    대화 세션 종료
    
    Args:
        session_id: 세션 ID
        
    Returns:
        대화 요약 정보
    """
    session = _conversation_sessions.get(session_id)
    
    if not session:
        raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
    
    # 대화 시간 계산
    duration = (datetime.now() - session.created_at).total_seconds()
    
    summary = {
        "session_id": session_id,
        "customer_name": session.customer_profile.get("name", "고객"),
        "turn_count": session.turn_count,
        "duration_seconds": duration,
        "conversation_history": session.conversation_history
    }
    
    # 세션 삭제
    del _conversation_sessions[session_id]
    
    print(f"[Conversation] 세션 종료: {session_id} (턴: {session.turn_count}, 시간: {duration:.1f}초)")
    
    return summary


def get_session_info(session_id: str) -> Optional[Dict[str, Any]]:
    """
    세션 정보 조회
    
    Args:
        session_id: 세션 ID
        
    Returns:
        세션 정보 딕셔너리 또는 None
    """
    session = _conversation_sessions.get(session_id)
    
    if not session:
        return None
    
    return session.to_dict()
