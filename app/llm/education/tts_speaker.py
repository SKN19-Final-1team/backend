from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import json

from app.llm.education.client import generate_text
from app.llm.education.text_refiner import unmask_text


# 페르소나 태그 → TTS instruct 매핑
_TAG_TO_INSTRUCT = {
    # 감정
    "angry": "화가 나서 짜증스럽게 말해",
    "frustrated": "불만이 있어 답답한 목소리로 말해",
    "impatient": "급하고 조급한 목소리로 빠르게 말해",
    "urgent": "다급하게 말해",
    "confused": "혼란스럽고 불안한 목소리로 말해",
    # 톤
    "friendly": "친근하고 밝은 목소리로 말해",
    "talkative": "수다스럽고 활기차게 말해",
    "direct": "단호하고 직접적으로 말해",
    "passive": "무관심하고 건조하게 말해",
    "disengaged": "지루한 듯 무심하게 말해",
    "cautious": "조심스럽고 신중하게 말해",
    "suspicious": "의심하는 듯 경계하며 말해",
    # 특수
    "elderly": "천천히 또박또박 말해",
    "vip": "격식을 차려 정중하게 말해",
    "foreign": "어눌한 한국어로 천천히 말해",
    "polite": "공손하고 부드럽게 말해",
    "emotional": "감정이 풍부하게 말해",
}

_SPEED_TO_INSTRUCT = {
    "slow": "천천히 말해",
    "fast": "빠르게 말해",
}


def build_tts_instruct(customer_profile: Dict[str, Any]) -> str:
    """페르소나 프로필에서 TTS instruct 문자열 생성"""
    parts = []

    # personality_tags에서 감정/말투 추출
    tags = customer_profile.get("personality_tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.strip("{}").split(",") if t.strip()]

    for tag in tags:
        desc = _TAG_TO_INSTRUCT.get(tag.strip().lower())
        if desc:
            parts.append(desc)

    # communication_style에서 speed 추출
    comm_style = customer_profile.get("communication_style", {})
    if isinstance(comm_style, str):
        import json as _json
        try:
            comm_style = _json.loads(comm_style)
        except Exception:
            comm_style = {}

    speed = comm_style.get("speed", "")
    speed_desc = _SPEED_TO_INSTRUCT.get(speed)
    if speed_desc and speed_desc not in parts:
        parts.append(speed_desc)

    return ", ".join(parts) if parts else ""


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

    # sLLM 응답 마스킹 해제 (TTS 전 정제)
    customer_name = session.customer_profile.get("name", "고객")
    customer_response = unmask_text(customer_response, customer_name=customer_name)

    # sLLM 응답을 터미널에 JSON으로 출력
    sllm_response_data = {
        "session_id": session_id,
        "turn": session.turn_count + 1,
        "agent_input": agent_message,
        "customer_response": customer_response
    }
    print("\n[Conversation] 🤖 sLLM 응답:")
    print(json.dumps(sllm_response_data, ensure_ascii=False, indent=2))
    
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
        audio_filename = f"response_{session.turn_count:03d}.mp3"
        audio_path = f"{output_dir}/{audio_filename}"
        
        # 음성 설정 (페르소나 기반 instruct 포함)
        voice_config = session.customer_profile.get("communication_style", {})
        if isinstance(voice_config, str):
            import json as _json
            try:
                voice_config = _json.loads(voice_config)
            except Exception:
                voice_config = {}
        voice_config = dict(voice_config)
        voice_config["instruct"] = build_tts_instruct(session.customer_profile)
        
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
