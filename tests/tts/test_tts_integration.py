"""
TTS 통합 테스트

테스트 흐름:
1. persona_generator로 고객 페르소나 생성
2. SIM_RUNPOD_URL의 sLLM과 대화 시작
3. sLLM 응답을 TTS로 변환
4. 오디오 자동 재생

사용법:
    conda activate final_env
    python tests/tts/test_tts_integration.py

환경변수:
    SIM_RUNPOD_URL: sLLM 서버 URL
    TTS_RUNPOD_URL: TTS 서버 URL
    RUNPOD_API_KEY: RunPod API 키
"""
import os
import sys
import time
import tempfile
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from app.llm.education.persona_generator import create_system_prompt
from app.llm.education.client import generate_text
from app.llm.education.tts_engine import generate_speech, check_server_health


def check_servers():
    """서버 연결 상태 확인"""
    print("=" * 60)
    print("   서버 연결 상태 확인")
    print("=" * 60)

    # sLLM 서버 확인
    sim_url = os.getenv("SIM_RUNPOD_URL")
    tts_url = os.getenv("TTS_RUNPOD_URL")

    print(f"sLLM 서버: {sim_url}")
    print(f"TTS 서버: {tts_url}")

    # TTS 서버 health check
    tts_ok = check_server_health()
    print(f"TTS 서버 상태: {'[OK] 정상' if tts_ok else '[FAIL] 연결 실패'}")

    if not tts_ok:
        print("TTS 서버에 연결할 수 없습니다. 환경변수를 확인하세요.")
        return False

    return True


def create_test_persona():
    """테스트용 페르소나 생성"""
    print("\n" + "=" * 60)
    print("   Step 1: 페르소나 생성")
    print("=" * 60)

    # 테스트용 고객 프로필
    customer_profile = {
        "name": "김민수",
        "age_group": "30대",
        "grade": "GOLD",
        "personality_tags": ["angry", "direct"],
        "communication_style": {
            "tone": "calm_professional",
            "speed": "fast"
        },
        "llm_guidance": "카드 결제 관련 문의가 있습니다."
    }

    # 시스템 프롬프트 생성
    system_prompt = create_system_prompt(
        customer_profile=customer_profile,
        difficulty="beginner"
    )

    print(f"고객명: {customer_profile['name']}")
    print(f"연령대: {customer_profile['age_group']}")
    print(f"등급: {customer_profile['grade']}")
    print(f"성격: {customer_profile['personality_tags']}")
    print(f"\n[시스템 프롬프트 미리보기 (처음 200자)]")
    print(system_prompt[:200] + "...")

    return system_prompt, customer_profile


def chat_with_sllm(system_prompt: str, agent_message: str):
    """sLLM과 대화"""
    print("\n" + "=" * 60)
    print("   Step 2: sLLM 대화")
    print("=" * 60)

    print(f"상담원 입력: \"{agent_message}\"")
    print("sLLM 응답 대기 중...")

    start_time = time.time()

    customer_response = generate_text(
        prompt=agent_message,
        system_prompt=system_prompt,
        temperature=0.3,
        max_tokens=200
    )

    elapsed = time.time() - start_time

    if not customer_response:
        customer_response = "죄송합니다, 잘 이해하지 못했습니다."

    print(f"sLLM 응답 ({elapsed:.2f}초): \"{customer_response}\"")

    return customer_response


def generate_and_play_tts(text: str, session_id: str = "test", turn: int = 1):
    """TTS 생성 및 재생"""
    print("\n" + "=" * 60)
    print("   Step 3: TTS 생성 및 재생")
    print("=" * 60)

    # 출력 경로 설정
    output_dir = tempfile.gettempdir()
    audio_filename = f"tts_test_{session_id}_{turn:03d}.mp3"
    audio_path = os.path.join(output_dir, audio_filename)

    # 음성 설정
    voice_config = {
        "speaker": "Sohee",
        "language": "Korean",
        "speed": "moderate"
    }

    print(f"TTS 변환 텍스트: \"{text}\"")
    print(f"음성 설정: {voice_config}")
    print("TTS 생성 중...")

    start_time = time.time()

    success = generate_speech(
        text=text,
        voice_config=voice_config,
        output_path=audio_path
    )

    elapsed = time.time() - start_time

    if success:
        print(f"[OK] TTS 생성 완료 ({elapsed:.2f}초)")
        print(f"파일 경로: {audio_path}")

        # 파일 크기 확인
        file_size = os.path.getsize(audio_path)
        print(f"파일 크기: {file_size:,} bytes")

        # 오디오 재생 (Windows)
        print("\n[PLAY] 오디오 재생 중...")
        if os.name == 'nt':
            os.system(f'start "" "{audio_path}"')
        else:
            print(f"재생하려면: play {audio_path}")

        return audio_path
    else:
        print(f"[FAIL] TTS 생성 실패")
        return None


def run_conversation_test():
    """인터랙티브 대화 테스트"""
    print("\n" + "=" * 60)
    print("   [TTS] TTS 통합 테스트 - 인터랙티브 모드")
    print("=" * 60)

    # 1. 서버 확인
    if not check_servers():
        return

    # 2. 페르소나 생성
    system_prompt, customer_profile = create_test_persona()

    print("\n" + "-" * 60)
    print("상담원 역할로 대화를 입력하세요.")
    print("고객(AI)이 응답하고 TTS로 음성을 재생합니다.")
    print("종료: 'quit' 또는 'q' 입력")
    print("-" * 60)

    turn = 1
    while True:
        try:
            agent_input = input(f"\n[상담원 턴 {turn}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n테스트를 종료합니다.")
            break

        if agent_input.lower() in ['quit', 'exit', 'q']:
            print("테스트를 종료합니다.")
            break

        if not agent_input:
            continue

        # sLLM 응답 생성
        customer_response = chat_with_sllm(system_prompt, agent_input)

        # TTS 생성 및 재생
        generate_and_play_tts(
            text=customer_response,
            session_id="conversation_test",
            turn=turn
        )

        turn += 1

    print("\n" + "=" * 60)
    print("   [OK] 테스트 완료")
    print("=" * 60)


def run_single_tts_test():
    """단일 TTS 테스트 (sLLM 없이)"""
    print("\n" + "=" * 60)
    print("   [TTS] 단일 TTS 테스트")
    print("=" * 60)

    if not check_servers():
        return

    test_text = "안녕하세요, 저는 테디카드 고객센터입니다. 무엇을 도와드릴까요?"

    audio_path = generate_and_play_tts(
        text=test_text,
        session_id="single_test",
        turn=1
    )

    if audio_path:
        print(f"\n[OK] 단일 TTS 테스트 완료: {audio_path}")


def interactive_mode():
    """인터랙티브 모드"""
    print("\n" + "=" * 60)
    print("   [TTS] TTS 인터랙티브 모드")
    print("=" * 60)

    if not check_servers():
        return

    # 페르소나 생성
    system_prompt, customer_profile = create_test_persona()

    print("\n상담원으로서 대화를 입력하세요. 종료하려면 'quit' 입력.")
    print("-" * 60)

    turn = 1
    while True:
        agent_input = input(f"\n[상담원 턴 {turn}] 입력: ").strip()

        if agent_input.lower() in ['quit', 'exit', 'q']:
            print("테스트를 종료합니다.")
            break

        if not agent_input:
            continue

        # sLLM 응답
        customer_response = chat_with_sllm(system_prompt, agent_input)

        # TTS 생성 및 재생
        generate_and_play_tts(
            text=customer_response,
            session_id="interactive",
            turn=turn
        )

        turn += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TTS 통합 테스트")
    parser.add_argument(
        "--mode",
        choices=["conversation", "single", "interactive"],
        default="conversation",
        help="테스트 모드 선택"
    )

    args = parser.parse_args()

    if args.mode == "conversation":
        run_conversation_test()
    elif args.mode == "single":
        run_single_tts_test()
    elif args.mode == "interactive":
        interactive_mode()
