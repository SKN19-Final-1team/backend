"""
OpenAI TTS API 테스트 클라이언트

Qwen3 TTS 대비 속도 비교 테스트

사용법:
    python tests/test_openai_tts.py

환경변수:
    OPENAI_API_KEY: OpenAI API 키 (필수)
"""
import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# OpenAI 클라이언트
client = OpenAI()

# 사용 가능한 음성
VOICES = {
    "alloy": "중성적인 음성",
    "echo": "남성 음성",
    "fable": "영국식 남성",
    "onyx": "깊은 남성",
    "nova": "여성 음성 (한국어 추천)",
    "shimmer": "부드러운 여성",
}


def generate_tts(
    text: str,
    voice: str = "nova",
    model: str = "tts-1",  # tts-1 (빠름) 또는 tts-1-hd (고품질)
    response_format: str = "mp3"
) -> tuple[bytes, float]:
    """
    OpenAI TTS 생성

    Returns:
        (audio_data, generation_time)
    """
    print(f"   [OpenAI] 요청 전송 중... (model={model}, voice={voice})")

    t0 = time.time()

    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=response_format
    )

    audio_data = response.content
    elapsed = time.time() - t0

    print(f"   [OpenAI] 응답 수신 완료 ({elapsed:.4f}초)")
    print(f"   [OpenAI] 데이터 크기: {len(audio_data)} bytes")

    return audio_data, elapsed


def generate_tts_streaming(
    text: str,
    voice: str = "nova",
    model: str = "tts-1"
) -> tuple[bytes, float, float]:
    """
    OpenAI TTS 스트리밍 생성

    Returns:
        (audio_data, first_chunk_time, total_time)
    """
    print(f"   [OpenAI Streaming] 요청 전송 중...")

    t0 = time.time()
    first_chunk_time = None
    chunks = []

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        response_format="mp3"
    ) as response:
        for chunk in response.iter_bytes(chunk_size=4096):
            if first_chunk_time is None:
                first_chunk_time = time.time() - t0
                print(f"   [OpenAI Streaming] 첫 청크 수신: {first_chunk_time:.4f}초")
            chunks.append(chunk)

    total_time = time.time() - t0
    audio_data = b''.join(chunks)

    print(f"   [OpenAI Streaming] 전체 완료: {total_time:.4f}초")
    print(f"   [OpenAI Streaming] 데이터 크기: {len(audio_data)} bytes")

    return audio_data, first_chunk_time, total_time


def play_audio(audio_data: bytes, suffix: str = ".mp3"):
    """오디오 재생 (Windows)"""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_data)
        temp_path = f.name

    print(f"   재생 중... ({temp_path})")

    if os.name == 'nt':
        subprocess.run(["cmd", "/c", "start", "", temp_path], shell=False)
        time.sleep(3)

    return temp_path


def run_comparison_test():
    """OpenAI TTS 속도 테스트"""
    print("=" * 60)
    print("   OpenAI TTS API 속도 테스트")
    print("=" * 60)

    test_texts = [
        "안녕하세요, 반가워요.",
        "오늘은 날씨가 참 좋네요.",
        "TTS 속도 테스트 중입니다.",
        "점심 메뉴 추천해주세요.",
        "빠르게 응답했으면 좋겠습니다."
    ]

    # 1. 일반 모드 테스트 (tts-1)
    print("\n[테스트 1] tts-1 모델 (빠른 모드)")
    print("-" * 40)

    total_time_fast = 0
    for i, text in enumerate(test_texts):
        print(f"\n[{i+1}/5] '{text}'")
        try:
            audio, elapsed = generate_tts(text, model="tts-1", voice="nova")
            total_time_fast += elapsed
        except Exception as e:
            print(f"   오류: {e}")

    avg_fast = total_time_fast / len(test_texts)
    print(f"\n[tts-1] 평균 응답 시간: {avg_fast:.4f}초")

    # 2. 고품질 모드 테스트 (tts-1-hd)
    print("\n[테스트 2] tts-1-hd 모델 (고품질 모드)")
    print("-" * 40)

    total_time_hd = 0
    for i, text in enumerate(test_texts):
        print(f"\n[{i+1}/5] '{text}'")
        try:
            audio, elapsed = generate_tts(text, model="tts-1-hd", voice="nova")
            total_time_hd += elapsed
        except Exception as e:
            print(f"   오류: {e}")

    avg_hd = total_time_hd / len(test_texts)
    print(f"\n[tts-1-hd] 평균 응답 시간: {avg_hd:.4f}초")

    # 3. 스트리밍 모드 테스트
    print("\n[테스트 3] 스트리밍 모드 (첫 응답 시간)")
    print("-" * 40)

    total_first_chunk = 0
    for i, text in enumerate(test_texts):
        print(f"\n[{i+1}/5] '{text}'")
        try:
            audio, first_chunk, total = generate_tts_streaming(text, voice="nova")
            total_first_chunk += first_chunk
        except Exception as e:
            print(f"   오류: {e}")

    avg_first_chunk = total_first_chunk / len(test_texts)
    print(f"\n[Streaming] 평균 첫 청크 시간: {avg_first_chunk:.4f}초")

    # 결과 요약
    print("\n" + "=" * 60)
    print("   테스트 결과 요약")
    print("=" * 60)
    print(f"tts-1 (빠름)     평균: {avg_fast:.4f}초")
    print(f"tts-1-hd (고품질) 평균: {avg_hd:.4f}초")
    print(f"스트리밍 첫 응답   평균: {avg_first_chunk:.4f}초")
    print("-" * 60)
    print("참고: Qwen3 TTS 평균 ~6초")
    print("=" * 60)

    # 마지막 오디오 재생
    print("\n마지막 결과 재생 중...")
    play_audio(audio)


def test_single():
    """단일 테스트"""
    print("단일 TTS 테스트")
    text = "안녕하세요, 오늘 하루도 좋은 하루 되세요."

    audio, elapsed = generate_tts(text, voice="nova", model="tts-1")
    print(f"\n생성 시간: {elapsed:.4f}초")

    play_audio(audio)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI TTS 테스트")
    parser.add_argument("--single", action="store_true", help="단일 테스트만 실행")
    args = parser.parse_args()

    if args.single:
        test_single()
    else:
        run_comparison_test()
