"""
RunPod Qwen3 TTS 테스트 클라이언트

사용법:
    python tests/test_runpod_tts.py
    
환경변수:
    TTS_RUNPOD_URL: RunPod TTS 서버 URL (예: http://xxx.xxx.xxx.xxx:8000)
"""
import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from dotenv import load_dotenv

load_dotenv()

# 서버 URL
TTS_RUNPOD_URL = os.getenv("TTS_RUNPOD_URL", "http://localhost:8000")


# 세션 재사용 (Connection Keep-Alive)
session = requests.Session()

def generate_tts(text: str, speaker: str = "Sohee", language: str = "Korean") -> bytes:
    """TTS 생성"""
    payload = {
        "text": text,
        "language": language,
        "speaker": speaker,
        "instruct": ""
    }
    
    print(f"   [Client] 서버 요청 전송 시작...")
    req_start = time.time()
    
    # session 객체 사용
    response = session.post(
        f"{TTS_RUNPOD_URL}/tts",
        json=payload,
        timeout=60
    )
    
    req_end = time.time()
    server_time = response.headers.get("X-Generation-Time", "N/A")
    latency = req_end - req_start
    
    if response.status_code != 200:
        raise Exception(f"TTS 생성 실패: {response.status_code} - {response.text[:200]}")
    
    content_size = len(response.content)
    print(f"   [Client] 응답 수신 완료 (네트워크+서버: {latency:.4f}초)")
    print(f"   [Server] 내부 생성 시간 헤더: {server_time}초")
    print(f"   [Network] 데이터 크기: {content_size} bytes")
    
    return response.content


def play_audio(audio_data: bytes):
    """오디오 재생 (Windows)"""
    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data)
        temp_path = f.name
    
    try:
        # Windows Media Player로 재생
        print(f"🔊 재생 중... ({temp_path})")
        
        # 방법 1: Windows start 명령어
        subprocess.run(["cmd", "/c", "start", "", temp_path], shell=False)
        
        # 재생 완료 대기 (대략적)
        time.sleep(3)
        
    finally:
        # 임시 파일 삭제 (약간의 지연 후)
        try:
            time.sleep(1)
            os.unlink(temp_path)
        except Exception:
            pass


def interactive_mode():
    """자동 테스트 모드"""
    print("=" * 60)
    print("   🎤 RunPod Qwen3 TTS 테스트")
    print("=" * 60)
    print(f"📡 서버: {TTS_RUNPOD_URL}")
    print("🚀 성능 측정을 위해 5회 연속 테스트를 수행합니다...")
    
    test_texts = [
        "안녕하세요, 반가워요.",
        "오늘은 날씨가 참 좋네요.",
        "TTS 속도 테스트 중입니다.",
        "점심 메뉴 추천해주세요.",
        "빠르게 응답했으면 좋겠습니다."
    ]
    
    total_latency = 0
    success_count = 0
    
    for i, text in enumerate(test_texts):
        print(f"\n[{i+1}/5] 요청: '{text}'")
        try:
            start_t = time.time()
            audio = generate_tts(text)
            latency = time.time() - start_t
            
            total_latency += latency
            success_count += 1
            
            # 마지막 결과만 재생
            if i == 4:
                file_path = os.path.join(tempfile.gettempdir(), f"tts_test_{int(time.time())}.wav")
                with open(file_path, "wb") as f:
                    f.write(audio)
                print(f"🔊 마지막 결과 재생: {file_path}")
                if os.name == 'nt':
                    os.system(f"start {file_path}")
                    
        except Exception as e:
            print(f"❌ 실패: {e}")
            
        time.sleep(0.5) # 과부하 방지

    print("\n" + "=" * 60)
    print("   테스트 결과 요약")
    print("=" * 60)
    print(f"총 요청 수: {len(test_texts)}")
    print(f"성공 요청 수: {success_count}")
    if success_count > 0:
        average_latency = total_latency / success_count
        print(f"평균 응답 시간 (네트워크+서버): {average_latency:.4f}초")
    else:
        print("성공한 요청이 없어 평균 응답 시간을 계산할 수 없습니다.")
    print("테스트 완료.")


if __name__ == "__main__":
    interactive_mode()
