import requests
import json
import time
import os
import tempfile
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# RunPod URL 가져오기
TTS_RUNPOD_URL = os.getenv("TTS_RUNPOD_URL", "http://localhost:8000")

print(f"📡 타겟 서버: {TTS_RUNPOD_URL}")

def test_qwen_tts(text: str, speaker: str = "Sohee", language: str = "Korean"):
    """
    최적화된 Qwen3 TTS 서버에 음성 합성을 요청합니다 (MP3 응답).
    """
    url = f"{TTS_RUNPOD_URL}/tts"
    payload = {
        "text": text,
        "language": language,
        "speaker": speaker,
        "instruct": ""
    }
    
    start_t = time.time()
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        latency = time.time() - start_t
        
        # 서버 헤더에서 상세 시간 정보 추출
        gen_time = response.headers.get("X-Generation-Time", "N/A")
        conv_time = response.headers.get("X-Conversion-Time", "N/A")
        total_time = response.headers.get("X-Total-Time", "N/A")
        
        # MP3 파일 저장
        audio_content = response.content
        temp_dir = tempfile.gettempdir()
        filename = f"qwen_tts_{speaker}_{int(time.time())}.mp3"
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(audio_content)
            
        file_size_kb = len(audio_content) / 1024
        
        print(f"   ✅ 생성 완료")
        print(f"      - 클라이언트 측 총 시간: {latency:.3f}초")
        print(f"      - 서버 생성 시간: {gen_time}초")
        print(f"      - MP3 변환 시간: {conv_time}초")
        print(f"      - 파일 크기: {file_size_kb:.1f} KB")
        print(f"      - 저장 위치: {file_path}")
        
        return latency, file_path, True
        
    except Exception as e:
        print(f"   ❌ 요청 실패: {e}")
        return 0, None, False

def main():
    print("="*60)
    print("   🎤 Qwen3 TTS v2.0 (MP3 Optimized) 테스트")
    print("="*60)
    
    # 테스트 시나리오
    scenarios = [
        ("안녕하세요, 최적화된 Qwen3 TTS 서버입니다.", "Sohee"),
        ("이제 MP3 압축으로 전송 속도가 빨라졌습니다.", "Sohee"),
        ("네트워크 지연 시간이 크게 줄어들었어요.", "Sohee"),
        ("파일 크기도 10분의 1로 작아졌습니다.", "Sohee"),
        ("테스트를 완료합니다. 감사합니다!", "Sohee")
    ]
    
    total_latency = 0
    success_count = 0
    
    for i, (text, speaker) in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] Speaker: {speaker}")
        print(f"   입력: '{text}'")
        
        latency, file_path, success = test_qwen_tts(text, speaker)
        
        if success:
            total_latency += latency
            success_count += 1
            
            # 마지막 파일 재생 (Windows)
            if i == len(scenarios) - 1 and os.name == 'nt':
                print(f"   🔊 마지막 결과 재생 중...")
                os.system(f"start {file_path}")
                
        time.sleep(0.5)

    print("\n" + "="*60)
    if success_count > 0:
        avg = total_latency / success_count
        print(f"📊 평균 응답 속도: {avg:.3f}초 (총 {success_count}회 성공)")
        print(f"💡 MP3 압축으로 네트워크 전송 시간이 대폭 단축되었습니다!")
    else:
        print("❌ 테스트 실패 (성공한 요청 없음)")
    print("="*60)

if __name__ == "__main__":
    main()
