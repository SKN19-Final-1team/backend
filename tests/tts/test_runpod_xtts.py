import requests
import json
import time
import os
import tempfile
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# RunPod URL 가져오기 (.env 파일에 TTS_RUNPOD_URL=http://xxx.xxx.xxx.xxx:xxxx 설정 필요)
TTS_RUNPOD_URL = os.getenv("TTS_RUNPOD_URL")
if not TTS_RUNPOD_URL:
    print("❌ .env 파일에 TTS_RUNPOD_URL이 설정되지 않았습니다.")
    # 기본값 (테스트용)
    TTS_RUNPOD_URL = "http://localhost:8000"

print(f"📡 타겟 서버: {TTS_RUNPOD_URL}")

def test_xtts(text: str, persona: str = "default_female"):
    """
    XTTS 서버에 음성 합성을 요청합니다.
    """
    url = f"{TTS_RUNPOD_URL}/tts"
    payload = {
        "text": text,
        "language": "ko",
        "persona": persona
    }
    
    start_t = time.time()
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        latency = time.time() - start_t
        
        # 파일 저장 (MP3)
        audio_content = response.content
        temp_dir = tempfile.gettempdir()
        filename = f"xtts_{persona}_{int(time.time())}.mp3"
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(audio_content)
            
        print(f"   ✅ 생성 완료 ({latency:.3f}초) -> {file_path}")
        return latency, file_path, True
        
    except Exception as e:
        print(f"   ❌ 요청 실패: {e}")
        return 0, None, False

def main():
    print("="*60)
    print("   🎤 RunPod XTTS v2 페르소나 테스트")
    print("="*60)
    
    # 테스트 시나리오
    scenarios = [
        ("안녕하세요, 저는 차분한 여성 목소리입니다.", "default_female"),
        ("반갑습니다! 저는 힘찬 남성 목소리입니다.", "default_male"),
        ("다양한 페르소나를 지원하는 XTTS 테스트 중입니다.", "default_female"),
        ("속도가 얼마나 빠른지 확인해 볼까요?", "default_male"),
        ("감사합니다.", "default_female")
    ]
    
    total_latency = 0
    success_count = 0
    
    for i, (text, persona) in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] Persona: {persona}")
        print(f"   입력: '{text}'")
        
        latency, file_path, success = test_xtts(text, persona)
        
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
    else:
        print("❌ 테스트 실패 (성공한 요청 없음)")
    print("="*60)

if __name__ == "__main__":
    main()
