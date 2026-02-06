import asyncio
import os
import time
import tempfile
import edge_tts

# 한국어 성우 목록
# ko-KR-SunHiNeural (여성, 차분함)
# ko-KR-InJoonNeural (남성, 안정감)
VOICE = "ko-KR-SunHiNeural"

async def generate_tts(text: str, output_file: str) -> float:
    """
    Edge TTS를 사용하여 텍스트를 음성으로 변환하고 저장합니다.
    소요 시간을 반환합니다.
    """
    print(f"   [EdgeTTS] 요청: '{text}'")
    start_time = time.time()
    
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(output_file)
    except Exception as e:
        print(f"   [Error] {e}")
        raise e
        
    end_time = time.time()
    latency = end_time - start_time
    
    file_size = os.path.getsize(output_file)
    print(f"   [EdgeTTS] 완료: {latency:.4f}초 (크기: {file_size} bytes)")
    return latency

async def main():
    print("=" * 60)
    print("   🎤 Edge TTS (Microsoft) 테스트")
    print("=" * 60)
    print(f"사용 목소리: {VOICE}")
    print("🚀 성능 측정을 위해 5회 연속 테스트를 수행합니다...")
    
    test_texts = [
        "안녕하세요, Edge TTS 테스트 중입니다.",
        "속도가 얼마나 빠른지 확인해 볼까요?",
        "이 방식은 별도의 GPU 서버가 필요 없습니다.",
        "마이크로소프트 서버를 이용하기 때문이죠.",
        "빠르게 응답했으면 좋겠습니다."
    ]
    
    total_latency = 0
    success_count = 0
    
    temp_dir = tempfile.gettempdir()
    
    for i, text in enumerate(test_texts):
        print(f"\n[{i+1}/5] 테스트 진행 중...")
        # mp3 포맷으로 저장됨
        output_path = os.path.join(temp_dir, f"edge_tts_{int(time.time())}.mp3")
        
        try:
            latency = await generate_tts(text, output_path)
            total_latency += latency
            success_count += 1
            
            # 마지막 파일만 재생
            if i == len(test_texts) - 1:
                print(f"🔊 마지막 결과 재생: {output_path}")
                if os.name == 'nt':
                    os.system(f"start {output_path}")
                else:
                    print("   (Windows가 아니라서 자동 재생 생략)")
                    
        except Exception as e:
            print(f"❌ 실패: {e}")
            
        # 너무 빠른 연속 요청 방지 (옵션)
        await asyncio.sleep(0.5)

    print("\n" + "=" * 60)
    print("   테스트 결과 요약")
    print("=" * 60)
    print(f"총 요청 수: {len(test_texts)}")
    print(f"성공 요청 수: {success_count}")
    if success_count > 0:
        avg_latency = total_latency / success_count
        print(f"평균 응답 시간: {avg_latency:.4f}초")
    else:
        print("평균 시간을 계산할 수 없습니다.")
    print("테스트 완료.")

if __name__ == "__main__":
    # Windows에서 asyncio 실행 시 정책 설정 (일부 환경에서 필요)
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
