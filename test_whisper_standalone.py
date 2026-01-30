"""
Whisper STT + sLLM 교정 단독 테스트
마이크로 음성을 녹음받아 STT 전사 및 텍스트 교정을 테스트합니다.
"""

import os
import sys
import asyncio
import wave
import io
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.audio.whisper import WhisperService

# 오디오 녹음 관련 임포트
try:
    import pyaudio
    import webrtcvad
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[WARNING] pyaudio 또는 webrtcvad가 설치되지 않았습니다.")
    print("설치 방법: pip install pyaudio webrtcvad")


class AudioRecorder:
    """실시간 음성 녹음 클래스"""
    
    def __init__(self, sample_rate=16000, frame_duration=30):
        """
        Args:
            sample_rate: 샘플링 레이트 (Hz)
            frame_duration: 프레임 길이 (ms)
        """
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.chunk_size = self.frame_size * 2  # bytes
        
        self.audio = None
        self.stream = None
        self.vad = None
        
    def start(self):
        """녹음 시작"""
        if not AUDIO_AVAILABLE:
            raise RuntimeError("pyaudio 또는 webrtcvad가 설치되지 않았습니다.")
        
        self.audio = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size
        )
        
        print("🎤 녹음 시작 (Enter 키를 눌러 종료)")
    
    def stop(self):
        """녹음 종료"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        
        print("⏹️  녹음 종료")
    
    def record_until_enter(self) -> bytes:
        """Enter 키를 누를 때까지 녹음"""
        import threading
        
        frames = []
        recording = True
        
        def wait_for_enter():
            nonlocal recording
            input()
            recording = False
        
        # Enter 대기 스레드 시작
        enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
        enter_thread.start()
        
        # 녹음
        while recording:
            try:
                data = self.stream.read(self.frame_size, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                print(f"[ERROR] 녹음 오류: {e}")
                break
        
        # WAV 파일로 변환
        return self._frames_to_wav(frames)
    
    def _frames_to_wav(self, frames: list) -> bytes:
        """프레임 리스트를 WAV 바이트로 변환"""
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        return wav_buffer.getvalue()


class WhisperTester:
    def __init__(self):
        self.results = []
        self.service = None
        
    async def on_transcription(self, text: str):
        """STT 결과 콜백"""
        print(f"\n✅ 최종 결과: {text}")
        self.results.append(text)
    
    async def test_microphone(self):
        """마이크 녹음 테스트"""
        print("\n" + "=" * 70)
        print("Whisper STT + sLLM 교정 마이크 테스트")
        print("=" * 70)
        
        if not AUDIO_AVAILABLE:
            print("[ERROR] pyaudio 또는 webrtcvad가 설치되지 않았습니다.")
            print("설치 방법: pip install pyaudio webrtcvad")
            return
        
        # WhisperService 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] OPENAI_API_KEY가 설정되지 않았습니다.")
            return
        
        self.service = WhisperService(api_key=api_key)
        
        # 이벤트 루프 가져오기
        loop = asyncio.get_event_loop()
        
        # 서비스 시작
        self.service.start(callback=self.on_transcription, loop=loop)
        
        # 녹음기 초기화
        recorder = AudioRecorder()
        
        while True:
            try:
                # 사용자 입력
                user_input = input("\n💬 명령 (record/r: 녹음, quit/q: 종료) > ").strip().lower()
                
                # 종료 명령
                if user_input in ['quit', 'q', '종료']:
                    print("\n테스트를 종료합니다.")
                    break
                
                # 녹음 명령
                if user_input in ['record', 'r', '녹음']:
                    # 녹음 시작
                    recorder.start()
                    audio_data = recorder.record_until_enter()
                    recorder.stop()
                    
                    print(f"\n📊 녹음 크기: {len(audio_data)} bytes")
                    print("⏳ STT 처리 중...")
                    
                    # 결과 초기화
                    self.results = []
                    
                    # 오디오 데이터 추가
                    self.service.add_audio(audio_data)
                    
                    # 처리 완료 대기 (최대 30초)
                    for i in range(30):
                        await asyncio.sleep(1)
                        if self.results:
                            break
                    
                    if not self.results:
                        print("[WARNING] 30초 내에 결과를 받지 못했습니다.")
                
            except KeyboardInterrupt:
                print("\n\n테스트를 종료합니다.")
                break
            except Exception as e:
                print(f"\n[ERROR] 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        
        # 서비스 종료
        self.service.stop()
        
        print("\n" + "=" * 70)
        print("테스트 완료")
        print("=" * 70)


async def main():
    """메인 함수"""
    tester = WhisperTester()
    await tester.test_microphone()


if __name__ == "__main__":
    asyncio.run(main())

