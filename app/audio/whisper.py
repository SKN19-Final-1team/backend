import threading
import queue
import io
import asyncio
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class WhisperService:
    def __init__(self, api_key: str = None):
        # API 클라이언트 초기화
        self.client = OpenAI(api_key=api_key)
        self.queue = queue.Queue()
        self.running = False
        self.thread = None
        self.loop = None
        self.callback = None

    def start(self, callback, loop: asyncio.AbstractEventLoop):
        # 백그라운드 작업
        self.callback = callback  # 결과가 나오면 호출할 함수
        self.loop = loop          # 메인 스레드의 이벤트 루프
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        # 작업 종료 및 자원 정리
        print("작업 스레드 종료")
        self.running = False
        self.queue.put(None)  # 종료 신호 전송
        if self.thread:
            self.thread.join()

    def add_audio(self, audio_data: bytes):
        # 오디오 데이터 추가
        self.queue.put(audio_data)

    def _worker(self):
        print("작업 스레드 시작")

        HALLUCINATION_KEYWORDS = [
            "시청해주셔서", "시청해 주셔서", "구독과 좋아요", 
            "재택 플러스", "MBC", "뉴스", "투데이", "먹방", "영상편집", "영상", "편집", "진심으로"
        ]
        
        while self.running:
            try:
                audio_data = self.queue.get()
                if audio_data is None: 
                    break

                # 오디오 처리
                audio_file = io.BytesIO(audio_data)
                audio_file.name = "audio.wav"

                # OpenAI API 호출
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko",
                )
                text = transcript.text.strip()

                # 할루시네이션 방지
                if not text:
                    self.queue.task_done()
                    continue

                if any(keyword in text for keyword in HALLUCINATION_KEYWORDS):
                    if len(text) < 20: 
                        self.queue.task_done()
                        continue

                # 비동기 콜백 함수를 메인 스케줄러에 등록
                if text and self.callback:
                    asyncio.run_coroutine_threadsafe(
                        self.callback(text), 
                        self.loop
                    )

                self.queue.task_done()

            except Exception as e:
                print(f"작업 스레드 오류 발생: {e}")
                self.queue.task_done()
                continue
        
        print("작업 스레드 종료")