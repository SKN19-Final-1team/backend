import threading
import queue
import io
import asyncio
import re
import time
from openai import OpenAI
from dotenv import load_dotenv
from app.core.prompt import WHISPER_PROMPT

load_dotenv()

# ── 할루시네이션 필터 ──────────────────────────────────────────
HALLUCINATION_KEYWORDS = [
    # YouTube / 방송 패턴
    "시청해주셔서", "시청해 주셔서", "구독과 좋아요", "구독 좋아요",
    "재택 플러스", "MBC", "뉴스", "투데이", "먹방",
    "영상편집", "영상", "편집", "진심으로",
    # 일반적 할루시네이션
    "오늘도 맛있게", "잘 먹었습니다", "감사합니다 여러분",
    "다음 시간에", "그럼 다음", "시간에 만나요",
    "자막 제공", "자막 by", "한국어 자막",
]

# 너무 짧은 응답 (1-2자) 필터
MIN_TEXT_LENGTH = 2

# 반복 패턴 필터 (같은 단어/문장이 3회 이상)
REPEAT_PATTERN = re.compile(r'(.{2,}?)\1{2,}')


def is_hallucination(text: str) -> bool:
    """Whisper 할루시네이션 여부 판정"""
    if not text or not text.strip():
        return True
    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return True
    if any(kw in text for kw in HALLUCINATION_KEYWORDS):
        return True
    if REPEAT_PATTERN.search(text):
        return True
    return False


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
                    prompt=WHISPER_PROMPT,
                )
                text = transcript.text.strip()

                # 할루시네이션 방지
                if is_hallucination(text):
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