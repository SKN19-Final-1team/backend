import threading
import queue
import io
import asyncio
import ahocorasick
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

KEYWORD_MAP = {
    "카드 분실": "분실 신고 접수",
    "분실": "분실 신고 접수",
    "한도 조회": "한도 상향 안내",
    "결제일": "결제일 변경 안내",
    "비밀번호": "비밀번호 재설정",
    "연체": "분할 납부 안내"
}


class WhisperService:
    def __init__(self, api_key: str = None):
        # API 클라이언트 초기화
        self.client = OpenAI(api_key=api_key)
        self.queue = queue.Queue()
        self.running = False
        self.thread = None
        self.loop = None
        self.callback = None

        # ahocorasick 오토마톤 초기화
        self.automaton = ahocorasick.Automaton()
        for key in KEYWORD_MAP.keys():
            self.automaton.add_word(key, key)
        self.automaton.make_automaton()

    def start(self, callback, loop: asyncio.AbstractEventLoop):
        # 백그라운드 작업
        self.callback = callback  # 결과가 나오면 호출할 함수
        self.loop = loop          # 메인 스레드의 이벤트 루프
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        # 작업 종료 및 자원 정리
        self.running = False
        self.queue.put(None)  # 종료 신호 전송
        if self.thread:
            self.thread.join()

    def add_audio(self, audio_data: bytes):
        # 오디오 데이터 추가
        self.queue.put(audio_data)

    def _find_keyword(self, text):
        for end_index, found_keyword in self.automaton.iter(text):
            next_keyword = KEYWORD_MAP.get(found_keyword)
            return {
                "current": found_keyword,
                "next": next_keyword
            }
        return None

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
                    prompt="이것은 카드사 상담 대화입니다. 이 상황을 고려하지만 문장을 들리는 대로 적으세요"
                )
                text = transcript.text.strip()

                keyword_info = self._find_keyword(text)

                # 비동기 콜백 함수를 메인 스케줄러에 등록
                if text and self.callback:
                    asyncio.run_coroutine_threadsafe(
                        self.callback(text, keyword_info), 
                        self.loop
                    )

                self.queue.task_done()

            except Exception as e:
                print(f"작업 스레드 오류 발생: {e}")
                self.queue.task_done()
                continue
        
        print("작업 스레드 종료")