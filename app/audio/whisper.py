import threading
import queue
import io
import asyncio
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# class WhisperService:
#     def __init__(self, api_key: str = None):
#         # API 클라이언트 초기화
#         self.client = OpenAI(api_key=api_key)
#         self.queue = queue.Queue()
#         self.running = False
#         self.thread = None
#         self.loop = None
#         self.callback = None

#     def start(self, callback, loop: asyncio.AbstractEventLoop):
#         # 백그라운드 작업
#         self.callback = callback  # 결과가 나오면 호출할 함수
#         self.loop = loop          # 메인 스레드의 이벤트 루프
#         self.running = True
#         self.thread = threading.Thread(target=self._worker, daemon=True)
#         self.thread.start()

#     def stop(self):
#         # 작업 종료 및 자원 정리
#         print("작업 스레드 종료")
#         self.running = False
#         self.queue.put(None)  # 종료 신호 전송
#         if self.thread:
#             self.thread.join()

#     def add_audio(self, audio_data: bytes):
#         # 오디오 데이터 추가
#         self.queue.put(audio_data)

#     def _worker(self):
#         print("작업 스레드 시작")

#         HALLUCINATION_KEYWORDS = [
#             "시청해주셔서", "시청해 주셔서", "구독과 좋아요", 
#             "재택 플러스", "MBC", "뉴스", "투데이", "먹방", "영상편집", "영상", "편집", "진심으로"
#         ]
        
#         while self.running:
#             try:
#                 audio_data = self.queue.get()
#                 if audio_data is None: 
#                     break

#                 # 오디오 처리
#                 audio_file = io.BytesIO(audio_data)
#                 audio_file.name = "audio.wav"

#                 # OpenAI API 호출
#                 transcript = self.client.audio.transcriptions.create(
#                     model="whisper-1",
#                     file=audio_file,
#                     language="ko",
#                 )
#                 text = transcript.text.strip()

#                 # 할루시네이션 방지
#                 if not text:
#                     self.queue.task_done()
#                     continue

#                 if any(keyword in text for keyword in HALLUCINATION_KEYWORDS):
#                     self.queue.task_done()
#                     continue

#                 # 비동기 콜백 함수를 메인 스케줄러에 등록
#                 if text and self.callback:
#                     asyncio.run_coroutine_threadsafe(
#                         self.callback(text), 
#                         self.loop
#                     )

#                 self.queue.task_done()

#             except Exception as e:
#                 print(f"작업 스레드 오류 발생: {e}")
#                 self.queue.task_done()
#                 continue
        
#         print("작업 스레드 종료")


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

    def _post_process(self, text: str) -> str:
        """
        [텍스트 교정 함수]
        STT로 전사된 텍스트를 sLLM으로 교정합니다.
        
        Args:
            text: Whisper API로부터 받은 원본 텍스트
        
        Returns:
            교정된 텍스트
        """
        try:
            # sLLM 교정 로직 임포트
            from app.llm.delivery.sllm_refiner import refinement_payload, parse_refinement_result
            from app.utils.runpod_connector import call_runpod
            
            # 1. Payload 생성
            payload = refinement_payload(text)
            
            # 2. LLM 호출
            llm_output = call_runpod(payload)
            
            # 3. 결과 파싱
            result = parse_refinement_result(llm_output, text)
            corrected_text = result["text"]
            
            print(f"[Whisper] 교정 전: {text}")
            print(f"[Whisper] 교정 후: {corrected_text}")
            
            return corrected_text
            
        except Exception as e:
            print(f"[Whisper] 텍스트 교정 실패: {e}")
            # 교정 실패 시 원본 반환
            return text

    def _worker(self):
        print("작업 스레드 시작")

        # [변경 1] 도메인 특화 프롬프트 정의
        # 카드사 상담 데이터의 인식률을 높이기 위해 핵심 용어와 문체를 미리 정의합니다.
        DOMAIN_PROMPT = (
            "이것은 카드사 고객 센터 상담 내용입니다. "
            "리볼빙, 선결제, 결제일, 한도 상향, 할부 수수료, 연체, 카드 론, "
            "단기카드대출, 장기카드대출, 비밀번호 초기화, 상담원 연결 등의 용어가 포함되어 있습니다. "
            "문장은 명확한 구어체로 작성해 주세요."
        )

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

                # [변경 2] OpenAI API 호출 시 prompt 파라미터 추가
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko",
                    prompt=DOMAIN_PROMPT  # <--- 프롬프트 주입
                )
                
                raw_text = transcript.text.strip()

                # [변경 3] 텍스트 교정 함수 연결
                # 원본 텍스트를 교정 함수로 전달하여 처리
                corrected_text = self._post_process(raw_text)

                # 할루시네이션 방지 (교정된 텍스트 기준)
                if not corrected_text:
                    self.queue.task_done()
                    continue

                if any(keyword in corrected_text for keyword in HALLUCINATION_KEYWORDS):
                    self.queue.task_done()
                    continue

                # 비동기 콜백 함수를 메인 스케줄러에 등록
                if corrected_text and self.callback:
                    asyncio.run_coroutine_threadsafe(
                        self.callback(corrected_text), 
                        self.loop
                    )

                self.queue.task_done()

            except Exception as e:
                print(f"작업 스레드 오류 발생: {e}")
                self.queue.task_done()
                continue
        
        print("작업 스레드 종료")