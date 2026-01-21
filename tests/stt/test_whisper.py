import asyncio
import websockets
import sounddevice as sd
import io
import wave
import webrtcvad
import time
import collections
import queue
import threading

# 설정
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
VAD_MODE = 3                  # 0~3 범위, 3이 가장 민감
PADDING_DURATION_MS = 300     # 음성 전후에 추가할 패딩(침묵) 시간
SILENT_CHUNKS_THRESHOLD = 20  # 이 개수의 연속된 침묵 프레임이 감지되면 종료

class Audio(object):
    """오디오 스트림을 처리하는 클래스"""
    def __init__(self, loop, async_queue):
        self.buffer_queue = queue.Queue()
        self.recording = True
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.num_voiced_frames = 0
        self.num_silent_frames = 0
        self.frames = []
        self.loop = loop
        self.async_queue = async_queue

    def frame_generator(self):
        """오디오 프레임 생성기"""
        while self.recording:
            try:
                data = self.buffer_queue.get(block=True, timeout=1)
                yield data
            except queue.Empty:
                break

    def callback_audio(self, indata, frames, time, status):
        """오디오 콜백 함수"""
        if status:
            print(f"오디오 상태: {status}")
        self.buffer_queue.put(bytes(indata))

    def vad_collector(self, threshold=SILENT_CHUNKS_THRESHOLD):
        """VAD를 통해 음성 구간 검출"""
        frame_duration_ms = FRAME_DURATION_MS
        frame_size = int(SAMPLE_RATE * frame_duration_ms / 1000)
        ring_buffer = collections.deque(maxlen=threshold)
        
        triggered = False
        for frame in self.frame_generator():
            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
            
            if not triggered:
                if is_speech:
                    triggered = True
                    print("음성 감지됨 - 녹음 시작")
                    for f in ring_buffer:
                        self.frames.append(f)
                    self.frames.append(frame)
                    self.num_voiced_frames += 1
                    ring_buffer.clear()
                else:
                    ring_buffer.append(frame)
            else:
                if is_speech:
                    self.frames.append(frame)
                    self.num_voiced_frames += 1
                    self.num_silent_frames = 0
                else:
                    self.frames.append(frame)
                    self.num_silent_frames += 1
                    
                    # 침묵이 일정 기간 계속되면 음성 세그먼트 종료
                    if self.num_silent_frames > threshold:
                        triggered = False
                        print(f"음성 세그먼트 종료")
                        
                        if self.num_voiced_frames > 0:
                            # 오디오 데이터를 메인 스레드로 안전하게 전달
                            audio_data = b''.join(self.frames)
                            if len(audio_data) > 1000:  # 너무 짧은 오디오는 무시
                                audio_buffer = io.BytesIO()
                                with wave.open(audio_buffer, 'wb') as wav_file:
                                    wav_file.setnchannels(CHANNELS)
                                    wav_file.setsampwidth(2)
                                    wav_file.setframerate(SAMPLE_RATE)
                                    wav_file.writeframes(audio_data)
                                
                                # 메인 이벤트 루프의 큐에 안전하게 데이터 추가
                                self.loop.call_soon_threadsafe(
                                    self.async_queue.put_nowait, 
                                    audio_buffer.getvalue()
                                )
                        
                        self.frames = []
                        self.num_voiced_frames = 0
                        self.num_silent_frames = 0
                        ring_buffer.clear()

async def ws_transcribe(duration=60):
    uri = "ws://127.0.0.1:8000/api/v1/ws"
    
    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue()
    audio = Audio(loop, audio_queue)

    frame_size = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
    stream = sd.InputStream(callback=audio.callback_audio, blocksize=frame_size, dtype='int16', channels=1, samplerate=SAMPLE_RATE)
    vad_thread = threading.Thread(target=audio.vad_collector)

    async with websockets.connect(uri) as websocket:
        print("웹소켓 연결 완료")

        with stream:
            vad_thread.start()
            start_time = time.time()
            
            # 오디오 전송
            while time.time() - start_time < duration:
                try:
                    audio_data = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                    await websocket.send(audio_data)
                    print(".", end="", flush=True)
                except asyncio.TimeoutError:
                    pass

        audio.recording = False
        vad_thread.join()

if __name__ == "__main__":
    asyncio.run(ws_transcribe())