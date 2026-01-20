import time
import gc
from llama_cpp import Llama

class ModelEvaluator:
    def __init__(self, model_path, model_name, n_gpu_layers=-1):
        self.model_path = model_path
        self.model_name = model_name
        self.n_gpu_layers = n_gpu_layers
        self.llm = None

    def load_model(self):
        print(f"모델 로드 중...: {self.model_name}")
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers, # -1: 모든 레이어 GPU 할당
                n_ctx=4096,                     # 컨텍스트 윈도우
                verbose=False
            )       
            print("✅ 모델 로드 완료")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")

    def unload_model(self):
        if self.llm:
            del self.llm
            self.llm = None
            gc.collect() # 가비지 컬렉터 강제 실행
            print(f"🗑️ 모델 해제 완료: {self.model_name}\n")

    def generate_and_measure(self, system_prompt, user_input):
        if not self.llm:
            return None

        # 1. 기본 시도: System Role과 User Role을 분리해서 전송
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        start_time = time.time()
        first_token_time = None
        token_count = 0
        response_text = ""

        try:
            # 스트리밍 방식으로 생성 시도
            stream = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.7,
                stream=True
            )
        except ValueError:
            # [에러 해결 핵심] System role not supported 에러 발생 시
            # 시스템 프롬프트를 유저 프롬프트 앞단에 합쳐서(Merge) 재시도
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\nAnswer the user's input based on the instructions above.\n\nUser Input: {user_input}"}
            ]
            stream = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.7,
                stream=True
            )

        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                content = delta['content']
                if not first_token_time:
                    first_token_time = time.time() # 첫 토큰 도착 시간 기록
                
                response_text += content
                token_count += 1

        end_time = time.time()
        
        # 지표 계산
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_time
        generation_time = end_time - first_token_time if first_token_time else 0
        tps = token_count / generation_time if generation_time > 0 else 0

        return {
            "model": self.model_name,
            "input": user_input,
            "response": response_text.strip(),
            "ttft": round(ttft, 4),
            "tps": round(tps, 2),
            "total_tokens": token_count
        }