import os
import time

# RunPod 설정
RUNPOD_IP = "213.192.2.91"
RUNPOD_PORT = "40127"
RUNPOD_API_KEY = "0211"
RUNPOD_MODEL_NAME = "kakaocorp.kanana-nano-2.1b-instruct"

from app.llm.sllm_refiner import refine_text


def main():
    print(f"Target: {RUNPOD_IP}:{RUNPOD_PORT}")
    
    while True:
        user_input = input("입력: ").strip()
        
        # 종료 조건
        if user_input.lower() in ['종료']:
            break
        
        s = time.time()
        result = refine_text(user_input)
        e = time.time()
        
        print(f"원본: {user_input}")
        print(f"교정: {result['text']}")
        print(f"키워드: {', '.join(result['keywords']) if result['keywords'] else '(없음)'}")
        print(f"소요시간: {e-s:.2f}s")

if __name__ == "__main__":
    main()
