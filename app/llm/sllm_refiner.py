import os
import json
import re
import requests
from typing import Optional, Dict, List
from llama_cpp import Llama

_sllm_model: Optional[Llama] = None
_model_loaded = False
_model_load_failed = False

# RunPod API 설정
RUNPOD_IP = "213.192.2.88"
RUNPOD_PORT = "40066"
RUNPOD_API_KEY = "0211"
RUNPOD_MODEL_NAME = "kakaocorp.kanana-nano-2.1b-instruct"

RUNPOD_API_URL = f"http://{RUNPOD_IP}:{RUNPOD_PORT}/v1/chat/completions"
_session = requests.Session()

# 로컬 모델 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "tests", "models", "kanana-nano-2.1b-instruct.Q4_K_M.gguf")
# MODEL_PATH = os.path.join(BASE_DIR, "tests", "models", "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf")

def refine_text(text: str) -> Dict[str, any]:
    """
    텍스트를 교정하고 키워드를 추출합니다.
    모델이 직접 교정과 키워드 추출을 수행합니다.
    """
    # 텍스트가 비어있다면 원본 반환
    if not text or not text.strip():
        return {"text": text, "keywords": []}

    try:
        # 개선된 프롬프트: 교정 + 키워드 추출을 한 번에 수행
        system_prompt = """당신은 금융 상담 전문 AI입니다.

**역할:**
1. 사용자 발화의 오타, 띄어쓰기, 문법 오류를 교정
2. 추임새(음, 그, 저기 등) 제거
3. 금융/카드 관련 핵심 키워드 추출

**키워드 추출 기준:**
- 카드명 (예: 나라사랑카드, 삼성카드, K-패스)
- 금융용어 (예: 리볼빙, 현금서비스, 할부)
- 사용자 의도 (예: 분실, 발급, 해지, 신청, 사용처, 혜택)
- 결제수단 (예: 삼성페이, 카카오페이, 네이버페이)

**주의사항:**
- 금융 전문용어는 정확히 유지 (예: 리볼빙, 일부결제금액이월약정)
- 카드명은 띄어쓰기 없이 (예: "나라사랑카드" O, "나라 사랑 카드" X)
- 키워드는 최대 3개까지만 추출
- 불필요한 일반 단어는 제외 (예: 문의, 안내)

**출력 형식 (JSON만 출력):**
{"text": "교정된문장", "keywords": ["키워드1", "키워드2"]}

**예시:**
입력: "리벌빙 취소 할라는데"
출력: {"text": "리볼빙 취소 하려고 하는데요", "keywords": ["리볼빙", "취소"]}

입력: "나라사랑 카드 분실했어요"
출력: {"text": "나라사랑카드를 분실했습니다", "keywords": ["나라사랑카드", "분실"]}"""

        payload = {
            "model": RUNPOD_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"입력: {text}"}
            ],
            "temperature": 0.1,
            "max_tokens": 256,  # 키워드 포함으로 토큰 증가
            "top_p": 0.9,
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RUNPOD_API_KEY}"
        }
        
        print(f"[RunPod] Sending request to: {RUNPOD_API_URL}")
        response = _session.post(RUNPOD_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"[RunPod] API 오류 ({response.status_code}): {response.text}")
            return {"text": text, "keywords": []}
        
        result = response.json()
        
        try:
            output = result['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError):
            print(f"[RunPod] 응답 구조가 예상과 다릅니다: {result}")
            return {"text": text, "keywords": []}

        try:
            # 마크다운 코드 블록 제거
            output = output.replace("```json", "").replace("```", "")
            
            # JSON 객체 추출 (text와 keywords 포함)
            json_match = re.search(r'\{[^}]*"text"[^}]*"keywords"[^}]*\}', output)
            if not json_match:
                # keywords 없이 text만 있는 경우도 처리
                json_match = re.search(r'\{"text":\s*"[^"]*"\}', output)
            
            if json_match:
                json_str = json_match.group(0)
                print(f"[RunPod] Extracted JSON: {json_str}")
            else:
                json_str = output.strip()
            
            parsed_result = json.loads(json_str)
            refined_text = parsed_result.get("text", text)
            keywords = parsed_result.get("keywords", [])
            
            # 키워드 정제: # 접두사 추가
            if keywords:
                keywords = [f"#{kw}" if not kw.startswith("#") else kw for kw in keywords]
            
            # 유효성 검사
            if refined_text and len(refined_text) > 0:
                return {"text": refined_text, "keywords": keywords}
            else:
                print(f"[RunPod] 교정 결과가 유효하지 않아 원본 반환")
                return {"text": text, "keywords": []}
                
        except json.JSONDecodeError:
            print(f"[RunPod] JSON 파싱 실패, 원본 반환: {output}")
            return {"text": text, "keywords": []}
            
    except requests.exceptions.RequestException as e:
        print(f"[RunPod] 네트워크 오류: {e}")
        return {"text": text, "keywords": []}
    except Exception as e:
        print(f"[RunPod] 처리 중 문제 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"text": text, "keywords": []}



def get_model_status() -> dict:
    return {
        "model_loaded": _model_loaded,
        "model_load_failed": _model_load_failed,
        "model_path": MODEL_PATH,
    }