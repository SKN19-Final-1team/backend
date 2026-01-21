import os
import json
import re
import requests
from typing import Optional, Dict, List
from llama_cpp import Llama

# 단어사전
from app.rag.vocab.rules import ACTION_SYNONYMS, CARD_NAME_SYNONYMS, PAYMENT_SYNONYMS, STOPWORDS, WEAK_INTENT_SYNONYMS

_sllm_model: Optional[Llama] = None
_model_loaded = False
_model_load_failed = False

# RunPod API 설정
RUNPOD_IP = "213.192.2.91"
RUNPOD_PORT = "40127"
RUNPOD_API_KEY = "0211"
RUNPOD_MODEL_NAME = "kakaocorp.kanana-nano-2.1b-instruct"

RUNPOD_API_URL = f"http://{RUNPOD_IP}:{RUNPOD_PORT}/v1/chat/completions"
_session = requests.Session()

# 로컬 모델 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# MODEL_PATH = os.path.join(BASE_DIR, "tests", "models", "Llama-3-Kor-BCCard-Finance-8B.Q4_K_M.gguf")
MODEL_PATH = os.path.join(BASE_DIR, "tests", "models", "kanana-nano-2.1b-instruct.Q4_K_M.gguf")
# MODEL_PATH = os.path.join(BASE_DIR, "tests", "models", "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf")

def _load_sllm_model() -> Optional[Llama]:
    global _sllm_model, _model_loaded, _model_load_failed
    
    if _model_loaded:
        return _sllm_model
    
    if _model_load_failed:
        return None
    
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"[sLLM] 모델 파일을 찾을 수 없음: {MODEL_PATH}")
            _model_load_failed = True
            return None
        
        _sllm_model = Llama(
            model_path=MODEL_PATH,
            n_ctx=512,
            n_threads=4,
            n_gpu_layers=-1,
            verbose=False,
        )
        
        _model_loaded = True
        return _sllm_model
        
    except Exception as e:
        print(f"[sLLM] 모델 로드 실패: {e}")
        _model_load_failed = True
        return None



def _extract_keywords_from_text(text: str) -> List[str]:
    """
    교정된 텍스트에서 단어사전을 활용하여 핵심 키워드를 추출합니다.
    발화자의 의도를 파악할 수 있는 키워드를 우선적으로 추출합니다.
    """
    if not text:
        return []
    
    keywords = []
    text_lower = text.lower()
    
    # 1. ACTION_SYNONYMS에서 키워드 추출 (사용자 행동 의도)
    for canonical, synonyms in ACTION_SYNONYMS.items():
        # canonical 자체가 텍스트에 포함되어 있는지 확인
        if canonical in text or canonical.lower() in text_lower:
            keywords.append(canonical)
            continue
        
        # synonyms 중 하나라도 포함되어 있으면 canonical을 키워드로 추가
        for synonym in synonyms:
            if synonym and (synonym in text or synonym.lower() in text_lower):
                keywords.append(canonical)
                break
    
    # 2. CARD_NAME_SYNONYMS에서 카드명 추출
    for canonical, synonyms in CARD_NAME_SYNONYMS.items():
        if canonical in text or canonical.lower() in text_lower:
            keywords.append(canonical)
            continue
        
        for synonym in synonyms:
            if synonym and (synonym in text or synonym.lower() in text_lower):
                keywords.append(canonical)
                break
    
    # 3. PAYMENT_SYNONYMS에서 결제 수단 추출
    for canonical, synonyms in PAYMENT_SYNONYMS.items():
        if canonical in text or canonical.lower() in text_lower:
            keywords.append(canonical)
            continue
        
        for synonym in synonyms:
            if synonym and (synonym in text or synonym.lower() in text_lower):
                keywords.append(canonical)
                break
    
    # 4. STOPWORDS 제거
    keywords = [kw for kw in keywords if kw.lower() not in STOPWORDS]
    
    # 5. 중복 제거 및 # 접두사 추가
    unique_keywords = []
    seen = set()
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(f"#{kw}" if not kw.startswith("#") else kw)
    
    return unique_keywords


def refine_text(text: str) -> Dict[str, any]:
    # 텍스트가 비어있다면 원본 반환
    if not text or not text.strip():
        return {"text": text, "keywords": []}

    try:
        system_prompt = """금융 텍스트 교정 전문가. 오타/띄어쓰기 교정, 추임새 제거.
출력 형식(JSON만): {"text": "교정된문장"}"""

        payload = {
            "model": RUNPOD_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"입력: {text}"}
            ],
            "temperature": 0.1,
            "max_tokens": 128,
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
            
            # 정규식으로 첫 번째 JSON 객체 추출
            json_match = re.search(r'\{"text":\s*"[^"]*"\}', output)
            if json_match:
                json_str = json_match.group(0)
                print(f"[RunPod] Extracted JSON: {json_str}")
            else:
                json_str = output.strip()
                print(f"[RunPod] Using full output: {json_str}")
            
            parsed_result = json.loads(json_str)
            refined_text = parsed_result.get("text", text)
            
            # 유효성 검사
            if refined_text and len(refined_text) > 0:
                # 교정된 텍스트에서 키워드 추출
                keywords = _extract_keywords_from_text(refined_text)
                print(f"[RunPod] Extracted keywords: {keywords}")
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
        return {"text": text, "keywords": []}


def get_model_status() -> dict:
    return {
        "model_loaded": _model_loaded,
        "model_load_failed": _model_load_failed,
        "model_path": MODEL_PATH,
    }