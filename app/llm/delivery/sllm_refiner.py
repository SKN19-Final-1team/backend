"""
Enhanced Text Refinement Module
STT 교정 및 화자분리 대화 교정 모듈

주요 기능:
1. refine_diarized_batch: 화자분리된 발화 리스트를 배치로 교정 (메인 기능)
2. correction_map: 단순 치환 교정 (JSON 로드)
3. sLLM interaction: RunPod API 연동

Author: Antigravity
"""

import json
import re
import os
from typing import Dict, List, Optional
from app.utils.runpod_connector import call_runpod

# Configuration
MODEL_NAME = "kanana-1.5-8b-instruct-2505-q4_k_m.gguf"
VOCAB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'rag', 'vocab', 'keywords_dict_refine.json')


# ==========================================
# 1. Helper Functions
# ==========================================

def load_correction_map() -> Dict[str, str]:
    """
    keywords_dict_refine.json에서 correction_map을 로드합니다.
    """
    try:
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('correction_map', {})
        else:
            print(f"[Refiner] 단어 사전 파일 없음: {VOCAB_PATH}")
            return {}
    except Exception as e:
        print(f"[Refiner] correction_map 로드 실패: {e}")
        return {}


def apply_correction_map(text: str, correction_map: Dict[str, str]) -> str:
    """
    텍스트에 단순 치환 교정(correction_map)을 적용합니다.
    """
    if not text:
        return ""
    
    result = text
    for error, correction in correction_map.items():
        if error in result:
            result = result.replace(error, correction)
    return result


def extract_json_content(text: str) -> Optional[str]:
    """
    LLM 출력에서 JSON 문자열을 추출합니다. (Code block 제거 및 괄호 매칭)
    """
    if not text:
        return None
        
    # 1. 마크다운 코드 블록 제거
    clean_text = text.replace("```json", "").replace("```", "").strip()
    
    # 2. 대괄호([]) 또는 중괄호({}) 균형 매칭
    # 배열([])이 우선순위 (배치 처리 때문)
    for bracket_pair in [('[', ']'), ('{', '}')]:
        start_char, end_char = bracket_pair
        
        bracket_count = 0
        start_idx = -1
        
        for i, char in enumerate(clean_text):
            if char == start_char:
                if start_idx == -1:
                    start_idx = i
                bracket_count += 1
            elif char == end_char:
                bracket_count -= 1
                if bracket_count == 0 and start_idx != -1:
                    return clean_text[start_idx : i+1]
                    
    return None


# ==========================================
# 2. Prompts
# ==========================================

def get_batch_refinement_prompt() -> str:
    """
    배치 교정용 시스템 프롬프트
    """
    return """당신은 금융 상담 STT 교정 AI입니다.

**역할:** 각 발화의 STT 오류를 교정합니다.

**교정 대상:**
1. 발음 오인식: "연예비"→"연회비", "바우저"→"바우처", "환도"→"한도"
2. 외국어 할루시네이션: 삭제
3. 불필요한 삽입어: 문맥에 맞지 않으면 삭제

**금지:**
- 원문 의미 변경
- 과도한 문체 변환

**출력 형식 (JSON 배열만 출력):**
[
  {"id": 1, "refined": "교정된 문장1"},
  {"id": 2, "refined": "교정된 문장2"}
]
"""


# ==========================================
# 3. Main Logic
# ==========================================

def refine_diarized_batch(utterances: List[Dict]) -> List[Dict]:
    """
    화자분리된 발화 리스트를 배치로 sLLM 교정합니다.
    
    Args:
        utterances: [{"speaker": "agent", "message": "..."}] 형태의 리스트
    
    Returns:
        교정된 발화 리스트 (원본 구조 유지)
    """
    if not utterances:
        return []
    
    # 1. correction_map 로드 및 1차 교정
    correction_map = load_correction_map()
    
    # 내부 처리를 위해 _corrected 필드 사용
    for utt in utterances:
        utt['_corrected'] = apply_correction_map(utt.get('message', ''), correction_map)
    
    # 2. 배치 입력 구성 (Prompt Engineering)
    input_lines = []
    for i, utt in enumerate(utterances, 1):
        speaker_kr = "상담원" if utt.get("speaker") == "agent" else "고객"
        # ID를 부여하여 매핑 정확도 향상
        input_lines.append(f"[{i}] ({speaker_kr}) {utt['_corrected']}")
    
    user_content = "다음 발화들을 교정하세요:\n\n" + "\n".join(input_lines)
    
    # 3. sLLM 호출
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": get_batch_refinement_prompt()},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
        "top_p": 0.9,
        "stream": False
    }
    
    try:
        output = call_runpod(payload)
    except Exception as e:
        print(f"[Refiner] sLLM 호출 실패: {e}")
        output = None
    
    # 4. 결과 파싱 및 반영
    # 기본값은 1차 교정된 텍스트
    refined_texts = [utt['_corrected'] for utt in utterances]
    
    if output:
        try:
            json_str = extract_json_content(output)
            if json_str:
                results = json.loads(json_str)
                
                # ID 기반으로 결과 매핑
                for i in range(len(utterances)):
                    case_id = i + 1
                    # 결과 리스트에서 해당 ID 찾기
                    found = next((r for r in results if r.get('id') == case_id), None)
                    
                    if found and found.get('refined'):
                         refined_texts[i] = found['refined']
            else:
                 print(f"[Refiner] JSON 추출 실패. Raw: {output[:100]}...")
                 
        except json.JSONDecodeError as e:
            print(f"[Refiner] JSON 파싱 에러: {e}")
        except Exception as e:
            print(f"[Refiner] 결과 처리 중 에러: {e}")
            
    # 5. 최종 결과 생성
    result = []
    for utt, refined_msg in zip(utterances, refined_texts):
        result.append({
            "speaker": utt.get("speaker", "unknown"),
            "message": refined_msg
        })
    
    return result


# ==========================================
# 4. Legacy / Utility (Optional)
# ==========================================

def get_model_name() -> str:
    return MODEL_NAME

# 필요한 경우 싱글턴 교정 함수 등을 여기에 남겨둘 수 있습니다.
# 현재는 서비스 로직인 refine_diarized_batch에 집중하여 정리했습니다.