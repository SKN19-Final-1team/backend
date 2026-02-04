import json
import re
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.core.prompt import REFINEMENT_PROMPT

load_dotenv()

VOCAB_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'rag', 'vocab', 'keywords_dict_refine.json'
)

_CORRECTION_MAP_CACHE: Optional[Dict[str, str]] = None

client = AsyncOpenAI(
    base_url=os.getenv("ACW_CORRECT_RUNPOD_URL"),
    api_key=os.getenv("RUNPOD_API_KEY")
)


def load_correction_map() -> Dict[str, str]:
    global _CORRECTION_MAP_CACHE
    
    if _CORRECTION_MAP_CACHE is not None:
        return _CORRECTION_MAP_CACHE

    try:
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _CORRECTION_MAP_CACHE = data.get('correction_map', {})
        else:
            print(f"[Refiner] 단어 사전 파일 없음: {VOCAB_PATH}")
            _CORRECTION_MAP_CACHE = {}
    except Exception as e:
        print(f"[Refiner] correction_map 로드 실패: {e}")
        _CORRECTION_MAP_CACHE = {}
        
    return _CORRECTION_MAP_CACHE


def apply_correction_map(text: str, correction_map: Dict[str, str]) -> str:
    if not text:
        return ""
    
    result = text
    for error, correction in correction_map.items():
        if error in result:
            result = result.replace(error, correction)
    return result


def extract_json_content(text: str) -> Optional[str]:
    if not text:
        return None
        
    clean_text = text.replace("```json", "").replace("```", "").strip()
    
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        start_idx = clean_text.find(start_char)
        if start_idx == -1:
            continue
            
        bracket_count = 0
        for i in range(start_idx, len(clean_text)):
            char = clean_text[i]
            if char == start_char:
                bracket_count += 1
            elif char == end_char:
                bracket_count -= 1
                
            if bracket_count == 0:
                return clean_text[start_idx : i+1]
                
    return None


async def refine_diarized_batch(utterances: List[Dict]) -> List[Dict]:
    if not utterances:
        return []
    
    correction_map = load_correction_map()
    
    for utt in utterances:
        utt['_corrected'] = apply_correction_map(utt.get('message', ''), correction_map)
    
    input_lines = []
    for i, utt in enumerate(utterances, 1):
        speaker_kr = "상담원" if utt.get("speaker") == "agent" else "고객"
        
        input_lines.append(f"[{i}] ({speaker_kr}) {utt['_corrected']}")
    
    user_content = "다음 발화들을 교정하세요:\n\n" + "\n".join(input_lines)

    result_json_str = None
    try:
        response = await client.chat.completions.create(
            model="kakaocorp/kanana-1.5-8b-instruct-2505",
            messages=[
                {"role": "system", "content": REFINEMENT_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=2048,
            stop=["[|", "[|end|]", "[|user|]", "\n"] 
        )
        result_content = response.choices[0].message.content
        result_json_str = extract_json_content(result_content)
        
    except Exception as e:
        print(f"[Refiner] sLLM 호출 실패: {e}")
    
    refined_texts = [utt['_corrected'] for utt in utterances]
    
    if result_json_str:
        try:
            results = json.loads(result_json_str)
            
            for i in range(len(utterances)):
                case_id = i + 1
                found = next((r for r in results if r.get('id') == case_id), None)
                
                if found and found.get('refined'):
                    refined_texts[i] = found['refined']
                    
        except json.JSONDecodeError as e:
            print(f"[Refiner] JSON 파싱 에러: {e}")
        except Exception as e:
            print(f"[Refiner] 결과 처리 중 에러: {e}")
            
    final_result = []
    for utt, refined_msg in zip(utterances, refined_texts):
        final_result.append({
            "speaker": utt.get("speaker", "unknown"),
            "message": refined_msg
        })
    
    return final_result