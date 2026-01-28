import json
from typing import Dict, List

MODEL_NAME = "kanana-nano-2.1b-instruct"

def get_model_name() -> str:
    return MODEL_NAME

def masking_prompt() -> str:
    return """개인정보 마스킹. 입력된 텍스트에서 아래 정보를 찾아 마스킹 처리. 부연설명 절대 금지
1. 인명(3글자) -> [고객명]
2. 전화번호(연속된 숫자) -> [전화번호]
3. 카드번호(연속된 숫자) -> [카드번호]
4. 계좌번호(연속된 숫자) -> [계좌번호]

[출력 형식]
{
    "original": "입력 텍스트",
    "masked": "마스킹된 텍스트",
    "detected_info": ["감지된 정보1", "감지된 정보2"]
}
"""

def user_message(text: str) -> str:
    return f"입력: {text}"

def masking_payload(
    text: str,
    temperature: float = 0.1,
    max_tokens: int = 200,
    top_p: float = 0.9
) -> Dict:
    """
    개인정보 마스킹 요청을 위한 페이로드를 생성합니다.
    """
    system_prompt = masking_prompt()
    user_msg = user_message(text)
    
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False
    }

def parse_masking_result(llm_output: str, original_text: str) -> Dict[str, any]:
    """
    LLM 출력 결과를 파싱하여 마스킹된 텍스트를 추출합니다.
    """
    if not llm_output:
        return {"text": original_text, "info": []}
    
    try:
        # JSON 파싱 시도
        result = json.loads(llm_output)
        masked_text = result.get("masked", original_text)
        detected_info = result.get("detected_info", [])
        return {"text": masked_text, "info": detected_info}
    except json.JSONDecodeError:
        print(f"[Masker] JSON 파싱 실패: {llm_output}")
        # 파싱 실패 시 원본 반환 (안전하게)
        return {"text": original_text, "info": []}
