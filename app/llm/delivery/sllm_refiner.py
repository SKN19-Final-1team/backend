import json
from typing import Dict

# 텍스트 교정에 사용할 모델
MODEL_NAME = "kanana-nano-2.1b-instruct"


def get_model_name() -> str:
    return MODEL_NAME


def refinement_prompt() -> str:
    return """금융 텍스트 교정. 발음 뭉개짐, 띄어쓰기 오류, 맞춤법 오류, 오타 수정. 아래 출력 형식 외 부연설명 절대 금지
[출력 형식]
{
    "original": "입력 텍스트",
    "refined" : "교정 텍스트"
}
"""


def user_message(text: str) -> str:
    return f"입력: {text}"


def refinement_payload(
    text: str,
    temperature: float = 0.1,
    max_tokens: int = 50,
    top_p: float = 0.9
) -> Dict:
    """
    텍스트 교정 요청을 위한 페이로드를 생성합니다.
    """
    system_prompt = refinement_prompt()
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


def parse_refinement_result(llm_output: str, original_text: str) -> Dict[str, any]:
    """
    LLM 출력 결과를 파싱하여 교정된 텍스트를 추출합니다.
    
    Args:
        llm_output: LLM의 원시 출력 텍스트
        original_text: 원본 텍스트 (파싱 실패 시 반환용)
    
    Returns:
        교정 결과 딕셔너리 {"text": str, "keywords": list}
    """
    if not llm_output:
        return {"text": original_text, "keywords": []}
    
    try:
        # JSON 파싱 시도
        result = json.loads(llm_output)
        refined_text = result.get("refined", original_text)
        return {"text": refined_text, "keywords": []}
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 원본 반환
        print(f"[Refiner] JSON 파싱 실패: {llm_output}")
        return {"text": original_text, "keywords": []}