import json
from typing import Dict

# 텍스트 교정에 사용할 모델
MODEL_NAME = "kanana-nano-2.1b-instruct"


def get_model_name() -> str:
    return MODEL_NAME


def refinement_prompt() -> str:
    return """당신은 금융권 상담 전문 교정 전문가입니다.

[역할]
STT(음성-텍스트 변환)로 전사된 상담 내용의 문법적 오류를 수정하고, 문맥상 어색하거나 이상한 표현을 자연스럽게 다듬습니다.

[교정 원칙]
1. **문법 교정**: 발음 오류, 띄어쓰기, 맞춤법, 조사 오류를 정확히 수정합니다.
2. **문맥 개선**: 문맥상 어색하거나 부자연스러운 표현을 자연스러운 구어체로 바꿉니다.
3. **의미 보존**: 원문의 의도와 내용을 절대 변경하지 않습니다.
4. **상담 흐름 유지**: 고객-상담사 간 대화의 자연스러운 흐름을 유지합니다.

[교정 예시]
- "테니카드로 결제해줘" → "테디카드로 결제해주세요"
- "배움카드 신청하고 싶어요" → "내일배움카드를 신청하고 싶어요"
- "나라 사랑 카드" → "나라사랑카드"
- "할부로 나눠서 결제 할수있나요" → "할부로 나눠서 결제할 수 있나요"
- "리볼빙이 뭐에요 그게" → "리볼빙이 뭔가요?"

[금융 용어 참고]
- 카드상품명: 정확한 띄어쓰기 유지 (예: "내일배움카드", "나라사랑카드")
- 금융용어: 리볼빙, 선결제, 할부, 연체, 한도, 수수료, 카드론 등

[출력]
교정된 상담 전문만 출력하세요. 다른 설명이나 주석을 추가하지 마세요.
"""


def user_message(text: str) -> str:
    return f"[상담 전문]\n{text}"


def refinement_payload(
    text: str,
    temperature: float = 0.4,
    max_tokens: int = 1024,
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
    
    # JSON 파싱 없이 LLM 출력을 그대로 교정된 텍스트로 사용
    return {"text": llm_output.strip(), "keywords": []}