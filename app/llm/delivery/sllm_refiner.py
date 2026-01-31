"""
Enhanced Text Refinement Module

개선사항:
1. Context-aware prompting (형태소 분석 및 단어 매칭 결과 활용)
2. Domain-specific corrections (금융권 STT 오류 패턴)
3. Improved JSON parsing with multiple fallback strategies
4. Conversation-level refinement support
"""

import json
import re
from typing import Dict, List, Tuple, Optional

# 텍스트 교정에 사용할 모델
MODEL_NAME = "kanana-nano-2.1b-instruct"


def get_model_name() -> str:
    return MODEL_NAME


def refinement_prompt() -> str:
    """
    기본 교정 프롬프트 (하위 호환성 유지)
    """
    return """금융 텍스트 교정. 발음 뭉개짐, 띄어쓰기 오류, 맞춤법 오류, 오타 수정. 아래 출력 형식 외 부연설명 절대 금지
[출력 형식]
{
    "original": "입력 텍스트",
    "refined" : "교정 텍스트"
}
"""


def refinement_prompt_with_context(
    text: str,
    known_entities: Optional[List[str]] = None,
    morphology_hints: Optional[List[Tuple[str, str]]] = None
) -> str:
    """
    컨텍스트 인식 교정 프롬프트 생성
    
    Args:
        text: 교정할 텍스트
        known_entities: 감지된 카드상품명 등 고유명사 리스트
        morphology_hints: 형태소 분석 결과 [(형태소, 품사), ...]
    
    Returns:
        컨텍스트가 포함된 프롬프트
    """
    # 기본 프롬프트
    prompt = """당신은 금융 상담 전문 텍스트 교정 AI입니다.

**역할:**
STT(음성인식)로 전사된 텍스트의 오류를 교정합니다.

**주요 교정 대상:**
1. STT 할루시네이션 (잘못 들은 단어)
   - "연예비" → "연회비"
   - "이길영업일" → "익일 영업일"
   - "나라사람카드" → "나라사랑카드"
   - "발송소리가" → "발송처리가"
   - "바우저" → "바우처"
   - "하나 낸" → "하나은행"
   - "출근할까요" → "출금할까요"
   - "통화주신" → "전화주신"

2. 발음 뭉개짐
   - "대꺼든요" → "됐거든요"
   - "할라는데" → "하려고 하는데"
   - "신청할려구요" → "신청하려고요"

3. 띄어쓰기 오류 (명백한 오류만)
   - "잠시 만" → "잠시만"
   - "무엇인 가요" → "무엇인가요"

**절대 금지 사항:**
1. 정확한 띄어쓰기를 임의로 변경 금지
   - "네, 손님" → "네, 손 님" (X)
   - "무엇인가요" → "무엇인 가요" (X)

2. 단어를 다른 단어로 확장/대체 금지
   - 카드상품명을 다른 카드명으로 바꾸지 말 것
   - 원문에 없는 단어 추가 금지

3. 문맥상 자연스러운 표현은 그대로 유지
   - 의미가 명확하면 원문 유지
   - 과도한 교정 절대 금지
"""
    
    # 알려진 고유명사 추가
    if known_entities and len(known_entities) > 0:
        entities_str = ", ".join(known_entities)
        prompt += f"""
**이 대화에서 감지된 카드상품명:**
{entities_str}

→ 위 상품명들은 정확히 보존하세요. 유사한 오타가 있으면 위 이름으로 교정하세요.
"""
    
    # 형태소 분석 힌트 추가 (선택적)
    if morphology_hints and len(morphology_hints) > 0:
        # 고유명사(NNP)만 추출
        proper_nouns = [morph for morph, pos in morphology_hints if pos == 'NNP']
        if proper_nouns:
            nouns_str = ", ".join(proper_nouns[:5])  # 최대 5개
            prompt += f"""
**형태소 분석 결과 (참고용 고유명사):**
{nouns_str}
"""
    
    # 출력 형식 지정
    prompt += """
**출력 형식 (JSON만 출력, 다른 설명 금지):**
{
    "original": "입력 텍스트",
    "refined": "교정된 텍스트"
}

**예시:**
입력: "하나 낸 계좌에서 출근할까요"
출력:
{
    "original": "하나 낸 계좌에서 출근할까요",
    "refined": "하나은행 계좌에서 출금할까요"
}
"""
    
    return prompt


def user_message(text: str) -> str:
    return f"입력: {text}"


def refinement_payload(
    text: str,
    temperature: float = 0.1,
    max_tokens: int = 50,
    top_p: float = 0.9
) -> Dict:
    """
    텍스트 교정 요청을 위한 페이로드를 생성합니다. (기존 함수 유지)
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


def refinement_payload_with_context(
    text: str,
    known_entities: Optional[List[str]] = None,
    morphology_hints: Optional[List[Tuple[str, str]]] = None,
    temperature: float = 0.1,
    max_tokens: int = 256,
    top_p: float = 0.9
) -> Dict:
    """
    컨텍스트 인식 교정 요청 페이로드 생성
    
    Args:
        text: 교정할 텍스트
        known_entities: 감지된 카드상품명 등
        morphology_hints: 형태소 분석 결과
        temperature: LLM temperature
        max_tokens: 최대 토큰 수 (교정 결과가 길 수 있으므로 증가)
        top_p: LLM top_p
    
    Returns:
        RunPod API 페이로드
    """
    system_prompt = refinement_prompt_with_context(text, known_entities, morphology_hints)
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


def extract_json_from_text(text: str) -> Optional[str]:
    """
    텍스트에서 JSON 객체 추출 (마크다운 코드 블록 등 제거)
    
    Args:
        text: LLM 출력 텍스트
    
    Returns:
        추출된 JSON 문자열 또는 None
    """
    if not text:
        return None
    
    # 1. 마크다운 코드 블록 제거
    text = text.replace("```json", "").replace("```", "")
    
    # 2. JSON 객체 패턴 매칭 (중괄호로 시작/끝)
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        # 가장 긴 매칭 결과 반환 (가장 완전한 JSON일 가능성 높음)
        return max(matches, key=len)
    
    # 3. 전체 텍스트가 JSON인 경우
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return text
    
    return None


def parse_refinement_result(llm_output: str, original_text: str) -> Dict[str, any]:
    """
    LLM 출력 결과를 파싱하여 교정된 텍스트를 추출합니다. (개선된 버전)
    
    Args:
        llm_output: LLM의 원시 출력 텍스트
        original_text: 원본 텍스트 (파싱 실패 시 반환용)
    
    Returns:
        교정 결과 딕셔너리 {
            "text": str,
            "keywords": list,
            "corrections": list (optional),
            "confidence": float (optional)
        }
    """
    if not llm_output:
        print(f"[Refiner] 빈 LLM 출력")
        return {"text": original_text, "keywords": []}
    
    # JSON 추출 시도
    json_str = extract_json_from_text(llm_output)
    
    if not json_str:
        print(f"[Refiner] JSON 추출 실패: {llm_output[:100]}")
        return {"text": original_text, "keywords": []}
    
    try:
        # JSON 파싱
        result = json.loads(json_str)
        
        # refined 필드 추출
        refined_text = result.get("refined", result.get("text", original_text))
        
        # 추가 정보 추출
        corrections = result.get("corrections", [])
        confidence = result.get("confidence", 0.0)
        
        # 유효성 검증
        if not refined_text or len(refined_text.strip()) == 0:
            print(f"[Refiner] 교정 결과가 비어있음")
            return {"text": original_text, "keywords": []}
        
        # 성공
        return {
            "text": refined_text,
            "keywords": [],  # 키워드 추출은 별도 모듈에서 처리
            "corrections": corrections,
            "confidence": confidence
        }
        
    except json.JSONDecodeError as e:
        print(f"[Refiner] JSON 파싱 실패: {e}")
        print(f"[Refiner] 추출된 JSON: {json_str[:200]}")
        return {"text": original_text, "keywords": []}
    except Exception as e:
        print(f"[Refiner] 예상치 못한 오류: {e}")
        return {"text": original_text, "keywords": []}


def validate_refinement(
    original: str,
    refined: str,
    known_entities: Optional[List[str]] = None
) -> Tuple[str, List[str]]:
    """
    교정 결과 검증 및 후처리
    
    Args:
        original: 원본 텍스트
        refined: 교정된 텍스트
        known_entities: 보존해야 할 고유명사 리스트
    
    Returns:
        (검증된 교정 텍스트, 경고 메시지 리스트)
    """
    warnings = []
    validated_text = refined
    
    # 1. 길이 검증 (교정 결과가 원본보다 3배 이상 길면 의심)
    if len(refined) > len(original) * 3:
        warnings.append(f"교정 결과가 원본보다 {len(refined) / len(original):.1f}배 길어짐")
    
    # 2. 고유명사 보존 검증
    if known_entities:
        for entity in known_entities:
            if entity in original and entity not in refined:
                warnings.append(f"고유명사 '{entity}'가 교정 과정에서 누락됨")
                # 복구 시도: 원본 유지
                validated_text = original
    
    # 3. 빈 결과 검증
    if not refined.strip():
        warnings.append("교정 결과가 비어있음")
        validated_text = original
    
    return validated_text, warnings