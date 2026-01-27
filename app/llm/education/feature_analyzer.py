"""
Feature Analyzer - 상담 내용 분석 모듈

과거 상담 기록을 분석하여 고객의 성향, 특성, 말투 등을 추출합니다.
"""
import json
from typing import Dict, Any, List
from app.llm.education.client import generate_json, generate_text


def analyze_consultation(consultation_content: str, customer_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    상담 내용을 분석하여 고객 특성을 추출
    
    Args:
        consultation_content: 상담 대화 내용 (전체 transcript)
        customer_info: 고객 기본 정보 (선택사항)
        
    Returns:
        분석 결과 dict:
        {
            "personality_tags": [str],  # 예: ["emotional", "expressive"]
            "communication_style": {
                "tone": str,  # "neutral", "empathetic", "respectful" 등
                "speed": str  # "slow", "moderate", "fast"
            },
            "llm_guidance": str,  # 상담원을 위한 가이드
            "age_group_inferred": str  # 추론된 연령대 (선택사항)
        }
    """
    
    system_prompt = """당신은 고객센터 상담 분석 전문가입니다.
상담 대화 내용을 분석하여 고객의 성향과 특성을 파악합니다.

다음 항목을 분석하세요:
1. personality_tags: 고객의 성격적 특성 (배열)
   - elderly: 고령층, 디지털 미숙
   - patient: 침착하고 인내심 있음
   - needs_repetition: 반복 설명 필요
   - emotional: 감정적
   - expressive: 표현이 풍부함
   - polite: 공손함
   - normal: 일반적인 응대

2. communication_style: 의사소통 스타일
   - tone: "neutral"(중립적), "empathetic"(공감적), "respectful"(존중)
   - speed: "slow"(천천히), "moderate"(보통), "fast"(빠르게)

3. llm_guidance: 상담원을 위한 응대 가이드 (1-2문장)

JSON 형식으로만 출력하세요."""

    prompt = f"""다음 상담 내용을 분석하세요:

{consultation_content}

분석 결과를 아래 JSON 형식으로 출력:
{{
    "personality_tags": ["tag1", "tag2"],
    "communication_style": {{
        "tone": "...",
        "speed": "..."
    }},
    "llm_guidance": "..."
}}"""

    result = generate_json(prompt, system_prompt, temperature=0.2, max_tokens=300)
    
    if not result:
        result = {}
        
    # 필수 키 검증 및 기본값 채우기
    defaults = {
        "personality_tags": ["normal", "polite"],
        "communication_style": {
            "tone": "neutral",
            "speed": "moderate"
        },
        "llm_guidance": "일반적인 응대로 친절하게 안내해주세요."
    }
    
    for key, default_val in defaults.items():
        if key not in result:
            print(f"[Feature Analyzer] Warning: Missing key '{key}' in analysis result. Using default.")
            result[key] = default_val
            
    return result

def format_analysis_for_db(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    분석 결과를 DB 저장 형식으로 변환
    
    Args:
        analysis: analyze_consultation 결과
        
    Returns:
        DB 저장 형식 (customer.json 스키마)
    """
    personality_tags_str = "{" + ",".join(analysis.get("personality_tags", [])) + "}"
    
    return {
        "personality_tags": personality_tags_str,
        "communication_style": json.dumps(analysis.get("communication_style", {}), ensure_ascii=False),
        "llm_guidance": analysis.get("llm_guidance", "")
    }
