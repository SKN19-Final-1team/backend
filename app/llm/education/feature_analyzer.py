import json
from typing import Dict, Any, List
from app.llm.education.client import generate_json, generate_text
from app.core.prompt import FEATURE_ANALYSIS_PROMPT


def analyze_consultation(consultation_content: str, customer_info: Dict[str, Any] = None) -> Dict[str, Any]:
    system_prompt = FEATURE_ANALYSIS_PROMPT

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
        "personality_tags": ["practical", "direct"],
        "communication_style": {
            "tone": "direct",
            "speed": "moderate"
        },
        "llm_guidance": "일반적인 응대로 친절하게 안내해주세요."
    }
    
    for key, default_val in defaults.items():
        if key not in result:
            print(f"[Feature Analyzer] Warning: Missing key '{key}' in analysis result. Using default.")
            result[key] = default_val
            
    return result

def summarize_consultation_flow(consultation_content: str) -> Dict[str, Any]:
    """
    상급 난이도용: 상담 전문을 요약하여 전체 흐름과 시나리오 생성

    Args:
        consultation_content: 상담 대화 내용 전체

    Returns:
        {
            "overall_flow": "전체 상담 흐름 요약 (2-3문장)",
            "consultation_scenario": {
                "customer_problem": "고객의 핵심 문제",
                "resolution_steps": ["단계1", "단계2", ...],
                "key_exchanges": ["핵심 대화1", "핵심 대화2", ...]
            }
        }
    """
    if not consultation_content:
        return {
            "overall_flow": "",
            "consultation_scenario": {
                "customer_problem": "",
                "resolution_steps": [],
                "key_exchanges": []
            }
        }

    system_prompt = """당신은 콜센터 교육 시나리오 전문가입니다.
상담 대화 내용을 분석하여 교육용 요약과 시나리오를 생성합니다.

다음 항목을 추출하세요:
1. overall_flow: 상담의 전체 흐름을 2-3문장으로 요약
2. consultation_scenario:
   - customer_problem: 고객이 가진 핵심 문제/문의 사항
   - resolution_steps: 문제 해결을 위해 진행된 단계들 (리스트)
   - key_exchanges: 상담에서 가장 중요한 대화 포인트들 (리스트)

JSON 형식으로만 출력하세요."""

    prompt = f"""다음 상담 내용을 분석하여 교육용 요약을 생성하세요:

{consultation_content[:3000]}

분석 결과를 아래 JSON 형식으로 출력:
{{
    "overall_flow": "전체 흐름 요약",
    "consultation_scenario": {{
        "customer_problem": "핵심 문제",
        "resolution_steps": ["단계1", "단계2"],
        "key_exchanges": ["핵심 대화1", "핵심 대화2"]
    }}
}}"""

    result = generate_json(prompt, system_prompt, temperature=0.3, max_tokens=500)

    if not result:
        result = {}

    # 기본값 채우기
    defaults = {
        "overall_flow": "상담 내용 요약을 생성할 수 없습니다.",
        "consultation_scenario": {
            "customer_problem": "",
            "resolution_steps": [],
            "key_exchanges": []
        }
    }

    for key, default_val in defaults.items():
        if key not in result:
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
