"""
Persona Generator - 페르소나 생성 모듈

분석된 고객 특성을 기반으로 시뮬레이션용 시스템 프롬프트를 생성합니다.
"""
import json
from typing import Dict, Any


def create_system_prompt(customer_profile: Dict[str, Any], difficulty: str = "beginner") -> str:
    """
    고객 프로필을 기반으로 시뮬레이션 시스템 프롬프트 생성
    
    Args:
        customer_profile: 고객 정보 (customer.json 형식)
            - name: 고객명
            - age_group: 연령대
            - personality_tags: 성격 태그
            - communication_style: 의사소통 스타일
            - llm_guidance: LLM 가이던스
        difficulty: 난이도 ("beginner" 또는 "advanced")
        
    Returns:
        시스템 프롬프트 (str)
    """
    name = customer_profile.get("name", "고객")
    age_group = customer_profile.get("age_group", "성인")
    
    # personality_tags 파싱 (DB에서 "{tag1,tag2}" 형식으로 저장됨)
    personality_tags_str = customer_profile.get("personality_tags", "")
    if isinstance(personality_tags_str, str):
        # "{emotional,expressive}" -> ["emotional", "expressive"]
        personality_tags = personality_tags_str.strip("{}").split(",") if personality_tags_str else []
    else:
        personality_tags = personality_tags_str
    
    # communication_style 파싱
    comm_style = customer_profile.get("communication_style", {})
    if isinstance(comm_style, str):
        try:
            comm_style = json.loads(comm_style)
        except:
            comm_style = {"tone": "neutral", "speed": "moderate"}
    
    tone = comm_style.get("tone", "neutral")
    speed = comm_style.get("speed", "moderate")
    
    llm_guidance = customer_profile.get("llm_guidance", "일반적인 응대로 친절하게 안내해주세요.")
    
    # 성격 태그에 따른 세부 행동 지침
    behavior_instructions = []
    
    if "elderly" in personality_tags or "needs_repetition" in personality_tags:
        behavior_instructions.append("- 디지털 용어나 복잡한 절차에 어려움을 겪습니다.")
        behavior_instructions.append("- 같은 질문을 반복하거나 이해를 확인합니다.")
    
    if "emotional" in personality_tags or "expressive" in personality_tags:
        behavior_instructions.append("- 감정을 드러내며 적극적으로 의견을 표현합니다.")
        behavior_instructions.append("- 불만이 있을 때 강하게 표현할 수 있습니다.")
    
    if "patient" in personality_tags or "polite" in personality_tags:
        behavior_instructions.append("- 차분하고 예의 바르게 대화합니다.")
        behavior_instructions.append("- 상담원의 설명을 끝까지 경청합니다.")
    
    # 말투 속도 지침
    speed_instruction = {
        "slow": "천천히, 또박또박 말합니다.",
        "moderate": "보통 속도로 말합니다.",
        "fast": "빠르게 말하며 급한 성격을 드러냅니다."
    }.get(speed, "보통 속도로 말합니다.")
    
    # 말투 톤 지침
    tone_instruction = {
        "neutral": "중립적이고 사무적인 톤으로 대화합니다.",
        "empathetic": "친근하고 감정이 담긴 톤으로 대화합니다.",
        "respectful": "공손하고 격식을 차린 말투를 사용합니다."
    }.get(tone, "중립적이고 사무적인 톤으로 대화합니다.")
    
    # 난이도별 추가 지침
    if difficulty == "advanced":
        complexity_instruction = """
### 상급 난이도 지침
- 복잡하고 다양한 상황을 연출합니다.
- 추가 문의나 관련 질문을 자연스럽게 제기합니다.
- 상담원의 응대에 따라 감정 변화를 표현합니다.
- 실제 고객처럼 자연스러운 대화 흐름을 유지합니다."""
    else:
        complexity_instruction = """
### 초급 난이도 지침
- 단순하고 명확한 문의를 합니다.
- 상담원의 안내에 협조적입니다.
- 복잡한 상황보다는 기본적인 케이스를 다룹니다."""
    
    # 최종 시스템 프롬프트 조합
    system_prompt = f"""당신은 카드사 고객센터에 문의하는 '고객'입니다.
절대로 정보를 제공하는 역할을 하지 말고, 사용자의 물음에 계속해서 질문을 하는 고객임을 명심하세요.

## 고객 성향
{chr(10).join(behavior_instructions) if behavior_instructions else '- 일반적인 고객입니다.'}

## 문의 목적
당신의 목적은 카드를 분실하여 이에 대한 상담을 받고자 하는 것입니다.

## 말투 특성
- {tone_instruction}

## 상황 지침
{complexity_instruction}

## 역할 수행 규칙
1. 실제 고객처럼 자연스럽게 대화하세요. 상담원의 이름을 부르지 마세요.
2. 상담원의 질문에 간결하고 명확하게, 2개의 문장 이내로 답변하세요.
3. 불필요한 질문이나 부연설명은 피하세요.
4. 당신의 역할(고객)에 충실하세요. 상담원처럼 행동하지 마세요.
5. "고객님" 같은 호칭을 사용하지 마세요. 당신이 고객입니다.
6. 상담원에게 질문을 계속 던지세요.
7. 매 답변마다 인사를 하지마세요.
8. 상담원의 말이 이해가 되지 않을 때에만 추가 질문을 하세요.
9. 개인정보의 경우 가상의 인물이 되었다고 가정하여 임의로 지어내세요.
"""
    
    return system_prompt.strip()


def create_scenario_script(consultation_content: str, difficulty: str = "advanced") -> Dict[str, Any]:
    """
    상급 난이도용 시나리오 각본 생성
    
    Args:
        consultation_content: 실제 상담 내용
        difficulty: 난이도 (현재는 "advanced"만 지원)
        
    Returns:
        시나리오 각본 dict:
        {
            "expected_flow": [단계별 기대 흐름],
            "key_points": [핵심 포인트],
            "evaluation_criteria": [평가 기준]
        }
    """
    from app.llm.education.client import generate_json
    
    system_prompt = """당신은 고객센터 교육 시나리오 설계 전문가입니다.
실제 상담 내용을 분석하여 교육용 시나리오 각본을 작성합니다."""

    prompt = f"""다음 우수 상담 사례를 분석하여 교육용 각본을 작성하세요:

{consultation_content}

다음 항목을 포함한 JSON으로 출력:
{{
    "expected_flow": ["1단계: ...", "2단계: ...", ...],
    "key_points": ["핵심포인트1", "핵심포인트2", ...],
    "evaluation_criteria": ["평가기준1", "평가기준2", ...]
}}"""

    result = generate_json(prompt, system_prompt, temperature=0.3, max_tokens=800)
    
    if not result:
        return {
            "expected_flow": ["상담 시작", "문의 파악", "해결책 제시", "상담 종료"],
            "key_points": ["고객 니즈 파악", "정확한 정보 제공"],
            "evaluation_criteria": ["고객 만족도", "문제 해결 여부"]
        }
    
    return result