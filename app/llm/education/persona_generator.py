"""
Persona Generator - 페르소나 생성 모듈

분석된 고객 특성을 기반으로 시뮬레이션용 시스템 프롬프트를 생성합니다.
"""
import json
from typing import Dict, Any, List, Optional
from app.core.prompt import PERSONA_BASIC_PROMPT, PERSONA_ADVANCED_PROMPT


# personality_tags → 한글 설명 매핑 (DB persona_types 기반)
TAG_DESCRIPTIONS = {
    # Normal Types
    "practical": "불필요한 말 없이 바로 본론으로 진행합니다.",
    "direct": "직접적이고 명확하게 의사를 표현합니다.",
    "efficient": "효율적인 처리를 원합니다.",
    "friendly": "친근하게 대화합니다.",
    "talkative": "사적인 이야기나 본인 상황을 길게 설명합니다.",
    "personal": "개인적인 이야기를 자주 합니다.",
    "cautious": "신중하게 확인합니다.",
    "security_conscious": "보안에 민감합니다.",
    "suspicious": "의심이 많고 확인을 자주 요청합니다.",
    "passive": "관심 없이 최소한의 답변만 합니다.",
    "disengaged": "설명을 듣지 않으려 합니다.",
    "minimal_response": "최소한의 답변만 합니다.",

    # Special Types
    "impatient": "급한 성격으로 빠른 처리를 원합니다.",
    "urgent": "긴급한 상황입니다.",
    "busy": "바쁜 상황입니다.",
    "detailed": "상세한 설명을 요구합니다.",
    "analytical": "꼼꼼하게 분석합니다.",
    "thorough": "철저하게 확인합니다.",
    "confused": "설명을 잘 이해하지 못합니다.",
    "needs_repetition": "반복적으로 확인합니다.",
    "patient_required": "인내심 있는 응대가 필요합니다.",
    "repeat_caller": "과거 상담 이력을 언급합니다.",
    "frustrated": "해결되지 않은 불만이 있습니다.",
    "unresolved": "이전에 해결되지 않은 문제가 있습니다.",
    "angry": "분노를 표현합니다.",
    "demanding": "요구가 많습니다.",
    "elderly": "고령층으로 디지털 기기 사용이 어렵습니다.",
    "digital_vulnerable": "디지털 취약계층입니다.",
    "phone_preferred": "전화 상담을 선호합니다.",
    "foreign": "한국어가 모국어가 아닙니다.",
    "language_barrier": "언어 소통에 어려움이 있습니다.",
    "simple_korean": "쉬운 한국어로 설명이 필요합니다.",
    "vip": "VIP 고객입니다.",
    "premium": "프리미엄 서비스를 기대합니다.",
    "high_expectation": "높은 수준의 서비스를 기대합니다.",

    # Legacy tags (기존 호환)
    "emotional": "감정을 드러내며 적극적으로 의견을 표현합니다.",
    "expressive": "표현이 풍부합니다.",
    "patient": "침착하고 인내심 있게 대화합니다.",
    "polite": "공손하게 대화합니다.",
    "normal": "일반적인 고객입니다.",
}

# tone → 한글 설명 매핑
TONE_DESCRIPTIONS = {
    "direct": "직접적이고 명확한 톤으로 대화합니다.",
    "warm": "친근하고 따뜻한 톤으로 대화합니다.",
    "formal": "격식을 차린 말투를 사용합니다.",
    "engaging": "관심을 끌려고 노력하며 대화합니다.",
    "concise": "간결하게 핵심만 말합니다.",
    "thorough": "상세하게 설명을 요구합니다.",
    "patient": "차분하고 인내심 있게 대화합니다.",
    "solution_focused": "해결책을 중심으로 대화합니다.",
    "calm_professional": "차분하고 전문적인 톤을 유지합니다.",
    "clear": "명확하게 의사를 전달합니다.",
    "premium_service": "고급 서비스를 기대하는 톤입니다.",
    # Legacy tones
    "neutral": "중립적이고 사무적인 톤으로 대화합니다.",
    "empathetic": "친근하고 감정이 담긴 톤으로 대화합니다.",
    "respectful": "공손하고 격식을 차린 말투를 사용합니다.",
}

# speed → 한글 설명 매핑
SPEED_DESCRIPTIONS = {
    "slow": "천천히, 또박또박 말합니다.",
    "moderate": "보통 속도로 말합니다.",
    "fast": "빠르게 말하며 급한 성격을 드러냅니다.",
}


def _parse_personality_tags(tags) -> List[str]:
    """Parse personality tags from various formats"""
    if isinstance(tags, str):
        # Handle DB format: "{tag1,tag2}"
        return [t.strip() for t in tags.strip("{}").split(",") if t.strip()]
    elif isinstance(tags, list):
        return [t.strip() for t in tags if t and isinstance(t, str)]
    return []


def _parse_communication_style(style) -> Dict[str, str]:
    """Parse communication style from various formats"""
    if isinstance(style, str):
        try:
            return json.loads(style)
        except:
            return {"tone": "direct", "speed": "moderate"}
    elif isinstance(style, dict):
        return style
    return {"tone": "direct", "speed": "moderate"}


def _build_personality_description(tags: List[str], persona_desc: str = None) -> str:
    """Build personality description from tags and persona description"""
    descriptions = []

    if persona_desc:
        descriptions.append(persona_desc)

    for tag in tags:
        tag_clean = tag.strip().lower()
        if tag_clean in TAG_DESCRIPTIONS:
            descriptions.append(f"- {TAG_DESCRIPTIONS[tag_clean]}")

    if not descriptions:
        return "- 일반적인 고객입니다."

    return "\n".join(descriptions)


def _build_speech_characteristics(comm_style: Dict[str, str], tags: List[str]) -> str:
    """Build speech characteristics from communication style"""
    tone = comm_style.get("tone", "direct")
    speed = comm_style.get("speed", "moderate")

    tone_desc = TONE_DESCRIPTIONS.get(tone, "중립적이고 사무적인 톤으로 대화합니다.")
    speed_desc = SPEED_DESCRIPTIONS.get(speed, "보통 속도로 말합니다.")

    characteristics = [tone_desc, speed_desc]

    # Add special characteristics based on tags
    if "angry" in tags or "frustrated" in tags:
        characteristics.append("불만이 있을 때 직접적으로 표현합니다.")
    if "elderly" in tags or "digital_vulnerable" in tags:
        characteristics.append("존댓말을 주로 사용합니다.")
    if "vip" in tags or "premium" in tags:
        characteristics.append("고급 서비스를 기대합니다.")

    return " ".join(characteristics)


def _build_inquiry_purpose(customer_profile: Dict[str, Any], scenario_context: Dict[str, Any] = None) -> str:
    """Build inquiry purpose from profile and scenario"""
    # Check if scenario context has customer problem
    if scenario_context and scenario_context.get("consultation_scenario"):
        problem = scenario_context["consultation_scenario"].get("customer_problem", "")
        if problem:
            return f"당신의 목적은 다음과 같습니다: {problem}"

    # Check if there's llm_guidance with specific purpose
    llm_guidance = customer_profile.get("llm_guidance", "")
    if llm_guidance and ":" in llm_guidance:
        return llm_guidance

    # Default inquiry purposes based on category (if available)
    category = customer_profile.get("category", "")
    category_purposes = {
        "분실/도난": "카드를 분실하여 정지 및 재발급에 대해 문의하고자 합니다.",
        "결제/승인": "카드 결제 관련 문의가 있습니다.",
        "한도": "카드 한도 조회 또는 변경에 대해 문의하고자 합니다.",
        "이용내역": "카드 이용내역을 확인하고자 합니다.",
        "수수료/연체": "수수료 또는 연체 관련 문의가 있습니다.",
        "포인트/혜택": "포인트 또는 혜택에 대해 문의하고자 합니다.",
        "정부지원": "정부지원 관련 문의가 있습니다.",
    }

    return category_purposes.get(category, "카드 관련 문의가 있습니다.")


def create_system_prompt(
    customer_profile: Dict[str, Any],
    difficulty: str = "beginner",
    scenario_context: Dict[str, Any] = None
) -> str:
    """
    고객 프로필을 기반으로 시뮬레이션 시스템 프롬프트 생성

    Args:
        customer_profile: 고객 정보 (DB + 분석 결과)
            - name: 고객명
            - age_group: 연령대
            - grade: 등급
            - personality_tags: 성격 태그
            - communication_style: 의사소통 스타일
            - llm_guidance: LLM 가이던스
            - persona_name: 페르소나 이름 (선택)
            - persona_description: 페르소나 설명 (선택)
        difficulty: 난이도 ("beginner" 또는 "advanced")
        scenario_context: 시나리오 컨텍스트 (advanced 모드에서 사용)
            - overall_flow: 전체 흐름 요약
            - consultation_scenario: 시나리오 상세

    Returns:
        시스템 프롬프트 (str)
    """
    # 1. Extract customer information
    name = customer_profile.get("name", "고객")
    age_group = customer_profile.get("age_group", "40대")
    grade = customer_profile.get("grade", "GENERAL")

    # 2. Parse personality_tags
    personality_tags = _parse_personality_tags(customer_profile.get("personality_tags", []))

    # 3. Parse communication_style
    comm_style = _parse_communication_style(customer_profile.get("communication_style", {}))

    # 4. Get persona information if available
    persona_name = customer_profile.get("persona_name", "")
    persona_description = customer_profile.get("persona_description", "")

    # 5. Build dynamic descriptions
    personality_description = _build_personality_description(personality_tags, persona_description)
    speech_characteristics = _build_speech_characteristics(comm_style, personality_tags)
    inquiry_purpose = _build_inquiry_purpose(customer_profile, scenario_context)

    # 6. Select appropriate template based on difficulty
    if difficulty == "advanced":
        scenario_summary = ""
        if scenario_context:
            scenario_summary = scenario_context.get("overall_flow", "")

        return PERSONA_ADVANCED_PROMPT.format(
            customer_name=name,
            age_group=age_group,
            grade=grade,
            persona_type=persona_name or "일반 고객",
            inquiry_purpose=inquiry_purpose,
            scenario_summary=scenario_summary or "시나리오 정보가 없습니다.",
            personality_description=personality_description,
            speech_characteristics=speech_characteristics
        ).strip()
    else:
        return PERSONA_BASIC_PROMPT.format(
            customer_name=name,
            age_group=age_group,
            grade=grade,
            inquiry_purpose=inquiry_purpose,
            personality_description=personality_description,
            speech_characteristics=speech_characteristics
        ).strip()


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
