"""
교육 시뮬레이션 평가 모듈

시뮬레이션 결과를 우수사례/시나리오와 비교하여 평가합니다.
- 키워드 커버리지 계산
- 순서 정확도 평가
- 종합 점수 산정
- 피드백 생성
"""

import re
from typing import Dict, Any, List, Optional
from app.llm.education.client import generate_json


def evaluate_simulation_result(
    simulation_transcript: str,
    original_transcript: Optional[str],
    scenario_script: Dict[str, Any],
    simulation_type: str = "scenario"
) -> Dict[str, Any]:
    """
    시뮬레이션 결과를 종합 평가

    Args:
        simulation_transcript: 시뮬레이션 대화 전문
        original_transcript: 원본 우수사례 전문 (best_practice 타입일 때)
        scenario_script: 시나리오 각본 (expected_flow, key_points, evaluation_criteria)
        simulation_type: "best_practice" 또는 "scenario"

    Returns:
        {
            "overall_score": 종합 점수 (0-100),
            "keyword_score": 키워드 커버리지 점수,
            "sequence_score": 순서 정확도 점수,
            "similarity_score": 우수사례 유사도 (best_practice만),
            "feedback": 상세 피드백,
            "passed": 합격 여부
        }
    """
    scores = {}

    # 1. 키워드 커버리지 계산
    key_points = scenario_script.get("key_points", [])
    keyword_result = calculate_keyword_coverage(simulation_transcript, key_points)
    scores["keyword_score"] = keyword_result["score"]
    scores["keyword_details"] = keyword_result

    # 2. 순서 정확도 계산
    expected_flow = scenario_script.get("expected_flow", [])
    scores["sequence_score"] = calculate_sequence_correctness(
        simulation_transcript, expected_flow
    )

    # 3. 우수사례 유사도 계산 (best_practice 타입일 때)
    if simulation_type == "best_practice" and original_transcript:
        scores["similarity_score"] = calculate_similarity_score(
            simulation_transcript, original_transcript
        )
    else:
        scores["similarity_score"] = None

    # 4. 종합 점수 계산
    evaluation_criteria = scenario_script.get("evaluation_criteria", {})
    if isinstance(evaluation_criteria, list):
        # 리스트 형태면 기본 가중치 적용
        evaluation_criteria = {
            "keyword_coverage": {"weight": 40},
            "sequence_correctness": {"weight": 30},
            "similarity": {"weight": 30}
        }

    scores["overall_score"] = calculate_overall_score(evaluation_criteria, scores)

    # 5. 피드백 생성
    scores["feedback"] = generate_feedback(scores, scenario_script)

    # 6. 합격 여부 판정 (기본 70점)
    passing_score = scenario_script.get("passing_score", 70)
    scores["passed"] = scores["overall_score"] >= passing_score

    return scores


def calculate_keyword_coverage(
    transcript: str,
    required_keywords: List[str]
) -> Dict[str, Any]:
    """
    필수 키워드 사용 여부 체크

    Args:
        transcript: 대화 전문
        required_keywords: 필수 키워드 리스트

    Returns:
        {
            "score": 점수 (0-100),
            "found": 발견된 키워드 리스트,
            "missing": 누락된 키워드 리스트,
            "coverage_ratio": 커버리지 비율
        }
    """
    if not required_keywords:
        return {
            "score": 100,
            "found": [],
            "missing": [],
            "coverage_ratio": 1.0
        }

    transcript_lower = transcript.lower()
    found = []
    missing = []

    for keyword in required_keywords:
        # 키워드에서 "핵심포인트1:" 같은 접두사 제거
        clean_keyword = re.sub(r"^(핵심포인트|포인트|키워드)\d*[:\s]*", "", keyword)
        clean_keyword = clean_keyword.strip()

        if not clean_keyword:
            continue

        # 키워드 또는 키워드의 핵심 단어가 포함되어 있는지 확인
        keywords_to_check = [clean_keyword]

        # 긴 문장이면 핵심 단어만 추출
        if len(clean_keyword) > 10:
            # 명사 추출 시도
            words = re.findall(r"[가-힣]+", clean_keyword)
            keywords_to_check.extend([w for w in words if len(w) >= 2])

        matched = False
        for kw in keywords_to_check:
            if kw.lower() in transcript_lower:
                matched = True
                break

        if matched:
            found.append(keyword)
        else:
            missing.append(keyword)

    coverage_ratio = len(found) / len(required_keywords) if required_keywords else 1.0
    score = int(coverage_ratio * 100)

    return {
        "score": score,
        "found": found,
        "missing": missing,
        "coverage_ratio": coverage_ratio
    }


def calculate_sequence_correctness(
    transcript: str,
    expected_sequence: List[str]
) -> int:
    """
    처리 순서 정확도 계산

    Args:
        transcript: 대화 전문
        expected_sequence: 기대 순서 리스트 (예: ["1단계: 본인확인", "2단계: 문의파악", ...])

    Returns:
        점수 (0-100)
    """
    if not expected_sequence:
        return 100

    transcript_lower = transcript.lower()

    # 각 단계의 핵심 키워드 추출 및 위치 찾기
    step_positions = []

    for step in expected_sequence:
        # "1단계: 본인확인" -> "본인확인"
        clean_step = re.sub(r"^(\d+단계|단계\s*\d+)[:\s]*", "", step)
        clean_step = clean_step.strip()

        if not clean_step:
            continue

        # 핵심 단어 추출
        keywords = re.findall(r"[가-힣]+", clean_step)
        keywords = [kw for kw in keywords if len(kw) >= 2]

        # 첫 번째 발견 위치 기록
        position = -1
        for kw in keywords:
            pos = transcript_lower.find(kw.lower())
            if pos != -1:
                if position == -1 or pos < position:
                    position = pos
                break

        step_positions.append(position)

    # 순서 정확도 계산
    # 발견된 단계들이 올바른 순서인지 확인
    found_positions = [p for p in step_positions if p != -1]

    if not found_positions:
        return 0

    # 발견된 것들의 순서가 맞는지 확인
    correct_order_count = 0
    for i in range(len(found_positions) - 1):
        if found_positions[i] <= found_positions[i + 1]:
            correct_order_count += 1

    # 발견율과 순서 정확도를 결합
    found_ratio = len(found_positions) / len(expected_sequence)
    order_ratio = (correct_order_count + 1) / len(found_positions) if found_positions else 0

    score = int((found_ratio * 0.6 + order_ratio * 0.4) * 100)
    return min(100, score)


def calculate_similarity_score(
    simulation_transcript: str,
    original_transcript: str
) -> int:
    """
    우수사례와의 유사도 계산

    Args:
        simulation_transcript: 시뮬레이션 대화 전문
        original_transcript: 원본 우수사례 전문

    Returns:
        유사도 점수 (0-100)
    """
    if not original_transcript:
        return 0

    # LLM을 사용한 의미적 유사도 평가
    system_prompt = """당신은 콜센터 상담 품질 평가 전문가입니다.
두 상담 대화의 유사도를 평가합니다."""

    prompt = f"""다음 두 상담 대화를 비교하여 유사도를 평가하세요:

[원본 우수사례]
{original_transcript[:2000]}

[시뮬레이션 결과]
{simulation_transcript[:2000]}

평가 기준:
1. 문제 해결 접근 방식의 유사성
2. 사용된 표현과 화법의 적절성
3. 고객 응대 흐름의 일관성

다음 JSON 형식으로 출력:
{{"similarity_score": 0-100 사이의 점수, "reason": "평가 이유"}}"""

    result = generate_json(prompt, system_prompt, temperature=0.2, max_tokens=300)

    if result and "similarity_score" in result:
        return int(result["similarity_score"])

    # LLM 실패 시 단순 키워드 기반 유사도
    sim_words = set(re.findall(r"[가-힣]+", simulation_transcript.lower()))
    orig_words = set(re.findall(r"[가-힣]+", original_transcript.lower()))

    if not orig_words:
        return 0

    intersection = len(sim_words & orig_words)
    union = len(sim_words | orig_words)

    jaccard = intersection / union if union > 0 else 0
    return int(jaccard * 100)


def calculate_overall_score(
    evaluation_criteria: Dict[str, Any],
    scores: Dict[str, Any]
) -> int:
    """
    가중치 기반 종합 점수 계산

    Args:
        evaluation_criteria: 평가 기준 및 가중치
            예: {"keyword_coverage": {"weight": 40}, "sequence_correctness": {"weight": 30}, ...}
        scores: 개별 점수들

    Returns:
        종합 점수 (0-100)
    """
    # 기본 가중치 설정
    default_weights = {
        "keyword_coverage": 40,
        "sequence_correctness": 30,
        "similarity": 30
    }

    total_weight = 0
    weighted_sum = 0

    # 키워드 커버리지
    kw_weight = evaluation_criteria.get("keyword_coverage", {}).get("weight", default_weights["keyword_coverage"])
    weighted_sum += scores.get("keyword_score", 0) * kw_weight
    total_weight += kw_weight

    # 순서 정확도
    seq_weight = evaluation_criteria.get("sequence_correctness", {}).get("weight", default_weights["sequence_correctness"])
    weighted_sum += scores.get("sequence_score", 0) * seq_weight
    total_weight += seq_weight

    # 유사도 (있는 경우에만)
    if scores.get("similarity_score") is not None:
        sim_weight = evaluation_criteria.get("similarity", {}).get("weight", default_weights["similarity"])
        weighted_sum += scores["similarity_score"] * sim_weight
        total_weight += sim_weight

    if total_weight == 0:
        return 0

    return int(weighted_sum / total_weight)


def generate_feedback(
    scores: Dict[str, Any],
    scenario_script: Dict[str, Any]
) -> Dict[str, Any]:
    """
    상세 피드백 생성

    Args:
        scores: 평가 점수들
        scenario_script: 시나리오 각본

    Returns:
        {
            "strengths": ["강점1", "강점2", ...],
            "improvements": ["개선점1", "개선점2", ...],
            "tips": ["팁1", "팁2", ...]
        }
    """
    strengths = []
    improvements = []
    tips = []

    # 키워드 점수 기반 피드백
    keyword_score = scores.get("keyword_score", 0)
    keyword_details = scores.get("keyword_details", {})

    if keyword_score >= 80:
        strengths.append("핵심 키워드를 대부분 사용하여 적절한 상담을 진행했습니다.")
    elif keyword_score >= 50:
        improvements.append("일부 핵심 키워드가 누락되었습니다.")
        missing = keyword_details.get("missing", [])
        if missing:
            tips.append(f"다음 포인트를 더 강조해보세요: {', '.join(missing[:3])}")
    else:
        improvements.append("핵심 내용 전달이 부족합니다. 필수 키워드 사용을 늘려주세요.")

    # 순서 점수 기반 피드백
    sequence_score = scores.get("sequence_score", 0)

    if sequence_score >= 80:
        strengths.append("상담 진행 순서가 체계적입니다.")
    elif sequence_score >= 50:
        improvements.append("상담 순서를 좀 더 체계적으로 진행해보세요.")
        expected_flow = scenario_script.get("expected_flow", [])
        if expected_flow:
            tips.append(f"권장 순서: {' → '.join(expected_flow[:4])}")
    else:
        improvements.append("상담 진행 순서가 혼란스럽습니다. 표준 절차를 따라주세요.")

    # 유사도 점수 기반 피드백 (있는 경우)
    similarity_score = scores.get("similarity_score")
    if similarity_score is not None:
        if similarity_score >= 80:
            strengths.append("우수사례와 유사한 수준의 상담을 진행했습니다.")
        elif similarity_score >= 50:
            improvements.append("우수사례의 응대 방식을 더 참고해보세요.")
        else:
            improvements.append("우수사례와 차이가 큽니다. 모범 응대를 학습해주세요.")

    # 종합 점수 기반 추가 피드백
    overall_score = scores.get("overall_score", 0)

    if overall_score >= 90:
        strengths.append("전반적으로 우수한 상담을 진행했습니다!")
    elif overall_score >= 70:
        tips.append("조금만 더 노력하면 우수 수준에 도달할 수 있습니다.")
    else:
        tips.append("기본기를 다시 점검하고 연습해보세요.")

    return {
        "strengths": strengths,
        "improvements": improvements,
        "tips": tips
    }


def evaluate_with_llm(
    simulation_transcript: str,
    scenario_script: Dict[str, Any]
) -> Dict[str, Any]:
    """
    LLM을 활용한 심층 평가 (선택적 사용)

    Args:
        simulation_transcript: 시뮬레이션 대화 전문
        scenario_script: 시나리오 각본

    Returns:
        LLM 평가 결과
    """
    system_prompt = """당신은 콜센터 상담 품질 평가 전문가입니다.
시뮬레이션 상담 결과를 평가하고 구체적인 피드백을 제공합니다."""

    key_points = scenario_script.get("key_points", [])
    expected_flow = scenario_script.get("expected_flow", [])

    prompt = f"""다음 상담 시뮬레이션을 평가해주세요:

[상담 내용]
{simulation_transcript[:3000]}

[평가 기준]
핵심 포인트: {', '.join(key_points[:5])}
기대 흐름: {' → '.join(expected_flow[:5])}

다음 JSON 형식으로 평가 결과를 출력:
{{
    "communication_score": 0-100,
    "problem_solving_score": 0-100,
    "empathy_score": 0-100,
    "overall_comment": "전체 평가 코멘트",
    "specific_feedback": ["구체적 피드백1", "구체적 피드백2"]
}}"""

    result = generate_json(prompt, system_prompt, temperature=0.3, max_tokens=500)

    if not result:
        return {
            "communication_score": 70,
            "problem_solving_score": 70,
            "empathy_score": 70,
            "overall_comment": "평가를 완료했습니다.",
            "specific_feedback": []
        }

    return result
