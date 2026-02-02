"""
교육 시뮬레이션 유사도 계산 모듈

시뮬레이션 결과를 원본 상담 사례와 비교하여 유사도를 계산합니다.
followup.py의 is_simulation 조건에서 호출됩니다.
"""
import re
from typing import Dict, Any

from app.db.base import get_connection
from app.llm.education.client import generate_json


async def calculate_consultation_similarity(
    simulation_transcript: str,
    consultation_id: str
) -> Dict[str, Any]:
    """
    시뮬레이션 결과와 원본 상담의 유사도 계산

    Args:
        simulation_transcript: 시뮬레이션 대화 전문
        consultation_id: 원본 상담 ID (consultation_documents 조회용)

    Returns:
        {
            "similarity_score": 0-100 점수,
            "matching_elements": ["일치 요소들"],
            "missing_elements": ["누락된 요소들"],
            "feedback": "평가 피드백"
        }
    """
    # 1. DB에서 원본 상담 내용 조회
    original_content = _get_original_consultation_content(consultation_id)

    if not original_content:
        return {
            "similarity_score": 0,
            "matching_elements": [],
            "missing_elements": [],
            "feedback": "원본 상담 내용을 찾을 수 없습니다.",
            "error": "Original consultation not found"
        }

    # 2. LLM 기반 의미적 유사도 평가
    similarity_result = _calculate_semantic_similarity(
        simulation_transcript,
        original_content
    )

    return similarity_result


def _get_original_consultation_content(consultation_id: str) -> str:
    """
    DB에서 원본 상담 내용(consultation_documents.content) 조회

    Args:
        consultation_id: 상담 ID

    Returns:
        상담 내용 문자열 (없으면 None)
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT content
                FROM consultation_documents
                WHERE consultation_id = %s
                LIMIT 1
            """, (consultation_id,))
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        print(f"[SimilarityCalculator] DB 조회 오류: {e}")
        return None
    finally:
        conn.close()


def _calculate_semantic_similarity(
    simulation: str,
    original: str
) -> Dict[str, Any]:
    """
    LLM을 사용한 의미적 유사도 평가

    Args:
        simulation: 시뮬레이션 상담 전문
        original: 원본 상담 전문

    Returns:
        유사도 평가 결과 딕셔너리
    """
    system_prompt = """당신은 콜센터 상담 품질 평가 전문가입니다.
두 상담을 비교하여 유사도를 평가합니다.

평가 기준:
1. 문제 해결 접근 방식의 유사성
2. 사용된 표현과 화법의 적절성
3. 고객 응대 흐름의 일관성
4. 핵심 정보 전달 여부"""

    prompt = f"""다음 두 상담을 비교하여 유사도를 평가하세요:

[원본 상담 (우수 사례)]
{original[:2000]}

[시뮬레이션 상담]
{simulation[:2000]}

다음 JSON 형식으로 출력:
{{
    "similarity_score": 0-100 사이의 점수,
    "matching_elements": ["잘 수행된 요소1", "잘 수행된 요소2"],
    "missing_elements": ["누락된 요소1", "개선 필요 요소2"],
    "feedback": "종합 평가 피드백 (2-3문장)"
}}"""

    result = generate_json(prompt, system_prompt, temperature=0.2, max_tokens=400)

    if result and "similarity_score" in result:
        # 점수 범위 검증
        score = result.get("similarity_score", 0)
        if isinstance(score, (int, float)):
            result["similarity_score"] = max(0, min(100, int(score)))
        return result

    # LLM 실패 시 키워드 기반 유사도 폴백
    return _keyword_similarity_fallback(simulation, original)


def _keyword_similarity_fallback(
    simulation: str,
    original: str
) -> Dict[str, Any]:
    """
    LLM 실패 시 키워드 기반 Jaccard 유사도 계산

    Args:
        simulation: 시뮬레이션 상담 전문
        original: 원본 상담 전문

    Returns:
        유사도 평가 결과 딕셔너리
    """
    # 한글 단어만 추출
    sim_words = set(re.findall(r"[가-힣]+", simulation.lower()))
    orig_words = set(re.findall(r"[가-힣]+", original.lower()))

    # Jaccard 유사도 계산
    intersection = sim_words & orig_words
    union = sim_words | orig_words

    score = int((len(intersection) / len(union) * 100)) if union else 0

    # 일치/누락 요소 추출 (상위 10개씩)
    matching = list(intersection)[:10]
    missing = list(orig_words - sim_words)[:10]

    return {
        "similarity_score": score,
        "matching_elements": matching,
        "missing_elements": missing,
        "feedback": f"키워드 기반 분석 결과입니다. 원본과 {len(intersection)}개의 핵심 단어가 일치합니다."
    }
