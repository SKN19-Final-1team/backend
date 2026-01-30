"""
전체 흐름:
1. STT 전사
2. 형태소 분석 - morphology_analyzer
3. 단어 매칭 및 교정 - vocabulary_matcher
4. 최종 문장 생성 및 저장 - sllm_refiner
"""

from typing import Dict, List, Tuple, Optional
from app.utils.runpod_connector import call_runpod

# 형태소 분석기
try:
    from app.llm.delivery.morphology_analyzer import (
        extract_nouns,
        extract_card_product_candidates,
        analyze_morphemes
    )
    MORPHOLOGY_AVAILABLE = True
except ImportError:
    MORPHOLOGY_AVAILABLE = False
    print("[Deliverer] 형태소 분석기 없음")

# 단어사전 매칭
try:
    from app.llm.delivery.vocabulary_matcher import (
        find_candidates,
        get_best_match
    )
    VOCABULARY_MATCHER_AVAILABLE = True
except ImportError:
    VOCABULARY_MATCHER_AVAILABLE = False
    print("[Deliverer] 단어사전 매칭 없음")

# sLLM 교정
from app.llm.delivery.sllm_refiner import (
    refinement_payload,
    parse_refinement_result
)


# 형태소 분석
def analyze_and_extract(text: str) -> Dict[str, any]:
    """
    교정할 대상을 좁히기 위해 명사(특히 고유명사)를 추출
    
    Args:
        text: STT로부터 받은 원본 텍스트
    
    Returns:
        {
            "morphemes": [(형태소, 품사), ...],
            "nouns": [명사 리스트],
            "card_candidates": [카드상품명 후보]
        }
    """
    if not MORPHOLOGY_AVAILABLE:
        return {
            "morphemes": [],
            "nouns": [],
            "card_candidates": []
        }
    
    try:
        morphemes = analyze_morphemes(text)
        nouns = extract_nouns(text)
        card_candidates = extract_card_product_candidates(text)
        
        return {
            "morphemes": morphemes,
            "nouns": nouns,
            "card_candidates": card_candidates
        }
    except Exception as e:
        print(f"[Deliverer] 형태소 분석 실패: {e}")
        return {
            "morphemes": [],
            "nouns": [],
            "card_candidates": []
        }


# 매칭 및 교정
def match_and_correct(text: str, candidates: List[str]) -> Dict[str, any]:
    """
    추출된 명사가 DB에 있는 정확한 카드상품명인지 확인하고 교정
    
    Args:
        text: 원본 텍스트
        candidates: 형태소 분석으로 추출된 카드상품명 후보
    
    Returns:
        {
            "matches": [(후보명, 유사도), ...],
            "best_match": 최적 매칭 결과 or None,
            "corrections": {원본: 교정본} 매핑
        }
    """
    if not VOCABULARY_MATCHER_AVAILABLE:
        return {
            "matches": [],
            "best_match": None,
            "corrections": {}
        }
    
    try:
        # 전체 텍스트에 대한 매칭
        matches = find_candidates(text, top_k=3, threshold=0.5)
        best_match = get_best_match(text, confidence_threshold=0.85)
        
        # 각 후보에 대한 교정 매핑 생성
        corrections = {}
        for candidate in candidates:
            candidate_matches = find_candidates(candidate, top_k=1, threshold=0.7)
            if candidate_matches:
                corrected_name, score = candidate_matches[0]
                if score >= 0.75:  # 충분히 높은 유사도
                    corrections[candidate] = corrected_name
        
        return {
            "matches": matches,
            "best_match": best_match,
            "corrections": corrections
        }
    except Exception as e:
        print(f"[Deliverer] 단어 매칭 실패: {e}")
        return {
            "matches": [],
            "best_match": None,
            "corrections": {}
        }


# 교정된 것을 텍스트에 적용
def apply_corrections(text: str, corrections: Dict[str, str]) -> str:
    """
    교정 매핑을 텍스트에 적용
    
    Args:
        text: 원본 텍스트
        corrections: {원본: 교정본} 매핑
    
    Returns:
        교정된 텍스트
    """
    corrected_text = text
    for original, corrected in corrections.items():
        corrected_text = corrected_text.replace(original, corrected)
    return corrected_text


# sLLM 교정
def refine_with_sllm(text: str, context: Optional[Dict] = None) -> Dict[str, any]:
    """
    sLLM이 문맥과 문법 오류를 다듬어 최종 교정
    
    Args:
        text: 단어 교정이 적용된 텍스트
        context: 추가 컨텍스트 (형태소 분석 결과 등)
    
    Returns:
        {
            "text": 최종 교정된 텍스트,
            "keywords": 추출된 키워드
        }
    """
    if not text or not text.strip():
        return {"text": text, "keywords": []}
    
    try:
        payload = refinement_payload(text)
        llm_output = call_runpod(payload)
        return parse_refinement_result(llm_output, text)
    except Exception as e:
        print(f"[Deliverer] sLLM 교정 실패: {e}")
        return {"text": text, "keywords": []}


def pipeline(text: str, use_sllm: bool = True) -> Dict[str, any]:
    """
    통합 텍스트 교정 파이프라인
    
    전체 흐름:
    1. STT 전사 (입력)
    2. 형태소 분석 (Targeting)
    3. 단어 매칭 및 교정 (Vocabulary Matching)
    4. 최종 문장 생성 (sLLM Refining)
    
    Args:
        text: STT로부터 받은 원본 텍스트
        use_sllm: sLLM 교정 사용 여부
    
    Returns:
        {
            "original": 원본 텍스트,
            "step2_morphology": 형태소 분석 결과,
            "step3_matching": 단어 매칭 결과,
            "step3_corrected": 단어 교정 적용 텍스트,
            "step4_refined": sLLM 최종 교정 텍스트,
            "keywords": 추출된 키워드
        }
    """
    # 1. 입력 (STT 전사)
    original_text = text
    
    # 2. 형태소 분석
    morphology_result = analyze_and_extract(text)
    
    # 3. 단어 매칭 및 교정
    matching_result = match_and_correct(text, morphology_result.get("card_candidates", []))
    corrected_text = apply_corrections(text, matching_result.get("corrections", {}))
    
    # 4. 최종 문장 생성
    if use_sllm:
        refine_result = refine_with_sllm(corrected_text, context=morphology_result)
        final_text = refine_result["text"]
        keywords = refine_result.get("keywords", [])
    else:
        final_text = corrected_text
        keywords = []
    
    return {
        "original": original_text,
        "step2_morphology": morphology_result,
        "step3_matching": matching_result,
        "step3_corrected": corrected_text,
        "step4_refined": final_text,
        "keywords": keywords
    }


def refine_text(text: str) -> Dict[str, any]:
    result = pipeline(text, use_sllm=True)
    return {
        "text": result["step4_refined"],
        "keywords": result["keywords"]
    }


def deliver(text: str) -> Dict[str, any]:
    refine_result = refine_text(text)
    refined_text = refine_result["text"]
    
    return {
        "original": text,
        "refined": refined_text,
        "keywords": refine_result.get("keywords", [])
    }
