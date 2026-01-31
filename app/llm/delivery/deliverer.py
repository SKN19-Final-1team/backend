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
        
        # 카드상품명 교정 비활성화 (사용자 요청)
        # 감지만 하고 교체하지 않음
        corrections = {}
        
        # [비활성화] 각 후보에 대한 교정 매핑 생성
        # for candidate in candidates:
        #     candidate_matches = find_candidates(candidate, top_k=1, threshold=0.7)
        #     if candidate_matches:
        #         corrected_name, score = candidate_matches[0]
        #         if score >= 0.75:  # 충분히 높은 유사도
        #             corrections[candidate] = corrected_name
        
        return {
            "matches": matches,
            "best_match": best_match,
            "corrections": corrections  # 항상 빈 딕셔너리
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
    2. 텍스트 레벨 교정 (correction_map - keywords_dict_refine.json)
    3. 형태소 분석 (Kiwipiepy)
    4. 단어 매칭 및 교정 (카드상품명 등)
    5. 최종 문장 생성 (sLLM - 문맥 기반 교정)
    
    Args:
        text: STT로부터 받은 원본 텍스트
        use_sllm: sLLM 교정 사용 여부
    
    Returns:
        {
            "original": 원본 텍스트,
            "step1_corrected": correction_map 적용 후 텍스트,
            "step2_morphology": 형태소 분석 결과,
            "step3_matching": 단어 매칭 결과,
            "refined": 최종 교정 텍스트,
            "keywords": 추출된 키워드
        }
    """
    from app.llm.delivery.morphology_analyzer import apply_text_corrections
    
    # 1. 입력 (STT 전사)
    original_text = text
    
    # 2. 텍스트 레벨 교정 (correction_map)
    step1_corrected = apply_text_corrections(text)
    
    # 3. 형태소 분석
    morphology_result = analyze_and_extract(step1_corrected)
    
    # 4. 단어 매칭 및 교정
    matching_result = match_and_correct(step1_corrected, morphology_result.get("card_candidates", []))
    corrected_text = apply_corrections(step1_corrected, matching_result.get("corrections", {}))
    
    # 5. sLLM 최종 교정 (문맥 기반)
    if use_sllm:
        refine_result = refine_with_sllm(corrected_text, context=morphology_result)
        final_text = refine_result["text"]
        keywords = refine_result.get("keywords", [])
    else:
        final_text = corrected_text
        keywords = []
    
    return {
        "original": original_text,
        "step1_corrected": step1_corrected,
        "step2_morphology": morphology_result,
        "step3_matching": matching_result,
        "refined": final_text,
        "keywords": keywords
    }


def refine_text(text: str) -> Dict[str, any]:
    """
    텍스트 교정 (파이프라인 래퍼)
    
    Args:
        text: 교정할 텍스트
    
    Returns:
        {
            "text": 교정된 텍스트,
            "keywords": 추출된 키워드
        }
    """
    result = pipeline(text, use_sllm=True)
    return {
        "text": result["step4_refined"],
        "keywords": result["keywords"]
    }


def deliver(text: str) -> Dict[str, any]:
    """
    텍스트 교정 및 전달 (레거시 호환)
    
    Args:
        text: 교정할 텍스트
    
    Returns:
        {
            "original": 원본 텍스트,
            "refined": 교정된 텍스트,
            "keywords": 추출된 키워드
        }
    """
    refine_result = refine_text(text)
    refined_text = refine_result["text"]
    
    return {
        "original": text,
        "refined": refined_text,
        "keywords": refine_result.get("keywords", [])
    }


def refine_conversation_text(text: str, use_sllm: bool = True) -> Dict[str, any]:
    """
    긴 대화 텍스트를 교정합니다.
    
    상담사와 고객의 발화가 구분 없이 이어붙어진 텍스트를 입력받아
    컨텍스트 인식 교정을 수행합니다.
    
    Args:
        text: 상담사와 고객의 발화가 이어붙어진 텍스트
        use_sllm: sLLM 교정 사용 여부 (기본값: True)
    
    Returns:
        {
            "original": 원본 텍스트,
            "refined": 최종 교정된 텍스트,
            "step2_morphology": 형태소 분석 결과,
            "step3_matching": 단어 매칭 결과,
            "step3_corrected": 단어 교정 적용 텍스트,
            "keywords": 추출된 키워드
        }
    
    Example:
        >>> text = "상담원 안수희입니다 나라사람카드 바우처 때문에"
        >>> result = refine_conversation_text(text)
        >>> print(result["refined"])
        "상담원 안수이입니다 나라사랑카드 바우처 때문에"
    """
    if not text or not text.strip():
        return {
            "original": text,
            "refined": text,
            "step2_morphology": {},
            "step3_matching": {},
            "step3_corrected": text,
            "keywords": []
        }
    
    # 파이프라인 실행
    result = pipeline(text, use_sllm=use_sllm)
    
    return {
        "original": result["original"],
        "refined": result["step4_refined"],
        "step2_morphology": result["step2_morphology"],
        "step3_matching": result["step3_matching"],
        "step3_corrected": result["step3_corrected"],
        "keywords": result.get("keywords", [])
    }


def interactive_refinement():
    """
    대화형 텍스트 교정 인터페이스
    
    사용자가 텍스트를 입력하면 즉시 교정 결과를 출력합니다.
    """
    print("=" * 70)
    print("대화 텍스트 교정 시스템")
    print("=" * 70)
    print("\n사용법:")
    print("1. 상담사와 고객의 발화가 이어진 텍스트를 입력하세요")
    print("2. 입력 완료 후 Enter를 두 번 누르면 교정이 시작됩니다")
    print("3. 'quit' 또는 'exit'를 입력하면 종료됩니다")
    print("=" * 70)
    
    while True:
        print("\n텍스트 입력 (완료 후 빈 줄 입력):")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            if line.strip().lower() in ["quit", "exit"]:
                print("\n프로그램을 종료합니다.")
                return
            lines.append(line)
        
        if not lines:
            print("텍스트가 입력되지 않았습니다. 다시 시도하세요.")
            continue
        
        text = " ".join(lines)
        
        print("\n" + "=" * 70)
        print("교정 중...")
        print("=" * 70)
        
        try:
            result = refine_conversation_text(text, use_sllm=True)
            
            print("\n[원본 텍스트]")
            print(result["original"])
            
            print("\n[교정된 텍스트]")
            print(result["refined"])
            
            print("\n[감지된 카드상품명]")
            card_candidates = result["step2_morphology"].get("card_candidates", [])
            if card_candidates:
                print(", ".join(card_candidates))
            else:
                print("없음")
            
            print("\n[단어 매칭 결과]")
            best_match = result["step3_matching"].get("best_match")
            if best_match:
                print(f"최적 매칭: {best_match}")
            else:
                print("매칭 없음")
            
            print("\n[추출된 키워드]")
            keywords = result.get("keywords", [])
            if keywords:
                print(", ".join(keywords))
            else:
                print("없음")
            
            print("\n" + "=" * 70)
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # 예시 테스트
    print("=== 예시 테스트 ===\n")
    
    test_cases = [
        "상담원 안수희입니다 무엇을 도와드릴까요 아 예 안녕하세요 나라사람카드 바우처 때문에 전화드렸는데요",
        "연예비 납부와 그 바우저 한개 선택하라고 해서 신청할려구요",
        "이길영업일날 문자로 발송소리가 될 것 같고요",
        "신세계상품권 등록 좀 해주세요"
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}]")
        print(f"입력: {test_text}")
        
        result = refine_conversation_text(test_text)
        print(f"출력: {result['refined']}")
        print("-" * 70)
    
    # 대화형 모드 시작
    print("\n\n대화형 모드를 시작하시겠습니까? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        interactive_refinement()
    else:
        print("\n프로그램을 종료합니다.")
