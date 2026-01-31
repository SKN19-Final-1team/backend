"""
Kiwipiepy + PyKoSpacing 기반 형태소 분석 모듈

개선사항:
- Java 의존성 제거 (PyKomoran → Kiwipiepy)
- 2-3배 빠른 처리 속도
- 멀티스레딩 지원
- 띄어쓰기 자동 교정 (PyKoSpacing)
"""

import sys
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
from pathlib import Path

# Kiwipiepy
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    print("[WARNING] Kiwipiepy not installed. Run: pip install kiwipiepy")
    Kiwi = None
    KIWI_AVAILABLE = False

# PyKoSpacing
try:
    from pykospacing import Spacing
    SPACING_AVAILABLE = True
except ImportError:
    print("[WARNING] PyKoSpacing not installed. Run: pip install pykospacing")
    Spacing = None
    SPACING_AVAILABLE = False

# 프로젝트 루트 경로
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from app.llm.delivery.vocabulary_matcher import load_card_products


# 전역 인스턴스 (싱글톤)
_kiwi_instance: Optional[Kiwi] = None
_spacing_instance: Optional[Spacing] = None
_user_dict_loaded: bool = False


def get_kiwi() -> Optional[Kiwi]:
    """
    Kiwi 인스턴스 반환 (싱글톤)
    
    Returns:
        Kiwi 인스턴스 또는 None
    """
    global _kiwi_instance, _user_dict_loaded
    
    if not KIWI_AVAILABLE:
        return None
    
    if _kiwi_instance is None:
        try:
            print("[MorphologyAnalyzer] Kiwipiepy 초기화 중...")
            
            # 오타 교정 활성화 (기본 + 연철 오타)
            _kiwi_instance = Kiwi(
                typos='basic_with_continual',
                typo_cost_threshold=2.5
            )
            
            # 사용자 사전 로드
            if not _user_dict_loaded:
                try:
                    products = load_card_products()
                    count = 0
                    
                    for product in products:
                        keyword = product["keyword"]
                        _kiwi_instance.add_user_word(keyword, "NNP")
                        count += 1
                        
                        # 동의어도 추가
                        for syn in product.get("synonyms", []):
                            if syn and syn.strip():
                                _kiwi_instance.add_user_word(syn, "NNP")
                                count += 1
                    
                    # 고빈도 STT 오류 패턴 등록
                    register_common_stt_errors(_kiwi_instance)
                    
                    _user_dict_loaded = True
                    print(f"[MorphologyAnalyzer] 사용자 사전 {count}개 단어 로드 완료")
                    print(f"[MorphologyAnalyzer] STT 오류 패턴 등록 완료")
                    
                except Exception as e:
                    print(f"[MorphologyAnalyzer] 사용자 사전 로드 실패: {e}")
            
        except Exception as e:
            print(f"[MorphologyAnalyzer] Kiwipiepy 초기화 실패: {e}")
            return None
    
    return _kiwi_instance


def register_common_stt_errors(kiwi: Kiwi):
    """
    고빈도 STT 오류 패턴을 기분석 형태로 등록
    
    keywords_dict_refine.json 파일에서 교정 맵을 로드하여 등록
    
    Args:
        kiwi: Kiwi 인스턴스
    """
    import json
    import os
    
    registered_count = 0
    
    try:
        # JSON 파일 경로
        json_path = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'rag', 'vocab', 'keywords_dict_refine.json'
        )
        json_path = os.path.normpath(json_path)
        
        # JSON 파일 로드
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            correction_map = data.get("correction_map", {})
            
            # Step 1: 타겟 형태소 먼저 등록 (교정 결과 단어들)
            target_words = set(correction_map.values())
            for word in target_words:
                try:
                    # 한글 단어만 등록 (영문, 숫자 제외)
                    if word and any('\uac00' <= c <= '\ud7a3' for c in word):
                        kiwi.add_user_word(word, "NNG")
                except Exception:
                    pass  # 이미 존재하면 무시
            
            # Step 2: 기분석 형태 등록 (오류 → 교정)
            for error_form, correct_form in correction_map.items():
                # 같은 단어면 스킵
                if error_form == correct_form:
                    continue
                
                try:
                    # 기분석 형태 등록
                    result = kiwi.add_pre_analyzed_word(
                        error_form,
                        [(correct_form, "NNG", 0, len(error_form))],
                        -2
                    )
                    if result:
                        registered_count += 1
                except Exception:
                    pass  # 등록 실패해도 계속 진행
            
            print(f"[MorphologyAnalyzer] keywords_dict_refine.json에서 {registered_count}개 오류 패턴 등록")
        else:
            print(f"[MorphologyAnalyzer] JSON 파일 없음: {json_path}")
            # 기본 패턴 등록 (fallback)
            _register_default_patterns(kiwi)
            
    except Exception as e:
        print(f"[MorphologyAnalyzer] STT 오류 패턴 등록 실패: {e}")
        # 기본 패턴 등록 (fallback)
        _register_default_patterns(kiwi)


def _register_default_patterns(kiwi: Kiwi):
    """기본 STT 오류 패턴 등록 (fallback)"""
    try:
        target_words = [
            ("하나은행", "NNP"),
            ("연회비", "NNG"),
            ("바우처", "NNG"),
            ("익일", "NNG"),
        ]
        
        for word, tag in target_words:
            try:
                kiwi.add_user_word(word, tag)
            except Exception:
                pass
        
        patterns = [
            ("하나낸", [("하나은행", "NNP", 0, 3)], -2),
            ("연예비", [("연회비", "NNG", 0, 3)], -2),
            ("바우저", [("바우처", "NNG", 0, 3)], -2),
            ("이길영업일", [("익일", "NNG", 0, 2), ("영업일", "NNG", 2, 3)], -2),
        ]
        
        count = 0
        for form, analyzed, score in patterns:
            try:
                if kiwi.add_pre_analyzed_word(form, analyzed, score):
                    count += 1
            except Exception:
                pass
        
        print(f"[MorphologyAnalyzer] 기본 패턴 {count}개 등록 (fallback)")
        
    except Exception as e:
        print(f"[MorphologyAnalyzer] 기본 패턴 등록 실패: {e}")



def get_spacing() -> Optional[Spacing]:
    """
    Spacing 인스턴스 반환 (싱글톤)
    
    Returns:
        Spacing 인스턴스 또는 None
    """
    global _spacing_instance
    
    if not SPACING_AVAILABLE:
        return None
    
    if _spacing_instance is None:
        try:
            print("[MorphologyAnalyzer] PyKoSpacing 초기화 중...")
            _spacing_instance = Spacing()
        except Exception as e:
            print(f"[MorphologyAnalyzer] PyKoSpacing 초기화 실패: {e}")
            return None
    
    return _spacing_instance


@lru_cache(maxsize=512)
def analyze_morphemes(text: str) -> List[Tuple[str, str]]:
    """
    형태소 분석 수행 (Kiwipiepy)
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        [(형태소, 품사), ...] 리스트
        
    Example:
        >>> analyze_morphemes("나라사랑카드 바우처")
        [('나라사랑카드', 'NNP'), ('바우처', 'NNG')]
    """
    kiwi = get_kiwi()
    
    if kiwi is None:
        return [(text, "UNKNOWN")]
    

    try:
        # Step 1: 텍스트 레벨 교정 (correction_map 적용)
        corrected_text = apply_text_corrections(text)
        
        # Step 2: 띄어쓰기 교정 (선택적)
        spacing = get_spacing()
        processed_text = corrected_text
        
        if spacing:
            try:
                processed_text = spacing(corrected_text)
            except Exception as e:
                print(f"[MorphologyAnalyzer] 띄어쓰기 교정 실패: {e}")
                processed_text = corrected_text
        
        # Step 3: 형태소 분석
        tokens = kiwi.tokenize(processed_text)
        
        # (형태소, 품사) 튜플로 변환
        result = [(token.form, token.tag) for token in tokens]
        
        return result
        
    except Exception as e:
        print(f"[MorphologyAnalyzer] 분석 오류: {e}")
        return [(text, "UNKNOWN")]


# 글로벌 캐시: correction_map
_correction_map_cache = None


def get_correction_map() -> dict:
    """
    keywords_dict_refine.json에서 correction_map 로드 (캐시됨)
    
    Returns:
        {오류형태: 교정형태} 딕셔너리
    """
    global _correction_map_cache
    
    if _correction_map_cache is not None:
        return _correction_map_cache
    
    import json
    import os
    
    try:
        json_path = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'rag', 'vocab', 'keywords_dict_refine.json'
        )
        json_path = os.path.normpath(json_path)
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            _correction_map_cache = data.get("correction_map", {})
        else:
            _correction_map_cache = {}
            
    except Exception as e:
        print(f"[MorphologyAnalyzer] correction_map 로드 실패: {e}")
        _correction_map_cache = {}
    
    return _correction_map_cache


def apply_text_corrections(text: str) -> str:
    """
    텍스트 레벨에서 STT 오류 교정 적용
    
    correction_map의 오류 패턴을 교정된 형태로 치환
    
    Args:
        text: 입력 텍스트
        
    Returns:
        교정된 텍스트
        
    Example:
        >>> apply_text_corrections("하나낸 계좌에서 연예비 납부")
        "하나은행 계좌에서 연회비 납부"
    """
    correction_map = get_correction_map()
    
    if not correction_map:
        return text
    
    result = text
    
    # 긴 패턴부터 먼저 교정 (겹침 방지)
    sorted_patterns = sorted(correction_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    for error_form, correct_form in sorted_patterns:
        # 같은 단어면 스킵
        if error_form == correct_form:
            continue
        
        # 텍스트에 오류 패턴이 있으면 교정
        if error_form in result:
            result = result.replace(error_form, correct_form)
    
    return result


def extract_nouns(text: str) -> List[str]:
    """
    텍스트에서 명사만 추출
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        명사 리스트
        
    Example:
        >>> extract_nouns("나라사랑카드 바우처를 신청합니다")
        ['나라사랑카드', '바우처', '신청']
    """
    morphemes = analyze_morphemes(text)
    noun_tags = {'NNG', 'NNP', 'NNB'}
    return [morph for morph, pos in morphemes if pos in noun_tags]


def extract_card_product_candidates(text: str) -> List[str]:
    """
    카드상품명 후보 추출
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        카드상품명 후보 리스트
        
    Example:
        >>> extract_card_product_candidates("나라사랑카드 바우처")
        ['나라사랑카드']
    """
    morphemes = analyze_morphemes(text)
    candidates = []
    
    # 1. 고유명사(NNP) - 사용자 사전에 등록된 카드상품명
    for morph, pos in morphemes:
        if pos == 'NNP':
            candidates.append(morph)
    
    # 2. 복합명사 처리 (연속된 명사 결합)
    noun_tags = {'NNG', 'NNP', 'NNB'}
    current_compound = []
    
    for morph, pos in morphemes:
        if pos in noun_tags:
            current_compound.append(morph)
        else:
            if len(current_compound) >= 2:
                compound = ''.join(current_compound)
                candidates.append(compound)
            current_compound = []
            
    # 마지막 복합명사 처리
    if len(current_compound) >= 2:
        compound = ''.join(current_compound)
        candidates.append(compound)
    
    # 중복 제거
    return list(set(candidates))


def normalize_with_morphology(text: str) -> str:
    """
    형태소 분석 기반 정규화 (명사만 추출)
    
    Args:
        text: 정규화할 텍스트
    
    Returns:
        명사만 공백으로 연결된 문자열
        
    Example:
        >>> normalize_with_morphology("나라사랑카드를 신청합니다")
        "나라사랑카드 신청"
    """
    nouns = extract_nouns(text)
    return ' '.join(nouns)


def get_user_dict_stats() -> Dict[str, any]:
    """
    사용자 사전 통계 반환
    
    Returns:
        사전 정보 딕셔너리
    """
    if not _user_dict_loaded:
        return {
            "exists": False,
            "engine": "Kiwipiepy",
            "kiwi_loaded": _kiwi_instance is not None,
            "spacing_loaded": _spacing_instance is not None
        }
    
    try:
        products = load_card_products()
        total_words = len(products)
        
        # 동의어 포함 총 단어 수 계산
        for product in products:
            total_words += len(product.get("synonyms", []))
        
        return {
            "exists": True,
            "engine": "Kiwipiepy",
            "products": len(products),
            "total_words": total_words,
            "kiwi_loaded": _kiwi_instance is not None,
            "spacing_loaded": _spacing_instance is not None
        }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e),
            "engine": "Kiwipiepy"
        }


# 멀티스레딩 지원 (대량 처리용)
def analyze_morphemes_batch(texts: List[str], num_workers: int = 4) -> List[List[Tuple[str, str]]]:
    """
    대량 텍스트 형태소 분석 (멀티스레딩)
    
    Args:
        texts: 텍스트 리스트
        num_workers: 워커 수 (기본값: 4)
    
    Returns:
        각 텍스트의 형태소 분석 결과 리스트
        
    Example:
        >>> texts = ["나라사랑카드", "신세계상품권"]
        >>> analyze_morphemes_batch(texts, num_workers=2)
        [[('나라사랑카드', 'NNP')], [('신세계상품권', 'NNP')]]
    """
    kiwi = get_kiwi()
    
    if kiwi is None:
        return [[(text, "UNKNOWN")] for text in texts]
    
    try:
        # 띄어쓰기 교정 (선택적)
        spacing = get_spacing()
        processed_texts = texts
        
        if spacing:
            try:
                processed_texts = [spacing(text) for text in texts]
            except Exception as e:
                print(f"[MorphologyAnalyzer] 배치 띄어쓰기 교정 실패: {e}")
                processed_texts = texts
        
        # 멀티스레딩 분석
        results = kiwi.tokenize(processed_texts, num_workers=num_workers)
        
        # 변환
        return [
            [(token.form, token.tag) for token in tokens]
            for tokens in results
        ]
        
    except Exception as e:
        print(f"[MorphologyAnalyzer] 배치 분석 오류: {e}")
        return [[(text, "UNKNOWN")] for text in texts]


# 하위 호환성 함수 (기존 코드 지원)
def create_user_dictionary() -> str:
    """
    하위 호환성 함수 (더 이상 사용되지 않음)
    
    Kiwipiepy는 메모리에 직접 사전을 로드하므로
    파일 경로를 반환할 필요가 없음
    
    Returns:
        빈 문자열
    """
    print("[MorphologyAnalyzer] create_user_dictionary() is deprecated with Kiwipiepy")
    return ""


if __name__ == "__main__":
    # 간단한 테스트
    print("=== Kiwipiepy + PyKoSpacing 테스트 ===\n")
    
    test_cases = [
        "나라사랑카드바우처신청",
        "연예비 납부와 그 바우저",
        "신세계상품권 등록",
    ]
    
    for text in test_cases:
        print(f"입력: {text}")
        morphemes = analyze_morphemes(text)
        print(f"형태소: {morphemes}")
        nouns = extract_nouns(text)
        print(f"명사: {nouns}")
        candidates = extract_card_product_candidates(text)
        print(f"카드상품명 후보: {candidates}")
        print("-" * 70)
    
    # 사용자 사전 통계
    stats = get_user_dict_stats()
    print(f"\n사용자 사전 통계: {stats}")