# """
# KOMORAN 형태소 분석기 통합 모듈

# 카드상품명 등 고유명사를 사용자사전에 등록하여
# 형태소 분석 정확도를 높이고, 조사 제거 및 복합명사 처리
# """

# import os
# import tempfile

# from typing import List, Dict, Optional, Tuple
# from functools import lru_cache
# from pathlib import Path

# os.environ["JAVA_TOOL_OPTIONS"] = (
#     "--add-opens=java.base/java.lang=ALL-UNNAMED "
#     "--add-opens=java.base/java.util=ALL-UNNAMED "
#     "--add-opens=java.base/java.io=ALL-UNNAMED"
# )

# try:
#     from PyKomoran import Komoran
# except ImportError:
#     print("[WARNING] PyKomoran not installed. Run: pip install PyKomoran")
#     Komoran = None

# # 프로젝트 루트 경로
# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
# from app.llm.delivery.vocabulary_matcher import load_card_products


# # 전역 KOMORAN 인스턴스 (싱글톤)
# _komoran_instance: Optional[Komoran] = None
# _user_dict_path: Optional[str] = None


# def create_user_dictionary() -> str:
#     """
#     DB에서 카드상품명을 로드하여 KOMORAN 사용자사전 파일 생성
    
#     Returns:
#         사용자사전 파일 경로
#     """
#     products = load_card_products()
    
#     # 임시 디렉토리에 사용자사전 파일 생성
#     temp_dir = tempfile.gettempdir()
#     user_dict_path = os.path.join(temp_dir, "komoran_card_products.txt")
    
#     with open(user_dict_path, 'w', encoding='utf-8') as f:
#         for product in products:
#             keyword = product["keyword"]
#             # KOMORAN 사용자사전 형식: 단어\t품사
#             # NNP: 고유명사
#             f.write(f"{keyword}\tNNP\n")
            
#             # 동의어도 추가
#             for syn in product.get("synonyms", []):
#                 if syn:
#                     f.write(f"{syn}\tNNP\n")
    
#     print(f"[MorphologyAnalyzer] 사용자사전 생성: {len(products)}개 카드상품명")
#     print(f"[MorphologyAnalyzer] 사전 경로: {user_dict_path}")
    
#     return user_dict_path


# def get_komoran() -> Optional[Komoran]:
#     """
#     KOMORAN 인스턴스 반환 (싱글톤 패턴)
    
#     Returns:
#         Komoran 인스턴스 또는 None
#     """
#     global _komoran_instance, _user_dict_path
    
#     if Komoran is None:
#         print("[MorphologyAnalyzer] PyKomoran 패키지가 설치되지 않았습니다.")
    
#     if _komoran_instance is None:
#         try:
#             # 사용자사전 생성
#             _user_dict_path = create_user_dictionary()
            
#             # KOMORAN 초기화 (STABLE 모델)
#             print(f"[MorphologyAnalyzer] KOMORAN 초기화 중...")
#             _komoran_instance = Komoran("STABLE")
            
#             # 사용자사전 설정
#             print(f"[MorphologyAnalyzer] 사용자사전 로드 중: {_user_dict_path}")
#             _komoran_instance.set_user_dic(_user_dict_path)
#             print("[MorphologyAnalyzer] KOMORAN 초기화 완료")
#         except Exception as e:
#             print(f"[MorphologyAnalyzer] KOMORAN 초기화 실패: {e}")
#             import traceback
#             traceback.print_exc()
#             return None
    
#     return _komoran_instance


# @lru_cache(maxsize=512)
# def analyze_morphemes(text: str) -> List[Tuple[str, str]]:
#     """
#     형태소 분석 수행 (캐싱)
    
#     Args:
#         text: 분석할 텍스트
    
#     Returns:
#         [(형태소, 품사), ...] 리스트
#     """
#     komoran = get_komoran()
    
#     if komoran is None:
#         # KOMORAN 없으면 원본 그대로 반환
#         print(f"[MorphologyAnalyzer] KOMORAN 없음, 원본 반환: {text}")
#         return [(text, "UNKNOWN")]
    
#     try:
#         result = komoran.pos(text)
#         print(f"[MorphologyAnalyzer] 분석 결과 타입: {type(result)}")
        
#         if not result:
#             print(f"[MorphologyAnalyzer] 빈 결과 반환")
#             return []
        
#         print(f"[MorphologyAnalyzer] 첫 번째 아이템 타입: {type(result[0])}")
#         print(f"[MorphologyAnalyzer] 첫 번째 아이템: {result[0]}")
        
#         # Token 객체를 (형태소, 품사) 튜플로 변환
#         morphemes = []
#         for item in result:
#             if hasattr(item, 'first') and hasattr(item, 'second'):
#                 # Token 객체인 경우
#                 morphemes.append((item.first, item.second))
#             elif isinstance(item, tuple) and len(item) == 2:
#                 # 이미 튜플인 경우
#                 morphemes.append(item)
#             else:
#                 print(f"[MorphologyAnalyzer] 알 수 없는 형식: {type(item)} - {item}")
        
#         print(f"[MorphologyAnalyzer] 변환된 형태소: {morphemes}")
#         return morphemes
            
#     except Exception as e:
#         print(f"[MorphologyAnalyzer] 분석 실패: {e}")
#         import traceback
#         traceback.print_exc()
#         return [(text, "UNKNOWN")]


# def extract_nouns(text: str) -> List[str]:
#     """
#     텍스트에서 명사만 추출
    
#     Args:
#         text: 입력 텍스트
    
#     Returns:
#         명사 리스트
#     """
#     morphemes = analyze_morphemes(text)
    
#     # 명사 품사: NNG(일반명사), NNP(고유명사), NNB(의존명사)
#     noun_tags = {'NNG', 'NNP', 'NNB'}
    
#     nouns = [morph for morph, pos in morphemes if pos in noun_tags]
#     return nouns


# def extract_card_product_candidates(text: str) -> List[str]:
#     """
#     텍스트에서 카드상품명 후보 추출
    
#     형태소 분석 결과에서 고유명사(NNP)를 우선 추출하고,
#     연속된 명사를 결합하여 복합명사 후보도 생성
    
#     Args:
#         text: 입력 텍스트
    
#     Returns:
#         카드상품명 후보 리스트
#     """
#     morphemes = analyze_morphemes(text)
    
#     candidates = []
    
#     # morphemes는 이미 (형태소, 품사) 튜플 리스트로 변환됨
#     # 1. 고유명사(NNP) 직접 추출 - 사용자사전에 등록된 카드상품명
#     for morph, pos in morphemes:
#         if pos == 'NNP':
#             candidates.append(morph)
    
#     # 2. 연속된 명사 결합 (복합명사 처리)
#     noun_tags = {'NNG', 'NNP', 'NNB'}
#     current_compound = []
    
#     for morph, pos in morphemes:
#         if pos in noun_tags:
#             current_compound.append(morph)
#         else:
#             if len(current_compound) >= 2:
#                 # 2개 이상 명사가 연속되면 결합
#                 candidates.append(''.join(current_compound))
#             current_compound = []
    
#     # 마지막 복합명사 처리
#     if len(current_compound) >= 2:
#         candidates.append(''.join(current_compound))
    
#     return list(set(candidates))  # 중복 제거


# def normalize_with_morphology(text: str) -> str:
#     """
#     형태소 분석 기반 텍스트 정규화
    
#     조사 제거 및 명사만 추출하여 정규화된 텍스트 생성
    
#     Args:
#         text: 입력 텍스트
    
#     Returns:
#         정규화된 텍스트
#     """
#     nouns = extract_nouns(text)
#     return ' '.join(nouns)


# def get_user_dict_stats() -> Dict[str, any]:
#     """
#     사용자사전 통계 반환
    
#     Returns:
#         통계 정보 딕셔너리
#     """
#     global _user_dict_path
    
#     if _user_dict_path is None or not os.path.exists(_user_dict_path):
#         return {"exists": False}
    
#     with open(_user_dict_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
    
#     return {
#         "exists": True,
#         "path": _user_dict_path,
#         "entries": len(lines),
#         "komoran_loaded": _komoran_instance is not None
#     }

"""
KOMORAN 형태소 분석기 통합 모듈
수정사항:
1. JAVA_TOOL_OPTIONS 환경변수 추가 (Java 17+ 호환성)
2. Token 객체 처리 로직 추가 (TypeError 해결)
"""

import os
import tempfile
import sys
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
from pathlib import Path

# [중요] PyKomoran 임포트 전 JVM 옵션 설정
os.environ["JAVA_TOOL_OPTIONS"] = (
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED"
)

try:
    from PyKomoran import Komoran
except ImportError:
    print("[WARNING] PyKomoran not installed. Run: pip install PyKomoran")
    Komoran = None

# 프로젝트 루트 경로 설정 (기존 코드 유지)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from app.llm.delivery.vocabulary_matcher import load_card_products


# 전역 KOMORAN 인스턴스 (싱글톤)
_komoran_instance: Optional[Komoran] = None
_user_dict_path: Optional[str] = None


def create_user_dictionary() -> str:
    """DB에서 카드상품명을 로드하여 사용자사전 생성"""
    products = load_card_products()
    temp_dir = tempfile.gettempdir()
    user_dict_path = os.path.join(temp_dir, "komoran_card_products.txt")
    
    with open(user_dict_path, 'w', encoding='utf-8') as f:
        for product in products:
            keyword = product["keyword"]
            f.write(f"{keyword}\tNNP\n")
            for syn in product.get("synonyms", []):
                if syn:
                    f.write(f"{syn}\tNNP\n")
    
    return user_dict_path


def get_komoran() -> Optional[Komoran]:
    """KOMORAN 인스턴스 반환 (싱글톤)"""
    global _komoran_instance, _user_dict_path
    
    if Komoran is None:
        return None
    
    if _komoran_instance is None:
        try:
            _user_dict_path = create_user_dictionary()
            print(f"[MorphologyAnalyzer] KOMORAN 초기화 (UserDict: {_user_dict_path})")
            _komoran_instance = Komoran("STABLE")
            _komoran_instance.set_user_dic(_user_dict_path)
        except Exception as e:
            print(f"[MorphologyAnalyzer] 초기화 실패: {e}")
            return None
    
    return _komoran_instance


@lru_cache(maxsize=512)
def analyze_morphemes(text: str) -> List[Tuple[str, str]]:
    """
    형태소 분석 수행
    수정: Token 객체를 (형태소, 품사) 튜플로 변환하여 반환
    """
    komoran = get_komoran()
    
    if komoran is None:
        return [(text, "UNKNOWN")]
    
    try:
        # komoran.pos()는 Token 객체의 리스트를 반환함
        tokens = komoran.pos(text)
        
        # [핵심 수정] Token 객체에서 get_morph(), get_pos()로 값을 꺼내 튜플로 변환
        result = []
        for token in tokens:
            # PyKomoran의 Token 객체 메서드 사용
            morph = token.get_morph()
            pos = token.get_pos()
            result.append((morph, pos))
            
        return result
        
    except Exception as e:
        print(f"[MorphologyAnalyzer] 분석 오류: {e}")
        # 오류 발생 시 원본 반환하여 파이프라인 중단 방지
        return [(text, "UNKNOWN")]


def extract_nouns(text: str) -> List[str]:
    """텍스트에서 명사만 추출"""
    # analyze_morphemes가 이제 튜플 리스트를 반환하므로 그대로 사용 가능
    morphemes = analyze_morphemes(text)
    noun_tags = {'NNG', 'NNP', 'NNB'}
    return [morph for morph, pos in morphemes if pos in noun_tags]


def extract_card_product_candidates(text: str) -> List[str]:
    """카드상품명 후보 추출"""
    morphemes = analyze_morphemes(text)
    candidates = []
    
    # 1. 고유명사(NNP)
    for morph, pos in morphemes:
        if pos == 'NNP':
            candidates.append(morph)
    
    # 2. 복합명사 처리
    noun_tags = {'NNG', 'NNP', 'NNB'}
    current_compound = []
    
    for morph, pos in morphemes:
        if pos in noun_tags:
            current_compound.append(morph)
        else:
            if len(current_compound) >= 2:
                candidates.append(''.join(current_compound))
            current_compound = []
            
    if len(current_compound) >= 2:
        candidates.append(''.join(current_compound))
    
    return list(set(candidates))


def normalize_with_morphology(text: str) -> str:
    nouns = extract_nouns(text)
    return ' '.join(nouns)


def get_user_dict_stats() -> Dict[str, any]:
    global _user_dict_path
    if _user_dict_path is None or not os.path.exists(_user_dict_path):
        return {"exists": False}
    with open(_user_dict_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return {
        "exists": True, 
        "path": _user_dict_path, 
        "entries": len(lines),
        "komoran_loaded": _komoran_instance is not None
    }