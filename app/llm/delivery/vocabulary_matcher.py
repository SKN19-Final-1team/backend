"""
카드상품명 어휘 매칭 모듈

DB에서 카드상품명을 로드하고 발음 유사도 및 편집거리 기반으로
입력 텍스트에서 정확한 상품명을 찾아내는 고속 매칭 엔진
"""

import re
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import Levenshtein
from jamo import h2j, j2hcj

# DB 연결
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from app.db.scripts.modules.connect_db import connect_db

# 형태소 분석기 (선택적)
try:
    from app.llm.delivery.morphology_analyzer import (
        extract_card_product_candidates,
        normalize_with_morphology
    )
    MORPHOLOGY_AVAILABLE = True
except ImportError:
    MORPHOLOGY_AVAILABLE = False
    print("[VocabularyMatcher] 형태소 분석기 없음 (선택사항)")


# 전역 캐시
_CARD_PRODUCTS_CACHE: Optional[List[Dict]] = None


def load_card_products(force_reload: bool = False) -> List[Dict]:
    """
    DB에서 카드상품명 로드 및 캐싱
    
    Args:
        force_reload: 캐시 무시하고 재로드
    
    Returns:
        카드상품명 리스트 [{"id": int, "keyword": str, "category": str}, ...]
    """
    global _CARD_PRODUCTS_CACHE
    
    if _CARD_PRODUCTS_CACHE is not None and not force_reload:
        return _CARD_PRODUCTS_CACHE
    
    conn = connect_db()
    cursor = conn.cursor()
    
    try:
        # id >= 2484인 카드상품 키워드 조회
        query = """
            SELECT id, keyword, category, synonyms, variations
            FROM keyword_dictionary
            WHERE id >= 2484 AND category = '카드상품'
            ORDER BY id
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        products = []
        for row in rows:
            products.append({
                "id": row[0],
                "keyword": row[1],
                "category": row[2],
                "synonyms": row[3] or [],
                "variations": row[4] or []
            })
        
        _CARD_PRODUCTS_CACHE = products
        print(f"[VocabularyMatcher] 카드상품명 {len(products)}개 로드 완료")
        
        return products
        
    except Exception as e:
        print(f"[VocabularyMatcher] DB 로드 실패: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def normalize_text(text: str) -> str:
    """
    텍스트 정규화: 띄어쓰기 제거, 소문자 변환
    """
    return re.sub(r'\s+', '', text).lower()


@lru_cache(maxsize=1024)
def decompose_hangul(text: str) -> str:
    """
    한글을 자모 단위로 분해 (캐싱)
    
    Example:
        "테디카드" -> "ㅌㅔㄷㅣㅋㅏㄷㅡ"
    """
    try:
        return j2hcj(h2j(text))
    except:
        return text


def phonetic_similarity(text1: str, text2: str) -> float:
    """
    발음 유사도 계산 (0.0 ~ 1.0)
    
    자모 분해 후 편집거리 기반 유사도 계산
    """
    # 정규화
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)
    
    # 자모 분해
    jamo1 = decompose_hangul(t1)
    jamo2 = decompose_hangul(t2)
    
    # 편집거리 계산
    distance = Levenshtein.distance(jamo1, jamo2)
    max_len = max(len(jamo1), len(jamo2))
    
    if max_len == 0:
        return 1.0
    
    # 유사도로 변환 (1 - normalized_distance)
    similarity = 1.0 - (distance / max_len)
    return similarity


def find_candidates(
    query: str,
    top_k: int = 3,
    threshold: float = 0.6,
    use_morphology: bool = True
) -> List[Tuple[str, float]]:
    """
    입력 쿼리에 대한 카드상품명 후보 추출
    
    개선사항:
    - 부분 매칭: "아이", "플러스" → "아이사랑 플러스 카드"
    - 키워드 조합 점수: 여러 키워드가 포함될수록 높은 점수
    
    Args:
        query: 입력 텍스트 (예: "아이 키우는데 무슨 플러스")
        top_k: 반환할 최대 후보 수
        threshold: 최소 유사도 임계값
        use_morphology: 형태소 분석 사용 여부
    
    Returns:
        [(상품명, 유사도), ...] 리스트 (유사도 내림차순)
    """
    products = load_card_products()
    
    if not products:
        return []
    
    candidates = []
    
    # 불용어 정의 (너무 일반적인 단어 제외)
    STOPWORDS = {'카드', '체크', '신용', '신용카드', '체크카드', 'card', 'check'}
    
    # 쿼리 키워드 추출 (정규화 전에 분리) + 불용어 제거
    query_keywords = []
    for word in query.split():
        if len(word) <= 1:
            continue
        normalized_word = normalize_text(word)
        
        # 불용어가 아니면 그대로 추가
        if normalized_word not in STOPWORDS:
            # 복합어에서 불용어 제거 (예: "배움카드" → "배움")
            cleaned_word = normalized_word
            for stopword in STOPWORDS:
                if stopword in cleaned_word:
                    cleaned_word = cleaned_word.replace(stopword, '')
            
            if len(cleaned_word) >= 2:  # 최소 2글자 이상
                query_keywords.append(cleaned_word)
    
    query_normalized = normalize_text(query)
    
    # 형태소 분석 기반 전처리 (사용 가능한 경우)
    morphology_candidates = []
    if use_morphology and MORPHOLOGY_AVAILABLE:
        try:
            # 1. 형태소 분석으로 카드상품명 후보 직접 추출
            morphology_candidates = extract_card_product_candidates(query)
            
            # 사용자사전에 등록된 카드상품명이 추출되면 높은 점수로 즉시 반환
            for candidate in morphology_candidates:
                for product in products:
                    if normalize_text(candidate) == normalize_text(product["keyword"]):
                        return [(product["keyword"], 0.98)]  # 형태소 분석 매칭은 매우 높은 확신도
            
            # 형태소 분석 결과로 직접 매칭 시도 (우선순위 높음)
            if morphology_candidates:
                morphology_matches = []
                for mc in morphology_candidates:
                    mc_normalized = normalize_text(mc)
                    for product in products:
                        keyword_normalized = normalize_text(product["keyword"])
                        
                        # 형태소 후보가 카드상품명에 포함되는지 확인
                        if mc_normalized in keyword_normalized:
                            # 길이 비율에 따라 점수 조정
                            length_ratio = len(mc_normalized) / len(keyword_normalized)
                            score = 0.90 + (length_ratio * 0.08)  # 0.90 ~ 0.98
                            morphology_matches.append((product["keyword"], score))
                        # 카드상품명이 형태소 후보에 포함되는 경우도 체크
                        elif keyword_normalized in mc_normalized:
                            morphology_matches.append((product["keyword"], 0.92))
                
                # 형태소 분석 기반 매칭이 있으면 우선 반환
                if morphology_matches:
                    morphology_matches.sort(key=lambda x: x[1], reverse=True)
                    return morphology_matches[:top_k]
            
            # 형태소 분석 결과를 쿼리 키워드에 추가
            if morphology_candidates:
                for mc in morphology_candidates:
                    normalized_mc = normalize_text(mc)
                    if normalized_mc not in query_keywords and len(normalized_mc) > 1:
                        query_keywords.append(normalized_mc)
                        
        except Exception as e:
            print(f"[VocabularyMatcher] 형태소 분석 실패: {e}")
    
    # 형태소 분석이 없는 경우, 쿼리 자체를 직접 매칭 시도
    if not morphology_candidates:
        query_matches = []
        for product in products:
            keyword_normalized = normalize_text(product["keyword"])
            
            # 쿼리가 카드상품명에 포함되는지 확인
            if query_normalized in keyword_normalized:
                length_ratio = len(query_normalized) / len(keyword_normalized)
                score = 0.88 + (length_ratio * 0.10)  # 0.88 ~ 0.98
                query_matches.append((product["keyword"], score))
            # 카드상품명이 쿼리에 포함되는 경우
            elif keyword_normalized in query_normalized:
                query_matches.append((product["keyword"], 0.95))
        
        # 직접 매칭 결과가 있으면 우선 반환
        if query_matches:
            query_matches.sort(key=lambda x: x[1], reverse=True)
            return query_matches[:top_k]
    
    
    # 각 상품명에 대해 유사도 계산
    for product in products:
        keyword = product["keyword"]
        keyword_normalized = normalize_text(keyword)
        
        # 1. 정확히 일치하는 경우 (최우선)
        if query_normalized == keyword_normalized:
            return [(keyword, 1.0)]
        
        # 2. 부분 문자열 포함 (양방향 체크)
        # 2-1. 카드상품명이 쿼리에 포함 (기존)
        if keyword_normalized in query_normalized:
            candidates.append((keyword, 0.95))
            continue
        
        # 2-2. 쿼리가 카드상품명에 포함 (신규 추가)
        # 예: "배움카드" in "내일배움 테디카드"
        if query_normalized in keyword_normalized:
            # 쿼리 길이 비율에 따라 점수 조정
            length_ratio = len(query_normalized) / len(keyword_normalized)
            score = 0.85 + (length_ratio * 0.1)  # 0.85 ~ 0.95
            candidates.append((keyword, score))
            continue
        
        # 3. 키워드 조합 매칭 (신규 추가)
        # "아이", "플러스" → "아이사랑 플러스 카드"
        if query_keywords:
            # 카드상품명을 단어 단위로 분리
            keyword_words = [normalize_text(word) for word in keyword.split() if len(word) > 1]
            
            # 쿼리 키워드가 카드상품명에 포함되는지 확인
            matched_keywords = []
            for qk in query_keywords:
                # 정확 매칭 또는 부분 포함
                if any(qk in kw or kw in qk for kw in keyword_words):
                    matched_keywords.append(qk)
                # 카드상품명 전체에 포함되는지도 확인
                elif qk in keyword_normalized:
                    matched_keywords.append(qk)
                else:
                    # 쿼리 키워드의 부분 문자열이 카드상품명에 포함되는지 확인
                    # 예: "배움카드" → "배움"이 "내일배움"에 포함
                    best_match_len = 0
                    best_match_str = ""
                    for i in range(len(qk)):
                        for j in range(i+2, len(qk)+1):  # 최소 2글자
                            substring = qk[i:j]
                            # 불용어 제외
                            if substring in STOPWORDS:
                                continue
                            if substring in keyword_normalized and len(substring) > best_match_len:
                                best_match_len = len(substring)
                                best_match_str = substring
                    
                    if best_match_len >= 2:
                        matched_keywords.append(best_match_str)  # 매칭된 부분만 추가
            
            if matched_keywords:
                # 매칭 점수 계산
                match_ratio = len(matched_keywords) / len(query_keywords)
                
                # 최소 2개 이상의 의미있는 키워드가 매칭되면 높은 점수
                if len(matched_keywords) >= 2:
                    combined_score = 0.7 + (match_ratio * 0.2)
                elif len(matched_keywords) == 1:
                    # 1개만 매칭되어도 키워드 길이가 길면 (3글자 이상) 높은 점수
                    matched_kw = matched_keywords[0]
                    if len(matched_kw) >= 3:
                        combined_score = 0.75 + (match_ratio * 0.15)
                    else:
                        combined_score = 0.5 + (match_ratio * 0.2)
                else:
                    combined_score = match_ratio * 0.5
                
                if combined_score >= threshold:
                    candidates.append((keyword, combined_score))
                    continue
        
        # 4. 발음 유사도 계산 (기존 로직)
        similarity = phonetic_similarity(query, keyword)
        
        if similarity >= threshold:
            candidates.append((keyword, similarity))
        
        # 5. 동의어/변형 체크
        for syn in product.get("synonyms", []):
            syn_similarity = phonetic_similarity(query, syn)
            if syn_similarity >= threshold:
                candidates.append((keyword, syn_similarity * 0.9))  # 동의어는 약간 낮은 점수
        
        for var in product.get("variations", []):
            var_similarity = phonetic_similarity(query, var)
            if var_similarity >= threshold:
                candidates.append((keyword, var_similarity * 0.9))
    
    # 유사도 내림차순 정렬 및 Top-K 반환
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def get_best_match(query: str, confidence_threshold: float = 0.85) -> Optional[str]:
    """
    가장 확신도 높은 매칭 결과 반환
    
    Args:
        query: 입력 텍스트
        confidence_threshold: 확신도 임계값 (이상이면 즉시 반환)
    
    Returns:
        매칭된 상품명 또는 None
    """
    candidates = find_candidates(query, top_k=1, threshold=0.6)
    
    if not candidates:
        return None
    
    best_match, score = candidates[0]
    
    # 확신도가 높으면 즉시 반환 (sLLM 호출 불필요)
    if score >= confidence_threshold:
        return best_match
    
    return None  # 확신도 낮으면 None 반환 (sLLM 검증 필요)
