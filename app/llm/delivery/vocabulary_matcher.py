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


def normalize_card_name(full_name: str) -> str:
    """
    카드 전체 상품명에서 핵심 키워드 추출

    Args:
        full_name: 전체 상품명 (예: "AK PLAZA 테디카드 Plus")

    Returns:
        정규화된 카드명 (예: "테디카드")

    규칙:
        1. 특정 카드명 패턴 우선 추출 (테디카드, 나라사랑카드 등)
        2. 부가 설명 제거 후 "카드" 추가
        3. 브랜드명, 부가 설명 제거
        4. 띄어쓰기 제거
    """
    # 정규화: 소문자, 띄어쓰기 제거
    normalized = re.sub(r'\s+', '', full_name.lower())

    # 특정 카드명 패턴 (우선순위 높음)
    CARD_NAME_PATTERNS = [
        r'(나라사랑카드)',
        r'(나라사랑)',  # "나라사랑 카드" → "나라사랑카드"
        r'(테디카드)',
        r'(그린카드)',
        r'(알뜰교통카드)',
        r'(국민행복카드)',
        r'(국민행복)',  # "국민행복(신용_체크)" → "국민행복카드"
    ]

    for pattern in CARD_NAME_PATTERNS:
        match = re.search(pattern, normalized)
        if match:
            matched_name = match.group(1)
            # "카드"가 없으면 추가
            if not matched_name.endswith('카드'):
                matched_name += '카드'
            return matched_name

    # 일반 패턴: "~카드" 형태 추출
    # 예: "akplaza테디카드plus" → "테디카드"
    card_pattern = r'([가-힣a-z0-9]+카드)'
    matches = re.findall(card_pattern, normalized)

    if matches:
        # 가장 긴 매칭 반환 (더 구체적인 카드명)
        longest_match = max(matches, key=len)

        # 브랜드명 제거 (선택적)
        # 예: "신한테디카드" → "테디카드"
        brand_prefixes = ['신한', '하나', '우리', '국민', '삼성', 'bc', 'kb', 'nh']
        for prefix in brand_prefixes:
            if longest_match.startswith(prefix):
                potential_name = longest_match[len(prefix):]
                if len(potential_name) >= 3:  # 최소 3글자 이상
                    return potential_name

        return longest_match

    # 매칭 실패 시 원본 반환
    return normalize_text(full_name)


def load_card_products(force_reload: bool = False) -> List[Dict]:
    """
    DB에서 카드상품명 로드 및 캐싱 (card_products 테이블 기반)

    Args:
        force_reload: 캐시 무시하고 재로드

    Returns:
        카드상품명 리스트 [{"id": str, "name": str, "normalized_name": str}, ...]
    """
    global _CARD_PRODUCTS_CACHE

    if _CARD_PRODUCTS_CACHE is not None and not force_reload:
        return _CARD_PRODUCTS_CACHE

    conn = connect_db()
    cursor = conn.cursor()

    try:
        # card_products 테이블에서 카드상품명 조회
        query = """
            SELECT id, name, card_type, brand
            FROM card_products
            ORDER BY id
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        products = []
        seen_normalized = set()  # 중복 제거용

        for row in rows:
            card_id = row[0]
            card_name = row[1]
            card_type = row[2]
            brand = row[3]

            # 카드명 정규화
            normalized_name = normalize_card_name(card_name)

            # 중복 제거: 동일한 정규화 이름은 1번만 추가
            if normalized_name not in seen_normalized:
                products.append({
                    "id": card_id,
                    "name": card_name,  # 전체 상품명
                    "normalized_name": normalized_name,  # 정규화된 카드명
                    "card_type": card_type,
                    "brand": brand
                })
                seen_normalized.add(normalized_name)

        _CARD_PRODUCTS_CACHE = products
        print(f"[VocabularyMatcher] 카드상품명 {len(products)}개 로드 완료 (정규화: {len(seen_normalized)}개)")

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
    STOPWORDS = {'카드', '체크', '신용', '신용카드', '체크카드', 'card', 'check', '있잖아요', '그거', '뭐시기', '기능'}

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

    # 짧은 쿼리 (2~4글자) 특별 처리: 부분 매칭 강화
    SHORT_KEYWORD_THRESHOLD = 4
    query_normalized_clean = re.sub(r'[^가-힣a-z0-9]', '', query_normalized)
    is_short_query = len(query_normalized_clean) <= SHORT_KEYWORD_THRESHOLD
    
    # 형태소 분석 기반 전처리 (사용 가능한 경우)
    morphology_candidates = []
    if use_morphology and MORPHOLOGY_AVAILABLE:
        try:
            # 1. 형태소 분석으로 카드상품명 후보 직접 추출
            morphology_candidates = extract_card_product_candidates(query)
            
            # 사용자사전에 등록된 카드상품명이 추출되면 높은 점수로 즉시 반환
            for candidate in morphology_candidates:
                for product in products:
                    normalized_name = product["normalized_name"]
                    if normalize_text(candidate) == normalize_text(normalized_name):
                        return [(normalized_name, 0.98)]  # 형태소 분석 매칭은 매우 높은 확신도

            # 형태소 분석 결과로 직접 매칭 시도 (우선순위 높음)
            if morphology_candidates:
                morphology_matches = []
                for mc in morphology_candidates:
                    mc_normalized = normalize_text(mc)
                    for product in products:
                        normalized_name = product["normalized_name"]
                        normalized_keyword = normalize_text(normalized_name)

                        # 형태소 후보가 정규화된 카드명에 포함되는지 확인
                        if mc_normalized in normalized_keyword:
                            # 길이 비율에 따라 점수 조정
                            length_ratio = len(mc_normalized) / len(normalized_keyword)
                            score = 0.90 + (length_ratio * 0.08)  # 0.90 ~ 0.98
                            morphology_matches.append((normalized_name, score))
                        # 정규화된 카드명이 형태소 후보에 포함되는 경우도 체크
                        elif normalized_keyword in mc_normalized:
                            morphology_matches.append((normalized_name, 0.92))

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
            normalized_name = product["normalized_name"]
            normalized_keyword = normalize_text(normalized_name)

            # 쿼리가 정규화된 카드명에 포함되는지 확인
            if query_normalized in normalized_keyword:
                length_ratio = len(query_normalized) / len(normalized_keyword)
                score = 0.88 + (length_ratio * 0.10)  # 0.88 ~ 0.98
                query_matches.append((normalized_name, score))
            # 정규화된 카드명이 쿼리에 포함되는 경우
            elif normalized_keyword in query_normalized:
                query_matches.append((normalized_name, 0.95))

        # 직접 매칭 결과가 있으면 우선 반환
        if query_matches:
            query_matches.sort(key=lambda x: x[1], reverse=True)
            return query_matches[:top_k]
    
    
    # 짧은 쿼리를 위한 특별 처리
    short_query_candidates = []

    # 각 상품명에 대해 유사도 계산
    for product in products:
        # Phase 3: 정규화된 카드명 사용
        full_name = product["name"]  # 전체 상품명
        normalized_name = product["normalized_name"]  # 정규화된 핵심 키워드

        # 매칭에는 전체 상품명과 정규화된 이름 둘 다 사용
        keyword = full_name
        keyword_normalized = normalize_text(full_name)
        normalized_keyword = normalize_text(normalized_name)

        # 1. 정규화된 카드명이 정확히 일치하는 경우 (최우선)
        if query_normalized == normalized_keyword:
            return [(normalized_name, 1.0)]

        # 2. 전체 상품명이 정확히 일치하는 경우
        if query_normalized == keyword_normalized:
            return [(normalized_name, 1.0)]

        # Phase 3: 짧은 쿼리 부분 매칭 (2~4글자)
        # 예: "테디" → "테디카드", "나라" → "나라사랑카드"
        if is_short_query and query_normalized_clean in normalized_keyword:
            # 쿼리 길이 비율에 따라 점수 조정
            length_ratio = len(query_normalized_clean) / len(normalized_keyword)
            score = 0.88 + (length_ratio * 0.10)  # 0.88 ~ 0.98
            short_query_candidates.append((normalized_name, score))
        
        # 3. 부분 문자열 포함 (정규화된 카드명과 비교)
        # 3-1. 정규화된 카드명이 쿼리에 포함
        if normalized_keyword in query_normalized:
            candidates.append((normalized_name, 0.95))
            continue

        # 3-2. 쿼리가 정규화된 카드명에 포함
        if query_normalized in normalized_keyword:
            length_ratio = len(query_normalized) / len(normalized_keyword)
            score = 0.85 + (length_ratio * 0.1)  # 0.85 ~ 0.95
            candidates.append((normalized_name, score))
            continue

        # 3-3. 전체 상품명으로도 매칭 시도 (보조)
        if keyword_normalized in query_normalized:
            candidates.append((normalized_name, 0.90))
            continue

        if query_normalized in keyword_normalized:
            length_ratio = len(query_normalized) / len(keyword_normalized)
            score = 0.80 + (length_ratio * 0.1)  # 0.80 ~ 0.90
            candidates.append((normalized_name, score))
            continue
        
        # 4. 키워드 조합 매칭
        # "테디", "카드" → "테디카드"
        if query_keywords:
            # 정규화된 카드명을 단어 단위로 분리
            normalized_words = [normalize_text(word) for word in normalized_name.split() if len(word) > 1]

            # 쿼리 키워드가 정규화된 카드명에 포함되는지 확인
            matched_keywords = []
            for qk in query_keywords:
                # 정확 매칭 또는 부분 포함
                if any(qk in nw or nw in qk for nw in normalized_words):
                    matched_keywords.append(qk)
                # 정규화된 카드명 전체에 포함되는지도 확인
                elif qk in normalized_keyword:
                    matched_keywords.append(qk)

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
                    candidates.append((normalized_name, combined_score))
                    continue

        # 5. 발음 유사도 계산 (정규화된 카드명과 비교)
        similarity = phonetic_similarity(query, normalized_name)

        if similarity >= threshold:
            candidates.append((normalized_name, similarity))

    # Phase 3: 짧은 쿼리 후보 우선 처리
    if short_query_candidates:
        short_query_candidates.sort(key=lambda x: x[1], reverse=True)
        # 짧은 쿼리 후보가 충분히 좋은 점수면 우선 반환
        if short_query_candidates[0][1] >= 0.88:
            return short_query_candidates[:top_k]
        # 그렇지 않으면 다른 후보와 합쳐서 정렬
        candidates.extend(short_query_candidates)

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
