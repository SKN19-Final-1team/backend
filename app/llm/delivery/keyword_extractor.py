"""
금융 키워드 추출 모듈

STT 텍스트에서 금융 관련 키워드를 추출하고 분류합니다.
- 카드상품명: 형태소 분석 + 발음 유사도 매칭
- 액션: 분실, 재발급, 신청 등
- 결제수단: 삼성페이, 애플페이 등
- 의도: 혜택, 연회비, 할인 등

RAG 파이프라인과 호환되는 형태로 키워드를 반환합니다.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

# 형태소 분석기
try:
    from app.llm.delivery.morphology_analyzer import (
        analyze_morphemes,
        extract_nouns,
        extract_card_product_candidates,
    )
    MORPHOLOGY_AVAILABLE = True
except ImportError:
    MORPHOLOGY_AVAILABLE = False
    print("[KeywordExtractor] 형태소 분석기 없음")

# 어휘 매칭기
try:
    from app.llm.delivery.vocabulary_matcher import (
        find_candidates,
        get_best_match,
    )
    VOCABULARY_MATCHER_AVAILABLE = True
except ImportError:
    VOCABULARY_MATCHER_AVAILABLE = False
    print("[KeywordExtractor] 어휘 매칭기 없음")

# 키워드 사전 (RAG)
try:
    from app.rag.vocab.keyword_dict import (
        ACTION_SYNONYMS,
        PAYMENT_SYNONYMS,
        WEAK_INTENT_SYNONYMS,
    )
    KEYWORD_DICT_AVAILABLE = True
except ImportError:
    KEYWORD_DICT_AVAILABLE = False
    ACTION_SYNONYMS = {}
    PAYMENT_SYNONYMS = {}
    WEAK_INTENT_SYNONYMS = {}
    print("[KeywordExtractor] 키워드 사전 없음")


# STT 오류 교정 사전 경로
CORRECTION_DICT_PATH = Path(__file__).parent.parent.parent / "rag" / "vocab" / "keywords_dict_refine.json"


# 액션 키워드
STRONG_ACTION_TOKENS = {
    "방법", "어떻게", "신청", "등록", "추가", "설정", "변경",
    "해지", "취소", "안됨", "오류", "실패", "인증", "재발급",
    "재발행", "재교부", "분실",
}

USAGE_STRONG_TERMS = {
    "분실", "도난", "재발급", "발급", "신청", "해지", "결제",
    "승인", "취소", "환불", "오류", "에러", "안돼", "불가",
    "거절", "신고", "등록", "인증", "서류", "사용", "사용처",
    "교통", "충전", "연체", "수수료", "이자", "이자율", "대출",
    "현금서비스", "카드론", "리볼빙", "입금", "납부", "한도",
}

# 정보성 의도 키워드
INFO_HINT_TERMS = {
    "혜택", "연회비", "조건", "추천", "비교", "좋아", "좋은",
    "괜찮", "정보", "소개", "어떤", "뭐가",
}

# 불용어
STOPWORDS = {
    "체크", "신용", "card", "check",  # "카드" 제거됨
    "신용카드", "체크카드", # 이들은 너무 일반적이라 여전히 불용어 처리 가능, 혹은 조건부로 이동 고민. 일단 유지.
    "있어요", "있나요", "있나", "뭐에요", "뭐예요", "뭐야",
    "어떻게", "어떤", "그", "이", "저", "그런", "이런",
}

# 조건부 불용어 (단독으로 쓰이면 불용어, 수식어가 있으면 유의미)
CONDITIONAL_STOPWORDS = {
    "카드", "문의", "번호", "질문", "상담",
}


@dataclass
class ExtractedKeywords:
    """추출된 키워드 결과"""
    card_names: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    payments: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)
    nouns: List[str] = field(default_factory=list)
    original_text: str = ""
    corrected_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    def to_rag_signals(self) -> Dict[str, List[str]]:
        """RAG Signals 호환 형식으로 변환"""
        return {
            "card_names": self.card_names,
            "actions": self.actions,
            "payments": self.payments,
            "weak_intents": self.intents,
        }

    def to_query(self) -> str:
        """RAG 검색용 쿼리 문자열 생성"""
        parts = []
        parts.extend(self.card_names)
        parts.extend(self.actions)
        parts.extend(self.intents)
        # 중복 제거하면서 순서 유지
        seen = set()
        unique_parts = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                unique_parts.append(p)
        return " ".join(unique_parts)

    def is_empty(self) -> bool:
        """키워드가 비어있는지 확인"""
        return not (self.card_names or self.actions or self.payments or self.intents)


class KeywordExtractor:
    """금융 키워드 추출기"""

    def __init__(self):
        self._correction_map: Dict[str, str] = {}
        self._action_keywords: Set[str] = set()
        self._payment_keywords: Dict[str, str] = {}  # 변형 -> 정규형
        self._intent_keywords: Dict[str, str] = {}   # 변형 -> 정규형
        self._initialized = False

    def _ensure_initialized(self):
        """지연 초기화"""
        if self._initialized:
            return

        # STT 오류 교정 사전 로드
        self._load_correction_map()

        # 액션 키워드 구축
        self._build_action_keywords()

        # 결제수단 키워드 구축
        self._build_payment_keywords()

        # 의도 키워드 구축
        self._build_intent_keywords()

        self._initialized = True
        print(f"[KeywordExtractor] 초기화 완료: 교정맵={len(self._correction_map)}, "
              f"액션={len(self._action_keywords)}, 결제수단={len(self._payment_keywords)}, "
              f"의도={len(self._intent_keywords)}")

    def _load_correction_map(self):
        """STT 오류 교정 사전 로드"""
        try:
            if CORRECTION_DICT_PATH.exists():
                data = json.loads(CORRECTION_DICT_PATH.read_text(encoding="utf-8"))
                self._correction_map = data.get("correction_map", {})
        except Exception as e:
            print(f"[KeywordExtractor] 교정 사전 로드 실패: {e}")
            self._correction_map = {}

    def _build_action_keywords(self):
        """액션 키워드 세트 구축"""
        self._action_keywords = STRONG_ACTION_TOKENS | USAGE_STRONG_TERMS

        # ACTION_SYNONYMS에서 추가
        if KEYWORD_DICT_AVAILABLE:
            for canonical, synonyms in ACTION_SYNONYMS.items():
                self._action_keywords.add(canonical)
                for syn in synonyms:
                    self._action_keywords.add(syn)

    def _build_payment_keywords(self):
        """결제수단 키워드 맵 구축 (변형 -> 정규형)"""
        for canonical, synonyms in PAYMENT_SYNONYMS.items():
            self._payment_keywords[canonical.lower()] = canonical
            self._payment_keywords[canonical.replace(" ", "").lower()] = canonical
            for syn in synonyms:
                self._payment_keywords[syn.lower()] = canonical
                self._payment_keywords[syn.replace(" ", "").lower()] = canonical

    def _build_intent_keywords(self):
        """의도 키워드 맵 구축 (변형 -> 정규형)"""
        # WEAK_INTENT_SYNONYMS에서 추가
        if KEYWORD_DICT_AVAILABLE:
            for canonical, synonyms in WEAK_INTENT_SYNONYMS.items():
                self._intent_keywords[canonical] = canonical
                for syn in synonyms:
                    self._intent_keywords[syn] = canonical

        # INFO_HINT_TERMS 추가
        for term in INFO_HINT_TERMS:
            if term not in self._intent_keywords:
                self._intent_keywords[term] = term

    def extract(self, text: str) -> ExtractedKeywords:
        """
        텍스트에서 금융 키워드 추출

        Args:
            text: 입력 텍스트 (STT 결과)

        Returns:
            ExtractedKeywords: 추출된 키워드
        """
        self._ensure_initialized()

        if not text or not text.strip():
            return ExtractedKeywords(original_text=text, corrected_text=text)

        # 1. STT 오류 교정
        corrected = self._correct_stt_errors(text)

        # 2. 카드상품명 추출
        card_names = self._extract_card_names(corrected)

        # 3. 액션 추출
        actions = self._extract_actions(corrected)

        # 4. 결제수단 추출
        payments = self._extract_payments(corrected)

        # 5. 의도 추출
        intents = self._extract_intents(corrected)

        # 6. 일반 명사 추출 (불용어 및 이미 추출된 것 제외)
        nouns = self._extract_nouns(corrected, card_names, actions, payments, intents)

        return ExtractedKeywords(
            card_names=card_names,
            actions=actions,
            payments=payments,
            intents=intents,
            nouns=nouns,
            original_text=text,
            corrected_text=corrected,
        )

    def _correct_stt_errors(self, text: str) -> str:
        """STT 오류 교정"""
        if not self._correction_map:
            return text

        corrected = text
        # 긴 키워드부터 먼저 교정 (부분 매칭 방지)
        sorted_keys = sorted(self._correction_map.keys(), key=len, reverse=True)

        for wrong, correct in ((k, self._correction_map[k]) for k in sorted_keys):
            if wrong in corrected:
                corrected = corrected.replace(wrong, correct)

        return corrected

    def _extract_card_names(self, text: str) -> List[str]:
        """카드상품명 추출"""
        card_names = []

        # 1. 형태소 분석으로 카드상품명 후보 추출
        if MORPHOLOGY_AVAILABLE:
            try:
                candidates = extract_card_product_candidates(text)
                for candidate in candidates:
                    if candidate and candidate not in card_names:
                        card_names.append(candidate)
            except Exception as e:
                print(f"[KeywordExtractor] 형태소 분석 실패: {e}")

        # 2. 어휘 매칭으로 카드상품명 검증/추가
        if VOCABULARY_MATCHER_AVAILABLE:
            try:
                # 전체 텍스트에서 매칭 시도
                matches = find_candidates(text, top_k=3, threshold=0.75)
                for match, score in matches:
                    if match and match not in card_names and score >= 0.75:
                        card_names.append(match)

                # 형태소 분석 결과 각각에 대해서도 매칭 시도
                if MORPHOLOGY_AVAILABLE:
                    nouns = extract_nouns(text)
                    for noun in nouns:
                        if len(noun) >= 2:
                            best = get_best_match(noun, confidence_threshold=0.80)
                            if best and best not in card_names:
                                card_names.append(best)
            except Exception as e:
                print(f"[KeywordExtractor] 어휘 매칭 실패: {e}")

        return card_names

    def _extract_actions(self, text: str) -> List[str]:
        """액션 키워드 추출"""
        actions = []

        # 형태소 분석으로 명사/동사 추출 후 액션 매칭
        tokens = set()
        if MORPHOLOGY_AVAILABLE:
            try:
                morphemes = analyze_morphemes(text)
                for morpheme, pos in morphemes:
                    # 명사(NNG, NNP, NNB) 및 동사(VV, VA)
                    if pos.startswith(('NN', 'VV', 'VA')):
                        tokens.add(morpheme)
            except Exception:
                pass

        # 공백 기준 토큰도 추가
        tokens.update(text.split())

        # 액션 키워드 매칭
        for token in tokens:
            token_clean = token.strip()
            if token_clean in self._action_keywords:
                if token_clean not in actions and token_clean not in STOPWORDS:
                    actions.append(token_clean)

        # 정규식 패턴 매칭 (복합 표현)
        action_patterns = [
            (r'(결제|승인).*(안|오류|실패|불가)', '결제오류'),
            (r'(카드|분실).*(신고|접수)', '분실신고'),
            (r'(한도).*(상향|올|높)', '한도상향'),
            (r'(한도).*(하향|낮|줄)', '한도하향'),
        ]

        for pattern, action in action_patterns:
            if re.search(pattern, text):
                if action not in actions:
                    actions.append(action)

        return actions

    def _extract_payments(self, text: str) -> List[str]:
        """결제수단 추출"""
        payments = []
        text_normalized = text.replace(" ", "").lower()

        # 결제수단 키워드 매칭
        for variant, canonical in self._payment_keywords.items():
            if variant in text_normalized:
                if canonical not in payments:
                    payments.append(canonical)

        return payments

    def _extract_intents(self, text: str) -> List[str]:
        """의도 키워드 추출"""
        intents = []

        # 형태소 분석으로 명사 추출
        tokens = set()
        if MORPHOLOGY_AVAILABLE:
            try:
                nouns = extract_nouns(text)
                tokens.update(nouns)
            except Exception:
                pass

        # 공백 기준 토큰도 추가
        tokens.update(text.split())

        # 의도 키워드 매칭
        for token in tokens:
            token_clean = token.strip()
            if token_clean in self._intent_keywords:
                canonical = self._intent_keywords[token_clean]
                if canonical not in intents and canonical not in STOPWORDS:
                    intents.append(canonical)

        return intents

    def _extract_nouns(
        self,
        text: str,
        card_names: List[str],
        actions: List[str],
        payments: List[str],
        intents: List[str],
    ) -> List[str]:
        """일반 명사 추출 (이미 추출된 키워드 제외) 및 조건부 불용어 처리"""
        nouns = []

        # 이미 추출된 키워드 세트
        extracted = set()
        for items in [card_names, actions, payments, intents]:
            for item in items:
                extracted.add(item)
                extracted.add(item.replace(" ", ""))

        if MORPHOLOGY_AVAILABLE:
            try:
                # 형태소 분석 결과를 직접 사용하여 문맥 파악
                morphemes = analyze_morphemes(text)
                
                for i, (morph, pos) in enumerate(morphemes):
                    # 명사(NNG, NNP, NNB)만 대상
                    if pos not in ('NNG', 'NNP', 'NNB'):
                        continue
                    
                    # 1. 기본 불용어 체크
                    if morph in STOPWORDS:
                        continue
                        
                    # 2. 이미 추출된 키워드 제외
                    if morph in extracted:
                        continue
                        
                    # 3. 2글자 미만 제외 (단, 특수 케이스 제외가능하지만 일단 유지)
                    if len(morph) < 2:
                        continue

                    # 4. 조건부 불용어(Context-aware Stopwords) 체크
                    if morph in CONDITIONAL_STOPWORDS:
                        if not self._has_meaningful_context(morphemes, i):
                            continue

                    nouns.append(morph)
                    
            except Exception as e:
                print(f"[KeywordExtractor] 명사 추출 실패: {e}")

        return nouns

    def _has_meaningful_context(self, morphemes: List[tuple], index: int) -> bool:
        """
        조건부 불용어가 유의미한 문맥(복합명사 등)에서 사용되었는지 확인
        
        Args:
            morphemes: 형태소 분석 결과 [(morph, pos), ...]
            index: 현재 불용어의 인덱스
            
        Returns:
            bool: 유지 여부 (True면 유지, False면 제거)
        """
        if index <= 0:
            return False
            
        prev_morph, prev_pos = morphemes[index - 1]
        
        # 바로 앞이 명사(NNG, NNP)이고, 불용어가 아닐 때만 유의미한 수식/복합 관계로 인정
        # 예: "청년(NNG) + 카드", "희망(NNG) + 카드" -> KEEP
        #     "만든(VV+ETM) + 카드" -> DROP (현재 로직 유지)
        if (prev_pos in ('NNG', 'NNP') and 
            prev_morph not in STOPWORDS and 
            prev_morph not in CONDITIONAL_STOPWORDS):
            return True
            
        return False

    def warmup(self):
        """
        초기화 및 형태소 분석기 사전 로드 (Warm-up)
        애플리케이션 시작 시점에 호출하여 첫 요청 지연을 방지합니다.
        """
        print("[KeywordExtractor] Warmup 시작...")
        self._ensure_initialized()
        
        if MORPHOLOGY_AVAILABLE:
            try:
                # 더미 텍스트로 형태소 분석기 로드 트리거
                analyze_morphemes("초기화")
                print("[KeywordExtractor] 형태소 분석기 로드 완료")
            except Exception as e:
                print(f"[KeywordExtractor] 형태소 분석기 워밍업 실패: {e}")
        
        print("[KeywordExtractor] Warmup 완료")


# 싱글톤 인스턴스
_extractor_instance: Optional[KeywordExtractor] = None


def get_extractor() -> KeywordExtractor:
    """KeywordExtractor 싱글톤 인스턴스 반환"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = KeywordExtractor()
    return _extractor_instance


def extract_keywords(text: str) -> ExtractedKeywords:
    """
    텍스트에서 금융 키워드 추출 (편의 함수)

    Args:
        text: 입력 텍스트 (STT 결과)

    Returns:
        ExtractedKeywords: 추출된 키워드

    Example:
        >>> result = extract_keywords("테디카드 분실 신고하려고요")
        >>> print(result.card_names)  # ['테디 베이직 카드']
        >>> print(result.actions)     # ['분실', '신고']
    """
    return get_extractor().extract(text)


def warmup():
    """모듈 웜업 (애플리케이션 시작 시 호출)"""
    get_extractor().warmup()


@lru_cache(maxsize=256)
def extract_keywords_cached(text: str) -> ExtractedKeywords:
    """
    캐싱된 키워드 추출 (동일 텍스트 재처리 방지)

    Note:
        dataclass는 hashable하지 않으므로 결과를 캐싱하려면
        이 함수를 사용하세요.
    """
    return extract_keywords(text)


def to_rag_query(keywords: ExtractedKeywords) -> str:
    """RAG 검색용 쿼리 문자열 생성"""
    return keywords.to_query()


def to_rag_signals(keywords: ExtractedKeywords) -> Dict[str, List[str]]:
    """RAG Signals 호환 형식으로 변환"""
    return keywords.to_rag_signals()

