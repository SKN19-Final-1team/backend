"""
임베딩 기반 금융 용어 정제 모듈

입력 텍스트의 오타/변형된 금융 용어를 임베딩 유사도 매칭으로 정제합니다.
하이브리드 방식: 편집 거리 + 임베딩 유사도
로컬 모델 사용 (빠른 처리)
"""

import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance

from dotenv import load_dotenv
load_dotenv()

# 단어사전 import
from app.rag.vocab.keyword_dict import (
    get_action_synonyms,
    get_card_name_synonyms,
    get_weak_intent_synonyms,
    PAYMENT_SYNONYMS,
)



class VocabEmbedder:
    """
    단어사전의 모든 금융 용어를 로컬 모델로 임베딩하고 캐싱합니다.
    """
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        Args:
            model_name: 사용할 임베딩 모델 (한국어 특화 로컬 모델)
                - jhgan/ko-sroberta-multitask (기본, 빠름)
                - BM-K/KoSimCSE-roberta-multitask (고성능)
                - jhgan/ko-sbert-nli (자연어 추론)
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.vocab_embeddings: Dict[str, np.ndarray] = {}
        self.vocab_terms: List[str] = []
        
        # 캐시 경로 설정
        cache_dir = Path(__file__).parent / "embeddings" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        # 모델명에서 특수문자 제거
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        self.cache_path = cache_dir / f"vocab_embeddings_{safe_model_name}.pkl"
        
        # 초기화
        self._initialize()
    
    def _initialize(self):
        """모델 로드 및 임베딩 로드/생성"""
        # 캐시가 있으면 로드, 없으면 생성
        if self.cache_path.exists():
            print(f"[VocabEmbedder] Loading cached embeddings from {self.cache_path}")
            self._load_cached_embeddings()
        else:
            print(f"[VocabEmbedder] Building new embeddings...")
            self._load_model()
            self._build_embeddings()
            self._save_embeddings()
    
    def _load_model(self):
        """로컬 임베딩 모델 로드"""
        if self.model is None:
            print(f"[VocabEmbedder] Loading local model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
    
    def _collect_vocab_terms(self) -> List[str]:
        """모든 금융 용어 수집"""
        terms = set()
        
        # 카드명
        card_synonyms = get_card_name_synonyms()
        for canonical, variants in card_synonyms.items():
            terms.add(canonical)
            terms.update(v for v in variants if v)
        
        # 결제수단
        for canonical, variants in PAYMENT_SYNONYMS.items():
            terms.add(canonical)
            terms.update(v for v in variants if v)
        
        # 행동 의도
        action_synonyms = get_action_synonyms()
        for canonical, variants in action_synonyms.items():
            terms.add(canonical)
            terms.update(v for v in variants if v)
        
        # 약한 의도
        weak_synonyms = get_weak_intent_synonyms()
        for canonical, variants in weak_synonyms.items():
            terms.add(canonical)
            terms.update(v for v in variants if v)
        
        return sorted(list(terms))
    
    def _build_embeddings(self):
        """모든 용어를 임베딩"""
        self.vocab_terms = self._collect_vocab_terms()
        print(f"[VocabEmbedder] Embedding {len(self.vocab_terms)} terms...")
        
        # 배치로 임베딩 (빠른 처리)
        embeddings = self.model.encode(
            self.vocab_terms,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # 딕셔너리로 저장
        self.vocab_embeddings = {
            term: emb for term, emb in zip(self.vocab_terms, embeddings)
        }
        
        print(f"[VocabEmbedder] Embeddings built: {len(self.vocab_embeddings)} terms")
    
    def _save_embeddings(self):
        """임베딩을 파일로 저장"""
        data = {
            "model_name": self.model_name,
            "vocab_terms": self.vocab_terms,
            "embeddings": self.vocab_embeddings
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[VocabEmbedder] Embeddings saved to {self.cache_path}")
    
    def _load_cached_embeddings(self):
        """캐시된 임베딩 로드"""
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)
        
        self.model_name = data["model_name"]
        self.vocab_terms = data["vocab_terms"]
        self.vocab_embeddings = data["embeddings"]
        
        print(f"[VocabEmbedder] Loaded {len(self.vocab_embeddings)} cached embeddings")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """텍스트를 임베딩 (로컬 모델 사용)"""
        if self.model is None:
            self._load_model()
        
        # 로컬 모델로 빠르게 임베딩
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding


class TextMatcher:
    """
    하이브리드 텍스트 매칭: 편집 거리 + 임베딩 유사도
    개선된 스코어링으로 정확도 향상
    """
    
    def __init__(
        self,
        vocab_embedder: VocabEmbedder,
        threshold: float = 0.65,  # 낮춤 (0.75 → 0.65)
        edit_weight: float = 0.4,  # 편집 거리 가중치
        emb_weight: float = 0.6    # 임베딩 가중치
    ):
        """
        Args:
            vocab_embedder: 단어사전 임베더
            threshold: 하이브리드 스코어 임계값 (0.0 ~ 1.0)
            edit_weight: 편집 거리 스코어 가중치
            emb_weight: 임베딩 유사도 가중치
        """
        self.vocab_embedder = vocab_embedder
        self.threshold = threshold
        self.edit_weight = edit_weight
        self.emb_weight = emb_weight
    
    def _dynamic_edit_distance_threshold(self, text: str) -> int:
        """
        단어 길이에 비례한 동적 편집 거리 임계값
        
        Args:
            text: 입력 텍스트
            
        Returns:
            편집 거리 임계값
        """
        # 최소 3, 최대 단어 길이의 1/3
        return max(3, len(text) // 3)
    
    def _hybrid_score(
        self,
        text: str,
        candidate: str,
        text_emb: Optional[np.ndarray] = None
    ) -> float:
        """
        편집 거리 + 임베딩 하이브리드 스코어
        
        Args:
            text: 입력 텍스트
            candidate: 후보 단어
            text_emb: 입력 텍스트 임베딩 (캐싱용)
            
        Returns:
            하이브리드 스코어 (0.0 ~ 1.0)
        """
        # 1. 편집 거리 스코어 (0~1, 높을수록 유사)
        edit_dist = levenshtein_distance(text, candidate)
        max_len = max(len(text), len(candidate))
        edit_score = 1 - (edit_dist / max_len) if max_len > 0 else 0
        
        # 2. 임베딩 유사도 (0~1)
        if text_emb is None:
            text_emb = self.vocab_embedder.get_embedding(text)
        
        text_emb_reshaped = text_emb.reshape(1, -1)
        cand_emb = self.vocab_embedder.vocab_embeddings[candidate].reshape(1, -1)
        emb_score = cosine_similarity(text_emb_reshaped, cand_emb)[0][0]
        
        # 3. 가중 평균
        final_score = self.edit_weight * edit_score + self.emb_weight * emb_score
        
        return float(final_score)
    
    def _filter_by_edit_distance(
        self,
        text: str,
        max_distance: int = None
    ) -> List[str]:
        """
        편집 거리로 후보 필터링
        
        Args:
            text: 검색할 텍스트
            max_distance: 최대 편집 거리 (None이면 동적 계산)
            
        Returns:
            편집 거리 임계값 이하인 용어 리스트
        """
        if max_distance is None:
            max_distance = self._dynamic_edit_distance_threshold(text)
        
        candidates = []
        for term in self.vocab_embedder.vocab_terms:
            dist = levenshtein_distance(text, term)
            if dist <= max_distance:
                candidates.append((term, dist))
        
        # 편집 거리 가까운 순으로 정렬
        candidates.sort(key=lambda x: x[1])
        
        return [term for term, _ in candidates]
    
    def find_similar_term(
        self,
        text: str,
        top_k: int = 1,
        use_hybrid: bool = True
    ) -> List[Tuple[str, float]]:
        """
        가장 유사한 금융 용어 찾기 (하이브리드 방식)
        
        Args:
            text: 검색할 텍스트
            top_k: 반환할 상위 결과 개수
            use_hybrid: 하이브리드 방식 사용 여부
            
        Returns:
            [(용어, 유사도), ...] 리스트
        """
        # 입력 텍스트 임베딩
        text_emb = self.vocab_embedder.get_embedding(text)
        
        # 하이브리드 방식: 편집 거리로 후보 필터링 후 하이브리드 스코어 계산
        if use_hybrid:
            # 1단계: 편집 거리로 후보 필터링 (동적 임계값 사용)
            candidates = self._filter_by_edit_distance(text)
            
            # 후보가 있으면 하이브리드 스코어 계산
            if candidates:
                scored_candidates = []
                for term in candidates:
                    score = self._hybrid_score(text, term, text_emb)
                    scored_candidates.append((term, score))
                
                # 스코어 높은 순으로 정렬
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # 임계값 이상만 반환
                results = [(term, score) for term, score in scored_candidates[:top_k] 
                           if score >= self.threshold]
                
                if results:
                    return results
        
        # 2단계: 후보가 없거나 하이브리드 미사용 시 전체 임베딩 검색 (Fallback)
        text_emb_reshaped = text_emb.reshape(1, -1)
        similarities = []
        for term, vocab_emb in self.vocab_embedder.vocab_embeddings.items():
            vocab_emb_reshaped = vocab_emb.reshape(1, -1)
            sim = cosine_similarity(text_emb_reshaped, vocab_emb_reshaped)[0][0]
            similarities.append((term, float(sim)))
        
        # 유사도 높은 순으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 임계값 이상만 반환
        results = [(term, sim) for term, sim in similarities[:top_k] if sim >= self.threshold]
        
        return results

    
    def refine_token(self, token: str) -> str:
        """
        단일 토큰을 정제
        
        Args:
            token: 정제할 토큰
            
        Returns:
            정제된 토큰 (매칭 실패 시 원본 반환)
        """
        # 너무 짧은 토큰은 스킵
        if len(token) < 2:
            return token
        
        # 유사한 용어 찾기
        matches = self.find_similar_term(token, top_k=1)
        
        if matches:
            matched_term, similarity = matches[0]
            # print(f"[TextMatcher] '{token}' → '{matched_term}' (sim: {similarity:.3f})")
            return matched_term
        
        return token
    
    def refine_text(self, text: str) -> str:
        """
        전체 텍스트를 정제
        
        Args:
            text: 정제할 텍스트
            
        Returns:
            정제된 텍스트
        """
        # 공백으로 토큰화 (간단한 방식)
        tokens = text.split()
        
        # 각 토큰 정제
        refined_tokens = [self.refine_token(token) for token in tokens]
        
        # 재조합
        return " ".join(refined_tokens)


# 싱글톤 인스턴스
_vocab_embedder: Optional[VocabEmbedder] = None
_text_matcher: Optional[TextMatcher] = None


def get_text_matcher(
    threshold: float = 0.65,
    model_name: str = "jhgan/ko-sroberta-multitask"
) -> TextMatcher:
    """TextMatcher 싱글톤 인스턴스 반환"""
    global _vocab_embedder, _text_matcher
    
    # 모델 변경 감지 또는 초기화 필요 시
    if _vocab_embedder is None or _vocab_embedder.model_name != model_name:
        _vocab_embedder = VocabEmbedder(model_name=model_name)
        _text_matcher = None  # embedder가 바뀌면 matcher도 재생성
    
    if _text_matcher is None or _text_matcher.threshold != threshold:
        _text_matcher = TextMatcher(_vocab_embedder, threshold=threshold)
    
    return _text_matcher


def refine_text_with_embedding(
    text: str,
    threshold: float = 0.65,
    model_name: str = "jhgan/ko-sroberta-multitask"
) -> Dict[str, any]:
    if not text or not text.strip():
        return {"text": text, "keywords": []}
    
    try:
        # TextMatcher 가져오기
        matcher = get_text_matcher(threshold=threshold, model_name=model_name)
        
        # 텍스트 정제
        refined_text = matcher.refine_text(text)
        
        # 간단한 키워드 추출 (정제된 텍스트에서 금융 용어만 추출)
        keywords = []
        for token in refined_text.split():
            if token in matcher.vocab_embedder.vocab_embeddings:
                keywords.append(f"#{token}")
        
        # 중복 제거
        keywords = list(dict.fromkeys(keywords))
        
        return {
            "text": refined_text,
            "keywords": keywords[:3]  # 최대 3개
        }
    
    except Exception as e:
        print(f"[refine_text_with_embedding] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"text": text, "keywords": []}
