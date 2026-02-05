"""
Cross-Encoder 리랭킹 모듈

초기 검색 결과를 정밀 재순위화하여 Precision 향상
- Cross-Encoder 모델 (sentence-transformers)
- LLM 기반 폴백 옵션
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache

_RERANK_ENABLED = os.getenv("RAG_RERANK", "1") != "0"
_RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "ms-marco-MiniLM-L-6-v2")
_RERANK_TOP_K = int(os.getenv("RAG_RERANK_TOP_K", "10"))
_USE_LLM_RERANK = os.getenv("RAG_RERANK_USE_LLM", "0") == "1"

_cross_encoder = None
_LLM_CLIENT = None


def _get_llm_client():
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        from openai import OpenAI
        _LLM_CLIENT = OpenAI()
    return _LLM_CLIENT


def _load_cross_encoder():
    """Cross-Encoder 모델 로드 (lazy loading)"""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(_RERANK_MODEL)
        return _cross_encoder
    except ImportError:
        print("[rerank] sentence-transformers not installed, using LLM fallback")
        return None
    except Exception as e:
        print(f"[rerank] Failed to load cross-encoder: {e}")
        return None


def _extract_doc_text(doc: Dict[str, Any]) -> str:
    """문서에서 텍스트 추출"""
    title = doc.get("title") or ""
    content = doc.get("content") or ""
    if title and content:
        return f"{title}\n{content[:500]}"
    return content[:500] or title


def rerank_with_cross_encoder(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Cross-Encoder로 리랭킹"""
    if not docs:
        return docs

    top_k = top_k or _RERANK_TOP_K
    encoder = _load_cross_encoder()

    if encoder is None:
        # 폴백: LLM 리랭킹 또는 원본 반환
        if _USE_LLM_RERANK:
            return rerank_with_llm(query, docs, top_k)
        return docs

    # 쿼리-문서 쌍 생성
    pairs = [(query, _extract_doc_text(doc)) for doc in docs]

    # 점수 계산
    scores = encoder.predict(pairs)

    # 점수 기준 정렬
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # 리랭킹 점수 추가
    result = []
    for doc, score in scored_docs[:top_k]:
        doc_copy = doc.copy()
        doc_copy["rerank_score"] = float(score)
        doc_copy["original_score"] = doc.get("score", 0)
        result.append(doc_copy)

    return result


LLM_RERANK_PROMPT = """다음 문서들이 질문에 얼마나 관련있는지 평가해주세요.

질문: {query}

문서들:
{docs}

각 문서의 관련성을 0-10 점수로 평가하고 JSON 배열로 응답해주세요.
[{{"doc_idx": 0, "score": 8, "reason": "직접 관련"}}, ...]
관련성 높은 순서대로 정렬해주세요."""


def rerank_with_llm(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """LLM 기반 리랭킹 (폴백)"""
    if not docs:
        return docs

    top_k = top_k or _RERANK_TOP_K
    docs_to_rerank = docs[:min(len(docs), 10)]  # LLM 비용 제한

    # 문서 텍스트 포맷
    docs_text = "\n\n".join([
        f"[문서 {i}]\n제목: {doc.get('title', 'N/A')}\n내용: {_extract_doc_text(doc)[:300]}"
        for i, doc in enumerate(docs_to_rerank)
    ])

    try:
        client = _get_llm_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "JSON 배열로만 응답하세요."},
                {"role": "user", "content": LLM_RERANK_PROMPT.format(query=query, docs=docs_text)},
            ],
            temperature=0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content
        # JSON 파싱
        try:
            rankings = json.loads(result_text)
            if isinstance(rankings, dict) and "rankings" in rankings:
                rankings = rankings["rankings"]
            elif isinstance(rankings, dict):
                rankings = list(rankings.values())[0] if rankings else []
        except:
            return docs[:top_k]

        # 점수 기준 재정렬
        score_map = {r.get("doc_idx", i): r.get("score", 0) for i, r in enumerate(rankings)}

        result = []
        for i, doc in enumerate(docs_to_rerank):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = score_map.get(i, 0)
            doc_copy["original_score"] = doc.get("score", 0)
            result.append(doc_copy)

        result.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return result[:top_k]

    except Exception as e:
        print(f"[rerank] LLM rerank failed: {e}")
        return docs[:top_k]


def rerank(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """메인 리랭킹 함수"""
    if not _RERANK_ENABLED or not docs:
        return docs[:top_k] if top_k else docs

    if _USE_LLM_RERANK:
        return rerank_with_llm(query, docs, top_k)

    return rerank_with_cross_encoder(query, docs, top_k)


class Reranker:
    """리랭커 클래스"""

    def __init__(self, use_llm: bool = False, top_k: int = 10):
        self.use_llm = use_llm
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        top_k = top_k or self.top_k
        if self.use_llm:
            return rerank_with_llm(query, docs, top_k)
        return rerank_with_cross_encoder(query, docs, top_k)


# 싱글톤
_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker(use_llm=_USE_LLM_RERANK, top_k=_RERANK_TOP_K)
    return _reranker
