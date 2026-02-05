"""
RAG 평가 지표 모듈

포함된 지표:
1. 검색 품질 지표: Precision@K, Recall@K, MRR, NDCG, Hit Rate
2. RAGAS 프레임워크: Faithfulness, Answer Relevancy, Context Precision
3. 시맨틱 유사도: Cosine Similarity
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.llm.base import get_openai_client


# ============================================================================
# 1. 검색 품질 지표 (Retrieval Quality Metrics)
# ============================================================================

@dataclass
class RetrievalMetrics:
    """검색 품질 지표 결과"""
    precision_at_k: float = 0.0      # Top-K 중 관련 문서 비율
    recall_at_k: float = 0.0         # 전체 관련 문서 중 검색된 비율
    mrr: float = 0.0                 # Mean Reciprocal Rank
    ndcg_at_k: float = 0.0           # Normalized DCG
    hit_rate: float = 0.0            # Top-K에 관련 문서 있는지


def precision_at_k(retrieved_docs: List[Dict], relevant_ids: set, k: int = 5) -> float:
    """
    Precision@K: Top-K 검색 결과 중 관련 문서의 비율

    Args:
        retrieved_docs: 검색된 문서 리스트 (순위순)
        relevant_ids: 관련 문서 ID set
        k: 상위 K개

    Returns:
        Precision@K 값 (0~1)
    """
    if not retrieved_docs or k <= 0:
        return 0.0

    top_k = retrieved_docs[:k]
    relevant_count = sum(1 for doc in top_k if doc.get("id") in relevant_ids)
    return relevant_count / k


def recall_at_k(retrieved_docs: List[Dict], relevant_ids: set, k: int = 5) -> float:
    """
    Recall@K: 전체 관련 문서 중 Top-K에 포함된 비율

    Args:
        retrieved_docs: 검색된 문서 리스트 (순위순)
        relevant_ids: 관련 문서 ID set
        k: 상위 K개

    Returns:
        Recall@K 값 (0~1)
    """
    if not relevant_ids:
        return 0.0

    top_k = retrieved_docs[:k]
    retrieved_relevant = sum(1 for doc in top_k if doc.get("id") in relevant_ids)
    return retrieved_relevant / len(relevant_ids)


def mean_reciprocal_rank(retrieved_docs: List[Dict], relevant_ids: set) -> float:
    """
    MRR (Mean Reciprocal Rank): 첫 번째 관련 문서의 역순위

    Args:
        retrieved_docs: 검색된 문서 리스트 (순위순)
        relevant_ids: 관련 문서 ID set

    Returns:
        MRR 값 (0~1)
    """
    for i, doc in enumerate(retrieved_docs):
        if doc.get("id") in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """DCG (Discounted Cumulative Gain) 계산"""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(retrieved_docs: List[Dict], relevance_map: Dict[str, float], k: int = 5) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain)
    순위를 고려한 검색 품질 점수

    Args:
        retrieved_docs: 검색된 문서 리스트 (순위순)
        relevance_map: 문서 ID -> 관련도 점수 (0~1)
        k: 상위 K개

    Returns:
        NDCG@K 값 (0~1)
    """
    if not retrieved_docs or not relevance_map:
        return 0.0

    # 실제 DCG 계산
    actual_scores = [relevance_map.get(doc.get("id"), 0.0) for doc in retrieved_docs[:k]]
    actual_dcg = dcg_at_k(actual_scores, k)

    # 이상적인 DCG 계산 (관련도 높은 순으로 정렬)
    ideal_scores = sorted(relevance_map.values(), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_scores, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def hit_rate(retrieved_docs: List[Dict], relevant_ids: set, k: int = 5) -> float:
    """
    Hit Rate: Top-K에 관련 문서가 하나라도 있는지

    Returns:
        1.0 (hit) or 0.0 (miss)
    """
    top_k = retrieved_docs[:k]
    for doc in top_k:
        if doc.get("id") in relevant_ids:
            return 1.0
    return 0.0


def calculate_retrieval_metrics(
    retrieved_docs: List[Dict],
    relevant_ids: set = None,
    relevance_map: Dict[str, float] = None,
    k: int = 5
) -> RetrievalMetrics:
    """모든 검색 품질 지표 계산"""
    relevant_ids = relevant_ids or set()
    relevance_map = relevance_map or {doc_id: 1.0 for doc_id in relevant_ids}

    return RetrievalMetrics(
        precision_at_k=precision_at_k(retrieved_docs, relevant_ids, k),
        recall_at_k=recall_at_k(retrieved_docs, relevant_ids, k),
        mrr=mean_reciprocal_rank(retrieved_docs, relevant_ids),
        ndcg_at_k=ndcg_at_k(retrieved_docs, relevance_map, k),
        hit_rate=hit_rate(retrieved_docs, relevant_ids, k),
    )


# ============================================================================
# 2. 시맨틱 유사도 지표 (Semantic Similarity Metrics)
# ============================================================================

@dataclass
class SemanticMetrics:
    """시맨틱 유사도 지표"""
    avg_similarity: float = 0.0          # 평균 쿼리-문서 유사도
    max_similarity: float = 0.0          # 최대 유사도
    min_similarity: float = 0.0          # 최소 유사도
    similarity_scores: List[float] = field(default_factory=list)


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """OpenAI 임베딩 생성"""
    try:
        from app.rag.retriever.db import embed_query
        return embed_query(text, model)
    except Exception:
        client = get_openai_client()
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """코사인 유사도 계산"""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def calculate_semantic_metrics(
    query: str,
    retrieved_docs: List[Dict],
    use_cached_embeddings: bool = True
) -> SemanticMetrics:
    """
    쿼리와 검색된 문서 간의 시맨틱 유사도 계산

    Args:
        query: 검색 쿼리
        retrieved_docs: 검색된 문서 리스트
        use_cached_embeddings: 문서에 임베딩이 있으면 사용
    """
    if not retrieved_docs:
        return SemanticMetrics()

    query_embedding = get_embedding(query)
    similarity_scores = []

    for doc in retrieved_docs:
        # 문서 임베딩 가져오기
        doc_embedding = doc.get("embedding")
        if not doc_embedding or not use_cached_embeddings:
            content = doc.get("content", "") or doc.get("title", "")
            if not content:
                continue
            doc_embedding = get_embedding(content[:2000])  # 토큰 제한

        sim = cosine_similarity(query_embedding, doc_embedding)
        similarity_scores.append(sim)

    if not similarity_scores:
        return SemanticMetrics()

    return SemanticMetrics(
        avg_similarity=float(np.mean(similarity_scores)),
        max_similarity=float(np.max(similarity_scores)),
        min_similarity=float(np.min(similarity_scores)),
        similarity_scores=similarity_scores,
    )


# ============================================================================
# 3. RAGAS 프레임워크 지표 (LLM-as-Judge)
# ============================================================================

@dataclass
class RAGASMetrics:
    """RAGAS 프레임워크 평가 지표"""
    faithfulness: float = 0.0        # 답변이 컨텍스트에 근거하는지
    answer_relevancy: float = 0.0    # 답변이 질문과 관련있는지
    context_precision: float = 0.0   # 컨텍스트가 질문에 유용한지
    context_recall: float = 0.0      # 필요한 정보가 컨텍스트에 있는지
    overall_score: float = 0.0       # 전체 점수


RAGAS_SYSTEM_PROMPT = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Evaluate the following based on the given criteria. Return ONLY a JSON object with numeric scores.
All scores must be between 0.0 and 1.0."""


def evaluate_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Faithfulness: 답변이 검색된 컨텍스트에 근거하는지 평가

    - 1.0: 답변의 모든 내용이 컨텍스트에 근거함
    - 0.0: 답변이 컨텍스트와 무관하거나 환각
    """
    if not answer or not contexts:
        return 0.0

    context_text = "\n---\n".join(contexts[:3])

    prompt = f"""Evaluate FAITHFULNESS: Does the answer contain ONLY information from the given contexts?

Context:
{context_text}

Answer:
{answer}

Score (0.0-1.0):
- 1.0 = All claims in answer are supported by context
- 0.5 = Some claims are supported, some are not
- 0.0 = Answer contains hallucinations or unsupported claims

Return JSON: {{"faithfulness": <score>}}"""

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": RAGAS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50,
        )
        import json
        result = json.loads(resp.choices[0].message.content)
        return float(result.get("faithfulness", 0.0))
    except Exception:
        return 0.0


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: 답변이 질문과 관련있는지 평가

    - 1.0: 답변이 질문을 완벽히 해결
    - 0.0: 답변이 질문과 무관
    """
    if not question or not answer:
        return 0.0

    prompt = f"""Evaluate ANSWER RELEVANCY: Does the answer address the question?

Question: {question}
Answer: {answer}

Score (0.0-1.0):
- 1.0 = Answer completely addresses the question
- 0.5 = Answer partially addresses the question
- 0.0 = Answer is irrelevant to the question

Return JSON: {{"answer_relevancy": <score>}}"""

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": RAGAS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50,
        )
        import json
        result = json.loads(resp.choices[0].message.content)
        return float(result.get("answer_relevancy", 0.0))
    except Exception:
        return 0.0


def evaluate_context_precision(question: str, contexts: List[str]) -> float:
    """
    Context Precision: 검색된 컨텍스트가 질문 해결에 유용한지

    - 1.0: 모든 컨텍스트가 유용
    - 0.0: 컨텍스트가 질문과 무관
    """
    if not question or not contexts:
        return 0.0

    context_text = "\n---\n".join(contexts[:3])

    prompt = f"""Evaluate CONTEXT PRECISION: Are the retrieved contexts useful for answering the question?

Question: {question}

Retrieved Contexts:
{context_text}

Score (0.0-1.0):
- 1.0 = All contexts are highly relevant and useful
- 0.5 = Some contexts are useful
- 0.0 = Contexts are irrelevant to the question

Return JSON: {{"context_precision": <score>}}"""

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": RAGAS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50,
        )
        import json
        result = json.loads(resp.choices[0].message.content)
        return float(result.get("context_precision", 0.0))
    except Exception:
        return 0.0


def evaluate_context_recall(question: str, contexts: List[str], ground_truth: str = None) -> float:
    """
    Context Recall: 질문에 필요한 정보가 컨텍스트에 있는지

    ground_truth가 없으면 질문 기반으로 추론
    """
    if not question or not contexts:
        return 0.0

    context_text = "\n---\n".join(contexts[:3])

    if ground_truth:
        prompt = f"""Evaluate CONTEXT RECALL: Do the contexts contain all information needed for the ground truth answer?

Question: {question}
Ground Truth Answer: {ground_truth}

Retrieved Contexts:
{context_text}

Score (0.0-1.0):
- 1.0 = Contexts contain all information needed
- 0.5 = Contexts contain some needed information
- 0.0 = Contexts miss critical information

Return JSON: {{"context_recall": <score>}}"""
    else:
        prompt = f"""Evaluate CONTEXT RECALL: Do the contexts contain enough information to answer the question?

Question: {question}

Retrieved Contexts:
{context_text}

Score (0.0-1.0):
- 1.0 = Contexts have sufficient information
- 0.5 = Contexts have partial information
- 0.0 = Contexts lack necessary information

Return JSON: {{"context_recall": <score>}}"""

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": RAGAS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50,
        )
        import json
        result = json.loads(resp.choices[0].message.content)
        return float(result.get("context_recall", 0.0))
    except Exception:
        return 0.0


def calculate_ragas_metrics(
    question: str,
    answer: str = None,
    contexts: List[str] = None,
    ground_truth: str = None,
) -> RAGASMetrics:
    """RAGAS 전체 지표 계산"""
    contexts = contexts or []

    faithfulness = evaluate_faithfulness(answer, contexts) if answer else 0.0
    answer_relevancy = evaluate_answer_relevancy(question, answer) if answer else 0.0
    context_precision = evaluate_context_precision(question, contexts)
    context_recall = evaluate_context_recall(question, contexts, ground_truth)

    # 전체 점수 (가중 평균)
    scores = [faithfulness, answer_relevancy, context_precision, context_recall]
    valid_scores = [s for s in scores if s > 0]
    overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    return RAGASMetrics(
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
        context_recall=context_recall,
        overall_score=overall,
    )


# ============================================================================
# 통합 평가 클래스
# ============================================================================

@dataclass
class RAGEvaluationResult:
    """전체 RAG 평가 결과"""
    query: str
    retrieval: RetrievalMetrics
    semantic: SemanticMetrics
    ragas: RAGASMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieval": {
                "precision_at_k": self.retrieval.precision_at_k,
                "recall_at_k": self.retrieval.recall_at_k,
                "mrr": self.retrieval.mrr,
                "ndcg_at_k": self.retrieval.ndcg_at_k,
                "hit_rate": self.retrieval.hit_rate,
            },
            "semantic": {
                "avg_similarity": self.semantic.avg_similarity,
                "max_similarity": self.semantic.max_similarity,
            },
            "ragas": {
                "faithfulness": self.ragas.faithfulness,
                "answer_relevancy": self.ragas.answer_relevancy,
                "context_precision": self.ragas.context_precision,
                "context_recall": self.ragas.context_recall,
                "overall_score": self.ragas.overall_score,
            },
        }


def evaluate_rag_response(
    query: str,
    retrieved_docs: List[Dict],
    answer: str = None,
    relevant_ids: set = None,
    ground_truth: str = None,
    k: int = 5,
    evaluate_semantic: bool = False,  # 비용 절감을 위해 기본 비활성화
    evaluate_ragas: bool = True,
) -> RAGEvaluationResult:
    """
    RAG 응답 종합 평가

    Args:
        query: 검색 쿼리
        retrieved_docs: 검색된 문서 리스트
        answer: 생성된 답변 (선택)
        relevant_ids: 관련 문서 ID set (Ground Truth)
        ground_truth: 정답 텍스트 (선택)
        k: Top-K
        evaluate_semantic: 시맨틱 유사도 평가 여부
        evaluate_ragas: RAGAS 평가 여부
    """
    # 1. 검색 품질 지표
    retrieval = calculate_retrieval_metrics(
        retrieved_docs, relevant_ids, k=k
    )

    # 2. 시맨틱 유사도 (선택)
    if evaluate_semantic:
        semantic = calculate_semantic_metrics(query, retrieved_docs)
    else:
        semantic = SemanticMetrics()

    # 3. RAGAS 평가 (선택)
    if evaluate_ragas:
        contexts = [doc.get("content", "") for doc in retrieved_docs[:3]]
        ragas = calculate_ragas_metrics(
            question=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
    else:
        ragas = RAGASMetrics()

    return RAGEvaluationResult(
        query=query,
        retrieval=retrieval,
        semantic=semantic,
        ragas=ragas,
    )


# ============================================================================
# 배치 평가 유틸리티
# ============================================================================

def aggregate_metrics(results: List[RAGEvaluationResult]) -> Dict[str, float]:
    """여러 평가 결과의 평균 계산"""
    if not results:
        return {}

    return {
        # Retrieval
        "avg_precision_at_k": np.mean([r.retrieval.precision_at_k for r in results]),
        "avg_recall_at_k": np.mean([r.retrieval.recall_at_k for r in results]),
        "avg_mrr": np.mean([r.retrieval.mrr for r in results]),
        "avg_ndcg_at_k": np.mean([r.retrieval.ndcg_at_k for r in results]),
        "avg_hit_rate": np.mean([r.retrieval.hit_rate for r in results]),
        # Semantic
        "avg_semantic_similarity": np.mean([r.semantic.avg_similarity for r in results if r.semantic.avg_similarity > 0]),
        # RAGAS
        "avg_faithfulness": np.mean([r.ragas.faithfulness for r in results if r.ragas.faithfulness > 0]),
        "avg_answer_relevancy": np.mean([r.ragas.answer_relevancy for r in results if r.ragas.answer_relevancy > 0]),
        "avg_context_precision": np.mean([r.ragas.context_precision for r in results if r.ragas.context_precision > 0]),
        "avg_context_recall": np.mean([r.ragas.context_recall for r in results if r.ragas.context_recall > 0]),
        "avg_ragas_overall": np.mean([r.ragas.overall_score for r in results if r.ragas.overall_score > 0]),
    }


def print_evaluation_summary(results: List[RAGEvaluationResult]):
    """평가 결과 요약 출력"""
    agg = aggregate_metrics(results)

    print("\n" + "=" * 70)
    print("RAG 평가 지표 요약")
    print("=" * 70)

    print("\n📊 검색 품질 지표 (Retrieval Quality)")
    print(f"  Precision@K:    {agg.get('avg_precision_at_k', 0):.3f}")
    print(f"  Recall@K:       {agg.get('avg_recall_at_k', 0):.3f}")
    print(f"  MRR:            {agg.get('avg_mrr', 0):.3f}")
    print(f"  NDCG@K:         {agg.get('avg_ndcg_at_k', 0):.3f}")
    print(f"  Hit Rate:       {agg.get('avg_hit_rate', 0):.3f}")

    if agg.get('avg_semantic_similarity', 0) > 0:
        print("\n📊 시맨틱 유사도 (Semantic Similarity)")
        print(f"  평균 유사도:     {agg.get('avg_semantic_similarity', 0):.3f}")

    if agg.get('avg_ragas_overall', 0) > 0:
        print("\n📊 RAGAS 지표 (LLM-as-Judge)")
        print(f"  Faithfulness:       {agg.get('avg_faithfulness', 0):.3f}")
        print(f"  Answer Relevancy:   {agg.get('avg_answer_relevancy', 0):.3f}")
        print(f"  Context Precision:  {agg.get('avg_context_precision', 0):.3f}")
        print(f"  Context Recall:     {agg.get('avg_context_recall', 0):.3f}")
        print(f"  Overall Score:      {agg.get('avg_ragas_overall', 0):.3f}")

    print("=" * 70)
