"""
RAG 검색 성능 테스트 스크립트

테스트 항목:
1. Retrieval 품질 테스트 (Precision, Recall, MRR)
2. 응답 속도 테스트 (지연 시간)
3. 쿼리 라우팅 정확도 테스트
"""

import asyncio
import time
import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import statistics

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag.pipeline.search import run_search, route
from app.rag.router.router import route_query
from app.rag.retriever.db import warmup_embed_cache

# RAG 평가 지표 모듈
from tests.rag_metrics import (
    RAGEvaluationResult,
    evaluate_rag_response,
    aggregate_metrics,
    print_evaluation_summary,
)

# Ground Truth 테스트 케이스 (test_suite.py)
from app.rag.tests.test_suite import TESTS as SUITE_TESTS


# ============================================================================
# 테스트 데이터셋 정의
# ============================================================================

@dataclass
class TestCase:
    """테스트 케이스"""
    query: str                              # 테스트 쿼리
    expected_route: str                     # 기대 라우팅 (card_info, card_usage, none)
    expected_keywords: List[str] = field(default_factory=list)  # 검색 결과에 포함되어야 할 키워드
    category: str = ""                      # 테스트 카테고리
    # Ground Truth (test_suite.py 형식)
    must_have_doc_ids: List[str] = field(default_factory=list)  # 반드시 검색되어야 할 문서 ID
    must_not_have_doc_ids: List[str] = field(default_factory=list)  # 검색되면 안되는 문서 ID
    must_have_answer_terms: List[str] = field(default_factory=list)  # 답변에 포함되어야 할 키워드
    must_not_have_answer_terms: List[str] = field(default_factory=list)  # 답변에 포함되면 안되는 키워드
    test_id: str = ""  # 테스트 ID


# 테스트 케이스 정의 (DB 카테고리 기준: 8대분류)
# 대분류: 결제/승인, 이용내역, 한도, 분실/도난, 수수료/연체, 포인트/혜택, 정부지원, 기타
TEST_CASES = [
    # ===== 카드 정보 조회 (card_info) =====
    # 포인트/혜택
    TestCase("나라사랑카드 혜택 알려주세요", "card_info", ["나라사랑카드", "혜택"], "포인트/혜택"),
    TestCase("K-패스 할인 되는 카드 추천해주세요", "card_info", ["K-패스", "할인"], "포인트/혜택"),
    TestCase("주유 할인 카드 있어요?", "card_info", ["주유", "할인"], "포인트/혜택"),
    TestCase("마일리지 적립 잘 되는 카드 추천해주세요", "card_info", ["마일리지", "적립"], "포인트/혜택"),

    # 기타 (카드 발급/정보)
    TestCase("국민행복카드 발급 조건이 뭐예요?", "card_info", ["국민행복카드", "발급"], "기타"),
    TestCase("신용카드 발급하려면 어떻게 해요?", "card_info", ["신용카드", "발급"], "기타"),

    # ===== 카드 사용법/서비스 안내 (card_usage) =====
    # 분실/도난
    TestCase("카드 분실신고 하려고요", "card_usage", ["분실"], "분실/도난"),
    TestCase("카드를 잃어버렸어요", "card_usage", ["분실", "도난"], "분실/도난"),
    TestCase("카드 도난당했어요 어떻게 해요?", "card_usage", ["도난"], "분실/도난"),

    # 결제/승인
    TestCase("결제가 안돼요", "card_usage", ["결제"], "결제/승인"),
    TestCase("결제일 변경하고 싶어요", "card_usage", ["결제일", "변경"], "결제/승인"),
    TestCase("이번 달 결제금액 알려주세요", "card_usage", ["결제", "금액"], "결제/승인"),

    # 한도
    TestCase("카드 한도 올리고 싶어요", "card_usage", ["한도"], "한도"),
    TestCase("이용한도 조회하려고요", "card_usage", ["한도", "조회"], "한도"),

    # 수수료/연체
    TestCase("리볼빙 신청하고 싶어요", "card_usage", ["리볼빙", "신청"], "수수료/연체"),
    TestCase("리볼빙 취소 방법", "card_usage", ["리볼빙", "취소"], "수수료/연체"),
    TestCase("연체이자 얼마예요?", "card_usage", ["연체", "이자"], "수수료/연체"),

    # 기타 (서비스)
    TestCase("카드론 신청하려고요", "card_usage", ["카드론", "신청"], "기타"),
    TestCase("현금서비스 이용방법", "card_usage", ["현금서비스"], "기타"),
    TestCase("삼성페이 등록하려고요", "card_usage", ["삼성페이", "등록"], "기타"),
    TestCase("카드 해지하려고요", "card_usage", ["해지"], "기타"),
    TestCase("카드 재발급 신청하려고요", "card_usage", ["재발급"], "기타"),

    # 정부지원
    TestCase("국민행복카드 바우처 사용법", "card_usage", ["국민행복카드", "바우처"], "정부지원"),

    # 이용내역
    TestCase("이번 달 카드 사용내역 보여주세요", "card_usage", ["사용내역", "이용내역"], "이용내역"),

    # 복합 쿼리
    TestCase("나라사랑카드 분실해서 재발급 받고 싶어요", "card_usage", ["나라사랑카드", "분실", "발급"], "분실/도난"),
    TestCase("리볼빙 서비스 신청했는데 취소하고 싶어요", "card_usage", ["리볼빙", "취소"], "수수료/연체"),
]


def _suite_to_testcase(t: Dict[str, Any]) -> TestCase:
    """test_suite.py 형식을 TestCase로 변환"""
    route = t.get("expect_route", "card_usage")
    # 카테고리 추정
    query = t.get("query", "").lower()
    if "분실" in query or "도난" in query or "잃어버" in query:
        category = "분실/도난"
    elif "결제" in query or "승인" in query:
        category = "결제/승인"
    elif "한도" in query:
        category = "한도"
    elif "리볼빙" in query or "연체" in query or "이자" in query or "수수료" in query:
        category = "수수료/연체"
    elif "혜택" in query or "포인트" in query or "적립" in query or "할인" in query:
        category = "포인트/혜택"
    elif "국민행복" in query or "바우처" in query or "다둥이" in query:
        category = "정부지원"
    elif "k패스" in query or "k-패스" in query:
        category = "포인트/혜택"
    else:
        category = "기타"

    return TestCase(
        query=t.get("query", ""),
        expected_route=route,
        expected_keywords=t.get("must_have_answer_terms", []),
        category=category,
        must_have_doc_ids=t.get("must_have_doc_ids", []),
        must_not_have_doc_ids=t.get("must_not_have_doc_ids", []),
        must_have_answer_terms=t.get("must_have_answer_terms", []),
        must_not_have_answer_terms=t.get("must_not_have_answer_terms", []),
        test_id=t.get("id", ""),
    )


# Ground Truth 테스트 케이스 (test_suite.py 기반)
GROUND_TRUTH_CASES = [_suite_to_testcase(t) for t in SUITE_TESTS]


# ============================================================================
# 성능 측정 클래스
# ============================================================================

@dataclass
class PerformanceMetrics:
    """성능 측정 결과"""
    # 속도 관련
    latencies: List[float] = field(default_factory=list)
    route_times: List[float] = field(default_factory=list)
    retrieve_times: List[float] = field(default_factory=list)

    # 품질 관련
    routing_correct: int = 0
    routing_total: int = 0
    keyword_hits: int = 0
    keyword_total: int = 0
    docs_returned: List[int] = field(default_factory=list)

    # 카테고리별 결과
    category_results: Dict[str, Dict[str, Any]] = field(default_factory=lambda: defaultdict(lambda: {
        "correct": 0, "total": 0, "latencies": []
    }))

    # RAG 평가 지표
    rag_evaluations: List[RAGEvaluationResult] = field(default_factory=list)

    def add_result(
        self,
        latency: float,
        route_time: float,
        retrieve_time: float,
        routing_correct: bool,
        keyword_hit_count: int,
        keyword_total_count: int,
        docs_count: int,
        category: str
    ):
        self.latencies.append(latency)
        self.route_times.append(route_time)
        self.retrieve_times.append(retrieve_time)

        if routing_correct:
            self.routing_correct += 1
        self.routing_total += 1

        self.keyword_hits += keyword_hit_count
        self.keyword_total += keyword_total_count
        self.docs_returned.append(docs_count)

        # 카테고리별 저장
        self.category_results[category]["total"] += 1
        if routing_correct:
            self.category_results[category]["correct"] += 1
        self.category_results[category]["latencies"].append(latency)

    def summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        return {
            "latency": {
                "mean_ms": statistics.mean(self.latencies) * 1000 if self.latencies else 0,
                "median_ms": statistics.median(self.latencies) * 1000 if self.latencies else 0,
                "p95_ms": self._percentile(self.latencies, 95) * 1000 if self.latencies else 0,
                "p99_ms": self._percentile(self.latencies, 99) * 1000 if self.latencies else 0,
                "min_ms": min(self.latencies) * 1000 if self.latencies else 0,
                "max_ms": max(self.latencies) * 1000 if self.latencies else 0,
            },
            "route_time": {
                "mean_ms": statistics.mean(self.route_times) * 1000 if self.route_times else 0,
            },
            "retrieve_time": {
                "mean_ms": statistics.mean(self.retrieve_times) * 1000 if self.retrieve_times else 0,
            },
            "routing_accuracy": self.routing_correct / self.routing_total if self.routing_total else 0,
            "keyword_precision": self.keyword_hits / self.keyword_total if self.keyword_total else 0,
            "avg_docs_returned": statistics.mean(self.docs_returned) if self.docs_returned else 0,
            "total_tests": self.routing_total,
        }

    def _percentile(self, data: List[float], p: int) -> float:
        if not data:
            return 0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


# ============================================================================
# 테스트 함수들
# ============================================================================

async def test_single_query(
    test_case: TestCase,
    top_k: int = 5,
    verbose: bool = False,
    evaluate_ragas: bool = False,
) -> Tuple[Dict[str, Any], PerformanceMetrics]:
    """단일 쿼리 테스트"""
    metrics = PerformanceMetrics()

    start_time = time.perf_counter()

    try:
        result = await run_search(
            query=test_case.query,
            top_k=top_k,
            enable_consult_search=False
        )

        end_time = time.perf_counter()
        latency = end_time - start_time
        route_time = result.t_route - result.t_start
        retrieve_time = result.t_retrieve - result.t_route

        # 라우팅 정확도 체크
        actual_route = result.routing.get("route") or result.routing.get("ui_route")
        routing_correct = actual_route == test_case.expected_route

        # 키워드 매칭 체크
        keyword_hits = 0
        if test_case.expected_keywords and result.docs:
            all_content = " ".join([
                str(doc.get("title", "")) + " " + str(doc.get("content", ""))
                for doc in result.docs
            ]).lower()

            for keyword in test_case.expected_keywords:
                if keyword.lower() in all_content:
                    keyword_hits += 1

        metrics.add_result(
            latency=latency,
            route_time=route_time,
            retrieve_time=retrieve_time,
            routing_correct=routing_correct,
            keyword_hit_count=keyword_hits,
            keyword_total_count=len(test_case.expected_keywords),
            docs_count=len(result.docs),
            category=test_case.category
        )

        # RAG 평가 지표 계산 (선택적)
        if evaluate_ragas and result.docs:
            # Ground Truth 문서 ID가 있으면 retrieval metrics 계산 가능
            relevant_ids = set(test_case.must_have_doc_ids) if test_case.must_have_doc_ids else None
            rag_eval = evaluate_rag_response(
                query=test_case.query,
                retrieved_docs=result.docs,
                answer=None,  # 답변 생성이 없으므로 None
                relevant_ids=relevant_ids,
                evaluate_semantic=False,
                evaluate_ragas=True,
            )
            metrics.rag_evaluations.append(rag_eval)

        detail = {
            "query": test_case.query,
            "expected_route": test_case.expected_route,
            "actual_route": actual_route,
            "routing_correct": routing_correct,
            "latency_ms": latency * 1000,
            "docs_count": len(result.docs),
            "keyword_hits": keyword_hits,
            "keyword_total": len(test_case.expected_keywords),
            "retrieval_mode": result.routing.get("retrieval_mode"),
            "cache_status": result.retrieve_cache_status,
        }

        if verbose and result.docs:
            detail["top_doc_title"] = result.docs[0].get("title", "N/A")[:50]
            detail["top_doc_score"] = result.docs[0].get("score", 0)

        return detail, metrics

    except Exception as e:
        return {"query": test_case.query, "error": str(e)}, metrics


async def run_performance_test(
    test_cases: List[TestCase] = None,
    top_k: int = 5,
    iterations: int = 1,
    verbose: bool = True,
    evaluate_ragas: bool = False,
) -> Dict[str, Any]:
    """전체 성능 테스트 실행"""

    if test_cases is None:
        test_cases = TEST_CASES

    # 임베딩 캐시 워밍업 (실제 검색 실행으로 캐시 채우기)
    print("=" * 70)
    print("캐시 워밍업 중...")
    for tc in test_cases:
        try:
            await run_search(tc.query, top_k=top_k)
        except Exception:
            pass
    print("워밍업 완료")

    print("=" * 70)
    print("RAG 검색 성능 테스트")
    print("=" * 70)
    print(f"테스트 케이스 수: {len(test_cases)}")
    print(f"반복 횟수: {iterations}")
    print(f"Top-K: {top_k}")
    print(f"RAGAS 평가: {'ON' if evaluate_ragas else 'OFF'}")
    print("=" * 70)

    all_metrics = PerformanceMetrics()
    all_details = []

    for iteration in range(iterations):
        if iterations > 1:
            print(f"\n[반복 {iteration + 1}/{iterations}]")

        for i, test_case in enumerate(test_cases, 1):
            detail, metrics = await test_single_query(
                test_case=test_case,
                top_k=top_k,
                verbose=verbose,
                evaluate_ragas=evaluate_ragas,
            )

            # 메트릭 병합
            all_metrics.latencies.extend(metrics.latencies)
            all_metrics.route_times.extend(metrics.route_times)
            all_metrics.retrieve_times.extend(metrics.retrieve_times)
            all_metrics.routing_correct += metrics.routing_correct
            all_metrics.routing_total += metrics.routing_total
            all_metrics.keyword_hits += metrics.keyword_hits
            all_metrics.keyword_total += metrics.keyword_total
            all_metrics.docs_returned.extend(metrics.docs_returned)
            all_metrics.rag_evaluations.extend(metrics.rag_evaluations)

            for cat, cat_data in metrics.category_results.items():
                all_metrics.category_results[cat]["total"] += cat_data["total"]
                all_metrics.category_results[cat]["correct"] += cat_data["correct"]
                all_metrics.category_results[cat]["latencies"].extend(cat_data["latencies"])

            all_details.append(detail)

            # 진행 상황 출력
            status = "✓" if detail.get("routing_correct", False) else "✗"
            latency = detail.get("latency_ms", 0)
            print(f"  [{i}/{len(test_cases)}] {status} {test_case.query[:40]:<40} | {latency:>7.1f}ms | docs: {detail.get('docs_count', 0)}")

    # 최종 결과 요약
    summary = all_metrics.summary()

    print("\n" + "=" * 70)
    print("테스트 결과 요약")
    print("=" * 70)

    print("\n📊 속도 성능")
    print(f"  평균 응답 시간:    {summary['latency']['mean_ms']:.1f}ms")
    print(f"  중앙값 응답 시간:  {summary['latency']['median_ms']:.1f}ms")
    print(f"  P95 응답 시간:     {summary['latency']['p95_ms']:.1f}ms")
    print(f"  P99 응답 시간:     {summary['latency']['p99_ms']:.1f}ms")
    print(f"  최소/최대:         {summary['latency']['min_ms']:.1f}ms / {summary['latency']['max_ms']:.1f}ms")

    print("\n📊 단계별 소요 시간")
    print(f"  라우팅 평균:       {summary['route_time']['mean_ms']:.1f}ms")
    print(f"  검색 평균:         {summary['retrieve_time']['mean_ms']:.1f}ms")

    print("\n📊 품질 지표")
    print(f"  라우팅 정확도:     {summary['routing_accuracy']*100:.1f}% ({all_metrics.routing_correct}/{all_metrics.routing_total})")
    print(f"  키워드 적중률:     {summary['keyword_precision']*100:.1f}% ({all_metrics.keyword_hits}/{all_metrics.keyword_total})")
    print(f"  평균 문서 수:      {summary['avg_docs_returned']:.1f}개")

    print("\n📊 카테고리별 결과")
    for cat, cat_data in sorted(all_metrics.category_results.items()):
        acc = cat_data["correct"] / cat_data["total"] * 100 if cat_data["total"] else 0
        avg_lat = statistics.mean(cat_data["latencies"]) * 1000 if cat_data["latencies"] else 0
        print(f"  {cat:<20} 정확도: {acc:>5.1f}% | 평균: {avg_lat:>6.1f}ms")

    # RAGAS 평가 결과 출력
    if all_metrics.rag_evaluations:
        print_evaluation_summary(all_metrics.rag_evaluations)

    print("\n" + "=" * 70)

    return {
        "summary": summary,
        "details": all_details,
        "category_results": dict(all_metrics.category_results),
        "rag_evaluations": [e.to_dict() for e in all_metrics.rag_evaluations] if all_metrics.rag_evaluations else [],
    }


async def test_routing_only(test_cases: List[TestCase] = None) -> Dict[str, Any]:
    """라우팅만 테스트 (DB 연결 없이)"""

    if test_cases is None:
        test_cases = TEST_CASES

    print("=" * 70)
    print("라우팅 정확도 테스트 (DB 연결 불필요)")
    print("=" * 70)

    correct = 0
    total = len(test_cases)
    results = []

    for test_case in test_cases:
        start = time.perf_counter()
        routing = route_query(test_case.query)
        elapsed = time.perf_counter() - start

        actual_route = routing.get("route") or routing.get("ui_route")
        is_correct = actual_route == test_case.expected_route

        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"  {status} {test_case.query[:50]:<50} | 예상: {test_case.expected_route:<12} | 실제: {actual_route:<12} | {elapsed*1000:.1f}ms")

        results.append({
            "query": test_case.query,
            "expected": test_case.expected_route,
            "actual": actual_route,
            "correct": is_correct,
            "latency_ms": elapsed * 1000,
            "routing_details": routing
        })

    print("\n" + "=" * 70)
    print(f"라우팅 정확도: {correct}/{total} ({correct/total*100:.1f}%)")
    print("=" * 70)

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "results": results
    }


# ============================================================================
# 메인 실행
# ============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG 검색 성능 테스트")
    parser.add_argument("--mode", choices=["full", "routing", "speed", "ground-truth"], default="full",
                       help="테스트 모드: full(전체), routing(라우팅만), speed(속도만), ground-truth(test_suite.py 기반)")
    parser.add_argument("--iterations", type=int, default=1, help="반복 횟수")
    parser.add_argument("--top-k", type=int, default=5, help="검색 결과 수")
    parser.add_argument("--verbose", action="store_true", help="상세 출력")
    parser.add_argument("--ragas", action="store_true", help="RAGAS 평가 지표 활성화 (LLM 호출)")

    args = parser.parse_args()

    if args.mode == "routing":
        await test_routing_only()
    elif args.mode == "speed":
        # 동일 쿼리 반복으로 속도만 측정
        speed_cases = [
            TestCase("카드 분실신고", "card_usage"),
            TestCase("나라사랑카드 혜택", "card_info"),
        ]
        await run_performance_test(
            test_cases=speed_cases,
            top_k=args.top_k,
            iterations=args.iterations * 5,
            verbose=args.verbose,
            evaluate_ragas=False,
        )
    elif args.mode == "ground-truth":
        # test_suite.py 기반 Ground Truth 테스트 (retrieval metrics 계산)
        await run_performance_test(
            test_cases=GROUND_TRUTH_CASES,
            top_k=args.top_k,
            iterations=args.iterations,
            verbose=args.verbose,
            evaluate_ragas=True,  # Ground Truth가 있으므로 RAGAS 활성화
        )
    else:
        await run_performance_test(
            top_k=args.top_k,
            iterations=args.iterations,
            verbose=args.verbose,
            evaluate_ragas=args.ragas,
        )


if __name__ == "__main__":
    asyncio.run(main())
