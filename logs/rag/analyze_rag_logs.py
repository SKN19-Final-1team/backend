#!/usr/bin/env python3
"""
RAG 검색 로그 분석 스크립트 - 개선 추적용

사용법:
    python analyze_rag_logs.py                    # 오늘 로그 분석
    python analyze_rag_logs.py rag_20260210.jsonl  # 특정 파일 분석
    python analyze_rag_logs.py --all               # 전체 로그 분석

측정 항목:
    1. 검색 응답 시간: search_time_ms 분포
    2. 유사도 점수: relevanceScore 분포 (0-100)
    3. 라우팅 분류: route별 검색 건수
    4. 문서 적중률: 검색 결과 0건인 비율
    5. 자주 검색되는 쿼리 패턴
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime


def load_logs(filepath):
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def analyze(entries):
    if not entries:
        print("분석할 로그가 없습니다.")
        return

    total = len(entries)

    print("=" * 70)
    print(f"  RAG 검색 로그 분석 리포트 ({entries[0]['ts'][:10]})")
    print("=" * 70)
    print(f"\n총 {total}건 검색")

    # ── 1. 검색 응답 시간 ──
    print("\n── 1. 검색 응답 시간 (ms) ──")
    times = [e.get("search_time_ms", 0) for e in entries if e.get("search_time_ms")]
    if times:
        avg_time = sum(times) / len(times)
        print(f"  평균: {avg_time:.0f}ms")
        print(f"  최소/최대: {min(times)}ms / {max(times)}ms")
        under_500 = sum(1 for t in times if t < 500)
        under_1000 = sum(1 for t in times if t < 1000)
        over_3000 = sum(1 for t in times if t >= 3000)
        print(f"  <500ms: {under_500}/{len(times)} ({under_500/len(times)*100:.0f}%)")
        print(f"  <1000ms: {under_1000}/{len(times)} ({under_1000/len(times)*100:.0f}%)")
        if over_3000:
            print(f"  >3000ms (느림): {over_3000}/{len(times)} ({over_3000/len(times)*100:.0f}%)")
    else:
        print("  시간 데이터 없음 (이전 버전 로그)")

    # ── 2. 유사도 점수 분포 ──
    print("\n── 2. 유사도 점수 분포 (0-100) ──")
    all_scores = []
    for e in entries:
        scores = e.get("relevance_scores", [])
        all_scores.extend(scores)
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        high = sum(1 for s in all_scores if s >= 80)
        mid = sum(1 for s in all_scores if 50 <= s < 80)
        low = sum(1 for s in all_scores if s < 50)
        print(f"  카드 수: {len(all_scores)}장 (평균 {len(all_scores)/total:.1f}장/쿼리)")
        print(f"  평균 유사도: {avg_score:.0f}%")
        print(f"  80%+ (높음): {high}장 ({high/len(all_scores)*100:.0f}%)")
        print(f"  50-79% (보통): {mid}장 ({mid/len(all_scores)*100:.0f}%)")
        print(f"  <50% (낮음): {low}장 ({low/len(all_scores)*100:.0f}%)")
    else:
        print("  유사도 데이터 없음 (이전 버전 로그)")

    # ── 3. 라우팅 분류 ──
    print("\n── 3. 라우팅 분류 ──")
    routes = Counter()
    for e in entries:
        routing = e.get("routing", {})
        route = routing.get("route", "unknown")
        routes[route] += 1
    for route, cnt in routes.most_common():
        print(f"    {route}: {cnt}건 ({cnt/total*100:.0f}%)")

    # ── 4. 문서 적중률 ──
    print("\n── 4. 문서 적중률 ──")
    zero_docs = sum(1 for e in entries if e.get("doc_count", 0) == 0)
    has_docs = total - zero_docs
    doc_counts = [e.get("doc_count", 0) for e in entries]
    avg_docs = sum(doc_counts) / total
    print(f"  검색 결과 있음: {has_docs}/{total} ({has_docs/total*100:.0f}%)")
    print(f"  검색 결과 없음: {zero_docs}/{total} ({zero_docs/total*100:.0f}%)")
    print(f"  평균 문서 수: {avg_docs:.1f}건")

    # ── 5. 자주 나오는 문서 ──
    print("\n── 5. 자주 검색된 문서 TOP 10 ──")
    doc_freq = Counter()
    for e in entries:
        for title in e.get("doc_titles", []):
            doc_freq[title] += 1
    for title, cnt in doc_freq.most_common(10):
        print(f"    {title[:40]}: {cnt}회")

    # ── 6. 세션별 검색 횟수 ──
    print("\n── 6. 세션별 검색 횟수 ──")
    sessions = Counter(e.get("sid", "unknown") for e in entries)
    print(f"  총 세션 수: {len(sessions)}")
    avg_per_session = total / len(sessions) if sessions else 0
    print(f"  세션당 평균 검색: {avg_per_session:.1f}회")

    # ── 개선 지표 요약 ──
    print("\n" + "=" * 70)
    print("  개선 지표 요약")
    print("=" * 70)
    print(f"  {'지표':<30} {'현재':<15} {'목표':<15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    if times:
        print(f"  {'평균 검색 시간':<26} {f'{avg_time:.0f}ms':<15} {'<500ms':<15}")
        print(f"  {'1초 이내 비율':<26} {f'{under_1000/len(times)*100:.0f}%':<15} {'95%+':<15}")
    if all_scores:
        print(f"  {'평균 유사도':<28} {f'{avg_score:.0f}%':<15} {'70%+':<15}")
        print(f"  {'80%+ 카드 비율':<24} {f'{high/len(all_scores)*100:.0f}%':<15} {'50%+':<15}")
    print(f"  {'문서 적중률':<28} {f'{has_docs/total*100:.0f}%':<15} {'95%+':<15}")
    print(f"  {'세션당 평균 검색':<24} {f'{avg_per_session:.1f}회':<15} {'-':<15}")
    print()


def main():
    log_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        all_entries = []
        for f in sorted(os.listdir(log_dir)):
            if f.startswith("rag_") and f.endswith(".jsonl"):
                all_entries.extend(load_logs(os.path.join(log_dir, f)))
        analyze(all_entries)
    elif len(sys.argv) > 1:
        filepath = os.path.join(log_dir, sys.argv[1])
        if not os.path.exists(filepath):
            print(f"파일 없음: {filepath}")
            return
        analyze(load_logs(filepath))
    else:
        today = datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(log_dir, f"rag_{today}.jsonl")
        if not os.path.exists(filepath):
            print(f"오늘 로그 없음: {filepath}")
            print("서버 테스트 후 다시 실행해주세요.")
            return
        analyze(load_logs(filepath))


if __name__ == "__main__":
    main()
