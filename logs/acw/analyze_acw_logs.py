#!/usr/bin/env python3
"""
ACW 로그 분석 스크립트 - 개선 추적용

사용법:
    python analyze_acw_logs.py                    # 오늘 로그 분석
    python analyze_acw_logs.py acw_20260210.jsonl  # 특정 파일 분석
    python analyze_acw_logs.py --all               # 전체 로그 분석

측정 항목:
    1. AI 요약 품질: result_length (이전: 1-2문장 ~50자 → 목표: 200자+), 섹션 구조 유무
    2. category_raw 분류율: 49개 세부 카테고리 중 매칭된 비율
    3. 응답 시간: 병렬 처리 시간
    4. 카테고리 분포: 어떤 카테고리가 가장 많이 나오는지
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime

VALID_CATEGORY_RAW = {
    "가상계좌 안내", "가상계좌 예약/취소", "갱신 수령지 변경", "결제계좌 안내/변경",
    "결제대금 안내", "결제일 안내/변경", "결제일 안내/변경/취소", "교육비",
    "금리인하요구권 안내/신청", "긴급 배송 신청", "단기카드대출 안내/실행",
    "도난/분실 신청/해제", "도시가스", "매출구분 변경", "명세서 재발송",
    "서비스 이용방법 안내", "선결제/즉시출금", "소득공제 확인서/종합소득세 안내",
    "쇼핑케어", "수기환급 처리", "승인취소/매출취소 안내", "신용공여기간 안내",
    "심사 진행사항 안내", "안심클릭/간편페이 안내", "연체대금 안내", "연체대금 즉시출금",
    "연회비 안내", "오토할부/오토캐쉬백 안내/신청/취소", "옵션 서비스 안내/변경",
    "이벤트 안내", "이용내역 안내", "이용방법 안내", "일부결제대금이월약정 등록",
    "일부결제대금이월약정 안내", "일부결제대금이월약정 해지", "입금내역 안내",
    "장기카드대출 실행", "장기카드대출 안내", "전화요금", "정부지원 바우처",
    "증명서/확인서 발급", "청구지 안내/변경", "특별한도/임시한도/일시한도 안내/신청",
    "포인트/마일리지 안내", "포인트/마일리지 전환등록", "프리미엄 바우처 안내/발급",
    "한도 안내", "한도상향 접수/처리", "해외결제 문의",
}


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
    real_calls = [e for e in entries if not e.get("is_simulation")]
    sim_calls = [e for e in entries if e.get("is_simulation")]

    print("=" * 70)
    print(f"  ACW 로그 분석 리포트 ({entries[0]['timestamp'][:10]})")
    print("=" * 70)
    print(f"\n총 {total}건 (실전: {len(real_calls)}, 시뮬레이션: {len(sim_calls)})")

    # ── 1. AI 요약 품질 ──
    print("\n── 1. AI 요약 품질 (result 필드) ──")
    result_lengths = [e["summary"]["result_length"] for e in entries]
    has_sections = sum(1 for e in entries if e["summary"]["result_has_sections"])

    avg_len = sum(result_lengths) / total
    min_len = min(result_lengths)
    max_len = max(result_lengths)

    print(f"  평균 길이: {avg_len:.0f}자  (이전 기준: ~50자)")
    print(f"  최소/최대: {min_len}자 / {max_len}자")
    print(f"  구조화 섹션 포함: {has_sections}/{total} ({has_sections/total*100:.0f}%)")

    # 200자 이상 비율
    over_200 = sum(1 for l in result_lengths if l >= 200)
    print(f"  200자 이상: {over_200}/{total} ({over_200/total*100:.0f}%)")

    # ── 2. category_raw 분류 정확도 ──
    print("\n── 2. category_raw 분류 정확도 ──")
    raw_values = [e["summary"]["category_raw"] for e in entries]
    filled = sum(1 for v in raw_values if v)
    valid = sum(1 for v in raw_values if v in VALID_CATEGORY_RAW)
    invalid = [v for v in raw_values if v and v not in VALID_CATEGORY_RAW]

    print(f"  채워진 비율: {filled}/{total} ({filled/total*100:.0f}%)")
    print(f"  유효한 값: {valid}/{total} ({valid/total*100:.0f}%)")
    if invalid:
        print(f"  유효하지 않은 값: {Counter(invalid).most_common(5)}")

    # ── 3. 카테고리 분포 ──
    print("\n── 3. 카테고리 분포 ──")
    main_dist = Counter(e["summary"]["category_main"] for e in entries)
    raw_dist = Counter(v for v in raw_values if v)
    print("  [대분류 TOP 5]")
    for cat, cnt in main_dist.most_common(5):
        print(f"    {cat}: {cnt}건 ({cnt/total*100:.0f}%)")
    print("  [세부분류 TOP 10]")
    for cat, cnt in raw_dist.most_common(10):
        print(f"    {cat}: {cnt}건")

    # ── 4. 응답 시간 ──
    print("\n── 4. 응답 시간 ──")
    times = [e["parallel_time_sec"] for e in entries]
    avg_time = sum(times) / total
    print(f"  평균: {avg_time:.2f}초")
    print(f"  최소/최대: {min(times):.2f}초 / {max(times):.2f}초")

    # ── 5. 이관 부서 분포 ──
    print("\n── 5. 이관 부서 분포 ──")
    transfer_dist = Counter(e["summary"]["transfer_dep"] for e in entries)
    for dep, cnt in transfer_dist.most_common():
        print(f"    {dep}: {cnt}건")

    # ── 6. 감정 점수 통계 ──
    print("\n── 6. 감정 점수 통계 ──")
    emotion_scores = [e["evaluation"]["emotion_score"] for e in entries if e["evaluation"].get("emotion_score")]
    if emotion_scores:
        print(f"  평균: {sum(emotion_scores)/len(emotion_scores):.1f}")
        print(f"  범위: {min(emotion_scores)} ~ {max(emotion_scores)}")

    # ── 개선 지표 요약 ──
    print("\n" + "=" * 70)
    print("  개선 지표 요약")
    print("=" * 70)
    print(f"  {'지표':<30} {'이전':<15} {'현재':<15} {'목표':<15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'AI 요약 평균 길이':<24} {'~50자':<15} {f'{avg_len:.0f}자':<15} {'200자+':<15}")
    print(f"  {'구조화 섹션 비율':<26} {'0%':<15} {f'{has_sections/total*100:.0f}%':<15} {'90%+':<15}")
    print(f"  {'category_raw 유효 분류율':<22} {'0%':<15} {f'{valid/total*100:.0f}%':<15} {'90%+':<15}")
    print(f"  {'평균 응답 시간':<26} {'-':<15} {f'{avg_time:.2f}초':<15} {'<10초':<15}")
    print()


def main():
    log_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # 전체 로그
        all_entries = []
        for f in sorted(os.listdir(log_dir)):
            if f.startswith("acw_") and f.endswith(".jsonl"):
                all_entries.extend(load_logs(os.path.join(log_dir, f)))
        analyze(all_entries)
    elif len(sys.argv) > 1:
        # 특정 파일
        filepath = os.path.join(log_dir, sys.argv[1])
        if not os.path.exists(filepath):
            print(f"파일 없음: {filepath}")
            return
        analyze(load_logs(filepath))
    else:
        # 오늘 로그
        today = datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(log_dir, f"acw_{today}.jsonl")
        if not os.path.exists(filepath):
            print(f"오늘 로그 없음: {filepath}")
            print("서버 테스트 후 다시 실행해주세요.")
            return
        analyze(load_logs(filepath))


if __name__ == "__main__":
    main()
