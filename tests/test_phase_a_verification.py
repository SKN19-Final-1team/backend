"""
Phase A 검증 테스트: STT 할루시네이션 필터 + Vocab Gate weak_intent 확장

실행:
  cd backend
  python -m tests.test_phase_a_verification

결과:
  - 할루시네이션 필터 테스트 (before/after 비교)
  - Vocab Gate 통과율 테스트 (before/after 비교)
  - 전체 개선 요약 출력
"""
import sys
import os
import re

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ══════════════════════════════════════════════════════════════
# 1. STT 할루시네이션 필터 검증
# ══════════════════════════════════════════════════════════════

# 이전 필터 (12개 키워드만, 반복/길이 체크 없음)
OLD_HALLUCINATION_KEYWORDS = [
    "시청해주셔서", "시청해 주셔서", "구독과 좋아요",
    "재택 플러스", "MBC", "뉴스", "투데이", "먹방",
    "영상편집", "영상", "편집", "진심으로"
]


def old_is_hallucination(text: str) -> bool:
    """이전 버전 필터 (prompt 없이 12개 키워드만)"""
    if not text or not text.strip():
        return True
    text = text.strip()
    if any(kw in text for kw in OLD_HALLUCINATION_KEYWORDS):
        return True
    return False


# 새 필터 import
from app.audio.whisper import is_hallucination as new_is_hallucination


# 테스트 케이스: (텍스트, 기대_할루시네이션_여부, 설명)
HALLUCINATION_TEST_CASES = [
    # 할루시네이션이어야 하는 것들 (True = 할루시네이션)
    ("", True, "빈 문자열"),
    (".", True, "1자 미만"),
    ("네", True, "1자 (최소 길이 미달)"),
    ("시청해주셔서 감사합니다", True, "YouTube 패턴"),
    ("구독과 좋아요 부탁드립니다", True, "YouTube 패턴"),
    ("MBC 뉴스입니다", True, "방송 패턴"),
    ("오늘도 맛있게 드세요", True, "일반 할루시네이션"),
    ("감사합니다 여러분 다음에 봐요", True, "일반 할루시네이션"),
    ("자막 제공 네이버", True, "자막 패턴"),
    ("다음 시간에 만나요", True, "방송 마무리 패턴"),
    ("아아아아아아아아아아", True, "반복 패턴"),
    ("네네네네네네", True, "반복 패턴"),
    ("하하하하하하하하", True, "반복 패턴"),

    # 정상 발화 (False = 유효한 전사)
    ("카드 분실했어요", False, "정상: 분실 신고"),
    ("잔액 확인 좀 해주세요", False, "정상: 잔액 조회"),
    ("네 알겠습니다", False, "정상: 짧은 응답"),
    ("연회비가 얼마예요?", False, "정상: 연회비 문의"),
    ("리볼빙 해지하고 싶어요", False, "정상: 해지 요청"),
    ("포인트 사용처가 어디예요?", False, "정상: 포인트 문의"),
    ("결제가 안 돼요", False, "정상: 결제 오류"),
    ("한도 좀 올려주세요", False, "정상: 한도 변경"),
]


def test_hallucination_filter():
    """할루시네이션 필터 개선 전/후 비교"""
    print("=" * 60)
    print("1. STT 할루시네이션 필터 검증")
    print("=" * 60)

    old_correct = 0
    new_correct = 0
    total = len(HALLUCINATION_TEST_CASES)
    improvements = []

    for text, expected, desc in HALLUCINATION_TEST_CASES:
        old_result = old_is_hallucination(text)
        new_result = new_is_hallucination(text)

        old_ok = old_result == expected
        new_ok = new_result == expected

        if old_ok:
            old_correct += 1
        if new_ok:
            new_correct += 1

        status = ""
        if not old_ok and new_ok:
            status = " ★ 개선"
            improvements.append(f"  {desc}: '{text[:30]}...' → 이제 {'차단' if expected else '통과'}됨")
        elif old_ok and not new_ok:
            status = " ⚠ 퇴보"
        elif not old_ok and not new_ok:
            status = " ✗ 둘 다 실패"

        marker = "✓" if new_ok else "✗"
        print(f"  [{marker}] {desc:<25} | old={'차단' if old_result else '통과':<4} new={'차단' if new_result else '통과':<4} expect={'차단' if expected else '통과':<4}{status}")

    print(f"\n  이전 정확도: {old_correct}/{total} ({old_correct/total*100:.0f}%)")
    print(f"  현재 정확도: {new_correct}/{total} ({new_correct/total*100:.0f}%)")
    print(f"  개선 항목: {len(improvements)}건")
    for imp in improvements:
        print(imp)

    return old_correct, new_correct, total


# ══════════════════════════════════════════════════════════════
# 2. Vocab Gate (weak_intent) 통과율 검증
# ══════════════════════════════════════════════════════════════

# 이전 weak_intent (5개만)
OLD_WEAK_INTENTS = {"혜택", "발급", "신청", "사용", "사용처"}

# 이전 버전 vocab match (간소화)
def old_has_vocab_match(query: str) -> bool:
    """이전 버전: weak_intent 5개만"""
    if not query or not query.strip():
        return False
    normalized = query.strip().lower()

    # ACTION_SYNONYMS에서 강한 액션 체크
    from app.rag.vocab.keyword_dict import ACTION_SYNONYMS
    for canon, terms in ACTION_SYNONYMS.items():
        for term in [canon, *terms]:
            if isinstance(term, str) and term.lower() in normalized:
                return True

    # PAYMENT_SYNONYMS 체크
    from app.rag.vocab.keyword_dict import PAYMENT_SYNONYMS
    for canon, terms in PAYMENT_SYNONYMS.items():
        for term in [canon, *terms]:
            if isinstance(term, str) and term.lower() in normalized:
                return True

    # 이전 weak_intent (5개만)
    for intent in OLD_WEAK_INTENTS:
        if intent in normalized:
            return True

    return False


# 새 버전 import
from app.rag.router.signals import has_vocab_match as new_has_vocab_match


# 테스트 케이스: (쿼리, 기대_통과_여부, 설명)
VOCAB_GATE_TEST_CASES = [
    # 반드시 통과해야 하는 것 (True)
    ("카드 분실했어요", True, "분실 = action"),
    ("잔액 확인 좀요", True, "확인 = new weak_intent"),
    ("리볼빙 해지하고 싶어요", True, "리볼빙+해지 = action"),
    ("왜 결제가 안 돼요?", True, "결제 = new weak_intent"),
    ("한도 좀 올려주세요", True, "한도 = new weak_intent"),
    ("포인트 얼마 있어요?", True, "포인트→혜택 = weak_intent"),
    ("Pay 신한카드 혜택", True, "카드명 + 혜택 = strong"),
    ("납부일이 언제예요?", True, "납부 = new weak_intent"),
    ("카드 해지 방법 알려주세요", True, "해지 = new weak_intent"),
    ("환불 처리 가능한가요?", True, "환불 = new weak_intent"),
    ("충전 방법이 어떻게 되나요?", True, "충전 = new weak_intent"),
    ("등록 절차 알려주세요", True, "등록 = new weak_intent"),
    ("조회가 안 되는데요", True, "조회 = new weak_intent"),
    ("상담 연결해주세요", True, "상담 = new weak_intent"),
    ("변경하고 싶어요", True, "변경 = new weak_intent"),
    ("이체 수수료 얼마예요?", True, "이체 = new weak_intent"),
    ("취소하고 싶습니다", True, "취소 = new weak_intent"),
    ("연회비가 얼마인가요?", True, "연회비 = action 관련"),

    # 통과해도 되고 안 해도 되는 경계 케이스
    ("이거 어떻게 써요?", None, "경계: 의도 불명확"),
    ("그냥 좀 알아보려고요", None, "경계: 약한 의도"),
    ("잘 안 되는데요", None, "경계: 문제 신고 암시"),
]


def test_vocab_gate():
    """Vocab Gate 통과율 개선 전/후 비교"""
    print("\n" + "=" * 60)
    print("2. Vocab Gate (weak_intent) 통과율 검증")
    print("=" * 60)

    old_pass = 0
    new_pass = 0
    old_correct = 0
    new_correct = 0
    evaluated = 0
    improvements = []

    for query, expected, desc in VOCAB_GATE_TEST_CASES:
        old_result = old_has_vocab_match(query)
        new_result = new_has_vocab_match(query)

        if expected is not None:
            evaluated += 1
            if old_result == expected:
                old_correct += 1
            if new_result == expected:
                new_correct += 1

        if old_result:
            old_pass += 1
        if new_result:
            new_pass += 1

        status = ""
        if expected is not None and not old_result and new_result and expected:
            status = " ★ 개선 (이전 차단→통과)"
            improvements.append(f"  \"{query}\" ({desc})")

        old_mark = "통과" if old_result else "차단"
        new_mark = "통과" if new_result else "차단"
        exp_mark = "통과" if expected else ("차단" if expected is False else "경계")
        marker = "✓" if (expected is None or new_result == expected) else "✗"
        print(f"  [{marker}] {desc:<25} | old={old_mark:<4} new={new_mark:<4} expect={exp_mark:<4}{status}")

    total = len(VOCAB_GATE_TEST_CASES)
    print(f"\n  전체 통과율: old={old_pass}/{total} ({old_pass/total*100:.0f}%) → new={new_pass}/{total} ({new_pass/total*100:.0f}%)")
    print(f"  평가 대상 정확도: old={old_correct}/{evaluated} ({old_correct/evaluated*100:.0f}%) → new={new_correct}/{evaluated} ({new_correct/evaluated*100:.0f}%)")
    print(f"  개선 항목: {len(improvements)}건")
    for imp in improvements:
        print(imp)

    return old_pass, new_pass, total, old_correct, new_correct, evaluated


# ══════════════════════════════════════════════════════════════
# 3. 전체 요약
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "━" * 60)
    print("  Phase A 검증 테스트 (STT 필터 + Vocab Gate)")
    print("━" * 60 + "\n")

    h_old, h_new, h_total = test_hallucination_filter()
    v_old_pass, v_new_pass, v_total, v_old_acc, v_new_acc, v_eval = test_vocab_gate()

    print("\n" + "━" * 60)
    print("  전체 개선 요약")
    print("━" * 60)
    print(f"\n  [STT 할루시네이션 필터]")
    print(f"    정확도: {h_old}/{h_total} ({h_old/h_total*100:.0f}%) → {h_new}/{h_total} ({h_new/h_total*100:.0f}%)")
    print(f"    변경사항:")
    print(f"      - WHISPER_PROMPT 적용 (Whisper API에 컨텍스트 제공)")
    print(f"      - 할루시네이션 키워드 12개 → 22개")
    print(f"      - 반복 패턴 감지 추가")
    print(f"      - 최소 길이(2자) 체크 추가")

    print(f"\n  [Vocab Gate]")
    print(f"    통과율: {v_old_pass}/{v_total} ({v_old_pass/v_total*100:.0f}%) → {v_new_pass}/{v_total} ({v_new_pass/v_total*100:.0f}%)")
    print(f"    정확도: {v_old_acc}/{v_eval} ({v_old_acc/v_eval*100:.0f}%) → {v_new_acc}/{v_eval} ({v_new_acc/v_eval*100:.0f}%)")
    print(f"    변경사항:")
    print(f"      - weak_intent 5개 → 20개")
    print(f"      - '안내'를 STOPWORDS에서 제거")

    all_pass = (h_new == h_total) and (v_new_acc == v_eval)
    print(f"\n  전체 결과: {'✅ PASS' if all_pass else '⚠ 일부 실패 있음'}")
    print("━" * 60 + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
