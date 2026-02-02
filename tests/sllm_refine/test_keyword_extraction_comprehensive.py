"""
키워드 추출 및 텍스트 보정 로직 종합 테스트

실제 카드사 고객센터에서 접수될 법한 모호하고 구어체적인 발화를 기반으로 테스트합니다.
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 테스트 대상 모듈
from app.llm.delivery.keyword_extractor import extract_keywords, ExtractedKeywords
from app.llm.delivery.morphology_analyzer import analyze_morphemes, apply_text_corrections


@dataclass
class TestCase:
    """테스트 케이스"""
    id: str
    category: str  # STT_CORRECTION, CARD_NAME, ACTION, PAYMENT, INTENT, COMPOUND
    input_text: str
    description: str
    expected_correction: Optional[str] = None  # 기대하는 보정 결과
    expected_card_names: List[str] = field(default_factory=list)
    expected_actions: List[str] = field(default_factory=list)
    expected_payments: List[str] = field(default_factory=list)
    expected_intents: List[str] = field(default_factory=list)
    difficulty: str = "normal"  # easy, normal, hard


@dataclass
class TestResult:
    """테스트 결과"""
    test_case: TestCase
    corrected_text: str
    extracted_keywords: ExtractedKeywords
    correction_passed: bool
    card_name_passed: bool
    action_passed: bool
    payment_passed: bool
    intent_passed: bool
    overall_passed: bool
    notes: List[str] = field(default_factory=list)


# =============================================================================
# 테스트 케이스 정의 - 실제 고객 발화 기반 구어체
# =============================================================================

TEST_CASES = [
    # =========================================================================
    # 카테고리 1: STT 발음 오류 보정 테스트
    # =========================================================================
    TestCase(
        id="STT-001",
        category="STT_CORRECTION",
        input_text="연예비가 얼마예요?",
        description="'연회비' STT 발음 오류 - 흔한 동음이의어 오류",
        expected_correction="연회비가 얼마예요?",
        expected_intents=["연회비"],
        difficulty="easy"
    ),
    TestCase(
        id="STT-002",
        category="STT_CORRECTION",
        input_text="바우저 신청하려구요",
        description="'바우처' STT 발음 오류",
        expected_correction="바우처 신청하려구요",
        expected_actions=["신청"],
        difficulty="easy"
    ),
    TestCase(
        id="STT-003",
        category="STT_CORRECTION",
        input_text="삼송페이 등녹이 안돼요",
        description="'삼성페이 등록' 복합 STT 오류",
        expected_correction="삼성페이 등록이 안돼요",
        expected_payments=["삼성페이"],
        expected_actions=["등록", "오류"],
        difficulty="normal"
    ),
    TestCase(
        id="STT-004",
        category="STT_CORRECTION",
        input_text="리벌빙 해지하고싶어요",
        description="'리볼빙' STT 발음 오류",
        expected_correction="리볼빙 해지하고싶어요",
        expected_actions=["해지"],
        difficulty="easy"
    ),
    TestCase(
        id="STT-005",
        category="STT_CORRECTION",
        input_text="하나낸에서 자동이채 걸려있는데요",
        description="'하나은행' + '자동이체' 복합 오류",
        expected_correction="하나은행에서 자동이체 걸려있는데요",
        expected_actions=["자동이체"],
        difficulty="normal"
    ),
    TestCase(
        id="STT-006",
        category="STT_CORRECTION",
        input_text="결재금앱 청구써 확인하려고요",
        description="'결제금액' + '청구서' 복합 오류",
        expected_correction="결제금액 청구서 확인하려고요",
        difficulty="hard"
    ),
    TestCase(
        id="STT-007",
        category="STT_CORRECTION",
        input_text="카드깝 할부로 변경할 수 있나요",
        description="'카드값' + '할부' 복합 오류",
        expected_correction="카드값 할부로 변경할 수 있나요",
        expected_actions=["할부", "변경"],
        difficulty="normal"
    ),
    TestCase(
        id="STT-008",
        category="STT_CORRECTION",
        input_text="이길 영업일에 입금되나요",
        description="'익일' STT 오류 - 금융 전문 용어",
        expected_correction="익일 영업일에 입금되나요",
        expected_actions=["입금"],
        difficulty="normal"
    ),

    # =========================================================================
    # 카테고리 2: 카드상품명 추출 테스트
    # =========================================================================
    TestCase(
        id="CARD-001",
        category="CARD_NAME",
        input_text="그... 나라 뭐시기 카드요, 군인들 쓰는 거",
        description="불완전한 카드명 언급 - 나라사랑카드",
        expected_card_names=["나라사랑카드"],
        difficulty="hard"
    ),
    TestCase(
        id="CARD-002",
        category="CARD_NAME",
        input_text="아이 있으면 뭐 좋은 거 있어요? 플러스 어쩌구 카드",
        description="불완전한 카드명 + 모호한 질문",
        expected_intents=["혜택"],
        difficulty="hard"
    ),
    TestCase(
        id="CARD-003",
        category="CARD_NAME",
        input_text="테디 기본 카드 있잖아요 그거요",
        description="약식 카드명 - 테디 베이직 카드",
        expected_card_names=["테디 베이직 카드"],
        difficulty="normal"
    ),
    TestCase(
        id="CARD-004",
        category="CARD_NAME",
        input_text="국민행복카드 신청하려구요",
        description="정확한 카드명",
        expected_card_names=["국민행복카드"],
        expected_actions=["신청"],
        difficulty="easy"
    ),
    TestCase(
        id="CARD-005",
        category="CARD_NAME",
        input_text="케이패스 등록 어떻게 해요",
        description="K-패스 약칭",
        expected_actions=["등록"],
        difficulty="normal"
    ),
    TestCase(
        id="CARD-006",
        category="CARD_NAME",
        input_text="알뜰카드로 교통 타면 얼마나 할인돼요",
        description="알뜰교통카드 약칭 + 혜택 문의",
        expected_intents=["할인"],
        difficulty="normal"
    ),

    # =========================================================================
    # 카테고리 3: 액션(행위) 추출 테스트
    # =========================================================================
    TestCase(
        id="ACT-001",
        category="ACTION",
        input_text="카드 잃어버렸어요 빨리 막아주세요",
        description="분실 신고 - 긴급 상황 구어체",
        expected_actions=["분실신고", "분실"],
        difficulty="normal"
    ),
    TestCase(
        id="ACT-002",
        category="ACTION",
        input_text="새 카드 하나 더 만들려고 하는데요",
        description="재발급/신규발급 - 간접 표현",
        expected_actions=["발급", "신청"],
        difficulty="normal"
    ),
    TestCase(
        id="ACT-003",
        category="ACTION",
        input_text="그냥 안 쓸래요 없애주세요",
        description="해지 요청 - 간접 표현",
        expected_actions=["해지"],
        difficulty="hard"
    ),
    TestCase(
        id="ACT-004",
        category="ACTION",
        input_text="한도 좀 올려주실 수 있나요 급해서요",
        description="한도상향 - 구어체 요청",
        expected_actions=["한도상향", "한도"],
        difficulty="normal"
    ),
    TestCase(
        id="ACT-005",
        category="ACTION",
        input_text="카드 결제가 왜 자꾸 튕겨요",
        description="결제오류 - 구어체 표현",
        expected_actions=["결제오류", "오류"],
        difficulty="normal"
    ),
    TestCase(
        id="ACT-006",
        category="ACTION",
        input_text="비번 까먹었어요 어떡해요",
        description="비밀번호 재설정 - 구어체",
        expected_actions=["비밀번호", "인증"],
        difficulty="normal"
    ),
    TestCase(
        id="ACT-007",
        category="ACTION",
        input_text="실수로 결제 취소 안 되나요",
        description="취소 요청",
        expected_actions=["취소"],
        difficulty="easy"
    ),
    TestCase(
        id="ACT-008",
        category="ACTION",
        input_text="카드 기간 다 됐는데 어떻게 해요",
        description="갱신/만료 - 간접 표현",
        expected_actions=["갱신", "만료"],
        difficulty="hard"
    ),

    # =========================================================================
    # 카테고리 4: 결제수단 추출 테스트
    # =========================================================================
    TestCase(
        id="PAY-001",
        category="PAYMENT",
        input_text="삼성페이에 카드 넣으려는데요",
        description="삼성페이 등록",
        expected_payments=["삼성페이"],
        expected_actions=["등록"],
        difficulty="easy"
    ),
    TestCase(
        id="PAY-002",
        category="PAYMENT",
        input_text="애플페이가 매장에서 안 먹혀요",
        description="애플페이 결제 오류 - 구어체",
        expected_payments=["애플페이"],
        expected_actions=["오류"],
        difficulty="normal"
    ),
    TestCase(
        id="PAY-003",
        category="PAYMENT",
        input_text="네이버 쓸 때 포인트 쌓여요?",
        description="네이버페이 - 약칭",
        expected_payments=["네이버페이"],
        expected_intents=["포인트", "적립"],
        difficulty="normal"
    ),
    TestCase(
        id="PAY-004",
        category="PAYMENT",
        input_text="폰으로 결제하려면 뭐 깔아야돼요",
        description="간접적 결제수단 문의 - 힌트만 있음",
        expected_actions=["등록"],
        difficulty="hard"
    ),
    TestCase(
        id="PAY-005",
        category="PAYMENT",
        input_text="카카오에서 결제했는데 취소하고싶어요",
        description="카카오페이 - 약칭",
        expected_payments=["카카오페이"],
        expected_actions=["취소"],
        difficulty="normal"
    ),

    # =========================================================================
    # 카테고리 5: 의도(인텐트) 추출 테스트
    # =========================================================================
    TestCase(
        id="INT-001",
        category="INTENT",
        input_text="이 카드 뭐가 좋아요?",
        description="혜택 문의 - 간접 표현",
        expected_intents=["혜택"],
        difficulty="normal"
    ),
    TestCase(
        id="INT-002",
        category="INTENT",
        input_text="1년에 얼마 내야돼요",
        description="연회비 문의 - 간접 표현",
        expected_intents=["연회비"],
        difficulty="hard"
    ),
    TestCase(
        id="INT-003",
        category="INTENT",
        input_text="여기 쓰면 얼마나 깎아줘요",
        description="할인 문의 - 구어체",
        expected_intents=["할인"],
        difficulty="hard"
    ),
    TestCase(
        id="INT-004",
        category="INTENT",
        input_text="돈 모으려면 어떤 카드가 좋아요",
        description="적립/캐시백 - 간접 표현",
        expected_intents=["적립", "캐시백"],
        difficulty="hard"
    ),
    TestCase(
        id="INT-005",
        category="INTENT",
        input_text="마일리지 쌓이는 카드 뭐 있어요",
        description="마일리지 적립 문의",
        expected_intents=["적립"],
        difficulty="normal"
    ),
    TestCase(
        id="INT-006",
        category="INTENT",
        input_text="저한테 맞는 카드 추천해주세요",
        description="카드 추천 요청",
        expected_intents=["추천"],
        difficulty="easy"
    ),

    # =========================================================================
    # 카테고리 6: 복합 시나리오 테스트
    # =========================================================================
    TestCase(
        id="COMP-001",
        category="COMPOUND",
        input_text="나라사람카드 연예비가 얼마예요",
        description="STT 오류 + 카드명 + 의도",
        expected_correction="나라사랑카드 연회비가 얼마예요",
        expected_card_names=["나라사랑카드"],
        expected_intents=["연회비"],
        difficulty="normal"
    ),
    TestCase(
        id="COMP-002",
        category="COMPOUND",
        input_text="삼송페이 등녹했는데 결재가 안돼요",
        description="STT 오류 + 결제수단 + 오류 상황",
        expected_correction="삼성페이 등록했는데 결제가 안돼요",
        expected_payments=["삼성페이"],
        expected_actions=["등록", "결제오류", "오류"],
        difficulty="hard"
    ),
    TestCase(
        id="COMP-003",
        category="COMPOUND",
        input_text="해외에서 긁었는데 갑이 두 번 빠졌어요",
        description="해외결제 + 중복결제 오류 - 구어체",
        expected_actions=["해외결제", "오류"],
        difficulty="hard"
    ),
    TestCase(
        id="COMP-004",
        category="COMPOUND",
        input_text="아 그 뭐냐 카드 막힘 풀어주세요",
        description="불완전한 발화 + 정지해제 요청",
        expected_actions=["정지"],
        difficulty="hard"
    ),
    TestCase(
        id="COMP-005",
        category="COMPOUND",
        input_text="할뿌로 바꿀 수 있어요? 무의자로요",
        description="STT 오류 + 할부 변경 + 무이자",
        expected_correction="할부로 바꿀 수 있어요? 무이자로요",
        expected_actions=["할부", "변경"],
        expected_intents=["무이자"],
        difficulty="normal"
    ),
    TestCase(
        id="COMP-006",
        category="COMPOUND",
        input_text="애플패이 티머니 충전 어떻게해요",
        description="STT 오류 + 결제수단 + 교통 충전",
        expected_correction="애플페이 티머니 충전 어떻게해요",
        expected_payments=["애플페이", "티머니"],
        expected_actions=["충전"],
        difficulty="normal"
    ),
    TestCase(
        id="COMP-007",
        category="COMPOUND",
        input_text="리보빙 쓰면 이짜가 얼마나 붙어요",
        description="STT 오류 + 리볼빙 + 이자 문의",
        expected_correction="리볼빙 쓰면 이자가 얼마나 붙어요",
        expected_intents=["이자"],
        difficulty="normal"
    ),
    TestCase(
        id="COMP-008",
        category="COMPOUND",
        input_text="계자에서 카드갑이 빠지는 날이 언제예요",
        description="STT 오류 + 결제일 문의",
        expected_correction="계좌에서 카드값이 빠지는 날이 언제예요",
        expected_actions=["결제일"],
        difficulty="normal"
    ),

    # =========================================================================
    # 카테고리 7: 극단적 모호성 / 엣지 케이스
    # =========================================================================
    TestCase(
        id="EDGE-001",
        category="EDGE",
        input_text="어... 그거요 그거",
        description="극도로 불완전한 발화",
        difficulty="hard"
    ),
    TestCase(
        id="EDGE-002",
        category="EDGE",
        input_text="카드요",
        description="단어 하나만 있는 경우",
        difficulty="hard"
    ),
    TestCase(
        id="EDGE-003",
        category="EDGE",
        input_text="",
        description="빈 입력",
        difficulty="easy"
    ),
    TestCase(
        id="EDGE-004",
        category="EDGE",
        input_text="ㅋㅋㅋ 아 진짜 뭐지 이게",
        description="비정형 텍스트",
        difficulty="hard"
    ),
    TestCase(
        id="EDGE-005",
        category="EDGE",
        input_text="결제결제결제 안돼요요요",
        description="반복된 단어",
        expected_actions=["결제오류", "오류"],
        difficulty="normal"
    ),
]


def run_test(test_case: TestCase) -> TestResult:
    """단일 테스트 케이스 실행"""
    notes = []

    # 키워드 추출
    result = extract_keywords(test_case.input_text)

    # 1. 텍스트 보정 확인
    correction_passed = True
    if test_case.expected_correction:
        if result.corrected_text != test_case.expected_correction:
            correction_passed = False
            notes.append(f"보정 불일치: 기대={test_case.expected_correction}, 실제={result.corrected_text}")
        else:
            notes.append("보정 성공")

    # 2. 카드명 추출 확인
    card_name_passed = True
    if test_case.expected_card_names:
        found = set(result.card_names)
        expected = set(test_case.expected_card_names)
        if not expected.issubset(found):
            card_name_passed = False
            missing = expected - found
            notes.append(f"카드명 누락: {missing}")
        else:
            notes.append(f"카드명 성공: {found}")

    # 3. 액션 추출 확인
    action_passed = True
    if test_case.expected_actions:
        found = set(result.actions)
        expected = set(test_case.expected_actions)
        # 최소 1개 이상 매칭되면 부분 성공
        if not expected.intersection(found):
            action_passed = False
            notes.append(f"액션 누락: 기대={expected}, 실제={found}")
        else:
            matched = expected.intersection(found)
            notes.append(f"액션 매칭: {matched}")

    # 4. 결제수단 추출 확인
    payment_passed = True
    if test_case.expected_payments:
        found = set(result.payments)
        expected = set(test_case.expected_payments)
        if not expected.issubset(found):
            payment_passed = False
            missing = expected - found
            notes.append(f"결제수단 누락: {missing}")
        else:
            notes.append(f"결제수단 성공: {found}")

    # 5. 의도 추출 확인
    intent_passed = True
    if test_case.expected_intents:
        found = set(result.intents)
        expected = set(test_case.expected_intents)
        # 최소 1개 이상 매칭되면 부분 성공
        if not expected.intersection(found):
            intent_passed = False
            notes.append(f"의도 누락: 기대={expected}, 실제={found}")
        else:
            matched = expected.intersection(found)
            notes.append(f"의도 매칭: {matched}")

    # 전체 패스 여부
    overall_passed = all([
        correction_passed,
        card_name_passed,
        action_passed,
        payment_passed,
        intent_passed
    ])

    return TestResult(
        test_case=test_case,
        corrected_text=result.corrected_text,
        extracted_keywords=result,
        correction_passed=correction_passed,
        card_name_passed=card_name_passed,
        action_passed=action_passed,
        payment_passed=payment_passed,
        intent_passed=intent_passed,
        overall_passed=overall_passed,
        notes=notes
    )


def run_all_tests() -> List[TestResult]:
    """모든 테스트 실행"""
    results = []
    for tc in TEST_CASES:
        result = run_test(tc)
        results.append(result)
    return results


def generate_report(results: List[TestResult]) -> str:
    """Markdown 형식 테스트 보고서 생성"""

    # 통계 계산
    total = len(results)
    passed = sum(1 for r in results if r.overall_passed)
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0

    # 카테고리별 통계
    category_stats = {}
    for r in results:
        cat = r.test_case.category
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0}
        category_stats[cat]["total"] += 1
        if r.overall_passed:
            category_stats[cat]["passed"] += 1

    # 난이도별 통계
    difficulty_stats = {"easy": {"total": 0, "passed": 0},
                        "normal": {"total": 0, "passed": 0},
                        "hard": {"total": 0, "passed": 0}}
    for r in results:
        diff = r.test_case.difficulty
        difficulty_stats[diff]["total"] += 1
        if r.overall_passed:
            difficulty_stats[diff]["passed"] += 1

    # 보고서 작성
    report = []
    report.append("# 키워드 추출 및 텍스트 보정 테스트 보고서")
    report.append("")
    report.append(f"**테스트 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 요약
    report.append("## 1. 테스트 요약")
    report.append("")
    report.append("| 항목 | 값 |")
    report.append("|------|-----|")
    report.append(f"| 총 테스트 케이스 | {total} |")
    report.append(f"| 통과 | {passed} |")
    report.append(f"| 실패 | {failed} |")
    report.append(f"| **통과율** | **{pass_rate:.1f}%** |")
    report.append("")

    # 카테고리별 결과
    report.append("## 2. 카테고리별 결과")
    report.append("")
    report.append("| 카테고리 | 총 케이스 | 통과 | 통과율 |")
    report.append("|----------|-----------|------|--------|")
    for cat, stats in sorted(category_stats.items()):
        rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        report.append(f"| {cat} | {stats['total']} | {stats['passed']} | {rate:.1f}% |")
    report.append("")

    # 난이도별 결과
    report.append("## 3. 난이도별 결과")
    report.append("")
    report.append("| 난이도 | 총 케이스 | 통과 | 통과율 |")
    report.append("|--------|-----------|------|--------|")
    for diff in ["easy", "normal", "hard"]:
        stats = difficulty_stats[diff]
        rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        diff_ko = {"easy": "쉬움", "normal": "보통", "hard": "어려움"}[diff]
        report.append(f"| {diff_ko} | {stats['total']} | {stats['passed']} | {rate:.1f}% |")
    report.append("")

    # 세부 결과
    report.append("## 4. 세부 테스트 결과")
    report.append("")

    for cat in sorted(set(r.test_case.category for r in results)):
        report.append(f"### {cat}")
        report.append("")

        cat_results = [r for r in results if r.test_case.category == cat]
        for r in cat_results:
            tc = r.test_case
            status = "✅ PASS" if r.overall_passed else "❌ FAIL"

            report.append(f"#### {tc.id}: {status}")
            report.append("")
            report.append(f"- **설명**: {tc.description}")
            report.append(f"- **난이도**: {tc.difficulty}")
            report.append(f"- **입력**: `{tc.input_text}`")
            report.append(f"- **보정 결과**: `{r.corrected_text}`")
            report.append("")

            report.append("**추출 결과:**")
            report.append(f"- 카드명: {r.extracted_keywords.card_names or '(없음)'}")
            report.append(f"- 액션: {r.extracted_keywords.actions or '(없음)'}")
            report.append(f"- 결제수단: {r.extracted_keywords.payments or '(없음)'}")
            report.append(f"- 의도: {r.extracted_keywords.intents or '(없음)'}")
            report.append(f"- 명사: {r.extracted_keywords.nouns or '(없음)'}")
            report.append("")

            if tc.expected_correction or tc.expected_card_names or tc.expected_actions or tc.expected_payments or tc.expected_intents:
                report.append("**기대 결과:**")
                if tc.expected_correction:
                    report.append(f"- 보정: `{tc.expected_correction}`")
                if tc.expected_card_names:
                    report.append(f"- 카드명: {tc.expected_card_names}")
                if tc.expected_actions:
                    report.append(f"- 액션: {tc.expected_actions}")
                if tc.expected_payments:
                    report.append(f"- 결제수단: {tc.expected_payments}")
                if tc.expected_intents:
                    report.append(f"- 의도: {tc.expected_intents}")
                report.append("")

            report.append("**검증 상세:**")
            for note in r.notes:
                report.append(f"- {note}")
            report.append("")
            report.append("---")
            report.append("")

    # 실패한 케이스 요약
    failed_cases = [r for r in results if not r.overall_passed]
    if failed_cases:
        report.append("## 5. 실패 케이스 요약")
        report.append("")
        report.append("| ID | 카테고리 | 설명 | 주요 실패 원인 |")
        report.append("|----|----------|------|----------------|")
        for r in failed_cases:
            tc = r.test_case
            causes = []
            if not r.correction_passed:
                causes.append("보정")
            if not r.card_name_passed:
                causes.append("카드명")
            if not r.action_passed:
                causes.append("액션")
            if not r.payment_passed:
                causes.append("결제수단")
            if not r.intent_passed:
                causes.append("의도")
            report.append(f"| {tc.id} | {tc.category} | {tc.description} | {', '.join(causes)} |")
        report.append("")

    # 개선 제안
    report.append("## 6. 개선 제안")
    report.append("")

    # 보정 실패 분석
    correction_failures = [r for r in results if not r.correction_passed and r.test_case.expected_correction]
    if correction_failures:
        report.append("### 6.1 텍스트 보정 개선")
        report.append("")
        report.append("다음 STT 오류 패턴을 `keywords_dict_refine.json`에 추가 검토:")
        report.append("")
        for r in correction_failures:
            report.append(f"- `{r.test_case.input_text}` → `{r.test_case.expected_correction}`")
        report.append("")

    # 키워드 추출 실패 분석
    action_failures = [r for r in results if not r.action_passed and r.test_case.expected_actions]
    if action_failures:
        report.append("### 6.2 액션 키워드 개선")
        report.append("")
        report.append("다음 구어체 표현에 대한 액션 매핑 검토:")
        report.append("")
        for r in action_failures:
            report.append(f"- `{r.test_case.input_text}` - 기대 액션: {r.test_case.expected_actions}")
        report.append("")

    intent_failures = [r for r in results if not r.intent_passed and r.test_case.expected_intents]
    if intent_failures:
        report.append("### 6.3 의도 추출 개선")
        report.append("")
        report.append("다음 간접 표현에 대한 의도 매핑 검토:")
        report.append("")
        for r in intent_failures:
            report.append(f"- `{r.test_case.input_text}` - 기대 의도: {r.test_case.expected_intents}")
        report.append("")

    report.append("---")
    report.append("")
    report.append("*이 보고서는 자동 생성되었습니다.*")

    return "\n".join(report)


if __name__ == "__main__":
    print("=" * 70)
    print("키워드 추출 및 텍스트 보정 테스트 시작")
    print("=" * 70)
    print()

    # 테스트 실행
    results = run_all_tests()

    # 간단한 콘솔 출력
    passed = sum(1 for r in results if r.overall_passed)
    total = len(results)

    print(f"테스트 완료: {passed}/{total} 통과 ({passed/total*100:.1f}%)")
    print()

    # 실패한 케이스 출력
    failed = [r for r in results if not r.overall_passed]
    if failed:
        print("실패한 케이스:")
        for r in failed:
            print(f"  - {r.test_case.id}: {r.test_case.description}")
            for note in r.notes:
                if "누락" in note or "불일치" in note:
                    print(f"      {note}")

    # 보고서 생성
    report = generate_report(results)

    # 파일로 저장
    report_path = Path(__file__).resolve().parent.parent / "Test Report.md"
    report_path.write_text(report, encoding="utf-8")
    print()
    print(f"보고서 저장: {report_path}")
