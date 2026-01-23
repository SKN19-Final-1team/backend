from typing import Any, Dict, List, Tuple

from app.rag.pipeline import RAGConfig, run_rag

TESTS: List[Dict[str, Any]] = [
    {"id": "T001", "query": "나라사랑 잃어버렸어요", "expect_route": "card_usage", "must_have_doc_ids": ["narasarang_faq_005"], "must_not_have_doc_ids": ["k패스_24", "k패스_30"], "notes": "나라사랑카드 분실 대응 핵심 케이스"},
    {"id": "T002", "query": "나라사랑카드 분실하면 어떻게 해요", "expect_route": "card_usage", "must_have_doc_ids": ["narasarang_faq_005"], "notes": "동의어/조사 변화"},
    {"id": "T003", "query": "해외 여행 중에 카드를 잃어버렸어요", "expect_route": "card_usage", "must_have_doc_ids": ["카드분실_도난_관련피해_예방_및_대응방법_merged"], "must_not_have_doc_ids": ["k패스_24"], "notes": "카드명 없는 분실 시나리오"},
    {"id": "T004", "query": "카드 도난당한 것 같아요", "expect_route": "card_usage", "must_have_doc_ids": ["카드분실_도난_관련피해_예방_및_대응방법_merged"], "notes": "분실/도난 intent 확장 확인"},
    {"id": "T005", "query": "분실 신고 어디서 해요", "expect_route": "card_usage", "must_have_doc_ids": ["카드분실_도난_관련피해_예방_및_대응방법_merged"], "notes": "카드명 없이 행동 중심 질문"},
    {"id": "T006", "query": "k패스 다자녀", "expect_route": "card_info", "must_have_doc_ids": ["k패스_2"], "must_not_have_doc_ids": ["k패스_24", "k패스_27"], "notes": "혜택 핵심 문서 상단 노출"},
    {"id": "T007", "query": "k패스 다자녀 혜택 신청", "expect_route": "card_info", "must_have_doc_ids": ["k패스_2"], "notes": "query 확장"},
    {"id": "T008", "query": "경기도 k패스 혜택", "expect_route": "card_info", "must_have_doc_ids": ["k패스_14"], "notes": "지역 혜택 가이드"},
    {"id": "T009", "query": "k패스 충남 혜택", "expect_route": "card_info", "must_have_doc_ids": ["k패스_13"], "notes": "지역별 분기"},
    {"id": "T010", "query": "k패스 체크카드 혜택", "expect_route": "card_info", "must_have_doc_ids": ["CARD-SHINHAN-K-패스-신한카드-체크"], "notes": "card_tbl 노출 확인"},
    {"id": "T011", "query": "단기카드대출", "expect_route": "card_usage", "must_have_doc_ids": ["카드상품별_거래조건_이자율__수수료_등__merged"], "notes": "현금서비스/단기대출 기본"},
    {"id": "T012", "query": "현금서비스 수수료", "expect_route": "card_usage", "must_have_doc_ids": ["카드상품별_거래조건_이자율__수수료_등__merged"], "notes": "동의어 매핑"},
    {"id": "T013", "query": "단기카드대출 리볼빙 되나요", "expect_route": "card_usage", "must_have_doc_ids": ["sinhan_terms_credit_신용카드_개인회원_약관_040"], "notes": "리볼빙 제외 조항"},
    {"id": "T014", "query": "리볼빙 신청 방법", "expect_route": "card_usage", "must_have_doc_ids": ["sinhan_terms_credit_신용카드_개인회원_약관_039"], "notes": "약관 기반"},
    {"id": "T015", "query": "신용카드 리볼빙 이자", "expect_route": "card_usage", "must_have_doc_ids": ["카드상품별_거래조건_이자율__수수료_등__merged"], "notes": "이자율 설명"},
    {"id": "T016", "query": "애플페이 등록이 안돼요", "expect_route": "card_usage", "notes": "payment intent 인식 확인 (doc id 유연)"},
    {"id": "T017", "query": "삼성페이 결제 오류", "expect_route": "card_usage", "notes": "payment synonyms 동작 확인"},
    {"id": "T018", "query": "카카오페이 카드 연결", "expect_route": "card_usage", "notes": "간편결제 가이드"},
    {"id": "T019", "query": "티머니 교통카드 등록", "expect_route": "card_usage", "notes": "교통카드 결제수단"},
    {"id": "T020", "query": "카드 재발급 어떻게 하나요", "expect_route": "card_usage", "notes": "재발급 기본 시나리오"},
    {"id": "T021", "query": "나라사랑카드 재발급", "expect_route": "card_usage", "must_have_doc_ids": ["narasarang_faq_006"], "notes": "나라사랑 재발급"},
    {"id": "T022", "query": "카드 발급 조건", "expect_route": "card_info", "notes": "발급은 info vs usage 경계 케이스"},
    {"id": "T023", "query": "신용카드 신청 서류", "expect_route": "card_usage", "notes": "신청 + usage"},
    {"id": "T024", "query": "카드 혜택 뭐가 좋아요", "expect_route": "card_info", "notes": "약한 의도 단독"},
    {"id": "T025", "query": "연회비 얼마에요", "expect_route": "card_info", "notes": "혜택/정보성"},
    {"id": "T026", "query": "카드 사용처", "expect_route": "card_usage", "notes": "weak intent routing"},
    {"id": "T027", "query": "카드 결제 안돼요", "expect_route": "card_usage", "notes": "장애/문제 상황"},
    {"id": "T028", "query": "이 카드 괜찮아요?", "expect_route": "card_info", "notes": "모호한 질문 fallback 확인"},
    {"id": "T029", "query": "문의 드립니다", "expect_route": "none", "notes": "검색 차단/안내 처리용"},
    {"id": "T030", "query": "그냥 궁금해서요", "expect_route": "none", "notes": "should_search false 기대"},
        # --- KT 으랏차차 신한카드 (※ 아래 KT_DOC_ID는 실제 적재된 문서 id로 교체 필요) ---
    {
        "id": "T031",
        "query": "KT 으랏차차 신한카드 통신요금 할인 한도 알려줘",
        "expect_route": "card_info",
        "must_have_doc_ids": ["CARD-SHINHAN-KT-으랏차차-신한카드"],
        "notes": "KT 통신요금 월 최대 3만3천원 할인 관련"
    },
    {
        "id": "T032",
        "query": "KT 으랏차차 전월 50만원이면 통신요금 할인 얼마야?",
        "expect_route": "card_info",
        "must_have_doc_ids": ["CARD-SHINHAN-KT-으랏차차-신한카드"],
        "notes": "전월실적 구간(50만원 이상/200만원 이상) 테이블 근거"
    },
    {
        "id": "T033",
        "query": "해외 원화결제(DCC) 차단 어떻게 해요",
        "expect_route": "card_usage",
        "notes": "해외원화결제(DCC) 사전차단 안내 문구 근거"
    },
    {
        "id": "T034",
        "query": "신한카드 고객센터 전화번호 뭐에요",
        "expect_route": "card_usage",
        "notes": "1544-7000 포함 문서 회수 확인(답변 텍스트 검증은 별도 필요)"
    },
    {
        "id": "T035",
        "query": "단기/장기 카드대출 전화번호 알려줘",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["카드대출 예약신청_merged"],
        "notes": "1544-0303 등 번호 포함 문서 회수 확인(답변 텍스트 검증은 별도 필요)"
    },

    # --- 국민행복카드_28 ---
    {
        "id": "T036",
        "query": "국민행복카드 통신료 자동납부 할인 되나요",
        "expect_route": "card_info",
        "must_have_doc_ids": ["CARD-SHINHAN-국민행복(신용_체크)"],
        "notes": "통신료 자동납부(3대 통신사) 혜택 문서"
    },
    {
        "id": "T037",
        "query": "국민행복카드 3대 통신사(SKT,KT,LGU+) 자동납부 혜택 알려줘",
        "expect_route": "card_info",
        "must_have_doc_ids": ["CARD-SHINHAN-국민행복(신용_체크)", "국민행복카드_28"],
        "notes": "SKT/KT/LG U+ 키워드 포함 케이스"
    },

    # --- 서울시다둥이행복카드_13 ---
    {
        "id": "T038",
        "query": "서울시다둥이행복카드 편의점 적립 혜택",
        "expect_route": "card_info",
        "must_have_doc_ids": ["dadungi_013"],
        "notes": "CU/GS25/세븐일레븐 5% 포인트 적립"
    },
    {
        "id": "T039",
        "query": "서울시다둥이행복카드 배달앱 포인트 적립 조건이 뭐야?",
        "expect_route": "card_info",
        "must_have_doc_ids": ["dadungi_013"],
        "notes": "주말 배달앱 건당 2만원 이상 결제 시 1천 포인트"
    },

    {
        "id": "T040",
        "query": "으랏차차",
        "expect_route": "none",
        "notes": "브랜드 단어 단독 입력 시 should_search false 기대(정책에 따라 조정)"
    },
]


def _doc_info(doc: Dict[str, Any]) -> Tuple[str, str, str]:
    return (doc.get("table"), doc.get("id") or doc.get("db_id"), doc.get("title"))


def _doc_id(doc: Dict[str, Any]) -> str:
    return str(doc.get("id") or doc.get("db_id") or "")


def _check(t: Dict[str, Any], res: Dict[str, Any]) -> Tuple[bool, bool, List[str], List[str], List[str]]:
    routing = res.get("routing", {})
    expect = t.get("expect_route")
    if expect == "none":
        route_ok = not routing.get("should_search", True)
    else:
        route_ok = routing.get("route") == expect
    doc_ids = [_doc_id(d) for d in res.get("docs", [])]
    must_have = [i for i in t.get("must_have_doc_ids", []) if i not in doc_ids]
    must_not = [i for i in t.get("must_not_have_doc_ids", []) if i in doc_ids]
    ok = route_ok and not must_have and not must_not
    return ok, route_ok, must_have, must_not, doc_ids


async def run_tests(tests: List[Dict[str, Any]], top_k: int = 4, show_all: bool = False):
    fails: List[str] = []
    for t in tests:
        res = await run_rag(t["query"], config=RAGConfig(top_k=top_k))
        ok, route_ok, must_have, must_not, doc_ids = _check(t, res)
        if show_all or not ok:
            print(f"{t['id']} | route_ok={route_ok} | missing={must_have} | forbidden={must_not}")
            print("docs:", [_doc_info(d) for d in res.get("docs", [])])
        if not ok:
            fails.append(t["id"])
    print(f"done: {len(tests)} total, fail {len(fails)} -> {fails}")
