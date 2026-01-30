from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import re

from app.rag.pipeline import RAGConfig, run_rag


# ----------------------------
# Utilities
# ----------------------------

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _doc_id(doc: Dict[str, Any]) -> str:
    """
    run_rag()가 반환하는 docs의 키가 환경마다 조금씩 다를 수 있어서
    id 후보 키를 넓게 잡아 안전하게 추출합니다.
    """
    for k in ("id", "db_id", "doc_id", "source_id", "document_id"):
        v = doc.get(k)
        if v:
            return str(v)
    return ""


def _doc_table(doc: Dict[str, Any]) -> str:
    for k in ("table", "source", "tbl", "collection"):
        v = doc.get(k)
        if v:
            return str(v)
    return ""


def _doc_title(doc: Dict[str, Any]) -> str:
    for k in ("title", "doc_title", "name"):
        v = doc.get(k)
        if v:
            return str(v)
    return ""


def _doc_info(doc: Dict[str, Any]) -> Tuple[str, str, str]:
    return (_doc_table(doc), _doc_id(doc), _doc_title(doc))


def _extract_answer_text(res: Dict[str, Any]) -> str:
    """
    run_rag 반환 payload에서 '답변 텍스트'가 들어갈 수 있는 키를 폭넓게 탐색합니다.
    (프로젝트마다 answer/guidanceScript/guidance_script/message/text 등이 다를 수 있음)
    """
    # 1) 가장 흔한 키
    for k in ("answer", "output", "text", "message", "final", "response"):
        v = res.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # 2) 카드 응답 구조가 있는 경우
    for k in ("guidanceScript", "guidance_script", "guidance", "script"):
        v = res.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # 3) currentSituation / nextStep (card_info path)
    def _list_to_text(items: Any) -> str:
        if not isinstance(items, list):
            return ""
        parts: List[str] = []
        for it in items:
            if isinstance(it, str) and it.strip():
                parts.append(it)
                continue
            if isinstance(it, dict):
                for key in ("text", "content", "message", "title"):
                    v = it.get(key)
                    if isinstance(v, str) and v.strip():
                        parts.append(v)
                        break
        return "\n".join(parts)

    cs_text = _list_to_text(res.get("currentSituation"))
    ns_text = _list_to_text(res.get("nextStep"))
    if cs_text or ns_text:
        return "\n".join([t for t in (cs_text, ns_text) if t])

    # 4) cards 배열이 있는 경우
    cards = res.get("cards")
    if isinstance(cards, list) and cards:
        parts: List[str] = []
        for c in cards:
            if not isinstance(c, dict):
                continue
            for ck in ("content", "text", "answer", "guidanceScript", "guidance_script"):
                cv = c.get(ck)
                if isinstance(cv, str) and cv.strip():
                    parts.append(cv)
        if parts:
            return "\n".join(parts)

    return ""


def _contains_any(text: str, terms: List[str]) -> bool:
    t = (text or "").lower()
    return any((term or "").lower() in t for term in terms)


# ----------------------------
# Test Set (강화 버전)
# - 기존 필드 유지
# - 답변 텍스트 검증 추가:
#   must_have_answer_terms / must_not_have_answer_terms
# - 힌트 필터가 docs에서 안 잡힐 때를 대비하여,
#   "명백 오답" 방지용 must_not을 적극 사용
# ----------------------------

TESTS: List[Dict[str, Any]] = [
    {
        "id": "T001",
        "query": "나라사랑 잃어버렸어요",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["narasarang_faq_005"],
        "must_not_have_doc_ids": ["k패스_24", "k패스_30"],
        "must_not_have_answer_terms": ["애플페이", "apple pay", "k-패스", "k패스"],
        "notes": "나라사랑카드 분실 대응 핵심 케이스",
    },
    {
        "id": "T002",
        "query": "나라사랑카드 분실하면 어떻게 해요",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["narasarang_faq_005"],
        "must_not_have_answer_terms": ["애플페이", "apple pay", "k-패스", "k패스"],
        "notes": "동의어/조사 변화",
    },
    {
        "id": "T003",
        "query": "해외 여행 중에 카드를 잃어버렸어요",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["카드분실_도난_관련피해_예방_및_대응방법_merged"],
        "must_not_have_doc_ids": ["k패스_24"],
        "must_not_have_answer_terms": ["애플페이", "apple pay"],
        "notes": "카드명 없는 분실 시나리오",
    },
    {
        "id": "T004",
        "query": "카드 도난당한 것 같아요",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["카드분실_도난_관련피해_예방_및_대응방법_merged"],
        "must_not_have_answer_terms": ["애플페이", "apple pay"],
        "notes": "분실/도난 intent 확장 확인",
    },
    {
        "id": "T005",
        "query": "분실 신고 어디서 해요",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["카드분실_도난_관련피해_예방_및_대응방법_merged"],
        "must_not_have_answer_terms": ["애플페이", "apple pay"],
        "notes": "카드명 없이 행동 중심 질문",
    },

    # --- K-패스 ---
    {
        "id": "T006",
        "query": "k패스 다자녀",
        "expect_route": "card_info",
        "must_have_doc_ids": ["k패스_2"],
        "must_not_have_doc_ids": ["k패스_24", "k패스_27"],
        "must_have_answer_terms": ["다자녀"],
        "must_not_have_answer_terms": ["애플페이", "apple pay"],
        "notes": "혜택 핵심 문서 상단 노출",
    },
    {
        "id": "T007",
        "query": "k패스 다자녀 혜택 신청",
        "expect_route": "card_info",
        "must_have_doc_ids": ["k패스_2"],
        "must_have_answer_terms": ["다자녀"],
        "must_not_have_answer_terms": ["애플페이", "apple pay"],
        "notes": "query 확장",
    },
    {
        "id": "T008",
        "query": "경기도 k패스 혜택",
        "expect_route": "card_info",
        "must_have_doc_ids": ["k패스_14"],
        "must_have_answer_terms": ["경기", "경기도"],
        "must_not_have_answer_terms": ["다자녀 혜택이 시행됩니다"],  # 다자녀 고정문구 방지(가성비)
        "notes": "지역 혜택 가이드",
    },
    {
        "id": "T009",
        "query": "k패스 충남 혜택",
        "expect_route": "card_info",
        "must_have_doc_ids": ["k패스_13"],
        "must_have_answer_terms": ["충남"],
        "must_not_have_answer_terms": ["다자녀 혜택이 시행됩니다"],
        "notes": "지역별 분기",
    },
    {
        "id": "T010",
        "query": "k패스 체크카드 혜택",
        "expect_route": "card_info",
        "must_have_doc_ids": ["CARD-SHINHAN-K-패스-신한카드-체크"],
        "must_have_answer_terms": ["체크"],
        "must_not_have_answer_terms": ["다자녀 혜택이 시행됩니다"],
        "notes": "card_tbl 노출 확인",
    },

    # --- 대출/리볼빙 ---
    {
        "id": "T011",
        "query": "단기카드대출",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["카드상품별_거래조건_이자율__수수료_등__merged"],
        "notes": "현금서비스/단기대출 기본",
    },
    {
        "id": "T012",
        "query": "현금서비스 수수료",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["카드상품별_거래조건_이자율__수수료_등__merged"],
        "notes": "동의어 매핑",
    },
    {
        "id": "T013",
        "query": "단기카드대출 리볼빙 되나요",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["sinhan_terms_credit_신용카드_개인회원_약관_040"],
        "notes": "리볼빙 제외 조항",
    },
    {
        "id": "T014",
        "query": "리볼빙 신청 방법",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["sinhan_terms_credit_신용카드_개인회원_약관_039"],
        "notes": "약관 기반",
    },
    {
        "id": "T015",
        "query": "신용카드 리볼빙 이자",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["카드상품별_거래조건_이자율__수수료_등__merged"],
        "notes": "이자율 설명",
    },

    # --- 간편결제 ---
    {
        "id": "T016",
        "query": "애플페이 등록이 안돼요",
        "expect_route": "card_usage",
        # 애플페이에서 K-패스 절차로 새는 걸 강하게 잡기
        "must_not_have_doc_ids": ["k패스_24", "k패스_5", "k패스_30"],
        "must_have_answer_terms": ["애플페이"],
        "must_not_have_answer_terms": ["k-패스", "k패스", "모바일 티머니"],
        "notes": "payment intent 인식 확인 (doc id 유연)",
    },
    {
        "id": "T017",
        "query": "삼성페이 결제 오류",
        "expect_route": "card_usage",
        "must_have_answer_terms": ["삼성페이"],
        "must_not_have_answer_terms": ["신용도 관리방법", "신용카드_관련주요_용어"],
        "notes": "payment synonyms 동작 확인",
    },
    {
        "id": "T018",
        "query": "카카오페이 카드 연결",
        "expect_route": "card_usage",
        "must_have_answer_terms": ["카카오페이"],
        "must_not_have_answer_terms": ["분실", "도난", "재발급"],  # 연결 문의가 분실로 새는 것 방지
        "notes": "간편결제 가이드",
    },

    # --- 교통 ---
    {
        "id": "T019",
        "query": "티머니 교통카드 등록",
        "expect_route": "card_usage",
        "must_have_answer_terms": ["티머니", "교통"],
        "notes": "교통카드 결제수단",
    },

    # --- 재발급 ---
    {
        "id": "T020",
        "query": "카드 재발급 어떻게 하나요",
        "expect_route": "card_usage",
        # 여기서 '테디카드 1577-6000' 같은 오염이 나오면 잡고 싶다면 아래를 켜도 됨
        "must_not_have_answer_terms": ["테디카드 고객센터", "1577-6000"],
        "notes": "재발급 기본 시나리오",
    },
    {
        "id": "T021",
        "query": "나라사랑카드 재발급",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["narasarang_faq_006"],
        "must_have_answer_terms": ["재발급"],
        "notes": "나라사랑 재발급",
    },

    # --- 카드 정보성 ---
    {
        "id": "T022",
        "query": "카드 발급 조건",
        "expect_route": "card_info",
        "must_have_answer_terms": ["신청"],
        "notes": "발급은 info vs usage 경계 케이스",
    },
    {
        "id": "T023",
        "query": "신용카드 신청 서류",
        "expect_route": "card_usage",
        "must_have_answer_terms": ["서류", "신청"],
        "notes": "신청 + usage",
    },
    {
        "id": "T024",
        "query": "카드 혜택 뭐가 좋아요",
        "expect_route": "card_info",
        "must_have_answer_terms": ["혜택"],
        "notes": "약한 의도 단독",
    },
    {
        "id": "T025",
        "query": "연회비 얼마에요",
        "expect_route": "card_info",
        "must_have_answer_terms": ["연회비"],
        "notes": "혜택/정보성",
    },
    {
        "id": "T026",
        "query": "카드 사용처",
        "expect_route": "card_usage",
        "notes": "weak intent routing",
    },
    {
        "id": "T027",
        "query": "카드 결제 안돼요",
        "expect_route": "card_usage",
        "must_not_have_answer_terms": ["할부항변권"],  # 결제오류가 할부항변권으로 튀는 케이스 방지
        "notes": "장애/문제 상황",
    },
    {
        "id": "T028",
        "query": "이 카드 괜찮아요?",
        "expect_route": "card_info",
        "must_not_have_answer_terms": ["분실", "도난", "재발급"],
        "notes": "모호한 질문 fallback 확인",
    },
    {
        "id": "T029",
        "query": "문의 드립니다",
        "expect_route": "none",
        "notes": "검색 차단/안내 처리용",
    },
    {
        "id": "T030",
        "query": "그냥 궁금해서요",
        "expect_route": "none",
        "notes": "should_search false 기대",
    },

    # --- KT 으랏차차 신한카드 ---
    {
        "id": "T031",
        "query": "KT 으랏차차 신한카드 통신요금 할인 한도 알려줘",
        "expect_route": "card_info",
        "must_have_doc_ids": ["CARD-SHINHAN-KT-으랏차차-신한카드"],
        "must_have_answer_terms": ["통신", "할인", "한도"],
        "must_not_have_answer_terms": ["신용정보 알림서비스 이용 수수료"],
        "notes": "KT 통신요금 월 최대 3만3천원 할인 관련",
    },
    {
        "id": "T032",
        "query": "KT 으랏차차 전월 50만원이면 통신요금 할인 얼마야?",
        "expect_route": "card_info",
        "must_have_doc_ids": ["CARD-SHINHAN-KT-으랏차차-신한카드"],
        "must_have_answer_terms": ["전월", "실적", "할인"],
        "must_not_have_answer_terms": ["전월실적별 통합할인한도 쇼핑"],  # 국민행복/다른 테이블로 새는 경우 방지
        "notes": "전월실적 구간(50만원 이상/200만원 이상) 테이블 근거",
    },

    # --- DCC / 전화번호 ---
    {
        "id": "T033",
        "query": "해외 원화결제(DCC) 차단 어떻게 해요",
        "expect_route": "card_usage",
        "must_have_answer_terms": ["dcc", "해외 원화결제", "원화결제"],
        "must_not_have_answer_terms": ["apple pay", "애플페이"],
        "notes": "해외원화결제(DCC) 사전차단 안내 문구 근거",
    },
    {
        "id": "T034",
        "query": "신한카드 고객센터 전화번호 뭐에요",
        "expect_route": "card_usage",
        # 답변 텍스트 검증이 핵심
        "must_have_answer_terms": ["1544-7000"],
        "must_not_have_answer_terms": ["신용정보 알림서비스 이용 수수료"],
        "notes": "1544-7000 포함 문서 회수 확인(답변 텍스트 검증)",
    },
    {
        "id": "T035",
        "query": "단기/장기 카드대출 전화번호 알려줘",
        "expect_route": "card_usage",
        "must_have_doc_ids": ["카드대출 예약신청_merged"],
        "must_have_answer_terms": ["전화", "번호"],
        "must_not_have_answer_terms": ["신용정보 알림서비스 이용 수수료"],
        "notes": "번호 포함 문서 회수 확인(답변 텍스트 검증)",
    },

    # --- 국민행복카드 ---
    {
        "id": "T036",
        "query": "국민행복카드 통신료 자동납부 할인 되나요",
        "expect_route": "card_info",
        "must_have_doc_ids": ["CARD-SHINHAN-국민행복(신용_체크)"],
        "must_have_answer_terms": ["통신", "자동납부"],
        "notes": "통신료 자동납부(3대 통신사) 혜택 문서",
    },
    {
        "id": "T037",
        "query": "국민행복카드 3대 통신사(SKT,KT,LGU+) 자동납부 혜택 알려줘",
        "expect_route": "card_info",
        "must_have_doc_ids": ["CARD-SHINHAN-국민행복(신용_체크)", "국민행복카드_28"],
        "must_have_answer_terms": ["통신사", "자동납부"],
        "notes": "SKT/KT/LG U+ 키워드 포함 케이스",
    },

    # --- 서울시다둥이행복카드 ---
    {
        "id": "T038",
        "query": "서울시다둥이행복카드 편의점 적립 혜택",
        "expect_route": "card_info",
        "must_have_doc_ids": ["dadungi_013"],
        "must_have_answer_terms": ["CU", "GS25", "세븐", "5%"],
        "notes": "CU/GS25/세븐일레븐 5% 포인트 적립",
    },
    {
        "id": "T039",
        "query": "서울시다둥이행복카드 배달앱 포인트 적립 조건이 뭐야?",
        "expect_route": "card_info",
        "must_have_doc_ids": ["dadungi_013"],
        "must_have_answer_terms": ["배달", "2만원", "1천"],
        "notes": "주말 배달앱 건당 2만원 이상 결제 시 1천 포인트",
    },

    {
        "id": "T040",
        "query": "으랏차차",
        "expect_route": "none",
        "notes": "브랜드 단어 단독 입력 시 should_search false 기대(정책에 따라 조정)",
    },
]


def split_test_sets(tests: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    guide_test: 상담/가이드 문장 품질 검증용(card_usage 중심)
    rag_test: 일반 RAG 검색/라우팅 검증용(card_info 및 기타)
    """
    guide_tests: List[Dict[str, Any]] = []
    rag_tests: List[Dict[str, Any]] = []
    for t in tests:
        if t.get("expect_route") == "card_usage":
            guide_tests.append(t)
        else:
            rag_tests.append(t)
    return guide_tests, rag_tests


GUIDE_TESTS, RAG_TESTS = split_test_sets(TESTS)


# ----------------------------
# Checker
# ----------------------------

def _check(t: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, Any]:
    routing = res.get("routing", {}) or {}
    expect = t.get("expect_route")

    if expect == "none":
        route_ok = not routing.get("should_search", True)
    else:
        route_ok = routing.get("route") == expect

    docs = res.get("docs", []) or []
    doc_ids = [_doc_id(d) for d in docs]

    missing_docs = [i for i in t.get("must_have_doc_ids", []) if i not in doc_ids]
    forbidden_docs = [i for i in t.get("must_not_have_doc_ids", []) if i in doc_ids]

    answer = _normalize_ws(_extract_answer_text(res))

    must_have_terms = t.get("must_have_answer_terms", []) or []
    must_not_terms = t.get("must_not_have_answer_terms", []) or []

    missing_terms = [term for term in must_have_terms if term and term.lower() not in answer.lower()]
    forbidden_terms = [term for term in must_not_terms if term and term.lower() in answer.lower()]

    ok = route_ok and not missing_docs and not forbidden_docs and not missing_terms and not forbidden_terms

    return {
        "ok": ok,
        "route_ok": route_ok,
        "missing_docs": missing_docs,
        "forbidden_docs": forbidden_docs,
        "missing_terms": missing_terms,
        "forbidden_terms": forbidden_terms,
        "doc_ids": doc_ids,
        "answer_preview": (answer[:220] + ("..." if len(answer) > 220 else "")),
        "route": routing.get("route"),
        "should_search": routing.get("should_search"),
    }


async def run_tests(
    tests: List[Dict[str, Any]] = TESTS,
    top_k: int = 4,
    show_all: bool = False,
    show_answer: bool = True,
    enable_consult_search: Optional[bool] = None,
):
    fails: List[str] = []

    for t in tests:
        cfg_kwargs = {"top_k": top_k}
        if enable_consult_search is not None:
            cfg_kwargs["enable_consult_search"] = enable_consult_search
        res = await run_rag(t["query"], config=RAGConfig(**cfg_kwargs))
        chk = _check(t, res)

        if show_all or not chk["ok"]:
            print(
                f"{t['id']} | ok={chk['ok']} | route_ok={chk['route_ok']} | "
                f"route={chk['route']} | should_search={chk['should_search']}"
            )
            if chk["missing_docs"] or chk["forbidden_docs"]:
                print(f"  docs: missing={chk['missing_docs']} forbidden={chk['forbidden_docs']}")
            if chk["missing_terms"] or chk["forbidden_terms"]:
                print(f"  terms: missing={chk['missing_terms']} forbidden={chk['forbidden_terms']}")
            print("  docs:", [_doc_info(d) for d in (res.get("docs", []) or [])])
            if show_answer:
                print("  answer:", chk["answer_preview"])
            print("-" * 80)

        if not chk["ok"]:
            fails.append(t["id"])

    print(f"done: {len(tests)} total, fail {len(fails)} -> {fails}")
    return len(fails) == 0


async def run_guide_tests(
    top_k: int = 4,
    show_all: bool = False,
    show_answer: bool = True,
):
    return await run_tests(
        GUIDE_TESTS,
        top_k=top_k,
        show_all=show_all,
        show_answer=show_answer,
        enable_consult_search=True,
    )


async def run_rag_tests(
    top_k: int = 4,
    show_all: bool = False,
    show_answer: bool = True,
):
    return await run_tests(
        RAG_TESTS,
        top_k=top_k,
        show_all=show_all,
        show_answer=show_answer,
        enable_consult_search=False,
    )
