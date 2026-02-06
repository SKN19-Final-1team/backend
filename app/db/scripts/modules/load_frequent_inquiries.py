"""
자주 찾는 문의 데이터 적재 모듈

frequent_inquiries 테이블에 실제 상담 데이터 분석 결과 적재
consultations 테이블의 category_raw를 분석하여 자주 찾는 문의 생성
"""

from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, RealDictCursor

from . import BATCH_SIZE, check_table_has_data


# 카테고리별 상세 콘텐츠 및 관련 문서 매핑
CATEGORY_CONTENT_MAP = {
    "청구문의/청구서": {
        "keyword": "청구서 확인",
        "question": "청구서를 확인하고 싶어요.",
        "content": """청구서 확인 및 청구 관련 안내입니다.

1. 청구서 확인 방법
   - 앱에서 '청구서' 메뉴에서 확인 가능
   - 이메일/SMS로 발송된 청구서 확인
   - 고객센터(1588-1234) 전화 문의

2. 청구 금액 이상 시
   - 거래 내역과 청구 금액 대조
   - 이의 제기는 청구일로부터 14일 이내
   - 고객센터에서 상세 내역 안내

3. 청구서 재발송
   - 앱에서 재발송 요청 가능
   - 이메일/우편 선택 가능""",
        "related_document_id": "billing-1-1-1",
        "related_document_title": "청구서 확인 안내",
        "related_document_regulation": "여신전문금융업법 제15조",
        "related_document_summary": "청구서 확인 방법과 청구 금액 이의 제기 절차를 안내합니다."
    },
    "이용내역 안내": {
        "keyword": "이용내역",
        "question": "카드 이용내역을 확인하고 싶어요.",
        "content": """카드 이용내역 조회 방법을 안내드립니다.

1. 실시간 이용내역 조회
   - 앱에서 '이용내역' 메뉴 선택
   - 최근 3개월 이용내역 즉시 조회
   - 카테고리별/기간별 필터링 가능

2. 이용내역 상세 정보
   - 가맹점명, 결제일시, 금액
   - 할부 정보 (할부 개월, 잔여 금액)
   - 포인트 적립/사용 내역

3. 이용내역 알림 설정
   - 결제 즉시 SMS/푸시 알림
   - 일일/주간/월간 이용 리포트
   - 특정 금액 이상 거래 알림""",
        "related_document_id": "usage-1-1-1",
        "related_document_title": "이용내역 조회 서비스",
        "related_document_regulation": "여신전문금융업법 제14조",
        "related_document_summary": "실시간 이용내역 조회 및 알림 서비스를 안내합니다."
    },
    "한도관련 문의/처리": {
        "keyword": "한도 조회/변경",
        "question": "신용한도를 확인하거나 변경하고 싶어요.",
        "content": """신용한도 조회 및 변경 방법을 안내드립니다.

1. 한도 조회
   - 앱에서 '한도 관리' 메뉴 선택
   - 총 한도, 사용 한도, 가용 한도 확인
   - 일시불/할부 한도 별도 확인

2. 한도 증액 신청
   - 앱에서 '한도 증액 신청' 선택
   - 실시간 자동 심사 (영업일 09:00~18:00)
   - 심사 결과 즉시 알림

3. 임시 한도 증액
   - 일시적 증액 (최대 7일)
   - 앱에서 즉시 신청 가능
   - 증액 한도: 기존 한도의 30%까지""",
        "related_document_id": "limit-1-1-1",
        "related_document_title": "신용한도 관리",
        "related_document_regulation": "여신전문금융업법 제6조",
        "related_document_summary": "신용한도 조회, 증액 신청, 임시 증액 서비스를 안내합니다."
    },
    "분실/도난 신청/처리": {
        "keyword": "카드 분실",
        "question": "카드를 분실했어요. 어떻게 해야 하나요?",
        "content": """카드 분실 시 즉시 처리 방법을 안내드립니다.

1. 즉시 카드 정지
   - 고객센터(1588-1234) 또는 앱에서 즉시 정지
   - 정지 처리 즉시 완료, SMS 확인

2. 재발급 신청
   - 카드 정지 후 재발급 신청 가능
   - 재발급 수수료 면제
   - 3-5 영업일 내 배송

3. 부정 사용 보상
   - 분실 신고 후 부정 사용 전액 보상
   - 신고 전 72시간 내 거래 보험 처리 가능""",
        "related_document_id": "card-1-1-1",
        "related_document_title": "카드 분실 신고",
        "related_document_regulation": "여신전문금융업법 제16조",
        "related_document_summary": "카드 분실 시 즉시 정지 및 재발급 절차를 안내합니다."
    },
    "결제방식 안내": {
        "keyword": "결제방식 변경",
        "question": "결제 방식을 변경하고 싶어요.",
        "content": """결제 방식 변경 안내입니다.

1. 일시불/할부 변경
   - 결제 후 할부 전환 가능 (결제일 전까지)
   - 앱에서 '할부 전환' 메뉴 선택
   - 2~12개월 할부 선택

2. 자동이체 설정
   - 결제 계좌 등록/변경
   - 결제일 변경 (매월 1일, 15일, 25일)
   - 자동이체 실패 시 재출금

3. 결제 수단 변경
   - 계좌 자동이체 ↔ 가상계좌 입금
   - 결제 계좌 변경 시 1-2일 소요""",
        "related_document_id": "payment-1-1-1",
        "related_document_title": "결제방식 안내",
        "related_document_regulation": "여신전문금융업법 제17조",
        "related_document_summary": "결제 방식 변경 및 자동이체 설정 방법을 안내합니다."
    }
}

# 기본 콘텐츠 (매핑에 없는 카테고리용)
DEFAULT_CONTENT_TEMPLATE = {
    "content": """해당 문의에 대한 안내입니다.

자세한 내용은 고객센터(1588-1234)로 문의하시거나
앱에서 관련 메뉴를 확인해 주세요.

- 운영시간: 평일 09:00~18:00
- 앱: 24시간 이용 가능""",
    "related_document_id": "general-1-1-1",
    "related_document_title": "일반 문의 안내",
    "related_document_regulation": "",
    "related_document_summary": "일반 문의 사항에 대한 안내입니다."
}


# 하드코딩 폴백 데이터 (consultations 데이터가 없을 때 사용)
FALLBACK_INQUIRIES_DATA = [
    {
        "id": 1,
        "keyword": "카드 분실",
        "question": "카드를 분실했어요. 어떻게 해야 하나요?",
        "count": 45,
        "trend": "up",
        "content": """카드 분실 시 즉시 처리 방법을 안내드립니다.

1. 즉시 카드 정지
   - 고객센터(1588-1234) 또는 앱을 통해 즉시 카드 사용을 정지할 수 있습니다.
   - 정지 처리는 즉시 완료되며, SMS로 확인 메시지를 발송해드립니다.

2. 재발급 신청
   - 카드 정지 후 재발급 신청이 가능합니다.
   - 재발급 수수료는 면제됩니다.
   - 신청 후 3-5 영업일 내 등록된 주소로 배송됩니다.

3. 부정 사용 보상
   - 분실 신고 접수 후 발생한 부정 사용은 전액 보상됩니다.
   - 신고 접수 이전 72시간 내 발생한 거래는 보험 처리로 보상 가능합니다.
   - 고액(100만원 초과)의 경우 경찰서 분실 신고 확인서가 필요합니다.

4. 주의사항
   - 법인카드는 담당자 승인이 필요합니다.
   - 가족카드는 주카드 회원의 동의가 필요합니다.
   - 해외 분실 시 긴급 카드 발급 서비스(수수료 $30)를 이용하실 수 있습니다.""",
        "related_document_id": "card-1-1-1",
        "related_document_title": "카드 즉시 사용 정지",
        "related_document_regulation": "여신전문금융업법 제16조",
        "related_document_summary": "고객의 카드 분실 신고 시 즉시 카드 사용을 정지하여 부정 사용을 방지합니다."
    },
    {
        "id": 2,
        "keyword": "해외 결제",
        "question": "해외에서 카드가 안 됩니다.",
        "count": 38,
        "trend": "up",
        "content": """해외에서 카드 사용이 안 되는 경우 원인과 해결 방법을 안내드립니다.

1. 해외 사용 차단 상태 확인
   - 보안을 위해 일부 카드는 해외 사용이 차단되어 있을 수 있습니다.
   - 고객센터 또는 앱을 통해 즉시 해제 가능합니다.
   - 해제 처리는 실시간으로 반영됩니다.

2. 일일 한도 초과
   - 해외 사용 일일 한도를 초과했을 가능성이 있습니다.
   - 앱에서 한도 조회 및 임시 증액이 가능합니다.
   - 증액 신청 후 즉시 반영됩니다.

3. 특정 국가 제한
   - 일부 국가는 금융 제재 또는 보안상 이유로 사용이 제한될 수 있습니다.
   - 해당 국가 확인 후 대체 결제 수단을 안내드립니다.

4. 가맹점 문제
   - 가맹점의 카드 단말기 또는 네트워크 문제일 수 있습니다.
   - 다른 가맹점에서 시도하거나 현금 인출 후 결제를 권장합니다.

5. 긴급 지원
   - 해외에서 카드 분실 시 긴급 카드 발급 서비스를 이용하실 수 있습니다.
   - 주요 공항에 테디카드 라운지가 있어 즉시 발급 가능합니다.""",
        "related_document_id": "card-2-1-1",
        "related_document_title": "해외 결제 차단 해제",
        "related_document_regulation": "외국환거래법 제3조",
        "related_document_summary": "해외 결제 차단 상태를 즉시 해제하고, 해외 사용 설정을 활성화합니다."
    },
    {
        "id": 3,
        "keyword": "포인트 적립",
        "question": "포인트가 적립 안 됐어요.",
        "count": 32,
        "trend": "same",
        "content": """포인트 적립이 되지 않은 경우 확인 사항과 해결 방법을 안내드립니다.

1. 적립 제외 업종 확인
   - 일부 업종(보험료, 세금, 공과금 등)은 포인트 적립이 제외됩니다.
   - 카드 약관에서 제외 업종 목록을 확인하실 수 있습니다.

2. 적립 반영 시간
   - 포인트는 거래 승인 후 1~3 영업일 내 적립됩니다.
   - 해외 거래의 경우 최대 7영업일이 소요될 수 있습니다.

3. 최소 적립 금액
   - 건당 1,000원 이상 거래부터 포인트가 적립됩니다.
   - 1,000원 미만 거래는 적립 대상이 아닙니다.

4. 실적 미달
   - 월 최소 이용 금액(30만원)을 충족해야 포인트가 적립됩니다.
   - 실적 미달 시 해당 월 포인트는 적립되지 않습니다.

5. 적립 누락 시 처리
   - 적립이 누락된 경우 고객센터로 문의하시면 확인 후 수동 적립해드립니다.
   - 거래 영수증과 카드 번호가 필요합니다.
   - 처리 기간은 3~5 영업일입니다.""",
        "related_document_id": "card-1-2-1",
        "related_document_title": "포인트 적립 정책",
        "related_document_regulation": "여신전문금융업법 제19조",
        "related_document_summary": "포인트 적립 조건, 제외 업종, 적립 시기를 안내합니다."
    },
    {
        "id": 4,
        "keyword": "연회비 환불",
        "question": "연회비 환불 받을 수 있나요?",
        "count": 28,
        "trend": "down",
        "content": """연회비 환불 조건과 신청 방법을 안내드립니다.

1. 환불 가능 조건
   - 카드 발급 후 1개월 이내 해지 시 전액 환불
   - 실적 조건 미충족으로 연회비가 부과된 경우, 부과 후 1개월 이내 해지 시 환불
   - 카드사 귀책 사유로 서비스 미제공 시 부분 환불

2. 환불 불가 조건
   - 카드 사용 실적이 있는 경우
   - 연회비 부과 후 1개월 경과
   - 회원의 귀책 사유로 서비스 미이용

3. 환불 신청 방법
   - 고객센터(1588-1234) 전화 신청
   - 앱 또는 홈페이지에서 온라인 신청
   - 신청 후 3~5 영업일 내 처리

4. 환불 금액
   - 사용 기간에 따라 일할 계산하여 환불
   - 환불 수수료는 없음
   - 환불 금액은 다음 달 청구액에서 차감

5. 연회비 면제 조건
   - 전년도 실적 조건 충족 시 다음 연도 연회비 면제
   - 실적 기준: 월 30만원 이상, 연 360만원 이상
   - 실적은 전년도 1월~12월 기준으로 산정""",
        "related_document_id": "card-3-1-1",
        "related_document_title": "연회비 환불 정책",
        "related_document_regulation": "여신전문금융업법 제18조",
        "related_document_summary": "연회비 환불 조건, 신청 방법, 면제 조건을 상세히 안내합니다."
    },
    {
        "id": 5,
        "keyword": "한도 증액",
        "question": "신용한도를 올리고 싶어요.",
        "count": 25,
        "trend": "up",
        "content": """신용한도 증액 신청 방법과 심사 기준을 안내드립니다.

1. 한도 증액 신청 방법
   - 앱에서 '한도 증액 신청' 메뉴 선택
   - 고객센터(1588-1234) 전화 신청
   - 신청 후 실시간 자동 심사

2. 심사 기준
   - 카드 사용 실적 (최근 6개월)
   - 연체 이력 확인 (24개월)
   - 신용평가 점수
   - 소득 대비 부채 비율

3. 즉시 증액 가능 조건
   - 최근 6개월 이상 정상 사용
   - 연체 이력 없음
   - 신용평가 등급 양호
   - 월 평균 이용액 30만원 이상

4. 증액 한도
   - 기본: 현재 한도의 30%까지 증액 가능
   - 우수 고객: 현재 한도의 50%까지 증액 가능
   - 최대 한도: 신용평가 등급에 따라 상이

5. 심사 기간
   - 자동 심사: 실시간 (영업일 기준 09:00~18:00)
   - 수동 심사: 1~2 영업일
   - 추가 서류 필요 시: 3~5 영업일

6. 주의사항
   - 증액 신청은 월 1회로 제한
   - 증액 거부 시 3개월 후 재신청 가능
   - 과도한 증액 신청은 신용평가에 영향을 줄 수 있음""",
        "related_document_id": "card-4-1-1",
        "related_document_title": "신용한도 증액 심사",
        "related_document_regulation": "여신전문금융업법 제6조",
        "related_document_summary": "신용한도 증액 신청 방법, 심사 기준, 즉시 증액 조건을 안내합니다."
    }
]


def analyze_consultations_for_inquiries(conn: psycopg2_connection, limit: int = 10, days: int = 14):
    """
    실제 상담 데이터(consultations)를 분석하여 자주 찾는 문의 목록 생성

    Args:
        conn: DB 연결
        limit: 상위 N개 문의 추출 (기본 10)
        days: 최근 N일 데이터 분석 (기본 14일 = 2주)

    Returns:
        list: 자주 찾는 문의 데이터 리스트
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # 최근 N일 기준 category_raw 상위 문의 유형 추출
        # + 이번 주 vs 저번 주 비교로 trend 계산
        cursor.execute("""
            WITH recent_data AS (
                SELECT
                    category_raw,
                    call_date,
                    -- 이번 주 (최근 7일)
                    CASE WHEN call_date >= CURRENT_DATE - INTERVAL '7 days' THEN 1 ELSE 0 END as this_week,
                    -- 저번 주 (8~14일 전)
                    CASE WHEN call_date >= CURRENT_DATE - INTERVAL '14 days'
                              AND call_date < CURRENT_DATE - INTERVAL '7 days' THEN 1 ELSE 0 END as last_week
                FROM consultations
                WHERE category_raw IS NOT NULL
                  AND category_raw != ''
                  AND call_date >= CURRENT_DATE - INTERVAL '%s days'
            )
            SELECT
                category_raw,
                COUNT(*) as cnt,
                SUM(this_week) as this_week_cnt,
                SUM(last_week) as last_week_cnt,
                CASE
                    WHEN SUM(this_week) > SUM(last_week) THEN 'up'
                    WHEN SUM(this_week) < SUM(last_week) THEN 'down'
                    ELSE 'same'
                END as trend
            FROM recent_data
            GROUP BY category_raw
            ORDER BY cnt DESC
            LIMIT %s
        """, (days, limit))

        top_categories = cursor.fetchall()

        if not top_categories:
            return None

        inquiries = []
        for idx, row in enumerate(top_categories, start=1):
            category = row['category_raw']
            count = row['cnt']
            trend = row.get('trend', 'same')  # SQL에서 계산된 trend 사용
            this_week = row.get('this_week_cnt', 0)
            last_week = row.get('last_week_cnt', 0)

            # 카테고리에 맞는 콘텐츠 매핑 찾기
            content_data = CATEGORY_CONTENT_MAP.get(category)

            if content_data:
                keyword = content_data["keyword"]
                question = content_data["question"]
                content = content_data["content"]
                doc_id = content_data["related_document_id"]
                doc_title = content_data["related_document_title"]
                doc_regulation = content_data["related_document_regulation"]
                doc_summary = content_data["related_document_summary"]
            else:
                # 매핑이 없으면 카테고리명 기반으로 생성
                keyword = category.split('/')[0] if '/' in category else category
                keyword = keyword[:20]  # 최대 20자
                question = f"{keyword} 관련 문의입니다."
                content = DEFAULT_CONTENT_TEMPLATE["content"]
                doc_id = DEFAULT_CONTENT_TEMPLATE["related_document_id"]
                doc_title = DEFAULT_CONTENT_TEMPLATE["related_document_title"]
                doc_regulation = DEFAULT_CONTENT_TEMPLATE["related_document_regulation"]
                doc_summary = DEFAULT_CONTENT_TEMPLATE["related_document_summary"]

            inquiries.append({
                "id": idx,
                "keyword": keyword,
                "question": question,
                "count": count,
                "trend": trend,  # 이번 주 vs 저번 주 비교 결과
                "this_week_cnt": this_week,
                "last_week_cnt": last_week,
                "content": content,
                "related_document_id": doc_id,
                "related_document_title": doc_title,
                "related_document_regulation": doc_regulation,
                "related_document_summary": doc_summary
            })

        return inquiries

    except Exception as e:
        print(f"[ERROR] 상담 데이터 분석 실패: {e}")
        return None
    finally:
        cursor.close()


def load_frequent_inquiries_data(conn: psycopg2_connection, force_reload: bool = False):
    """
    자주 찾는 문의 데이터 적재

    1차: consultations 테이블 분석하여 실제 데이터 기반 생성
    2차: 상담 데이터가 없으면 폴백 데이터 사용

    Args:
        conn: DB 연결
        force_reload: True면 기존 데이터 삭제 후 재적재
    """
    print("\n" + "=" * 60)
    print("[LOAD] 자주 찾는 문의 데이터 적재")
    print("=" * 60)

    # 이미 데이터가 있는지 확인
    has_data, count = check_table_has_data(conn, "frequent_inquiries")
    if has_data and not force_reload:
        print(f"[INFO] frequent_inquiries 테이블에 이미 데이터가 있습니다. (건수: {count}건) - 적재 스킵")
        return True

    cursor = conn.cursor()

    try:
        # 기존 데이터 삭제 (force_reload 또는 신규)
        if force_reload and has_data:
            cursor.execute("DELETE FROM frequent_inquiries")
            print("[INFO] 기존 데이터 삭제 완료")

        # 1차: 실제 상담 데이터 분석
        print("[INFO] 실제 상담 데이터 분석 중...")
        inquiries_data = analyze_consultations_for_inquiries(conn, limit=10)

        if inquiries_data:
            print(f"[INFO] 상담 데이터 기반 {len(inquiries_data)}건 생성")
            data_source = "실제 상담 데이터 분석"
        else:
            # 2차: 폴백 데이터 사용
            print("[WARN] 상담 데이터가 없어 폴백 데이터 사용")
            inquiries_data = FALLBACK_INQUIRIES_DATA
            data_source = "폴백 데이터"

        print(f"[INFO] 자주 찾는 문의 데이터 적재 중... ({len(inquiries_data)}건, 출처: {data_source})")

        insert_query = """
            INSERT INTO frequent_inquiries (
                id, keyword, question, count, trend, content,
                related_document_id, related_document_title,
                related_document_regulation, related_document_summary,
                is_active, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true, NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                keyword = EXCLUDED.keyword,
                question = EXCLUDED.question,
                count = EXCLUDED.count,
                trend = EXCLUDED.trend,
                content = EXCLUDED.content,
                related_document_id = EXCLUDED.related_document_id,
                related_document_title = EXCLUDED.related_document_title,
                related_document_regulation = EXCLUDED.related_document_regulation,
                related_document_summary = EXCLUDED.related_document_summary,
                updated_at = NOW()
        """

        batch_data = [
            (
                item["id"],
                item["keyword"],
                item["question"],
                item["count"],
                item["trend"],
                item["content"],
                item["related_document_id"],
                item["related_document_title"],
                item["related_document_regulation"],
                item["related_document_summary"]
            )
            for item in inquiries_data
        ]

        execute_batch(cursor, insert_query, batch_data, page_size=BATCH_SIZE)
        conn.commit()

        print(f"[INFO] 자주 찾는 문의 적재 완료: {len(batch_data)}건")

        # 적재 결과 출력
        print("[INFO] 적재된 자주 찾는 문의 (최근 2주 기준):")
        for item in inquiries_data[:5]:
            this_week = item.get('this_week_cnt', '-')
            last_week = item.get('last_week_cnt', '-')
            trend_arrow = {'up': '↑', 'down': '↓', 'same': '→'}.get(item['trend'], '→')
            print(f"  - {item['keyword']}: {item['count']}건 (이번주:{this_week}, 저번주:{last_week}) {trend_arrow}")
        if len(inquiries_data) > 5:
            print(f"  ... 외 {len(inquiries_data) - 5}건")

        return True

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] 자주 찾는 문의 데이터 적재 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cursor.close()
