
# 카드 / 정부지원 프로그램 명칭 동의어
# canonical: 내부 표준 명칭
# values: 사용자 발화에서 인식할 수 있는 다양한 표현
CARD_NAME_SYNONYMS = {
    "K-패스": ["K-패스", "K패스", "k패스", "케이패스", "케이 패스"],
    "국민행복카드": ["국민행복카드", "국민 행복카드", "행복카드"],
    "나라사랑카드": ["나라사랑", "나라사랑카드", "나라사랑 카드"],
    "시니어카드": ["시니어", "시니어카드", "시니어 카드"],
    "서울시다둥이행복카드": [
        "서울시다둥이행복카드",
        "서울시 다둥이 행복카드",
        "서울시 다둥이카드",
        "서울시 다둥이 카드",
        "다둥이카드",
        "다둥이 카드",
        "다둥이",
        "다자녀",
        "다자녀카드",
        "다자녀 카드",
        "다둥이 행복카드",
    ],
    "네이버페이카드": ["네이버페이카드", "네이버페이 카드", "네이버카드", "네이버 카드"],
    "쿠팡와우카드": ["쿠팡와우카드", "쿠팡 와우 카드", "쿠팡와우", "쿠팡 와우", "와우카드"],
    "민생회복 소비쿠폰": [
        "민생회복 소비쿠폰",
        "민생회복쿠폰",
        "민생 회복 소비쿠폰",
        "민생회복 쿠폰",
        "민생 회복 쿠폰",
        "민생회복 소비 쿠폰",
        "민생 회복 소비 쿠폰",
        "민생회복",
        "민생 회복",
    ],
}

# 명확한 사용자 행동 의도
ACTION_SYNONYMS = {
    "분실": [
        "분실",
        "분실했",
        "분실했어",
        "분실했어요",
        "도난",
        "잃어버",
        "잃어버렸",
        "잃어버렸어요",
        "없어졌",
        "못 찾",
        "못찾",
        "못 찾겠",
        "못찾겠",
        "두고 왔",
        "두고왔",
    ],
    "재발급": ["재발급", "다시 발급", "새로 발급", "재발행"],
    "해지": ["해지", "없애", "없앨래"],
    "결제취소": ["결제취소", "결제 취소", "승인취소", "승인 취소"],
    "한도": ["한도", "limit", "얼마까지", "상향", "증액"],
    "연회비": ["연회비", "연 회비"],
    "교통": ["교통", "교통카드", "교통 카드", "후불 교통", "후불교통"],
    "적립": ["적립", "포인트 적립"],
    "청구할인": ["청구할인", "청구 할인"],
}

# 단독으로는 의미가 약한 일반 의도
WEAK_INTENT_SYNONYMS = {
    "발급": ["발급", "발급 방법", "발급방법"],
    "신청": ["신청", "신청 방법", "신청방법"],
    "사용": ["사용", "사용 방법", "사용방법", "사용법"],
    "사용처": ["사용처", "사용 가능한 곳", "어디서 사용"],
    "혜택": ["혜택"],
}

# 결제 수단 관련 키워드
PAYMENT_SYNONYMS = {
    "애플페이": ["애플페이", "애플 페이", "apple pay", "applepay"],
    "삼성페이": ["삼성페이", "삼성 페이", "samsung pay"],
    "네이버페이": ["네이버페이", "네이버 페이", "naver pay"],
    "카카오페이": ["카카오페이", "카카오 페이", "kakao pay"],
    "티머니": ["티머니", "티 머니", "t머니", "t-money", "tmoney"],
}

# 라우팅 힌트용
ROUTE_CARD_INFO = "card_info"
ROUTE_CARD_USAGE = "card_usage"

# ACTION 중에서도 즉시 대응 가능한 intent만 허용
ACTION_ALLOWLIST = {"분실"}
PAYMENT_ALLOWLIST = set(PAYMENT_SYNONYMS.keys())

# 약한 의도 단독 등장 시 기본 라우팅 힌트
WEAK_INTENT_ROUTE_HINTS = {
    "혜택": ROUTE_CARD_INFO,
    "발급": ROUTE_CARD_USAGE,
    "신청": ROUTE_CARD_USAGE,
    "사용": ROUTE_CARD_USAGE,
    "사용처": ROUTE_CARD_USAGE,
}

STOPWORDS = {"n/a", "na", "none", "문의", "안내"}

# vocab.json 생성을 위한 그룹 정의
# - type: 키워드 분류
# - route: 기본 라우팅 대상 테이블
# - cooldown_sec: 동일 키워드 재트리거 방지 시간
# - synonyms: canonical ↔ synonym 매핑
# - filter_key: 검색 시 metadata filter에 사용
VOCAB_GROUPS = [
    {
        "type": "CARD/PROGRAM",
        "route": "card_tbl",
        "cooldown_sec": 8,
        "synonyms": CARD_NAME_SYNONYMS,
        "filter_key": "card_name",
    },
    {
        "type": "INTENT",
        "route": "guide_tbl",
        "cooldown_sec": 5,
        "synonyms": ACTION_SYNONYMS,
        "filter_key": "intent",
    },
    {
        "type": "WEAK_INTENT",
        "route": "guide_tbl",
        "cooldown_sec": 5,
        "synonyms": WEAK_INTENT_SYNONYMS,
        "filter_key": "weak_intent",
    },
    {
        "type": "PAYMENT",
        "route": "card_tbl",
        "cooldown_sec": 6,
        "synonyms": PAYMENT_SYNONYMS,
        "filter_key": "payment_method",
    },
]
