RRF_K = 60
QUERY_TITLE_WEIGHT = 2
QUERY_CONTENT_WEIGHT = 1
CARD_META_WEIGHT = 3
TITLE_SCORE_WEIGHT = 0.001
ISSUANCE_HINT_TOKENS = ("발급", "신청", "재발급", "대상", "서류")
CATEGORY_MATCH_TOKENS = ("발급", "신청", "재발급", "대상", "서류", "적립", "혜택")
ISSUANCE_TITLE_BONUS = {
    "발급 대상": 6,
    "발급": 4,
    "신청": 3,
    "구비": 2,
    "서류": 2,
    "신규": 2,
}
ISSUANCE_TITLE_DEMOTE = ("유의사항", "공통 제외", "적립", "혜택")
CATEGORY_TITLE_BONUS = {
    "적립 서비스": 4,
    "일상 생활비": 3,
    "필수 생활비": 3,
    "포인트 적립": 2,
}
CATEGORY_TITLE_DEMOTE = ("유의사항", "공통 제외")
KEYWORD_STOPWORDS = {"카드"}
PRIORITY_TERMS_BY_CATEGORY = {
    "발급": ["발급 대상"],
    "신청": ["발급 대상"],
    "재발급": ["발급 대상"],
    "적립": ["적립 서비스", "일상 생활비 적립", "필수 생활비 적립", "포인트 적립"],
}
ISSUE_TERMS = {"발급", "신청", "재발급", "대상", "서류"}
BENEFIT_TERMS = {"적립", "혜택", "할인", "포인트"}
REISSUE_TERMS = {"재발급", "재발행"}
REISSUE_TITLE_PENALTY = 4
MIN_GUIDE_CONTENT_LEN = 60
