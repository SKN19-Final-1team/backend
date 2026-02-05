from app.rag.retriever.tuning import get_current_preset, get_tuning_value

_PRESET = get_current_preset()
RRF_K = int(get_tuning_value(_PRESET, "rrf_k", "RAG_RRF_K"))
CARD_META_WEIGHT = 8
CATEGORY_MATCH_TOKENS = ("발급", "신청", "재발급", "대상", "서류", "적립", "혜택")
KEYWORD_STOPWORDS = {"카드"}
PRIORITY_TERMS_BY_CATEGORY = {
    "발급": ["발급 대상"],
    "신청": ["발급 대상"],
    "재발급": ["발급 대상"],
    "적립": ["적립 서비스", "일상 생활비 적립", "필수 생활비 적립", "포인트 적립"],
}
ISSUE_TERMS = {
    "발급",
    "신청",
    "재발급",
    "대상",
    "서류",
    "오류",
    "에러",
    "분실",
    "도난",
    "잃어버",
    "해지",
    "탈회",
    "취소",
    "사용내역",
    "이용내역",
    "거래내역",
    "결제일",
    "한도",
    "이용한도",
    "리볼빙",
    "조회",
}
BENEFIT_TERMS = {"적립", "혜택", "할인", "포인트"}
REISSUE_TERMS = {"재발급", "재발행"}
MIN_GUIDE_CONTENT_LEN = 60
