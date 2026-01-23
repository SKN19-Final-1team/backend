from __future__ import annotations

POLICY_PINS = [
    {
        "name": "narasarang_loss",
        "table": "service_guide_documents",
        "doc_ids": [
            "narasarang_faq_005",
            "narasarang_faq_006",
            "narasarang_faq_010",
        ],
        "card_names": ["나라사랑"],
        "tokens": ["나라사랑"],
    },
    {
        "name": "fees_interest",
        "table": "service_guide_documents",
        "doc_ids": ["카드상품별_거래조건_이자율__수수료_등__merged"],
        "tokens": ["수수료", "이자율", "이자", "현금서비스", "카드론", "카드상품별", "대출", "카드대출", "단기카드대출"],
    },
    {
        "name": "revolving_terms",
        "table": "service_guide_documents",
        "doc_ids": [
            "sinhan_terms_credit_신용카드_개인회원_약관_039",
            "sinhan_terms_credit_신용카드_개인회원_약관_040",
        ],
        "tokens": ["약관", "리볼빙", "조항", "규정"],
    },
    {
        "name": "kookminhappy_guide",
        "table": "service_guide_documents",
        "doc_ids": ["국민행복카드_28"],
        "tokens": ["국민행복", "통신사", "자동납부", "통신요금", "통신료"],
    },
    {
        "name": "kookminhappy_card",
        "table": "card_products",
        "doc_ids": ["CARD-SHINHAN-국민행복(신용_체크)"],
        "tokens": ["국민행복", "자동납부", "통신요금", "통신료"],
    },
    {
        "name": "dadungi_benefit",
        "table": "service_guide_documents",
        "doc_ids": ["dadungi_013"],
        "tokens": ["다둥이", "배달앱", "편의점", "서울시다둥이"],
    },
    {
        "name": "loss_general",
        "table": "service_guide_documents",
        "doc_ids": ["카드분실_도난_관련피해_예방_및_대응방법_merged"],
        "tokens": ["분실", "도난", "잃어버", "분실신고"],
        "exclude_card_names": ["나라사랑"],
    },
]
