from __future__ import annotations

DOC_SOURCE_FILTERS = {
    "hyundai_applepay": "id LIKE 'hyundai_applepay_%'",
    "guide_merged": "id LIKE '%_merged'",
    "guide_general": "id NOT LIKE 'sinhan_terms_%' AND id NOT LIKE '%_merged' AND id NOT LIKE 'hyundai_applepay_%'",
    "guide_all": "id NOT LIKE 'sinhan_terms_%' AND id NOT LIKE 'hyundai_applepay_%'",  # merged + general 통합
    "guide_with_terms": "id NOT LIKE 'hyundai_applepay_%'",  # merged + general + terms
    "terms": "id LIKE 'sinhan_terms_%'",
}


# Policy Pins(핀)는 boost(가중치)로만 약하게 적용, 최소 1개 보장, top_k 다양성(card/guide) 유지가 원칙
ALLOWED_SCOPE_FILTERS = frozenset(DOC_SOURCE_FILTERS.values())
