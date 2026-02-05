"""
RAG 검색 가중치 튜닝 설정

실험을 위한 다양한 프리셋 제공.
환경변수 RAG_TUNING_PRESET 으로 프리셋 선택 (default: balanced)
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TuningPreset:
    """검색 가중치 프리셋"""
    name: str
    description: str

    # RRF 파라미터
    rrf_k: int = 60  # 낮을수록 상위 순위에 가중치

    # 부스트 가중치
    boost_card: float = 0.2       # 카드명 매칭
    boost_intent: float = 0.15    # 의도 키워드 매칭
    boost_payment: float = 0.1    # 결제 관련 매칭
    boost_weak: float = 0.05      # 약한 매칭
    boost_category: float = 0.05  # 카테고리 매칭
    boost_guide: float = 0.004    # 가이드 문서
    boost_guide_coverage: float = 0.01
    boost_intent_title: float = 0.02

    # 페널티
    penalty_card_guide: float = 0.06
    card_top_bonus: float = 0.6

    # 검색 밸런스
    vector_weight: float = 1.0    # 벡터 검색 가중치
    keyword_weight: float = 1.0   # 키워드 검색 가중치


# 프리셋 정의
PRESETS: Dict[str, TuningPreset] = {
    # 기본 설정 (현재 프로덕션)
    "balanced": TuningPreset(
        name="balanced",
        description="기본 균형 설정 (현재 프로덕션)",
        rrf_k=60,
        boost_card=0.2,
        boost_intent=0.15,
    ),

    # 정밀도 최적화 - 상위 문서 정확도 향상
    "precision": TuningPreset(
        name="precision",
        description="Precision@K 최적화 - 상위 결과 정확도 향상",
        rrf_k=30,              # 상위 순위 가중치 증가
        boost_card=0.35,       # 카드명 매칭 강화
        boost_intent=0.25,     # 의도 매칭 강화
        boost_payment=0.15,
        boost_weak=0.03,
        boost_category=0.08,
        boost_guide=0.008,
        boost_guide_coverage=0.015,
        boost_intent_title=0.04,
        penalty_card_guide=0.1,
        card_top_bonus=0.8,
    ),

    # 재현율 최적화 - 관련 문서 발굴 향상
    "recall": TuningPreset(
        name="recall",
        description="Recall@K 최적화 - 더 많은 관련 문서 발굴",
        rrf_k=80,              # 순위 차이 감소
        boost_card=0.15,
        boost_intent=0.12,
        boost_payment=0.08,
        boost_weak=0.08,       # 약한 매칭도 살림
        boost_category=0.1,    # 카테고리 매칭 강화
        boost_guide=0.01,
        boost_guide_coverage=0.02,
        penalty_card_guide=0.03,
        card_top_bonus=0.4,
    ),

    # 벡터 검색 중심
    "vector_heavy": TuningPreset(
        name="vector_heavy",
        description="시맨틱 검색(벡터) 중심",
        rrf_k=40,
        boost_card=0.25,
        boost_intent=0.2,
        vector_weight=1.5,     # 벡터 검색 가중치 증가
        keyword_weight=0.7,
    ),

    # 키워드 검색 중심
    "keyword_heavy": TuningPreset(
        name="keyword_heavy",
        description="키워드 매칭 중심",
        rrf_k=50,
        boost_card=0.3,
        boost_intent=0.25,
        boost_payment=0.15,
        vector_weight=0.7,
        keyword_weight=1.5,
    ),

    # 실험용 - 공격적 튜닝
    "aggressive": TuningPreset(
        name="aggressive",
        description="공격적 튜닝 - Precision 극대화",
        rrf_k=20,              # 매우 낮은 RRF_K
        boost_card=0.5,        # 카드명 매칭 극대화
        boost_intent=0.4,
        boost_payment=0.2,
        boost_weak=0.02,
        boost_category=0.1,
        boost_guide=0.015,
        boost_guide_coverage=0.025,
        boost_intent_title=0.06,
        penalty_card_guide=0.15,
        card_top_bonus=1.0,
    ),
}


def get_current_preset() -> TuningPreset:
    """현재 활성화된 프리셋 반환"""
    preset_name = os.getenv("RAG_TUNING_PRESET", "balanced")
    return PRESETS.get(preset_name, PRESETS["balanced"])


def get_preset(name: str) -> Optional[TuningPreset]:
    """이름으로 프리셋 조회"""
    return PRESETS.get(name)


def list_presets() -> Dict[str, str]:
    """사용 가능한 프리셋 목록"""
    return {name: preset.description for name, preset in PRESETS.items()}


# 환경변수 오버라이드 지원
def get_tuning_value(preset: TuningPreset, key: str, env_key: str) -> float:
    """프리셋 값 또는 환경변수 오버라이드 반환"""
    env_val = os.getenv(env_key)
    if env_val is not None:
        try:
            return float(env_val)
        except ValueError:
            pass
    return getattr(preset, key)
