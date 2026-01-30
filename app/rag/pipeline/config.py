from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    top_k: int = 4
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    no_route_answer: str = "카드명/상황을 조금 더 구체적으로 말씀해 주세요."
    include_docs: bool = True
    include_consult_docs: bool = False
    normalize_keywords: bool = True
    strict_guidance_script: bool = True
    llm_card_top_n: int = 2
    enable_consult_search: bool = True
