from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

from llm.rag_answerer import generate_answer, generate_detail_cards
from rag.retriever import retrieve_docs, retrieve_multi
from rag.router import route_query
from rag.vocab.rules import STOPWORDS


# --- 설정 ---
@dataclass(frozen=True)
class RAGConfig:
    top_k: int = 5
    model: str = "gpt-4.1"
    temperature: float = 0.2
    no_route_answer: str = "카드명/상황을 조금 더 구체적으로 말씀해 주세요."
    include_docs: bool = True
    normalize_keywords: bool = False
    strict_guidance_script: bool = True


# --- 라우팅 ---
def route(query: str) -> Dict[str, Any]:
    return route_query(query)


# --- 문서 우선순위 ---
def _is_definition_title(title: str) -> bool:
    if not title:
        return False
    lowered = title.lower()
    for marker in ("란", "정의", "소개", "무엇", "무엇인가요"):
        if marker in lowered:
            return True
    return False


def _promote_definition_doc(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for idx, doc in enumerate(docs):
        title = doc.get("title") or ""
        if _is_definition_title(title):
            if idx == 0:
                return docs
            return [docs[idx], *docs[:idx], *docs[idx + 1 :]]
    return docs


# --- 키워드/토큰 정제 ---
_TERM_WS_RE = re.compile(r"\s+")
_TERM_CLEAN_RE = re.compile(r"[^\w가-힣]+")
_PARTICLE_SUFFIXES = (
    "으로",
    "에서",
    "에게",
    "으로",
    "로",
    "와",
    "과",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "도",
    "만",
    "요",
    "죠",
)
_KEYWORD_STOPWORDS = {
    "무엇",
    "무엇인가요",
    "뭔가요",
    "뭐야",
    "뭐에요",
    "뭐예요",
    "뭔지",
    "어떤",
    "어떻게",
    "왜",
    "언제",
    "어디",
    "가능",
    "되나요",
    "되요",
    "해주세요",
    "알려줘",
    "알려주세요",
    "알려줘요",
    "방법",
    "소개",
    "정의",
    "란",
    "카드",
}
_ISSUE_FILTER_TOKENS = ("발급", "신청", "재발급", "대상", "서류")
_BENEFIT_FILTER_TOKENS = ("적립", "혜택", "유의", "제외", "포인트", "할인")


def _unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _strip_particle(term: str) -> str:
    for suffix in _PARTICLE_SUFFIXES:
        if term.endswith(suffix) and len(term) > len(suffix) + 1:
            return term[: -len(suffix)]
    return term


def _normalize_text(text: str) -> str:
    return _TERM_WS_RE.sub(" ", text.strip().lower())


def _strict_guidance_script(script: str, docs: List[Dict[str, Any]]) -> str:
    if not script:
        return ""
    content = " ".join(doc.get("content") or "" for doc in docs).strip()
    if not content:
        return ""
    normalized_content = _normalize_text(content)
    sentences = [s.strip() for s in re.split(r"[.!?\\n]+", script) if s.strip()]
    for sentence in sentences:
        if _normalize_text(sentence) not in normalized_content:
            return ""
    return script


def _extract_query_terms(query: str) -> List[str]:
    text = _TERM_WS_RE.sub(" ", query.strip().lower())
    raw_terms = [term for term in text.split(" ") if term]
    stopwords = {word.lower() for word in STOPWORDS}
    terms = []
    for term in raw_terms:
        term = _TERM_CLEAN_RE.sub("", term)
        if not term:
            continue
        if term in stopwords or term in _KEYWORD_STOPWORDS:
            continue
        if term.endswith("란") and len(term) > 1:
            term = term[:-1]
        term = _strip_particle(term)
        if not term:
            continue
        if term in stopwords or term in _KEYWORD_STOPWORDS:
            continue
        if term.isdigit():
            continue
        if len(term) < 2:
            continue
        terms.append(term)
    return _unique_in_order(terms)


def _collect_query_keywords(query: str, routing: Dict[str, Any], normalize: bool) -> List[str]:
    if normalize:
        matched = routing.get("matched") or {}
        keywords: List[str] = []
        for key in ("card_names", "actions", "payments", "weak_intents"):
            keywords.extend(matched.get(key) or [])
        if not keywords:
            keywords = _extract_query_terms(query)
    else:
        keywords = _extract_query_terms(query)
    normalized = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        if not kw.startswith("#"):
            kw = f"#{kw}"
        normalized.append(kw)
    return _unique_in_order(normalized)


def _text_has_any(text: str, tokens: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(token in lowered for token in tokens)


# --- 카드 후처리 ---
def _split_cards_by_query(cards: List[Dict[str, Any]], query: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    terms = set(_extract_query_terms(query))
    mode: Optional[str] = None
    if terms.intersection(_ISSUE_FILTER_TOKENS):
        mode = "ISSUE"
    elif terms.intersection(_BENEFIT_FILTER_TOKENS):
        mode = "BENEFIT"

    if not mode:
        return cards[:2], cards[2:4]

    def _blocked(card: Dict[str, Any]) -> bool:
        text = f"{card.get('title') or ''} {card.get('content') or ''}"
        if mode == "ISSUE":
            return _text_has_any(text, _BENEFIT_FILTER_TOKENS)
        return _text_has_any(text, _ISSUE_FILTER_TOKENS)

    kept = [card for card in cards if not _blocked(card)]
    blocked = [card for card in cards if _blocked(card)]

    current = kept[:2]
    next_step = kept[2:4]
    if len(current) < 2:
        needed = 2 - len(current)
        current.extend(blocked[:needed])
        blocked = blocked[needed:]
    if len(next_step) < 2:
        needed = 2 - len(next_step)
        next_step.extend(blocked[:needed])
    return current, next_step


def _omit_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, val in value.items():
            cleaned_val = _omit_empty(val)
            if cleaned_val in ("", None, [], {}):
                continue
            cleaned[key] = cleaned_val
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            cleaned_item = _omit_empty(item)
            if cleaned_item in ("", None, [], {}):
                continue
            cleaned_list.append(cleaned_item)
        return cleaned_list
    return value


# --- 검색 ---
async def retrieve(
    query: str,
    routing: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    filters = routing.get("filters") or {}
    route_name = routing.get("route")

    sources = set()
    if route_name == "card_info":
        sources.add("card_tbl")
    elif filters.get("card_name"):
        sources.update({"card_tbl", "guide_tbl"})
    if filters.get("payment_method"):
        sources.update({"card_tbl", "guide_tbl"})
    if filters.get("intent") or filters.get("weak_intent"):
        sources.add("guide_tbl")
    if not sources:
        sources.update({"card_tbl", "guide_tbl"})

    return await retrieve_multi(
        query=query,
        routing=routing,
        tables=sorted(sources),
        top_k=top_k,
    )


# --- 답변 생성 ---
def answer(
    query: str,
    docs: List[Dict[str, Any]],
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    return generate_answer(query=query, docs=docs, model=model, temperature=temperature)


# --- 파이프라인 ---
async def run_rag(query: str, config: Optional[RAGConfig] = None) -> Dict[str, Any]:
    cfg = config or RAGConfig()
    routing = route(query)

    if not routing.get("should_route"):
        return {
            "currentSituation": [],
            "nextStep": [],
            "guidanceScript": cfg.no_route_answer,
            "routing": routing,
            "meta": {"model": None, "doc_count": 0, "context_chars": 0},
        }

    docs = await retrieve(query=query, routing=routing, top_k=cfg.top_k)
    if routing.get("route") == "card_info":
        docs = _promote_definition_doc(docs)
    cards, guidance_script = generate_detail_cards(
        query=query,
        docs=docs,
        model=cfg.model,
        temperature=0.0,
    )
    if cfg.strict_guidance_script:
        guidance_script = _strict_guidance_script(guidance_script, docs)
    query_keywords = _collect_query_keywords(query, routing, cfg.normalize_keywords)
    for card in cards:
        card["keywords"] = query_keywords
    cards = [_omit_empty(card) for card in cards]
    current_cards, next_cards = _split_cards_by_query(cards, query)

    response = {
        "currentSituation": current_cards,
        "nextStep": next_cards,
        "guidanceScript": guidance_script or "",
        "routing": routing,
        "meta": {"model": cfg.model, "doc_count": len(docs), "context_chars": 0},
    }
    if cfg.include_docs:
        response["docs"] = docs
    return response
