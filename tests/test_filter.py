import os
import time
import asyncio

from app.llm.sllm_refiner import refine_text
from app.rag.pipeline import run_rag, RAGConfig


def main():    
    while True:
        user_input = input("\n입력: ").strip()
        
        # 종료 조건
        if user_input.lower() in ['종료']:
            break
        
        s = time.time()
        
        # 1. 텍스트 교정 및 키워드 추출
        result = refine_text(user_input)
        refined_text = result['text']
        keywords = result['keywords']
        
        # # 2. RAG 검색 쿼리 결정
        # # 키워드가 있으면 키워드로 검색, 없으면 교정된 문장으로 검색
        # if keywords:
        #     search_query = " ".join(kw.lstrip("#") for kw in keywords)
        # else:
        #     search_query = refined_text
        
        # # 3. RAG 검색 실행
        # config = RAGConfig(top_k=5)
        # rag_result = asyncio.run(run_rag(search_query, config))
        
        e = time.time()
        
        # 4. 결과 출력
        print(f"원본: {user_input}")
        print(f"교정: {refined_text}")
        print(f"키워드: {', '.join(keywords) if keywords else '(없음)'}")
        # print(f"검색쿼리: {search_query}")
        # print(f"\n[RAG 검색 결과]")
        # print(f"라우팅: {rag_result['routing'].get('route', 'N/A')}")
        # print(f"검색문서: {rag_result['meta']['doc_count']}개")
        # print(f"안내스크립트: {rag_result['guidanceScript'][:100]}..." if len(rag_result['guidanceScript']) > 100 else f"안내스크립트: {rag_result['guidanceScript']}")
        # print(f"\n현재상황 카드: {len(rag_result['currentSituation'])}개")
        # for i, card in enumerate(rag_result['currentSituation'][:2], 1):  # 최대 2개만 출력
        #     print(f"  {i}. {card.get('title', 'N/A')}")
        # print(f"다음단계 카드: {len(rag_result['nextStep'])}개")
        # for i, card in enumerate(rag_result['nextStep'][:2], 1):  # 최대 2개만 출력
        #     print(f"  {i}. {card.get('title', 'N/A')}")
        
        print(f"\n소요시간: {e-s:.2f}s")

if __name__ == "__main__":
    main()

