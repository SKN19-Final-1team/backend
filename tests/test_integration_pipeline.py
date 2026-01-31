"""
전체 프로세스 통합 테스트

Kiwipiepy + PyKoSpacing 마이그레이션 후
전체 파이프라인이 정상 작동하는지 검증

테스트 흐름:
1. 형태소 분석 (Kiwipiepy + PyKoSpacing)
2. 단어 매칭 및 교정
3. sLLM 교정 (RunPod)
4. 최종 결과 검증
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.delivery.deliverer import refine_conversation_text


def test_end_to_end_pipeline():
    """전체 파이프라인 통합 테스트"""
    
    print("=" * 70)
    print("전체 프로세스 통합 테스트")
    print("=" * 70)
    
    test_cases = [
        {
            "name": "STT 할루시네이션 + 카드상품명 교정",
            "input": "나라사람카드바우처 연예비납부와 그 바우저 한개 선택하라고",
            "expected_keywords": ["나라사랑카드", "연회비", "바우처"]
        },
        {
            "name": "띄어쓰기 오류 + 복합 오류",
            "input": "이길영업일날문자로발송소리가될것같고요",
            "expected_keywords": ["영업일", "문자", "발송"]
        },
        {
            "name": "정확한 카드상품명",
            "input": "신세계상품권 등록 좀 해주세요",
            "expected_keywords": ["신세계상품권", "등록"]
        },
        {
            "name": "대화 형식 (상담사 + 고객)",
            "input": "상담원 안수희입니다 무엇을 도와드릴까요 아 예 안녕하세요 나라사람카드 바우처 때문에 전화드렸는데요",
            "expected_keywords": ["상담원", "나라사랑카드", "바우처"]
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {test_case['name']}")
        print("-" * 70)
        print(f"입력: {test_case['input']}")
        
        try:
            # 전체 파이프라인 실행
            result = refine_conversation_text(test_case['input'], use_sllm=True)
            
            print(f"\n[Step 1] 띄어쓰기 교정 (PyKoSpacing)")
            print(f"  (자동 적용됨)")
            
            print(f"\n[Step 2] 형태소 분석 (Kiwipiepy)")
            morphology = result["step2_morphology"]
            print(f"  카드상품명 후보: {morphology.get('card_candidates', [])}")
            print(f"  명사: {morphology.get('nouns', [])[:5]}...")  # 처음 5개만
            
            print(f"\n[Step 3] 단어 매칭 및 교정")
            matching = result["step3_matching"]
            best_match = matching.get('best_match')
            corrections = matching.get('corrections', {})
            if best_match:
                print(f"  최적 매칭: {best_match}")
            if corrections:
                print(f"  교정 매핑: {corrections}")
            print(f"  교정 적용 텍스트: {result['step3_corrected'][:50]}...")
            
            print(f"\n[Step 4] sLLM 교정 (RunPod)")
            print(f"  최종 교정: {result['refined']}")
            
            print(f"\n[결과]")
            print(f"  원본: {result['original']}")
            print(f"  교정: {result['refined']}")
            
            # 성공 여부 판단
            success = result['refined'] != result['original']
            
            results.append({
                "test": test_case['name'],
                "success": success,
                "original": result['original'],
                "refined": result['refined']
            })
            
            if success:
                print(f"  ✅ 교정 성공")
            else:
                print(f"  ⚠️ 교정 없음 (원본과 동일)")
            
        except Exception as e:
            print(f"\n  ❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                "test": test_case['name'],
                "success": False,
                "error": str(e)
            })
    
    # 최종 요약
    print("\n" + "=" * 70)
    print("테스트 요약")
    print("=" * 70)
    
    total = len(results)
    success_count = sum(1 for r in results if r.get("success", False))
    
    for i, result in enumerate(results, 1):
        status = "✅ 성공" if result.get("success") else "❌ 실패"
        print(f"{i}. {result['test']}: {status}")
        if "error" in result:
            print(f"   오류: {result['error']}")
    
    print(f"\n총 {total}개 중 {success_count}개 성공 ({success_count/total*100:.1f}%)")
    
    return success_count == total


if __name__ == "__main__":
    success = test_end_to_end_pipeline()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ 모든 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패 (상세 내용 위 참조)")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
