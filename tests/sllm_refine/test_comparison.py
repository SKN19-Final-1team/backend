"""
STT 오류 교정 방법 비교 테스트

비교 대상:
1. correction_map (keywords_dict_refine.json)
2. symspellpy-ko
3. sLLM (RunPod)
4. 전체 파이프라인 (correction_map + symspellpy-ko + sLLM)

동일 조건에서 비교하여 가장 효과적인 조합 찾기

사용법:
    C:\\Users\\bsjun\\anaconda3\\envs\\final_env\\python.exe tests/sllm_refine/test_comparison.py
"""

import sys
import json
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# 기존 모듈
from app.llm.delivery.morphology_analyzer import apply_text_corrections, get_correction_map

# symspellpy-ko
try:
    from symspellpy_ko import KoSymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False

# sLLM
try:
    from app.llm.delivery.sllm_refiner import SLMRefiner
    SLLM_AVAILABLE = True
except ImportError:
    SLLM_AVAILABLE = False

# 테스트 케이스
TEST_CASES = [
    {
        "input": "하나낸 계좌에서 먼저 출금할까요",
        "expected": "하나은행 계좌에서 먼저 출금할까요",
        "errors": ["하나낸→하나은행"]
    },
    {
        "input": "연예비 납부와 그 바우저 한개 선택",
        "expected": "연회비 납부와 그 바우처 한개 선택",
        "errors": ["연예비→연회비", "바우저→바우처"]
    },
    {
        "input": "결채 금액이 얼마인가요",
        "expected": "결제 금액이 얼마인가요",
        "errors": ["결채→결제"]
    },
    {
        "input": "발송소리가 될것같아요",
        "expected": "발송처리가 될것같아요",
        "errors": ["발송소리가→발송처리가 (문맥 의존)"]
    },
    {
        "input": "이길 영업일에 처리됩니다",
        "expected": "익일 영업일에 처리됩니다",
        "errors": ["이길→익일"]
    },
    {
        "input": "채크카드로 할부 가능한가요",
        "expected": "체크카드로 할부 가능한가요",
        "errors": ["채크카드→체크카드"]
    },
]


# symspellpy-ko 인스턴스 (싱글톤)
_symspell_instance = None

def get_symspell():
    """KoSymSpell 인스턴스 반환"""
    global _symspell_instance
    
    if not SYMSPELL_AVAILABLE:
        return None
    
    if _symspell_instance is None:
        print("[SymSpell] 초기화 중...")
        _symspell_instance = KoSymSpell()
        _symspell_instance.load_korean_dictionary(decompose_korean=True, load_bigrams=True)
        
        # 금융 전문 용어 추가 (correction_map에서)
        correction_map = get_correction_map()
        added = 0
        for word in set(correction_map.values()):
            try:
                if word and any('\uac00' <= c <= '\ud7a3' for c in word):
                    _symspell_instance.create_dictionary_entry(word, 1000)
                    added += 1
            except:
                pass
        print(f"[SymSpell] 금융 용어 {added}개 추가")
    
    return _symspell_instance


def method_correction_map(text: str) -> str:
    """방법 1: correction_map만 사용"""
    return apply_text_corrections(text)


def method_symspell(text: str) -> str:
    """방법 2: symspellpy-ko만 사용"""
    sym = get_symspell()
    if sym is None:
        return text
    
    try:
        suggestions = sym.lookup_compound(text, max_edit_distance=2)
        if suggestions:
            return suggestions[0].term
    except Exception as e:
        pass
    
    return text


def method_correction_map_then_symspell(text: str) -> str:
    """방법 3: correction_map → symspellpy-ko"""
    # Step 1: correction_map
    step1 = apply_text_corrections(text)
    
    # Step 2: symspellpy-ko
    step2 = method_symspell(step1)
    
    return step2


def method_sllm(text: str) -> str:
    """방법 4: sLLM만 사용"""
    if not SLLM_AVAILABLE:
        return text
    
    try:
        refiner = SLMRefiner()
        result = refiner.refine_with_sllm(text)
        return result if result else text
    except Exception as e:
        print(f"[sLLM] 오류: {e}")
        return text


def method_full_pipeline(text: str) -> str:
    """방법 5: correction_map → symspellpy-ko → sLLM (전체 파이프라인)"""
    # Step 1: correction_map
    step1 = apply_text_corrections(text)
    
    # Step 2: symspellpy-ko
    step2 = method_symspell(step1)
    
    # Step 3: sLLM
    step3 = method_sllm(step2)
    
    return step3


def calculate_similarity(result: str, expected: str) -> float:
    """문자열 유사도 계산 (0.0 ~ 1.0)"""
    if result == expected:
        return 1.0
    
    # 간단한 단어 일치율
    result_words = set(result.replace(" ", ""))
    expected_words = set(expected.replace(" ", ""))
    
    if not expected_words:
        return 1.0 if not result_words else 0.0
    
    intersection = result_words & expected_words
    return len(intersection) / len(expected_words)


def run_comparison():
    """비교 테스트 실행"""
    print("=" * 80)
    print("STT 오류 교정 방법 비교 테스트")
    print("=" * 80)
    
    methods = [
        ("1. correction_map", method_correction_map),
        ("2. symspellpy-ko", method_symspell),
        ("3. correction_map + symspell", method_correction_map_then_symspell),
    ]
    
    # sLLM은 시간이 오래 걸리므로 선택적
    include_sllm = input("\nsLLM 테스트 포함? (y/n, 시간 소요): ").strip().lower() == 'y'
    if include_sllm and SLLM_AVAILABLE:
        methods.append(("4. sLLM", method_sllm))
        methods.append(("5. 전체 파이프라인", method_full_pipeline))
    
    results = {name: {"correct": 0, "partial": 0, "failed": 0, "time": 0} for name, _ in methods}
    
    for case in TEST_CASES:
        input_text = case["input"]
        expected = case["expected"]
        
        print(f"\n{'='*80}")
        print(f"[테스트] {input_text}")
        print(f"[기대값] {expected}")
        print(f"[오류]   {', '.join(case['errors'])}")
        print("-" * 80)
        
        for name, method in methods:
            start = time.time()
            try:
                result = method(input_text)
            except Exception as e:
                result = f"ERROR: {e}"
            elapsed = time.time() - start
            
            results[name]["time"] += elapsed
            
            # 평가
            if result == expected:
                status = "✅ 완전 일치"
                results[name]["correct"] += 1
            elif result != input_text and calculate_similarity(result, expected) > 0.7:
                status = "⚠️ 부분 교정"
                results[name]["partial"] += 1
            else:
                status = "❌ 실패"
                results[name]["failed"] += 1
            
            print(f"{name:30} | {result}")
            print(f"{'':30} | {status} ({elapsed*1000:.0f}ms)")
    
    # 최종 결과
    print("\n" + "=" * 80)
    print("📊 최종 결과")
    print("=" * 80)
    
    print(f"\n{'방법':30} | {'완전일치':8} | {'부분교정':8} | {'실패':8} | {'시간':10}")
    print("-" * 80)
    
    for name, stats in results.items():
        total = stats["correct"] + stats["partial"] + stats["failed"]
        print(f"{name:30} | {stats['correct']:8} | {stats['partial']:8} | {stats['failed']:8} | {stats['time']*1000:8.0f}ms")
    
    print("\n" + "=" * 80)


def main():
    print("\n🚀 STT 오류 교정 방법 비교 테스트 시작\n")
    
    print("[사용 가능한 도구]")
    print(f"  - correction_map: ✅ ({len(get_correction_map())}개 패턴)")
    print(f"  - symspellpy-ko:  {'✅' if SYMSPELL_AVAILABLE else '❌ 설치 필요'}")
    print(f"  - sLLM (RunPod):  {'✅' if SLLM_AVAILABLE else '❌ 설치 필요'}")
    
    if SYMSPELL_AVAILABLE:
        get_symspell()  # 미리 로딩
    
    run_comparison()
    
    print("✅ 테스트 완료")


if __name__ == "__main__":
    main()
