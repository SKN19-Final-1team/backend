"""
symspellpy-ko 테스트 모듈

기존 코드와 독립적으로 작성됨 - 필요 없으면 파일만 삭제하면 됨

테스트 목적:
- 문맥 기반 STT 오류 교정 가능 여부 확인
- 발송소리가 → 발송처리가 같은 복합 오류 처리 가능 여부

사용법:
    C:\\Users\\bsjun\\anaconda3\\envs\\final_env\\python.exe tests/sllm_refine/test_symspellpy_ko.py
"""

import sys
from pathlib import Path

# 패키지 설치 확인
try:
    from symspellpy_ko import KoSymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False
    print("❌ symspellpy-ko 설치 필요: pip install symspellpy-ko")


def get_symspell():
    """KoSymSpell 인스턴스 생성"""
    if not SYMSPELL_AVAILABLE:
        return None
    
    sym_spell = KoSymSpell()
    sym_spell.load_korean_dictionary(decompose_korean=True, load_bigrams=True)
    return sym_spell


def test_basic_correction():
    """기본 오타 교정 테스트"""
    print("=" * 70)
    print("symspellpy-ko 기본 교정 테스트")
    print("=" * 70)
    
    sym = get_symspell()
    if sym is None:
        print("❌ symspellpy-ko 사용 불가")
        return
    
    test_cases = [
        "하나낸 계좌에서",
        "연예비 납부",
        "바우저 신청",
        "발송소리가 될것같아요",
        "이길 영업일에 처리됩니다",
    ]
    
    for text in test_cases:
        print(f"\n[입력] {text}")
        
        try:
            # lookup_compound: 복합어 교정
            suggestions = sym.lookup_compound(text, max_edit_distance=2)
            if suggestions:
                corrected = suggestions[0].term
                print(f"[교정] {corrected}")
                
                if text != corrected:
                    print("✅ 교정됨")
                else:
                    print("⚠️ 변경 없음")
            else:
                print("⚠️ 교정 결과 없음")
        except Exception as e:
            print(f"❌ 오류: {e}")
        
        print("-" * 70)


def test_stt_error_correction():
    """STT 오류 교정 테스트"""
    print("\n" + "=" * 70)
    print("symspellpy-ko STT 오류 교정 테스트")
    print("=" * 70)
    
    sym = get_symspell()
    if sym is None:
        print("❌ symspellpy-ko 사용 불가")
        return
    
    # STT 오류 케이스
    stt_test_cases = [
        ("결채 금액이 얼마인가요", "결제 금액이 얼마인가요"),
        ("채크카드로 할부 가능한가요", "체크카드로 할부 가능한가요"),
        ("연예비 납부와 그 바우저 한개 선택", "연회비 납부와 그 바우처 한 개 선택"),
        ("발송소리가 될것같아요", "발송처리가 될 것 같아요"),  # 문맥 의존
        ("이길 영업일에 처리됩니다", "익일 영업일에 처리됩니다"),
    ]
    
    success_count = 0
    
    for input_text, expected in stt_test_cases:
        print(f"\n[입력] {input_text}")
        print(f"[기대] {expected}")
        
        try:
            suggestions = sym.lookup_compound(input_text, max_edit_distance=2)
            if suggestions:
                corrected = suggestions[0].term
                print(f"[결과] {corrected}")
                
                # 기대값과 비교
                if corrected == expected or expected in corrected:
                    print("✅ 기대값 일치")
                    success_count += 1
                elif input_text != corrected:
                    print("⚠️ 부분 교정됨")
                else:
                    print("❌ 교정 실패")
            else:
                print("❌ 교정 결과 없음")
        except Exception as e:
            print(f"❌ 오류: {e}")
        
        print("-" * 70)
    
    print(f"\n📊 결과: {success_count}/{len(stt_test_cases)} 성공")


def test_word_segmentation():
    """단어 분리 테스트"""
    print("\n" + "=" * 70)
    print("symspellpy-ko 단어 분리 (띄어쓰기) 테스트")
    print("=" * 70)
    
    sym = get_symspell()
    if sym is None:
        print("❌ symspellpy-ko 사용 불가")
        return
    
    test_cases = [
        "발송소리가될것같아요",
        "연예비납부해주세요",
        "결제일변경해주세요",
    ]
    
    for text in test_cases:
        print(f"\n[입력] {text}")
        
        try:
            # word_segmentation: 띄어쓰기 교정
            result = sym.word_segmentation(text)
            if result:
                print(f"[결과] {result.corrected_string}")
            else:
                print("⚠️ 결과 없음")
        except Exception as e:
            print(f"❌ 오류: {e}")
        
        print("-" * 70)


def test_single_word_lookup():
    """단일 단어 교정 테스트"""
    print("\n" + "=" * 70)
    print("symspellpy-ko 단일 단어 교정 테스트")
    print("=" * 70)
    
    sym = get_symspell()
    if sym is None:
        print("❌ symspellpy-ko 사용 불가")
        return
    
    test_words = [
        "연예비",
        "바우저",
        "결채",
        "채크카드",
        "하나낸",
    ]
    
    for word in test_words:
        print(f"\n[입력] {word}")
        
        try:
            suggestions = sym.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                for i, sug in enumerate(suggestions[:3]):
                    print(f"  [{i+1}] {sug.term} (거리: {sug.distance}, 빈도: {sug.count})")
            else:
                print("⚠️ 교정 결과 없음")
        except Exception as e:
            print(f"❌ 오류: {e}")
        
        print("-" * 70)


def main():
    """메인 함수"""
    print("\n🚀 symspellpy-ko 테스트 시작\n")
    
    if not SYMSPELL_AVAILABLE:
        print("❌ symspellpy-ko가 설치되지 않았습니다.")
        print("설치: pip install symspellpy-ko")
        return
    
    print("📥 한국어 사전 로딩 중...")
    
    # 1. 단일 단어 교정 테스트
    test_single_word_lookup()
    
    # 2. 기본 교정 테스트
    test_basic_correction()
    
    # 3. STT 오류 교정 테스트
    test_stt_error_correction()
    
    # 4. 단어 분리 테스트
    test_word_segmentation()
    
    print("\n" + "=" * 70)
    print("✅ 모든 테스트 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
