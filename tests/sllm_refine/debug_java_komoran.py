"""
PyKomoran Java 연동 디버그
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Java 환경 확인")
print("=" * 70)

# JPype 확인
try:
    import jpype
    print(f"✓ JPype 설치됨: {jpype.__version__}")
    print(f"  JVM 경로: {jpype.getDefaultJVMPath()}")
    
    # JVM 시작 확인
    if not jpype.isJVMStarted():
        print("  JVM 시작 시도...")
        jpype.startJVM(jpype.getDefaultJVMPath())
        print("  ✓ JVM 시작 성공")
    else:
        print("  ✓ JVM 이미 실행 중")
        
except ImportError:
    print("✗ JPype 설치 안 됨")
    print("  설치: pip install JPype1")
    sys.exit(1)
except Exception as e:
    print(f"✗ JVM 시작 실패: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("PyKomoran 초기화 테스트")
print("=" * 70)

try:
    from PyKomoran import Komoran
    
    print("STABLE 모델 초기화 중...")
    komoran = Komoran("STABLE")
    print("✓ Komoran 초기화 성공")
    
    # 테스트
    test_text = "테디카드로 결제해줘"
    print(f"\n테스트 입력: {test_text}")
    
    # 다양한 메서드 시도
    print("\n1. pos() 메서드:")
    result_pos = komoran.pos(test_text)
    print(f"   결과: {result_pos}")
    
    print("\n2. nouns() 메서드:")
    result_nouns = komoran.nouns(test_text)
    print(f"   결과: {result_nouns}")
    
    print("\n3. morphes() 메서드:")
    result_morphes = komoran.morphes(test_text)
    print(f"   결과: {result_morphes}")
    
    print("\n4. get_plain_text() 메서드:")
    result_plain = komoran.get_plain_text(test_text)
    print(f"   결과: {result_plain}")
    
    # 사용자사전 테스트
    print("\n" + "=" * 70)
    print("사용자사전 테스트")
    print("=" * 70)
    
    from app.llm.delivery.morphology_analyzer import create_user_dictionary
    dict_path = create_user_dictionary()
    
    print(f"\n사용자사전 로드: {dict_path}")
    komoran.set_user_dic(dict_path)
    print("✓ 사용자사전 설정 완료")
    
    print(f"\n테스트 입력: {test_text}")
    result_pos = komoran.pos(test_text)
    print(f"pos() 결과: {result_pos}")
    
    result_nouns = komoran.nouns(test_text)
    print(f"nouns() 결과: {result_nouns}")
    
except Exception as e:
    print(f"\n✗ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("디버그 완료")
print("=" * 70)
