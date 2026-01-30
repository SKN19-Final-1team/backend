"""
형태소 분석기 간단 디버그 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("KOMORAN 설치 확인")
print("=" * 70)

try:
    from PyKomoran import Komoran
    print("✓ PyKomoran 설치됨")
    
    # 기본 KOMORAN 테스트 (DEFAULT 모델 사용)
    print("\n기본 KOMORAN 테스트 (사용자사전 없음):")
    komoran_basic = Komoran("STABLE")  # STABLE 또는 LIGHT 모델
    test_text = "테디카드로 결제해줘"
    result = komoran_basic.pos(test_text)
    print(f"입력: {test_text}")
    print(f"결과: {result}")
    
except ImportError as e:
    print(f"✗ PyKomoran 설치 안 됨: {e}")
    print("\n설치 방법:")
    print("  conda activate vectordb")
    print("  pip install PyKomoran")
    sys.exit(1)
except Exception as e:
    print(f"✗ PyKomoran 초기화 실패: {e}")
    print("\nPyKomoran 사용법:")
    print("  Komoran('STABLE')  # 안정 버전")
    print("  Komoran('LIGHT')   # 경량 버전")
    sys.exit(1)
    
except ImportError as e:
    print(f"✗ PyKomoran 설치 안 됨: {e}")
    print("\n설치 방법:")
    print("  conda activate vectordb")
    print("  pip install PyKomoran")
    sys.exit(1)

print("\n" + "=" * 70)
print("사용자사전 생성 테스트")
print("=" * 70)

try:
    from app.llm.delivery.morphology_analyzer import (
        create_user_dictionary,
        get_user_dict_stats
    )
    
    # 사용자사전 생성
    dict_path = create_user_dictionary()
    print(f"\n사용자사전 경로: {dict_path}")
    
    # 사전 내용 확인
    with open(dict_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"등록된 단어 수: {len(lines)}")
        print("\n처음 10개 단어:")
        for line in lines[:10]:
            print(f"  {line.strip()}")
    
    # 사용자사전으로 KOMORAN 초기화
    print("\n사용자사전 적용 KOMORAN 테스트:")
    komoran_custom = Komoran("STABLE")
    komoran_custom.set_user_dic(dict_path)
    
    test_cases = [
        "테디카드로 결제해줘",
        "나라사랑카드 혜택이 뭐야",
    ]
    
    for text in test_cases:
        result = komoran_custom.pos(text)
        print(f"\n입력: {text}")
        print(f"결과: {result}")
        
        # 명사만 추출
        nouns = [morph for morph, pos in result if pos in {'NNG', 'NNP', 'NNB'}]
        print(f"명사: {nouns}")
    
except Exception as e:
    print(f"\n✗ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("디버그 완료")
print("=" * 70)
