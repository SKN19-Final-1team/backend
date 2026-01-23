"""
Persona Generator 테스트
"""
import json
from app.llm.education.persona_generator import create_system_prompt, create_scenario_script


def test_persona_generation():
    """샘플 고객 데이터로 페르소나 생성 테스트"""
    
    # customer.json에서 몇 가지 샘플 로드
    with open("tests/education/customer.json", "r", encoding="utf-8") as f:
        customers = json.load(f)
    
    print("=" * 60)
    print("Persona Generator 테스트")
    print("=" * 60)
    
    # 테스트 1: 초급 난이도
    customer1 = customers[0]  # 김소영
    print(f"\n[테스트 1] 초급 난이도 - {customer1['name']} ({customer1['age_group']})")
    print("-" * 60)
    
    beginner_prompt = create_system_prompt(customer1, difficulty="beginner")
    print(beginner_prompt)
    
    # 테스트 2: 상급 난이도
    customer2 = customers[2]  # 정종현
    print(f"\n\n[테스트 2] 상급 난이도 - {customer2['name']} ({customer2['age_group']})")
    print("-" * 60)
    
    advanced_prompt = create_system_prompt(customer2, difficulty="advanced")
    print(advanced_prompt)
    
    print("\n\n✅ 테스트 완료")


if __name__ == "__main__":
    test_persona_generation()
