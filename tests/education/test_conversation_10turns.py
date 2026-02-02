"""
가상 고객 페르소나와의 10턴 대화 테스트

sLLM 모델(kanana-nano-2.1B-customer-emotional)에 페르소나를 부여하고
실제 상담 시뮬레이션 대화를 진행합니다.
"""
import os
import sys
import json
from datetime import datetime

# 프로젝트 경로 추가
sys.path.insert(0, r"c:\SKN19\backend")
os.chdir(r"c:\SKN19\backend")

from dotenv import load_dotenv
load_dotenv()

print("=" * 70)
print("가상 고객 페르소나 10턴 대화 테스트")
print("=" * 70)

# ============================================================
# 1. 모듈 임포트 및 설정
# ============================================================
from app.llm.education.client import generate_text, RUNPOD_MODEL_NAME
from app.llm.education.tts_speaker import (
    initialize_conversation,
    process_agent_input,
    end_conversation
)
from app.llm.education.persona_generator import create_system_prompt

print(f"\n[설정] 사용 모델: {RUNPOD_MODEL_NAME}")

# ============================================================
# 2. 테스트용 고객 프로필 설정
# ============================================================
# 실제 DB에서 가져올 고객 정보를 시뮬레이션
customer_profile = {
    "name": "김영희",
    "gender": "female",
    "age_group": "50대",
    "grade": "GOLD",
    "card_type": "테디카드 프리미엄",
    "personality_tags": ["elderly", "emotional", "needs_repetition"],
    "communication_style": {
        "tone": "emotional",
        "speed": "slow"
    },
    "llm_guidance": "고령층 고객으로 디지털 기기 사용에 어려움이 있습니다. 천천히 쉬운 말로 설명하고, 공감하며 경청해주세요."
}

print(f"\n[고객 프로필]")
print(f"  이름: {customer_profile['name']}")
print(f"  연령대: {customer_profile['age_group']}")
print(f"  등급: {customer_profile['grade']}")
print(f"  성격: {customer_profile['personality_tags']}")
print(f"  말투: {customer_profile['communication_style']}")

# ============================================================
# 3. 시스템 프롬프트 생성
# ============================================================
system_prompt = create_system_prompt(customer_profile, difficulty="advanced")

print(f"\n[시스템 프롬프트]")
print("-" * 50)
print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
print("-" * 50)

# ============================================================
# 4. 대화 세션 초기화
# ============================================================
session_id = f"test_conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
session = initialize_conversation(session_id, system_prompt, customer_profile)

print(f"\n[세션 초기화] ID: {session_id}")

# ============================================================
# 5. 10턴 대화 시나리오 (카드 분실 상담)
# ============================================================
agent_messages = [
    "안녕하세요, 테디카드 고객센터입니다. 무엇을 도와드릴까요?",
    "네, 카드 분실이시군요. 걱정이 많으셨겠습니다. 먼저 본인 확인을 위해 성함과 생년월일 말씀해주시겠어요?",
    "네, 김영희 고객님 확인되었습니다. 분실하신 카드가 테디카드 프리미엄 맞으시죠?",
    "네, 지금 바로 카드 정지 처리 도와드리겠습니다. 혹시 부정 사용된 내역이 있으신지 확인해보셨나요?",
    "다행히 부정 사용은 없으셨네요. 카드 정지 처리 완료되었습니다. 재발급 신청도 함께 도와드릴까요?",
    "네, 재발급 신청 진행하겠습니다. 기존 주소로 배송해드릴까요, 아니면 다른 주소로 변경하실까요?",
    "기존 주소로 배송 접수했습니다. 3~5 영업일 내로 도착 예정입니다. 카드 수령 전까지 임시 카드가 필요하시면 가까운 영업점 방문하시면 발급 가능합니다.",
    "네, 영업점은 인터넷 검색이 어려우시면 1588-0000으로 전화하셔도 안내받으실 수 있어요. 또 궁금하신 점 있으실까요?",
    "네, 포인트는 그대로 유지됩니다. 걱정 안 하셔도 돼요. 새 카드 받으시면 바로 사용 가능합니다.",
    "네, 도움이 되셨다니 다행입니다. 다른 문의사항 있으시면 언제든 연락주세요. 좋은 하루 되세요!"
]

conversation_log = []

print("\n" + "=" * 70)
print("대화 시작")
print("=" * 70)

for turn, agent_msg in enumerate(agent_messages, 1):
    print(f"\n[턴 {turn}/10]")
    print(f"상담원: {agent_msg}")

    try:
        # AI 고객 응답 생성
        result = process_agent_input(
            session_id=session_id,
            agent_message=agent_msg,
            input_mode="text"
        )

        customer_response = result["customer_response"]
        print(f"고객({customer_profile['name']}): {customer_response}")

        # 로그 저장
        conversation_log.append({
            "turn": turn,
            "agent": agent_msg,
            "customer": customer_response,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"[오류] 응답 생성 실패: {e}")
        conversation_log.append({
            "turn": turn,
            "agent": agent_msg,
            "customer": f"ERROR: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })

# ============================================================
# 6. 대화 종료 및 결과 저장
# ============================================================
print("\n" + "=" * 70)
print("대화 종료")
print("=" * 70)

try:
    summary = end_conversation(session_id)
    print(f"\n[세션 요약]")
    print(f"  총 턴 수: {summary['turn_count']}")
    print(f"  소요 시간: {summary['duration_seconds']:.1f}초")
except Exception as e:
    print(f"[오류] 세션 종료 실패: {e}")
    summary = {"error": str(e)}

# 결과 파일 저장
result = {
    "test_info": {
        "date": datetime.now().isoformat(),
        "model": RUNPOD_MODEL_NAME,
        "session_id": session_id
    },
    "customer_profile": customer_profile,
    "system_prompt": system_prompt,
    "conversation_log": conversation_log,
    "summary": summary
}

output_file = r"c:\SKN19\backend\test_conversation_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n[결과 저장] {output_file}")

# ============================================================
# 7. 대화 품질 분석
# ============================================================
print("\n" + "=" * 70)
print("대화 품질 분석")
print("=" * 70)

if conversation_log:
    # 응답 길이 분석
    response_lengths = [len(log["customer"]) for log in conversation_log if "ERROR" not in log["customer"]]
    if response_lengths:
        avg_length = sum(response_lengths) / len(response_lengths)
        print(f"\n[응답 통계]")
        print(f"  성공한 응답: {len(response_lengths)}/{len(conversation_log)}")
        print(f"  평균 응답 길이: {avg_length:.1f}자")
        print(f"  최소 응답 길이: {min(response_lengths)}자")
        print(f"  최대 응답 길이: {max(response_lengths)}자")

    # 페르소나 일관성 확인 (키워드 체크)
    persona_keywords = ["어머", "죄송", "걱정", "네", "감사"]  # 감정적/공손한 표현
    keyword_matches = 0
    for log in conversation_log:
        if "ERROR" not in log["customer"]:
            for kw in persona_keywords:
                if kw in log["customer"]:
                    keyword_matches += 1
                    break

    print(f"\n[페르소나 일관성]")
    print(f"  감정적/공손한 표현 사용: {keyword_matches}/{len(conversation_log)} 턴")

print("\n테스트 완료!")
