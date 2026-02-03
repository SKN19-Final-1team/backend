"""
Consultations 확장 필드 데이터 생성 모듈

기존 consultations 테이블의 확장 필드에 실제 데이터를 채워넣음:
- call_end_time: 통화 종료 시간 (call_time + call_duration)
- acw_duration: 후처리 시간 (1-5분)
- transcript: 화자 분리된 상담 전문 (JSONB)
- ai_summary: AI 생성 상담 요약
- agent_notes: 상담사 메모
- follow_up_schedule: 후속 일정 (10% 비율)
- transfer_department: 이관 부서 (5% 비율)
- transfer_notes: 이관 전달 사항
- sentiment: 감정 분석 결과
- emotion_score: 감정 점수 (0-100)
- satisfaction_score: 만족도 (1-5)
- feedback_text: 피드백 텍스트
- feedback_emotions: 감정 태그 (JSONB)
- recording_file_path: 녹취 파일 경로
- recording_duration: 녹취 시간
- recording_file_size: 녹취 파일 크기
- referenced_documents: 참조 문서 목록

멱등성: RANDOM_SEED=42로 동일한 결과 보장
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, Json as PsycopgJson


RANDOM_SEED = 42

# 감정 분포: positive 40%, neutral 45%, negative 15%
SENTIMENT_DISTRIBUTION = ['positive'] * 40 + ['neutral'] * 45 + ['negative'] * 15

# 감정 태그
EMOTION_TAGS = {
    'positive': ['satisfied', 'grateful', 'relieved', 'happy'],
    'neutral': ['calm', 'neutral', 'patient'],
    'negative': ['frustrated', 'angry', 'disappointed', 'anxious', 'confused']
}

# 이관 부서 목록
TRANSFER_DEPARTMENTS = [
    '카드발급팀', '결제심사팀', '한도관리팀', '분쟁처리팀',
    '해외서비스팀', 'VIP전담팀', '기술지원팀', '민원처리팀'
]

# 후속 일정 템플릿
FOLLOW_UP_TEMPLATES = [
    "3일 후 카드 수령 확인 전화",
    "1주일 후 서비스 만족도 확인",
    "결제일 전 납부 안내 연락",
    "심사 결과 통보 후 안내 전화",
    "재발급 카드 배송 확인",
    None, None, None, None, None  # 50%는 후속 일정 없음
]

# 카테고리별 대화 템플릿
TRANSCRIPT_TEMPLATES = {
    "분실/도난": [
        {"speaker": "agent", "message": "안녕하세요, 테디카드 고객센터입니다. 무엇을 도와드릴까요?"},
        {"speaker": "customer", "message": "네, 카드를 분실했어요. 정지 처리 부탁드립니다."},
        {"speaker": "agent", "message": "네, 본인 확인을 위해 성함과 생년월일 말씀해 주시겠어요?"},
        {"speaker": "customer", "message": "{customer_name}이고요, 생년월일은 {birth_date}입니다."},
        {"speaker": "agent", "message": "확인되었습니다. 카드 정지 처리 도와드리겠습니다."},
        {"speaker": "agent", "message": "분실 신고 접수 완료되었습니다. 재발급 신청도 함께 진행하시겠어요?"},
        {"speaker": "customer", "message": "네, 재발급도 부탁드려요."},
        {"speaker": "agent", "message": "재발급 신청 완료되었습니다. 3-5 영업일 내 배송됩니다."},
        {"speaker": "customer", "message": "감사합니다."},
        {"speaker": "agent", "message": "더 필요하신 사항 있으실까요?"},
        {"speaker": "customer", "message": "아니요, 없습니다."},
        {"speaker": "agent", "message": "테디카드를 이용해 주셔서 감사합니다. 좋은 하루 되세요."},
    ],
    "결제/승인": [
        {"speaker": "agent", "message": "안녕하세요, 테디카드 고객센터입니다."},
        {"speaker": "customer", "message": "결제가 안 되는데 확인 좀 해주세요."},
        {"speaker": "agent", "message": "네, 본인 확인을 위해 성함과 생년월일 말씀해 주세요."},
        {"speaker": "customer", "message": "{customer_name}, {birth_date}입니다."},
        {"speaker": "agent", "message": "확인 중입니다. 잠시만 기다려 주세요."},
        {"speaker": "agent", "message": "확인 결과, 한도 초과로 결제가 거절되었습니다."},
        {"speaker": "customer", "message": "한도가 얼마나 남았나요?"},
        {"speaker": "agent", "message": "현재 잔여 한도는 {remaining_limit}원입니다."},
        {"speaker": "customer", "message": "알겠습니다. 감사합니다."},
        {"speaker": "agent", "message": "더 문의사항 있으실까요?"},
        {"speaker": "customer", "message": "없습니다."},
        {"speaker": "agent", "message": "이용해 주셔서 감사합니다."},
    ],
    "이용내역": [
        {"speaker": "agent", "message": "안녕하세요, 테디카드입니다."},
        {"speaker": "customer", "message": "이번 달 이용내역 확인하고 싶어요."},
        {"speaker": "agent", "message": "본인 확인 부탁드립니다. 성함과 생년월일요."},
        {"speaker": "customer", "message": "{customer_name}, {birth_date}요."},
        {"speaker": "agent", "message": "이번 달 총 이용금액은 {total_amount}원입니다."},
        {"speaker": "customer", "message": "상세 내역도 알 수 있나요?"},
        {"speaker": "agent", "message": "네, 문자로 발송해 드리겠습니다."},
        {"speaker": "customer", "message": "감사합니다."},
        {"speaker": "agent", "message": "좋은 하루 되세요."},
    ],
    "한도": [
        {"speaker": "agent", "message": "안녕하세요, 테디카드 고객센터입니다."},
        {"speaker": "customer", "message": "한도 증액 신청하고 싶습니다."},
        {"speaker": "agent", "message": "본인 확인을 위해 정보 말씀해 주세요."},
        {"speaker": "customer", "message": "{customer_name}, {birth_date}입니다."},
        {"speaker": "agent", "message": "현재 한도는 {current_limit}원입니다. 희망 한도가 있으신가요?"},
        {"speaker": "customer", "message": "{desired_limit}원으로 올려주세요."},
        {"speaker": "agent", "message": "심사 후 결과를 문자로 안내드리겠습니다."},
        {"speaker": "customer", "message": "얼마나 걸리나요?"},
        {"speaker": "agent", "message": "보통 1-2 영업일 소요됩니다."},
        {"speaker": "customer", "message": "알겠습니다."},
        {"speaker": "agent", "message": "감사합니다. 좋은 하루 되세요."},
    ],
    "수수료/연체": [
        {"speaker": "agent", "message": "안녕하세요, 테디카드입니다."},
        {"speaker": "customer", "message": "연체 수수료가 나왔는데 확인해 주세요."},
        {"speaker": "agent", "message": "본인 확인 부탁드립니다."},
        {"speaker": "customer", "message": "{customer_name}, {birth_date}입니다."},
        {"speaker": "agent", "message": "확인 결과, 결제일에 잔액 부족으로 연체되었습니다."},
        {"speaker": "customer", "message": "연체료가 얼마인가요?"},
        {"speaker": "agent", "message": "연체료는 {late_fee}원입니다."},
        {"speaker": "customer", "message": "지금 바로 납부할게요."},
        {"speaker": "agent", "message": "즉시출금으로 처리해 드리겠습니다."},
        {"speaker": "customer", "message": "감사합니다."},
        {"speaker": "agent", "message": "처리 완료되었습니다. 좋은 하루 되세요."},
    ],
    "포인트/혜택": [
        {"speaker": "agent", "message": "안녕하세요, 테디카드 고객센터입니다."},
        {"speaker": "customer", "message": "포인트 잔액 확인하고 싶어요."},
        {"speaker": "agent", "message": "본인 확인 부탁드립니다."},
        {"speaker": "customer", "message": "{customer_name}, {birth_date}입니다."},
        {"speaker": "agent", "message": "현재 포인트 잔액은 {points}P입니다."},
        {"speaker": "customer", "message": "현금으로 전환 가능한가요?"},
        {"speaker": "agent", "message": "네, 1만 포인트부터 전환 가능합니다."},
        {"speaker": "customer", "message": "전환 신청할게요."},
        {"speaker": "agent", "message": "처리 완료되었습니다. 2-3일 내 입금됩니다."},
        {"speaker": "customer", "message": "감사합니다."},
    ],
    "정부지원": [
        {"speaker": "agent", "message": "안녕하세요, 테디카드입니다."},
        {"speaker": "customer", "message": "정부지원금 바우처 사용 방법 알려주세요."},
        {"speaker": "agent", "message": "어떤 바우처인지 말씀해 주시겠어요?"},
        {"speaker": "customer", "message": "문화누리 카드요."},
        {"speaker": "agent", "message": "문화누리 카드는 가맹점에서 바로 결제하시면 됩니다."},
        {"speaker": "customer", "message": "잔액 확인은 어떻게 하나요?"},
        {"speaker": "agent", "message": "앱이나 ARS로 확인 가능합니다."},
        {"speaker": "customer", "message": "알겠습니다. 감사합니다."},
        {"speaker": "agent", "message": "좋은 하루 되세요."},
    ],
    "기타": [
        {"speaker": "agent", "message": "안녕하세요, 테디카드 고객센터입니다."},
        {"speaker": "customer", "message": "문의드릴 게 있어서요."},
        {"speaker": "agent", "message": "네, 말씀해 주세요."},
        {"speaker": "customer", "message": "{inquiry_content}"},
        {"speaker": "agent", "message": "확인해 드리겠습니다. 잠시만요."},
        {"speaker": "agent", "message": "확인 결과 말씀드리겠습니다."},
        {"speaker": "customer", "message": "알겠습니다."},
        {"speaker": "agent", "message": "더 문의사항 있으신가요?"},
        {"speaker": "customer", "message": "없습니다."},
        {"speaker": "agent", "message": "이용해 주셔서 감사합니다."},
    ],
}

# AI 요약 템플릿
AI_SUMMARY_TEMPLATES = {
    "분실/도난": [
        "고객이 카드 분실을 신고하여 즉시 정지 처리 완료. 재발급 신청도 함께 진행됨.",
        "분실 카드 정지 및 재발급 접수 완료. 배송지 확인 후 발송 예정.",
        "도난 신고 접수 및 카드 사용 정지 처리. 부정 사용 내역 없음 확인.",
    ],
    "결제/승인": [
        "결제 오류 문의. 한도 초과로 인한 거절 건으로 확인 및 안내 완료.",
        "승인 거절 원인 확인 요청. 일시적 시스템 오류로 재시도 안내.",
        "해외 결제 차단 해제 요청. 본인 확인 후 해외 결제 활성화 처리.",
    ],
    "이용내역": [
        "이번 달 카드 이용내역 조회 및 상세 내역 문자 발송 완료.",
        "특정 결제 건 확인 요청. 정상 승인 건으로 확인 및 안내.",
        "이용내역 불일치 문의. 가맹점 정보 확인 후 정상 처리 안내.",
    ],
    "한도": [
        "한도 증액 신청 접수. 심사 후 결과 문자 안내 예정.",
        "현재 한도 조회 및 임시 한도 상향 요청. 당일 처리 완료.",
        "한도 조회 요청. 잔여 한도 및 결제 가능 금액 안내 완료.",
    ],
    "수수료/연체": [
        "연체 수수료 발생 원인 확인 및 즉시출금으로 연체 해소 처리.",
        "연회비 면제 조건 문의. 조건 충족으로 면제 처리 완료.",
        "수수료 이의 제기. 확인 결과 정상 청구 건으로 안내.",
    ],
    "포인트/혜택": [
        "포인트 잔액 조회 및 현금 전환 신청 처리 완료.",
        "이벤트 혜택 문의. 참여 조건 및 적립 일정 안내.",
        "마일리지 전환 요청. 항공사 마일리지로 전환 처리 완료.",
    ],
    "정부지원": [
        "문화누리 바우처 사용 방법 및 가맹점 정보 안내 완료.",
        "정부지원금 카드 잔액 조회 및 사용처 안내.",
        "바우처 충전 일정 및 사용 기한 안내 완료.",
    ],
    "기타": [
        "일반 문의 사항 확인 및 관련 정보 안내 완료.",
        "서비스 이용 방법 문의. 상세 안내 및 가이드 발송.",
        "기타 문의 접수 및 처리 완료.",
    ],
}

# 상담사 메모 템플릿
AGENT_NOTES_TEMPLATES = [
    "고객 요청 사항 정상 처리 완료",
    "추가 문의 가능성 있음. 후속 연락 필요할 수 있음",
    "VIP 고객 - 우선 처리 완료",
    "재문의 시 이전 상담 내역 참고 필요",
    "고객 만족도 높음",
    "관련 부서 이관 없이 완결 처리",
    "고객 요청에 따라 문자 안내 발송",
    "시스템 처리 완료. 결과 확인 필요",
    None, None,  # 20%는 메모 없음
]

# 피드백 텍스트 템플릿
FEEDBACK_TEMPLATES = {
    "positive": [
        "친절하게 안내해 주셔서 감사합니다.",
        "빠른 처리 감사드립니다.",
        "설명이 명확해서 이해하기 쉬웠어요.",
        "문제가 바로 해결되어서 좋았습니다.",
        None,
    ],
    "neutral": [
        "보통이었습니다.",
        "필요한 건 처리되었습니다.",
        None, None, None,
    ],
    "negative": [
        "대기 시간이 너무 길었습니다.",
        "설명이 좀 부족했어요.",
        "처리가 오래 걸렸습니다.",
        None,
    ],
}

# 이관 메모 템플릿
TRANSFER_NOTES_TEMPLATES = [
    "고객 요청에 따라 전문 상담 필요",
    "심사 관련 문의로 해당 부서 이관",
    "VIP 고객 전담 상담 요청",
    "기술적 문제 확인 필요",
    "민원 접수에 따른 이관",
]


def calculate_call_end_time(call_time, call_duration_str: str):
    """call_time + call_duration으로 종료 시간 계산"""
    if not call_time or not call_duration_str:
        return None

    try:
        parts = call_duration_str.split(':')
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
        elif len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            minutes += hours * 60
        else:
            return None

        base = datetime(2000, 1, 1, call_time.hour, call_time.minute, call_time.second)
        end = base + timedelta(minutes=minutes, seconds=seconds)
        return end.time()
    except:
        return None


def generate_acw_duration(rng) -> str:
    """후처리 시간 생성 (1-5분)"""
    minutes = rng.randint(1, 5)
    seconds = rng.randint(0, 59)
    return f"{minutes:02d}:{seconds:02d}"


def generate_transcript(category_main: str, customer_name: str, call_time, call_duration_str: str, rng) -> dict:
    """상담 전문 생성"""
    templates = TRANSCRIPT_TEMPLATES.get(category_main, TRANSCRIPT_TEMPLATES["기타"])

    birth_date = f"{rng.randint(1960, 2000)}년 {rng.randint(1, 12)}월 {rng.randint(1, 28)}일"

    variables = {
        "customer_name": customer_name,
        "birth_date": birth_date,
        "remaining_limit": f"{rng.randint(10, 500) * 10000:,}",
        "total_amount": f"{rng.randint(10, 300) * 10000:,}",
        "current_limit": f"{rng.randint(100, 1000) * 10000:,}",
        "desired_limit": f"{rng.randint(200, 2000) * 10000:,}",
        "late_fee": f"{rng.randint(1, 10) * 1000:,}",
        "points": f"{rng.randint(1000, 50000):,}",
        "inquiry_content": "카드 관련 문의드립니다.",
    }

    messages = []

    duration_seconds = 300
    if call_duration_str:
        try:
            parts = call_duration_str.split(':')
            if len(parts) == 2:
                duration_seconds = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                duration_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except:
            pass

    base_seconds = call_time.hour * 3600 + call_time.minute * 60 + call_time.second if call_time else 9 * 3600
    interval = duration_seconds / len(templates) if templates else 30

    for i, template in enumerate(templates):
        msg_seconds = base_seconds + int(i * interval)
        h = (msg_seconds // 3600) % 24
        m = (msg_seconds % 3600) // 60
        s = msg_seconds % 60

        message_text = template["message"]
        for var, val in variables.items():
            message_text = message_text.replace("{" + var + "}", val)

        messages.append({
            "speaker": template["speaker"],
            "message": message_text,
            "timestamp": f"{h:02d}:{m:02d}:{s:02d}"
        })

    return {"messages": messages}


def generate_ai_summary(category_main: str, rng) -> str:
    """AI 요약 생성"""
    templates = AI_SUMMARY_TEMPLATES.get(category_main, AI_SUMMARY_TEMPLATES["기타"])
    return rng.choice(templates)


def generate_agent_notes(rng) -> Optional[str]:
    """상담사 메모 생성 (80% 확률)"""
    return rng.choice(AGENT_NOTES_TEMPLATES)


def generate_follow_up_schedule(rng) -> Optional[str]:
    """후속 일정 생성 (20% 확률)"""
    if rng.random() < 0.2:
        return rng.choice([t for t in FOLLOW_UP_TEMPLATES if t])
    return None


def generate_transfer_info(rng) -> Tuple[Optional[str], Optional[str]]:
    """이관 정보 생성 (5% 확률)"""
    if rng.random() < 0.05:
        dept = rng.choice(TRANSFER_DEPARTMENTS)
        notes = rng.choice(TRANSFER_NOTES_TEMPLATES)
        return dept, notes
    return None, None


def generate_sentiment(rng) -> str:
    """감정 분석 결과 생성"""
    return rng.choice(SENTIMENT_DISTRIBUTION)


def generate_emotion_score(sentiment: str, rng) -> int:
    """감정 점수 생성 (0-100)"""
    if sentiment == "positive":
        return rng.randint(70, 100)
    elif sentiment == "neutral":
        return rng.randint(40, 69)
    else:
        return rng.randint(0, 39)


def generate_satisfaction_score(sentiment: str, rng) -> int:
    """만족도 점수 생성 (1-5)"""
    if sentiment == "positive":
        return rng.randint(4, 5)
    elif sentiment == "neutral":
        return rng.randint(3, 4)
    else:
        return rng.randint(1, 3)


def generate_feedback_text(sentiment: str, rng) -> Optional[str]:
    """피드백 텍스트 생성 (60% 확률)"""
    templates = FEEDBACK_TEMPLATES.get(sentiment, FEEDBACK_TEMPLATES["neutral"])
    return rng.choice(templates)


def generate_feedback_emotions(sentiment: str, rng) -> List[str]:
    """감정 태그 생성"""
    tags = EMOTION_TAGS.get(sentiment, EMOTION_TAGS["neutral"])
    num_tags = rng.randint(1, min(3, len(tags)))
    return rng.sample(tags, num_tags)


def generate_recording_info(consultation_id: str, call_duration_str: str, rng) -> Tuple[str, str, int]:
    """녹취 정보 생성"""
    file_path = f"/recordings/{consultation_id}.wav"
    recording_duration = call_duration_str if call_duration_str else "05:00"

    try:
        parts = recording_duration.split(':')
        if len(parts) == 2:
            minutes = int(parts[0]) + int(parts[1]) / 60
        elif len(parts) == 3:
            minutes = int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
        else:
            minutes = 5
    except:
        minutes = 5

    file_size = int(minutes * 1024 * 1024 * (0.8 + rng.random() * 0.4))

    return file_path, recording_duration, file_size


def generate_referenced_documents(category_main: str, rng) -> list:
    """참조 문서 목록 생성 (30% 확률)"""
    if rng.random() > 0.3:
        return []

    doc_types = ["service_guide", "product_info", "faq", "notice"]
    docs = []

    num_docs = rng.randint(1, 3)
    for i in range(num_docs):
        doc_type = rng.choice(doc_types)
        docs.append({
            "step_number": i + 1,
            "doc_id": f"DOC-{rng.randint(1000, 9999)}",
            "doc_type": doc_type,
            "title": f"{category_main} 관련 {doc_type} 문서",
            "used": rng.choice([True, True, False])
        })

    return docs


def populate_extended_fields(conn: psycopg2_connection, batch_size: int = 500):
    """consultations 테이블의 확장 필드에 데이터 채우기"""
    print("\n" + "=" * 60)
    print("[2/4] Consultations 확장 필드 데이터 생성")
    print("=" * 60)

    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                c.id, c.customer_id, c.category_main, c.call_time, c.call_duration,
                c.processing_timeline, cust.name as customer_name
            FROM consultations c
            LEFT JOIN customers cust ON c.customer_id = cust.id
            ORDER BY c.id
        """)

        rows = cursor.fetchall()
        total = len(rows)
        print(f"[INFO] 총 상담 건수: {total}")

        rng = random.Random(RANDOM_SEED)
        print(f"[INFO] 랜덤 시드: {RANDOM_SEED}")

        update_query = """
            UPDATE consultations SET
                call_end_time = %s,
                acw_duration = %s,
                transcript = %s,
                ai_summary = %s,
                agent_notes = %s,
                follow_up_schedule = %s,
                transfer_department = %s,
                transfer_notes = %s,
                sentiment = %s,
                emotion_score = %s,
                satisfaction_score = %s,
                feedback_text = %s,
                feedback_emotions = %s,
                recording_file_path = %s,
                recording_duration = %s,
                recording_file_size = %s,
                referenced_documents = %s,
                updated_at = NOW()
            WHERE id = %s
        """

        update_batch = []
        stats = {'follow_up': 0, 'transfer': 0, 'feedback': 0}

        for i, row in enumerate(rows):
            consultation_id = row[0]
            customer_id = row[1]
            category_main = row[2] or "기타"
            call_time = row[3]
            call_duration = row[4]
            processing_timeline = row[5]
            customer_name = row[6] or "고객"

            call_end_time = calculate_call_end_time(call_time, call_duration)
            acw_duration = generate_acw_duration(rng)
            transcript = generate_transcript(category_main, customer_name, call_time, call_duration, rng)
            ai_summary = generate_ai_summary(category_main, rng)
            agent_notes = generate_agent_notes(rng)
            follow_up_schedule = generate_follow_up_schedule(rng)
            transfer_dept, transfer_notes = generate_transfer_info(rng)
            sentiment = generate_sentiment(rng)
            emotion_score = generate_emotion_score(sentiment, rng)
            satisfaction_score = generate_satisfaction_score(sentiment, rng)
            feedback_text = generate_feedback_text(sentiment, rng)
            feedback_emotions = generate_feedback_emotions(sentiment, rng)
            recording_path, recording_dur, recording_size = generate_recording_info(consultation_id, call_duration, rng)
            referenced_docs = generate_referenced_documents(category_main, rng)

            if follow_up_schedule:
                stats['follow_up'] += 1
            if transfer_dept:
                stats['transfer'] += 1
            if feedback_text:
                stats['feedback'] += 1

            update_batch.append((
                call_end_time,
                acw_duration,
                PsycopgJson(transcript),
                ai_summary,
                agent_notes,
                follow_up_schedule,
                transfer_dept,
                transfer_notes,
                sentiment,
                emotion_score,
                satisfaction_score,
                feedback_text,
                PsycopgJson(feedback_emotions),
                recording_path,
                recording_dur,
                recording_size,
                PsycopgJson(referenced_docs) if referenced_docs else None,
                consultation_id
            ))

            if len(update_batch) >= batch_size:
                execute_batch(cursor, update_query, update_batch, page_size=batch_size)
                conn.commit()
                print(f"[INFO] {i + 1}/{total} 처리 완료 ({(i + 1) / total * 100:.1f}%)")
                update_batch = []

        if update_batch:
            execute_batch(cursor, update_query, update_batch, page_size=len(update_batch))
            conn.commit()
            print(f"[INFO] {total}/{total} 처리 완료 (100%)")

        print(f"\n[INFO] 통계:")
        print(f"  - follow_up_schedule 있음: {stats['follow_up']}건 ({stats['follow_up']/total*100:.1f}%)")
        print(f"  - transfer_department 있음: {stats['transfer']}건 ({stats['transfer']/total*100:.1f}%)")
        print(f"  - feedback_text 있음: {stats['feedback']}건 ({stats['feedback']/total*100:.1f}%)")

        cursor.close()
        return True

    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] 확장 필드 데이터 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    from connect_db import connect_db

    conn = connect_db()
    if conn:
        populate_extended_fields(conn)
        conn.close()
