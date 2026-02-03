"""
Simulation 데이터 생성 모듈

simulation_results 테이블의 점수 및 참조 데이터 생성:
- similarity_score: 유사도 점수 (0-100)
- keyword_match_score: 키워드 매칭 점수 (0-100)
- document_match_score: 문서 매칭 점수 (0-100)
- sequence_match_score: 시퀀스 매칭 점수 (0-100)
- time_comparison: 시간 비교 정보 (JSONB)
- ai_customer_reactions: AI 고객 반응 (JSONB)
- recording_file_path: 녹취 파일 경로
- recording_transcript: 녹취 전문 (JSONB)

멱등성: RANDOM_SEED=42로 동일한 결과 보장
"""

import random
from datetime import datetime

from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import execute_batch, Json as PsycopgJson


RANDOM_SEED = 42

# AI 고객 반응 템플릿
AI_REACTIONS = {
    "positive": [
        {"reaction": "satisfied", "message": "네, 잘 이해했습니다. 감사합니다."},
        {"reaction": "grateful", "message": "빠르게 처리해 주셔서 감사해요."},
        {"reaction": "relieved", "message": "다행이네요, 걱정했는데 해결됐네요."},
    ],
    "neutral": [
        {"reaction": "understanding", "message": "네, 알겠습니다."},
        {"reaction": "waiting", "message": "그럼 기다려 볼게요."},
        {"reaction": "confirming", "message": "그렇군요, 확인했습니다."},
    ],
    "negative": [
        {"reaction": "frustrated", "message": "왜 이렇게 복잡한 거죠?"},
        {"reaction": "impatient", "message": "좀 더 빨리 처리 안 되나요?"},
        {"reaction": "confused", "message": "잘 이해가 안 되는데요."},
    ],
}

# 한글 난이도를 영어로 매핑 (DB 값 → 내부 처리용)
DIFFICULTY_MAP = {
    "초급": "easy",
    "중급": "medium",
    "고급": "hard",
}

# 시나리오별 기본 난이도 (scenario_id → difficulty, fallback용)
SCENARIO_DIFFICULTY = {
    "SIM-001": "easy",
    "SIM-002": "medium",
    "SIM-003": "medium",
    "SIM-004": "hard",
    "SIM-005": "hard",
}


def generate_scores(difficulty: str, rng) -> dict:
    """난이도에 따른 점수 생성"""
    if difficulty == "easy":
        base = rng.randint(70, 90)
    elif difficulty == "medium":
        base = rng.randint(50, 80)
    else:  # hard
        base = rng.randint(30, 70)

    # 각 점수에 약간의 변동 추가
    return {
        "similarity": min(100, max(0, base + rng.randint(-10, 10))),
        "keyword_match": min(100, max(0, base + rng.randint(-15, 15))),
        "document_match": min(100, max(0, base + rng.randint(-10, 10))),
        "sequence_match": min(100, max(0, base + rng.randint(-20, 20))),
    }


def generate_time_comparison(total_score: int, rng) -> dict:
    """시간 비교 정보 생성"""
    # 원본 상담 시간 (3-10분)
    original_duration = rng.randint(180, 600)

    # 시뮬레이션 시간 (점수에 따라 변동)
    # 점수가 높으면 원본과 비슷, 낮으면 더 오래 걸림
    time_factor = 1 + (100 - total_score) / 100  # 1.0 ~ 2.0
    sim_duration = int(original_duration * time_factor * (0.8 + rng.random() * 0.4))

    return {
        "original_duration_sec": original_duration,
        "simulation_duration_sec": sim_duration,
        "time_difference_sec": sim_duration - original_duration,
        "efficiency_ratio": round(original_duration / sim_duration, 2) if sim_duration > 0 else 1.0
    }


def generate_ai_reactions(total_score: int, rng) -> list:
    """AI 고객 반응 목록 생성"""
    # 점수에 따라 반응 결정
    if total_score >= 70:
        reaction_pool = AI_REACTIONS["positive"] + AI_REACTIONS["neutral"]
    elif total_score >= 40:
        reaction_pool = AI_REACTIONS["neutral"] + AI_REACTIONS["negative"][:1]
    else:
        reaction_pool = AI_REACTIONS["negative"] + AI_REACTIONS["neutral"][:1]

    # 3-5개 반응 선택
    num_reactions = rng.randint(3, 5)
    reactions = []

    for i in range(num_reactions):
        reaction = rng.choice(reaction_pool).copy()
        reaction["step"] = i + 1
        reaction["timestamp"] = f"{rng.randint(0, 9)}:{rng.randint(0, 59):02d}"
        reactions.append(reaction)

    return reactions


def generate_recording_transcript(rng) -> dict:
    """녹취 전문 생성 (간단한 버전)"""
    messages = [
        {"speaker": "agent", "message": "안녕하세요, 시뮬레이션을 시작하겠습니다.", "timestamp": "00:00"},
        {"speaker": "ai_customer", "message": "네, 문의드릴 게 있어요.", "timestamp": "00:05"},
        {"speaker": "agent", "message": "네, 말씀해 주세요.", "timestamp": "00:10"},
        {"speaker": "ai_customer", "message": "카드 관련해서요.", "timestamp": "00:15"},
        {"speaker": "agent", "message": "어떤 부분이 궁금하신가요?", "timestamp": "00:20"},
    ]

    # 추가 대화 (3-7개)
    additional = rng.randint(3, 7)
    for i in range(additional):
        timestamp = f"{(i + 1) // 2}:{((i + 1) * 30) % 60:02d}"
        if i % 2 == 0:
            messages.append({
                "speaker": "ai_customer",
                "message": f"추가 질문 {i + 1}입니다.",
                "timestamp": timestamp
            })
        else:
            messages.append({
                "speaker": "agent",
                "message": f"답변 드리겠습니다.",
                "timestamp": timestamp
            })

    messages.append({
        "speaker": "agent",
        "message": "시뮬레이션을 종료합니다. 수고하셨습니다.",
        "timestamp": f"{len(messages) // 2}:00"
    })

    return {"messages": messages}


def populate_simulation_data(conn: psycopg2_connection, batch_size: int = 100):
    """simulation_results 테이블의 데이터 채우기"""
    print("\n" + "=" * 60)
    print("[4/4] Simulation 데이터 생성")
    print("=" * 60)

    cursor = conn.cursor()
    rng = random.Random(RANDOM_SEED)
    print(f"[INFO] 랜덤 시드: {RANDOM_SEED}")

    try:
        # 기존 데이터 조회 (difficulty 컬럼 사용)
        cursor.execute("""
            SELECT sr.id, sr.scenario_id, sr.employee_id, ss.difficulty
            FROM simulation_results sr
            LEFT JOIN simulation_scenarios ss ON sr.scenario_id = ss.id
            ORDER BY sr.id
        """)
        rows = cursor.fetchall()
        total = len(rows)
        print(f"[INFO] 총 시뮬레이션 결과 수: {total}")

        if total == 0:
            print("[INFO] 시뮬레이션 결과가 없습니다. 건너뜁니다.")
            cursor.close()
            return True

        update_query = """
            UPDATE simulation_results SET
                similarity_score = %s,
                keyword_match_score = %s,
                document_match_score = %s,
                sequence_match_score = %s,
                time_comparison = %s,
                ai_customer_reactions = %s,
                recording_file_path = %s,
                recording_transcript = %s
            WHERE id = %s
        """

        update_batch = []

        for row in rows:
            result_id = row[0]
            scenario_id = row[1]
            employee_id = row[2]
            difficulty_kr = row[3]  # '초급', '중급', '고급' (한글)

            # 한글 난이도를 영어로 변환, 없으면 시나리오 기본값 사용
            difficulty = DIFFICULTY_MAP.get(difficulty_kr) or SCENARIO_DIFFICULTY.get(scenario_id, "medium")

            # 점수 생성
            scores = generate_scores(difficulty, rng)
            total_score = (scores["similarity"] + scores["keyword_match"] +
                           scores["document_match"] + scores["sequence_match"]) // 4

            # 시간 비교 정보
            time_comparison = generate_time_comparison(total_score, rng)

            # AI 반응
            ai_reactions = generate_ai_reactions(total_score, rng)

            # 녹취 정보
            recording_path = f"/simulations/{result_id}_recording.wav"
            recording_transcript = generate_recording_transcript(rng)

            update_batch.append((
                scores["similarity"],
                scores["keyword_match"],
                scores["document_match"],
                scores["sequence_match"],
                PsycopgJson(time_comparison),
                PsycopgJson(ai_reactions),
                recording_path,
                PsycopgJson(recording_transcript),
                result_id
            ))

        if update_batch:
            execute_batch(cursor, update_query, update_batch, page_size=batch_size)
            conn.commit()

        print(f"[INFO] {total}건 업데이트 완료")

        # 통계 확인
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                AVG(similarity_score)::int as avg_similarity,
                AVG(keyword_match_score)::int as avg_keyword,
                COUNT(recording_file_path) as has_recording
            FROM simulation_results
        """)
        stats = cursor.fetchone()
        print(f"[INFO] 통계: avg_similarity={stats[1]}, avg_keyword={stats[2]}, recordings={stats[3]}/{stats[0]}")

        cursor.close()
        return True

    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] simulation 데이터 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    from connect_db import connect_db

    conn = connect_db()
    if conn:
        populate_simulation_data(conn)
        conn.close()
