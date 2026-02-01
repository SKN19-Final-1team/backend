"""
교육 시뮬레이션 결과 관리 모듈

시뮬레이션 결과를 DB에 저장하고 학습 분석 데이터를 업데이트합니다.
"""

import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.db.base import get_connection


def save_simulation_result(
    employee_id: str,
    simulation_type: str,
    original_consultation_id: Optional[str],
    scenario_id: Optional[str],
    scores: Dict[str, Any],
    feedback_data: Dict[str, Any],
    call_duration: int,
    recording_transcript: str,
    ai_customer_reactions: Optional[List[Dict]] = None
) -> str:
    """
    시뮬레이션 결과를 DB에 저장

    Args:
        employee_id: 직원 ID
        simulation_type: "best_practice" 또는 "scenario"
        original_consultation_id: 원본 상담 ID (best_practice일 때)
        scenario_id: 시나리오 ID (scenario일 때)
        scores: 평가 점수 딕셔너리
        feedback_data: 피드백 데이터
        call_duration: 통화 시간 (초)
        recording_transcript: 녹취 전문
        ai_customer_reactions: AI 고객 반응 기록

    Returns:
        생성된 결과 ID
    """
    conn = get_connection()
    result_id = str(uuid.uuid4())
    simulation_consultation_id = f"SIM-{employee_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    try:
        with conn.cursor() as cur:
            query = """
                INSERT INTO simulation_results (
                    id,
                    employee_id,
                    simulation_type,
                    original_consultation_id,
                    scenario_id,
                    simulation_consultation_id,
                    overall_score,
                    similarity_score,
                    keyword_match_score,
                    document_match_score,
                    sequence_match_score,
                    time_comparison,
                    feedback_data,
                    call_duration,
                    recording_transcript,
                    ai_customer_reactions,
                    created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                )
            """

            # 점수 추출
            overall_score = scores.get("overall_score", 0)
            similarity_score = scores.get("similarity_score")
            keyword_score = scores.get("keyword_score", 0)
            sequence_score = scores.get("sequence_score", 0)

            # 시간 비교 데이터
            time_comparison = json.dumps({
                "actual_duration": call_duration,
                "recorded_at": datetime.now().isoformat()
            })

            cur.execute(query, (
                result_id,
                employee_id,
                simulation_type,
                original_consultation_id,
                scenario_id,
                simulation_consultation_id,
                overall_score,
                similarity_score,
                keyword_score,
                0,  # document_match_score (미구현)
                sequence_score,
                time_comparison,
                json.dumps(feedback_data, ensure_ascii=False),
                call_duration,
                recording_transcript,
                json.dumps(ai_customer_reactions or [], ensure_ascii=False)
            ))

            conn.commit()
            print(f"[ResultManager] 시뮬레이션 결과 저장 완료: {result_id}")

            # 학습 분석 업데이트
            update_learning_analytics(employee_id, conn)

            return result_id

    except Exception as e:
        conn.rollback()
        print(f"[ResultManager] 결과 저장 실패: {e}")
        raise
    finally:
        conn.close()


def update_learning_analytics(employee_id: str, conn=None) -> None:
    """
    직원의 학습 분석 데이터 업데이트

    Args:
        employee_id: 직원 ID
        conn: DB 연결 (없으면 새로 생성)
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True

    try:
        with conn.cursor() as cur:
            # 해당 직원의 모든 시뮬레이션 결과 조회
            cur.execute("""
                SELECT
                    overall_score,
                    simulation_type,
                    scenario_id,
                    created_at
                FROM simulation_results
                WHERE employee_id = %s
                ORDER BY created_at DESC
            """, (employee_id,))

            results = cur.fetchall()

            if not results:
                return

            # 통계 계산
            total_simulations = len(results)
            scores = [r[0] for r in results if r[0] is not None]
            average_score = sum(scores) / len(scores) if scores else 0
            passed_count = sum(1 for s in scores if s >= 70)
            pass_rate = (passed_count / len(scores) * 100) if scores else 0

            # 최근 10개 vs 이전 10개 비교로 개선율 계산
            recent_scores = scores[:10]
            older_scores = scores[10:20] if len(scores) > 10 else []

            improvement_rate = 0
            if recent_scores and older_scores:
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)
                if older_avg > 0:
                    improvement_rate = ((recent_avg - older_avg) / older_avg) * 100

            # 카테고리별 성과 (시나리오 ID 기준)
            category_performance = {}
            for r in results:
                scenario_id = r[2]
                score = r[0]
                if scenario_id and score is not None:
                    if scenario_id not in category_performance:
                        category_performance[scenario_id] = {"scores": [], "count": 0}
                    category_performance[scenario_id]["scores"].append(score)
                    category_performance[scenario_id]["count"] += 1

            # 평균 계산
            for k, v in category_performance.items():
                v["average"] = sum(v["scores"]) / len(v["scores"]) if v["scores"] else 0
                del v["scores"]

            # 강점/약점 분석
            sorted_categories = sorted(
                category_performance.items(),
                key=lambda x: x[1].get("average", 0),
                reverse=True
            )
            strengths = [k for k, v in sorted_categories[:3] if v.get("average", 0) >= 70]
            weaknesses = [k for k, v in sorted_categories[-3:] if v.get("average", 0) < 70]

            # 완료한 시나리오 목록
            completed_scenarios = list(set(r[2] for r in results if r[2]))

            # UPSERT (있으면 업데이트, 없으면 삽입)
            cur.execute("""
                INSERT INTO employee_learning_analytics (
                    employee_id,
                    total_simulations,
                    average_score,
                    pass_rate,
                    improvement_rate,
                    strengths,
                    weaknesses,
                    category_performance,
                    completed_scenarios,
                    last_simulation_date,
                    updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                )
                ON CONFLICT (employee_id) DO UPDATE SET
                    total_simulations = EXCLUDED.total_simulations,
                    average_score = EXCLUDED.average_score,
                    pass_rate = EXCLUDED.pass_rate,
                    improvement_rate = EXCLUDED.improvement_rate,
                    strengths = EXCLUDED.strengths,
                    weaknesses = EXCLUDED.weaknesses,
                    category_performance = EXCLUDED.category_performance,
                    completed_scenarios = EXCLUDED.completed_scenarios,
                    last_simulation_date = NOW(),
                    updated_at = NOW()
            """, (
                employee_id,
                total_simulations,
                round(average_score, 2),
                round(pass_rate, 2),
                round(improvement_rate, 2),
                json.dumps(strengths, ensure_ascii=False),
                json.dumps(weaknesses, ensure_ascii=False),
                json.dumps(category_performance, ensure_ascii=False),
                completed_scenarios
            ))

            conn.commit()
            print(f"[ResultManager] 학습 분석 업데이트 완료: {employee_id}")

    except Exception as e:
        print(f"[ResultManager] 학습 분석 업데이트 실패: {e}")
        # 에러가 발생해도 메인 저장은 완료되었으므로 raise하지 않음
    finally:
        if should_close:
            conn.close()


def get_employee_analytics(employee_id: str) -> Optional[Dict[str, Any]]:
    """
    직원의 학습 분석 데이터 조회

    Args:
        employee_id: 직원 ID

    Returns:
        학습 분석 데이터 딕셔너리
    """
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    total_simulations,
                    average_score,
                    pass_rate,
                    improvement_rate,
                    strengths,
                    weaknesses,
                    category_performance,
                    completed_scenarios,
                    last_simulation_date,
                    updated_at
                FROM employee_learning_analytics
                WHERE employee_id = %s
            """, (employee_id,))

            row = cur.fetchone()

            if not row:
                return None

            return {
                "employee_id": employee_id,
                "total_simulations": row[0],
                "average_score": float(row[1]) if row[1] else 0,
                "pass_rate": float(row[2]) if row[2] else 0,
                "improvement_rate": float(row[3]) if row[3] else 0,
                "strengths": json.loads(row[4]) if row[4] else [],
                "weaknesses": json.loads(row[5]) if row[5] else [],
                "category_performance": json.loads(row[6]) if row[6] else {},
                "completed_scenarios": row[7] or [],
                "last_simulation_date": row[8].isoformat() if row[8] else None,
                "updated_at": row[9].isoformat() if row[9] else None
            }

    except Exception as e:
        print(f"[ResultManager] 학습 분석 조회 실패: {e}")
        return None
    finally:
        conn.close()


def get_simulation_history(
    employee_id: str,
    limit: int = 10,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    직원의 시뮬레이션 히스토리 조회

    Args:
        employee_id: 직원 ID
        limit: 조회 개수
        offset: 시작 위치

    Returns:
        시뮬레이션 결과 리스트
    """
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    simulation_type,
                    scenario_id,
                    overall_score,
                    similarity_score,
                    keyword_match_score,
                    sequence_match_score,
                    call_duration,
                    created_at
                FROM simulation_results
                WHERE employee_id = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, (employee_id, limit, offset))

            rows = cur.fetchall()

            return [{
                "id": row[0],
                "simulation_type": row[1],
                "scenario_id": row[2],
                "overall_score": row[3],
                "similarity_score": row[4],
                "keyword_match_score": row[5],
                "sequence_match_score": row[6],
                "call_duration": row[7],
                "created_at": row[8].isoformat() if row[8] else None
            } for row in rows]

    except Exception as e:
        print(f"[ResultManager] 히스토리 조회 실패: {e}")
        return []
    finally:
        conn.close()
