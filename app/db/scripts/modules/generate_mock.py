"""
Mock 데이터 생성 모듈

시뮬레이션 교육 Mock 데이터, 감사 로그 Mock 데이터 생성
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from psycopg2.extensions import connection as psycopg2_connection

from . import check_table_has_data


def generate_mock_simulation_data(conn: psycopg2_connection):
    """시뮬레이션 교육 Mock 데이터 생성"""
    print("\n" + "=" * 60)
    print("[9/12] 시뮬레이션 교육 Mock 데이터 생성")
    print("=" * 60)

    random.seed(42)

    # 이미 데이터가 있는지 확인
    has_data, count = check_table_has_data(conn, "simulation_results")
    if has_data:
        print(f"[INFO] simulation_results 테이블에 이미 데이터가 있습니다. ({count}건) - 스킵")
        return True

    cursor = conn.cursor()

    try:
        # 상담사 목록 조회 (전원)
        cursor.execute("""
            SELECT id FROM employees
            WHERE id != 'EMP-TEDDY-DEFAULT' AND status = 'active'
            ORDER BY id
        """)
        employees = [row[0] for row in cursor.fetchall()]

        if not employees:
            print("[WARNING] 상담사 데이터가 없어 Mock 데이터 생성을 건너뜉니다.")
            cursor.close()
            return True

        # 시나리오 목록 조회 (10_setup_simulation_tables.sql에서 이미 생성됨)
        cursor.execute("SELECT id FROM simulation_scenarios LIMIT 10")
        scenarios = [row[0] for row in cursor.fetchall()]

        if not scenarios:
            print("[WARNING] 시뮬레이션 시나리오가 없어 Mock 데이터 생성을 건너뜉니다.")
            cursor.close()
            return True

        print(f"[INFO] 사용 가능한 시나리오: {len(scenarios)}개")

        # simulation_results Mock 데이터 생성 (실제 스키마에 맞춤)
        print("[INFO] simulation_results Mock 데이터 생성 중...")
        result_batch = []
        base_date = datetime(2026, 1, 1)  # 고정 기준일 (재현성)

        for emp_id in employees:
            # 각 상담사별 1~3개의 시뮬레이션 결과
            num_results = random.randint(1, 3)
            for i in range(num_results):
                scenario_id = random.choice(scenarios)
                score = random.randint(60, 100)
                duration = random.randint(180, 600)  # 3~10분
                passed = score >= 70
                attempt_date = base_date + timedelta(days=random.randint(0, 30))
                sim_consult_id = f"SIM-{emp_id}-{scenario_id}-{i+1:02d}"

                feedback = {
                    "strengths": ["적절한 인사", "문제 파악", "해결책 제시"] if score >= 80 else ["적절한 인사"],
                    "improvements": [] if score >= 80 else ["더 빠른 해결책 제시 필요"],
                    "expert_tips": ["고객 공감 표현을 더 자주 사용하세요"]
                }

                result_batch.append((
                    emp_id,
                    'scenario',  # simulation_type
                    scenario_id,
                    sim_consult_id,  # simulation_consultation_id (unique)
                    score,  # overall_score
                    passed,
                    random.randint(70, 95),  # objective_completion_rate
                    random.randint(60, 100),  # document_usage_score
                    random.randint(60, 100),  # keyword_coverage_score
                    random.randint(60, 100),  # sequence_correctness_score
                    random.randint(70, 100),  # customer_satisfaction_score
                    json.dumps(feedback),  # feedback_data
                    duration,  # call_duration
                    attempt_date,  # call_started_at
                    attempt_date + timedelta(seconds=duration)  # call_ended_at
                ))

        if result_batch:
            cursor.executemany("""
                INSERT INTO simulation_results (
                    employee_id, simulation_type, scenario_id, simulation_consultation_id,
                    overall_score, passed, objective_completion_rate, document_usage_score,
                    keyword_coverage_score, sequence_correctness_score, customer_satisfaction_score,
                    feedback_data, call_duration, call_started_at, call_ended_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
            """, result_batch)
            conn.commit()
            print(f"[INFO] simulation_results 생성: {len(result_batch)}건")

        # employee_learning_analytics Mock 데이터 생성 (전원 + 전필드)
        print("[INFO] employee_learning_analytics Mock 데이터 생성 중 (전원)...")
        analytics_batch = []

        for emp_id in employees:
            total_sims = random.randint(3, 15)
            total_bp = random.randint(0, total_sims // 2)
            total_scenario = total_sims - total_bp
            avg_score = random.uniform(70.0, 95.0)
            avg_similarity = random.uniform(65.0, 95.0)
            avg_obj_completion = random.uniform(70.0, 98.0)
            pass_rate = random.uniform(60.0, 100.0)
            bp_pass_rate = random.uniform(55.0, 100.0) if total_bp > 0 else 0.0
            scenario_pass_rate = random.uniform(60.0, 100.0) if total_scenario > 0 else 0.0
            improvement = random.uniform(-5.0, 25.0)

            strengths = [
                {"skill": "문서 활용", "score": random.randint(70, 100)},
                {"skill": "키워드 사용", "score": random.randint(70, 100)}
            ]
            weaknesses = [
                {"skill": "고객 공감", "score": random.randint(50, 70)}
            ] if avg_score < 85 else []

            category_perf = {
                "카드분실": {"attempts": random.randint(1, 5), "avg_score": random.randint(70, 95), "pass_rate": random.randint(60, 100), "best_score": random.randint(80, 100)},
                "결제취소": {"attempts": random.randint(1, 3), "avg_score": random.randint(65, 90), "pass_rate": random.randint(55, 95), "best_score": random.randint(75, 100)}
            }

            completed = sorted(random.sample(
                [s for s in scenarios],
                k=min(random.randint(1, len(scenarios)), len(scenarios))
            ))
            unlocked = sorted(set(completed) | set(random.sample(
                [s for s in scenarios],
                k=min(len(completed) + random.randint(0, 2), len(scenarios))
            )))

            last_sim_date = base_date + timedelta(days=random.randint(0, 30))
            last_sim_score = random.randint(65, 100)

            analytics_batch.append((
                emp_id,
                total_sims,  # total_simulations
                total_bp,  # total_best_practice_simulations
                total_scenario,  # total_scenario_simulations
                round(avg_score, 2),  # average_score
                round(avg_similarity, 2),  # average_similarity_score
                round(avg_obj_completion, 2),  # average_objective_completion
                round(pass_rate, 2),  # pass_rate
                round(bp_pass_rate, 2),  # best_practice_pass_rate
                round(scenario_pass_rate, 2),  # scenario_pass_rate
                round(improvement, 2),  # improvement_rate
                json.dumps(strengths),  # strengths
                json.dumps(weaknesses),  # weaknesses
                json.dumps(category_perf),  # category_performance
                completed,  # completed_scenarios
                unlocked,  # unlocked_scenarios
                random.randint(1800, 7200),  # total_learning_time_seconds
                last_sim_date,  # last_simulation_at
                last_sim_score,  # last_simulation_score
            ))

        if analytics_batch:
            cursor.executemany("""
                INSERT INTO employee_learning_analytics (
                    employee_id, total_simulations, total_best_practice_simulations,
                    total_scenario_simulations, average_score, average_similarity_score,
                    average_objective_completion, pass_rate, best_practice_pass_rate,
                    scenario_pass_rate, improvement_rate,
                    strengths, weaknesses, category_performance,
                    completed_scenarios, unlocked_scenarios,
                    total_learning_time_seconds, last_simulation_at, last_simulation_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                          %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
                ON CONFLICT (employee_id) DO NOTHING
            """, analytics_batch)
            conn.commit()
            print(f"[INFO] employee_learning_analytics 생성: {len(analytics_batch)}건")

        cursor.close()
        return True

    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] 시뮬레이션 Mock 데이터 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_mock_audit_data(conn: psycopg2_connection):
    """감사 로그 Mock 데이터 생성"""
    print("\n" + "=" * 60)
    print("[10/12] 감사 로그 Mock 데이터 생성")
    print("=" * 60)

    random.seed(43)

    # 이미 데이터가 있는지 확인
    has_data, count = check_table_has_data(conn, "recording_download_logs")
    if has_data:
        print(f"[INFO] recording_download_logs 테이블에 이미 데이터가 있습니다. ({count}건) - 스킵")
        return True

    cursor = conn.cursor()

    try:
        # 상담사 및 상담 목록 조회
        cursor.execute("""
            SELECT e.id as emp_id, c.id as consult_id
            FROM employees e
            CROSS JOIN LATERAL (
                SELECT id FROM consultations WHERE agent_id = e.id LIMIT 3
            ) c
            WHERE e.id != 'EMP-TEDDY-DEFAULT' AND e.status = 'active'
            LIMIT 100
        """)
        emp_consult_pairs = cursor.fetchall()

        if not emp_consult_pairs:
            print("[WARNING] 상담사-상담 매핑 데이터가 없어 Mock 데이터 생성을 건너뜉니다.")
            cursor.close()
            return True

        # recording_download_logs Mock 데이터 생성
        print("[INFO] recording_download_logs Mock 데이터 생성 중...")
        download_batch = []
        base_date = datetime(2026, 1, 10)  # 고정 기준일 (재현성)

        for emp_id, consult_id in emp_consult_pairs:
            download_type = random.choice(['txt', 'wav', 'mp3'])
            download_date = base_date + timedelta(days=random.randint(0, 14), hours=random.randint(9, 18))

            download_batch.append((
                consult_id,
                emp_id,
                download_type,
                f"192.168.1.{random.randint(10, 250)}",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
                f"{consult_id}_recording.{download_type}",
                f"/recordings/{consult_id[:20]}/{consult_id}_recording.{download_type}",
                random.randint(50000, 5000000),
                download_date
            ))

        if download_batch:
            cursor.executemany("""
                INSERT INTO recording_download_logs (consultation_id, downloaded_by, download_type, download_ip, download_user_agent, file_name, file_path, file_size, downloaded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, download_batch)
            conn.commit()
            print(f"[INFO] recording_download_logs 생성: {len(download_batch)}건")

        # audit_logs Mock 데이터 생성
        print("[INFO] audit_logs Mock 데이터 생성 중...")
        audit_batch = []
        actions = ['RECORDING_DOWNLOAD', 'CUSTOMER_VIEW', 'CONSULTATION_VIEW', 'LOGIN', 'LOGOUT']

        for emp_id, consult_id in emp_consult_pairs[:50]:
            action = random.choice(actions)
            resource_type = 'consultation' if 'CONSULT' in action or 'RECORDING' in action else 'customer'
            audit_date = base_date + timedelta(days=random.randint(0, 14), hours=random.randint(9, 18))

            audit_batch.append((
                emp_id,
                action,
                resource_type,
                consult_id,
                f"192.168.1.{random.randint(10, 250)}",
                "Mozilla/5.0 Chrome/120.0.0.0",
                '{"source": "web_app"}',
                audit_date
            ))

        if audit_batch:
            cursor.executemany("""
                INSERT INTO audit_logs (user_id, action, resource_type, resource_id, ip_address, user_agent, details, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            """, audit_batch)
            conn.commit()
            print(f"[INFO] audit_logs 생성: {len(audit_batch)}건")

        cursor.close()
        return True

    except Exception as e:
        conn.rollback()
        cursor.close()
        print(f"[ERROR] 감사 로그 Mock 데이터 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
