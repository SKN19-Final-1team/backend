"""
헬스체크 엔드포인트
배포 환경에서 서버 상태를 모니터링하기 위한 엔드포인트
"""
import os
from fastapi import APIRouter, HTTPException, status
from redis import Redis
from app.db.base import get_connection

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    기본 헬스 체크
    서버가 실행 중인지만 확인 (외부 의존성 체크 X)
    """
    return {
        "status": "healthy",
        "service": "CALL:ACT Backend",
        "version": "1.0.0"
    }


@router.get("/readiness")
async def readiness_check():
    """
    레디니스 체크 (배포 환경용)
    DB, Redis 연결 상태 확인
    실패 시 503 반환
    """
    checks = {
        "database": False,
        "redis": False,
    }
    errors = []

    # 1. 데이터베이스 연결 확인
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        checks["database"] = True
    except Exception as e:
        errors.append(f"Database: {str(e)}")

    # 2. Redis 연결 확인
    try:
        redis_url = os.getenv("DIALOGUE_REDIS_URL", "redis://localhost:6379/0")
        redis_client = Redis.from_url(redis_url, socket_connect_timeout=2)
        redis_client.ping()
        checks["redis"] = True
    except Exception as e:
        errors.append(f"Redis: {str(e)}")

    # 모든 체크가 성공하면 200, 하나라도 실패하면 503
    all_healthy = all(checks.values())

    response = {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks
    }

    if errors:
        response["errors"] = errors

    if not all_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response
        )

    return response


@router.get("/liveness")
async def liveness_check():
    """
    라이브니스 체크 (Kubernetes용)
    프로세스가 살아있는지만 확인
    """
    return {"status": "alive"}
