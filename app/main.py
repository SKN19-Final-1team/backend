from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.v1.routers import api_router
from app.llm.delivery.keyword_extractor import warmup

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시 워밍업 실행 (형태소 분석기 로드 등)
    # 서버 실행 중 계속 유지되어야 하는 리소스 초기화를 여기서 수행합니다.
    warmup()
    yield
    # 애플리케이션 종료 시 정리 작업 (필요 시)

app = FastAPI(
    title="CALL:ACT",
    description="API documentation",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")