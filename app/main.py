from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.v1.routers import api_router
from app.llm.follow_up.summarizer import SLLMSummarizer

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 로컬 서버 시작 시 통신용 객체 생성하여 state에 저장
    app.state.summarizer = SLLMSummarizer()
    yield
    # 종료 시 정리 (필요한 경우)
    del app.state.summarizer

app = FastAPI(
    title="CALL:ACT",
    description="API documentation",
    version="1.0.0",
    lifespan=lifespan # lifespan 등록
)

app.include_router(api_router, prefix="/api/v1")