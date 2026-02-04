from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.routers import api_router
from app.llm.delivery.keyword_extractor import warmup

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시 워밍업 실행 (형태소 분석기 로드 등)
    # 서버 실행 중 계속 유지되어야 하는 리소스 초기화를 여기서 수행합니다.
    warmup(silent=True)
    yield
    # 애플리케이션 종료 시 정리 작업 (필요 시)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://callact.vercel.app"
]

app = FastAPI(
    title="CALL:ACT",
    description="API documentation",
    version="1.0.0",
    lifespan=lifespan,
    redirect_slashes=False,  # trailing slash redirect 비활성화 - CORS 문제 방지
)

# CORS 설정 - Frontend 개발 서버 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 퍼블릭 IP 개방을 위해 모든 origin 허용
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")