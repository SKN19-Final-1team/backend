from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.v1.routers import api_router
from app.llm.delivery.keyword_extractor import warmup
from app.rag.retriever.db import warmup_embed_cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시 워밍업 실행
    warmup(silent=True)  # 형태소 분석기 로드
    warmup_embed_cache()  # 자주 쓰는 쿼리 임베딩 사전 캐싱
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

# TTS 오디오 파일 정적 서빙
app.mount("/static", StaticFiles(directory="app/llm/education"), name="static")