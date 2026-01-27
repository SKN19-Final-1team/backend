from fastapi import APIRouter
from app.api.v1.endpoints import websocket, followup, education

api_router = APIRouter()

# 웹소켓 라우터
api_router.include_router(websocket.router, tags=["websocket"])
api_router.include_router(followup.router, prefix="/followup", tags=["followup"])
api_router.include_router(education.router, prefix="/education", tags=["education"])