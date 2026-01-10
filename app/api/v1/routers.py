from fastapi import APIRouter
from api.v1.endpoints import websocket

api_router = APIRouter()

# 웹소켓 라우터
api_router.include_router(websocket.router, tags=["websocket"])
api_router.include_router(rag.router, tags=["rag"])
