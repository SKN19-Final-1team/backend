from fastapi import APIRouter
from app.api.v1.endpoints import call_websocket, followup, education, edu_websocket, rag_frontend, customers

api_router = APIRouter()

# 웹소켓 라우터
api_router.include_router(call_websocket.router, tags=["websocket"])
api_router.include_router(edu_websocket.router, tags=["websocket"])
api_router.include_router(rag_frontend.router, prefix="/rag", tags=["rag"])
api_router.include_router(followup.router, prefix="/followup", tags=["followup"])
api_router.include_router(education.router, prefix="/education", tags=["education"])

# 고객 정보 API
api_router.include_router(customers.router, prefix="/customers", tags=["customers"])
