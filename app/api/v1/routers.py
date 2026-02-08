from fastapi import APIRouter
from app.api.v1.endpoints import call_websocket, followup, education, edu_websocket, rag_frontend, customers, notices, employees, consultations, frequent_inquiries, health

api_router = APIRouter()

# 웹소켓 라우터
api_router.include_router(call_websocket.router, tags=["websocket"])
api_router.include_router(edu_websocket.router, tags=["websocket"])
api_router.include_router(rag_frontend.router, prefix="/rag", tags=["rag"])
api_router.include_router(followup.router, prefix="/followup", tags=["followup"])
api_router.include_router(education.router, prefix="/education", tags=["education"])

# 고객 정보 API
api_router.include_router(customers.router, prefix="/customers", tags=["customers"])

# 공지사항 API
api_router.include_router(notices.router, prefix="/notices", tags=["notices"])

# 상담사 API
api_router.include_router(employees.router, prefix="/employees", tags=["employees"])

# 상담 저장 API
api_router.include_router(consultations.router, prefix="/consultations", tags=["consultations"])

# 자주 찾는 문의 API
api_router.include_router(frequent_inquiries.router, prefix="/frequent-inquiries", tags=["frequent-inquiries"])

# 헬스체크
api_router.include_router(health.router, tags=["health"])
