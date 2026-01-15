from fastapi import FastAPI
from app.api.v1.routers import api_router

app = FastAPI(
    title = "CALL:ACT",
    description = "API documentation",
    version = "1.0.0",
)

app.include_router(api_router, prefix="/api/v1")
