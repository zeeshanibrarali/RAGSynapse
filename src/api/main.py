"""
RAGSynapse FastAPI backend — v2
Production REST API for document intelligence.
Author: Zeeshan Ibrar

Endpoints:
  GET  /health              — system health check
  GET  /documents           — list stored documents
  POST /documents/upload    — ingest a new document
  POST /chat                — ask a question
  POST /chat/stream         — streaming answer
  POST /eval/run            — RAGAS evaluation
  GET  /eval/results        — fetch MLflow results
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, documents, eval
from api.models import HealthResponse

app = FastAPI(
    title="RAGSynapse API",
    description="Production RAG system with multi-model support",
    version="2.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc UI
)

# CORS — allow Streamlit and any React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(eval.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Check status of all connected services."""
    import redis as redis_lib
    import httpx

    # Redis
    try:
        r = redis_lib.Redis(host="redis", port=6379)
        r.ping()
        redis_status = "healthy"
    except Exception:
        redis_status = "unreachable"

    # Ollama
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
            )
            ollama_status = "healthy" if resp.status_code == 200 else "degraded"
    except Exception:
        ollama_status = "unreachable"

    # MLflow
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            )
            mlflow_status = "healthy" if resp.status_code == 200 else "degraded"
    except Exception:
        mlflow_status = "unreachable"

    overall = "healthy" if redis_status == "healthy" else "degraded"

    return HealthResponse(
        status=overall,
        redis=redis_status,
        ollama=ollama_status,
        mlflow=mlflow_status,
    )