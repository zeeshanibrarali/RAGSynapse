"""
Pydantic models for RAGSynapse FastAPI.
Author: Zeeshan Ibrar
"""
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    provider: str = Field(default="ollama")
    model: Optional[str] = None
    session_id: Optional[str] = None


class SourceNode(BaseModel):
    source: str
    text_preview: str
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    provider: str
    model: str
    sources: list[SourceNode]
    latency_ms: float


class DocumentInfo(BaseModel):
    filename: str
    doc_hash: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class UploadResponse(BaseModel):
    status: str
    filename: str
    nodes_ingested: int
    message: str


class EvalRequest(BaseModel):
    questions: list[str]
    ground_truths: list[str]
    run_name: Optional[str] = None
    provider: str = Field(default="ollama")


class EvalResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    overall_score: float
    num_questions: int
    run_id: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    redis: str
    ollama: str
    mlflow: str