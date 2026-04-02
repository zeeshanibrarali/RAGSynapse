import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.models import ChatRequest, ChatResponse, SourceNode
from api.dependencies import get_chat_engine

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question about ingested documents.
    Returns answer with source citations and latency.
    """
    try:
        engine = get_chat_engine(
            provider=request.provider,
            model=request.model,
        )

        t0 = time.perf_counter()
        response = engine.chat(request.question)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract source nodes
        sources = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for node in response.source_nodes:
                sources.append(SourceNode(
                    source=node.metadata.get("source", "unknown"),
                    text_preview=node.text[:300],
                    score=float(node.score) if node.score else None,
                ))

        from ragsynapse.llm.model_factory import DEFAULT_MODELS, LLMProvider
        used_model = request.model or DEFAULT_MODELS.get(
            LLMProvider(request.provider), "unknown"
        )

        return ChatResponse(
            answer=str(response),
            provider=request.provider,
            model=used_model,
            sources=sources,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint — returns tokens as they generate.
    Use for real-time UI updates.
    """
    async def generate():
        try:
            engine = get_chat_engine(
                provider=request.provider,
                model=request.model,
            )
            response = engine.stream_chat(request.question)
            for token in response.response_gen:
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: ERROR: {e}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )