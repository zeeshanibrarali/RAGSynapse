import os
from llama_index.core.indices.base import BaseIndex
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ragsynapse.llm.model_factory import get_llm


def get_conversation_engine(
    embed_model: HuggingFaceEmbedding,
    vector_store: RedisVectorStore,
    provider: str = None,
    model: str = None,
) -> BaseIndex.as_chat_engine:
    """
    Initialize and return a Chat engine using the provided embed model and vector store.

    Args:
    - embed_model: HuggingFace embedding model for query encoding
    - vector_store: Redis vector store from the ingestion pipeline
    - provider: LLM provider — 'openai', 'anthropic', or 'ollama'
                Defaults to LLM_PROVIDER env var (falls back to 'openai')
    - model: Specific model name. If None, uses provider default.

    Returns:
    - Chat engine with selected LLM backend
    """
    llm = get_llm(provider=provider, model=model)

    chat_engine = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    ).as_chat_engine(
    llm=llm,
    chat_mode="condense_question",   # faster than default for CPU
    verbose=False,
    )

    return chat_engine