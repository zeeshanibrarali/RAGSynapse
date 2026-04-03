import os
from llama_index.core.indices.base import BaseIndex
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ragsynapse.llm.model_factory import get_llm


def get_conversation_engine(
    embed_model: HuggingFaceEmbedding,
    vector_store: RedisVectorStore,
    provider: str = None,
    model: str = None,
    active_document: str = None,
) -> BaseIndex.as_chat_engine:
    """
    Initialize chat engine with optional document filter.
    When active_document is set, only chunks from that file are retrieved.
    """
    llm = get_llm(provider=provider, model=model)

    # Build metadata filter if a specific document is selected
    filters = None
    if active_document:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="source",
                    value=active_document,
                    operator=FilterOperator.EQ,
                )
            ]
        )

    chat_engine = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    ).as_chat_engine(
        llm=llm,
        chat_mode="condense_question",
        verbose=True,
        similarity_top_k=3,
        filters=filters,
    )

    return chat_engine