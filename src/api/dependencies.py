"""
FastAPI dependency injection — pipeline and LLM initialization.
Uses functools.lru_cache instead of st.cache_resource.
Author: Zeeshan Ibrar
"""

import os
import toml
from pathlib import Path
from functools import lru_cache

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core import VectorStoreIndex

# Load config
current_dir = Path(__file__).parent
config_path = current_dir / ".." / ".." / "config.toml"
with open(config_path, "r") as f:
    params = toml.load(f)


@lru_cache(maxsize=1)
def get_embed_model() -> HuggingFaceEmbedding:
    """Cached embedding model — loads once, reused on every request."""
    return HuggingFaceEmbedding(
        model_name=params["embed_model"]["model_name"],
        cache_folder=params["embed_model"]["cache_folder"],
        embed_batch_size=params["embed_model"]["embed_batch_size"],
    )


@lru_cache(maxsize=1)
def get_vector_store() -> RedisVectorStore:
    """Cached Redis vector store connection."""
    return RedisVectorStore(
        index_name=params["redis"]["vector_index_name"],
        index_prefix=params["redis"]["vector_index_prefix"],
        redis_url="redis://" + params["redis"]["host_name"] + ":" + str(params["redis"]["port_no"]),
        metadata_fields=["source", "page_num"],
    )


@lru_cache(maxsize=1)
def get_doc_store() -> RedisDocumentStore:
    """Cached Redis document store connection."""
    return RedisDocumentStore.from_host_and_port(
        params["redis"]["host_name"],
        params["redis"]["port_no"],
        namespace=params["redis"]["doc_store_name"],
    )


def get_ingestion_pipeline() -> IngestionPipeline:
    """Returns a fresh ingestion pipeline (not cached — stateful)."""
    embed_model = get_embed_model()
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=256, chunk_overlap=50),
            embed_model,
        ],
        docstore=get_doc_store(),
        vector_store=get_vector_store(),
        cache=IngestionCache(
            cache=RedisCache.from_host_and_port(
                params["redis"]["host_name"],
                params["redis"]["port_no"],
            ),
            collection=params["redis"]["cache_name"],
        ),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )


def get_query_engine(provider: str = "ollama", model: str = None):
    """Returns a LlamaIndex query engine with selected LLM."""
    import sys
    sys.path.insert(0, str(current_dir / ".."))
    from ragsynapse.llm.model_factory import get_llm

    llm = get_llm(provider=provider, model=model)
    return VectorStoreIndex.from_vector_store(
        get_vector_store(),
        embed_model=get_embed_model(),
    ).as_query_engine(
        llm=llm,
        similarity_top_k=3,
    )


def get_chat_engine(provider: str = "ollama", model: str = None):
    """Returns a LlamaIndex chat engine with selected LLM."""
    import sys
    sys.path.insert(0, str(current_dir / ".."))
    from ragsynapse.llm.model_factory import get_llm

    llm = get_llm(provider=provider, model=model)
    return VectorStoreIndex.from_vector_store(
        get_vector_store(),
        embed_model=get_embed_model(),
    ).as_chat_engine(
        llm=llm,
        chat_mode="condense_question",
        similarity_top_k=3,
    )