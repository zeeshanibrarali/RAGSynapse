import toml
import os
from pathlib import Path
import streamlit as st
import json
import redis as redis_client

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

current_dir = Path(__file__).parent
config_path = current_dir / '..' / '..' / '..' / 'config.toml'
with open(config_path, 'r') as f:
    params = toml.load(f)


def get_text_nodes(documents, pipeline):
    """
    Run documents through the ingestion pipeline.
    Returns list of nodes or None on failure.
    """
    try:
        nodes = pipeline.run(documents=documents, show_progress=True)
        return nodes
    except Exception as e:
        print(f"Pipeline error: {e}")
        return None

def get_stored_documents() -> list[str]:
    """
    Query Redis docstore and return list of unique source filenames
    already ingested. Keys in DocStore_v1/metadata are the filenames directly.
    """
    try:
        import redis as redis_lib
        import json

        r = redis_lib.Redis(
            host=params['redis']['host_name'],
            port=params['redis']['port_no'],
            decode_responses=True
        )

        # Metadata is stored as a hash — keys are filenames
        metadata_key = f"{params['redis']['doc_store_name']}/metadata"
        all_fields = r.hgetall(metadata_key)

        if not all_fields:
            return []

        # Keys are the filenames — e.g. "Python certificate.pdf"
        filenames = sorted([
            fname for fname in all_fields.keys()
            if fname.endswith(('.pdf', '.docx', '.txt'))
        ])

        return filenames

    except Exception as e:
        print(f"Could not fetch stored documents: {e}")
        return []

@st.cache_resource
def get_pipeline() -> dict:
    """
    Initialize embedding model and ingestion pipeline.
    Uses SentenceSplitter for reliable chunking across all document sizes.
    
    RAGSynapse v2 changes vs original:
    - Replaced SemanticSplitterNodeParser (produced 1-2 chunks) with
      SentenceSplitter (produces 10-50+ chunks per document)
    - Reduced chunk_size to 256 for better retrieval granularity
    - Added chunk_overlap of 50 to preserve context at boundaries
    """

    embed_model = HuggingFaceEmbedding(
        model_name=params['embed_model']['model_name'],
        cache_folder=params['embed_model']['cache_folder'],
        embed_batch_size=params['embed_model']['embed_batch_size']
    )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=256,    # small chunks = more of them = better retrieval
                chunk_overlap=50,  # overlap preserves context at boundaries
            ),
            embed_model,
        ],

        docstore=RedisDocumentStore.from_host_and_port(
            params['redis']['host_name'],
            params['redis']['port_no'],
            namespace=params['redis']['doc_store_name']
        ),

        vector_store=RedisVectorStore(
            index_name=params['redis']['vector_index_name'],
            index_prefix=params['redis']['vector_index_prefix'],
            redis_url="redis://" + params['redis']['host_name'] + ":" + str(params['redis']['port_no']),
            metadata_fields=["source", "page_num"],
        ),

        cache=IngestionCache(
            cache=RedisCache.from_host_and_port(
                params['redis']['host_name'],
                params['redis']['port_no']
            ),
            collection=params['redis']['cache_name'],
        ),

        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    return {'pipeline': pipeline, 'embed_model': embed_model}