import json
import os
import tempfile
import redis as redis_lib
import toml
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from llama_index.core import Document

from api.models import DocumentsResponse, DocumentInfo, UploadResponse
from api.dependencies import get_ingestion_pipeline

router = APIRouter(prefix="/documents", tags=["documents"])

current_dir = Path(__file__).parent
config_path = current_dir / ".." / ".." / ".." / "config.toml"
with open(config_path, "r") as f:
    params = toml.load(f)


def _get_stored_docs() -> list[DocumentInfo]:
    r = redis_lib.Redis(
        host=params["redis"]["host_name"],
        port=params["redis"]["port_no"],
        decode_responses=True,
    )
    metadata_key = f"{params['redis']['doc_store_name']}/metadata"
    all_fields = r.hgetall(metadata_key)
    docs = []
    for fname, meta_str in all_fields.items():
        if fname.endswith((".pdf", ".docx", ".txt")):
            meta = json.loads(meta_str)
            docs.append(DocumentInfo(
                filename=fname,
                doc_hash=meta.get("doc_hash", ""),
            ))
    return sorted(docs, key=lambda d: d.filename)


@router.get("", response_model=DocumentsResponse)
async def list_documents():
    """List all documents currently stored in Redis."""
    docs = _get_stored_docs()
    return DocumentsResponse(documents=docs, total=len(docs))


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document into the RAG pipeline.
    Supports PDF, DOCX, TXT.
    """
    allowed = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not supported. Use: {allowed}"
        )

    # Save to temp file
    content = await file.read()
    with tempfile.NamedTemporaryFile(
        suffix=ext, delete=False, dir="/tmp"
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Build LlamaIndex document
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(tmp_path)
            text = ""
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text() or ""
                text += f"PAGE_NUM={page_num}\n{page_text}\n"
        else:
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        document = Document(
            text=text,
            metadata={"source": file.filename, "file_name": file.filename},
        )

        pipeline = get_ingestion_pipeline()
        nodes = pipeline.run(documents=[document], show_progress=False)

        return UploadResponse(
            status="success",
            filename=file.filename,
            nodes_ingested=len(nodes),
            message=f"Ingested {len(nodes)} chunks into Redis",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)