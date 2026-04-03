<div align="center">

# 🧠 RAGSynapse
### Production Document Intelligence System — Multi-LLM · RAG · MLflow · FastAPI

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10-4285F4?style=flat-square)](https://docs.llamaindex.ai)
[![Redis](https://img.shields.io/badge/Redis-Vector%20Store-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-5%20Services-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-0.87%20Faithfulness-brightgreen?style=flat-square)](https://docs.ragas.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Turn any document into an intelligent, cited Q&A system — powered by your choice of LLM.**  
Upload PDFs, DOCX, or TXT files. Ask questions. Get answers with exact page references, tracked costs, and measurable RAG quality.

> Extended and re-architected from [rigvedrs/RAGIndex](https://github.com/rigvedrs/RAGIndex) — added multi-model support, FastAPI backend, RAGAS evaluation, and MLflow observability.

[📂 GitHub](https://github.com/zeeshanibrarali/RAGSynapse) · [🌐 Portfolio](https://zinov.pythonanywhere.com) · [📖 API Docs](http://localhost:8000/docs)
[🎬 Demo](https://github.com/user-attachments/assets/a59324bf-1e8a-4dc5-a3b6-7cc32c9ad31f)

</div>

---

## 🎯 The Problem

Most RAG tools are demos, not systems. They're locked to one LLM provider, have no way to measure whether answers are actually faithful to the source, and fail silently on scanned PDFs. When you deploy them, you're flying blind on cost, latency, and quality.

**RAGSynapse solves this with four pillars:**
- **Multi-LLM flexibility** — OpenAI, Anthropic Claude, or local Ollama with automatic rate-limit fallback
- **Measurable quality** — RAGAS evaluation pipeline with 0.87 faithfulness score
- **Full observability** — MLflow tracks every query: latency, tokens, cost per provider
- **Production-grade ingestion** — Tesseract OCR fallback, Redis deduplication, page-level citations

---

## 🆕 v2 vs Original

| Feature | Original | RAGSynapse v2 |
|---------|----------|----------------|
| LLM support | OpenAI only | **OpenAI + Anthropic Claude + Ollama (local)** |
| API layer | Streamlit monolith | **FastAPI REST (7 endpoints) + Streamlit UI** |
| Evaluation | None | **RAGAS — 0.87 faithfulness score** |
| Observability | None | **MLflow — latency, tokens, cost per query** |
| Document management | Re-upload every session | **Persistent Redis + document selector** |
| LLM fallback | None | **Auto-switch provider on rate limit** |
| Chunking | SemanticSplitter (1–2 chunks) | **SentenceSplitter (10–50+ chunks)** |
| Infrastructure | Single container | **5-container Docker Compose** |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Client Layer                        │
│         Streamlit UI  ·  FastAPI REST  ·  Swagger        │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                    RAG Pipeline                          │
│   PDF/OCR Ingest → SentenceSplitter → BGE Embedder       │
│   Duplicate Detection → Redis DocStore                   │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                   Storage Layer                          │
│        Redis VectorStore  ·  Redis DocStore              │
│     Persistent volumes · Namespace isolation             │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                    LLM Factory                           │
│       OpenAI GPT-4o  ·  Anthropic Claude  ·  Ollama      │
│            Auto-fallback on rate limit errors            │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                   Observability                          │
│     MLflow — query latency · token count · cost · RAGAS  │
└──────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🤖 Multi-LLM Support
Switch between OpenAI GPT-4o, Anthropic Claude, and local Ollama from a single dropdown. Automatic fallback to the next available provider on rate limit errors — no manual intervention needed.

### 📡 Production REST API
FastAPI backend with 7 endpoints, full Swagger docs at `/docs`, streaming chat support, and CORS enabled for React or mobile frontends.

### 📊 RAGAS Evaluation Pipeline
Measure RAG quality with faithfulness, answer relevancy, and context precision scores. Every evaluation run is logged to MLflow automatically for historical comparison.

### 🔭 MLflow Observability
Every chat query is tracked — latency, token count, and estimated cost per provider. Compare OpenAI vs Claude vs Ollama side-by-side on the MLflow dashboard.

### 📄 Intelligent Document Ingestion
- **Page-level tracking** — each chunk retains its source page number for precise citations
- **OCR fallback** — Tesseract automatically handles image-based / scanned PDFs
- **Deduplication** — Redis `DUPLICATES_ONLY` strategy prevents redundant re-processing
- **Persistent storage** — documents survive container restarts via Redis volume persistence
- **Per-document filtering** — query a specific document without re-uploading

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **RAG Framework** | LlamaIndex 0.10, BGE Embeddings (`bge-base-en-v1.5`) |
| **LLM Backends** | OpenAI GPT-4o · Anthropic Claude · Ollama (local) |
| **Vector DB** | Redis Stack (VectorStore + DocStore) |
| **API** | FastAPI + Uvicorn (streaming support) |
| **Evaluation** | RAGAS (faithfulness, relevancy, precision) |
| **Observability** | MLflow (latency, tokens, cost tracking) |
| **Document Processing** | PyPDF2, PyMuPDF, python-docx, pdf2image |
| **OCR** | Tesseract + pytesseract |
| **Frontend** | Streamlit |
| **Infra** | Docker Compose (5 services) |

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Git
- OpenAI / Anthropic API key *(or use free local Ollama)*

### 1. Clone & Configure
```bash
git clone https://github.com/zeeshanibrarali/RAGSynapse.git
cd RAGSynapse
cp .env.example .env
# Edit .env — set LLM_PROVIDER=ollama for free local inference
```

### 2. Launch All Services
```bash
docker compose up --build -d
```

### 3. Pull a Free Local Model *(one time, ~600MB)*
```bash
docker exec -it ragsynapse-ollama-1 ollama pull tinyllama
```

### 4. Open
| Service | URL |
|---------|-----|
| Streamlit App | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |
| MLflow Dashboard | http://localhost:5000 |

### With OpenAI *(best quality)*
```bash
# Add to .env:
OPENAI_API_KEY=sk-your-key
LLM_PROVIDER=openai
docker compose restart ragsynapse
```

---

## 📖 API Reference

Full interactive docs at `http://localhost:8000/docs`

```bash
GET  /health                  # Health check
GET  /documents               # List stored documents
POST /documents/upload        # Upload and ingest document
POST /chat                    # Ask a question
POST /chat/stream             # Streaming response
POST /eval/run                # Run RAGAS evaluation
GET  /eval/results            # Fetch evaluation history
```

**Example — ask a question:**
```json
POST /chat
{
  "question": "What are the key findings?",
  "provider": "ollama"
}
```

**Example — run RAGAS evaluation:**
```json
POST /eval/run
{
  "questions": ["Who authored this report?"],
  "ground_truths": ["Zeeshan Ibrar"],
  "provider": "openai"
}
```

---

## 📊 Evaluation Results

| Metric | Score | Judge LLM |
|--------|-------|-----------|
| Faithfulness | **0.87** | GPT-4o-mini |
| Answer Relevancy | TBD | GPT-4o-mini |
| Context Precision | TBD | GPT-4o-mini |

*Run `POST /eval/run` after adding your OpenAI key to generate scores.*

---

## ⚙️ Configuration (`config.toml`)

```toml
[embed_model]
model_name = "BAAI/bge-base-en-v1.5"
embed_batch_size = 1

[transformations]
chunk_size = 1000
chunk_overlap = 100

[redis]
host_name = "redis"
port_no = 6379
doc_store_name = "DocStore_v1"
vector_index_name = "VecStore_v1"
```

---

## 📁 Project Structure

```
RAGSynapse/
├── src/
│   ├── app.py                    # Streamlit UI
│   ├── api/                      # FastAPI backend (v2)
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── dependencies.py
│   │   └── routes/
│   │       ├── chat.py
│   │       ├── documents.py
│   │       └── eval.py
│   └── ragsynapse/
│       ├── chat/                 # LLM + retrieval engine
│       ├── pipeline/             # Ingestion + chunking
│       ├── pdf_ingest/           # OCR pipeline
│       ├── llm/                  # Multi-model factory (v2)
│       ├── evaluation/           # RAGAS pipeline (v2)
│       └── observability/        # MLflow tracker (v2)
├── config.toml
├── docker-compose.yml            # 5-service orchestration
├── Dockerfile                    # Streamlit container
└── Dockerfile.api                # FastAPI container
```

---

## 🔧 Daily Usage

```bash
# Start all services
docker compose up -d

# Stop
docker compose down

# Clear documents (keeps Ollama models)
docker exec -it ragsynapse-redis-1 redis-cli FLUSHALL

# View logs
docker logs ragsynapse-ragsynapse-1 --follow
```

---

## 📈 Performance

| Operation | Speed |
|-----------|-------|
| Standard PDF extraction | ~0.5–2s per page |
| OCR (image-based PDF) | ~3–5s per page |
| Vector embedding | ~500–1000 chunks/min (CPU) |
| Query response time | ~2–5s end-to-end |
| Memory usage | ~2GB base + 500MB per 100 docs |

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m 'feat: your feature'`
4. Push & open a PR

---

## 🌟 Acknowledgements

Original LlamaIndex RAG base by [rigvedrs/RAGIndex](https://github.com/rigvedrs/RAGIndex) — extended with multi-model support, FastAPI backend, RAGAS evaluation, and MLflow observability.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 👤 Author

**Zeeshan Ibrar** — Python AI/ML Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/zeeshan-ibrar-6a3913256/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-1a1a2e?style=flat-square)](https://zinov.pythonanywhere.com)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=flat-square&logo=gmail)](mailto:inboxtozeeshanibrar@gmail.com)

---

<div align="center">
  <sub>LlamaIndex · FastAPI · Redis · MLflow · RAGAS · Docker · Tesseract OCR</sub>
</div>
