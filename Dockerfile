FROM python:3.11.5-slim-bookworm

WORKDIR /ragsynapse

# ── System deps in ONE layer ─────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# ── Pip upgrade ───────────────────────────────────────────────────────
RUN pip install --upgrade pip

# ── Torch CPU-only (separate step — keeps it cacheable) ──────────────
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# ── Copy ONLY requirements first (cache this layer independently) ─────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── NOW copy the rest of the code ─────────────────────────────────────
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "./src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.maxUploadSize=4000"]