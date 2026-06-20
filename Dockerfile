# Air India RAG Chatbot — production image
# Build:  docker build -t air-india-chatbot .
# Run:    docker run -p 8000:8000 --env-file .env air-india-chatbot
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# 1) Install CPU-only PyTorch first (avoids the ~2 GB CUDA build → much smaller image),
#    then the rest of the pinned dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# 2) App code + the PRE-BUILT index (so the container runs without re-ingesting).
COPY config.py main.py ./
COPY src/ ./src/
COPY static/ ./static/
COPY chroma_db/ ./chroma_db/
COPY data/ ./data/

# 3) Bake the reranker model into the image so there is NO download at runtime.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 4) Run as a non-root user (security best practice).
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness check against the app's /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# Bind 0.0.0.0 so the container is reachable from the host / EC2.
# The GOOGLE_API_KEY is provided at runtime (--env-file or -e), never baked in.
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
