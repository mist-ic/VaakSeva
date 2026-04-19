# VaakSeva Backend — GPU-enabled Dockerfile
#
# Base: NVIDIA CUDA 12.4 + cuDNN 9 (required for PyTorch 2.11 GPU inference)
# Target hardware: Vast.ai / RunPod A10 or L40S (24-48 GB VRAM)
#
# Build args:
#   CUDA_BASE  -- set to "cpu" to build a CPU-only image without CUDA
#
# GPU build (default):
#   docker build -t vaakseva-backend .
#
# CPU build (local dev, no GPU):
#   docker build --build-arg CUDA_BASE=cpu -t vaakseva-backend-cpu .

ARG CUDA_BASE=cuda

# --- GPU stage (CUDA 12.4 + cuDNN 9) ---
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04 AS base-cuda

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# --- CPU stage (lightweight, no CUDA) ---
FROM python:3.12-slim AS base-cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Select final base ---
FROM base-${CUDA_BASE} AS final

WORKDIR /app

# Python dependencies (cache layer — rebuilt only when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY data/schemes_structured.json ./data/schemes_structured.json
COPY dashboard/ ./dashboard/

# Create runtime directories
RUN mkdir -p logs data/user_memory data/audio_cache data/processed

# Model weights mount point (bind-mount /models on production GPU pod)
# Sarvam-30B GGUF, Veena 3B will be loaded from here at runtime
RUN mkdir -p /models

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
