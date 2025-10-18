FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    git \
    curl \
    ffmpeg \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.in requirements-base.in constraints-base.txt ./

ARG PYTORCH_INDEX_URL_CU128=https://download.pytorch.org/whl/cu128

RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel

RUN /bin/bash -lc "set -euo pipefail; \
    /opt/venv/bin/pip install -c constraints-base.txt -r requirements.in --index-url ${PYTORCH_INDEX_URL_CU128}; \
    /opt/venv/bin/pip install faiss-gpu-cu12 || /opt/venv/bin/pip install 'faiss-gpu-cu12[fix_cuda]'"

ENV PATH="/opt/venv/bin:${PATH}"

COPY conf ./conf
COPY src ./src
COPY scripts ./scripts
