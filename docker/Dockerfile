# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_ONLY_BINARY=:all: \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    git \
    curl \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.in requirements-base.in constraints-base.txt ./

ARG PYTORCH_INDEX_URL_CU128=https://download.pytorch.org/whl/cu128

RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=tmpfs,target=/tmp \
    /bin/bash -lc "set -euo pipefail; \
    /opt/venv/bin/pip install -c constraints-base.txt \
        hydra-core==1.3.2 \
        omegaconf==2.3.0 \
        antlr4-python3-runtime==4.9.3"

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=tmpfs,target=/tmp \
    /bin/bash -lc "set -euo pipefail; \
    /opt/venv/bin/pip install -c constraints-base.txt -r requirements.in --extra-index-url ${PYTORCH_INDEX_URL_CU128}; \
    /opt/venv/bin/pip install faiss-gpu-cu12"

ENV PATH="/opt/venv/bin:${PATH}"

COPY conf ./conf
COPY src ./src
COPY scripts ./scripts
