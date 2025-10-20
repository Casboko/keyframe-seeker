# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    git \
    curl \
    ffmpeg

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

RUN python3 -m venv /opt/venv

WORKDIR /workspace

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=tmpfs,target=/tmp \
    uv pip install --python=/opt/venv/bin/python \
        wheel

COPY requirements.in requirements-base.in constraints-base.txt pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=tmpfs,target=/tmp \
    uv pip install --python=/opt/venv/bin/python \
        --constraint constraints-base.txt \
        hydra-core==1.3.2 \
        omegaconf==2.3.0 \
        antlr4-python3-runtime==4.9.3

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=tmpfs,target=/tmp \
    uv pip install --python=/opt/venv/bin/python \
        --constraint constraints-base.txt \
        -r requirements.in \
    && uv pip install --python=/opt/venv/bin/python \
        --constraint constraints-base.txt \
        faiss-gpu-cu12

ENV PATH="/opt/venv/bin:${PATH}"

COPY conf ./conf
COPY src ./src
COPY scripts ./scripts
