# syntax=docker/dockerfile:1.6
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    build-essential pkg-config \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /opt

# ---- Pin PyTorch nightly CPU wheel (matches AutoParallel README) ----
# AutoParallel README: "This currently works on PyTorch 2.8.0.dev20250506."  :contentReference[oaicite:1]{index=1}
ARG TORCH_WHL_URL="https://download.pytorch.org/whl/nightly/cpu/torch-2.8.0.dev20250506%2Bcpu-cp311-cp311-linux_x86_64.whl"
RUN python3 -m pip install --no-cache-dir "${TORCH_WHL_URL}"

# ---- AutoParallel repo (default: meta-pytorch/autoparallel main) ----
ARG AUTOPARALLEL_REPO="https://github.com/meta-pytorch/autoparallel.git"
ARG AUTOPARALLEL_REF="main"
RUN git clone "${AUTOPARALLEL_REPO}" /opt/autoparallel && \
    cd /opt/autoparallel && \
    git checkout "${AUTOPARALLEL_REF}" && \
    python3 -m pip install -e .

# ---- Runtime deps for your scripts ----
RUN python3 -m pip install --no-cache-dir \
    transformers accelerate safetensors sentencepiece tiktoken \
    numpy packaging psutil

# ---- HuggingFace cache paths (mounted from host) ----
ENV HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

RUN mkdir -p /cache/huggingface

WORKDIR /workspace
CMD ["/bin/bash"]
