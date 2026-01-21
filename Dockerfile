# syntax=docker/dockerfile:1.6
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11

# Optional proxy args (passed from docker-compose build.args)
ARG http_proxy
ARG https_proxy
ARG no_proxy

# ---- System deps ----
# Behind HTTPS MITM proxy:
# - Bootstrap CA certs by temporarily disabling apt HTTPS verification
# - Avoid security.ubuntu.com by routing all pockets through archive.ubuntu.com
RUN set -eux; \
    printf '%s\n' \
      'Acquire::https::Verify-Peer "false";' \
      'Acquire::https::Verify-Host "false";' \
      > /etc/apt/apt.conf.d/99insecure-proxy; \
    sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list; \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list; \
    sed -i 's|https://security.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      build-essential pkg-config \
      python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip; \
    rm -f /etc/apt/apt.conf.d/99insecure-proxy; \
    apt-get update; \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# pip tooling
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /opt

# ---- Pin PyTorch nightly (matches AutoParallel README) ----
# IMPORTANT: avoid direct wheel URL (often blocked/403 by enterprise gateways).
# Use PyTorch nightly CPU index instead.
# We still pin exact version to 2.8.0.dev20250506+cpu.
ARG TORCH_VERSION="2.8.0.dev20250506+cpu"
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cpu"

RUN python3 -m pip install --no-cache-dir \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}"

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
