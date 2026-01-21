# syntax=docker/dockerfile:1.6
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11

# Optional proxy args (passed from docker-compose build.args)
ARG http_proxy
ARG https_proxy
ARG no_proxy

# ---- System deps ----
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
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /opt

# ---- Pin PyTorch nightly CPU (available in current nightly index) ----
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cpu"
ARG TORCH_VERSION="2.11.0.dev20260121+cpu"
RUN python3 -m pip install --no-cache-dir \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}"

# ---- AutoParallel (copied from build context to avoid GitHub TLS issues) ----
# Expect: third_party/autoparallel exists in build context
COPY third_party/autoparallel /opt/autoparallel
RUN cd /opt/autoparallel && python3 -m pip install -e .

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
