# syntax=docker/dockerfile:1.6
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11

# Optional proxy args (passed from docker-compose build.args)
ARG http_proxy
ARG https_proxy
ARG no_proxy

# ---- System deps ----
# This environment is behind an HTTPS MITM proxy.
# Bootstrap approach:
#   1) Temporarily disable apt HTTPS verification (bootstrap only)
#   2) Force all Ubuntu pockets (including -security) to use archive.ubuntu.com over HTTPS
#      to avoid security.ubuntu.com MITM trust issues
#   3) Install ca-certificates and toolchain
#   4) Remove the insecure apt config and do a normal apt-get update
RUN set -eux; \
    # 1) temporarily disable apt https verification (bootstrap only)
    cat >/etc/apt/apt.conf.d/99insecure-proxy <<'EOF' \
Acquire::https::Verify-Peer "false"; \
Acquire::https::Verify-Host "false"; \
EOF \
    ; \
    # 2) force sources to https and avoid security.ubuntu.com
    sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list; \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list; \
    sed -i 's|https://security.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list; \
    \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      build-essential pkg-config \
      python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip \
    ; \
    # 3) remove bootstrap-only insecure config
    rm -f /etc/apt/apt.conf.d/99insecure-proxy; \
    # 4) normal update (now without security.ubuntu.com)
    apt-get update; \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /opt

# ---- Pin PyTorch nightly CPU wheel (matches AutoParallel README) ----
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
