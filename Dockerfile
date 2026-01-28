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
      ninja-build \
      python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python3-pip \
      libopenblas-dev liblapack-dev \
      libssl-dev zlib1g-dev libffi-dev libomp-dev libnuma-dev \
      patchelf; \
    rm -f /etc/apt/apt.conf.d/99insecure-proxy; \
    apt-get update; \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# ---- venv (avoid root-user pip warning, isolate deps) ----
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /opt

# ---- Build PyTorch from local source tar (version pinned) ----
ARG TORCH_VERSION="2.8.0.dev20250506+cpu"
COPY pytorch-src-20250506.tar.gz /opt/pytorch-src-20250506.tar.gz

# ====================== CPU LIMIT (40%) - REMOVE THIS BLOCK TO UNLIMIT ======================
# This block limits *build parallelism* to <= 40% of available logical CPUs inside the container.
# NOTE: Hard CPU quota enforcement must be done via docker/compose (--cpus/--cpuset-cpus).
ARG CPU_LIMIT_PERCENT="40"
ENV CPU_LIMIT_PERCENT=${CPU_LIMIT_PERCENT}
# ============================================================================================

RUN set -eux; \
    HTTP_PROXY="${http_proxy}" HTTPS_PROXY="${https_proxy}" NO_PROXY="${no_proxy}" \
    http_proxy="${http_proxy}" https_proxy="${https_proxy}" no_proxy="${no_proxy}" \
      tar -xzf /opt/pytorch-src-20250506.tar.gz -C /opt; \
    \
    # Fix git "dubious ownership" during build/version steps
    git config --global --add safe.directory /opt/pytorch; \
    git config --global --add safe.directory /opt/pytorch/third_party/ideep/mkl-dnn; \
    \
    cd /opt/pytorch; \
    \
    # Ensure sufficiently new CMake/Ninja
    python -m pip install --no-cache-dir --retries 10 --timeout 120 -U cmake ninja; \
    \
    python -m pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt; \
    \
    # -------- compute build parallelism cap (<= 40% logical CPUs) --------
    NPROC="$(getconf _NPROCESSORS_ONLN)"; \
    JOBS="$(( (NPROC * CPU_LIMIT_PERCENT) / 100 ))"; \
    if [ "${JOBS}" -lt 1 ]; then JOBS=1; fi; \
    echo "[CPU LIMIT] nproc=${NPROC}, percent=${CPU_LIMIT_PERCENT}%, MAX_JOBS=${JOBS}"; \
    \
    export PYTORCH_BUILD_VERSION="${TORCH_VERSION}"; \
    export PYTORCH_BUILD_NUMBER="1"; \
    export USE_CUDA=0 USE_ROCM=0; \
    \
    # Limit parallel build threads (soft cap)
    export MAX_JOBS="${JOBS}"; \
    export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"; \
    \
    # Prevent threaded libs from inflating CPU usage during checks/build
    export OMP_NUM_THREADS=1; \
    export MKL_NUM_THREADS=1; \
    export OPENBLAS_NUM_THREADS=1; \
    \
    python setup.py bdist_wheel; \
    python -m pip install --no-cache-dir --retries 10 --timeout 120 dist/torch-*.whl; \
    \
    # IMPORTANT: don't import torch inside source tree (it shadows installed torch)
    cd /; \
    python -c "import torch; expected='${TORCH_VERSION}'; got=torch.__version__; print('torch.__version__ =', got); \
                (got==expected) or (_ for _ in ()).throw(SystemExit(f'[FATAL] torch version mismatch: expected={expected}, got={got}'))"

# ---- AutoParallel (copied from build context to avoid GitHub TLS issues) ----
# Expect: third_party/autoparallel exists in build context
COPY third_party/autoparallel /opt/autoparallel
RUN cd /opt/autoparallel && python -m pip install -e .

# ---- Runtime deps for your scripts ----
RUN python -m pip install --no-cache-dir \
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
