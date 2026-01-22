#!/usr/bin/env bash
set -euo pipefail

# ---- Tunables (override via env) ----
MESHES="${MESHES:-8x8,4x16,4x4x4}"
OUTDIR="${OUTDIR:-outputs}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
BATCH="${BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-16}"

DTYPE="${DTYPE:-fp32}"
ALLOW_CPU_LOW_PRECISION="${ALLOW_CPU_LOW_PRECISION:-0}"

CAPTURE_JOINT="${CAPTURE_JOINT:-1}"                # MUST be 1 for档1
SAVE_JOINT_READABLE="${SAVE_JOINT_READABLE:-1}"
RECORD_ENV="${RECORD_ENV:-1}"
DO_DIM_PERMUTE_SENS="${DO_DIM_PERMUTE_SENS:-1}"

docker compose build

docker compose run --rm ap bash -lc "
  python3 -c 'import torch; print(\"torch=\", torch.__version__)'
  python3 -c 'import autoparallel; print(\"autoparallel import ok\")'

  python3 ./ap_llama3_cpu_experiment.py \
    --model_path /models/Meta-Llama-3-8B \
    --meshes ${MESHES} \
    --outdir ${OUTDIR} \
    --timeout_s ${TIMEOUT_S} \
    --batch ${BATCH} \
    --seq_len ${SEQ_LEN} \
    --dtype ${DTYPE} \
    --allow_cpu_low_precision ${ALLOW_CPU_LOW_PRECISION} \
    --capture_joint ${CAPTURE_JOINT} \
    --save_joint_readable ${SAVE_JOINT_READABLE} \
    --record_env ${RECORD_ENV} \
    --do_dim_permute_sensitivity ${DO_DIM_PERMUTE_SENS} \
    --write_csv 1
"
