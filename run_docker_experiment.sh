#!/usr/bin/env bash
set -euo pipefail

# 你可以按需改 meshes / timeout / seq_len
MESHES="${MESHES:-8x8,4x16,4x4x4}"
OUTDIR="${OUTDIR:-outputs}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
BATCH="${BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-16}"

# 默认稳：CPU fp32，不冒险影响 export/solver
DTYPE="${DTYPE:-fp32}"
ALLOW_CPU_LOW_PRECISION="${ALLOW_CPU_LOW_PRECISION:-0}"

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
    --save_fx 0 \
    --save_fx_readable 1 \
    --write_csv 1 \
    --record_env 1 \
    --do_dim_permute_sensitivity 1
"
