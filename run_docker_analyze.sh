#!/usr/bin/env bash
set -euo pipefail

# ---- Tunables (override via env) ----
OUTDIR="${OUTDIR:-outputs}"
REPORT_DIR="${REPORT_DIR:-}"        # default: <OUTDIR>/analysis/<ts>
W_COMM="${W_COMM:-0.8}"
W_RESHARD="${W_RESHARD:-1.2}"
W_SOLVE_TIME="${W_SOLVE_TIME:-0.05}"

# ---- Run analysis as host user (FIX: avoid root-owned outputs) ----
docker compose run --rm \
  --user "$(id -u):$(id -g)" \
  ap bash -lc "
    python3 -c 'import torch; print(\"torch=\", torch.__version__)'

    test -d ${OUTDIR} || (echo \"OUTDIR not found: ${OUTDIR}\" && exit 2)

    python3 ./ap_llama3_cpu_analyze.py \
      --outdir ${OUTDIR} \
      ${REPORT_DIR:+--report_dir ${REPORT_DIR}} \
      --w_comm ${W_COMM} \
      --w_reshard ${W_RESHARD} \
      --w_solve_time ${W_SOLVE_TIME}
  "
