#!/usr/bin/env bash
set -euo pipefail

# ---- Tunables (override via env) ----
OUTDIR="${OUTDIR:-outputs}"
REPORT_DIR="${REPORT_DIR:-}"        # default: <OUTDIR>/analysis/<ts>
W_COMM="${W_COMM:-0.8}"
W_RESHARD="${W_RESHARD:-1.2}"
W_SOLVE_TIME="${W_SOLVE_TIME:-0.05}"

docker compose run --rm \
  --user "$(id -u):$(id -g)" \
  --volume /etc/passwd:/etc/passwd:ro \
  --volume /etc/group:/etc/group:ro \
  --env HOME=/workspace \
  ap bash -lc "
    set -euo pipefail

    python3 -c 'import torch; print(\"torch=\", torch.__version__)'
    test -d ${OUTDIR} || (echo \"OUTDIR not found: ${OUTDIR}\" && exit 2)

    python3 ./ap_llama3_cpu_analyze.py \
      --outdir ${OUTDIR} \
      ${REPORT_DIR:+--report_dir ${REPORT_DIR}} \
      --w_comm ${W_COMM} \
      --w_reshard ${W_RESHARD} \
      --w_solve_time ${W_SOLVE_TIME}
  "
