#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${OUTDIR:-outputs}"

# 如果你知道 memory_cost 的单位并且可与预算对比，再设置 MEMORY_BUDGET_GB，否则留空/0
MEMORY_BUDGET_GB="${MEMORY_BUDGET_GB:-0}"

docker compose run --rm ap bash -lc "
  if [ ! -f ${OUTDIR}/runs.jsonl ]; then
    echo \"ERROR: ${OUTDIR}/runs.jsonl not found. Run experiment first.\"
    exit 2
  fi

  if [ \"${MEMORY_BUDGET_GB}\" != \"0\" ]; then
    python3 ./ap_llama3_cpu_analyze.py --outdir ${OUTDIR} --memory_budget_gb ${MEMORY_BUDGET_GB}
  else
    python3 ./ap_llama3_cpu_analyze.py --outdir ${OUTDIR}
  fi
"
