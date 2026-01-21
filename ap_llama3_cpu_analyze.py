#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ap_llama3_cpu_analyze.py

结果分析脚本（与实验/记录脚本解耦）：
- 读取实验脚本输出的 outputs/runs.jsonl（以及每个 run_dir 下的 summary/meta/autoparallel_out.json 等）
- 计算派生指标（total_cost、reshard_share、comm_share、comm/compute 等）
- 在不 profiling 的前提下给出“是否值得继续”的直观判定（green/yellow/red + 原因）
- 生成：
  - <outdir>/analysis/analysis_runs.csv（逐条 run 的增强表）
  - <outdir>/analysis/analysis_best_by_mesh.csv（每个 mesh 选择一个“最值得”的 run）
  - <outdir>/analysis/analysis_report.md（可读报告）
  - <outdir>/analysis/analysis_summary.json（结构化汇总）

用法示例：
  python3 ap_llama3_cpu_analyze.py --outdir outputs
  python3 ap_llama3_cpu_analyze.py --runs_jsonl outputs/runs.jsonl
  python3 ap_llama3_cpu_analyze.py --outdir outputs --memory_budget_gb 64
  python3 ap_llama3_cpu_analyze.py --runs_jsonl outputs/runs.jsonl --runs_jsonl other_outputs/runs.jsonl
"""

import argparse
import csv
import json
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# -----------------------------
# Numeric parsing / safe ops
# -----------------------------
_NUM_RE = re.compile(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?")


def to_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, bool):
            return None
        return float(x)
    # try string parse (e.g. "1.23", "tensor(1.23)", "1e-3")
    s = str(x)
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if b == 0:
        return None
    return a / b


def finite(x: Optional[float]) -> bool:
    return x is not None and isinstance(x, float) and math.isfinite(x)


# -----------------------------
# Worth assessment (no profiling)
# -----------------------------
def assess_worth(
    solve_status: str,
    timeout: bool,
    fallback: bool,
    costs: Dict[str, Any],
    scale: Dict[str, Any],
    memory_budget_gb: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    给出直观“是否值得继续”的判定（不依赖 profiling）。
    只用：status/timeout/fallback + cost breakdown（compute/comm/reshard/memory）+ scale（vars/cons/cands）。

    thresholds 默认：
      - red_reshard_share: 0.30
      - yellow_reshard_share: 0.15
      - yellow_comm_share: 0.60
      - red_memory_budget_ratio: 0.90
      - yellow_memory_budget_ratio: 0.75
    """
    th = {
        "red_reshard_share": 0.30,
        "yellow_reshard_share": 0.15,
        "yellow_comm_share": 0.60,
        "red_memory_budget_ratio": 0.90,
        "yellow_memory_budget_ratio": 0.75,
    }
    if thresholds:
        th.update({k: float(v) for k, v in thresholds.items()})

    status = (solve_status or "").lower()
    reasons: List[str] = []

    if timeout:
        return {
            "verdict": "red",
            "reasons": ["solver timeout (not actionable without reducing search space)"],
            "recommendation": "Reduce candidate space / test smaller subgraph / increase timeout.",
        }

    if status in {"error", "fail"} or "error" in status:
        return {
            "verdict": "red",
            "reasons": [f"solver error status={solve_status}"],
            "recommendation": "Fix solver/API wiring or unsupported ops before judging strategies.",
        }

    if fallback:
        return {
            "verdict": "red",
            "reasons": [f"fallback path (status={solve_status}); no real strategy to assess"],
            "recommendation": "Ensure autoparallel is installed and entrypoint produces a real strategy.",
        }

    c_compute = to_float(costs.get("compute"))
    c_comm = to_float(costs.get("comm"))
    c_reshard = to_float(costs.get("reshard"))
    c_memory = to_float(costs.get("memory"))

    parts = [v for v in [c_compute, c_comm, c_reshard] if finite(v)]
    total_cost = sum(parts) if parts else None

    if not finite(total_cost) or total_cost <= 0:
        return {
            "verdict": "yellow",
            "reasons": ["cost breakdown missing/unparseable; cannot judge quality reliably"],
            "recommendation": "Expose numeric costs (compute/comm/reshard/memory) from autoparallel output.",
        }

    reshard_share = safe_div(c_reshard, total_cost)
    comm_share = safe_div(c_comm, total_cost)
    compute_share = safe_div(c_compute, total_cost)

    verdict = "green"

    # reshard share
    if finite(reshard_share):
        if reshard_share > th["red_reshard_share"]:
            verdict = "red"
            reasons.append(f"reshard_share={reshard_share:.2f} > {th['red_reshard_share']:.2f} (excessive layout churn)")
        elif reshard_share > th["yellow_reshard_share"] and verdict != "red":
            verdict = "yellow"
            reasons.append(f"reshard_share={reshard_share:.2f} > {th['yellow_reshard_share']:.2f} (moderate churn)")

    # comm share sanity
    if verdict != "red" and finite(comm_share):
        if comm_share > th["yellow_comm_share"]:
            verdict = "yellow"
            reasons.append(f"comm_share={comm_share:.2f} > {th['yellow_comm_share']:.2f} (communication-heavy)")

    # optional memory budget
    if memory_budget_gb is not None and finite(c_memory):
        ratio = safe_div(c_memory, memory_budget_gb)
        if finite(ratio):
            if ratio > th["red_memory_budget_ratio"]:
                verdict = "red"
                reasons.append(f"memory_cost/budget={ratio:.2f} > {th['red_memory_budget_ratio']:.2f} (high OOM risk)")
            elif ratio > th["yellow_memory_budget_ratio"] and verdict != "red":
                verdict = "yellow"
                reasons.append(f"memory_cost/budget={ratio:.2f} > {th['yellow_memory_budget_ratio']:.2f} (high memory pressure)")

    # if still green but no concrete reasons, add a gentle note
    if verdict == "green" and not reasons:
        reasons.append("no obvious structural red flags in cost breakdown (profiling still needed for real speed)")

    # recommendation
    if verdict == "green":
        rec = "Proceed to minimal runtime validation (small seq_len/batch) and then selective profiling if needed."
    elif verdict == "yellow":
        rec = "Worth limited validation; consider tightening solver constraints or reducing reshard/comm if possible."
    else:
        rec = "Not worth profiling yet; address reshard/memory risk or solver feasibility first."

    # include scale hints (do not judge; just record)
    nv = scale.get("num_vars")
    nc = scale.get("num_constraints")
    nd = scale.get("num_candidates")
    if nv is not None or nc is not None or nd is not None:
        reasons.append(f"scale(vars={nv}, cons={nc}, cands={nd})")

    return {"verdict": verdict, "reasons": reasons, "recommendation": rec}


# -----------------------------
# Aggregation / selection
# -----------------------------
def normalize_run(row: Dict[str, Any], source_label: str) -> Dict[str, Any]:
    """将 runs.jsonl 每条记录规范化，并补充派生指标字段。"""
    mesh_tag = row.get("mesh_tag") or ""
    mesh_shape = row.get("mesh_shape") or []
    solve_status = row.get("solve_status") or ""
    timeout = bool(row.get("timeout", False))
    fallback = bool(row.get("fallback", False))
    scale = row.get("scale") or {}
    costs = row.get("costs") or {}
    run_dir = row.get("run_dir") or ""

    c_compute = to_float(costs.get("compute"))
    c_comm = to_float(costs.get("comm"))
    c_reshard = to_float(costs.get("reshard"))
    c_memory = to_float(costs.get("memory"))

    parts = [v for v in [c_compute, c_comm, c_reshard] if finite(v)]
    total_cost = sum(parts) if parts else None

    reshard_share = safe_div(c_reshard, total_cost) if finite(total_cost) else None
    comm_share = safe_div(c_comm, total_cost) if finite(total_cost) else None
    compute_share = safe_div(c_compute, total_cost) if finite(total_cost) else None

    comm_to_compute = safe_div(c_comm, c_compute)
    reshard_to_total = reshard_share

    # timestamp from run_dir if possible: outputs/<mesh_tag>/<timestamp>/
    ts = ""
    try:
        # take last component
        ts = os.path.basename(run_dir.rstrip("/"))
    except Exception:
        ts = ""

    return {
        "source": source_label,
        "mesh_tag": mesh_tag,
        "mesh_shape": "x".join(map(str, mesh_shape)) if isinstance(mesh_shape, list) else str(mesh_shape),
        "timestamp": ts,
        "solve_status": solve_status,
        "timeout": int(timeout),
        "fallback": int(fallback),
        "solve_time_s": to_float(row.get("solve_time_s")),
        "total_time_s": to_float(row.get("total_time_s")),
        "num_vars": scale.get("num_vars"),
        "num_constraints": scale.get("num_constraints"),
        "num_candidates": scale.get("num_candidates"),
        "compute_cost": c_compute,
        "comm_cost": c_comm,
        "reshard_cost": c_reshard,
        "memory_cost": c_memory,
        "total_cost": total_cost,
        "reshard_share": reshard_to_total,
        "comm_share": comm_share,
        "compute_share": compute_share,
        "comm_to_compute": comm_to_compute,
        "run_dir": run_dir,
    }


def select_best_by_mesh(runs: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    为每个 (source, mesh_tag) 选择一个“最值得”的 run。
    选择规则（偏向不 profiling 的策略评估）：
      1) verdict: green > yellow > red
      2) 在同 verdict 内，total_cost 低者优先（若无 total_cost，用 solve_time_s 低者）
      3) 再用 reshard_share 低者作为 tie-break
    """
    rank = {"green": 0, "yellow": 1, "red": 2}

    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in runs:
        key = (r.get("source", ""), r.get("mesh_tag", ""))
        if key not in best:
            best[key] = r
            continue

        a = r
        b = best[key]

        ra = rank.get(a.get("worth_verdict", "red"), 2)
        rb = rank.get(b.get("worth_verdict", "red"), 2)
        if ra != rb:
            if ra < rb:
                best[key] = a
            continue

        # same verdict: compare total_cost if available
        ta = a.get("total_cost")
        tb = b.get("total_cost")

        if finite(ta) and finite(tb) and ta != tb:
            if ta < tb:
                best[key] = a
            continue
        if finite(ta) and not finite(tb):
            best[key] = a
            continue
        if not finite(ta) and finite(tb):
            continue

        # fallback: solve_time
        sa = a.get("solve_time_s")
        sb = b.get("solve_time_s")
        if finite(sa) and finite(sb) and sa != sb:
            if sa < sb:
                best[key] = a
            continue
        if finite(sa) and not finite(sb):
            best[key] = a
            continue

        # tie-break: lower reshard_share
        sha = a.get("reshard_share")
        shb = b.get("reshard_share")
        if finite(sha) and finite(shb) and sha != shb:
            if sha < shb:
                best[key] = a

    return best


# -----------------------------
# Report generation
# -----------------------------
def format_pct(x: Optional[float]) -> str:
    if not finite(x):
        return ""
    return f"{100.0 * x:.1f}%"


def format_f(x: Optional[float]) -> str:
    if not finite(x):
        return ""
    # keep compact
    return f"{x:.6g}"


def make_markdown_report(outdir: str, all_runs: List[Dict[str, Any]], best_map: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
    now = datetime.utcnow().isoformat() + "Z"

    # summary counts
    counts = {}
    for r in all_runs:
        v = r.get("worth_verdict", "unknown")
        counts[v] = counts.get(v, 0) + 1

    lines = []
    lines.append(f"# AutoParallel Results Report")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Input outdir: `{outdir}`")
    lines.append("")
    lines.append("## Verdict counts")
    for k in ["green", "yellow", "red", "unknown"]:
        if k in counts:
            lines.append(f"- {k}: {counts[k]}")
    lines.append("")

    # Best per mesh (per source)
    lines.append("## Best run per mesh (per source)")
    lines.append("")
    lines.append("| source | mesh_tag | verdict | total_cost | reshard_share | comm_share | solve_time_s | run_dir |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")

    best_items = sorted(best_map.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    for (source, mesh_tag), r in best_items:
        lines.append(
            f"| {source} | {mesh_tag} | {r.get('worth_verdict','')} | {format_f(r.get('total_cost'))} | "
            f"{format_pct(r.get('reshard_share'))} | {format_pct(r.get('comm_share'))} | {format_f(r.get('solve_time_s'))} | "
            f"`{r.get('run_dir','')}` |"
        )

    lines.append("")
    lines.append("## Notes on interpretation (no profiling)")
    lines.append("")
    lines.append("- `reshard_share` high typically indicates frequent layout changes (often not worth scaling).")
    lines.append("- `comm_share` high suggests communication-heavy strategy; may be acceptable depending on target interconnect, but requires validation.")
    lines.append("- This report does not claim real runtime speed; it is a structural/cost-model-based screening tool.")

    return "\n".join(lines) + "\n"


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="", help="Experiment outdir (e.g., outputs). If set, default runs_jsonl=outdir/runs.jsonl and analysis outputs under outdir/analysis/.")
    p.add_argument("--runs_jsonl", action="append", default=[], help="Path to runs.jsonl. Can be provided multiple times for comparing multiple result sets.")
    p.add_argument("--label", action="append", default=[], help="Optional label for each runs_jsonl. If omitted, basename(parent_dir) is used.")
    p.add_argument("--analysis_dir", default="", help="Where to write analysis outputs. Default: <outdir>/analysis or ./analysis if outdir not set.")
    p.add_argument("--memory_budget_gb", type=float, default=0.0, help="Optional memory budget for screening (only used if >0 and memory_cost is numeric/comparable).")
    p.add_argument("--red_reshard_share", type=float, default=0.30)
    p.add_argument("--yellow_reshard_share", type=float, default=0.15)
    p.add_argument("--yellow_comm_share", type=float, default=0.60)
    p.add_argument("--red_memory_budget_ratio", type=float, default=0.90)
    p.add_argument("--yellow_memory_budget_ratio", type=float, default=0.75)
    args = p.parse_args()

    # determine input runs.jsonl list
    runs_paths: List[str] = list(args.runs_jsonl)
    if args.outdir and not runs_paths:
        runs_paths = [os.path.join(args.outdir, "runs.jsonl")]

    if not runs_paths:
        raise SystemExit("ERROR: Provide --outdir or at least one --runs_jsonl")

    # labels
    labels: List[str] = list(args.label)
    while len(labels) < len(runs_paths):
        rp = runs_paths[len(labels)]
        parent = os.path.basename(os.path.dirname(os.path.abspath(rp)))
        labels.append(parent or f"set{len(labels)}")

    # analysis_dir
    analysis_dir = args.analysis_dir
    if not analysis_dir:
        if args.outdir:
            analysis_dir = os.path.join(args.outdir, "analysis")
        else:
            analysis_dir = os.path.join(".", "analysis")
    ensure_dir(analysis_dir)

    thresholds = {
        "red_reshard_share": args.red_reshard_share,
        "yellow_reshard_share": args.yellow_reshard_share,
        "yellow_comm_share": args.yellow_comm_share,
        "red_memory_budget_ratio": args.red_memory_budget_ratio,
        "yellow_memory_budget_ratio": args.yellow_memory_budget_ratio,
    }
    mem_budget = args.memory_budget_gb if args.memory_budget_gb and args.memory_budget_gb > 0 else None

    # load and normalize
    all_runs: List[Dict[str, Any]] = []
    raw_counts: List[Dict[str, Any]] = []

    for rp, lab in zip(runs_paths, labels):
        if not os.path.exists(rp):
            raise SystemExit(f"ERROR: runs_jsonl not found: {rp}")
        rows = read_jsonl(rp)
        raw_counts.append({"label": lab, "runs_jsonl": rp, "num_rows": len(rows)})
        for row in rows:
            nr = normalize_run(row, source_label=lab)
            worth = assess_worth(
                solve_status=nr["solve_status"],
                timeout=bool(nr["timeout"]),
                fallback=bool(nr["fallback"]),
                costs={
                    "compute": nr["compute_cost"],
                    "comm": nr["comm_cost"],
                    "reshard": nr["reshard_cost"],
                    "memory": nr["memory_cost"],
                },
                scale={
                    "num_vars": nr["num_vars"],
                    "num_constraints": nr["num_constraints"],
                    "num_candidates": nr["num_candidates"],
                },
                memory_budget_gb=mem_budget,
                thresholds=thresholds,
            )
            nr["worth_verdict"] = worth["verdict"]
            nr["worth_reasons"] = " | ".join(worth.get("reasons", []))
            nr["worth_recommendation"] = worth.get("recommendation", "")
            all_runs.append(nr)

    # select best per mesh per source
    best_map = select_best_by_mesh(all_runs)

    # write enriched runs CSV
    runs_csv_fields = [
        "source",
        "mesh_tag",
        "mesh_shape",
        "timestamp",
        "solve_status",
        "timeout",
        "fallback",
        "solve_time_s",
        "total_time_s",
        "num_vars",
        "num_constraints",
        "num_candidates",
        "compute_cost",
        "comm_cost",
        "reshard_cost",
        "memory_cost",
        "total_cost",
        "reshard_share",
        "comm_share",
        "compute_share",
        "comm_to_compute",
        "worth_verdict",
        "worth_reasons",
        "worth_recommendation",
        "run_dir",
    ]
    write_csv(os.path.join(analysis_dir, "analysis_runs.csv"), all_runs, runs_csv_fields)

    # write best-by-mesh CSV
    best_rows = []
    for (source, mesh_tag), r in sorted(best_map.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        best_rows.append(r)
    write_csv(os.path.join(analysis_dir, "analysis_best_by_mesh.csv"), best_rows, runs_csv_fields)

    # report.md
    report_md = make_markdown_report(args.outdir or "", all_runs, best_map)
    write_text(os.path.join(analysis_dir, "analysis_report.md"), report_md)

    # summary.json
    verdict_counts = {}
    for r in all_runs:
        v = r.get("worth_verdict", "unknown")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    summary = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "inputs": raw_counts,
        "analysis_dir": os.path.abspath(analysis_dir),
        "memory_budget_gb": mem_budget,
        "thresholds": thresholds,
        "verdict_counts": verdict_counts,
        "num_runs": len(all_runs),
        "num_best": len(best_map),
    }
    write_json(os.path.join(analysis_dir, "analysis_summary.json"), summary)

    # concise stdout
    print(f"[OK] analysis_dir: {os.path.abspath(analysis_dir)}")
    print(f"[OK] runs: {len(all_runs)} | best_by_mesh: {len(best_map)} | verdict_counts: {verdict_counts}")
    print(f"[OK] wrote: analysis_runs.csv, analysis_best_by_mesh.csv, analysis_report.md, analysis_summary.json")


if __name__ == "__main__":
    main()
