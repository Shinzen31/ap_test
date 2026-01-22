#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ap_llama3_cpu_analyze.py  (ANALYSIS ONLY)

输入：实验脚本 ap_llama3_cpu_experiment.py 产出的 outputs/ 目录
输出：report_dir 下
  - summary_table.csv
  - summary_table.json
  - report.md
  - redflags.json
  - best_by_mesh.json
  - sensitivity_summary.json

不做 profiling：
  只基于 cost breakdown（compute/comm/reshard/memory）、shares、solve_status、fallback/timeout 等做结构性判断。

使用：
python3 ap_llama3_cpu_analyze.py --outdir outputs

可调打分：
python3 ap_llama3_cpu_analyze.py --outdir outputs --w_comm 0.7 --w_reshard 1.3 --w_solve_time 0.05
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
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


# -----------------------------
# Parsing helpers
# -----------------------------
def to_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def mesh_shape_str(mesh_shape) -> str:
    if isinstance(mesh_shape, list):
        return "x".join(str(x) for x in mesh_shape)
    return str(mesh_shape)


def verdict_rank(verdict: str) -> int:
    v = (verdict or "").lower()
    if v == "green":
        return 0
    if v == "yellow":
        return 1
    if v == "red":
        return 2
    return 3


# -----------------------------
# Load runs
# -----------------------------
def load_runs(outdir: str) -> List[Dict[str, Any]]:
    runs_path = os.path.join(outdir, "runs.jsonl")
    runs: List[Dict[str, Any]] = []

    if os.path.exists(runs_path):
        with open(runs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    runs.append(json.loads(line))
                except Exception:
                    continue
        if runs:
            return runs

    # fallback: scan summary.json
    for p in glob.glob(os.path.join(outdir, "*", "*", "summary.json")):
        try:
            runs.append(read_json(p))
        except Exception:
            pass
    return runs


def find_joint_capture_dirs(outdir: str) -> List[str]:
    base = os.path.join(outdir, "_joint_capture")
    if not os.path.isdir(base):
        return []
    dirs = []
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            dirs.append(p)
    return dirs


def load_joint_capture_meta(joint_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(joint_dir, "joint_export_meta.json")
    fail_path = os.path.join(joint_dir, "joint_capture_failed.json")
    out: Dict[str, Any] = {"joint_dir": joint_dir}
    if os.path.exists(meta_path):
        out["ok"] = True
        out["meta"] = read_json(meta_path)
    elif os.path.exists(fail_path):
        out["ok"] = False
        out["fail"] = read_json(fail_path)
    else:
        out["ok"] = False
        out["fail"] = {"error": "missing joint_export_meta.json and joint_capture_failed.json"}
    return out


# -----------------------------
# Scoring / red flags
# -----------------------------
def compute_shares(costs: Dict[str, Any]) -> Dict[str, Optional[float]]:
    c_compute = to_float(costs.get("compute"))
    c_comm = to_float(costs.get("comm"))
    c_reshard = to_float(costs.get("reshard"))
    parts = [v for v in [c_compute, c_comm, c_reshard] if v is not None]
    total = sum(parts) if parts else None
    return {
        "total_cost": total,
        "compute": c_compute,
        "comm": c_comm,
        "reshard": c_reshard,
        "compute_share": safe_div(c_compute, total),
        "comm_share": safe_div(c_comm, total),
        "reshard_share": safe_div(c_reshard, total),
        "memory_cost": to_float(costs.get("memory")),
    }


def quality_score(
    verdict: str,
    timeout: bool,
    fallback: bool,
    solve_time_s: Optional[float],
    shares: Dict[str, Optional[float]],
    w_comm: float,
    w_reshard: float,
    w_solve_time: float,
) -> float:
    """
    低越好：综合结构性风险/代价占比。
    - 基础：green<yellow<red<unknown
    - penalty：timeout/fallback/缺失成本
    - weighted：comm_share、reshard_share、solve_time
    """
    base = verdict_rank(verdict) * 10.0

    if timeout:
        base += 100.0
    if fallback:
        base += 50.0

    total_cost = shares.get("total_cost")
    if total_cost is None:
        base += 20.0  # missing cost breakdown

    comm_share = shares.get("comm_share")
    reshard_share = shares.get("reshard_share")

    if comm_share is not None:
        base += w_comm * comm_share * 100.0
    else:
        base += 5.0

    if reshard_share is not None:
        base += w_reshard * reshard_share * 100.0
    else:
        base += 5.0

    if solve_time_s is not None:
        base += w_solve_time * float(solve_time_s)

    return float(base)


def redflags(row: Dict[str, Any], shares: Dict[str, Optional[float]]) -> List[str]:
    flags: List[str] = []
    status = (row.get("solve_status") or "").lower()
    if row.get("timeout"):
        flags.append("timeout")
    if row.get("fallback"):
        flags.append("fallback")
    if "error" in status or status in {"fail", "no_autoparallel", "autoparallel_found_no_entry"}:
        flags.append(f"bad_status:{row.get('solve_status')}")

    rs = shares.get("reshard_share")
    cs = shares.get("comm_share")
    if rs is not None and rs > 0.30:
        flags.append(f"high_reshard_share:{rs:.2f}")
    if cs is not None and cs > 0.70:
        flags.append(f"high_comm_share:{cs:.2f}")

    if shares.get("total_cost") is None:
        flags.append("missing_numeric_costs")

    return flags


# -----------------------------
# Reporting
# -----------------------------
def format_float(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def build_summary_rows(
    runs: List[Dict[str, Any]],
    w_comm: float,
    w_reshard: float,
    w_solve_time: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in runs:
        costs = r.get("costs") or {}
        shares = compute_shares(costs)

        solve_time_s = to_float(r.get("solve_time_s"))
        verdict = r.get("worth_verdict") or r.get("worth", {}).get("verdict")  #兼容旧结构

        score = quality_score(
            verdict=str(verdict) if verdict is not None else "",
            timeout=bool(r.get("timeout", False)),
            fallback=bool(r.get("fallback", False)),
            solve_time_s=solve_time_s,
            shares=shares,
            w_comm=w_comm,
            w_reshard=w_reshard,
            w_solve_time=w_solve_time,
        )

        row = {
            "mesh_tag": r.get("mesh_tag") or "",
            "mesh_shape": mesh_shape_str(r.get("mesh_shape")),
            "solve_status": r.get("solve_status") or "",
            "timeout": int(bool(r.get("timeout", False))),
            "fallback": int(bool(r.get("fallback", False))),
            "solve_time_s": format_float(solve_time_s, 4),
            "vars": (r.get("scale") or {}).get("num_vars", ""),
            "cons": (r.get("scale") or {}).get("num_constraints", ""),
            "cands": (r.get("scale") or {}).get("num_candidates", ""),
            "compute_cost": format_float(shares.get("compute"), 6),
            "comm_cost": format_float(shares.get("comm"), 6),
            "reshard_cost": format_float(shares.get("reshard"), 6),
            "memory_cost": format_float(shares.get("memory_cost"), 6),
            "total_cost": format_float(shares.get("total_cost"), 6),
            "compute_share": format_float(shares.get("compute_share"), 4),
            "comm_share": format_float(shares.get("comm_share"), 4),
            "reshard_share": format_float(shares.get("reshard_share"), 4),
            "worth_verdict": (verdict or ""),
            "quality_score": format_float(score, 4),
            "run_dir": r.get("run_dir") or "",
        }
        rows.append(row)
    return rows


def pick_best_by_mesh(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mesh: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        k = row.get("mesh_shape") or row.get("mesh_tag") or "unknown"
        by_mesh.setdefault(k, []).append(row)

    best: Dict[str, Any] = {}
    for k, arr in by_mesh.items():
        arr2 = sorted(
            arr,
            key=lambda x: (
                verdict_rank(str(x.get("worth_verdict"))),
                float(x.get("quality_score") or 1e18),
            ),
        )
        best[k] = {"best": arr2[0], "top3": arr2[:3]}
    return best


def summarize_sensitivity(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    items = []
    for r in runs:
        sens = r.get("dim_sensitivity")
        if not sens or not isinstance(sens, dict):
            continue
        if not sens.get("supported", False):
            continue
        items.append(
            {
                "mesh_shape": mesh_shape_str(r.get("mesh_shape")),
                "mesh_tag": r.get("mesh_tag"),
                "changed": bool(sens.get("changed")),
                "base_mesh": sens.get("base_mesh"),
                "perm_mesh": sens.get("perm_mesh"),
                "run_dir": r.get("run_dir"),
            }
        )

    changed = sum(1 for x in items if x.get("changed"))
    total = len(items)
    return {
        "total_checked": total,
        "changed_count": changed,
        "changed_ratio": (changed / total) if total else None,
        "items": items,
    }


def make_report_md(
    outdir: str,
    joint_meta_list: List[Dict[str, Any]],
    rows_sorted: List[Dict[str, Any]],
    best_by_mesh: Dict[str, Any],
    redflag_map: Dict[str, Any],
    sens_summary: Dict[str, Any],
) -> str:
    ts = datetime.utcnow().isoformat() + "Z"

    def md_escape(s: str) -> str:
        return (s or "").replace("|", "\\|")

    lines: List[str] = []
    lines.append(f"# AutoParallel Experiment Report\n")
    lines.append(f"- Generated: `{ts}`\n")
    lines.append(f"- Outdir: `{outdir}`\n")

    # Joint capture summary
    lines.append("\n## Joint fwd+bwd capture\n")
    if not joint_meta_list:
        lines.append("- No joint capture directory found under `_joint_capture/`.\n")
    else:
        for jm in joint_meta_list[-3:]:  # show last few
            if jm.get("ok"):
                meta = jm.get("meta") or {}
                lines.append(f"- `{jm.get('joint_dir')}`: **OK** method=`{meta.get('method')}` status=`{meta.get('status')}` capture_time_s=`{meta.get('capture_time_s')}`\n")
            else:
                fail = jm.get("fail") or {}
                lines.append(f"- `{jm.get('joint_dir')}`: **FAIL** error=`{fail.get('error')}`\n")

    # Best by mesh
    lines.append("\n## Best strategy per mesh (no profiling)\n")
    for mesh, obj in best_by_mesh.items():
        b = obj["best"]
        lines.append(
            f"- Mesh `{mesh}` best: verdict=`{b.get('worth_verdict')}` "
            f"score=`{b.get('quality_score')}` "
            f"reshard_share=`{b.get('reshard_share')}` comm_share=`{b.get('comm_share')}` "
            f"solve_status=`{b.get('solve_status')}`\n"
        )

    # Red flags
    lines.append("\n## Red flags\n")
    if not redflag_map["items"]:
        lines.append("- None\n")
    else:
        lines.append(f"- Total flagged runs: `{len(redflag_map['items'])}`\n")
        for it in redflag_map["items"][:10]:
            lines.append(
                f"  - mesh=`{it['mesh_shape']}` verdict=`{it['worth_verdict']}` "
                f"score=`{it['quality_score']}` flags=`{', '.join(it['flags'])}` run_dir=`{it['run_dir']}`\n"
            )

    # Sensitivity
    lines.append("\n## Mesh dimension swap sensitivity (record-only)\n")
    lines.append(f"- Checked: `{sens_summary.get('total_checked')}`; changed: `{sens_summary.get('changed_count')}`; ratio: `{sens_summary.get('changed_ratio')}`\n")

    # Top table
    lines.append("\n## Ranked runs (top 15)\n")
    header = ["mesh_shape", "worth_verdict", "quality_score", "reshard_share", "comm_share", "solve_status", "run_dir"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows_sorted[:15]:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(str(r.get("mesh_shape", ""))),
                    md_escape(str(r.get("worth_verdict", ""))),
                    md_escape(str(r.get("quality_score", ""))),
                    md_escape(str(r.get("reshard_share", ""))),
                    md_escape(str(r.get("comm_share", ""))),
                    md_escape(str(r.get("solve_status", ""))),
                    md_escape(str(r.get("run_dir", ""))),
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--report_dir", default="", help="default: <outdir>/analysis/<timestamp>")
    p.add_argument("--w_comm", type=float, default=0.8)
    p.add_argument("--w_reshard", type=float, default=1.2)
    p.add_argument("--w_solve_time", type=float, default=0.05)
    p.add_argument("--topk", type=int, default=15)
    args = p.parse_args()

    outdir = args.outdir
    runs = load_runs(outdir)
    if not runs:
        raise SystemExit(f"No runs found under {outdir}. Expect runs.jsonl or */*/summary.json")

    # report dir
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_dir = args.report_dir.strip() or os.path.join(outdir, "analysis", ts)
    ensure_dir(report_dir)

    # joint capture meta
    joint_dirs = find_joint_capture_dirs(outdir)
    joint_meta_list = [load_joint_capture_meta(d) for d in joint_dirs]
    write_json(os.path.join(report_dir, "joint_capture_meta.json"), {"items": joint_meta_list})

    # compute rows
    rows = build_summary_rows(runs, args.w_comm, args.w_reshard, args.w_solve_time)

    # sort: verdict first, then score
    rows_sorted = sorted(
        rows,
        key=lambda x: (
            verdict_rank(str(x.get("worth_verdict"))),
            float(x.get("quality_score") or 1e18),
        ),
    )

    # red flags
    flagged = []
    for r, raw in zip(rows, runs):
        shares = {
            "total_cost": to_float(r.get("total_cost")),
            "comm_share": to_float(r.get("comm_share")),
            "reshard_share": to_float(r.get("reshard_share")),
        }
        flags = redflags(raw, {
            "total_cost": to_float(raw.get("derived", {}).get("total_cost")) if isinstance(raw.get("derived"), dict) else None,
            "comm_share": to_float(raw.get("derived", {}).get("comm_share")) if isinstance(raw.get("derived"), dict) else None,
            "reshard_share": to_float(raw.get("derived", {}).get("reshard_share")) if isinstance(raw.get("derived"), dict) else None,
        })
        if flags:
            flagged.append(
                {
                    "mesh_shape": r.get("mesh_shape"),
                    "worth_verdict": r.get("worth_verdict"),
                    "quality_score": r.get("quality_score"),
                    "flags": flags,
                    "run_dir": r.get("run_dir"),
                }
            )
    redflag_map = {"items": flagged}
    write_json(os.path.join(report_dir, "redflags.json"), redflag_map)

    # best per mesh
    best_by_mesh = pick_best_by_mesh(rows)
    write_json(os.path.join(report_dir, "best_by_mesh.json"), best_by_mesh)

    # sensitivity summary
    sens_summary = summarize_sensitivity(runs)
    write_json(os.path.join(report_dir, "sensitivity_summary.json"), sens_summary)

    # export tables
    write_json(os.path.join(report_dir, "summary_table.json"), {"rows": rows_sorted})
    fieldnames = [
        "mesh_tag",
        "mesh_shape",
        "solve_status",
        "timeout",
        "fallback",
        "solve_time_s",
        "vars",
        "cons",
        "cands",
        "compute_cost",
        "comm_cost",
        "reshard_cost",
        "memory_cost",
        "total_cost",
        "compute_share",
        "comm_share",
        "reshard_share",
        "worth_verdict",
        "quality_score",
        "run_dir",
    ]
    write_csv(os.path.join(report_dir, "summary_table.csv"), rows_sorted, fieldnames)

    # markdown report
    md = make_report_md(outdir, joint_meta_list, rows_sorted, best_by_mesh, redflag_map, sens_summary)
    write_text(os.path.join(report_dir, "report.md"), md)

    # also write "latest" pointer file
    write_text(os.path.join(outdir, "analysis_latest.txt"), report_dir + "\n")

    print(f"[OK] analysis report written to: {report_dir}")
    print(f"[OK] summary_table.csv, report.md generated.")
    print(f"[OK] analysis_latest.txt -> {report_dir}")


if __name__ == "__main__":
    main()
