#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ap_llama3_cpu_experiment.py  (EXPERIMENT ONLY)

目标（档1：最小完整）：
1) 生成 joint fwd+bwd graph（必须，失败则记录并退出）
2) 基于 joint graph 调用 autoparallel solver，记录策略与 cost breakdown（含 bwd ops 的前提：solver 输入为 joint graph）
3) 落盘所有中间产物，确保后续分析无需重跑
4) 记录基础指标 + 非 profiling 的“策略是否优质”直观指标（worth_verdict + cost shares）

运行示例：
python3 ap_llama3_cpu_experiment.py \
  --model_path /models/Meta-Llama-3-8B \
  --meshes 8x8,4x16,4x4x4 \
  --outdir outputs \
  --timeout_s 1800 \
  --batch 1 --seq_len 16 \
  --dtype fp32 \
  --capture_joint 1
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import torch

# -----------------------------
# Warning + cache root fixes (ONLY what user asked)
# -----------------------------
# Root fix for Transformers cache warning: prefer HF_HOME (Transformers v5 direction)
os.environ.setdefault("HF_HOME", "/cache/huggingface")

# Narrow warning filters: only suppress the specific noisy FutureWarnings seen in logs.
warnings.filterwarnings(
    "ignore",
    message=r"Using `TRANSFORMERS_CACHE` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch_dtype` is deprecated! Use `dtype` instead!*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"export\(f, \*args, \*\*kwargs\) is deprecated.*",
    category=FutureWarning,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None


# -----------------------------
# Utils: mesh / io
# -----------------------------
def parse_mesh(s: str) -> List[int]:
    s = s.strip().lower()
    if "x" in s:
        parts = [int(x) for x in s.split("x") if x]
        if not parts:
            raise ValueError(f"bad mesh: {s}")
        return parts
    return [int(s)]


def mesh_tag(shape: List[int]) -> str:
    return "x".join(str(x) for x in shape)


def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def write_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_csv(path: str, row: dict):
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def make_run_dir(outdir: str, tag: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(outdir, tag, ts)
    ensure_dir(run_dir)
    return run_dir


# -----------------------------
# Robust serialization helpers
# -----------------------------
def _json_friendly(obj: Any, max_depth: int = 6, _depth: int = 0) -> Any:
    if _depth > max_depth:
        return {"__truncated__": True, "repr": repr(obj)[:2000]}

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            v = obj.item()
            if isinstance(v, (bool, int, float)):
                return v
        except Exception:
            pass

    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)

    if isinstance(obj, (list, tuple)):
        return [_json_friendly(x, max_depth=max_depth, _depth=_depth + 1) for x in obj]

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[str(k)] = _json_friendly(v, max_depth=max_depth, _depth=_depth + 1)
        return out

    if isinstance(obj, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
            "requires_grad": bool(obj.requires_grad),
        }

    if hasattr(obj, "__dict__"):
        try:
            d = {}
            for k, v in obj.__dict__.items():
                if str(k).startswith("_"):
                    continue
                d[str(k)] = _json_friendly(v, max_depth=max_depth, _depth=_depth + 1)
            return {"__object__": True, "type": str(type(obj)), "dict": d, "repr": repr(obj)[:2000]}
        except Exception:
            pass

    return {"__repr__": True, "type": str(type(obj)), "repr": repr(obj)[:2000]}


def try_torch_save(path: str, obj: Any) -> Dict[str, Any]:
    ensure_dir(os.path.dirname(path))
    try:
        torch.save(obj, path)
        return {"ok": True, "path": path}
    except Exception as e:
        return {"ok": False, "path": path, "error": repr(e)}


def capture_pip_freeze() -> str:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace")
    except Exception as e:
        return f"[pip freeze failed] {repr(e)}"


def capture_torch_env() -> str:
    try:
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()
    except Exception as e:
        return f"[torch env info unavailable] {repr(e)}"


def example_inputs_meta(example_inputs: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    metas = []
    for i, t in enumerate(example_inputs):
        if isinstance(t, torch.Tensor):
            metas.append({"index": i, "shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)})
        else:
            metas.append({"index": i, "type": str(type(t)), "repr": repr(t)[:500]})
    return {"num_inputs": len(example_inputs), "inputs": metas}


# -----------------------------
# Model: Llama3-8B
# -----------------------------
def _from_pretrained_dtype_kw(dtype: torch.dtype) -> Dict[str, Any]:
    """
    ONLY for warning root-fix:
    - Prefer new `dtype=` kw if supported by current transformers
    - Fallback to legacy `torch_dtype=` for compatibility
    """
    try:
        import inspect

        sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        if "dtype" in sig.parameters:
            return {"dtype": dtype}
    except Exception:
        pass
    return {"torch_dtype": dtype}


def load_llama3(model_path: str, device: str, dtype: str, allow_cpu_low_precision: bool = False):
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError("transformers not available. Please `pip install transformers accelerate`.")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"model_path not found or not a directory: {model_path}")

    torch_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(dtype.lower(), torch.float32)

    # CPU-only：默认强制 fp32（稳定优先），除非显式允许低精度
    if device == "cpu" and not allow_cpu_low_precision:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **_from_pretrained_dtype_kw(torch_dtype),  # root fix for `torch_dtype` deprecation warning
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return model, tok


def build_example_inputs(tok, batch: int, seq_len: int, device: str) -> Tuple[torch.Tensor, ...]:
    text = "Hello world. " * max(1, seq_len // 3)
    enc = tok(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    input_ids = enc["input_ids"].repeat(batch, 1).to(device)
    return (input_ids,)


# -----------------------------
# Joint fwd+bwd graph capture (must)
# -----------------------------
def _disable_kv_cache(model) -> Dict[str, Any]:
    state = {"cfg": None, "gen": None}
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        state["cfg"] = model.config.use_cache
        model.config.use_cache = False
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "use_cache"):
        state["gen"] = model.generation_config.use_cache
        model.generation_config.use_cache = False
    return state


def _restore_kv_cache(model, state: Dict[str, Any]):
    if state.get("cfg") is not None and hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = state["cfg"]
    if state.get("gen") is not None and hasattr(model, "generation_config") and hasattr(model.generation_config, "use_cache"):
        model.generation_config.use_cache = state["gen"]


def _extract_graph_module(obj: Any) -> Optional[torch.fx.GraphModule]:
    """
    Root fix for the reported error:
    Torch 2.11 dev joint-export outputs are not stable:
    - may be GraphModule
    - may be tuple/list containing GM
    - may be ExportedProgram-like with .graph_module / .module() / etc.
    This extractor is recursive and best-effort.
    """
    if obj is None:
        return None

    if isinstance(obj, torch.fx.GraphModule):
        return obj

    if isinstance(obj, (tuple, list)):
        for x in obj:
            gm = _extract_graph_module(x)
            if gm is not None:
                return gm
        return None

    # common attribute patterns
    for attr in ["graph_module", "gm", "module", "graph"]:
        if hasattr(obj, attr):
            try:
                v = getattr(obj, attr)
                if callable(v):
                    v = v()
                gm = _extract_graph_module(v)
                if gm is not None:
                    return gm
            except Exception:
                pass

    return None


def capture_joint_fx(model, example_inputs: Tuple[torch.Tensor, ...], run_dir: str, save_readable: bool = True) -> Dict[str, Any]:
    """
    必须产出 joint fwd+bwd graph（否则档1不成立）。
    尝试顺序（best-effort）：
      1) torch.export.aot_export_joint_with_descriptors (若存在)
      2) torch._functorch.aot_autograd.aot_export_joint_simple (若存在)
    成功后保存：
      - run_dir/joint_gm.pt
      - run_dir/joint_graph.txt
      - run_dir/joint_export_raw.pkl (完整原始对象)
      - run_dir/joint_export_raw.json (json-friendly)
      - run_dir/joint_export_meta.json
    """
    meta = {
        "torch": torch.__version__,
        "status": None,
        "method": None,
        "note": None,
    }

    input_ids = example_inputs[0]

    # loss fn：必须能产生对参数的梯度 => 返回 loss 标量
    def loss_fn(input_ids_):
        out = model(input_ids=input_ids_, use_cache=False, return_dict=True)
        loss = out.logits.float().mean()
        return loss

    st = _disable_kv_cache(model)
    t0 = time.perf_counter()
    raw = None
    gm = None
    try:
        # Try torch.export API
        try:
            import torch.export as texp  # type: ignore

            if hasattr(texp, "aot_export_joint_with_descriptors"):
                meta["method"] = "torch.export.aot_export_joint_with_descriptors"
                raw = texp.aot_export_joint_with_descriptors(loss_fn, (input_ids,), {})
                gm = _extract_graph_module(raw)

                # Always dump raw attempt for postmortem
                try:
                    write_text(os.path.join(run_dir, "joint_export_raw_repr.txt"), repr(raw)[:200000])
                except Exception:
                    pass
                try:
                    try_torch_save(os.path.join(run_dir, "joint_export_raw.pt"), raw)
                except Exception:
                    pass
        except Exception as e:
            meta["note"] = f"[torch.export attempt failed] {repr(e)}"

        # Try functorch fallback
        if gm is None:
            try:
                from torch._functorch.aot_autograd import aot_export_joint_simple  # type: ignore

                meta["method"] = "torch._functorch.aot_autograd.aot_export_joint_simple"
                raw = aot_export_joint_simple(loss_fn, (input_ids,))
                gm = _extract_graph_module(raw)

                # Always dump raw attempt for postmortem
                try:
                    write_text(os.path.join(run_dir, "joint_export_raw_repr.txt"), repr(raw)[:200000])
                except Exception:
                    pass
                try:
                    try_torch_save(os.path.join(run_dir, "joint_export_raw.pt"), raw)
                except Exception:
                    pass
            except Exception as e:
                meta["note"] = (meta["note"] or "") + f" | [aot_export_joint_simple failed] {repr(e)}"

        if gm is None:
            meta["status"] = "fail_no_graphmodule"
            meta["capture_time_s"] = time.perf_counter() - t0
            write_json(os.path.join(run_dir, "joint_export_meta.json"), meta)

            # Dump json-friendly raw if present
            if raw is not None:
                try:
                    write_json(os.path.join(run_dir, "joint_export_raw.json"), _json_friendly(raw, max_depth=6))
                except Exception as e:
                    write_json(
                        os.path.join(run_dir, "joint_export_raw.json"),
                        {"__error__": repr(e), "repr": repr(raw)[:2000]},
                    )

            raise RuntimeError(
                "Joint export did not yield a torch.fx.GraphModule. "
                "See joint_export_meta.json and joint_export_raw.* for details."
            )

        meta["status"] = "ok_joint_fx"
        meta["capture_time_s"] = time.perf_counter() - t0

        # Save artifacts (maximal)
        write_json(os.path.join(run_dir, "joint_export_meta.json"), meta)

        try:
            torch.save(gm, os.path.join(run_dir, "joint_gm.pt"))
        except Exception as e:
            write_text(os.path.join(run_dir, "joint_gm_save_error.txt"), repr(e))

        if save_readable:
            try:
                write_text(os.path.join(run_dir, "joint_graph.txt"), str(gm.graph))
            except Exception as e:
                write_text(os.path.join(run_dir, "joint_graph.txt"), f"[failed] {repr(e)}")

        if raw is not None:
            pkl_info = try_torch_save(os.path.join(run_dir, "joint_export_raw.pkl"), raw)
            write_json(os.path.join(run_dir, "joint_export_raw_save.json"), pkl_info)
            try:
                write_json(os.path.join(run_dir, "joint_export_raw.json"), _json_friendly(raw, max_depth=6))
            except Exception as e:
                write_json(os.path.join(run_dir, "joint_export_raw.json"), {"__error__": repr(e), "repr": repr(raw)[:2000]})

        return {"gm": gm, "meta": meta}

    finally:
        _restore_kv_cache(model, st)


# -----------------------------
# Worth assessment (no profiling)
# -----------------------------
def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = str(x)
    # parse first float-like token
    import re

    m = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def assess_worth(solve_status: str, timeout: bool, fallback: bool, costs: Dict[str, Any]) -> Dict[str, Any]:
    """
    非 profiling 的直观判定：green/yellow/red
    依据：compute/comm/reshard 的占比结构（以及 timeout/fallback）。
    """
    status = (solve_status or "").lower()
    if timeout:
        return {"verdict": "red", "reasons": ["timeout"], "recommendation": "Reduce candidate space / smaller graph / raise timeout."}
    if "error" in status or status in {"fail"}:
        return {"verdict": "red", "reasons": [f"status={solve_status}"], "recommendation": "Fix solver/export issues first."}
    if fallback:
        return {"verdict": "red", "reasons": [f"fallback ({solve_status})"], "recommendation": "Ensure solver produces real strategy/costs."}

    c_compute = _to_float(costs.get("compute"))
    c_comm = _to_float(costs.get("comm"))
    c_reshard = _to_float(costs.get("reshard"))
    parts = [v for v in [c_compute, c_comm, c_reshard] if v is not None]
    total = sum(parts) if parts else None
    if total is None or total <= 0:
        return {"verdict": "yellow", "reasons": ["missing_costs"], "recommendation": "Expose numeric cost breakdown."}

    reshard_share = _safe_div(c_reshard, total)
    comm_share = _safe_div(c_comm, total)

    verdict = "green"
    reasons = []

    if reshard_share is not None:
        if reshard_share > 0.30:
            verdict = "red"
            reasons.append(f"reshard_share={reshard_share:.2f} (>0.30)")
        elif reshard_share > 0.15:
            verdict = "yellow"
            reasons.append(f"reshard_share={reshard_share:.2f} (>0.15)")

    if verdict != "red" and comm_share is not None and comm_share > 0.60:
        verdict = "yellow"
        reasons.append(f"comm_share={comm_share:.2f} (>0.60)")

    if not reasons:
        reasons.append("no obvious structural red flags in cost breakdown")

    rec = {
        "green": "Proceed to minimal runtime validation; profiling later if needed.",
        "yellow": "Worth limited validation; try reduce reshard/comm if possible.",
        "red": "Not worth profiling yet; address reshard/solver feasibility first.",
    }[verdict]

    return {"verdict": verdict, "reasons": reasons, "recommendation": rec, "reshard_share": reshard_share, "comm_share": comm_share, "total_cost": total}


# -----------------------------
# Solver shim (autoparallel)
# -----------------------------
def try_autoparallel_solve(gm, mesh_shape: List[int], run_dir: str) -> Dict[str, Any]:
    """
    尽量兼容不同 autoparallel API：
    - import autoparallel / autoparallel.api / autoparallel.solver
    - 探测入口函数名
    - 调用并记录 raw/pickle/json-friendly
    """
    t0 = time.perf_counter()

    ap = None
    ap_import_name = None
    for name in ["autoparallel", "autoparallel.api", "autoparallel.solver"]:
        try:
            ap = __import__(name, fromlist=["*"])
            ap_import_name = name
            break
        except Exception:
            continue

    if ap is None:
        return {
            "solve_status": "no_autoparallel",
            "solve_time_s": time.perf_counter() - t0,
            "timeout": False,
            "fallback": True,
            "scale": {"num_vars": None, "num_constraints": None, "num_candidates": None},
            "costs": {"compute": None, "comm": None, "reshard": None, "memory": None},
            "strategy": None,
            "raw_repr": "",
            "autoparallel_import": None,
            "autoparallel_entry": None,
        }

    entry = None
    entry_name = None
    for fn in [
        "solve",
        "autoparallel_solve",
        "auto_parallelize",
        "autoparallelize",
        "run_autoparallel",
        "compile_autoparallel",
    ]:
        if hasattr(ap, fn):
            entry = getattr(ap, fn)
            entry_name = fn
            break

    if entry is None:
        return {
            "solve_status": "autoparallel_found_no_entry",
            "solve_time_s": time.perf_counter() - t0,
            "timeout": False,
            "fallback": True,
            "scale": {"num_vars": None, "num_constraints": None, "num_candidates": None},
            "costs": {"compute": None, "comm": None, "reshard": None, "memory": None},
            "strategy": None,
            "raw_repr": "",
            "autoparallel_import": ap_import_name,
            "autoparallel_entry": None,
        }

    out = None
    call_note = None
    try:
        # Best-effort signature
        out = entry(gm, mesh_shape=mesh_shape, run_dir=run_dir)
        call_note = "called entry(gm, mesh_shape=..., run_dir=...)"
    except TypeError:
        try:
            out = entry(gm, mesh_shape=mesh_shape)
            call_note = "called entry(gm, mesh_shape=...)"
        except TypeError:
            out = entry(gm)
            call_note = "called entry(gm)"
    except Exception as e:
        return {
            "solve_status": "error",
            "error": repr(e),
            "solve_time_s": time.perf_counter() - t0,
            "timeout": False,
            "fallback": False,
            "scale": {"num_vars": None, "num_constraints": None, "num_candidates": None},
            "costs": {"compute": None, "comm": None, "reshard": None, "memory": None},
            "strategy": None,
            "raw_repr": "",
            "autoparallel_import": ap_import_name,
            "autoparallel_entry": entry_name,
            "call_note": call_note,
        }

    def pick(obj, keys):
        if isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    return obj[k]
        for k in keys:
            if hasattr(obj, k):
                return getattr(obj, k)
        return None

    res = {
        "solve_status": "ok",
        "solve_time_s": time.perf_counter() - t0,
        "timeout": False,
        "fallback": False,
        "scale": {
            "num_vars": pick(out, ["num_vars", "n_vars", "variables", "n_variables"]),
            "num_constraints": pick(out, ["num_constraints", "n_constraints", "constraints", "n_cons"]),
            "num_candidates": pick(out, ["num_candidates", "n_candidates", "candidates", "n_cands"]),
        },
        "costs": {
            "compute": pick(out, ["compute_cost", "compute", "flops_cost"]),
            "comm": pick(out, ["comm_cost", "comm", "communication_cost"]),
            "reshard": pick(out, ["reshard_cost", "redistribute_cost", "reshard"]),
            "memory": pick(out, ["memory_cost", "memory", "peak_memory"]),
        },
        "strategy": pick(out, ["strategy"]) if pick(out, ["strategy"]) is not None else (out if isinstance(out, dict) else None),
        "raw_repr": repr(out)[:6000],
        "autoparallel_import": ap_import_name,
        "autoparallel_entry": entry_name,
        "call_note": call_note,
    }

    # Persist raw outputs
    try:
        write_json(os.path.join(run_dir, "autoparallel_out.json"), _json_friendly(out, max_depth=6))
    except Exception as e:
        write_json(os.path.join(run_dir, "autoparallel_out.json"), {"__error__": repr(e), "repr": repr(out)[:2000]})

    pkl_info = try_torch_save(os.path.join(run_dir, "autoparallel_out.pkl"), out)
    write_json(os.path.join(run_dir, "autoparallel_out_save.json"), pkl_info)

    return res


# -----------------------------
# Timeout runner
# -----------------------------
def run_with_timeout(fn, timeout_s: int) -> Dict[str, Any]:
    import multiprocessing as mp

    def _worker(q):
        t0 = time.perf_counter()
        try:
            out = fn()
            if out is None:
                out = {}
            out.setdefault("solve_time_s", time.perf_counter() - t0)
            q.put(("ok", out))
        except Exception as e:
            q.put(("err", {"solve_status": "error", "error": repr(e), "solve_time_s": time.perf_counter() - t0}))

    q = mp.Queue()
    p = mp.Process(target=_worker, args=(q,), daemon=True)
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(5)
        return {
            "solve_status": "timeout",
            "timeout": True,
            "fallback": False,
            "solve_time_s": float(timeout_s),
            "scale": {"num_vars": None, "num_constraints": None, "num_candidates": None},
            "costs": {"compute": None, "comm": None, "reshard": None, "memory": None},
            "strategy": None,
            "raw_repr": "",
        }

    if q.empty():
        return {
            "solve_status": "error",
            "timeout": False,
            "fallback": False,
            "solve_time_s": None,
            "scale": {"num_vars": None, "num_constraints": None, "num_candidates": None},
            "costs": {"compute": None, "comm": None, "reshard": None, "memory": None},
            "strategy": None,
            "raw_repr": "",
        }

    _, payload = q.get()
    payload.setdefault("timeout", False)
    payload.setdefault("fallback", False)
    payload.setdefault("scale", {"num_vars": None, "num_constraints": None, "num_candidates": None})
    payload.setdefault("costs", {"compute": None, "comm": None, "reshard": None, "memory": None})
    payload.setdefault("strategy", None)
    payload.setdefault("raw_repr", "")
    return payload


# -----------------------------
# Optional: mesh dim swap sensitivity (record-only)
# -----------------------------
def dim_permute_sensitivity_record(gm, mesh_shape: List[int], run_dir: str, timeout_s: int) -> Dict[str, Any]:
    if len(mesh_shape) < 2:
        return {"supported": False}

    perm = list(mesh_shape)
    perm[0], perm[1] = perm[1], perm[0]

    def solve_base():
        return try_autoparallel_solve(gm, mesh_shape, run_dir)

    def solve_perm():
        return try_autoparallel_solve(gm, perm, run_dir)

    base = run_with_timeout(solve_base, timeout_s)
    alt = run_with_timeout(solve_perm, timeout_s)

    base_sig = {"status": base.get("solve_status"), "scale": base.get("scale"), "costs": base.get("costs")}
    alt_sig = {"status": alt.get("solve_status"), "scale": alt.get("scale"), "costs": alt.get("costs")}

    return {
        "supported": True,
        "base_mesh": mesh_shape,
        "perm_mesh": perm,
        "changed": base_sig != alt_sig,
        "base_sig": base_sig,
        "perm_sig": alt_sig,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model_path", required=True)
    p.add_argument("--mesh", default="")
    p.add_argument("--meshes", default="")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--timeout_s", type=int, default=1800)

    p.add_argument("--device", default="cpu", choices=["cpu"])
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=16)

    # 必须 joint
    p.add_argument("--capture_joint", type=int, default=1, help="1=must capture joint fwd+bwd graph; fail stops. default 1")

    # 输出控制
    p.add_argument("--write_csv", type=int, default=1)
    p.add_argument("--record_env", type=int, default=1)
    p.add_argument("--save_joint_readable", type=int, default=1)

    # 可选：mesh swap 敏感性（会额外做两次 solve）
    p.add_argument("--do_dim_permute_sensitivity", type=int, default=1)

    # 内存节省开关（默认关，避免影响稳定性）
    p.add_argument("--allow_cpu_low_precision", type=int, default=0)

    args = p.parse_args()

    if not args.mesh and not args.meshes:
        raise ValueError("Provide --mesh or --meshes")

    mesh_list = []
    if args.meshes:
        mesh_list.extend([x.strip() for x in args.meshes.split(",") if x.strip()])
    if args.mesh:
        mesh_list.append(args.mesh.strip())

    ensure_dir(args.outdir)

    # 记录环境（只做一次）
    if args.record_env:
        write_text(os.path.join(args.outdir, "pip_freeze.txt"), capture_pip_freeze())
        write_text(os.path.join(args.outdir, "torch_env.txt"), capture_torch_env())

    write_json(
        os.path.join(args.outdir, "experiment_args.json"),
        {
            "argv": sys.argv,
            "args": vars(args),
            "time_utc": datetime.utcnow().isoformat() + "Z",
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
        },
    )

    # Load model once
    model, tok = load_llama3(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        allow_cpu_low_precision=bool(args.allow_cpu_low_precision),
    )
    example_inputs = build_example_inputs(tok, args.batch, args.seq_len, device=args.device)

    # 先生成一次 joint graph（与 mesh 无关）：
    # 为满足“档1必须 joint”，若失败则直接退出（但会把失败信息落盘到 outdir 根目录）。
    joint_cache = None
    if args.capture_joint:
        # 放到一个稳定位置：outdir/_joint_capture/<ts>/
        joint_dir = os.path.join(args.outdir, "_joint_capture", datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
        ensure_dir(joint_dir)
        write_json(os.path.join(joint_dir, "example_inputs.json"), example_inputs_meta(example_inputs))
        try:
            joint_cache = capture_joint_fx(model, example_inputs, run_dir=joint_dir, save_readable=bool(args.save_joint_readable))
        except Exception as e:
            # 落盘失败原因并退出（符合“必须 joint”）
            write_json(
                os.path.join(joint_dir, "joint_capture_failed.json"),
                {"ok": False, "error": repr(e), "torch": torch.__version__},
            )
            raise

    gm_joint = joint_cache["gm"]
    joint_meta = joint_cache["meta"]

    # 每个 mesh：调用 solver（输入 joint graph）
    for mesh_str in mesh_list:
        mesh_shape = parse_mesh(mesh_str)
        tag = mesh_tag(mesh_shape)
        run_dir = make_run_dir(args.outdir, tag)

        meta = {
            "mesh": {"raw": mesh_str, "shape": mesh_shape, "tag": tag},
            "time_utc": datetime.utcnow().isoformat() + "Z",
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
            "model": {
                "path": args.model_path,
                "dtype": args.dtype,
                "device": args.device,
                "allow_cpu_low_precision": bool(args.allow_cpu_low_precision),
            },
            "capture": {"batch": args.batch, "seq_len": args.seq_len, "joint": True},
            "joint_capture_meta": joint_meta,
            "run_dir": run_dir,
            "argv": sys.argv,
        }
        write_json(os.path.join(run_dir, "meta.json"), meta)

        # 为每个 run_dir 再保存一份 joint graph 的只读引用信息（不复制大 pt）
        write_json(
            os.path.join(run_dir, "joint_reference.json"),
            {
                "joint_capture_dir": os.path.dirname(os.path.join(args.outdir, "_joint_capture")),
                "note": "Joint artifacts saved under outdir/_joint_capture/<ts>/ (gm.pt, graph.txt, raw.pkl, meta.json).",
            },
        )

        t_all0 = time.perf_counter()

        def solve_fn():
            return try_autoparallel_solve(gm_joint, mesh_shape, run_dir)

        solve_res = run_with_timeout(solve_fn, args.timeout_s)

        # 非 profiling 判断
        worth = assess_worth(
            solve_status=str(solve_res.get("solve_status")),
            timeout=bool(solve_res.get("timeout", False)),
            fallback=bool(solve_res.get("fallback", False)),
            costs=solve_res.get("costs", {}) or {},
        )

        # dim swap 记录（可选）
        sens = None
        if args.do_dim_permute_sensitivity and len(mesh_shape) >= 2:
            sens = dim_permute_sensitivity_record(gm_joint, mesh_shape, run_dir, timeout_s=args.timeout_s)

        total_s = time.perf_counter() - t_all0

        summary = {
            "mesh_tag": tag,
            "mesh_shape": mesh_shape,
            "solve_status": solve_res.get("solve_status"),
            "solve_time_s": solve_res.get("solve_time_s"),
            "timeout": bool(solve_res.get("timeout", False)),
            "fallback": bool(solve_res.get("fallback", False)),
            "scale": solve_res.get("scale", {}),
            "costs": solve_res.get("costs", {}),
            "raw_repr": solve_res.get("raw_repr", ""),
            "autoparallel_import": solve_res.get("autoparallel_import"),
            "autoparallel_entry": solve_res.get("autoparallel_entry"),
            "call_note": solve_res.get("call_note"),
            "dim_sensitivity": sens,
            "total_time_s": total_s,
            "run_dir": run_dir,
            # 关键：不 profiling 也能看出策略是否“结构上”优质
            "worth_verdict": worth.get("verdict"),
            "worth_reasons": worth.get("reasons"),
            "worth_recommendation": worth.get("recommendation"),
            "derived": {
                "total_cost": worth.get("total_cost"),
                "reshard_share": worth.get("reshard_share"),
                "comm_share": worth.get("comm_share"),
            },
            # joint capture meta（用于证明该 run 是 joint）
            "joint_capture": {"status": joint_meta.get("status"), "method": joint_meta.get("method")},
        }

        write_json(os.path.join(run_dir, "summary.json"), summary)
        append_jsonl(os.path.join(args.outdir, "runs.jsonl"), summary)

        if args.write_csv:
            append_csv(
                os.path.join(args.outdir, "runs.csv"),
                {
                    "mesh_tag": tag,
                    "mesh_shape": "x".join(map(str, mesh_shape)),
                    "solve_status": summary["solve_status"],
                    "solve_time_s": summary["solve_time_s"],
                    "timeout": int(summary["timeout"]),
                    "fallback": int(summary["fallback"]),
                    "vars": summary["scale"].get("num_vars", ""),
                    "cons": summary["scale"].get("num_constraints", ""),
                    "cands": summary["scale"].get("num_candidates", ""),
                    "compute_cost": summary["costs"].get("compute", ""),
                    "comm_cost": summary["costs"].get("comm", ""),
                    "reshard_cost": summary["costs"].get("reshard", ""),
                    "memory_cost": summary["costs"].get("memory", ""),
                    "worth_verdict": summary["worth_verdict"],
                    "reshard_share": summary["derived"].get("reshard_share", ""),
                    "comm_share": summary["derived"].get("comm_share", ""),
                    "total_time_s": total_s,
                    "run_dir": run_dir,
                },
            )

        print(f"[OK] mesh={tag} verdict={summary['worth_verdict']} -> {run_dir}")


if __name__ == "__main__":
    main()
