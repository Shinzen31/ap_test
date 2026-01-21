'''

如何运行（实验部分）

最常用（稳态、最少风险）：

python3 ap_llama3_cpu_experiment.py \
  --model_path ~/models/Meta-Llama-3-8B \
  --meshes 8x8,4x16,4x4x4 \
  --outdir outputs \
  --timeout_s 1800 \
  --batch 1 --seq_len 16 \
  --save_fx 0


如果你想尝试省内存（仅在你确认不会影响测试时）：

python3 ap_llama3_cpu_experiment.py \
  --model_path ~/models/Meta-Llama-3-8B \
  --mesh 8x8 \
  --dtype bf16 \
  --allow_cpu_low_precision 1 \
  --batch 1 --seq_len 16
  
  '''




import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch

# transformers 需要你本机已安装，并且 llama3 权重在本地或可访问
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None


# -----------------------------
# Utils: mesh / io
# -----------------------------
def parse_mesh(s: str):
    s = s.strip().lower()
    if "x" in s:
        parts = [int(x) for x in s.split("x") if x]
        if not parts:
            raise ValueError(f"bad mesh: {s}")
        return parts
    return [int(s)]


def mesh_tag(shape):
    return "x".join(str(x) for x in shape)


def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)


def write_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_csv(path, row: dict):
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def make_run_dir(outdir, tag):
    # 以 mesh 数字命名的目录；每次运行只生成一个时间戳子目录
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(outdir, tag, ts)
    ensure_dir(run_dir)
    return run_dir


# -----------------------------
# Robust serialization helpers
# -----------------------------
def _json_friendly(obj: Any, max_depth: int = 6, _depth: int = 0) -> Any:
    """
    尽可能把对象转为 JSON 友好结构，无法序列化的用 repr 兜底。
    注意：这是“记录原始信息”的工具，不做任何分析判断。
    """
    if _depth > max_depth:
        return {"__truncated__": True, "repr": repr(obj)[:2000]}

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # torch / numpy scalar
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            v = obj.item()
            if isinstance(v, (bool, int, float)):
                return v
        except Exception:
            pass

    # torch dtype / device
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [_json_friendly(x, max_depth=max_depth, _depth=_depth + 1) for x in obj]

    # dict
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                kk = str(k)
            except Exception:
                kk = repr(k)
            out[kk] = _json_friendly(v, max_depth=max_depth, _depth=_depth + 1)
        return out

    # torch Tensor: 只记录 shape/dtype/device，避免爆文件
    if isinstance(obj, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }

    # objects: 尝试提取 __dict__ 的浅层信息
    if hasattr(obj, "__dict__"):
        try:
            d = {}
            for k, v in obj.__dict__.items():
                if k.startswith("_"):
                    continue
                d[str(k)] = _json_friendly(v, max_depth=max_depth, _depth=_depth + 1)
            return {
                "__object__": True,
                "type": f"{type(obj)}",
                "dict": d,
                "repr": repr(obj)[:2000],
            }
        except Exception:
            pass

    # fallback
    return {"__repr__": True, "type": f"{type(obj)}", "repr": repr(obj)[:2000]}


def try_torch_save(path: str, obj: Any) -> Dict[str, Any]:
    """
    尝试用 torch.save 保存任意对象（pickle），用于“尽量全记录”。
    若失败，返回 error 信息，不抛异常。
    """
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
    # 尽量不引入额外依赖：优先 torch 自带的 collect_env（若存在）
    try:
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()
    except Exception as e:
        return f"[torch env info unavailable] {repr(e)}"


def example_inputs_meta(example_inputs: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    metas = []
    for i, t in enumerate(example_inputs):
        if isinstance(t, torch.Tensor):
            metas.append(
                {
                    "index": i,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "device": str(t.device),
                }
            )
        else:
            metas.append({"index": i, "type": str(type(t)), "repr": repr(t)[:500]})
    return {"num_inputs": len(example_inputs), "inputs": metas}


# -----------------------------
# Model: Llama3-8B
# -----------------------------
def load_llama3(model_path: str, device: str, dtype: str, allow_cpu_low_precision: bool = False):
    """
    仅修改点：支持你给的 ModelScope 本地缓存目录作为 model_path。
    - 直接对本地目录调用 transformers.from_pretrained
    - local_files_only=True 避免误触发联网
    - trust_remote_code=True 提升兼容性（一般不影响 Llama3）

    额外（受控开关）：
    - allow_cpu_low_precision=1 时才允许 CPU 使用 bf16/fp16（默认仍强制 fp32，避免影响 autoparallel 测试稳定性）
    """
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError("transformers not available. Please `pip install transformers accelerate`.")

    # 提醒但不阻塞：temp 目录可能被清理
    if "/._____temp/" in model_path:
        print(
            "[WARN] You are using a ModelScope temp cache path; it may be cleaned. "
            "Prefer the non-temp path under hub/models/."
        )

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"model_path not found or not a directory: {model_path}")

    torch_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(dtype.lower(), torch.float32)

    # CPU-only：默认强制 fp32 更稳（只有显式开关才允许低精度）
    if device == "cpu" and not allow_cpu_low_precision:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    # 有些 Llama tokenizer 没有 pad_token，给一个安全兜底
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return model, tok


def build_example_inputs(tok, batch: int, seq_len: int, device: str):
    # 用极小序列做图捕获，避免 CPU 上 8B 过慢
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
# Graph capture (CPU-friendly)
# -----------------------------
def capture_fx(model, example_inputs, save_fx_path: str = "", save_fx_readable_path: str = ""):
    """
    torch 2.11 dev 下，torch._dynamo.export 对输出类型要求更严格：
    - transformers 默认可能返回包含 DynamicCache 的 ModelOutput，导致 export 失败
    - 解决：强制 use_cache=False，并且 wrapper forward 只返回 logits(Tensor)
    """
    meta = {"torch": torch.__version__, "status": None, "note": None}
    t0 = time.perf_counter()

    try:
        with torch.no_grad():
            # ---- Force-disable cache at config level (both config and generation_config if present) ----
            orig_cfg_use_cache = None
            orig_gen_use_cache = None

            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                orig_cfg_use_cache = model.config.use_cache
                model.config.use_cache = False

            if hasattr(model, "generation_config") and hasattr(model.generation_config, "use_cache"):
                orig_gen_use_cache = model.generation_config.use_cache
                model.generation_config.use_cache = False

            try:
                input_ids = example_inputs[0]

                # Wrapper: return ONLY logits (Tensor), never DynamicCache/past_key_values
                def _fw(input_ids_):
                    out = model(input_ids=input_ids_, use_cache=False, return_dict=True)
                    return out.logits

                exported = torch._dynamo.export(_fw, input_ids, aten_graph=True)
                fwd_gm = exported[0] if isinstance(exported, tuple) else exported

            finally:
                # restore cache flags
                if orig_cfg_use_cache is not None:
                    model.config.use_cache = orig_cfg_use_cache
                if orig_gen_use_cache is not None:
                    model.generation_config.use_cache = orig_gen_use_cache

        meta["status"] = "ok_fwd_fx"

    except Exception as e:
        meta["status"] = "fail"
        meta["note"] = repr(e)
        raise RuntimeError(
            "Failed to capture FX via torch._dynamo.export. "
            "On torch 2.11 dev, this commonly happens if outputs include non-Tensor objects "
            "(e.g., transformers DynamicCache)."
        ) from e

    meta["capture_time_s"] = time.perf_counter() - t0

    if save_fx_path:
        ensure_dir(os.path.dirname(save_fx_path))
        torch.save(fwd_gm, save_fx_path)

    if save_fx_readable_path:
        try:
            write_text(save_fx_readable_path, str(fwd_gm.graph))
        except Exception as e:
            write_text(save_fx_readable_path, f"[failed to write readable fx graph] {repr(e)}")

    return {"gm": fwd_gm, "meta": meta}



# -----------------------------
# Solver/compile shim (real autoparallel if available)
# -----------------------------
def try_autoparallel_solve(gm, example_inputs, mesh_shape, run_dir):
    """
    统一返回结构（尽量保持你原风格）：
      {
        solve_status, solve_time_s, timeout, fallback,
        scale: {num_vars, num_constraints, num_candidates},
        costs: {compute, comm, reshard, memory},
        strategy: <raw or best-effort>,
        raw_repr: ...,
        out_json: ... (json-friendly best-effort),
        out_pickle: ... (torch.save result),
      }

    注意：本文件只做“实验记录”，不做策略优劣判断。
    """
    t0 = time.perf_counter()

    # 尝试导入 autoparallel（如果你本机装了 meta-pytorch/autoparallel）
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
            "solve_status": "offline_no_autoparallel",
            "solve_time_s": time.perf_counter() - t0,
            "timeout": False,
            "fallback": True,
            "scale": {"num_vars": None, "num_constraints": None, "num_candidates": None},
            "costs": {"compute": None, "comm": None, "reshard": None, "memory": None},
            "strategy": {"note": "autoparallel not installed; offline placeholder"},
            "raw_repr": "",
            "out_json": None,
            "out_pickle": None,
            "autoparallel_import": None,
            "autoparallel_entry": None,
        }

    # 入口函数名做容错探测
    entry = None
    entry_name = None
    for fn in ["solve", "autoparallel_solve", "auto_parallelize", "autoparallelize", "run_autoparallel", "compile_autoparallel"]:
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
            "strategy": {"note": "autoparallel imported but no known entry; wire in manually"},
            "raw_repr": "",
            "out_json": None,
            "out_pickle": None,
            "autoparallel_import": ap_import_name,
            "autoparallel_entry": None,
        }

    # 调入口（你可能需要按你本地 autoparallel 签名微调 kwargs）
    out = None
    call_note = None
    try:
        out = entry(gm, mesh_shape=mesh_shape, example_inputs=example_inputs, run_dir=run_dir)
        call_note = "called entry(gm, mesh_shape=..., example_inputs=..., run_dir=...)"
    except TypeError:
        out = entry(gm)
        call_note = "called entry(gm) (fallback signature)"
    except Exception as e:
        # 保持记录完整：把异常也记录下来
        return {
            "solve_status": "error",
            "error": repr(e),
            "solve_time_s": time.perf_counter() - t0,
            "timeout": False,
            "fallback": False,
            "scale": {"num_vars": None, "num_constraints": None, "num_candidates": None},
            "costs": {"compute": None, "comm": None, "reshard": None, "memory": None},
            "strategy": {"note": "exception raised during entry() call"},
            "raw_repr": "",
            "out_json": None,
            "out_pickle": None,
            "autoparallel_import": ap_import_name,
            "autoparallel_entry": entry_name,
            "call_note": call_note,
        }

    res = {
        "solve_status": "ok",
        "solve_time_s": time.perf_counter() - t0,
        "timeout": False,
        "fallback": False,
        "scale": {},
        "costs": {},
        "strategy": {},
        "raw_repr": repr(out)[:4000],  # 记录更长一些，便于后续分析
        "autoparallel_import": ap_import_name,
        "autoparallel_entry": entry_name,
        "call_note": call_note,
    }

    # 尽量从 out 中扒 scale/costs/strategy（字段名按常见别名探测）
    def pick(obj, keys):
        if isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    return obj[k]
        for k in keys:
            if hasattr(obj, k):
                return getattr(obj, k)
        return None

    res["scale"]["num_vars"] = pick(out, ["num_vars", "n_vars", "variables", "n_variables"])
    res["scale"]["num_constraints"] = pick(out, ["num_constraints", "n_constraints", "constraints", "n_cons"])
    res["scale"]["num_candidates"] = pick(out, ["num_candidates", "n_candidates", "candidates", "n_cands"])

    res["costs"]["compute"] = pick(out, ["compute_cost", "compute", "flops_cost"])
    res["costs"]["comm"] = pick(out, ["comm_cost", "comm", "communication_cost"])
    res["costs"]["reshard"] = pick(out, ["reshard_cost", "redistribute_cost", "reshard"])
    res["costs"]["memory"] = pick(out, ["memory_cost", "memory", "peak_memory"])

    strat = pick(out, ["strategy"])
    if strat is None and isinstance(out, dict):
        strat = out
    res["strategy"] = strat if strat is not None else {"note": "strategy not found; inspect raw_repr"}

    # 记录：JSON 友好形式（best-effort）
    try:
        res["out_json"] = _json_friendly(out, max_depth=6)
    except Exception as e:
        res["out_json"] = {"__error__": repr(e), "repr": repr(out)[:2000]}

    # 记录：尽力保存完整对象（pickle）
    pkl_path = os.path.join(run_dir, "autoparallel_out.pkl")
    res["out_pickle"] = try_torch_save(pkl_path, out)

    # 同时把 json-friendly 落盘（失败也不影响主流程）
    try:
        write_json(os.path.join(run_dir, "autoparallel_out.json"), res["out_json"])
    except Exception as e:
        write_json(os.path.join(run_dir, "autoparallel_out.json"), {"__error__": repr(e), "repr": repr(out)[:2000]})

    return res


# -----------------------------
# Timeout runner (single-file)
# -----------------------------
def run_with_timeout(fn, timeout_s: int):
    # 用 multiprocessing 来硬超时（Windows 下也可用，但注意 spawn 开销）
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
            "strategy": {"note": "timeout"},
            "raw_repr": "",
            "out_json": None,
            "out_pickle": None,
            "autoparallel_import": None,
            "autoparallel_entry": None,
        }

    if q.empty():
        return {
            "solve_status": "error",
            "timeout": False,
            "fallback": False,
            "solve_time_s": None,
            "scale": {"num_vars": None, "num_constraints": None, "num_candidates": None},
            "costs": {"compute": None, "comm": None, "reshard": None, "memory": None},
            "strategy": {"note": "no payload"},
            "raw_repr": "",
            "out_json": None,
            "out_pickle": None,
            "autoparallel_import": None,
            "autoparallel_entry": None,
        }

    status, payload = q.get()
    payload.setdefault("timeout", False)
    payload.setdefault("fallback", False)
    payload.setdefault("scale", {"num_vars": None, "num_constraints": None, "num_candidates": None})
    payload.setdefault("costs", {"compute": None, "comm": None, "reshard": None, "memory": None})
    payload.setdefault("strategy", {"note": "missing"})
    payload.setdefault("raw_repr", "")
    payload.setdefault("out_json", None)
    payload.setdefault("out_pickle", None)
    payload.setdefault("autoparallel_import", None)
    payload.setdefault("autoparallel_entry", None)
    return payload


# -----------------------------
# Mesh sensitivity (record-only, no judgement)
# -----------------------------
def dim_permute_sensitivity_record(gm, example_inputs, mesh_shape, run_dir):
    """
    仅记录：swap mesh 前两维后，solver 输出的关键摘要是否发生变化。
    不做“好坏”判断；后续分析脚本再解释。
    """
    if len(mesh_shape) < 2:
        return {"supported": False}

    perm = list(mesh_shape)
    perm[0], perm[1] = perm[1], perm[0]

    base = try_autoparallel_solve(gm, example_inputs, mesh_shape, run_dir)
    alt = try_autoparallel_solve(gm, example_inputs, perm, run_dir)

    base_sig = {
        "solve_status": base.get("solve_status"),
        "scale": base.get("scale"),
        "costs": base.get("costs"),
        "raw": (base.get("raw_repr", "")[:800] if base.get("raw_repr") else ""),
    }
    alt_sig = {
        "solve_status": alt.get("solve_status"),
        "scale": alt.get("scale"),
        "costs": alt.get("costs"),
        "raw": (alt.get("raw_repr", "")[:800] if alt.get("raw_repr") else ""),
    }

    return {
        "supported": True,
        "base_mesh": mesh_shape,
        "perm_mesh": perm,
        "changed": base_sig != alt_sig,
        "base_sig": base_sig,
        "perm_sig": alt_sig,
        "note": "Record-only. Use analysis script to interpret sensitivity.",
    }


# -----------------------------
# Main (experiment + record)
# -----------------------------
def main():
    p = argparse.ArgumentParser()

    # 你原有参数（保持）
    p.add_argument("--model_path", required=True, help="Local path or HF id for Llama3-8B")
    p.add_argument("--mesh", default="", help="Single mesh: 64 | 8x8 | 4x16 | 4x4x4")
    p.add_argument("--meshes", default="", help="Comma-separated meshes to batch run, e.g. 64,8x8,4x16")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--timeout_s", type=int, default=1800)

    # CPU-only graph/compile: 强烈建议 tiny seq/batch
    p.add_argument("--device", default="cpu", choices=["cpu"])
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=16)

    # 落盘控制：默认极简（但本脚本会额外记录更多原始文件，便于后续分析）
    p.add_argument("--save_fx", type=int, default=0, help="1 to save fx graphmodule (.pt). default 0")
    p.add_argument("--save_fx_readable", type=int, default=1, help="1 to save a readable fx graph text file. default 1")
    p.add_argument("--write_csv", type=int, default=1)
    p.add_argument("--debug_dump", type=int, default=0, help="1 to dump extra raw objects (may be large)")

    # 记录额外环境信息（建议保持 1）
    p.add_argument("--record_env", type=int, default=1, help="1 to record pip freeze and torch env info. default 1")

    # mesh 维度 swap 敏感性（会额外做一次求解；纯记录，不分析）
    p.add_argument("--do_dim_permute_sensitivity", type=int, default=1)

    # 内存节省：默认不启用，避免影响 autoparallel 测试稳定性
    p.add_argument(
        "--allow_cpu_low_precision",
        type=int,
        default=0,
        help="1 to allow bf16/fp16 on CPU (may reduce memory; may affect export/ops support). Default 0 keeps fp32.",
    )

    args = p.parse_args()

    if not args.mesh and not args.meshes:
        raise ValueError("Provide --mesh or --meshes")

    mesh_list = []
    if args.meshes:
        mesh_list.extend([x.strip() for x in args.meshes.split(",") if x.strip()])
    if args.mesh:
        mesh_list.append(args.mesh.strip())

    # Load model once (成本很高，别在每个 mesh 重复加载)
    model, tok = load_llama3(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        allow_cpu_low_precision=bool(args.allow_cpu_low_precision),
    )
    example_inputs = build_example_inputs(tok, args.batch, args.seq_len, device=args.device)

    # Capture once（图本身与 mesh 无关；mesh 只影响 placements/求解）
    fx_cache = None

    # 全局记录：环境、版本、命令行参数（只写一次，放 outdir 根目录）
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

    # 每个 mesh 逐个 run
    for mesh_str in mesh_list:
        mesh_shape = parse_mesh(mesh_str)
        tag = mesh_tag(mesh_shape)
        run_dir = make_run_dir(args.outdir, tag)

        # 记录 meta（尽量全）
        meta = {
            "mesh": {"raw": mesh_str, "shape": mesh_shape, "tag": tag},
            "time": datetime.utcnow().isoformat() + "Z",
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
            "capture": {"batch": args.batch, "seq_len": args.seq_len},
            "run_dir": run_dir,
            "argv": sys.argv,
        }

        # 记录 example inputs meta
        write_json(os.path.join(run_dir, "example_inputs.json"), example_inputs_meta(example_inputs))

        # capture fx once
        if fx_cache is None:
            fx_pt_path = os.path.join(run_dir, "fx.pt") if args.save_fx else ""
            fx_txt_path = os.path.join(run_dir, "fx_readable.txt") if args.save_fx_readable else ""
            fx_cache = capture_fx(model, example_inputs, save_fx_path=fx_pt_path, save_fx_readable_path=fx_txt_path)
        else:
            # 若 save_fx=1 但你希望每个 mesh 都落一份 fx，也可以在这里复制；默认不做
            pass

        gm = fx_cache["gm"]
        meta["fx_meta"] = fx_cache["meta"]

        # 先写 meta，确保中断也有记录
        write_json(os.path.join(run_dir, "meta.json"), meta)

        t_all0 = time.perf_counter()

        # solve with timeout
        def solve_fn():
            return try_autoparallel_solve(gm, example_inputs, mesh_shape, run_dir)

        solve_res = run_with_timeout(solve_fn, args.timeout_s)

        # dim permute sensitivity record (optional; extra solve)
        sens = None
        if args.do_dim_permute_sensitivity and len(mesh_shape) >= 2:
            sens = dim_permute_sensitivity_record(gm, example_inputs, mesh_shape, run_dir)

        total_s = time.perf_counter() - t_all0

        # 记录汇总（注意：这里只是“实验记录”，不做任何好坏判断）
        summary = {
            "mesh_tag": tag,
            "mesh_shape": mesh_shape,
            "solve_status": solve_res.get("solve_status"),
            "solve_time_s": solve_res.get("solve_time_s"),
            "timeout": bool(solve_res.get("timeout", False)),
            "fallback": bool(solve_res.get("fallback", False)),
            "scale": solve_res.get("scale", {}),
            "costs": solve_res.get("costs", {}),
            # 原始信息保留：repr + json-friendly + pickle 保存结果
            "raw_repr": solve_res.get("raw_repr", ""),
            "autoparallel_import": solve_res.get("autoparallel_import"),
            "autoparallel_entry": solve_res.get("autoparallel_entry"),
            "call_note": solve_res.get("call_note"),
            "out_pickle": solve_res.get("out_pickle"),
            "dim_sensitivity": sens,
            "total_time_s": total_s,
            "run_dir": run_dir,
            "meta_path": os.path.join(run_dir, "meta.json"),
        }

        # 每个 mesh 写 summary.json；全局聚合写 runs.jsonl（可用于后续分析脚本读取）
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
                    "total_time_s": total_s,
                    "run_dir": run_dir,
                },
            )

        if args.debug_dump:
            # 可选：把更深的 json-friendly out 落盘（可能较大）
            try:
                write_json(os.path.join(run_dir, "autoparallel_out_deep.json"), _json_friendly(solve_res.get("strategy"), max_depth=8))
            except Exception as e:
                write_json(os.path.join(run_dir, "autoparallel_out_deep.json"), {"__error__": repr(e)})

        print(f"[OK] mesh={tag} -> {run_dir}")


if __name__ == "__main__":
    main()
