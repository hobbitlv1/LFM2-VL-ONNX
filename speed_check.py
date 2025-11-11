from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

DEFAULT_VALIDATION_SET = Path("quant/validation_dataset.pt")
REFERENCE_CANDIDATES = (
    Path("onnx_optimum/model.onnx"),
    Path("onnx_optimum/preprocessed/model.pre.ort.onnx"),
    Path("onnx_optimum/model.inferred.onnx"),

)
MODEL_CANDIDATES = (
    Path("quant/model_dynamic.quant.onnx"),
    Path("quant/model.quant.onnx"),
)


class BenchRes:
    def __init__(self, path, prov, runs, warmup, avg, p50, p90, p95, p99, tput, lats):
        self.model_path = path
        self.provider = prov
        self.runs = runs
        self.warmup = warmup
        self.avg_ms = avg
        self.p50_ms = p50
        self.p90_ms = p90
        self.p95_ms = p95
        self.p99_ms = p99
        self.throughput_ips = tput
        self.latencies_ms = lats

    def to_dict(self):
        return {
            "model_path": str(self.model_path),
            "provider": self.provider,
            "runs": self.runs,
            "warmup": self.warmup,
            "avg_ms": self.avg_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "throughput_ips": self.throughput_ips,
            "latencies_ms": self.latencies_ms,
        }


def to_np(tensor):
    arr = tensor.detach().cpu().numpy()
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32, copy=False)
    elif arr.dtype == np.int32:
        arr = arr.astype(np.int64, copy=False)
    return np.ascontiguousarray(arr)


def dedup(items):
    seen = set()
    result = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def load_inputs(ds_path, idx):
    ds_path = ds_path.expanduser()
    if not ds_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {ds_path}")

    data = torch.load(ds_path, map_location="cpu")
    samps = data.get("samples") or []
    if not samps:
        raise RuntimeError(f"Dataset {ds_path} does not contain any validation samples.")

    i = idx % len(samps)
    entry = samps[i]
    tens = entry.get("inputs")
    if not tens:
        raise RuntimeError(f"Sample #{i} is missing input tensors.")

    return {name: to_np(t) for name, t in tens.items()}


def check_path(path):
    cand = path.expanduser()
    if not cand.exists():
        raise FileNotFoundError(f"Model not found: {cand}")
    return cand


def find_models(ref, cands):
    sel = []

    if ref:
        sel.append(check_path(ref))
    else:
        for cand in REFERENCE_CANDIDATES:
            if cand.exists():
                sel.append(cand)
                break

    if cands:
        for item in cands:
            sel.append(check_path(item))
    else:
        for cand in MODEL_CANDIDATES:
            if cand.exists():
                sel.append(cand)

    sel = dedup(sel)
    if not sel:
        raise RuntimeError(
            "No ONNX models to benchmark. Provide --reference/--candidate paths or "
            "drop models into the standard locations."
        )
    return sel


def find_provs(req, cpu_only):
    avail = ort.get_available_providers()
    if not avail:
        raise RuntimeError("onnxruntime did not report any available execution providers.")

    provs = []
    if req:
        for prov in req:
            if prov not in avail:
                raise ValueError(f"Requested provider {prov} is not available. Available providers: {avail}")
            if cpu_only and prov != "CPUExecutionProvider":
                raise ValueError("CPU-only mode only supports CPUExecutionProvider.")
            provs.append(prov)
    else:
        provs.append("CPUExecutionProvider")
        if not cpu_only and "CUDAExecutionProvider" in avail:
            provs.append("CUDAExecutionProvider")

    provs = dedup(provs)
    if cpu_only and provs != ["CPUExecutionProvider"]:
        provs = ["CPUExecutionProvider"]

    return provs


def bench_model(path, inputs, prov, runs, warmup, intra, inter, opt):
    if runs <= 0:
        raise ValueError("Number of benchmark runs must be positive.")
    if warmup < 0:
        raise ValueError("Warmup iterations cannot be negative.")

    opts = ort.SessionOptions()
    if intra is not None:
        opts.intra_op_num_threads = int(intra)
    if inter is not None:
        opts.inter_op_num_threads = int(inter)
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL if opt
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )

    sess = ort.InferenceSession(str(path), providers=[prov], sess_options=opts)

    for _ in range(warmup):
        sess.run(None, inputs)

    lats = []
    for _ in range(runs):
        start = time.perf_counter()
        sess.run(None, inputs)
        end = time.perf_counter()
        lats.append((end - start) * 1000.0)

    lats.sort()
    arr = np.asarray(lats, dtype=np.float64)
    avg = float(arr.mean())
    p50, p90, p95, p99 = (float(x) for x in np.percentile(arr, [50, 90, 95, 99]))
    tput = 1000.0 / avg if avg > 0 else float("inf")

    return BenchRes(path, prov, runs, warmup, avg, p50, p90, p95, p99, tput, list(lats))


def show_results(res):
    lines = []
    baselines = {}

    for r in res:
        base = baselines.setdefault(r.provider, r.avg_ms)
        speedup = (base / r.avg_ms) if (base and r.avg_ms) else 1.0
        line = (
            f"{r.provider:<22} {r.model_path}\n"
            f"  avg: {r.avg_ms:.2f} ms | p50: {r.p50_ms:.2f} | "
            f"p90: {r.p90_ms:.2f} | p95: {r.p95_ms:.2f} | p99: {r.p99_ms:.2f}\n"
            f"  throughput: {r.throughput_ips:.2f} infer/s | "
        )
        lines.append(line)
    return lines


def save_res(res, out):
    out = out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in res]
    out.write_text(json.dumps(data, indent=2))
    print(f"Saved raw measurements to {out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime latency for FP32 and quantized models.")
    parser.add_argument(
        "--reference", type=Path, default=None,
        help=f"Path to the baseline FP32 model. Defaults to the first existing path in "
        f"{[str(p) for p in REFERENCE_CANDIDATES]}.",
    )
    parser.add_argument(
        "--candidate", dest="candidates", action="append", type=Path, default=None,
        help="Additional ONNX model to benchmark. Repeat for multiple models.",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_VALIDATION_SET,
        help="Path to the prepared validation dataset (.pt) that provides sample inputs.",
    )
    parser.add_argument(
        "--sample-index", type=int, default=0,
        help="Which validation sample to feed into the models (default: 0).",
    )
    parser.add_argument(
        "--providers", nargs="+", default=None,
        help="Explicit list of execution providers to test. Defaults to CPU plus CUDA when available.",
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Restrict benchmarking to CPUExecutionProvider.",
    )
    parser.add_argument(
        "--runs", type=int, default=50,
        help="Number of timed inference runs per model/provider.",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup runs to perform before timing.",
    )
    parser.add_argument(
        "--intra-op-threads", type=int, default=None,
        help="Optional override for ORT intra_op_num_threads.",
    )
    parser.add_argument(
        "--inter-op-threads", type=int, default=None,
        help="Optional override for ORT inter_op_num_threads.",
    )
    parser.add_argument(
        "--disable-graph-optimizations", action="store_true",
        help="Disable ONNX Runtime graph optimizations when creating sessions.",
    )
    parser.add_argument(
        "--save-json", type=Path, default=None,
        help="Optional path to write the raw benchmark measurements as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        inputs = load_inputs(args.dataset, args.sample_index)
    except Exception as exc:
        raise SystemExit(f"Failed to load validation inputs: {exc}") from exc

    try:
        models = find_models(args.reference, args.candidates)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    try:
        provs = find_provs(args.providers, args.cpu_only)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    prov_list = ", ".join(provs)
    print(f"Benchmarking providers: [{prov_list}]")
    print(f"Loaded sample #{args.sample_index} from {args.dataset} (shape set: {list(inputs)})")

    res = []
    fails = []
    for prov in provs:
        for path in models:
            print(f"\nBenchmarking {path.name} on {prov}...")
            try:
                r = bench_model(
                    path, inputs, prov, args.runs, args.warmup,
                    args.intra_op_threads, args.inter_op_threads,
                    not args.disable_graph_optimizations,
                )
                res.append(r)
            except Exception as exc:
                fail = f"{path} on {prov}: {exc}"
                fails.append(fail)
                print(f"  FAILED: {exc}")

    if res:
        print("BENCHMARK RESULTS")
        for line in show_results(res):
            print(line)

        if args.save_json:
            save_res(res, args.save_json)
    else:
        raise SystemExit("All benchmarks failed; nothing to report.")

    if fails:
        print("\nSome benchmarks failed:")
        for fail in fails:
            print(f"- {fail}")


if __name__ == "__main__":
    main()
