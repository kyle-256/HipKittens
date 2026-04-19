#!/usr/bin/env python3
"""R38 Opt E: bench every shape's _R38E module with correctness gate.

Same harness as bench_all_42_R37.py but loads from build_R38E/. Module names
are derived from R38_BEST_VARIANTS_v3.BEST_VARIANTS_V3 + macro signature
(matches build_R38E.module_name_for).
"""
import json
import os
import subprocess
import sys
import sysconfig
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R38E")

sys.path.insert(0, SCRIPT_DIR)
from R38_BEST_VARIANTS_v3 import BEST_VARIANTS_V3

WARMUP = 200
ITERS = 500
TRIM = 0.10
SHAPE_TIMEOUT = 600
CORRECTNESS_GATE = 0.995

ALL_SHAPES = [
    (16384,  4096,  2048, 2995.0), (16384,  4096,  3072, 3492.3),
    (16384,  6144,  2048, 3047.6), (32768,  4096,  2048, 3131.8),
    (32768,  4096,  3072, 3630.6), (32768,  6144,  2048, 3239.9),
    (16384, 14336,  2048, 3301.3), (16384, 28672,  2048, 3482.3),
    (32768, 14336,  2048, 3351.4), (32768, 28672,  2048, 3353.4),
    (4096,   4096,  16384, 4642.1), (4096,  14336,  16384, 5013.0),
    (6144,   4096,  16384, 4428.1), (4096,   4096,   8192, 3959.9),
    (4096,   4096,  32768, 5152.8), (4096,   6144,  32768, 3784.2),
    (4096,  14336,   8192, 4345.8), (4096,  28672,  32768, 5649.9),
    (4096,  32768,  4096, 4166.5), (4096,  32768,  6144, 4548.6),
    (4096,  32768,  14336, 5296.1), (4096,  32768,  28672, 5568.2),
    (4096,  32768, 128256, 5781.1), (4096, 128256,  32768, 3195.3),
    (6144,   4096,   8192, 3822.0), (6144,  32768,   4096, 4291.0),
    (14336,  4096,  32768, 5245.4), (14336, 32768,   4096, 4462.6),
    (16384,  4096,   4096, 3951.8), (16384,  4096,   6144, 4259.9),
    (16384,  4096,   7168, 4443.2), (16384,  4096,  14336, 5142.1),
    (16384,  4096,  28672, 5525.3), (16384,  6144,   4096, 4042.5),
    (16384, 14336,   4096, 4255.8), (16384, 28672,   4096, 4411.7),
    (28672,  4096,   8192, 4810.0), (28672,  4096,  16384, 5350.6),
    (28672, 32768,   4096, 4466.6), (32768,  4096,   7168, 4666.8),
    (32768,  4096,  14336, 5223.4), (128256, 32768,  4096, 4536.4),
]


def macro_signature(macro_overrides):
    if not macro_overrides:
        return ""
    parts = [f"{k}{v}" for k, v in sorted(macro_overrides.items())]
    return "_" + "_".join(parts)


def module_name_for(n_dim, k_dim, parent_tag, macro_overrides):
    sig = macro_signature(macro_overrides)
    return f"tk_mxfp4_gluon_cpp_n{n_dim}_k{k_dim}_{parent_tag}{sig}_R38E"


def make_runner_script(module_name, so_path, m, n, k, comp):
    return f"""\
import sys, math, json, importlib.util, torch
torch.manual_seed(0)
WARMUP, ITERS, TRIM, GATE = {WARMUP}, {ITERS}, {TRIM}, {CORRECTNESS_GATE}
M, N, K = {m}, {n}, {k}
COMP = {comp}

def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda') << 4) | \\
           torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')

def preshuffle(se):
    r, kb = se.shape
    pr = math.ceil(r/64)*64
    pk = math.ceil(kb/8)*8
    raw = torch.full((pr,pk),0x7F,dtype=torch.uint8,device=se.device)
    raw[:r,:kb] = (se.to(torch.int16)+127).to(torch.uint8)
    sh = raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh = sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)

try:
    spec = importlib.util.spec_from_file_location({module_name!r}, {so_path!r})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    A = gen_fp4(M, K); B = gen_fp4(N, K)
    sc_a_correct = torch.full((M, K//32), -4, dtype=torch.int8, device='cuda')
    sc_b_correct = torch.full((N, K//32), -4, dtype=torch.int8, device='cuda')
    A_sc_correct = preshuffle(sc_a_correct); B_sc_correct = preshuffle(sc_b_correct)
    sc_a = torch.randint(-2,3,(M,K//32),dtype=torch.int8,device='cuda')
    sc_b = torch.randint(-2,3,(N,K//32),dtype=torch.int8,device='cuda')
    A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)
    C = torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')

    run_correct = lambda: mod.gemm_rcr(A,B,A_sc_correct,B_sc_correct,C)
    run = lambda: mod.gemm_rcr(A,B,A_sc,B_sc,C)

    C.zero_(); run_correct(); torch.cuda.synchronize()
    finite_frac = float(torch.isfinite(C.float()).sum().item()) / float(C.numel())
    finite_frac = round(finite_frac, 6)

    if finite_frac < GATE:
        out = {{
            "M": M, "N": N, "K": K, "tflops": None, "avg_ms": None, "comp": COMP,
            "status": "WRONG_OUTPUT", "kernel_finite": finite_frac,
        }}
        print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
        sys.exit(0)

    for _ in range(WARMUP): run()
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); run(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    times.sort()
    trim = int(len(times) * TRIM)
    if trim > 0: times = times[trim:-trim]
    avg = sum(times) / len(times)
    tflops = 2.0 * M * N * K / (avg * 1e-3) / 1e12

    out = {{
        "M": M, "N": N, "K": K, "tflops": round(tflops, 1),
        "avg_ms": round(avg, 4), "comp": COMP, "status": "OK",
        "kernel_finite": finite_frac,
    }}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")

except torch.cuda.OutOfMemoryError:
    out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None,
           "comp": COMP, "status": "OOM", "kernel_finite": None}}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
except Exception as e:
    out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None,
           "comp": COMP, "status": f"ERR:{{e}}"[:80], "kernel_finite": None}}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
"""


def bench_one_shape(m, n, k, comp, gpu_id):
    parent_tag, macro_overrides = BEST_VARIANTS_V3[(m, n, k)]
    module_name = module_name_for(n, k, parent_tag, macro_overrides)
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    sig = macro_signature(macro_overrides)
    var_label = parent_tag + sig + "_R38E"

    if not os.path.exists(so_path):
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "MISSING_SO", "kernel_finite": None,
                "best_variant": var_label, "macro_overrides": macro_overrides}

    script = make_runner_script(module_name, so_path, m, n, k, comp)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        r = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=SHAPE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "TIMEOUT", "kernel_finite": None,
                "best_variant": var_label, "macro_overrides": macro_overrides}

    if r.returncode != 0:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "CRASH", "kernel_finite": None,
                "best_variant": var_label, "macro_overrides": macro_overrides,
                "stderr_tail": r.stderr[-200:]}

    try:
        s = r.stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = r.stdout.index("BENCH_JSON_END")
        out = json.loads(r.stdout[s:e].strip())
        out["best_variant"] = var_label
        out["macro_overrides"] = macro_overrides
        return out
    except (ValueError, json.JSONDecodeError):
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "PARSE_FAIL", "kernel_finite": None,
                "best_variant": var_label, "macro_overrides": macro_overrides}


def main():
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    if len(sys.argv) > 1:
        gpus = [int(g) for g in sys.argv[1].split(",")]

    print(f"R38 Opt E Correctness-Gated Bench - GPUs: {gpus}")
    print(f"Shapes: {len(ALL_SHAPES)}, warmup={WARMUP}, iters={ITERS}, trim={TRIM}, "
          f"gate kernel_finite>={CORRECTNESS_GATE}")
    print("=" * 110)

    gpu_tasks = {g: [] for g in gpus}
    for i, (m, n, k, comp) in enumerate(ALL_SHAPES):
        gpu = gpus[i % len(gpus)]
        gpu_tasks[gpu].append((i, m, n, k, comp))

    results = [None] * len(ALL_SHAPES)
    lock = threading.Lock()
    t0 = time.time()

    def run_gpu(gpu_id):
        for (i, m, n, k, comp) in gpu_tasks[gpu_id]:
            r = bench_one_shape(m, n, k, comp, gpu_id)
            with lock:
                results[i] = r
                tag = r["best_variant"]
                if r["status"] == "OK":
                    ratio = r["tflops"] / comp * 100
                    flag = "WIN" if r["tflops"] >= comp else "LOSE"
                    print(f"  [{i+1:>2}/42] {m:>6}x{n:>6}x{k:>6}  "
                          f"{r['tflops']:>7.1f} vs {comp:>7.1f} ({ratio:>5.1f}%) "
                          f"finite={r['kernel_finite']:.4f}  {flag}  GPU{gpu_id}", flush=True)
                else:
                    fin = r.get("kernel_finite")
                    fin_s = f"{fin:.4f}" if fin is not None else "N/A"
                    print(f"  [{i+1:>2}/42] {m:>6}x{n:>6}x{k:>6}  "
                          f"{r['status']}  finite={fin_s}  GPU{gpu_id}", flush=True)

    threads = []
    for g in gpus:
        t = threading.Thread(target=run_gpu, args=(g,))
        t.start(); threads.append(t)
    for t in threads: t.join()

    elapsed = time.time() - t0
    print()
    print("=" * 110)
    wins = losses = wrong = errs = 0
    for r in results:
        if r is None: errs += 1; continue
        if r["status"] == "OK":
            if r["tflops"] >= r["comp"]: wins += 1
            else: losses += 1
        elif r["status"] == "WRONG_OUTPUT": wrong += 1
        else: errs += 1

    print(f"WIN: {wins}/42  LOSE: {losses}/42  WRONG_OUTPUT: {wrong}/42  ERR: {errs}/42  ({elapsed/60:.1f} min)")

    out_path = os.path.join(SCRIPT_DIR, "bench_all42_results_R38_optE.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": "R38_optE",
            "warmup": WARMUP, "iters": ITERS, "trim_frac": TRIM,
            "correctness_gate_finite": CORRECTNESS_GATE,
            "n_shapes": len(ALL_SHAPES),
            "wins": wins, "losses": losses, "wrong_output": wrong, "errors": errs,
            "elapsed_minutes": round(elapsed / 60, 1),
            "gpus": gpus,
            "results": [r for r in results if r],
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
