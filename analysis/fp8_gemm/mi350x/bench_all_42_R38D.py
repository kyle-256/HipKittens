#!/usr/bin/env python3
"""R38 Opt D: Bench every candidate variant for the 19 R37 WRONG_OUTPUT shapes
with the correctness gate. For each shape, find the fastest variant with
kernel_finite >= 0.995. Combine with R37's 14 WIN entries to form
R38_BEST_VARIANTS_v2.
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
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R38D")

WARMUP = 200
ITERS = 500
TRIM = 0.10
SHAPE_TIMEOUT = 600
CORRECTNESS_GATE = 0.995


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


def bench_one_variant(m, n, k, comp, parent_tag, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{n}_k{k}_{parent_tag}_R38D"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")

    if not os.path.exists(so_path):
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "MISSING_SO", "kernel_finite": None,
                "variant": parent_tag}

    script = make_runner_script(module_name, so_path, m, n, k, comp)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        r = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=SHAPE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "TIMEOUT", "kernel_finite": None,
                "variant": parent_tag}

    if r.returncode != 0:
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "CRASH", "kernel_finite": None,
                "variant": parent_tag,
                "stderr_tail": r.stderr[-200:]}

    try:
        s = r.stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = r.stdout.index("BENCH_JSON_END")
        out = json.loads(r.stdout[s:e].strip())
        out["variant"] = parent_tag
        return out
    except (ValueError, json.JSONDecodeError):
        return {"M": m, "N": n, "K": k, "tflops": None, "avg_ms": None,
                "comp": comp, "status": "PARSE_FAIL", "kernel_finite": None,
                "variant": parent_tag}


def load_candidates():
    with open(os.path.join(SCRIPT_DIR, "R38D_candidates.json")) as f:
        return json.load(f)


def main():
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    if len(sys.argv) > 1:
        gpus = [int(g) for g in sys.argv[1].split(",")]

    cands = load_candidates()

    # Build a flat task list: (m, n, k, comp, variant_tag).
    tasks = []
    for shape_str, info in cands["shapes"].items():
        if info["status"] != "have_candidates":
            continue
        m, n, k, comp = info["M"], info["N"], info["K"], info["comp"]
        for c in info["candidates"]:
            tasks.append((m, n, k, comp, c["variant"]))

    print(f"R38D Bench - GPUs: {gpus}")
    print(f"Tasks: {len(tasks)}, warmup={WARMUP}, iters={ITERS}, trim={TRIM}, "
          f"gate kernel_finite>={CORRECTNESS_GATE}")
    print("=" * 110)

    gpu_tasks = {g: [] for g in gpus}
    for i, t in enumerate(tasks):
        gpu = gpus[i % len(gpus)]
        gpu_tasks[gpu].append((i, *t))

    results = [None] * len(tasks)
    lock = threading.Lock()
    t0 = time.time()

    def run_gpu(gpu_id):
        for (i, m, n, k, comp, tag) in gpu_tasks[gpu_id]:
            r = bench_one_variant(m, n, k, comp, tag, gpu_id)
            with lock:
                results[i] = r
                if r["status"] == "OK":
                    ratio = r["tflops"] / comp * 100
                    flag = "WIN" if r["tflops"] >= comp else "LOSE"
                    print(f"  [{i+1:>3}/{len(tasks)}] {m:>6}x{n:>6}x{k:>6} {tag:<48} "
                          f"{r['tflops']:>7.1f} ({ratio:>5.1f}%) finite={r['kernel_finite']:.4f}  {flag}  GPU{gpu_id}", flush=True)
                else:
                    fin = r.get("kernel_finite")
                    fin_s = f"{fin:.4f}" if fin is not None else "N/A"
                    print(f"  [{i+1:>3}/{len(tasks)}] {m:>6}x{n:>6}x{k:>6} {tag:<48} "
                          f"{r['status']}  finite={fin_s}  GPU{gpu_id}", flush=True)

    threads = []
    for g in gpus:
        t = threading.Thread(target=run_gpu, args=(g,))
        t.start(); threads.append(t)
    for t in threads: t.join()

    elapsed = time.time() - t0
    print()
    print("=" * 110)

    # Aggregate per-shape
    per_shape = {}
    for r in results:
        if r is None:
            continue
        key = (r["M"], r["N"], r["K"])
        per_shape.setdefault(key, []).append(r)

    summary = {}
    for (m, n, k), entries in per_shape.items():
        comp = entries[0]["comp"]
        ok_entries = [e for e in entries if e["status"] == "OK" and e.get("kernel_finite", 0) >= CORRECTNESS_GATE]
        ok_entries.sort(key=lambda e: -e["tflops"])
        wrong = [e for e in entries if e["status"] == "WRONG_OUTPUT"]
        crash = [e for e in entries if e["status"] not in ("OK", "WRONG_OUTPUT")]
        if ok_entries:
            best = ok_entries[0]
            summary[f"{m}x{n}x{k}"] = {
                "M": m, "N": n, "K": k, "comp": comp,
                "best_variant": best["variant"],
                "best_tflops": best["tflops"],
                "best_ratio_vs_comp": round(best["tflops"]/comp*100, 1),
                "best_finite": best["kernel_finite"],
                "n_correct": len(ok_entries),
                "n_wrong": len(wrong),
                "n_crash": len(crash),
                "all_correct_sorted": [
                    {"v": e["variant"], "t": e["tflops"], "f": e["kernel_finite"]} for e in ok_entries
                ],
            }
        else:
            summary[f"{m}x{n}x{k}"] = {
                "M": m, "N": n, "K": k, "comp": comp,
                "best_variant": None,
                "best_tflops": None,
                "n_correct": 0,
                "n_wrong": len(wrong),
                "n_crash": len(crash),
                "wrong_variants": [e["variant"] for e in wrong],
                "crash_variants": [(e["variant"], e["status"]) for e in crash],
            }

    out_path = os.path.join(SCRIPT_DIR, "bench_all42_results_R38_optD.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": "R38_optD",
            "warmup": WARMUP, "iters": ITERS, "trim_frac": TRIM,
            "correctness_gate_finite": CORRECTNESS_GATE,
            "n_tasks": len(tasks),
            "n_shapes": len(per_shape),
            "elapsed_minutes": round(elapsed / 60, 1),
            "gpus": gpus,
            "per_shape": summary,
            "raw_results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    n_with_correct = sum(1 for v in summary.values() if v["best_variant"] is not None)
    n_winning = sum(1 for v in summary.values() if v["best_variant"] is not None and v["best_tflops"] >= v["comp"])
    print(f"Per-shape: {n_with_correct}/{len(per_shape)} have at least one correct variant; "
          f"{n_winning}/{len(per_shape)} have a WIN (correct AND ≥comp).")
    print(f"Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
