#!/usr/bin/env python3
"""R42 Opt A Phase 1: NaN positional diagnostic for cluster-B shapes.

For each cluster-B shape (PASS_3/5..FLAKE_2/5 with fin_min < 0.999):
1. Load the .so from R41 manifest.
2. Run kernel N_PROBE times with random scales (re-seeded each run for
   independent draws), record:
   - finite_frac per run
   - bad-cell positions (row,col) per run as a set
   - decomposition: count of (NaN, +inf, -inf, +bf16max overflow ~3.39e38, -bf16max)
3. Compute positional consistency: |intersection| / |union| across runs.
   - 1.0 = deterministic (same cells every run)
   - 0.0 = race (no overlap)

Output JSON: {shape: {fin_per_run, bad_cell_intersect_pct, bad_cell_decomp, ...}}

Run as a worker pool over GPUs 0,1.
"""
import json
import math
import os
import subprocess
import sys
import threading
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

WARMUP = 0           # diagnostic only - no warmup needed for finite check
N_PROBE = 5          # runs per shape for positional analysis (can override)
SHAPE_TIMEOUT = 700
RANDOM_SEED_BASE = 42  # we use seed_base + run_idx for independent draws
INPUT_REUSE = True   # if True, same A,B,scales every run (deterministic-seeded)


def make_diagnostic_script(module_name, so_path, m, n, k, n_probe, seed_base, input_reuse):
    return f"""\
import sys, math, json, importlib.util, torch
WARMUP = {WARMUP}
N_PROBE = {n_probe}
SEED_BASE = {seed_base}
INPUT_REUSE = {input_reuse}
M, N, K = {m}, {n}, {k}

FP4_TBL = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0,
], dtype=torch.float32, device='cuda')

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

    BF16_MAX = 3.3895e38  # bf16 finite max approximation
    BF16_MAX_LOWER = 3.0e38  # threshold for "near max" overflow detection

    per_run = []
    fixed_inputs = None

    for ri in range(N_PROBE):
        torch.manual_seed(SEED_BASE + (0 if INPUT_REUSE else ri))
        if INPUT_REUSE and fixed_inputs is not None:
            A, B, sc_a, sc_b, A_sc, B_sc = fixed_inputs
        else:
            A = gen_fp4(M, K); B = gen_fp4(N, K)
            sc_a = torch.randint(-2, 3, (M, K//32), dtype=torch.int8, device='cuda')
            sc_b = torch.randint(-2, 3, (N, K//32), dtype=torch.int8, device='cuda')
            A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)
            if INPUT_REUSE:
                fixed_inputs = (A, B, sc_a, sc_b, A_sc, B_sc)

        C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
        mod.gemm_rcr(A, B, A_sc, B_sc, C)
        torch.cuda.synchronize()

        Cf = C.float()
        finite_mask = torch.isfinite(Cf)
        nan_mask = torch.isnan(Cf)
        pos_inf_mask = torch.isinf(Cf) & (Cf > 0)
        neg_inf_mask = torch.isinf(Cf) & (Cf < 0)
        # bf16 overflow: |c| > 3e38 but finite
        # Note: bf16 max is ~3.39e38; values that overflow saturate to inf typically,
        # but compiler-generated incorrect values may land near max if not full overflow.
        bf16_overflow_mask = finite_mask & (Cf.abs() > BF16_MAX_LOWER)

        n_total = M * N
        n_finite = int(finite_mask.sum().item())
        n_nan = int(nan_mask.sum().item())
        n_pinf = int(pos_inf_mask.sum().item())
        n_ninf = int(neg_inf_mask.sum().item())
        n_overflow = int(bf16_overflow_mask.sum().item())
        finite_frac = n_finite / n_total

        # Positions of bad cells: union of (NaN, Inf)
        bad_mask = ~finite_mask
        # If too many bad cells, sample positions
        bad_count = int(bad_mask.sum().item())
        # Convert bad cell positions to flat indices, capped to 50k
        flat_idx = torch.nonzero(bad_mask.flatten(), as_tuple=False).squeeze(-1)
        if flat_idx.numel() > 50000:
            # Sort and take stable subset (deterministic for same kernel output)
            flat_idx_sorted, _ = torch.sort(flat_idx)
            sample = flat_idx_sorted[:50000]
        else:
            sample, _ = torch.sort(flat_idx)
        idx_list = sample.cpu().tolist()

        per_run.append({{
            "run_idx": ri,
            "finite_frac": round(finite_frac, 6),
            "n_total": n_total,
            "n_finite": n_finite,
            "n_nan": n_nan,
            "n_pinf": n_pinf,
            "n_ninf": n_ninf,
            "n_bf16_overflow_finite": n_overflow,
            "n_bad": bad_count,
            "bad_idx_sample_count": len(idx_list),
            "bad_idx_sample": idx_list,
        }})

    # Compute positional consistency across runs
    sets = [set(r["bad_idx_sample"]) for r in per_run]
    if sets and all(len(s) > 0 for s in sets):
        intersect = set.intersection(*sets)
        union = set.union(*sets)
        # Jaccard
        jaccard = len(intersect) / len(union) if union else 0.0
        # Pairwise overlap
        pair_overlaps = []
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                a, b = sets[i], sets[j]
                u = a | b
                if u:
                    pair_overlaps.append(len(a & b) / len(u))
        avg_pair_jaccard = sum(pair_overlaps) / len(pair_overlaps) if pair_overlaps else 0.0
    elif sets and any(len(s) == 0 for s in sets) and any(len(s) > 0 for s in sets):
        jaccard = 0.0
        avg_pair_jaccard = 0.0
    else:
        jaccard = 1.0  # all empty (no bad cells)
        avg_pair_jaccard = 1.0

    # Strip raw bad_idx_sample for compactness in main JSON (keep only count)
    per_run_summary = []
    for r in per_run:
        rr = dict(r); rr.pop("bad_idx_sample", None); per_run_summary.append(rr)

    out = {{
        "M": M, "N": N, "K": K,
        "n_probe": N_PROBE,
        "input_reuse": INPUT_REUSE,
        "seed_base": SEED_BASE,
        "per_run": per_run_summary,
        "jaccard_intersect_over_union": round(jaccard, 6),
        "avg_pairwise_jaccard": round(avg_pair_jaccard, 6),
        "fin_min": round(min(r["finite_frac"] for r in per_run), 6),
        "fin_max": round(max(r["finite_frac"] for r in per_run), 6),
        "fin_mean": round(sum(r["finite_frac"] for r in per_run)/len(per_run), 6),
        "n_bad_min": min(r["n_bad"] for r in per_run),
        "n_bad_max": max(r["n_bad"] for r in per_run),
        # Aggregate decomposition (sum across runs)
        "agg_n_nan": sum(r["n_nan"] for r in per_run),
        "agg_n_pinf": sum(r["n_pinf"] for r in per_run),
        "agg_n_ninf": sum(r["n_ninf"] for r in per_run),
        "agg_n_bf16_overflow_finite": sum(r["n_bf16_overflow_finite"] for r in per_run),
    }}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")

except torch.cuda.OutOfMemoryError:
    out = {{"M": M, "N": N, "K": K, "status": "OOM"}}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
except Exception as e:
    out = {{"M": M, "N": N, "K": K, "status": f"ERR:{{e}}"[:200]}}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
"""


def diag_one_shape(m, n, k, so_path, source, gpu_id, n_probe, input_reuse):
    if not os.path.exists(so_path):
        return {"M": m, "N": n, "K": k, "status": "MISSING_SO", "source": source}

    module_name = os.path.basename(so_path).split(".")[0]
    script = make_diagnostic_script(module_name, so_path, m, n, k, n_probe,
                                    RANDOM_SEED_BASE, input_reuse)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        r = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=SHAPE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"M": m, "N": n, "K": k, "status": "TIMEOUT", "source": source}

    if r.returncode != 0:
        return {"M": m, "N": n, "K": k, "status": "CRASH", "source": source,
                "stderr_tail": r.stderr[-200:]}

    try:
        s = r.stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = r.stdout.index("BENCH_JSON_END")
        out = json.loads(r.stdout[s:e].strip())
        out["source"] = source
        out["so_path"] = so_path
        return out
    except (ValueError, json.JSONDecodeError):
        return {"M": m, "N": n, "K": k, "status": "PARSE_FAIL", "source": source,
                "stdout_tail": r.stdout[-300:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--n-probe", type=int, default=N_PROBE)
    ap.add_argument("--input-reuse", action="store_true", default=True,
                    help="If true, fix inputs (same seed every run); only kernel-side variation")
    ap.add_argument("--input-vary", dest="input_reuse", action="store_false",
                    help="Vary inputs each run")
    ap.add_argument("--cluster-b-json", default="R42_OPT_A_CLUSTER_B_SHAPES.json")
    ap.add_argument("--out", default="R42_OPT_A_PHASE1_DIAGNOSTIC.json")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    cb = json.load(open(os.path.join(SCRIPT_DIR, args.cluster_b_json)))
    print(f"R42 Opt A Phase1: {len(cb)} cluster-B shapes, n_probe={args.n_probe}, "
          f"input_reuse={args.input_reuse}, GPUs={gpus}")

    jobs = [(c["shape"], c["so_path"], c["source"]) for c in cb]
    gpu_tasks = {g: [] for g in gpus}
    for i, j in enumerate(jobs):
        gpu_tasks[gpus[i % len(gpus)]].append((i, j))

    results = [None] * len(jobs)
    lock = threading.Lock()
    t0 = time.time()
    done = [0]

    def run_gpu(gpu_id):
        for (i, (shape, so_path, source)) in gpu_tasks[gpu_id]:
            m, n, k = [int(x) for x in shape.split("x")]
            r = diag_one_shape(m, n, k, so_path, source, gpu_id, args.n_probe,
                              args.input_reuse)
            with lock:
                results[i] = r
                done[0] += 1
                jac = r.get("avg_pairwise_jaccard")
                fmin = r.get("fin_min")
                fmax = r.get("fin_max")
                if jac is not None and fmin is not None:
                    print(f"  [{done[0]:>2}/{len(jobs)}] {shape:20s} [{source}] "
                          f"fin_min={fmin:.4f} fin_max={fmax:.4f} jaccard={jac:.3f} "
                          f"GPU{gpu_id}", flush=True)
                else:
                    print(f"  [{done[0]:>2}/{len(jobs)}] {shape:20s} [{source}] "
                          f"status={r.get('status', 'OK')} GPU{gpu_id}", flush=True)

    threads = []
    for g in gpus:
        t = threading.Thread(target=run_gpu, args=(g,))
        t.start(); threads.append(t)
    for t in threads: t.join()

    elapsed = time.time() - t0

    out_path = os.path.join(SCRIPT_DIR, args.out)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": "R42_OPT_A_PHASE1",
            "n_probe": args.n_probe,
            "input_reuse": args.input_reuse,
            "gpus": gpus,
            "elapsed_minutes": round(elapsed / 60, 2),
            "shapes": {(c["shape"]): r for c, r in zip(cb, results)},
        }, f, indent=2)

    print(f"\nDone in {elapsed/60:.1f} min. Output: {out_path}")

    # Summary table
    print("\n=== Phase 1 Summary ===")
    print(f"{'shape':22s} {'src':6s} {'fin_min':>8s} {'fin_max':>8s} {'jaccard':>8s} "
          f"{'verdict':>14s}")
    for c, r in zip(cb, results):
        shape = c["shape"]
        src = c["source"]
        fmin = r.get("fin_min")
        fmax = r.get("fin_max")
        jac = r.get("avg_pairwise_jaccard")
        # Heuristic verdict
        if jac is None or fmin is None:
            v = r.get("status", "ERR")
        elif jac >= 0.95:
            v = "DETERMINISTIC"
        elif jac >= 0.50:
            v = "PARTIAL_RACE"
        else:
            v = "RACE"
        print(f"{shape:22s} {src:6s} {fmin if fmin is not None else 0.0:>8.4f} "
              f"{fmax if fmax is not None else 0.0:>8.4f} "
              f"{jac if jac is not None else 0.0:>8.3f} {v:>14s}")


if __name__ == "__main__":
    main()
