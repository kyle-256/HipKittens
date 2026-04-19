#!/usr/bin/env python3
"""R41 Opt D — narrow R40C re-bench (single shape, 5 runs).

Imports the worktree R40C bench module and calls bench_one_shape() N times for
the target shape. Avoids needing --only on the R40C bench harness.
"""
import json, os, sys, time, importlib.util

WORKTREE = "/shared_nfs/kyle/test/HipKittens/.claude/worktrees/agent-afccfa0b/analysis/fp8_gemm/mi350x"
sys.path.insert(0, WORKTREE)

# Import the R40C bench module so its BUILD_DIR / find_so / runner script work
spec = importlib.util.spec_from_file_location("bench_r40c", os.path.join(WORKTREE, "bench_all_42_R40C.py"))
bench_r40c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench_r40c)


def main():
    # The shape (M, N, K) and competitor TFLOPS
    target = (16384, 4096, 14336)
    comp = None
    for s in bench_r40c.ALL_SHAPES:
        if (s[0], s[1], s[2]) == target:
            comp = s[3]
            break
    if comp is None:
        print("FATAL: shape not in ALL_SHAPES")
        sys.exit(1)
    M, N, K = target
    GPU = int(os.environ.get("GPU", "1"))
    N_RUNS = int(os.environ.get("N_RUNS", "5"))

    print(f"R41D R40C re-bench: shape={target} comp={comp} GPU={GPU} runs={N_RUNS}")
    runs = []
    t0 = time.time()
    for i in range(N_RUNS):
        ts = time.time()
        r = bench_r40c.bench_one_shape(M, N, K, comp, GPU)
        dt = time.time() - ts
        wcf = r.get("wrong_cell_frac")
        snr = r.get("snr_med_db")
        fin = r.get("kernel_finite")
        tflops = r.get("tflops")
        status = r.get("status")
        print(f"  run{i+1}/{N_RUNS}  status={status} tflops={tflops} wcf={wcf} snr_med={snr} finite={fin}  ({dt:.1f}s)", flush=True)
        runs.append(r)
    elapsed = time.time() - t0

    # Summary
    oks = [r for r in runs if r.get("status") == "OK"]
    wcfs = [r.get("wrong_cell_frac") for r in runs if r.get("wrong_cell_frac") is not None]
    pass_count = 0
    for r in runs:
        if r.get("status") == "OK":
            wcf = r.get("wrong_cell_frac", 1.0)
            snr = r.get("snr_med_db", -999)
            fin = r.get("kernel_finite", 0)
            if wcf < 0.02 and snr >= 10.0 and fin >= 0.99:
                pass_count += 1

    if wcfs:
        wcf_mean = sum(wcfs) / len(wcfs)
        wcf_max = max(wcfs)
        wcf_min = min(wcfs)
        wcf_std = (sum((x - wcf_mean) ** 2 for x in wcfs) / len(wcfs)) ** 0.5
    else:
        wcf_mean = wcf_max = wcf_min = wcf_std = None

    promoted = (wcf_max is not None and wcf_max < 0.02 and wcf_std is not None and wcf_std < 0.01)

    out = {
        "shape": list(target), "comp": comp,
        "n_runs": N_RUNS, "pass_count": pass_count,
        "wcf_mean": wcf_mean, "wcf_std": wcf_std, "wcf_max": wcf_max, "wcf_min": wcf_min,
        "promote_under_5run_gate": promoted,
        "tflops_per_run": [r.get("tflops") for r in runs],
        "wcf_per_run": wcfs,
        "snr_med_per_run": [r.get("snr_med_db") for r in runs],
        "finite_per_run": [r.get("kernel_finite") for r in runs],
        "status_per_run": [r.get("status") for r in runs],
        "elapsed_seconds": round(elapsed, 1),
        "runs_full": runs,
    }
    out_path = "/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/R41D_R40C_5RUN_n14336.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"PASS_COUNT: {pass_count}/{N_RUNS}")
    print(f"wcf_mean={wcf_mean} wcf_std={wcf_std} wcf_max={wcf_max}")
    print(f"PROMOTE: {promoted}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
