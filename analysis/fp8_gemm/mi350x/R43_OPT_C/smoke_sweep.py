#!/usr/bin/env python3
"""R43 Opt C — Phase 1 Smoke Sweep.

For each (shape, candidate.so) combination, run a single-rep bench
at warmup=200/iters=500/trim=0.10 with FINITE_GATE=0.98.

Output: R43_OPT_C_SWEEP_SMOKE.json
"""
import os, sys, json, time, math, subprocess, threading, argparse

SCRIPT_DIR = "/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x"
WARMUP = 200
ITERS = 500
TRIM = 0.10
SHAPE_TIMEOUT = 700
SNR_THRESHOLD_DB = 10.0
WRONG_CELL_GATE = 0.02
FINITE_GATE = 0.98
RANDOM_SEED = 42

# Comp tflops table — lookup by shape
COMP_TFLOPS = {
    "14336x4096x32768": 5245.4,
    "4096x28672x32768": 5649.9,
    "4096x32768x128256": 5781.1,
    "4096x4096x32768": 5152.8,
    "4096x6144x32768": 3784.2,
    "6144x4096x16384": 4428.1,
    "4096x14336x16384": 5013.0,
    "14336x32768x4096": 4462.6,
    "16384x4096x14336": 5142.1,
    "28672x32768x4096": 4466.6,
    "6144x4096x8192": 3822.0,
    "4096x14336x8192": 4345.8,
    "4096x32768x4096": 4166.5,
    "6144x32768x4096": 4291.0,
}


def make_runner_script(module_name, so_path, m, n, k, comp):
    return f"""\
import sys, math, json, importlib.util, torch
torch.manual_seed({RANDOM_SEED})
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
SNR_DB_GATE = {SNR_THRESHOLD_DB}
WRONG_CELL_GATE = {WRONG_CELL_GATE}
FINITE_GATE = {FINITE_GATE}
M, N, K = {m}, {n}, {k}
COMP = {comp}

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

def dequant_fp4_rows(packed_uint8, K, row_lo, row_hi):
    pk = packed_uint8[row_lo:row_hi]
    lo = (pk & 0x0F).to(torch.int64)
    hi = ((pk >> 4) & 0x0F).to(torch.int64)
    rows, cols = pk.shape
    out = torch.empty(rows, cols * 2, dtype=torch.float32, device=pk.device)
    out[:, 0::2] = FP4_TBL[lo]
    out[:, 1::2] = FP4_TBL[hi]
    return out[:, :K]

def apply_scales_rows(data_f32, scale_exp_i8, K_dim, row_lo, row_hi, block=32):
    sc = scale_exp_i8[row_lo:row_hi].to(torch.float32)
    scales = torch.pow(2.0, sc)
    rows, kb = scales.shape
    se = scales.unsqueeze(-1).expand(rows, kb, block).reshape(rows, -1)[:, :K_dim]
    return data_f32 * se

try:
    spec = importlib.util.spec_from_file_location({module_name!r}, {so_path!r})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    A = gen_fp4(M, K); B = gen_fp4(N, K)
    sc_a = torch.randint(-2, 3, (M, K//32), dtype=torch.int8, device='cuda')
    sc_b = torch.randint(-2, 3, (N, K//32), dtype=torch.int8, device='cuda')
    A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

    run = lambda: mod.gemm_rcr(A, B, A_sc, B_sc, C)
    C.zero_(); run(); torch.cuda.synchronize()
    C0 = C.clone()
    finite_frac = float(torch.isfinite(C0.float()).sum().item()) / float(C0.numel())
    finite_frac = round(finite_frac, 6)

    MAX_M_ROWS = 1024
    MAX_N_ROWS = 1024
    m_rows = min(M, MAX_M_ROWS)
    n_rows = min(N, MAX_N_ROWS)

    A_blk = dequant_fp4_rows(A, K, 0, m_rows)
    B_blk = dequant_fp4_rows(B, K, 0, n_rows)
    A_scaled = apply_scales_rows(A_blk, sc_a, K, 0, m_rows)
    B_scaled = apply_scales_rows(B_blk, sc_b, K, 0, n_rows)
    del A_blk, B_blk

    C_ref = torch.matmul(A_scaled, B_scaled.T)
    del A_scaled, B_scaled

    C_test = C0[:m_rows, :n_rows].float()
    base_mask = torch.isfinite(C_ref) & torch.isfinite(C_test)

    REL_TOL = 0.05
    ABS_TOL = 1e-2
    n_valid = int(base_mask.sum().item())

    if n_valid < 1024:
        snr_db = float('-inf')
        snr_med_db = float('-inf')
        wrong_cell_frac = 1.0
    else:
        ref_d = C_ref.double()
        test_d = C_test.double()
        abs_diff = (test_d - ref_d).abs()
        abs_ref = ref_d.abs()
        abs_test = test_d.abs()
        catastrophic = (abs_test > 100.0 * abs_ref + 1e-3) & (abs_diff > REL_TOL * abs_ref + ABS_TOL)
        catastrophic = catastrophic & base_mask
        wrong_cell_frac = float(catastrophic.sum().item()) / float(base_mask.sum().item())

        good_mask = base_mask & ~catastrophic
        n_good = int(good_mask.sum().item())
        if n_good < 1024:
            snr_db = float('-inf'); snr_med_db = float('-inf')
        else:
            rv = ref_d[good_mask]; tv = test_d[good_mask]
            sig = float((rv * rv).mean().item())
            err = float(((tv - rv) ** 2).mean().item())
            snr_db = 10.0 * math.log10(sig / err) if (sig > 0.0 and err > 0.0) else float('-inf')

            row_n = good_mask.sum(dim=1).clamp_min(1)
            row_sig = (ref_d ** 2 * good_mask).sum(dim=1) / row_n
            row_err = ((test_d - ref_d) ** 2 * good_mask).sum(dim=1) / row_n
            valid_rows = good_mask.any(dim=1) & (row_sig > 0) & (row_err > 0)
            if int(valid_rows.sum().item()) < 8:
                snr_med_db = float('-inf')
            else:
                row_snr = 10.0 * torch.log10(row_sig[valid_rows] / row_err[valid_rows])
                snr_med_db = float(row_snr.median().item())

    correct = ((snr_med_db >= SNR_DB_GATE)
               and (wrong_cell_frac < WRONG_CELL_GATE)
               and (finite_frac >= FINITE_GATE))

    if not correct:
        out = {{
            "M": M, "N": N, "K": K, "tflops": None, "avg_ms": None, "comp": COMP,
            "status": "WRONG_OUTPUT", "kernel_finite": finite_frac,
            "snr_db": snr_db if math.isfinite(snr_db) else None,
            "snr_med_db": snr_med_db if math.isfinite(snr_med_db) else None,
            "snr_n_valid": n_valid,
            "wrong_cell_frac": round(wrong_cell_frac, 6),
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
        "snr_db": round(snr_db, 2) if math.isfinite(snr_db) else None,
        "snr_med_db": round(snr_med_db, 2) if math.isfinite(snr_med_db) else None,
        "snr_n_valid": n_valid,
        "wrong_cell_frac": round(wrong_cell_frac, 6),
    }}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")

except torch.cuda.OutOfMemoryError:
    out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None,
           "comp": COMP, "status": "OOM", "kernel_finite": None}}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
except Exception as e:
    out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None,
           "comp": COMP, "status": f"ERR:{{e}}"[:120], "kernel_finite": None}}
    print("BENCH_JSON_START"); print(json.dumps(out)); print("BENCH_JSON_END")
"""


def bench_one(shape, m, n, k, comp, so_path, gpu_id):
    if not os.path.exists(so_path):
        return {"shape": shape, "so": so_path, "gpu": gpu_id,
                "status": "MISSING_SO", "tflops": None, "kernel_finite": None}
    module_name = os.path.basename(so_path).split(".")[0]
    script = make_runner_script(module_name, so_path, m, n, k, comp)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=SHAPE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"shape": shape, "so": so_path, "gpu": gpu_id,
                "status": "TIMEOUT", "tflops": None, "kernel_finite": None}
    if r.returncode != 0:
        return {"shape": shape, "so": so_path, "gpu": gpu_id,
                "status": "CRASH", "tflops": None, "kernel_finite": None,
                "stderr_tail": r.stderr[-200:]}
    try:
        s = r.stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = r.stdout.index("BENCH_JSON_END")
        out = json.loads(r.stdout[s:e].strip())
        out["shape"] = shape
        out["so"] = so_path
        out["gpu"] = gpu_id
        return out
    except (ValueError, json.JSONDecodeError):
        return {"shape": shape, "so": so_path, "gpu": gpu_id,
                "status": "PARSE_FAIL", "tflops": None, "kernel_finite": None,
                "stdout_tail": r.stdout[-200:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="5,6,7")
    ap.add_argument("--inventory",
                    default=os.path.join(SCRIPT_DIR, "R43_OPT_C", "candidate_inventory.json"))
    ap.add_argument("--out",
                    default=os.path.join(SCRIPT_DIR, "R43_OPT_C_SWEEP_SMOKE.json"))
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    with open(args.inventory) as f:
        inventory = json.load(f)

    # Build job list
    jobs = []
    for shape, info in inventory.items():
        m, n, k = info["M"], info["N"], info["K"]
        comp = COMP_TFLOPS[shape]
        for so in info["candidates"]:
            jobs.append((shape, m, n, k, comp, so))

    print(f"R43 OPT C SMOKE SWEEP: {len(jobs)} jobs across {len(gpus)} GPUs", flush=True)
    print(f"  warmup={WARMUP}, iters={ITERS}, trim={TRIM}, GATE_FIN={FINITE_GATE}", flush=True)

    # Round-robin assign jobs to GPUs
    gpu_jobs = {g: [] for g in gpus}
    for i, j in enumerate(jobs):
        gpu_jobs[gpus[i % len(gpus)]].append((i, j))

    results = [None] * len(jobs)
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def worker(gpu_id):
        for (i, (shape, m, n, k, comp, so)) in gpu_jobs[gpu_id]:
            r = bench_one(shape, m, n, k, comp, so, gpu_id)
            with lock:
                results[i] = r
                done[0] += 1
                tflops = r.get("tflops")
                fin = r.get("kernel_finite")
                wcf = r.get("wrong_cell_frac")
                fin_s = f"{fin:.4f}" if isinstance(fin, float) else "--"
                wcf_s = f"{wcf:.4f}" if isinstance(wcf, float) else "--"
                tflops_s = f"{tflops:>7.1f}" if isinstance(tflops, float) else "------- "
                pct = (tflops / comp * 100) if isinstance(tflops, float) else None
                pct_s = f"{pct:5.1f}%" if pct is not None else "  --  "
                so_short = os.path.basename(so).replace("tk_mxfp4_gluon_cpp_", "").split(".")[0]
                print(f"  [{done[0]:>3}/{len(jobs)}] {shape:<22} G{gpu_id} "
                      f"{r['status']:<12} t={tflops_s} {pct_s} fin={fin_s} wcf={wcf_s} {so_short}",
                      flush=True)

    threads = []
    for g in gpus:
        t = threading.Thread(target=worker, args=(g,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    # Per-shape filter: keep candidates that PASSED the gate AND beat current p50 by >= 3%
    # Current p50 is from inventory baseline.
    candidates_to_5run = {}
    for shape, info in inventory.items():
        cur_p50 = info["current_tflops_p50"]
        threshold = cur_p50 * 1.03
        rows = [r for r in results if r.get("shape") == shape]
        promoted = []
        for r in rows:
            if r["status"] != "OK":
                continue
            t = r.get("tflops")
            fin = r.get("kernel_finite")
            wcf = r.get("wrong_cell_frac")
            if t is None or fin is None or wcf is None:
                continue
            if fin < FINITE_GATE:
                continue
            if wcf >= WRONG_CELL_GATE:
                continue
            if t >= threshold:
                promoted.append({
                    "so": r["so"],
                    "tflops": t,
                    "fin": fin,
                    "wcf": wcf,
                    "delta_vs_current_pct": (t / cur_p50 - 1) * 100,
                })
        promoted.sort(key=lambda x: x["tflops"], reverse=True)
        candidates_to_5run[shape] = {
            "current_so": info["current_so"],
            "current_tflops": cur_p50,
            "comp_tflops": info["comp_tflops"],
            "n_candidates": len(rows),
            "n_passed_gate_and_beat_3pct": len(promoted),
            "promoted": promoted,
        }

    summary = {
        "round": "R43_OPT_C_PHASE1_SMOKE",
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup": WARMUP, "iters": ITERS, "trim": TRIM,
        "finite_gate": FINITE_GATE, "wcf_gate": WRONG_CELL_GATE,
        "gpus": gpus, "n_jobs": len(jobs),
        "elapsed_minutes": round(elapsed / 60, 2),
        "raw_results": results,
        "candidates_to_5run": candidates_to_5run,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== SMOKE DONE in {elapsed/60:.1f} min ===")
    for shape, sd in candidates_to_5run.items():
        print(f"  {shape}: {sd['n_passed_gate_and_beat_3pct']}/{sd['n_candidates']} promoted "
              f"(top: {sd['promoted'][0]['tflops']:.1f}@+{sd['promoted'][0]['delta_vs_current_pct']:.1f}%) "
              if sd['promoted'] else f"  {shape}: 0/{sd['n_candidates']} promoted (none beat +3%)")
    print(f"\nOutput: {args.out}")


if __name__ == "__main__":
    main()
