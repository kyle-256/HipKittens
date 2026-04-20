#!/usr/bin/env python3
"""R58 Opt O / I-2 / O-1 — L8 HK kernel probe on (4096, 32768, 128256).

Tests whether HK 256x256 lgk2 v12 (R40B safe build, never benched against L8 since R52D2B
AITER swap) can beat the current AITER 256x256 baseline (R52D2B reviewer p50 = 97.75%).

Bench MANDATORY: warmup=200, iters=500, trim=0.10, fresh seed per run.

Modes:
  --mode smoke    1 (or N) seeds, full correctness gate + perf
  --mode 10run    10 INDEPENDENT seeds [101,202,303,404,505,606,707,808,909,1010]

Per-cell HK harness mirrors bench_R44D_10run.py invocation pattern (gemm_rcr).

Default candidate (HK_VARIANT=R40B):
  /shared_nfs/.../build_R40B/tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_btw_all_R40B_safe...so
Fallback candidate (HK_VARIANT=R37):
  /shared_nfs/.../build_R37/tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all_R37...so

GATES:
- VC strict 10-run gate: n_OK>=8/10 AND wcf_max<0.02 AND wcf_std<0.01 AND fin_min>=0.97
- SMOKE escalate gate: pct_comp >= 98.75% (current AITER 97.75% + 1.0pp)
- SMOKE STOP rule: pct_comp < 96.75% -> ACCEPT_FALLBACK without 10-run
- WIN gate: pct_comp >= 100.0%
- PROMOTE gate: 10-run PASS AND pct_comp >= 100.0% AND >=+1.0pp over AITER 97.75%
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATES = {
    "R40B": os.path.join(
        SCRIPT_DIR, "build_R40B",
        "tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_btw_all_R40B_safe."
        "cpython-310-x86_64-linux-gnu.so"),
    "R37": os.path.join(
        SCRIPT_DIR, "build_R37",
        "tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all_R37."
        "cpython-310-x86_64-linux-gnu.so"),
}

WARMUP = 200
ITERS = 500
TRIM_FRAC = 0.10
SNR_DB_GATE = 10.0
WCF_GATE = 0.02
FINITE_GATE = 0.97
M, N, K = 4096, 32768, 128256
COMPETITOR_TFLOPS = 5781.1  # bench_all_42.py competitor for (4096, 32768, 128256)
AITER_R52D2B_PCT_COMP = 97.75  # current baseline (R57 reviewer p50)
SMOKE_ESCALATE_PCT = 98.75
SMOKE_STOP_PCT = 96.75


def make_runner_script(module_name: str, so_path: str, seed: int, do_perf: bool) -> str:
    return f"""\
import sys, math, json, importlib.util, torch
torch.manual_seed({seed})
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM_FRAC}
SNR_DB_GATE = {SNR_DB_GATE}
WCF_GATE = {WCF_GATE}
FIN_GATE = {FINITE_GATE}
M, N, K = {M}, {N}, {K}
COMP = {COMPETITOR_TFLOPS}
DO_PERF = {int(do_perf)}

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

def emit(d):
    print("BENCH_JSON_START"); print(json.dumps(d)); print("BENCH_JSON_END")

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
        snr_db = float('-inf'); snr_med_db = float('-inf'); wrong_cell_frac = 1.0
    else:
        ref_d = C_ref.double(); test_d = C_test.double()
        abs_diff = (test_d - ref_d).abs()
        abs_ref = ref_d.abs(); abs_test = test_d.abs()
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

    correct = (snr_med_db >= SNR_DB_GATE
               and wrong_cell_frac < WCF_GATE
               and finite_frac >= FIN_GATE)

    if not correct or not DO_PERF:
        out = {{
            "M": M, "N": N, "K": K,
            "tflops": None, "avg_ms": None,
            "comp": COMP,
            "status": "WRONG_OUTPUT" if not correct else "CORRECT_NO_PERF",
            "kernel_finite": finite_frac,
            "snr_db": snr_db if math.isfinite(snr_db) else None,
            "snr_med_db": snr_med_db if math.isfinite(snr_med_db) else None,
            "snr_n_valid": n_valid,
            "wrong_cell_frac": round(wrong_cell_frac, 6),
            "seed": {seed},
        }}
        emit(out); sys.exit(0)

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
        "M": M, "N": N, "K": K,
        "tflops": round(tflops, 1), "avg_ms": round(avg, 4),
        "comp": COMP,
        "status": "OK",
        "kernel_finite": finite_frac,
        "snr_db": round(snr_db, 2) if math.isfinite(snr_db) else None,
        "snr_med_db": round(snr_med_db, 2) if math.isfinite(snr_med_db) else None,
        "snr_n_valid": n_valid,
        "wrong_cell_frac": round(wrong_cell_frac, 6),
        "seed": {seed},
    }}
    emit(out)

except torch.cuda.OutOfMemoryError:
    out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None,
           "comp": COMP, "status": "OOM", "kernel_finite": None, "seed": {seed}}}
    emit(out)
except Exception as e:
    import traceback
    tb = traceback.format_exc()
    out = {{"M": M, "N": N, "K": K, "tflops": None, "avg_ms": None,
           "comp": COMP, "status": f"ERR:{{e}}"[:200],
           "kernel_finite": None, "seed": {seed}, "tb": tb[-800:]}}
    emit(out)
"""


def run_one_seed(so_path, seed, gpu_id, do_perf, timeout=2400):
    module_name = os.path.basename(so_path).split(".")[0]
    script = make_runner_script(module_name, so_path, seed, do_perf)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"M": M, "N": N, "K": K, "status": "TIMEOUT", "seed": seed,
                "comp": COMPETITOR_TFLOPS, "kernel_finite": None}
    if r.returncode != 0:
        return {"M": M, "N": N, "K": K, "status": "CRASH", "seed": seed,
                "comp": COMPETITOR_TFLOPS, "kernel_finite": None,
                "stderr_tail": r.stderr[-400:],
                "stdout_tail": r.stdout[-400:]}
    try:
        s = r.stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = r.stdout.index("BENCH_JSON_END")
        return json.loads(r.stdout[s:e].strip())
    except (ValueError, json.JSONDecodeError):
        return {"M": M, "N": N, "K": K, "status": "PARSE_FAIL", "seed": seed,
                "comp": COMPETITOR_TFLOPS, "kernel_finite": None,
                "stdout_tail": r.stdout[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "10run"], default="smoke")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--variant", choices=list(CANDIDATES.keys()), default="R40B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    so_path = CANDIDATES[args.variant]
    if not os.path.exists(so_path):
        print(f"FATAL: SO not found: {so_path}", file=sys.stderr)
        sys.exit(2)

    gpu_list = [int(g) for g in str(args.gpu).split(",") if g.strip()]
    if not gpu_list:
        gpu_list = [0]

    tag = args.tag or f"R58_OPT_O1_{args.variant}"
    if args.mode == "smoke":
        out_path = args.out or os.path.join(SCRIPT_DIR, f"{tag}_SMOKE.json")
        log_path = os.path.join(SCRIPT_DIR, f"{tag}_SMOKE.log")
        seeds = args.seeds or [101]
        do_perf = True
    else:
        out_path = args.out or os.path.join(SCRIPT_DIR, f"{tag}_10RUN.json")
        log_path = os.path.join(SCRIPT_DIR, f"{tag}_10RUN.log")
        seeds = args.seeds or [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
        do_perf = True  # 10run also measures perf so we get p50

    log_lines = []
    log_lines.append(f"R58 I-2/O-1 bench - mode={args.mode}, variant={args.variant}, "
                     f"gpu={gpu_list}, shape=({M},{N},{K})")
    log_lines.append(f"  so_path={so_path}")
    log_lines.append(f"  warmup={WARMUP} iters={ITERS} trim={TRIM_FRAC}")
    log_lines.append(f"  AITER R52D2B baseline pct_comp = {AITER_R52D2B_PCT_COMP}%")
    log_lines.append(f"  SMOKE escalate gate >= {SMOKE_ESCALATE_PCT}%; SMOKE STOP < {SMOKE_STOP_PCT}%")
    log_lines.append("=" * 90)
    for ln in log_lines:
        print(ln)

    results = []
    for i, seed in enumerate(seeds):
        gpu_id = gpu_list[i % len(gpu_list)]
        t0 = time.time()
        r = run_one_seed(so_path, seed, gpu_id, do_perf)
        dt = time.time() - t0
        r["wall_s"] = round(dt, 1)
        r["gpu_id"] = gpu_id
        results.append(r)
        line = (f"  seed={seed:>5} gpu={gpu_id} status={r.get('status'):<22} "
                f"fin={r.get('kernel_finite')!s:<8} "
                f"wcf={r.get('wrong_cell_frac')!s:<10} "
                f"snr_med={r.get('snr_med_db')!s:<8} "
                f"tflops={r.get('tflops')!s:<10} ({dt:.1f}s)")
        print(line, flush=True)
        log_lines.append(line)
        if r.get("status") in ("CRASH", "TIMEOUT", "PARSE_FAIL"):
            log_lines.append(f"    stderr: {r.get('stderr_tail','')[:300]}")
            log_lines.append(f"    stdout: {r.get('stdout_tail','')[:300]}")

    fins = [r.get("kernel_finite") for r in results if r.get("kernel_finite") is not None]
    wcfs = [r.get("wrong_cell_frac") for r in results if r.get("wrong_cell_frac") is not None]
    snrs = [r.get("snr_med_db") for r in results if r.get("snr_med_db") is not None]
    tflops_vals = [r.get("tflops") for r in results if r.get("tflops") is not None]

    n_OK = sum(1 for r in results
               if r.get("status") in ("OK", "CORRECT_NO_PERF")
               and (r.get("kernel_finite") is not None and r["kernel_finite"] >= FINITE_GATE)
               and (r.get("wrong_cell_frac") is not None and r["wrong_cell_frac"] < WCF_GATE)
               and (r.get("snr_med_db") is not None and r["snr_med_db"] >= SNR_DB_GATE))

    pct_comp_first = (round(tflops_vals[0] / COMPETITOR_TFLOPS * 100.0, 2)
                      if tflops_vals else None)
    pct_comp_p50 = None
    if tflops_vals:
        med_t = statistics.median(tflops_vals)
        pct_comp_p50 = round(med_t / COMPETITOR_TFLOPS * 100.0, 2)

    summary = {
        "mode": args.mode,
        "variant": args.variant,
        "so_path": so_path,
        "shape": [M, N, K],
        "competitor_tflops": COMPETITOR_TFLOPS,
        "aiter_baseline_pct_comp": AITER_R52D2B_PCT_COMP,
        "n_runs": len(results),
        "n_OK": n_OK,
        "fin_min": min(fins) if fins else None,
        "fin_max": max(fins) if fins else None,
        "wcf_min": min(wcfs) if wcfs else None,
        "wcf_max": max(wcfs) if wcfs else None,
        "wcf_std": (statistics.pstdev(wcfs) if len(wcfs) >= 2 else 0.0),
        "snr_min": min(snrs) if snrs else None,
        "snr_max": max(snrs) if snrs else None,
        "snr_med": (statistics.median(snrs) if snrs else None),
        "tflops_first": tflops_vals[0] if tflops_vals else None,
        "tflops_max": max(tflops_vals) if tflops_vals else None,
        "tflops_p50": (statistics.median(tflops_vals) if tflops_vals else None),
        "pct_comp_first": pct_comp_first,
        "pct_comp_p50": pct_comp_p50,
        "perf_delta_pp_vs_aiter": (round(pct_comp_p50 - AITER_R52D2B_PCT_COMP, 2)
                                   if pct_comp_p50 is not None else None),
        "passes_10run_strict_gate": (
            len(seeds) == 10 and n_OK >= 8
            and (max(wcfs) if wcfs else 1.0) < WCF_GATE
            and (statistics.pstdev(wcfs) if len(wcfs) >= 2 else 1.0) < 0.01
            and (min(fins) if fins else 0.0) >= FINITE_GATE
        ),
    }

    # SMOKE decision flags
    if args.mode == "smoke" and pct_comp_first is not None:
        if pct_comp_first < SMOKE_STOP_PCT:
            summary["smoke_decision"] = "STOP_ACCEPT_FALLBACK"
        elif pct_comp_first < SMOKE_ESCALATE_PCT:
            summary["smoke_decision"] = "STOP_ACCEPT_FALLBACK"
        else:
            summary["smoke_decision"] = "ESCALATE"
    elif args.mode == "smoke":
        summary["smoke_decision"] = "STOP_DEAD"

    log_lines.append("-" * 90)
    log_lines.append("SUMMARY: " + json.dumps(summary))
    print(log_lines[-1])

    full = {"summary": summary, "results": results}
    with open(out_path, "w") as f:
        json.dump(full, f, indent=2)
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nWrote {out_path}")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
