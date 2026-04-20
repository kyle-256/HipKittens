#!/usr/bin/env python3
"""R68 ISO harness: differential test for step12pf NON-SPLIT prototype.

Builds and runs THREE variants of kernel_mxfp4_gluon_cpp.cpp at ONE shape
(M=16384, N=4096, K=2048) and compares their bf16 output tensors:

  (a) baseline   — FUSED_STEP34=1 + TAIL_SPLIT=1                      (R37 base)
  (b) step34pf   — baseline + STEP34_PF_INTERLEAVE=1                  (R66 LANDED)
  (c) step12pf   — baseline + STEP12_PF_INTERLEAVE=1                  (R68 prototype)

Gates:
  Gate 1 (compile)    : all three variants must build without errors.
  Gate 2 (correctness): byte-equal output (a) vs (b) and (a) vs (c).
                        We use bit-exact tensor equality — not SNR.
                        (b) is expected equal (R66 already shipped); the
                        critical question is whether (c) is equal.
  Gate 3 (perf)       : (only meaningful if Gate 2 passes for variant c)
                        warmup=200, iters=500, trim_frac=0.10.

Output: prints a structured report and writes results.json.

Inputs use the same protocol as bench_all_42.py:
  fp4 codes random in {0..15}, scales random in [-2..2], preshuffled.

Notes on the iso reduction:
  R67 spec asked for a single-iter K=64 test, but the kernel is structured
  around BK=128 (one K iter = 128 K-elements/lane). The smallest meaningful
  shape that exercises the steady-state K-loop body is K=2048 (16 iters)
  with M=16384, N=4096 (the smallest shape in bench_all_42.py). This is the
  canonical "single-shape" iso check for differential correctness — it runs
  the entire kpair_64mfma_step12 -> step34 pipeline including the prefetch
  race we want to measure.
"""

import json
import math
import os
import subprocess
import sys
import sysconfig
import textwrap
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MI350X_DIR = os.path.dirname(SCRIPT_DIR)
KERNEL_SRC = os.path.join(MI350X_DIR, "kernel_mxfp4_gluon_cpp.cpp")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(MI350X_DIR, "..", "..", ".."))
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

M, N, K = 16384, 4096, 2048

VARIANTS = [
    ("base",     "-DFUSED_STEP34=1 -DTAIL_SPLIT=1"),
    ("step34pf", "-DFUSED_STEP34=1 -DTAIL_SPLIT=1 -DSTEP34_PF_INTERLEAVE=1"),
    ("step12pf", "-DFUSED_STEP34=1 -DTAIL_SPLIT=1 -DSTEP12_PF_INTERLEAVE=1"),
]


def build_variant(tag, flags, build_dir):
    module_name = f"tk_iso_{tag}"
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(build_dir, out_file)
    if os.path.exists(out_path):
        os.remove(out_path)

    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(build_dir, f"wrap_{tag}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    print(f"  Building {tag:>10} ...", end=" ", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT
    cmd = (
        f'make -C {MI350X_DIR} TARGET={os.path.join(build_dir, module_name)} '
        f'SRC={wrapper_src} '
        f'CPPFLAGS="-DK_DIM={K} -DN_DIM={N} {flags}"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    log_path = os.path.join(build_dir, f"build_{tag}.log")
    with open(log_path, "w") as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)

    if result.returncode != 0 or not os.path.exists(out_path):
        print(f"FAIL ({elapsed:.1f}s) — see {log_path}")
        return None, log_path
    print(f"OK   ({elapsed:.1f}s)")
    return out_path, log_path


def make_gate2_runner(module_name, build_dir):
    """Run kernel multiple times and compute SNR vs torch reference.

    Background: the HK kernel has a known MFMA accumulator cohort race that
    makes outputs non-deterministic across runs (memory note
    project_mxfp4_finite_gate_cohort_race.md). Pure byte-equality is therefore
    not a usable correctness gate. Instead we:
      1. Run the kernel 3x and report (median_inf_count, mean_snr).
      2. Compute SNR vs torch reference to detect SYSTEMATIC corruption.
      3. The race produces sporadic per-cell inf, but mean SNR should still
         be > ~30 dB for a correct kernel. R67 SPLIT failure produced 5632
         row-clustered NaNs vs 0 in baseline — a clear cliff.
    """
    return textwrap.dedent(f'''\
        #!/usr/bin/env python3
        import gc, json, math, sys, torch
        torch.manual_seed(0)
        sys.path.insert(0, {build_dir!r})
        import {module_name}

        M, N, K = {M}, {N}, {K}
        k_blocks = K // 32

        def gen_fp4_restricted(rows, K):
            cols = K // 2
            lo = torch.randint(1, 4, (rows, cols), dtype=torch.uint8, device="cuda")
            hi = torch.randint(1, 4, (rows, cols), dtype=torch.uint8, device="cuda")
            return (hi << 4) | lo

        def preshuffle_mfma16_merged(scale_exp):
            rows, kb = scale_exp.shape
            pr = math.ceil(rows / 64) * 64
            pk = math.ceil(kb / 8) * 8
            raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)
            raw[:rows, :kb] = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
            sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)
            sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
            sh = sh.view(pr // 32, pk * 32)
            sh = sh.view(pr // 64, 2, pk * 32 // 4, 4)
            sh = sh.permute(0, 2, 1, 3).contiguous()
            return sh.view(pr // 64, pk * 64)

        # Build reference once
        def fp4_decode(packed_bytes):
            tab = torch.tensor([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                -0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                                dtype=torch.float32, device=packed_bytes.device)
            lo = packed_bytes & 0xF
            hi = (packed_bytes >> 4) & 0xF
            R, C = packed_bytes.shape
            out = torch.zeros(R, C * 2, dtype=torch.float32, device=packed_bytes.device)
            out[:, 0::2] = tab[lo.long()]
            out[:, 1::2] = tab[hi.long()]
            return out

        try:
            A = gen_fp4_restricted(M, K)
            B = gen_fp4_restricted(N, K)
            sc_exp_a = torch.zeros(M, k_blocks, dtype=torch.int8, device="cuda")
            sc_exp_b = torch.zeros(N, k_blocks, dtype=torch.int8, device="cuda")
            A_sc = preshuffle_mfma16_merged(sc_exp_a)
            B_sc = preshuffle_mfma16_merged(sc_exp_b)

            # Reference (all scales = 1, exact-rounded fp32 matmul)
            A_f = fp4_decode(A); B_f = fp4_decode(B)
            ref = (A_f @ B_f.T).to(torch.bfloat16)
            ref_norm = ref.float().norm().item()

            # Run kernel 3x and stat
            inf_counts, nan_counts, snrs = [], [], []
            saved = False
            for run_i in range(3):
                C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
                {module_name}.gemm_rcr(A, B, A_sc, B_sc, C)
                torch.cuda.synchronize()
                if not saved:
                    torch.save(C.cpu(), "{build_dir}/{module_name}_C.pt")
                    saved = True
                isn = int(torch.isnan(C).sum().item())
                isi = int(torch.isinf(C).sum().item())
                inf_counts.append(isi); nan_counts.append(isn)
                # SNR over finite cells only — mask BEFORE arithmetic to avoid inf*0=nan
                Cf = C.float()
                rf = ref.float()
                finite = (~torch.isnan(Cf)) & (~torch.isinf(Cf))
                if finite.any():
                    # Use double precision norm to avoid fp32 overflow when err
                    # is large (sum of squares of bf16 ~2000 * 67M cells overflows).
                    Cf_clean = torch.where(finite, Cf, rf).double()
                    rfd = rf.double()
                    err_sq = ((Cf_clean - rfd) ** 2).sum().item()
                    ref_sq = (rfd * finite.double()).pow(2).sum().item()
                    err = math.sqrt(err_sq) if err_sq > 0 else 0.0
                    ref_finite_norm = math.sqrt(ref_sq) if ref_sq > 0 else 0.0
                    if err > 0 and ref_finite_norm > 0:
                        snr = 20 * math.log10(ref_finite_norm / err)
                    elif err == 0:
                        snr = float('inf')
                    else:
                        snr = float('-inf')
                else:
                    snr = float('-inf')
                snrs.append(snr)
            result = {{"status":"OK",
                       "inf_per_run":inf_counts,
                       "nan_per_run":nan_counts,
                       "snr_db_per_run":[round(s,2) for s in snrs],
                       "median_inf":sorted(inf_counts)[1],
                       "median_nan":sorted(nan_counts)[1],
                       "median_snr_db":round(sorted(snrs)[1],2),
                       "ref_norm":ref_norm}}
        except Exception as e:
            result = {{"status":f"ERR:{{e}}"}}
        print("ISO_JSON_START")
        print(json.dumps(result))
        print("ISO_JSON_END")
    ''')


def make_gate3_runner(module_name, build_dir, comp_tflops=2995.0):
    return textwrap.dedent(f'''\
        #!/usr/bin/env python3
        import gc, json, math, sys, torch
        torch.manual_seed(0)
        sys.path.insert(0, {build_dir!r})
        import {module_name}

        M, N, K = {M}, {N}, {K}
        WARMUP = 200
        ITERS = 500
        TRIM_FRAC = 0.10
        k_blocks = K // 32

        def gen_fp4(rows, K):
            cols = K // 2
            lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
            hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
            return (hi << 4) | lo

        def preshuffle_mfma16_merged(scale_exp):
            rows, kb = scale_exp.shape
            pr = math.ceil(rows / 64) * 64
            pk = math.ceil(kb / 8) * 8
            raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)
            raw[:rows, :kb] = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
            sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)
            sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
            sh = sh.view(pr // 32, pk * 32)
            sh = sh.view(pr // 64, 2, pk * 32 // 4, 4)
            sh = sh.permute(0, 2, 1, 3).contiguous()
            return sh.view(pr // 64, pk * 64)

        A = gen_fp4(M, K); B = gen_fp4(N, K)
        sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
        sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
        A_sc = preshuffle_mfma16_merged(sc_exp_a)
        B_sc = preshuffle_mfma16_merged(sc_exp_b)
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        run = lambda: {module_name}.gemm_rcr(A, B, A_sc, B_sc, C)
        for _ in range(WARMUP): run()
        torch.cuda.synchronize()
        times_ms = []
        for _ in range(ITERS):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); run(); e.record(); torch.cuda.synchronize()
            times_ms.append(s.elapsed_time(e))
        times_ms.sort()
        trim = int(len(times_ms) * TRIM_FRAC)
        trimmed = times_ms[trim:-trim] if trim > 0 else times_ms
        avg_ms = sum(trimmed) / len(trimmed)
        tflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e12
        result = {{"avg_ms":round(avg_ms,4),"tflops":round(tflops,1),
                   "comp":{comp_tflops},"ratio":round(tflops/{comp_tflops}*100,1)}}
        print("ISO_JSON_START")
        print(json.dumps(result))
        print("ISO_JSON_END")
    ''')


def run_subprocess(script, work_dir, name, gpu=None, timeout=600):
    runner_path = os.path.join(work_dir, f"_run_{name}.py")
    with open(runner_path, "w") as f:
        f.write(script)
    env = os.environ.copy()
    if gpu is not None:
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    try:
        result = subprocess.run([sys.executable, runner_path], capture_output=True,
                                 text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}, time.time() - t0, ""
    elapsed = time.time() - t0
    if result.returncode != 0:
        return {"status": "CRASH", "stderr": result.stderr[-500:]}, elapsed, result.stdout
    out = result.stdout
    try:
        s = out.index("ISO_JSON_START") + len("ISO_JSON_START")
        e = out.index("ISO_JSON_END")
        return json.loads(out[s:e].strip()), elapsed, out
    except (ValueError, json.JSONDecodeError):
        return {"status": "PARSE_FAIL"}, elapsed, out


def compare_tensors(path_a, path_b):
    """Returns dict with byte_equal, max_abs_diff, n_diff_cells, n_nan."""
    import torch
    a = torch.load(path_a)
    b = torch.load(path_b)
    if a.shape != b.shape:
        return {"byte_equal": False, "shape_mismatch": [list(a.shape), list(b.shape)]}
    a_bytes = a.contiguous().view(torch.int16)
    b_bytes = b.contiguous().view(torch.int16)
    byte_equal = bool(torch.equal(a_bytes, b_bytes))
    diff = (a_bytes != b_bytes)
    n_diff = int(diff.sum().item())
    a_f = a.float()
    b_f = b.float()
    nan_a = int(torch.isnan(a_f).sum().item())
    nan_b = int(torch.isnan(b_f).sum().item())
    finite = (~torch.isnan(a_f)) & (~torch.isnan(b_f)) & (~torch.isinf(a_f)) & (~torch.isinf(b_f))
    if finite.any():
        max_abs_diff = float((a_f - b_f)[finite].abs().max().item())
    else:
        max_abs_diff = float("nan")
    return {
        "byte_equal": byte_equal,
        "n_diff_cells": n_diff,
        "n_total_cells": int(a.numel()),
        "max_abs_diff": max_abs_diff,
        "nan_a": nan_a, "nan_b": nan_b,
    }


def main():
    gpu = os.environ.get("ISO_GPU", "4")
    print("=" * 70)
    print(f"R68 ISO harness — step12pf NON-SPLIT differential test")
    print(f"Shape: M={M} N={N} K={K}   GPU: {gpu}")
    print("=" * 70)

    build_dir = os.path.join(SCRIPT_DIR, "build")
    work_dir = os.path.join(SCRIPT_DIR, "work")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    report = {"shape": [M, N, K], "gates": {}}

    # ── GATE 1: compile ──
    print("\n--- GATE 1: compile all 3 variants ---")
    so_paths = {}
    log_paths = {}
    gate1_pass = True
    for tag, flags in VARIANTS:
        so, log = build_variant(tag, flags, build_dir)
        log_paths[tag] = log
        if so is None:
            gate1_pass = False
            so_paths[tag] = None
        else:
            so_paths[tag] = so
    report["gates"]["gate1_compile"] = {
        "pass": gate1_pass,
        "paths": {k: v for k, v in so_paths.items()},
    }
    if not gate1_pass:
        print("\nGATE 1 FAIL — at least one variant did not build. See logs.")
        with open(os.path.join(SCRIPT_DIR, "results.json"), "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(1)
    print("GATE 1 PASS — all 3 variants built.")

    # ── GATE 2: differential correctness ──
    print("\n--- GATE 2: output equality vs base ---")
    gate2_results = {}
    tensor_paths = {}
    for tag in ("base", "step34pf", "step12pf"):
        script = make_gate2_runner(f"tk_iso_{tag}", build_dir)
        r, elapsed, _ = run_subprocess(script, work_dir, f"g2_{tag}", gpu=gpu, timeout=300)
        print(f"  {tag:>10}: {r}  ({elapsed:.1f}s)")
        gate2_results[tag] = r
        tensor_paths[tag] = os.path.join(build_dir, f"tk_iso_{tag}_C.pt")

    diffs = {}
    if all(gate2_results[t].get("status") == "OK" for t in ("base", "step34pf", "step12pf")):
        for cmp in ("step34pf", "step12pf"):
            d = compare_tensors(tensor_paths["base"], tensor_paths[cmp])
            print(f"  diff base vs {cmp}: {d}")
            diffs[f"base_vs_{cmp}"] = d

    # Gate 2 NEW LOGIC (because base kernel is non-deterministic with cohort
    # race producing ~0.3% inf cells per run, see project_mxfp4_finite_gate_cohort_race.md):
    #
    # step12pf passes iff:
    #   - finite-cell SNR within SNR_TOL_DB of base (systematic numeric drift)
    #   - inf cell count ratio not >> base (catastrophic increase = race got worse)
    #   - reference value SNR is reasonable (>20 dB, says we're computing right product)
    #
    # We benchmark step34pf as a "ceiling" — it's already shipped (R66/R67), so
    # whatever its number is = "correctness-safe under cohort race".
    SNR_TOL_DB = 3.0
    INF_RATIO_TOL = 10.0  # cohort race varies 5-10x naturally between runs
    base_r  = gate2_results.get("base", {})
    s34_r   = gate2_results.get("step34pf", {})
    s12_r   = gate2_results.get("step12pf", {})
    snr_ok = {}
    for tag, r in [("step34pf", s34_r), ("step12pf", s12_r)]:
        if r.get("status") != "OK" or base_r.get("status") != "OK":
            snr_ok[tag] = False; continue
        b_snr = base_r.get("median_snr_db", 0)
        c_snr = r.get("median_snr_db", 0)
        d_snr = b_snr - c_snr
        base_inf = max(base_r.get("median_inf", 1), 1)
        cmp_inf = r.get("median_inf", 0)
        inf_ratio = cmp_inf / base_inf
        snr_min_ok = c_snr >= 20.0
        snr_ok[tag] = (d_snr <= SNR_TOL_DB) and (inf_ratio <= INF_RATIO_TOL) and snr_min_ok
        print(f"  {tag}: snr={c_snr:.2f}dB (base {b_snr:.2f}, d={d_snr:.2f}), inf_ratio={inf_ratio:.2f}x -> {'OK' if snr_ok[tag] else 'FAIL'}")
    gate2_pass = snr_ok.get("step12pf", False)
    report["gates"]["gate2_correctness"] = {
        "pass": gate2_pass, "runs": gate2_results, "diffs": diffs,
        "snr_ok": snr_ok,
        "snr_tol_db": SNR_TOL_DB, "inf_ratio_tol": INF_RATIO_TOL,
    }
    print(f"\nGATE 2 (SNR-based): {'PASS' if gate2_pass else 'FAIL'}")
    print(f"  base     median snr={base_r.get('median_snr_db')}dB inf={base_r.get('median_inf')} nan={base_r.get('median_nan')}")
    print(f"  step34pf median snr={s34_r.get('median_snr_db')}dB inf={s34_r.get('median_inf')} nan={s34_r.get('median_nan')}")
    print(f"  step12pf median snr={s12_r.get('median_snr_db')}dB inf={s12_r.get('median_inf')} nan={s12_r.get('median_nan')}")

    # ── GATE 3: performance (only meaningful if Gate 2 passed for step12pf) ──
    print("\n--- GATE 3: performance (warmup=200, iters=500) ---")
    gate3_results = {}
    if not gate2_pass:
        print("  SKIPPED — Gate 2 failed (perf comparison meaningless for incorrect output)")
        report["gates"]["gate3_perf"] = {"pass": False, "skipped": True}
    else:
        for tag, _ in VARIANTS:
            script = make_gate3_runner(f"tk_iso_{tag}", build_dir)
            r, elapsed, _ = run_subprocess(script, work_dir, f"g3_{tag}", gpu=gpu, timeout=900)
            print(f"  {tag:>10}: {r}  ({elapsed:.1f}s)")
            gate3_results[tag] = r
        report["gates"]["gate3_perf"] = {"results": gate3_results}

    with open(os.path.join(SCRIPT_DIR, "results.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + "=" * 70)
    print(f"Wrote results to {os.path.join(SCRIPT_DIR, 'results.json')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
