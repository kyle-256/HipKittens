#!/usr/bin/env python3
"""R32-OptA bench: SNR check + bench on L6 for K_LOOP_SYNC_EVERY_2 / NONTEMPORAL variants.

warmup=200, iters=500, trim=0.10 per benchmark-rules.md.

Usage:
  python3 bench_R32_optA.py --gpu N --snr-only           # SNR gate only
  python3 bench_R32_optA.py --gpu N --variants V1_kls2 --reps 5
"""
import os, sys, json, subprocess, textwrap, time, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

M, N, K = 4096, 32768, 128256
INCUMBENT_TFLOPS = 5354.0
COMP_TFLOPS = 5781.1

VARIANTS = [
    ("incumbent",   "tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all"),
    ("V1_kls2",     "tk_mxfp4_gluon_cpp_n32768_k128256_R32A_ts_lgk2_v12_memc_btw_all_kls2"),
    ("V2_nt",       "tk_mxfp4_gluon_cpp_n32768_k128256_R32A_ts_lgk2_v12_memc_btw_all_nt"),
    ("V3_kls2_nt",  "tk_mxfp4_gluon_cpp_n32768_k128256_R32A_ts_lgk2_v12_memc_btw_all_kls2_nt"),
]


def make_script(module_name, m, n, k, comp, snr_check=False, snr_only=False):
    snr_emit = '"snr_db": snr_db, ' if snr_check else ''
    if snr_check:
        # Full-K reference SNR check: compute ref over the FULL K=128256
        # for a small snr_M x snr_N corner, then compare against the kernel's
        # output corner. Reuses the SAME A/B/sc tensors used for timing so
        # we get an apples-to-apples comparison.
        snr_compute = (
            "try:\n"
            "    snr_M = 64; snr_N = 64\n"
            "    C_chk = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')\n"
            "    MOD.gemm_rcr(A, B, A_sc, B_sc, C_chk)\n"
            "    LUT = torch.tensor([0,0.5,1,1.5,2,3,4,6,-0,-0.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32, device='cuda')\n"
            "    def dec(p):\n"
            "        lo = p & 0xF; hi = (p >> 4) & 0xF\n"
            "        out = torch.empty(p.shape[0], p.shape[1]*2, dtype=torch.float32, device='cuda')\n"
            "        out[:, 0::2] = LUT[lo.long()]; out[:, 1::2] = LUT[hi.long()]\n"
            "        return out\n"
            # Full-K reference using the original UN-preshuffled scales
            "    Af = dec(A[:snr_M, :])\n"
            "    Bf = dec(B[:snr_N, :])\n"
            "    sa = sc_exp_a[:snr_M, :].float().repeat_interleave(32, dim=1)\n"
            "    sb = sc_exp_b[:snr_N, :].float().repeat_interleave(32, dim=1)\n"
            "    Af = Af * (2.0 ** sa)\n"
            "    Bf = Bf * (2.0 ** sb)\n"
            "    ref = (Af @ Bf.T).to(torch.float32)\n"
            "    got = C_chk[:snr_M, :snr_N].float()\n"
            "    err = got - ref\n"
            "    sig_pow = (ref ** 2).mean().item()\n"
            "    err_pow = (err ** 2).mean().item() + 1e-30\n"
            "    snr_db = 10.0 * math.log10(sig_pow / err_pow) if sig_pow > 0 else float('nan')\n"
            "    n_nan_got = torch.isnan(got).sum().item(); n_nan_ref = torch.isnan(ref).sum().item()\n"
            "    print(f'SNR sig_pow={sig_pow:.6e} err_pow={err_pow:.6e} snr_db={snr_db:.2f} n_nan_got={n_nan_got} n_nan_ref={n_nan_ref}')\n"
            "except Exception as _e:\n"
            "    snr_db = None\n"
            "    print(f'SNR EXCEPTION: {_e}')\n"
        )
    else:
        snr_compute = "snr_db = None\n"

    # Skip the timing loop in --snr-only mode (just SNR + return)
    if snr_only:
        timing_block = (
            "tflops = None; avg_ms = None\n"
        )
    else:
        timing_block = (
            "run = lambda: MOD.gemm_rcr(A, B, A_sc, B_sc, C)\n"
            "for _ in range(WARMUP):\n"
            "    run()\n"
            "torch.cuda.synchronize()\n"
            "times_ms = []\n"
            "for _ in range(ITERS):\n"
            "    s = torch.cuda.Event(enable_timing=True)\n"
            "    e = torch.cuda.Event(enable_timing=True)\n"
            "    s.record(); run(); e.record(); torch.cuda.synchronize()\n"
            "    times_ms.append(s.elapsed_time(e))\n"
            "times_ms.sort()\n"
            "tc = int(len(times_ms) * TRIM_FRAC)\n"
            "trimmed = times_ms[tc:-tc] if tc > 0 else times_ms\n"
            "avg_ms = sum(trimmed) / len(trimmed)\n"
            "tflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e12\n"
        )

    script = textwrap.dedent(f'''\
        #!/usr/bin/env python3
        import gc, json, math, sys, torch
        torch.manual_seed(0)
        sys.path.insert(0, {BUILD_DIR!r})
        import {module_name} as MOD

        M, N, K = {m}, {n}, {k}
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

        try:
            A = gen_fp4(M, K)
            B = gen_fp4(N, K)
            sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
            sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
            A_sc = preshuffle_mfma16_merged(sc_exp_a)
            B_sc = preshuffle_mfma16_merged(sc_exp_b)
            C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

            __TIMING_BLOCK__

            __SNR_COMPUTE__

            result = {{"M": M, "N": N, "K": K,
                       "avg_ms": (round(avg_ms, 4) if avg_ms is not None else None),
                       "tflops": (round(tflops, 1) if tflops is not None else None),
                       {snr_emit}"comp": {comp}, "status": "OK"}}
        except Exception as ex:
            result = {{"M": M, "N": N, "K": K, "status": f"ERR:{{ex}}"}}

        print("BENCH_JSON_START")
        print(json.dumps(result))
        print("BENCH_JSON_END")
    ''')
    snr_indented = "\n".join(("    " + ln) if ln else "" for ln in snr_compute.splitlines())
    script = script.replace("    __SNR_COMPUTE__", snr_indented)
    timing_indented = "\n".join(("    " + ln) if ln else "" for ln in timing_block.splitlines())
    script = script.replace("    __TIMING_BLOCK__", timing_indented)
    return script


def run_bench(tag, module_name, gpu_id, snr_check=False, snr_only=False):
    script = make_script(module_name, M, N, K, COMP_TFLOPS, snr_check=snr_check, snr_only=snr_only)
    suffix = ""
    if snr_only: suffix = "_snronly"
    elif snr_check: suffix = "_snr"
    script_path = os.path.join(SCRIPT_DIR, f"_R32A_run_{tag}{suffix}.py")
    with open(script_path, "w") as f:
        f.write(script)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    r = subprocess.run(["python3", script_path], capture_output=True, text=True, env=env, timeout=1500)
    dt = time.time() - t0
    out = r.stdout
    result = {"tag": tag, "module": module_name, "gpu_id": gpu_id, "wall_sec": round(dt, 1)}
    # Capture full stdout (includes SNR diag print)
    if "SNR" in out:
        for line in out.splitlines():
            if line.startswith("SNR "):
                result["snr_diag"] = line
    if "BENCH_JSON_START" in out and "BENCH_JSON_END" in out:
        s = out.split("BENCH_JSON_START")[1].split("BENCH_JSON_END")[0].strip()
        try:
            result.update(json.loads(s))
        except Exception as e:
            result["status"] = f"PARSE_ERR:{e}"
            result["stderr_tail"] = r.stderr[-1000:]
            result["stdout_tail"] = out[-1000:]
    else:
        result["status"] = f"NO_JSON (rc={r.returncode})"
        result["stderr_tail"] = r.stderr[-2000:]
        result["stdout_tail"] = out[-1000:]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Subset of variant tags to run")
    ap.add_argument("--reps", type=int, default=1, help="Repetitions per variant")
    ap.add_argument("--snr", action="store_true", help="Run with SNR check on first rep")
    ap.add_argument("--snr-only", action="store_true", help="Only run SNR (skip timing)")
    ap.add_argument("--out", type=str, default=None, help="Override output JSON path")
    args = ap.parse_args()
    todo = VARIANTS
    if args.variants:
        todo = [(t, m) for t, m in VARIANTS if t in args.variants]
    print(f"R32-A bench on GPU {args.gpu}, {len(todo)} variants, reps={args.reps}, snr={args.snr}, snr_only={args.snr_only}")
    results = []
    for tag, module in todo:
        per_variant = []
        for rep in range(args.reps):
            do_snr = (args.snr and rep == 0) or args.snr_only
            print(f"\n[{tag}] rep {rep+1}/{args.reps} on GPU {args.gpu} (snr={do_snr}, snr_only={args.snr_only})...", flush=True)
            res = run_bench(tag, module, args.gpu, snr_check=do_snr, snr_only=args.snr_only)
            tf = res.get("tflops")
            ratio = (tf / INCUMBENT_TFLOPS * 100) if tf else 0
            tf_str = f"{tf:>8}" if tf is not None else "    NONE"
            print(f"  {tag:14s} rep{rep+1} TFLOPS={tf_str} ratio_vs_inc={ratio:.2f}% snr_db={res.get('snr_db','-')} wall={res['wall_sec']}s status={res.get('status','?')}", flush=True)
            if res.get("snr_diag"):
                print(f"    {res['snr_diag']}", flush=True)
            per_variant.append(res)
        tflops_list = [r.get("tflops") for r in per_variant if r.get("tflops") is not None]
        if tflops_list:
            mean_tf = sum(tflops_list) / len(tflops_list)
            best_tf = max(tflops_list)
            worst_tf = min(tflops_list)
            print(f"  {tag:14s} SUMMARY mean={mean_tf:.1f} best={best_tf:.1f} worst={worst_tf:.1f}")
        results.append({"tag": tag, "module": module, "reps": per_variant})

    out_json = args.out or os.path.join(SCRIPT_DIR, "R32_OPT_A_BENCH_RESULTS.json")
    with open(out_json, "w") as f:
        json.dump({
            "M": M, "N": N, "K": K,
            "incumbent_tflops": INCUMBENT_TFLOPS,
            "comp_tflops": COMP_TFLOPS,
            "warmup": 200, "iters": 500, "trim_frac": 0.10,
            "gpu_id": args.gpu,
            "reps_per_variant": args.reps,
            "results": results,
        }, f, indent=2)
    print(f"\nResults: {out_json}")


if __name__ == "__main__":
    main()
