#!/usr/bin/env python3
"""R31-OptB bench: single-rep on L6 for each XCD variant (+ incumbent for sanity).

Runs each module in its own subprocess (clean import / GPU isolation).
warmup=200, iters=500, trim=0.10 per benchmark-rules.md.
"""
import os, sys, json, subprocess, textwrap, time, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

M, N, K = 4096, 32768, 128256
INCUMBENT_TFLOPS = 5353.9
COMP_TFLOPS = 5781.1

VARIANTS = [
    ("incumbent",    "tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all"),
    ("V1_pxcd",      "tk_mxfp4_gluon_cpp_n32768_k128256_R31B_ts_lgk2_v12_memc_btw_all_V1_pxcd"),
    ("V2_sremap",    "tk_mxfp4_gluon_cpp_n32768_k128256_R31B_ts_lgk2_v12_memc_btw_all_V2_sremap"),
    ("V3_pxcd_b4",   "tk_mxfp4_gluon_cpp_n32768_k128256_R31B_ts_lgk2_v12_memc_btw_all_V3_pxcd_b4"),
    ("V4_sremap_gm8","tk_mxfp4_gluon_cpp_n32768_k128256_R31B_ts_lgk2_v12_memc_btw_all_V4_sremap_gm8"),
]


def make_script(module_name, m, n, k, comp, with_snr=False, reps=1):
    snr_block = ""
    if with_snr:
        snr_block = (
            "    try:\n"
            "        snr_M, snr_K, snr_N = 64, 256, 64\n"
            "        k_blocks_s = snr_K // 32\n"
            "        C_chk = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')\n"
            "        MOD.gemm_rcr(A, B, A_sc, B_sc, C_chk)\n"
            "        LUT = torch.tensor([0,0.5,1,1.5,2,3,4,6,-0,-0.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32, device='cuda')\n"
            "        def decode_fp4(packed):\n"
            "            lo = packed & 0xF\n"
            "            hi = (packed >> 4) & 0xF\n"
            "            out = torch.empty(packed.shape[0], packed.shape[1]*2, dtype=torch.float32, device='cuda')\n"
            "            out[:, 0::2] = LUT[lo.long()]\n"
            "            out[:, 1::2] = LUT[hi.long()]\n"
            "            return out\n"
            "        Af = decode_fp4(A[:snr_M, :snr_K//2])\n"
            "        Bf = decode_fp4(B[:snr_N, :snr_K//2])\n"
            "        sa = sc_exp_a[:snr_M, :k_blocks_s].float().repeat_interleave(32, dim=1)\n"
            "        sb = sc_exp_b[:snr_N, :k_blocks_s].float().repeat_interleave(32, dim=1)\n"
            "        Af = Af * (2.0 ** sa)\n"
            "        Bf = Bf * (2.0 ** sb)\n"
            "        ref = (Af @ Bf.T).to(torch.float32)\n"
            "        got = C_chk[:snr_M, :snr_N].float()\n"
            "        err = got - ref\n"
            "        sig_pow = (ref ** 2).mean().item()\n"
            "        err_pow = (err ** 2).mean().item() + 1e-30\n"
            "        snr_db = 10.0 * math.log10(sig_pow / err_pow) if sig_pow > 0 else float('nan')\n"
            "        result['snr_db'] = round(snr_db, 2)\n"
            "    except Exception as e:\n"
            "        result['snr_db'] = None\n"
            "        result['snr_err'] = str(e)[:200]\n"
        )
    body = (
        "#!/usr/bin/env python3\n"
        "import gc, json, math, sys, torch\n"
        "torch.manual_seed(0)\n"
        f"sys.path.insert(0, {BUILD_DIR!r})\n"
        f"import {module_name} as MOD\n"
        "\n"
        f"M, N, K = {m}, {n}, {k}\n"
        "WARMUP = 200\n"
        "ITERS = 500\n"
        "TRIM_FRAC = 0.10\n"
        f"REPS = {reps}\n"
        "k_blocks = K // 32\n"
        "\n"
        "def gen_fp4(rows, K):\n"
        "    cols = K // 2\n"
        "    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')\n"
        "    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')\n"
        "    return (hi << 4) | lo\n"
        "\n"
        "def preshuffle_mfma16_merged(scale_exp):\n"
        "    rows, kb = scale_exp.shape\n"
        "    pr = math.ceil(rows / 64) * 64\n"
        "    pk = math.ceil(kb / 8) * 8\n"
        "    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)\n"
        "    raw[:rows, :kb] = (scale_exp.to(torch.int16) + 127).to(torch.uint8)\n"
        "    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)\n"
        "    sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()\n"
        "    sh = sh.view(pr // 32, pk * 32)\n"
        "    sh = sh.view(pr // 64, 2, pk * 32 // 4, 4)\n"
        "    sh = sh.permute(0, 2, 1, 3).contiguous()\n"
        "    return sh.view(pr // 64, pk * 64)\n"
        "\n"
        "try:\n"
        "    A = gen_fp4(M, K)\n"
        "    B = gen_fp4(N, K)\n"
        "    sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device='cuda')\n"
        "    sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device='cuda')\n"
        "    A_sc = preshuffle_mfma16_merged(sc_exp_a)\n"
        "    B_sc = preshuffle_mfma16_merged(sc_exp_b)\n"
        "    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')\n"
        "\n"
        "    run = lambda: MOD.gemm_rcr(A, B, A_sc, B_sc, C)\n"
        "    for _ in range(WARMUP):\n"
        "        run()\n"
        "    torch.cuda.synchronize()\n"
        "\n"
        "    rep_tflops = []\n"
        "    rep_avg_ms = []\n"
        "    for r_i in range(REPS):\n"
        "        times_ms = []\n"
        "        for _ in range(ITERS):\n"
        "            s = torch.cuda.Event(enable_timing=True)\n"
        "            e = torch.cuda.Event(enable_timing=True)\n"
        "            s.record(); run(); e.record(); torch.cuda.synchronize()\n"
        "            times_ms.append(s.elapsed_time(e))\n"
        "        times_ms.sort()\n"
        "        tc = int(len(times_ms) * TRIM_FRAC)\n"
        "        trimmed = times_ms[tc:-tc] if tc > 0 else times_ms\n"
        "        avg_ms = sum(trimmed) / len(trimmed)\n"
        "        rep_avg_ms.append(round(avg_ms, 4))\n"
        "        rep_tflops.append(round(2.0 * M * N * K / (avg_ms * 1e-3) / 1e12, 1))\n"
        "\n"
        "    tflops = sum(rep_tflops) / len(rep_tflops)\n"
        f"    result = {{'M': M, 'N': N, 'K': K,\n"
        "               'rep_tflops': rep_tflops, 'rep_avg_ms': rep_avg_ms,\n"
        "               'tflops_mean': round(tflops, 1),\n"
        "               'tflops_min': min(rep_tflops), 'tflops_max': max(rep_tflops),\n"
        f"               'comp': {comp}, 'status': 'OK'}}\n"
        f"{snr_block}"
        "except Exception as ex:\n"
        f"    result = {{'M': M, 'N': N, 'K': K, 'status': 'ERR:' + str(ex)}}\n"
        "\n"
        "print('BENCH_JSON_START')\n"
        "print(json.dumps(result))\n"
        "print('BENCH_JSON_END')\n"
    )
    return body


def run_bench(tag, module_name, gpu_id, with_snr=False, reps=1):
    script = make_script(module_name, M, N, K, COMP_TFLOPS, with_snr=with_snr, reps=reps)
    snr_tag = "_snr" if with_snr else ""
    rep_tag = f"_r{reps}" if reps > 1 else ""
    script_path = os.path.join(SCRIPT_DIR, f"_R31B_run_{tag}{snr_tag}{rep_tag}.py")
    with open(script_path, "w") as f:
        f.write(script)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    r = subprocess.run(["python3", script_path], capture_output=True, text=True, env=env, timeout=1800)
    dt = time.time() - t0
    out = r.stdout
    result = {"tag": tag, "module": module_name, "gpu_id": gpu_id, "wall_sec": round(dt, 1)}
    if "BENCH_JSON_START" in out and "BENCH_JSON_END" in out:
        s = out.split("BENCH_JSON_START")[1].split("BENCH_JSON_END")[0].strip()
        try:
            result.update(json.loads(s))
        except Exception as e:
            result["status"] = f"PARSE_ERR:{e}"
            result["stderr_tail"] = r.stderr[-1000:]
    else:
        result["status"] = f"NO_JSON (rc={r.returncode})"
        result["stderr_tail"] = r.stderr[-2000:]
        result["stdout_tail"] = out[-1000:]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--snr", action="store_true", help="Enable SNR check (one variant only ideally)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    todo = VARIANTS
    if args.variants:
        todo = [(t, m) for t, m in VARIANTS if t in args.variants]
    print(f"R31-B bench on GPU {args.gpu}, {len(todo)} variants, reps={args.reps}, snr={args.snr}")
    results = []
    for tag, module in todo:
        print(f"\n[{tag}] running on GPU {args.gpu}...", flush=True)
        res = run_bench(tag, module, args.gpu, with_snr=args.snr, reps=args.reps)
        tflops = res.get("tflops_mean", res.get("tflops", 0))
        ratio = (tflops / INCUMBENT_TFLOPS * 100) if tflops else 0
        snr = res.get("snr_db", "")
        snr_str = f" snr={snr}dB" if snr != "" else ""
        print(f"  {tag:18s} TFLOPS_mean={tflops:>8} ratio_vs_inc={ratio:.2f}%{snr_str} wall={res['wall_sec']}s status={res.get('status','?')}", flush=True)
        results.append(res)

    out_json = args.out or os.path.join(SCRIPT_DIR, "R31_OPT_B_BENCH_RESULTS.json")
    with open(out_json, "w") as f:
        json.dump({
            "M": M, "N": N, "K": K,
            "incumbent_tflops": INCUMBENT_TFLOPS,
            "comp_tflops": COMP_TFLOPS,
            "warmup": 200, "iters": 500, "trim_frac": 0.10,
            "gpu_id": args.gpu, "reps": args.reps, "snr_check": args.snr,
            "results": results,
        }, f, indent=2)
    print(f"\nResults: {out_json}")


if __name__ == "__main__":
    main()
