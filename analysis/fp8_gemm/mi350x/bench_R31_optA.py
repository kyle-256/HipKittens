#!/usr/bin/env python3
"""R31-OptA bench: single-rep on L6 for each UNROLL_K variant (+ incumbent).

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
    ("incumbent", "tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all"),
    ("u1",  "tk_mxfp4_gluon_cpp_n32768_k128256_R31A_ts_lgk2_v12_memc_btw_all_u1"),
    ("u2",  "tk_mxfp4_gluon_cpp_n32768_k128256_R31A_ts_lgk2_v12_memc_btw_all_u2"),
    ("u4",  "tk_mxfp4_gluon_cpp_n32768_k128256_R31A_ts_lgk2_v12_memc_btw_all_u4"),
    ("u16", "tk_mxfp4_gluon_cpp_n32768_k128256_R31A_ts_lgk2_v12_memc_btw_all_u16"),
    ("u32", "tk_mxfp4_gluon_cpp_n32768_k128256_R31A_ts_lgk2_v12_memc_btw_all_u32"),
]


def make_script(module_name, m, n, k, comp, snr_check=False):
    snr_block = ""
    if snr_check:
        snr_block = textwrap.dedent('''
            # SNR check vs torch reference (single small instance)
            try:
                snr_M = 64
                snr_K = 256
                snr_N = 64
                k_blocks_s = snr_K // 32
                A_s = gen_fp4(snr_M, snr_K)
                B_s = gen_fp4(snr_N, snr_K)
                sc_a = torch.randint(-2, 3, (snr_M, k_blocks_s), dtype=torch.int8, device="cuda")
                sc_b = torch.randint(-2, 3, (snr_N, k_blocks_s), dtype=torch.int8, device="cuda")
                # Build padded versions via the standard preshuffle
                A_p = gen_fp4(M, K); B_p = gen_fp4(N, K)
                A_p[:snr_M, :snr_K//2] = A_s
                B_p[:snr_N, :snr_K//2] = B_s
                sc_pa = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
                sc_pb = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
                sc_pa[:snr_M, :k_blocks_s] = sc_a
                sc_pb[:snr_N, :k_blocks_s] = sc_b
                A_psc = preshuffle_mfma16_merged(sc_pa)
                B_psc = preshuffle_mfma16_merged(sc_pb)
                C_chk = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
                MOD.gemm_rcr(A_p, B_p, A_psc, B_psc, C_chk)
                # Compare top-left snr_M x snr_N corner
                # Decode A_p[:snr_M, :snr_K//2] and B_p[:snr_N, :snr_K//2]
                def decode_fp4(packed):
                    LUT = torch.tensor([0,0.5,1,1.5,2,3,4,6,-0,-0.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32, device="cuda")
                    lo = packed & 0xF
                    hi = (packed >> 4) & 0xF
                    out = torch.empty(packed.shape[0], packed.shape[1]*2, dtype=torch.float32, device="cuda")
                    out[:, 0::2] = LUT[lo.long()]
                    out[:, 1::2] = LUT[hi.long()]
                    return out
                Af = decode_fp4(A_p[:snr_M, :snr_K//2])
                Bf = decode_fp4(B_p[:snr_N, :snr_K//2])
                # Apply scales (block of 32)
                sa = (sc_pa[:snr_M, :k_blocks_s].float()).repeat_interleave(32, dim=1)
                sb = (sc_pb[:snr_N, :k_blocks_s].float()).repeat_interleave(32, dim=1)
                Af = Af * (2.0 ** sa)
                Bf = Bf * (2.0 ** sb)
                ref = (Af @ Bf.T).to(torch.bfloat16)
                got = C_chk[:snr_M, :snr_N]
                err = (got.float() - ref.float())
                sig_pow = (ref.float() ** 2).mean().item()
                err_pow = (err ** 2).mean().item() + 1e-30
                snr_db = 10.0 * math.log10(sig_pow / err_pow) if sig_pow > 0 else float("nan")
            except Exception as e:
                snr_db = None
        ''')
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

            run = lambda: MOD.gemm_rcr(A, B, A_sc, B_sc, C)
            for _ in range(WARMUP):
                run()
            torch.cuda.synchronize()
            times_ms = []
            for _ in range(ITERS):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); run(); e.record(); torch.cuda.synchronize()
                times_ms.append(s.elapsed_time(e))
            times_ms.sort()
            tc = int(len(times_ms) * TRIM_FRAC)
            trimmed = times_ms[tc:-tc] if tc > 0 else times_ms
            avg_ms = sum(trimmed) / len(trimmed)
            tflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e12
            result = {{"M": M, "N": N, "K": K, "avg_ms": round(avg_ms, 4),
                       "tflops": round(tflops, 1), "comp": {comp}, "status": "OK"}}
        except Exception as ex:
            result = {{"M": M, "N": N, "K": K, "status": f"ERR:{{ex}}"}}

        print("BENCH_JSON_START")
        print(json.dumps(result))
        print("BENCH_JSON_END")
    ''')
    return script


def run_bench(tag, module_name, gpu_id):
    script = make_script(module_name, M, N, K, COMP_TFLOPS)
    script_path = os.path.join(SCRIPT_DIR, f"_R31A_run_{tag}.py")
    with open(script_path, "w") as f:
        f.write(script)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    r = subprocess.run(["python3", script_path], capture_output=True, text=True, env=env, timeout=900)
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
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Subset of variant tags to run")
    args = ap.parse_args()
    todo = VARIANTS
    if args.variants:
        todo = [(t, m) for t, m in VARIANTS if t in args.variants]
    print(f"R31-A bench on GPU {args.gpu}, {len(todo)} variants, sequential")
    results = []
    for tag, module in todo:
        print(f"\n[{tag}] running on GPU {args.gpu}...", flush=True)
        res = run_bench(tag, module, args.gpu)
        ratio = res.get("tflops", 0) / INCUMBENT_TFLOPS * 100 if res.get("tflops") else 0
        print(f"  {tag:10s} TFLOPS={res.get('tflops','?'):>8} ratio_vs_inc={ratio:.2f}% wall={res['wall_sec']}s status={res.get('status','?')}", flush=True)
        results.append(res)

    out_json = os.path.join(SCRIPT_DIR, "R31_OPT_A_BENCH_RESULTS.json")
    with open(out_json, "w") as f:
        json.dump({
            "M": M, "N": N, "K": K,
            "incumbent_tflops": INCUMBENT_TFLOPS,
            "comp_tflops": COMP_TFLOPS,
            "warmup": 200, "iters": 500, "trim_frac": 0.10,
            "gpu_id": args.gpu,
            "results": results,
        }, f, indent=2)
    print(f"\nResults: {out_json}")


if __name__ == "__main__":
    main()
