#!/usr/bin/env python3
"""R33-OptA bench: 5-rep bench at warmup=200 iters=500 trim=10% on L6.

Usage:
  python3 bench_R33_optA.py --gpu N --reps 5
  python3 bench_R33_optA.py --gpu N --variants V1_relax15 --reps 1
"""
import os, sys, json, subprocess, textwrap, time, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

M, N, K = 4096, 32768, 128256
INCUMBENT_TFLOPS = 5354.0
COMP_TFLOPS = 5781.1
WIN_BAR = INCUMBENT_TFLOPS + 0.6 * INCUMBENT_TFLOPS / 100.0  # +0.6pp = 5386 TFLOPS

VARIANTS = [
    ("incumbent",        "tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all"),
    ("V1_relax15",       "tk_mxfp4_gluon_cpp_n32768_k128256_R33A_v12_memc_btw_all_relax15"),
    ("V2_relax25",       "tk_mxfp4_gluon_cpp_n32768_k128256_R33A_v12_memc_btw_all_relax25"),
    ("V3_relax10",       "tk_mxfp4_gluon_cpp_n32768_k128256_R33A_v12_memc_btw_all_relax10"),
    ("V4_snop1",         "tk_mxfp4_gluon_cpp_n32768_k128256_R33A_v12_memc_btw_all_snop1"),
    ("V5_relax15_snop1", "tk_mxfp4_gluon_cpp_n32768_k128256_R33A_v12_memc_btw_all_relax15_snop1"),
    ("V6_relax25_snop1", "tk_mxfp4_gluon_cpp_n32768_k128256_R33A_v12_memc_btw_all_relax25_snop1"),
]


def make_script(module_name, m, n, k):
    return textwrap.dedent(f'''\
        import gc, json, math, sys, torch
        torch.manual_seed(0)
        sys.path.insert(0, {BUILD_DIR!r})
        import {module_name} as MOD

        M, N, K = {m}, {n}, {k}
        WARMUP = 200; ITERS = 500; TRIM_FRAC = 0.10
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
            A = gen_fp4(M, K); B = gen_fp4(N, K)
            sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
            sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
            A_sc = preshuffle_mfma16_merged(sc_exp_a)
            B_sc = preshuffle_mfma16_merged(sc_exp_b)
            C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            run = lambda: MOD.gemm_rcr(A, B, A_sc, B_sc, C)
            for _ in range(WARMUP): run()
            torch.cuda.synchronize()

            times_ms = []
            for _ in range(ITERS):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); run(); e.record()
                torch.cuda.synchronize()
                times_ms.append(s.elapsed_time(e))
            times_ms.sort()
            tc = int(len(times_ms) * TRIM_FRAC)
            trimmed = times_ms[tc:-tc] if tc > 0 else times_ms
            avg_ms = sum(trimmed) / len(trimmed)
            tflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e12
            print(json.dumps({{"status": "OK", "avg_ms": round(avg_ms, 4), "tflops": round(tflops, 2)}}))
        except Exception as ex:
            print(json.dumps({{"status": "FAIL", "error": str(ex)[:300]}}))
    ''')


def run_one(module_name, gpu):
    script = make_script(module_name, M, N, K)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           env=env, capture_output=True, text=True, timeout=600)
        dt = time.time() - t0
        out = r.stdout.strip().splitlines()
        if not out:
            return {"status": "NO_OUTPUT", "wall_s": round(dt, 1), "stderr": r.stderr[-1000:]}
        # Find the JSON line
        for line in reversed(out):
            line = line.strip()
            if line.startswith("{"):
                try:
                    js = json.loads(line)
                    js["wall_s"] = round(dt, 1)
                    return js
                except:
                    pass
        return {"status": "NO_JSON", "wall_s": round(dt, 1), "stdout": "\n".join(out)[-500:]}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "wall_s": round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--out", default="R33_OPT_A_BENCH_RESULTS.json")
    args = p.parse_args()

    selected = VARIANTS if args.variants is None else [v for v in VARIANTS if v[0] in args.variants]
    print(f"Bench R33-A: {len(selected)} variants × {args.reps} reps on GPU{args.gpu}")
    print(f"  WIN_BAR = {WIN_BAR:.1f} TFLOPS (incumbent {INCUMBENT_TFLOPS} + 0.6pp)")
    print()

    results = {}
    for tag, mod_name in selected:
        reps_data = []
        print(f"[{tag}]")
        for r in range(args.reps):
            res = run_one(mod_name, args.gpu)
            if res.get("status") == "OK":
                tf = res["tflops"]
                marker = "WIN" if tf >= WIN_BAR else "ok"
                print(f"  rep{r+1}: {tf:.1f} TFLOPS  ({marker})  wall={res['wall_s']}s")
            else:
                print(f"  rep{r+1}: FAIL {res.get('status')}  wall={res.get('wall_s','?')}s err={res.get('error', res.get('stderr','')[:200])}")
                # First-fail = drop remaining reps for this variant
                reps_data.append(res); break
            reps_data.append(res)

        ok_tflops = [r["tflops"] for r in reps_data if r.get("status") == "OK"]
        if ok_tflops:
            sorted_tf = sorted(ok_tflops)
            n = len(sorted_tf)
            p10 = sorted_tf[max(0, int(n * 0.1))]
            p50 = sorted_tf[n // 2]
            p90 = sorted_tf[min(n - 1, int(n * 0.9))]
            best = sorted_tf[-1]
            mean_tf = sum(sorted_tf) / n
            verdict = "WIN" if best >= WIN_BAR else ("LOSS" if best < INCUMBENT_TFLOPS else "TIE")
            ratio = mean_tf / INCUMBENT_TFLOPS * 100
            print(f"  -> mean={mean_tf:.1f} best={best:.1f} p10={p10:.1f} p50={p50:.1f} p90={p90:.1f} ({ratio:.2f}% incumbent) [{verdict}]")
            results[tag] = {"reps": reps_data, "mean": mean_tf, "best": best,
                            "p10": p10, "p50": p50, "p90": p90,
                            "ratio_vs_incumbent": ratio, "verdict": verdict}
        else:
            results[tag] = {"reps": reps_data, "verdict": "DEAD-BY-CRASH"}
            print(f"  -> DEAD-BY-CRASH")
        print()

    out_path = os.path.join(SCRIPT_DIR, args.out)
    with open(out_path, "w") as f:
        json.dump({"WIN_BAR": WIN_BAR, "INCUMBENT_TFLOPS": INCUMBENT_TFLOPS,
                   "COMP_TFLOPS": COMP_TFLOPS, "results": results}, f, indent=2)
    print(f"Results: {out_path}")

    print("\n=== SUMMARY ===")
    print(f"  {'variant':22s} {'best':>8s} {'mean':>8s} {'ratio':>8s} verdict")
    for tag, r in results.items():
        if "best" in r:
            print(f"  {tag:22s} {r['best']:>8.1f} {r['mean']:>8.1f} {r['ratio_vs_incumbent']:>7.2f}% {r['verdict']}")
        else:
            print(f"  {tag:22s} {'-':>8s} {'-':>8s} {'-':>8s} {r['verdict']}")


if __name__ == "__main__":
    main()
