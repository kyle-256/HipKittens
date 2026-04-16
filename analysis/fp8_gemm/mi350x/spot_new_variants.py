#!/usr/bin/env python3
"""Spot-test near-threshold shapes with NEW cross-product variants not yet in the 90-variant suite.

Tests genuinely new combinations:
- PF_N=2/1 (untested PF depths)
- PF4 × LGK2/V12 (PF depth × barrier/wait timing)
- EXT_BR × LGK2 (external BR × LDS wait)
- GM × LGK (tile scheduling × LDS wait)
- TAIL_VMCNT (separate tail barrier VMCNT)
- Asymmetric PF (STEP3_PF_N != STEP4_PF_N)

Usage:
    HIP_VISIBLE_DEVICES=1 python3 spot_new_variants.py [shape_idx]
    # shape_idx 0-6 for the 7 near-threshold shapes
    # or: HIP_VISIBLE_DEVICES=1 python3 spot_new_variants.py M N K competitor_tflops
"""
import sys, os, subprocess, json, math, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

WARMUP = 200
ITERS = 500
TRIM = 0.10

# 7 near-threshold LOSE shapes (97-99.5%)
NEAR_THRESHOLD_SHAPES = [
    (6144,  32768,  4096,  4291.0),   # 0: 99.5% ts_lgk2
    (32768, 28672,  2048,  3353.4),   # 1: 99.1% ts_gm2_v12
    (4096,  14336,  8192,  4345.8),   # 2: 98.9% lgk2
    (6144,   4096, 16384,  4428.1),   # 3: 98.8% ts_v4
    (4096,  32768,  4096,  4166.5),   # 4: 98.6% ts_gm2_v12
    (16384,  4096, 14336,  5142.1),   # 5: 98.4% ts_pf4
    (28672,  4096,  8192,  4810.0),   # 6: 97.5% gm8_v12
]

# NEW variants (not in the existing 90-variant suite)
NEW_VARIANTS = [
    # PF depth = 2 (never tested)
    ("pf2", "-DSTEP3_PF_N=2 -DSTEP4_PF_N=2"),
    ("ts_pf2", "-DTAIL_SPLIT=1 -DSTEP3_PF_N=2 -DSTEP4_PF_N=2"),
    ("ts_pf2_lgk2", "-DTAIL_SPLIT=1 -DSTEP3_PF_N=2 -DSTEP4_PF_N=2 -DSTEP12_BR_LGKMCNT=2"),
    # PF depth = 1 (minimal PF interleaving)
    ("pf1", "-DSTEP3_PF_N=1 -DSTEP4_PF_N=1"),
    ("ts_pf1", "-DTAIL_SPLIT=1 -DSTEP3_PF_N=1 -DSTEP4_PF_N=1"),
    # Asymmetric PF (different PF depth for Step3 vs Step4)
    ("pf_asym_2_8", "-DSTEP3_PF_N=2 -DSTEP4_PF_N=8"),
    ("ts_pf_asym_2_8", "-DTAIL_SPLIT=1 -DSTEP3_PF_N=2 -DSTEP4_PF_N=8"),
    ("pf_asym_8_2", "-DSTEP3_PF_N=8 -DSTEP4_PF_N=2"),
    ("ts_pf_asym_8_2", "-DTAIL_SPLIT=1 -DSTEP3_PF_N=8 -DSTEP4_PF_N=2"),
    # PF4 × LGK2 (PF depth × LDS wait — genuinely new cross)
    ("pf4_lgk2", "-DSTEP3_PF_N=4 -DSTEP4_PF_N=4 -DSTEP12_BR_LGKMCNT=2"),
    ("ts_pf4_lgk2", "-DTAIL_SPLIT=1 -DSTEP3_PF_N=4 -DSTEP4_PF_N=4 -DSTEP12_BR_LGKMCNT=2"),
    # PF4 × V12 (PF depth × barrier VMCNT)
    ("pf4_v12", "-DSTEP3_PF_N=4 -DSTEP4_PF_N=4 -DSTEP3_BARRIER_VMCNT=12"),
    ("ts_pf4_v12", "-DTAIL_SPLIT=1 -DSTEP3_PF_N=4 -DSTEP4_PF_N=4 -DSTEP3_BARRIER_VMCNT=12"),
    # PF4 × NO_EMBED (PF depth × barrier embedding)
    ("ts_pf4_no_embed", "-DTAIL_SPLIT=1 -DSTEP3_PF_N=4 -DSTEP4_PF_N=4 -DSTEP3_EMBED_BARRIER=0"),
    # EXT_BR × LGK2 (external BR × LDS wait — genuinely new)
    ("ext_br_lgk2", "-DSTEP4_EXTERNAL_BR_PREFETCH=1 -DSTEP12_BR_LGKMCNT=2"),
    ("ts_ext_br_lgk2", "-DTAIL_SPLIT=1 -DSTEP4_EXTERNAL_BR_PREFETCH=1 -DSTEP12_BR_LGKMCNT=2"),
    ("ts_ext_br_lgk2_v12", "-DTAIL_SPLIT=1 -DSTEP4_EXTERNAL_BR_PREFETCH=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12"),
    # GM × LGK (tile scheduling × LDS wait — genuinely new)
    ("gm2_lgk2", "-DGROUP_SIZE_M=2 -DSTEP12_BR_LGKMCNT=2"),
    ("ts_gm2_lgk2", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP12_BR_LGKMCNT=2"),
    ("gm8_lgk2", "-DGROUP_SIZE_M=8 -DSTEP12_BR_LGKMCNT=2"),
    ("ts_gm8_lgk2", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP12_BR_LGKMCNT=2"),
    ("ts_gm2_lgk2_v12", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12"),
    ("ts_gm8_lgk2_v12", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12"),
    # TAIL_VMCNT tuning (NEW flag — different VMCNT for tail barrier)
    ("ts_tv0", "-DTAIL_SPLIT=1 -DTAIL_BARRIER_VMCNT=0"),
    ("ts_tv0_lgk2", "-DTAIL_SPLIT=1 -DTAIL_BARRIER_VMCNT=0 -DSTEP12_BR_LGKMCNT=2"),
    ("ts_tv4", "-DTAIL_SPLIT=1 -DTAIL_BARRIER_VMCNT=4"),
    ("ts_tv0_v12", "-DTAIL_SPLIT=1 -DTAIL_BARRIER_VMCNT=0 -DSTEP3_BARRIER_VMCNT=12"),
    ("ts_tv16", "-DTAIL_SPLIT=1 -DTAIL_BARRIER_VMCNT=16"),
    ("ts_tv0_no_embed", "-DTAIL_SPLIT=1 -DTAIL_BARRIER_VMCNT=0 -DSTEP3_EMBED_BARRIER=0"),
    ("ts_tv0_no_embed_v12", "-DTAIL_SPLIT=1 -DTAIL_BARRIER_VMCNT=0 -DSTEP3_EMBED_BARRIER=0 -DSTEP3_BARRIER_VMCNT=12"),
]

# Also include best-known variants for reference/comparison
REFERENCE_VARIANTS = [
    ("default", ""),
    ("ts_lgk2", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2"),
    ("ts_gm2_v12", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12"),
    ("lgk2", "-DSTEP12_BR_LGKMCNT=2"),
    ("ts_v4", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=4"),
    ("ts_pf4", "-DTAIL_SPLIT=1 -DSTEP3_PF_N=4 -DSTEP4_PF_N=4"),
    ("gm8_v12", "-DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12"),
    ("ts_no_embed_v12", "-DTAIL_SPLIT=1 -DSTEP3_EMBED_BARRIER=0 -DSTEP3_BARRIER_VMCNT=12"),
    ("ts_v16", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=16"),
]


def build(n, k, tag, flags, build_dir):
    so_name = f"tk_mxfp4_gluon_cpp_n{n}_k{k}_{tag}.cpython-310-x86_64-linux-gnu.so"
    so_path = os.path.join(build_dir, so_name)
    if os.path.exists(so_path):
        return so_path
    cmd = (
        f"/opt/rocm/bin/hipcc {SCRIPT_DIR}/kernel_mxfp4_gluon_cpp.cpp "
        f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
        f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
        f"-I/opt/rocm/include/hip "
        f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
        f"-shared -fPIC -std=c++20 -w "
        f"-DN_DIM={n} -DK_DIM={k} {flags} "
        f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm "
        f"-o {so_path}"
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"  BUILD FAIL [{tag}]: {r.stderr[-200:]}", file=sys.stderr)
        return None
    return so_path


def bench_one(so_path, m, n, k, tag):
    script = f"""
import sys, math, torch, importlib.util
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda') << 4) | torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda')
def preshuffle(se):
    r, kb = se.shape; pr = math.ceil(r/64)*64; pk = math.ceil(kb/8)*8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=se.device)
    raw[:r,:kb] = (se.to(torch.int16)+127).to(torch.uint8)
    sh = raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh = sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)
spec = importlib.util.spec_from_file_location('tk_mxfp4_gluon_cpp', '{so_path}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
M, N, K = {m}, {n}, {k}
A = gen_fp4(M, K); B = gen_fp4(N, K)
sc_a = torch.randint(-2, 3, (M, K//32), dtype=torch.int8, device='cuda')
sc_b = torch.randint(-2, 3, (N, K//32), dtype=torch.int8, device='cuda')
A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)
C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
run = lambda: mod.gemm_rcr(A, B, A_sc, B_sc, C)
for _ in range(WARMUP): run()
torch.cuda.synchronize()
times = []
for _ in range(ITERS):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort(); trim = int(len(times)*TRIM)
times = times[trim:-trim] if trim > 0 else times
avg = sum(times)/len(times)
t = 2.0*M*N*K/(avg*1e-3)/1e12
import json
print(json.dumps({{"tflops": round(t, 1), "ms": round(avg, 4)}}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout.strip())
    except Exception:
        return None


def main():
    if len(sys.argv) == 5:
        m, n, k, comp = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
        shapes = [(m, n, k, comp)]
    elif len(sys.argv) == 2:
        idx = int(sys.argv[1])
        shapes = [NEAR_THRESHOLD_SHAPES[idx]]
    else:
        shapes = NEAR_THRESHOLD_SHAPES

    build_dir = os.path.join(SCRIPT_DIR, "build_spot_new")
    os.makedirs(build_dir, exist_ok=True)

    all_variants = REFERENCE_VARIANTS + NEW_VARIANTS
    results = {}

    for m, n, k, comp in shapes:
        shape_key = f"{m}x{n}x{k}"
        print(f"\n{'='*70}")
        print(f"Shape: {shape_key}  (competitor: {comp:.1f} TFLOPS)")
        print(f"{'='*70}")

        best_t, best_tag = 0, "?"
        shape_results = []

        for tag, flags in all_variants:
            print(f"  [{tag:30s}] ", end="", flush=True)
            so = build(n, k, tag, flags, build_dir)
            if so is None:
                print("BUILD FAIL")
                continue
            r = bench_one(so, m, n, k, tag)
            if r is None:
                print("BENCH FAIL")
                continue
            tflops = r["tflops"]
            ratio = tflops / comp * 100
            marker = " ***NEW BEST***" if tflops > best_t else ""
            win = " WIN!" if ratio >= 100 else ""
            print(f"{tflops:7.1f} T  ({ratio:5.1f}%){win}{marker}")
            shape_results.append((tag, tflops, ratio))
            if tflops > best_t:
                best_t = tflops
                best_tag = tag

        print(f"\n  BEST: {best_tag} = {best_t:.1f} TFLOPS ({best_t/comp*100:.1f}%)")
        results[shape_key] = {"best_tag": best_tag, "best_tflops": best_t,
                              "ratio": round(best_t / comp * 100, 1),
                              "all": shape_results}

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    wins = 0
    for sk, rd in results.items():
        w = "WIN" if rd["ratio"] >= 100 else "LOSE"
        if w == "WIN":
            wins += 1
        print(f"  {sk:30s}  {rd['best_tag']:30s}  {rd['best_tflops']:7.1f}T  {rd['ratio']:5.1f}%  {w}")
    print(f"\n  Total WINs among tested shapes: {wins}/{len(results)}")

    # Save results
    out_path = os.path.join(SCRIPT_DIR, "spot_new_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
