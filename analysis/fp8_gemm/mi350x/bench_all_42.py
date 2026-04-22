#!/usr/bin/env python3
"""Benchmark HipKittens MXFP4 kernel across all 42 shapes vs aiter ASM.

HipKittens-only — uses our own compiled kernel for every shape.
No aiter binary substitution.

Per-shape auto-tune over a small set of (compile_flag, GROUP_SIZE_M) variants.
hipEvent timing: warmup 200, iters 500, trimmed mean (10% each end).
Each shape runs in its own subprocess for crash isolation.

Usage:
    python3 bench_all_42.py                  # all 42 shapes
    python3 bench_all_42.py M N K            # single shape
    BENCH_VARIANTS=default,gm6 python3 ...   # limit variants (env)
    BENCH_GPUS=4,5,6,7 python3 ...           # parallel GPU set (env)
"""

import json
import math
import os
import subprocess
import sys
import sysconfig
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

# All 42 shapes: (M, N, K, competitor_tflops)
ALL_SHAPES = [
    (16384,  4096,  2048, 2995.0),
    (16384,  4096,  3072, 3492.3),
    (16384,  6144,  2048, 3047.6),
    (32768,  4096,  2048, 3131.8),
    (32768,  4096,  3072, 3630.6),
    (32768,  6144,  2048, 3239.9),
    (16384, 14336,  2048, 3301.3),
    (16384, 28672,  2048, 3482.3),
    (32768, 14336,  2048, 3351.4),
    (32768, 28672,  2048, 3353.4),
    (4096,   4096,  16384, 4642.1),
    (4096,  14336,  16384, 5013.0),
    (6144,   4096,  16384, 4428.1),
    (4096,   4096,   8192, 3959.9),
    (4096,   4096,  32768, 5152.8),
    (4096,   6144,  32768, 3784.2),
    (4096,  14336,   8192, 4345.8),
    (4096,  28672,  32768, 5649.9),
    (4096,  32768,   4096, 4166.5),
    (4096,  32768,   6144, 4548.6),
    (4096,  32768,  14336, 5296.1),
    (4096,  32768,  28672, 5568.2),
    (4096,  32768, 128256, 5781.1),
    (4096, 128256,  32768, 3195.3),
    (6144,   4096,   8192, 3822.0),
    (6144,  32768,   4096, 4291.0),
    (14336,  4096,  32768, 5245.4),
    (14336, 32768,   4096, 4462.6),
    (16384,  4096,   4096, 3951.8),
    (16384,  4096,   6144, 4259.9),
    (16384,  4096,   7168, 4443.2),
    (16384,  4096,  14336, 5142.1),
    (16384,  4096,  28672, 5525.3),
    (16384,  6144,   4096, 4042.5),
    (16384, 14336,   4096, 4255.8),
    (16384, 28672,   4096, 4411.7),
    (28672,  4096,   8192, 4810.0),
    (28672,  4096,  16384, 5350.6),
    (28672, 32768,   4096, 4466.6),
    (32768,  4096,   7168, 4666.8),
    (32768,  4096,  14336, 5223.4),
    (128256, 32768,  4096, 4536.4),
]

# Variant set — production-relevant only.
# All variants use FUSED_STEP34=1 + TAIL_SPLIT=1 (proven best baseline).
# Per memory `project_mxfp4_bpreshuffle_debug.md`:
#   default GM=4 wins on most shapes
#   GM=6 wins on large-M shapes (32768x14336x2048, 32768x28672x2048)
#   GLOBAL_B=1 fixes K=28672 CRASH shapes
_BASE = "-DFUSED_STEP34=1 -DTAIL_SPLIT=1"
DEFAULT_VARIANTS = [
    ("default",   _BASE),
    ("gm6",       _BASE + " -DGROUP_SIZE_M=6"),
    ("gb",        _BASE + " -DGLOBAL_B=1"),
    ("gb_gm6",    _BASE + " -DGLOBAL_B=1 -DGROUP_SIZE_M=6"),
    # R62 OPT-2: UNROLL_K variants help large-K wide-N LOSE shapes.
    # Sweep on 8 worst LOSE shapes (4 GPUs):
    #   unr2  best on (4096,32768,128256): 74.8% -> 76.3% (+1.5pp)
    #   unr16 best on (4096,28672,32768):  81.9% -> 83.3% (+1.4pp)
    ("unr2",      _BASE + " -DUNROLL_K=2"),
    ("unr16",     _BASE + " -DUNROLL_K=16"),
    # R63 OPT-1: GM=8 unlocks new WIN on 16384x6144x4096 (99.5% -> 102.2%).
    # Regresses some other shapes (per-shape autotune protects them).
    ("gm8",       _BASE + " -DGROUP_SIZE_M=8"),
    # R64 OPT-1+4: 3 NEW WINs unlocked via newly-discovered macro switches
    # and 2-knob combos.
    #   we1       locks 16384x4096x6144  (98.1% -> 100.6%)  via -DWAVES_PER_EU_1
    #   tbv16     locks 4096x4096x8192   (98.8% -> 101.3%)  via -DTAIL_BARRIER_VMCNT=16
    #   unr2_gm6  locks 32768x28672x2048 (99.1% -> 100.3%)  via UNROLL_K=2 + GM=6
    #   gb_unr2   backup for 16384x4096x6144 (100.2%)
    ("we1",       _BASE + " -DWAVES_PER_EU_1"),
    ("tbv16",     _BASE + " -DTAIL_BARRIER_VMCNT=16"),
    ("unr2_gm6",  _BASE + " -DUNROLL_K=2 -DGROUP_SIZE_M=6"),
    ("gb_unr2",   _BASE + " -DGLOBAL_B=1 -DUNROLL_K=2"),
    # R66 axis-A Opt-1: interleave 16 buffer_load_dwordx4 ... lds prefetches
    # into the kpair_64mfma_step34 asm block (true 4:1:1 MFMA:ds_read:bufload
    # pattern, mirrors aiter scheduling). Replaces post-block emit_pf_tail<0>.
    ("step34pf",  _BASE + " -DSTEP34_PF_INTERLEAVE=1"),
    # R67: cross-product of step34pf with knobs that were tuned on top of OLD
    # step34. With step34pf as new base, the same knobs may unlock new shapes.
    ("step34pf_gm6",   _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DGROUP_SIZE_M=6"),
    ("step34pf_gm8",   _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DGROUP_SIZE_M=8"),
    ("step34pf_unr2",  _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DUNROLL_K=2"),
    ("step34pf_we1",   _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DWAVES_PER_EU_1"),
    ("step34pf_tbv16", _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DTAIL_BARRIER_VMCNT=16"),
    # R68 OPT-1: triple-knob cross-product variants on top of step34pf to
    # capture boundary residue at 95-99.5% (10 candidate shapes).
    ("step34pf_gm6_unr2",   _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DGROUP_SIZE_M=6 -DUNROLL_K=2"),
    ("step34pf_gm8_we1",    _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DGROUP_SIZE_M=8 -DWAVES_PER_EU_1"),
    ("step34pf_gm6_tbv16",  _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DGROUP_SIZE_M=6 -DTAIL_BARRIER_VMCNT=16"),
    ("step34pf_unr2_tbv16", _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DUNROLL_K=2 -DTAIL_BARRIER_VMCNT=16"),
    ("step34pf_gb_gm6",     _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DGLOBAL_B=1 -DGROUP_SIZE_M=6"),
    ("step34pf_we1_unr2",   _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DWAVES_PER_EU_1 -DUNROLL_K=2"),
    # R70: UNROLL_K=4 reduces I-cache pressure on K-heavy shapes (K≥14336).
    # Compiler default unrolls 8-51× for large K; unr4 keeps it moderate.
    ("step34pf_unr4",       _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DUNROLL_K=4"),
    ("step34pf_unr4_tbv16", _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DUNROLL_K=4 -DTAIL_BARRIER_VMCNT=16"),
    ("step34pf_unr4_we1",   _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DUNROLL_K=4 -DWAVES_PER_EU_1"),
    ("step34pf_unr4_gm6",   _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DUNROLL_K=4 -DGROUP_SIZE_M=6"),
    # STEP12_SPLIT_PF: split step12 into 2x 32-MFMA + 4 prefetches between halves.
    # +1.7pp on K=128256, -1.5pp on K=32768. Autotune picks best per-shape.
    ("step34pf_s12split",       _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DSTEP12_SPLIT_PF=1"),
    ("step34pf_s12split_unr4",  _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DSTEP12_SPLIT_PF=1 -DUNROLL_K=4"),
    ("step34pf_s12split_gm6_tbv16", _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DSTEP12_SPLIT_PF=1 -DGROUP_SIZE_M=6 -DTAIL_BARRIER_VMCNT=16"),
    # R70+ aggressive sweep: best combos for hard losers (no NEW WINs but +0.8-2pp uplift)
    ("step34pf_unr6",          _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DUNROLL_K=6"),
    ("step34pf_unr3",          _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DUNROLL_K=3"),
    ("step34pf_unr4_gm8_we1",  _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DUNROLL_K=4 -DGROUP_SIZE_M=8 -DWAVES_PER_EU_1"),
    # R70+: triple-knob combos that unlock borderline shapes (16384x4096x14336 → WIN)
    ("step34pf_gm6_we1",        _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DGROUP_SIZE_M=6 -DWAVES_PER_EU_1"),
    ("step34pf_unr4_gm6_tbv16", _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DUNROLL_K=4 -DGROUP_SIZE_M=6 -DTAIL_BARRIER_VMCNT=16"),
    ("step34pf_gm6_tbv16_we1", _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DGROUP_SIZE_M=6 -DTAIL_BARRIER_VMCNT=16 -DWAVES_PER_EU_1"),
    ("step34pf_unr2_gm6_tbv16",_BASE + " -DSTEP34_PF_INTERLEAVE=1 -DUNROLL_K=2 -DGROUP_SIZE_M=6 -DTAIL_BARRIER_VMCNT=16"),
    # SPREAD_DS_READ: distribute ds_reads evenly across 32 MFMAs instead of front-loading
    ("step34pf_spread",         _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DSPREAD_DS_READ=1"),
    ("step34pf_spread_unr4",    _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DSPREAD_DS_READ=1 -DUNROLL_K=4"),
    ("step34pf_spread_gm6",     _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DSPREAD_DS_READ=1 -DGROUP_SIZE_M=6"),
    ("step34pf_spread_we1",     _BASE + " -DSTEP34_PF_INTERLEAVE=1 -DSPREAD_DS_READ=1 -DWAVES_PER_EU_1"),
    # Fused 128-MFMA: all 4 steps in one asm block, eliminates extract_tile overhead
    ("fused128",               _BASE + " -DFUSED_128=1"),
    ("fused128_gm6",           _BASE + " -DFUSED_128=1 -DGROUP_SIZE_M=6"),
    ("fused128_we1",           _BASE + " -DFUSED_128=1 -DWAVES_PER_EU_1"),
    ("fused128_unr4",          _BASE + " -DFUSED_128=1 -DUNROLL_K=4"),
]


def get_unique_nk_pairs(shapes):
    return sorted(set((n, k) for _, n, k, _ in shapes))


def module_name_for_nk(n, k):
    return f"tk_mxfp4_n{n}_k{k}"


def build_for_nk(n_dim, k_dim, build_dir, extra_cppflags="", suffix=""):
    module_name = module_name_for_nk(n_dim, k_dim) + suffix
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(build_dir, out_file)

    if os.path.exists(out_path):
        return out_path

    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(build_dir, f"wrap_n{n_dim}_k{k_dim}{suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    print(f"  Compiling N={n_dim}, K={k_dim} {suffix} ...", end=" ", flush=True)
    t0 = time.time()

    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT

    cmd = (
        f'make -C {SCRIPT_DIR} TARGET={os.path.join(build_dir, module_name)} '
        f'SRC={wrapper_src} '
        f'CPPFLAGS="-DK_DIM={k_dim} -DN_DIM={n_dim} {extra_cppflags}"'
    )
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, env=env
    )
    elapsed = time.time() - t0

    if result.returncode != 0 or not os.path.exists(out_path):
        print(f"FAILED ({elapsed:.1f}s)")
        sys.stderr.write(result.stderr[-2000:] + "\n")
        return None

    print(f"OK ({elapsed:.1f}s)")
    return out_path


def make_runner_script(module_name, so_dir, m, n, k, comp_tflops):
    return textwrap.dedent(f'''\
        #!/usr/bin/env python3
        import gc, json, math, sys, torch
        torch.manual_seed(0)
        sys.path.insert(0, {so_dir!r})
        import {module_name}

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
            run = lambda: {module_name}.gemm_rcr(A, B, A_sc, B_sc, C)

            for _ in range(WARMUP):
                run()
            torch.cuda.synchronize()

            times_ms = []
            for _ in range(ITERS):
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                run()
                end_evt.record()
                torch.cuda.synchronize()
                times_ms.append(start_evt.elapsed_time(end_evt))

            times_ms.sort()
            trim_count = int(len(times_ms) * TRIM_FRAC)
            trimmed = times_ms[trim_count:-trim_count] if trim_count > 0 else times_ms
            avg_ms = sum(trimmed) / len(trimmed)
            tflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e12
            result = {{"M":M,"N":N,"K":K,"avg_ms":round(avg_ms,4),
                      "tflops":round(tflops,1),"comp":{comp_tflops},"status":"OK"}}
        except torch.cuda.OutOfMemoryError:
            result = {{"M":M,"N":N,"K":K,"avg_ms":None,"tflops":None,
                      "comp":{comp_tflops},"status":"OOM"}}
        except Exception as e:
            result = {{"M":M,"N":N,"K":K,"avg_ms":None,"tflops":None,
                      "comp":{comp_tflops},"status":f"ERR:{{e}}"}}

        print("BENCH_JSON_START")
        print(json.dumps(result))
        print("BENCH_JSON_END")
    ''')


def run_single(m, n, k, comp, build_dir, work_dir, suffix, gpu=None):
    module_name = module_name_for_nk(n, k) + suffix
    script = make_runner_script(module_name, build_dir, m, n, k, comp)
    runner_path = os.path.join(work_dir, f"_run_{m}_{n}_{k}{suffix}.py")
    with open(runner_path, "w") as f:
        f.write(script)
    env = os.environ.copy()
    if gpu is not None:
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
    try:
        result = subprocess.run(
            [sys.executable, runner_path],
            capture_output=True, text=True, timeout=900, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"M":m,"N":n,"K":k,"avg_ms":None,"tflops":None,
                "comp":comp,"status":"TIMEOUT"}
    if result.returncode != 0:
        return {"M":m,"N":n,"K":k,"avg_ms":None,"tflops":None,
                "comp":comp,"status":"CRASH"}
    stdout = result.stdout
    try:
        s = stdout.index("BENCH_JSON_START") + len("BENCH_JSON_START")
        e = stdout.index("BENCH_JSON_END")
        return json.loads(stdout[s:e].strip())
    except (ValueError, json.JSONDecodeError):
        return {"M":m,"N":n,"K":k,"avg_ms":None,"tflops":None,
                "comp":comp,"status":"PARSE_FAIL"}


def shape_worker(args):
    m, n, k, comp, build_dir, work_dir, variants, gpu = args
    best_r = None
    best_t = 0.0
    best_tag = ""
    for tag, _flags in variants:
        r = run_single(m, n, k, comp, build_dir, work_dir, "_" + tag, gpu=gpu)
        t = r.get("tflops") or 0.0
        if t > best_t:
            best_t = t
            best_r = r
            best_tag = tag
    if best_r is None:
        best_r = r
    best_r["best_tag"] = best_tag
    return best_r


def main():
    target_shapes = ALL_SHAPES
    if len(sys.argv) >= 4:
        m, n, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        comp_map = {(M,N,K): C for (M,N,K,C) in ALL_SHAPES}
        comp = comp_map.get((m,n,k), 0.0)
        target_shapes = [(m, n, k, comp)]

    variant_filter = os.environ.get("BENCH_VARIANTS", "").strip()
    if variant_filter:
        wanted = set(variant_filter.split(","))
        variants = [(t, f) for (t, f) in DEFAULT_VARIANTS if t in wanted]
    else:
        variants = DEFAULT_VARIANTS

    gpus = os.environ.get("BENCH_GPUS", "").strip()
    gpu_list = [int(x) for x in gpus.split(",")] if gpus else [None]
    parallel = len(gpu_list) > 1

    print("=" * 70)
    print(f"HipKittens MXFP4 — {len(target_shapes)} shape(s), "
          f"{len(variants)} variant(s), {len(gpu_list)} GPU(s)")
    print(f"Variants: {[t for t,_ in variants]}")
    print(f"GPUs: {gpu_list}")
    print("=" * 70)

    build_dir = os.environ.get("BENCH_BUILD_DIR", os.path.join(SCRIPT_DIR, "build_all42"))
    work_dir = os.environ.get("BENCH_WORK_DIR", os.path.join(SCRIPT_DIR, "work_all42"))
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Build phase
    nk_pairs = sorted(set((n,k) for _,n,k,_ in target_shapes))
    print(f"\n--- Build Phase ({len(nk_pairs)} (N,K) pairs × {len(variants)} variants) ---")
    for n_val, k_val in nk_pairs:
        for tag, flags in variants:
            so_path = build_for_nk(n_val, k_val, build_dir,
                                   extra_cppflags=flags, suffix="_" + tag)
            if so_path is None and tag == "default":
                print(f"FATAL: default variant compile failed for N={n_val}, K={k_val}")
                sys.exit(1)

    print("\n--- Benchmark Phase ---")
    all_results = []

    if parallel:
        tasks = []
        for i, (m,n,k,comp) in enumerate(target_shapes):
            gpu = gpu_list[i % len(gpu_list)]
            tasks.append((m,n,k,comp,build_dir,work_dir,variants,gpu))
        with ProcessPoolExecutor(max_workers=len(gpu_list)) as ex:
            futures = {ex.submit(shape_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                r = fut.result()
                all_results.append(r)
                if r["status"] == "OK":
                    ratio = r["tflops"]/r["comp"]*100.0
                    tag = "WIN" if r["tflops"] >= r["comp"] else "LOSE"
                    print(f"  {r['M']:>6}x{r['N']:>6}x{r['K']:>6}  "
                          f"{r['tflops']:>7.1f} / {r['comp']:>7.1f}  "
                          f"({ratio:>5.1f}%)  {tag}  [{r['best_tag']}]")
                else:
                    print(f"  {r['M']:>6}x{r['N']:>6}x{r['K']:>6}  {r['status']}")
        all_results.sort(key=lambda r: ALL_SHAPES.index(
            next(s for s in ALL_SHAPES if s[0]==r["M"] and s[1]==r["N"] and s[2]==r["K"])
        ) if any(s[0]==r["M"] and s[1]==r["N"] and s[2]==r["K"] for s in ALL_SHAPES) else 999)
    else:
        for idx, (m,n,k,comp) in enumerate(target_shapes):
            print(f"[{idx+1:>2}/{len(target_shapes)}] {m:>6}x{n:>6}x{k:>6}",
                  end="  ", flush=True)
            r = shape_worker((m,n,k,comp,build_dir,work_dir,variants,gpu_list[0]))
            all_results.append(r)
            if r["status"] == "OK":
                ratio = r["tflops"]/r["comp"]*100.0
                tag = "WIN" if r["tflops"] >= r["comp"] else "LOSE"
                print(f"{r['tflops']:>7.1f} / {r['comp']:>7.1f}  "
                      f"({ratio:>5.1f}%)  {tag}  [{r['best_tag']}]")
            else:
                print(r["status"])

    # Summary
    print("\n" + "=" * 70)
    wins = sum(1 for r in all_results
               if r["status"]=="OK" and r["tflops"] >= r["comp"])
    losses = sum(1 for r in all_results
                 if r["status"]=="OK" and r["tflops"] < r["comp"])
    errors = sum(1 for r in all_results if r["status"] != "OK")
    valid = [r for r in all_results if r["status"] == "OK"]
    avg_ratio = (sum(r["tflops"]/r["comp"] for r in valid) / len(valid) * 100
                 if valid else 0.0)
    print(f"WIN: {wins}/{len(target_shapes)}  LOSE: {losses}  ERR: {errors}  "
          f"mean: {avg_ratio:.1f}%")
    print("=" * 70)

    results_file = os.path.join(SCRIPT_DIR, "bench_all42_results.json")
    with open(results_file, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "variants": [t for t,_ in variants],
                   "wins": wins, "losses": losses, "errors": errors,
                   "mean_ratio": round(avg_ratio, 2),
                   "results": all_results}, f, indent=2)
    print(f"Results: {results_file}")


if __name__ == "__main__":
    main()
