#!/usr/bin/env python3
"""R38 Opt F: Build BEST_VARIANTS with R37_FIX_B + R38F_TAIL_DRAIN ON.

Hypothesis F (per R38C verdict): the structural fix for the 6 WIN→WRONG_OUTPUT
and 6 CRASH→WRONG_OUTPUT demotions in R37_FIX_B + R25-C is to insert an
explicit `s_waitcnt vmcnt(0)[+s_barrier]` on the tail iters where R25-C drops
the prefetch. This forces any in-flight buffer_load_to_lds from the previous
iter to land before the next iter's s_barrier releases the LDS double-buffer
slot for reuse — eliminating the stale-LDS-read race that R38B/R38C revealed.

Variants tested separately via R38F_VARIANT={1,2,3,4}:
  F1: s_waitcnt vmcnt(0) only
  F2: s_waitcnt vmcnt(0) + s_barrier  (DEFAULT)
  F3: s_waitcnt 0  (everything)
  F4: s_waitcnt vmcnt(0) lgkmcnt(0)

Strips `-mllvm -amdgpu-sched-strategy=max-memory-clause` (same as R37/R38B/R38C).
Module names use a `_R38F{V}` suffix to keep .so files distinct per variant.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")),
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

SCHED_BAD = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

# Same 42-shape list and BEST_VARIANTS table used by R37/R38B/R38C.
BEST_VARIANTS = {
    (16384,  4096,  2048): "ts_v12_gm7_memc_pfoff4_kx2048_btw_all",
    (16384,  4096,  3072): "ts_gm6_v12_memc_dc_pfoff4",
    (16384,  6144,  2048): "ts_lgk2_gm6_v12_memc_pfoff4",
    (32768,  4096,  2048): "ts_lgk2_gm6_v12_memc_pfoff4",
    (32768,  4096,  3072): "ts_lgk2_memc_btw_all",
    (32768,  6144,  2048): "ts_lgk2_gm6_v12_memc_pfoff4",
    (16384, 14336,  2048): "ts_v12_gm7_memc_pfoff4_kx2048_btw_all",
    (16384, 28672,  2048): "ts_lgk2_gm6_v12_memc_pfoff4",
    (32768, 14336,  2048): "ts_gm6_v12_memc_dc_pfoff4",
    (32768, 28672,  2048): "ts_lgk2_gm6_v12_memc_pfoff4",
    (4096,   4096, 16384): "ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",
    (4096,  14336, 16384): "ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",
    (6144,   4096, 16384): "ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",
    (4096,   4096,  8192): "ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all",
    (4096,   4096, 32768): "ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all",
    (4096,   6144, 32768): "ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all",
    (4096,  14336,  8192): "ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all",
    (4096,  28672, 32768): "ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all",
    (4096,  32768,  4096): "ts_lgk2_gm7_v12_memc_pfoff14",
    (4096,  32768,  6144): "ts_v12_gm7_memc_pfoff19_kx6144_btw_all",
    (4096,  32768, 14336): "ts_v12_tv0_memc_btw_all",
    (4096,  32768, 28672): "ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all",
    (4096,  32768,128256): "ts_lgk2_v12_memc_btw_all",
    (4096, 128256, 32768): "ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all",
    (6144,   4096,  8192): "ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all",
    (6144,  32768,  4096): "ts_gm7_v12_memc_dc_pfoff14",
    (14336,  4096, 32768): "ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all",
    (14336, 32768,  4096): "ts_lgk2_gm7_v12_memc_pfoff14",
    (16384,  4096,  4096): "ts_gm7_v12_memc_dc_pfoff14",
    (16384,  4096,  6144): "ts_v12_gm7_memc_pfoff19_kx6144_btw_all",
    (16384,  4096,  7168): "ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all",
    (16384,  4096, 14336): "ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all",
    (16384,  4096, 28672): "ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all",
    (16384,  6144,  4096): "ts_lgk2_gm7_v12_memc_pfoff14",
    (16384, 14336,  4096): "ts_gm7_v12_memc_dc_pfoff14",
    (16384, 28672,  4096): "ts_gm7_v12_memc_dc_pfoff14",
    (28672,  4096,  8192): "ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all",
    (28672,  4096, 16384): "ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",
    (28672, 32768,  4096): "ts_gm7_v12_memc_dc_pfoff14",
    (32768,  4096,  7168): "ts_v12_tv0_gm7_memc_pfoff24_kx7168_btw_all",
    (32768,  4096, 14336): "ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all",
    (128256, 32768,  4096): "ts_lgk2_gm7_v12_memc_pfoff14",
}

# 9 CRASH shapes from R37_LEADERBOARD.md.
CRASH_SHAPES = [
    (4096,  32768,  6144),
    (14336, 32768,  4096),
    (16384,  4096,  3072),
    (16384, 28672,  2048),
    (28672, 32768,  4096),
    (32768,  4096,  2048),
    (32768,  6144,  2048),
    (32768, 28672,  2048),
    (128256, 32768, 4096),
]

# 14 R37 WIN shapes (verify no regression).
WIN_SHAPES = [
    (4096,    4096,  8192),
    (4096,    4096, 16384),
    (4096,   14336,  8192),
    (4096,   14336, 16384),
    (6144,    4096,  8192),
    (6144,    4096, 16384),
    (16384,   4096,  4096),
    (16384,   4096,  6144),
    (16384,   4096,  7168),
    (16384,   6144,  2048),
    (16384,   6144,  4096),
    (28672,   4096,  8192),
    (28672,   4096, 16384),
    (32768,   4096,  7168),
]


def parse_variants_from_bench():
    bench_path = os.path.join(SCRIPT_DIR, "bench_all_42.py")
    with open(bench_path) as f:
        src = f.read()
    pat = re.compile(r"variants\s*=\s*\[(.*?)^\s*\]", re.DOTALL | re.MULTILINE)
    m = pat.search(src)
    if not m:
        raise RuntimeError("could not parse variants from bench_all_42.py")
    src_list = "variants = [\n" + m.group(1) + "\n]"
    ns = {}
    exec(src_list, ns)
    return ns["variants"]


def strip_bad_sched(flags):
    out = re.sub(r"\s*-mllvm\s+-amdgpu-sched-strategy=max-memory-clause\b", "", flags)
    return out.strip()


def build_one(n_dim, k_dim, parent_tag, base_cppflags, variant, build_dir):
    suffix = f"R38F{variant}"
    module_name = f"tk_mxfp4_gluon_cpp_n{n_dim}_k{k_dim}_{parent_tag}_{suffix}"
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(build_dir, out_file)
    key = f"N={n_dim},K={k_dim},{parent_tag}_{suffix}"

    if os.path.exists(out_path):
        return (key, "cached", 0.0)

    with open(KERNEL_SRC) as f:
        kernel_src_text = f.read()
    patched = kernel_src_text.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},",
    )
    wrapper_src = os.path.join(build_dir, f"wrap_n{n_dim}_k{k_dim}_{parent_tag}_{suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    extra_cppflags = strip_bad_sched(base_cppflags)
    # R38F is layered on top of R38B (always-emit closes the CRASH; R38F drain
    # closes the stale-LDS-read WRONG_OUTPUT R38B reveals).
    extra_cppflags = (
        f"-DR38B_TAIL_FIX=1 -DR38F_TAIL_DRAIN=1 -DR38F_VARIANT={variant} "
        + extra_cppflags
    )

    t0 = time.time()
    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT
    cmd = (
        f'make -C {SCRIPT_DIR} TARGET={os.path.join(build_dir, module_name)} '
        f'SRC={wrapper_src} '
        f'CPPFLAGS="-DK_DIM={k_dim} -DN_DIM={n_dim} {extra_cppflags}"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0 or not os.path.exists(out_path):
        return (key, "FAILED", elapsed, result.stderr[-1500:])

    return (key, "ok", elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, default=2,
                    help="R38F_VARIANT 1/2/3/4 (default 2 = vmcnt(0)+s_barrier)")
    ap.add_argument("--all", action="store_true", help="Build all 42 shapes")
    ap.add_argument("--crash-and-win", action="store_true",
                    help="Build only the 9 CRASH + 14 WIN shapes (default)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shape", type=str, default=None,
                    help="Build only this shape, e.g. 32768x4096x2048")
    args = ap.parse_args()

    build_dir = os.path.join(SCRIPT_DIR, f"build_R38F{args.variant}")
    os.makedirs(build_dir, exist_ok=True)
    variants = parse_variants_from_bench()
    suffix_to_flags = {}
    for sfx, flags in variants:
        tag = sfx.lstrip("_") if sfx else "default"
        suffix_to_flags[tag] = flags

    if args.shape:
        m, n, k = (int(x) for x in args.shape.split("x"))
        shapes_to_build = [(m, n, k)]
    elif args.all:
        shapes_to_build = list(BEST_VARIANTS.keys())
    else:
        shapes_to_build = list(set(CRASH_SHAPES + WIN_SHAPES))

    tasks = []
    plan = []
    for (m, n, k) in shapes_to_build:
        parent_tag = BEST_VARIANTS[(m, n, k)]
        if parent_tag not in suffix_to_flags:
            print(f"  ERROR: variant {parent_tag} not in bench_all_42.py")
            sys.exit(1)
        flags = suffix_to_flags[parent_tag]
        tasks.append((n, k, parent_tag, flags, args.variant, build_dir))
        plan.append((m, n, k, parent_tag))

    unique_tasks = list({(t[0], t[1], t[2]): t for t in tasks}.values())
    print(f"R38F builder: {len(plan)} shape entries, {len(unique_tasks)} unique builds, R38F_VARIANT={args.variant}")
    print(f"  (R37_FIX_B=1 + R38F_TAIL_DRAIN=1; sched-strategy stripped)")

    t_start = time.time()
    completed = built = cached = failed = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_one, *t): t for t in unique_tasks}
        for fut in as_completed(futs):
            res = fut.result()
            completed += 1
            key, status = res[0], res[1]
            if status == "cached":
                cached += 1
            elif status == "ok":
                built += 1
                el = res[2]
                print(f"  [{completed:>3}/{len(unique_tasks)}] BUILT {key} ({el:.1f}s)")
            else:
                failed += 1
                err = res[3] if len(res) > 3 else ""
                failures.append((key, err))
                print(f"  [{completed:>3}/{len(unique_tasks)}] FAILED {key}")

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s: built={built}, cached={cached}, failed={failed}")
    if failures:
        print("\nFailures:")
        for key, err in failures[:10]:
            print(f"  {key}")
            for line in err.strip().split("\n")[-12:]:
                print(f"    {line}")

    suffix = f"R38F{args.variant}"
    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variant": args.variant,
        "shapes_to_R38F": {
            f"{m}x{n}x{k}": f"tk_mxfp4_gluon_cpp_n{n}_k{k}_{tag}_{suffix}"
            for (m, n, k, tag) in plan
        },
        "built": built,
        "cached": cached,
        "failed": failed,
    }
    manifest_path = os.path.join(SCRIPT_DIR, f"R38F{args.variant}_BUILD_MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: R38F{args.variant}_BUILD_MANIFEST.json")


if __name__ == "__main__":
    main()
