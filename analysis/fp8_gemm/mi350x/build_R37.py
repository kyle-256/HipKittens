#!/usr/bin/env python3
"""R37: Build BEST_VARIANTS with R37_FIX_B (default ON in source) — NO -DFUSED_STEP34=1.

Key change vs R36_F34: we STRIP `-mllvm -amdgpu-sched-strategy=max-memory-clause`
from each variant's CPPFLAGS because that scheduler flag is incompatible with the
fused step34 path (it reorders the buffer_load_to_lds prefetch intrinsics across
iteration boundaries, corrupting the LDS double-buffer state machine).

Module names use a `_R37` suffix (and we still tag with the parent variant) to keep
the .so files distinct from any prior R35/R36 binaries.
"""
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
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R37")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")),
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

# R37: scheduler flag that breaks fused step34 — strip from all variant flags.
SCHED_BAD = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

ALL_SHAPES = [
    (16384,  4096,  2048, 2995.0), (16384,  4096,  3072, 3492.3),
    (16384,  6144,  2048, 3047.6), (32768,  4096,  2048, 3131.8),
    (32768,  4096,  3072, 3630.6), (32768,  6144,  2048, 3239.9),
    (16384, 14336,  2048, 3301.3), (16384, 28672,  2048, 3482.3),
    (32768, 14336,  2048, 3351.4), (32768, 28672,  2048, 3353.4),
    (4096,   4096,  16384, 4642.1), (4096,  14336,  16384, 5013.0),
    (6144,   4096,  16384, 4428.1), (4096,   4096,   8192, 3959.9),
    (4096,   4096,  32768, 5152.8), (4096,   6144,  32768, 3784.2),
    (4096,  14336,   8192, 4345.8), (4096,  28672,  32768, 5649.9),
    (4096,  32768,  4096, 4166.5), (4096,  32768,  6144, 4548.6),
    (4096,  32768,  14336, 5296.1), (4096,  32768,  28672, 5568.2),
    (4096,  32768, 128256, 5781.1), (4096, 128256,  32768, 3195.3),
    (6144,   4096,   8192, 3822.0), (6144,  32768,   4096, 4291.0),
    (14336,  4096,  32768, 5245.4), (14336, 32768,   4096, 4462.6),
    (16384,  4096,   4096, 3951.8), (16384,  4096,   6144, 4259.9),
    (16384,  4096,   7168, 4443.2), (16384,  4096,  14336, 5142.1),
    (16384,  4096,  28672, 5525.3), (16384,  6144,   4096, 4042.5),
    (16384, 14336,   4096, 4255.8), (16384, 28672,   4096, 4411.7),
    (28672,  4096,   8192, 4810.0), (28672,  4096,  16384, 5350.6),
    (28672, 32768,   4096, 4466.6), (32768,  4096,   7168, 4666.8),
    (32768,  4096,  14336, 5223.4), (128256, 32768,  4096, 4536.4),
]

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
    # Remove `-mllvm -amdgpu-sched-strategy=max-memory-clause` (any spacing).
    out = re.sub(r"\s*-mllvm\s+-amdgpu-sched-strategy=max-memory-clause\b", "", flags)
    return out.strip()


def build_one(n_dim, k_dim, parent_tag, base_cppflags):
    module_name = f"tk_mxfp4_gluon_cpp_n{n_dim}_k{k_dim}_{parent_tag}_R37"
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(BUILD_DIR, out_file)
    key = f"N={n_dim},K={k_dim},{parent_tag}_R37"

    if os.path.exists(out_path):
        return (key, "cached", 0.0)

    with open(KERNEL_SRC) as f:
        kernel_src_text = f.read()
    patched = kernel_src_text.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},",
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n_dim}_k{k_dim}_{parent_tag}_R37.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    extra_cppflags = strip_bad_sched(base_cppflags)
    # R37_FIX_B is ON by default in source; nothing to add.

    t0 = time.time()
    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT
    cmd = (
        f'make -C {SCRIPT_DIR} TARGET={os.path.join(BUILD_DIR, module_name)} '
        f'SRC={wrapper_src} '
        f'CPPFLAGS="-DK_DIM={k_dim} -DN_DIM={n_dim} {extra_cppflags}"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0 or not os.path.exists(out_path):
        return (key, "FAILED", elapsed, result.stderr[-1500:])

    return (key, "ok", elapsed)


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 24

    os.makedirs(BUILD_DIR, exist_ok=True)
    variants = parse_variants_from_bench()
    suffix_to_flags = {}
    for sfx, flags in variants:
        tag = sfx.lstrip("_") if sfx else "default"
        suffix_to_flags[tag] = flags

    tasks = []
    plan = []
    for (m, n, k), parent_tag in BEST_VARIANTS.items():
        if parent_tag not in suffix_to_flags:
            print(f"  ERROR: variant {parent_tag} not in bench_all_42.py")
            sys.exit(1)
        flags = suffix_to_flags[parent_tag]
        tasks.append((n, k, parent_tag, flags))
        plan.append((m, n, k, parent_tag))

    unique_tasks = list(set(tasks))
    print(f"R37 builder: {len(plan)} shape entries, {len(unique_tasks)} unique builds")
    print(f"  (R37_FIX_B=1 default; -mllvm -amdgpu-sched-strategy=max-memory-clause stripped)")

    t_start = time.time()
    completed = built = cached = failed = 0
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
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
            for line in err.strip().split("\n")[-3:]:
                print(f"    {line}")

    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shapes_to_R37": {
            f"{m}x{n}x{k}": f"tk_mxfp4_gluon_cpp_n{n}_k{k}_{tag}_R37"
            for (m, n, k, tag) in plan
        },
        "built": built,
        "cached": cached,
        "failed": failed,
    }
    with open(os.path.join(SCRIPT_DIR, "R37_BUILD_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Manifest: R37_BUILD_MANIFEST.json")


if __name__ == "__main__":
    main()
