#!/usr/bin/env python3
"""R42 Opt B: Fix the 2 carry-over CRASH shapes at K=28672.

Targets:
  (M=4096,  N=32768, K=28672) — competitor 5568.2 TFLOPS
  (M=16384, N=4096,  K=28672) — competitor 5525.3 TFLOPS

Both have FAIL_CRASH (HSA fault) since R37+ era. Parent variant is
`_ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all` (memc stripped per R40B).

Sweep cells (4 per shape):
  cell 'fence0_po104' = R41A_EXTRACT_TILE_FENCE=0, R25C kept at 104  (baseline)
  cell 'fence1_po104' = R41A_EXTRACT_TILE_FENCE=1, R25C kept at 104  (Phase 1, hypothesis b)
  cell 'fence0_po0'   = R41A_EXTRACT_TILE_FENCE=0, R25C overridden to 0 (Phase 1, hypothesis d)
  cell 'fence1_po0'   = R41A_EXTRACT_TILE_FENCE=1, R25C overridden to 0 (combined)

Per-cell macro stack (over R40B base):
  -DFUSED_STEP34=1
  -DR41A_DEEP_K_FIX=1
  -DR41A_PFOFF_OVERRIDE=<0|104>            (104 = no-op, 0 = disable tail-pf-off)
  -DR41A_EXTRACT_TILE_FENCE=<0|1>
  base CPPFLAGS minus '-mllvm -amdgpu-sched-strategy=max-memory-clause' (memc stripped)

Note: R41A_PFOFF_OVERRIDE only takes effect when != 0. To disable tail-pf-off
we override R25C_TAIL_PF_OFF_ITERS to 0 directly (bypassing the override
guard) by INSTEAD compiling with -DR25C_TAIL_PF_OFF_ITERS=0 (overrides the
parent flags' value). Cell 'fence*_po0' uses that path.

Module suffix: _R42B_<cell>
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
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R42B")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")),
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

# 2 K=28672 CRASH shapes
SHAPES = [
    (4096,  32768, 28672),
    (16384, 4096,  28672),
]

# Parent variant — must match an entry in bench_all_42.py 'variants' list
PARENT_VARIANT = "ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all"

# 4 cells: (cell_label, fence, override_pf_off_to_zero)
CELLS = [
    ("fence0_po104", 0, False),
    ("fence1_po104", 1, False),
    ("fence0_po0",   0, True),
    ("fence1_po0",   1, True),
]

sys.path.insert(0, SCRIPT_DIR)


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


def safe_tag(parent_tag):
    out = re.sub(r"(?<![A-Za-z0-9])memc(?![A-Za-z0-9])", "", parent_tag)
    out = re.sub(r"_memc(?=_|$)", "", out)
    out = re.sub(r"__+", "_", out)
    return out.strip("_")


def module_name_for(n_dim, k_dim, parent_tag, cell_label):
    return (f"tk_mxfp4_gluon_cpp_n{n_dim}_k{k_dim}_"
            f"{safe_tag(parent_tag)}_R42B_{cell_label}")


def build_one(n_dim, k_dim, parent_tag, base_cppflags, cell_label, fence, zero_pfoff):
    module_name = module_name_for(n_dim, k_dim, parent_tag, cell_label)
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(BUILD_DIR, out_file)
    key = f"N={n_dim},K={k_dim},cell={cell_label}"

    if os.path.exists(out_path):
        return (key, "cached", 0.0, module_name)

    with open(KERNEL_SRC) as f:
        kernel_src_text = f.read()
    patched = kernel_src_text.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},",
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n_dim}_k{k_dim}_{cell_label}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    extra_cppflags = strip_bad_sched(base_cppflags)
    # If we want to zero pfoff, strip the parent's R25C_TAIL_PF_OFF_ITERS=104 first
    # so our redefine wins.
    if zero_pfoff:
        extra_cppflags = re.sub(r"-DR25C_TAIL_PF_OFF_ITERS=\d+", "", extra_cppflags).strip()
        pfoff_macro = "-DR25C_TAIL_PF_OFF_ITERS=0"
    else:
        pfoff_macro = ""
    extra_macros = (
        f"-DFUSED_STEP34=1 "
        f"-DR41A_DEEP_K_FIX=1 "
        f"-DR41A_EXTRACT_TILE_FENCE={fence} "
        f"{pfoff_macro}"
    ).strip()
    extra_cppflags = (extra_macros + " " + extra_cppflags).strip()

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
        return (key, "FAILED", elapsed, module_name, result.stderr[-1500:])
    return (key, "ok", elapsed, module_name)


def main():
    args = sys.argv[1:]
    max_workers = int(args[0]) if args else 8

    os.makedirs(BUILD_DIR, exist_ok=True)
    variants = parse_variants_from_bench()
    suffix_to_flags = {}
    for sfx, flags in variants:
        tag = sfx.lstrip("_") if sfx else "default"
        suffix_to_flags[tag] = flags

    if PARENT_VARIANT not in suffix_to_flags:
        print(f"ERROR: parent variant {PARENT_VARIANT} not found in bench_all_42.py")
        print(f"  available with 'kx28672': "
              f"{[s for s in suffix_to_flags if 'kx28672' in s]}")
        sys.exit(1)
    base_flags = suffix_to_flags[PARENT_VARIANT]
    print(f"Parent variant: {PARENT_VARIANT}")
    print(f"Base flags: {base_flags}")

    # Unique builds keyed on (n,k,cell)
    plan = []
    for (m, n, k) in SHAPES:
        for (cell, fence, zero) in CELLS:
            plan.append((m, n, k, PARENT_VARIANT, base_flags, cell, fence, zero))

    unique = {}
    for (m, n, k, parent_tag, flags, cell, fence, zero) in plan:
        unique[(n, k, cell)] = (n, k, parent_tag, flags, cell, fence, zero)
    unique_tasks = list(unique.values())

    print(f"R42B builder: {len(plan)} (M,N,K,cell) entries, "
          f"{len(unique_tasks)} unique builds")
    print(f"  Cells: {[c[0] for c in CELLS]}")
    print(f"  Forced macros: FUSED_STEP34=1, R41A_DEEP_K_FIX=1")
    print(f"  memc sched-strategy stripped from CPPFLAGS")

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
                print(f"  [{completed:>3}/{len(unique_tasks)}] CACHED {key}", flush=True)
            elif status == "ok":
                built += 1
                el = res[2]
                print(f"  [{completed:>3}/{len(unique_tasks)}] BUILT {key} ({el:.1f}s)", flush=True)
            else:
                failed += 1
                err = res[4] if len(res) > 4 else ""
                failures.append((key, err))
                print(f"  [{completed:>3}/{len(unique_tasks)}] FAILED {key}", flush=True)

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s: built={built}, cached={cached}, failed={failed}")
    if failures:
        print("\nFailures:")
        for key, err in failures[:10]:
            print(f"  {key}")
            for line in err.strip().split("\n")[-8:]:
                print(f"    {line}")

    shape_cell_modules = {}
    for (m, n, k, parent_tag, flags, cell, fence, zero) in plan:
        shape_key = f"{m}x{n}x{k}"
        shape_cell_modules.setdefault(shape_key, {})[cell] = (
            module_name_for(n, k, parent_tag, cell)
        )

    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round": "R42B",
        "parent_variant": PARENT_VARIANT,
        "shapes": [list(s) for s in SHAPES],
        "cells": [{"label": c[0], "fence": c[1], "zero_pfoff": c[2]} for c in CELLS],
        "forced_macros": {
            "FUSED_STEP34": 1,
            "R41A_DEEP_K_FIX": 1,
        },
        "shapes_to_modules_per_cell": shape_cell_modules,
        "built": built,
        "cached": cached,
        "failed": failed,
    }
    out_manifest = "R42B_BUILD_MANIFEST.json"
    with open(os.path.join(SCRIPT_DIR, out_manifest), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {out_manifest}")


if __name__ == "__main__":
    main()
