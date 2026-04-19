#!/usr/bin/env python3
"""R43 Opt A: build A.fix1 (R43A_GATE_PF_TAIL_KBOUND) for the 2 K=28672 CRASH shapes.

A.fix1 wraps the unconditional `emit_pf_tail<0>(...)` calls in the FUSED_STEP34
path with `if (bt < k_byte_iters - 2) { ... }` to skip the racey final-iter
prefetch. Default OFF macro; R41A behavior preserved when unset.

Reference:
- R42_OPT_B_VERDICT.md (CRASH localized to FUSED_STEP34=1 AND TAIL_SPLIT=1).
- bench rules: warmup=200, iters=500, trim=0.10.
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
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R43A")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")),
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

# Two CRASH shapes per R43_DECIDER_PER_SHAPE.json
SHAPES = [
    (4096,  32768, 28672, 5568.2),
    (16384, 4096,  28672, 5525.3),
]

# R41 integration manifest assigns these the R40B parent variant
PARENT_VARIANT = "ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all"

# Cells (each builds a separate .so) — R43A_GATE_PF_TAIL_KBOUND value:
#   p2b_ctrl   = 0  (control = R42 baseline CRASH)
#   p2b_fix1   = 1  (skip ALL pf on clamped iter)
#   p2b_fix1b  = 2  (skip only B-half on clamped iter)
#   p2b_fix1c  = 3  (skip only A-half on clamped iter)
#   p2b_fix1d  = 4  (replace data prefetch with L2-only on clamped iter)
CELLS = [
    ("p2b_ctrl",  0),
    ("p2b_fix1",  1),
    ("p2b_fix1b", 2),
    ("p2b_fix1c", 3),
    ("p2b_fix1d", 4),
    ("p2b_fix1e", 5),  # R38B-style late pf_*_p construction inside FUSED path
    ("p2b_fix1f", 6),  # vmcnt(0) fence before emit_pf_tail (drain in-flight loads)
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
            f"{safe_tag(parent_tag)}_R43A_{cell_label}")


def build_one(n_dim, k_dim, parent_tag, base_cppflags, cell_label, fix_value):
    module_name = module_name_for(n_dim, k_dim, parent_tag, cell_label)
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(BUILD_DIR, out_file)
    key = f"N={n_dim},K={k_dim},cell={cell_label}"

    if os.path.exists(out_path):
        return (key, "cached", 0.0, module_name, out_path)

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
    fused_macro = "-DFUSED_STEP34=1 "
    fix_macro = f"-DR43A_GATE_PF_TAIL_KBOUND={fix_value} " if fix_value else ""
    extra_cppflags = (fix_macro + fused_macro + extra_cppflags).strip()

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
        return (key, "FAILED", elapsed, module_name, out_path, result.stderr[-1500:])
    return (key, "ok", elapsed, module_name, out_path)


def main():
    args = sys.argv[1:]
    max_workers = int(args[0]) if args else 4

    os.makedirs(BUILD_DIR, exist_ok=True)
    variants = parse_variants_from_bench()
    suffix_to_flags = {}
    for sfx, flags in variants:
        tag = sfx.lstrip("_") if sfx else "default"
        suffix_to_flags[tag] = flags

    if PARENT_VARIANT not in suffix_to_flags:
        print(f"ERROR: parent variant {PARENT_VARIANT} not found in bench_all_42.py")
        sys.exit(1)
    base_flags = suffix_to_flags[PARENT_VARIANT]
    print(f"Parent variant: {PARENT_VARIANT}")
    print(f"Base flags: {base_flags}")

    plan = []
    for (m, n, k, comp) in SHAPES:
        for (cell, fix_value) in CELLS:
            plan.append((m, n, k, PARENT_VARIANT, base_flags, cell, fix_value))

    unique = {}
    for (m, n, k, parent_tag, flags, cell, fix_value) in plan:
        unique[(n, k, cell)] = (n, k, parent_tag, flags, cell, fix_value)
    unique_tasks = list(unique.values())

    print(f"R43A builder: {len(plan)} entries, {len(unique_tasks)} unique builds")
    print(f"  Cells: {[c[0] for c in CELLS]}")

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
                err = res[5] if len(res) > 5 else ""
                failures.append((key, err))
                print(f"  [{completed:>3}/{len(unique_tasks)}] FAILED {key}", flush=True)

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s: built={built}, cached={cached}, failed={failed}")
    if failures:
        for key, err in failures[:10]:
            print(f"  FAILED {key}")
            for line in err.strip().split("\n")[-12:]:
                print(f"    {line}")

    shape_cell_modules = {}
    for (m, n, k, parent_tag, flags, cell, fix_value) in plan:
        shape_key = f"{m}x{n}x{k}"
        mod = module_name_for(n, k, parent_tag, cell)
        so = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
        shape_cell_modules.setdefault(shape_key, {})[cell] = {
            "module": mod,
            "so_path": so,
            "macros": f"FUSED_STEP34=1, R43A_GATE_PF_TAIL_KBOUND={fix_value}",
        }

    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round": "R43A",
        "parent_variant": PARENT_VARIANT,
        "shapes": [list(s) for s in SHAPES],
        "cells": [c[0] for c in CELLS],
        "shapes_to_modules_per_cell": shape_cell_modules,
        "built": built, "cached": cached, "failed": failed,
    }
    out_manifest = "R43A_BUILD_MANIFEST.json"
    with open(os.path.join(SCRIPT_DIR, out_manifest), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {out_manifest}")


if __name__ == "__main__":
    main()
