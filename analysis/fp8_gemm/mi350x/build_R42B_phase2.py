#!/usr/bin/env python3
"""R42 Opt B Phase 2: localize the K=28672 aperture violation by knob sweep.

Phase 1 confirmed: R41A fence + tail-pf-off=0 don't fix CRASH.
The CRASH is HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (a true OOB read).

Phase 2 sweeps:
  cell 'baseline'      = R40B parent variant w/ FUSED_STEP34=1 (control - confirms CRASH)
  cell 'no_fused'      = parent flags WITHOUT FUSED_STEP34 (does FUSED_STEP34 cause it?)
  cell 'no_btw'        = remove BARRIER_TO_WAITCNT_ALL=1 (R23-G barrier rewrite)
  cell 'gm1'           = override GROUP_SIZE_M=1 (eliminate partial-group swizzle)
  cell 'no_lgk2'       = remove STEP12_BR_LGKMCNT=2
  cell 'no_tailsplit'  = remove TAIL_SPLIT=1
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
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R42B_phase2")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")),
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

SHAPES = [
    (4096,  32768, 28672),
    (16384, 4096,  28672),
]

PARENT_VARIANT = "ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all"

# Each cell: (label, transform_fn(base_flags) -> str)
def t_baseline(f):
    return f
def t_no_fused(f):
    return f.replace("-DFUSED_STEP34=1", "")  # already not in base; cell name reflects no add
def t_no_btw(f):
    return f.replace("-DBARRIER_TO_WAITCNT_ALL=1", "")
def t_gm1(f):
    f = re.sub(r"-DGROUP_SIZE_M=\d+", "-DGROUP_SIZE_M=1", f)
    return f
def t_no_lgk2(f):
    return f.replace("-DSTEP12_BR_LGKMCNT=2", "")
def t_no_tailsplit(f):
    return f.replace("-DTAIL_SPLIT=1", "")
def t_no_r25c(f):
    f = re.sub(r"-DR25C_TAIL_PF_OFF_ITERS=\d+", "-DR25C_TAIL_PF_OFF_ITERS=0", f)
    return f

# Keep FUSED_STEP34=1 for all cells EXCEPT 'no_fused', because FUSED_STEP34
# was forced by R40B integration as part of the working pipeline.
CELLS = [
    ("baseline",     t_baseline,    True),
    ("no_fused",     t_no_fused,    False),
    ("no_btw",       t_no_btw,      True),
    ("gm1",          t_gm1,         True),
    ("no_lgk2",      t_no_lgk2,     True),
    ("no_tailsplit", t_no_tailsplit, True),
    ("no_r25c",      t_no_r25c,     True),
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
            f"{safe_tag(parent_tag)}_R42Bp2_{cell_label}")


def build_one(n_dim, k_dim, parent_tag, base_cppflags, cell_label, transform, with_fused):
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

    extra_cppflags = transform(strip_bad_sched(base_cppflags))
    fused_macro = "-DFUSED_STEP34=1 " if with_fused else ""
    extra_cppflags = (fused_macro + extra_cppflags).strip()

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
        sys.exit(1)
    base_flags = suffix_to_flags[PARENT_VARIANT]
    print(f"Parent variant: {PARENT_VARIANT}")
    print(f"Base flags: {base_flags}")

    plan = []
    for (m, n, k) in SHAPES:
        for (cell, transform, fused) in CELLS:
            plan.append((m, n, k, PARENT_VARIANT, base_flags, cell, transform, fused))

    unique = {}
    for (m, n, k, parent_tag, flags, cell, transform, fused) in plan:
        unique[(n, k, cell)] = (n, k, parent_tag, flags, cell, transform, fused)
    unique_tasks = list(unique.values())

    print(f"R42B Phase2 builder: {len(plan)} entries, {len(unique_tasks)} unique builds")
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
                err = res[4] if len(res) > 4 else ""
                failures.append((key, err))
                print(f"  [{completed:>3}/{len(unique_tasks)}] FAILED {key}", flush=True)

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s: built={built}, cached={cached}, failed={failed}")
    if failures:
        for key, err in failures[:10]:
            print(f"  FAILED {key}")
            for line in err.strip().split("\n")[-8:]:
                print(f"    {line}")

    shape_cell_modules = {}
    for (m, n, k, parent_tag, flags, cell, transform, fused) in plan:
        shape_key = f"{m}x{n}x{k}"
        shape_cell_modules.setdefault(shape_key, {})[cell] = (
            module_name_for(n, k, parent_tag, cell)
        )

    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round": "R42B_phase2",
        "parent_variant": PARENT_VARIANT,
        "shapes": [list(s) for s in SHAPES],
        "cells": [c[0] for c in CELLS],
        "shapes_to_modules_per_cell": shape_cell_modules,
        "built": built, "cached": cached, "failed": failed,
    }
    out_manifest = "R42B_PHASE2_BUILD_MANIFEST.json"
    with open(os.path.join(SCRIPT_DIR, out_manifest), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {out_manifest}")


if __name__ == "__main__":
    main()
