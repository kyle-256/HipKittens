#!/usr/bin/env python3
"""R44 Opt A Phase 2 — combine wcf-clean variants (R38B_pfoff{8,32}) with
R38F_TAIL_DRAIN to force in-flight loads to land before next iter reads.

Hypothesis: shape B has wcf=0 with R38B_pfoff{8,32} but fin floor ~0.96.
The remaining ~3-4% inf cells are caused by in-flight buffer_load_to_lds
loads that haven't landed when the next iter's MFMA reads from that LDS
slot. R38F variants 1/2/4 add explicit s_waitcnt vmcnt(0) [+ s_barrier]
on the tail iters.

Cells:
  R38B_pfoff8_R38F1   = R25C=8  + R38B + R38F (drain only, no barrier)
  R38B_pfoff8_R38F2   = R25C=8  + R38B + R38F (drain + barrier)
  R38B_pfoff8_R38F4   = R25C=8  + R38B + R38F (drain GMEM + LDS)
  R38B_pfoff32_R38F1  = R25C=32 + R38B + R38F1
  R38B_pfoff32_R38F2  = R25C=32 + R38B + R38F2
  R38B_pfoff32_R38F4  = R25C=32 + R38B + R38F4
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
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R44A")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")),
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

SHAPES = [
    (4096,  32768, 28672),
    (16384, 4096,  28672),
]

BASE_FLAGS = (
    "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DGROUP_SIZE_M=7 "
    "-DR25C_K_LIMIT=32768 -DR25C_K_EXACT=28672 "
    "-DBARRIER_TO_WAITCNT_ALL=1"
)

CELLS = [
    ("R38B_pfoff8_R38F1",
     "-DR25C_TAIL_PF_OFF_ITERS=8 -DR38B_TAIL_FIX=1 -DR38F_TAIL_DRAIN=1 -DR38F_VARIANT=1"),
    ("R38B_pfoff8_R38F2",
     "-DR25C_TAIL_PF_OFF_ITERS=8 -DR38B_TAIL_FIX=1 -DR38F_TAIL_DRAIN=1 -DR38F_VARIANT=2"),
    ("R38B_pfoff8_R38F4",
     "-DR25C_TAIL_PF_OFF_ITERS=8 -DR38B_TAIL_FIX=1 -DR38F_TAIL_DRAIN=1 -DR38F_VARIANT=4"),
    ("R38B_pfoff32_R38F1",
     "-DR25C_TAIL_PF_OFF_ITERS=32 -DR38B_TAIL_FIX=1 -DR38F_TAIL_DRAIN=1 -DR38F_VARIANT=1"),
    ("R38B_pfoff32_R38F2",
     "-DR25C_TAIL_PF_OFF_ITERS=32 -DR38B_TAIL_FIX=1 -DR38F_TAIL_DRAIN=1 -DR38F_VARIANT=2"),
    ("R38B_pfoff32_R38F4",
     "-DR25C_TAIL_PF_OFF_ITERS=32 -DR38B_TAIL_FIX=1 -DR38F_TAIL_DRAIN=1 -DR38F_VARIANT=4"),
    # Also try R38B + R38F (no R25C clamp at all) — drain forced everywhere
    ("R38B_no_r25c_R38F2",
     "-DR25C_TAIL_PF_OFF_ITERS=0 -DR38B_TAIL_FIX=1 -DR38F_TAIL_DRAIN=1 -DR38F_VARIANT=2"),
    ("R38B_no_r25c_R38F4",
     "-DR25C_TAIL_PF_OFF_ITERS=0 -DR38B_TAIL_FIX=1 -DR38F_TAIL_DRAIN=1 -DR38F_VARIANT=4"),
]


def safe_tag():
    return "ts_lgk2_gm7_kx28672_btw_all"


def module_name_for(n_dim, k_dim, cell_label):
    return (f"tk_mxfp4_gluon_cpp_n{n_dim}_k{k_dim}_"
            f"{safe_tag()}_R44A_{cell_label}")


def build_one(n_dim, k_dim, cell_label, extra_cell_macros):
    module_name = module_name_for(n_dim, k_dim, cell_label)
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

    extra_cppflags = (extra_cell_macros + " " + BASE_FLAGS).strip()

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
    print(f"R44 Opt A Phase 2 builder: {len(CELLS)} cells x {len(SHAPES)} shapes")

    tasks = []
    for (m, n, k) in SHAPES:
        for (cell, extras) in CELLS:
            tasks.append((n, k, cell, extras))
    unique = {}
    for (n, k, cell, extras) in tasks:
        unique[(n, k, cell)] = (n, k, cell, extras)
    unique_tasks = list(unique.values())

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

    # Update manifest by appending
    manifest_path = os.path.join(SCRIPT_DIR, "R44A_BUILD_MANIFEST.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"shapes_to_modules_per_cell": {}, "cells": []}

    for (m, n, k) in SHAPES:
        for (cell, _) in CELLS:
            shape_key = f"{m}x{n}x{k}"
            manifest["shapes_to_modules_per_cell"].setdefault(shape_key, {})[cell] = (
                module_name_for(n, k, cell)
            )
    existing_labels = {c["label"] for c in manifest.get("cells", [])}
    for (cell, extras) in CELLS:
        if cell not in existing_labels:
            manifest.setdefault("cells", []).append({"label": cell, "extras": extras})
    manifest["last_phase"] = "R44A_phase2"
    manifest["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
