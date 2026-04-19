#!/usr/bin/env python3
"""R40 Opt B: build BEST_VARIANTS_v3 with FUSED_STEP34=1 + R25C_TAIL_PF_OFF_ITERS=0.

Hypothesis: R35 already proved FUSED_STEP34=1 produces 0.06% non-finite vs
6-8% on the non-fused (R37_FIX_B) path. R37_FIX_B was supposed to backport
FUSED_STEP34's clean step34 emission, but R39B shows it is INCOMPLETE on
36/42 shapes (cluster B + C upper-left 128x128 corruption persists).

This harness ENABLES the original FUSED_STEP34=1 path and also drops
R25C_TAIL_PF_OFF_ITERS (which made R36 mechanical-append CRASH).

Build-flag fork only — NO kernel source edits. For every entry in
R38_BEST_VARIANTS_v3.BEST_VARIANTS_V3:
  1. Append `-DFUSED_STEP34=1`
  2. Append `-DR25C_TAIL_PF_OFF_ITERS=0`
  3. Strip `-mllvm -amdgpu-sched-strategy=max-memory-clause` (memc — breaks fused step34)
  4. Strip `_memc` from variant tag for naming, append `_R40B_safe` suffix
  5. Drop per-shape macro overrides (R38B_TAIL_FIX) — they only apply to non-FUSED path

Module name suffix: `_R40B_safe` (distinct .so files vs _R38E and _R37 builds).
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
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R40B")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")),
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
SCHED_BAD = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

# Forced macros for R40B safe path
R40B_MACROS = {
    "FUSED_STEP34": 1,
    "R25C_TAIL_PF_OFF_ITERS": 0,
}

sys.path.insert(0, SCRIPT_DIR)
from R38_BEST_VARIANTS_v3 import BEST_VARIANTS_V3


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
    """Strip `_memc` tokens from the variant tag for module naming."""
    # remove _memc occurrences (token separated by '_' delimiters)
    out = re.sub(r"(?<![A-Za-z0-9])memc(?![A-Za-z0-9])", "", parent_tag)
    out = re.sub(r"_memc(?=_|$)", "", out)
    # collapse double underscores
    out = re.sub(r"__+", "_", out)
    out = out.strip("_")
    return out


def macro_flags(macro_overrides):
    return " ".join(f"-D{k}={v}" for k, v in sorted(macro_overrides.items()))


def module_name_for(n_dim, k_dim, parent_tag):
    return f"tk_mxfp4_gluon_cpp_n{n_dim}_k{k_dim}_{safe_tag(parent_tag)}_R40B_safe"


def build_one(n_dim, k_dim, parent_tag, base_cppflags):
    module_name = module_name_for(n_dim, k_dim, parent_tag)
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(BUILD_DIR, out_file)
    key = f"N={n_dim},K={k_dim},{parent_tag}->{safe_tag(parent_tag)}_R40B_safe"

    if os.path.exists(out_path):
        return (key, "cached", 0.0)

    with open(KERNEL_SRC) as f:
        kernel_src_text = f.read()
    patched = kernel_src_text.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},",
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n_dim}_k{k_dim}_{safe_tag(parent_tag)}_R40B.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    extra_cppflags = strip_bad_sched(base_cppflags)
    extra_macros = macro_flags(R40B_MACROS)
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
        return (key, "FAILED", elapsed, result.stderr[-1500:])

    return (key, "ok", elapsed)


# Subset of shapes for smoke test
SMOKE_SHAPES = {
    (4096, 4096, 8192),
    (16384, 4096, 14336),
    (32768, 6144, 2048),
    (4096, 28672, 32768),
    (4096, 4096, 16384),
}


def main():
    smoke = "--smoke" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--smoke"]
    max_workers = int(args[0]) if args else 16

    os.makedirs(BUILD_DIR, exist_ok=True)
    variants = parse_variants_from_bench()
    suffix_to_flags = {}
    for sfx, flags in variants:
        tag = sfx.lstrip("_") if sfx else "default"
        suffix_to_flags[tag] = flags

    plan = []  # (m,n,k,parent_tag,flags)
    for (m, n, k), (parent_tag, _macro_overrides) in BEST_VARIANTS_V3.items():
        if smoke and (m, n, k) not in SMOKE_SHAPES:
            continue
        if parent_tag not in suffix_to_flags:
            print(f"  ERROR: variant {parent_tag} not in bench_all_42.py")
            sys.exit(1)
        flags = suffix_to_flags[parent_tag]
        plan.append((m, n, k, parent_tag, flags))

    # Unique builds keyed on (n,k,safe_tag)
    unique = {}
    for (m, n, k, parent_tag, flags) in plan:
        unique[(n, k, safe_tag(parent_tag))] = (n, k, parent_tag, flags)
    unique_tasks = list(unique.values())

    mode = "SMOKE" if smoke else "FULL"
    print(f"R40B builder [{mode}]: {len(plan)} shape entries, {len(unique_tasks)} unique builds")
    print(f"  Forced macros: {R40B_MACROS}")
    print(f"  memc sched-strategy stripped from CPPFLAGS")
    print(f"  module suffix: _R40B_safe (vs base _memc tag)")

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
                err = res[3] if len(res) > 3 else ""
                failures.append((key, err))
                print(f"  [{completed:>3}/{len(unique_tasks)}] FAILED {key}", flush=True)

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s: built={built}, cached={cached}, failed={failed}")
    if failures:
        print("\nFailures:")
        for key, err in failures[:10]:
            print(f"  {key}")
            for line in err.strip().split("\n")[-5:]:
                print(f"    {line}")

    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round": "R40B",
        "mode": mode,
        "forced_macros": R40B_MACROS,
        "shapes_to_module": {
            f"{m}x{n}x{k}": module_name_for(n, k, parent_tag)
            for (m, n, k, parent_tag, _) in plan
        },
        "built": built,
        "cached": cached,
        "failed": failed,
    }
    out_manifest = "R40B_BUILD_MANIFEST_SMOKE.json" if smoke else "R40B_BUILD_MANIFEST.json"
    with open(os.path.join(SCRIPT_DIR, out_manifest), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {out_manifest}")


if __name__ == "__main__":
    main()
