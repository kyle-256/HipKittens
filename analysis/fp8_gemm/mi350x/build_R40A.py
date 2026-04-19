#!/usr/bin/env python3
"""R40 Opt A: Build BEST_VARIANTS_v3 with R40A_PF_FENCE=1 layered on top.

R40A insertions in kernel_mxfp4_gluon_cpp.cpp (R37_FIX_B && !FUSED_STEP34 path):
  1) `asm volatile("" ::: "memory")` BEFORE every kpair_64mfma_step12 call.
  2) `make_pf_params(...)` block deferred from TOP-of-iter to AFTER kpair_64mfma_step34.

Hypothesis: prevents the compiler from hoisting buffer_load_to_lds prefetches
across the step12 boundary, where they can land in an LDS slot the in-flight
MFMA is reading. Targets cluster-A "borderline" 24-50 dB shapes.

Same harness as build_R38E.py: layer R40A on top of each shape's existing
macro_overrides dict; strip `-mllvm -amdgpu-sched-strategy=max-memory-clause`
(documented enemy of the fused step34 path).

Module names use a `_R40A` suffix.
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
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R40A")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")),
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

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


def macro_signature(macro_overrides):
    if not macro_overrides:
        return ""
    parts = [f"{k}{v}" for k, v in sorted(macro_overrides.items())]
    return "_" + "_".join(parts)


def macro_flags(macro_overrides):
    return " ".join(f"-D{k}={v}" for k, v in sorted(macro_overrides.items()))


def module_name_for(n_dim, k_dim, parent_tag, macro_overrides):
    sig = macro_signature(macro_overrides)
    return f"tk_mxfp4_gluon_cpp_n{n_dim}_k{k_dim}_{parent_tag}{sig}_R40A"


def build_one(n_dim, k_dim, parent_tag, base_cppflags, macro_overrides):
    module_name = module_name_for(n_dim, k_dim, parent_tag, macro_overrides)
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(BUILD_DIR, out_file)
    sig = macro_signature(macro_overrides)
    key = f"N={n_dim},K={k_dim},{parent_tag}{sig}_R40A"

    if os.path.exists(out_path):
        return (key, "cached", 0.0)

    with open(KERNEL_SRC) as f:
        kernel_src_text = f.read()
    patched = kernel_src_text.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},",
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n_dim}_k{k_dim}_{parent_tag}{sig}_R40A.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    extra_cppflags = strip_bad_sched(base_cppflags)
    extra_macros = macro_flags(macro_overrides)
    if extra_macros:
        extra_cppflags = extra_macros + " " + extra_cppflags

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

    # Optional shape filter: --only "M,N,K;M,N,K"
    only_shapes = None
    for arg in sys.argv[1:]:
        if arg.startswith("--only="):
            only_shapes = set()
            for s in arg.split("=", 1)[1].split(";"):
                m, n, k = [int(x) for x in s.split(",")]
                only_shapes.add((m, n, k))

    os.makedirs(BUILD_DIR, exist_ok=True)
    variants = parse_variants_from_bench()
    suffix_to_flags = {}
    for sfx, flags in variants:
        tag = sfx.lstrip("_") if sfx else "default"
        suffix_to_flags[tag] = flags

    plan = []  # (m,n,k,parent_tag,macro_overrides,flags)
    for (m, n, k), (parent_tag, macro_overrides) in BEST_VARIANTS_V3.items():
        if only_shapes is not None and (m, n, k) not in only_shapes:
            continue
        if parent_tag not in suffix_to_flags:
            print(f"  ERROR: variant {parent_tag} not in bench_all_42.py")
            sys.exit(1)
        flags = suffix_to_flags[parent_tag]
        # Layer R40A_PF_FENCE=1 on top of any existing per-shape overrides.
        ovr = dict(macro_overrides)
        ovr["R40A_PF_FENCE"] = 1
        plan.append((m, n, k, parent_tag, ovr, flags))

    unique = {}
    for (m, n, k, parent_tag, macro_overrides, flags) in plan:
        sig = macro_signature(macro_overrides)
        unique[(n, k, parent_tag, sig)] = (n, k, parent_tag, flags, macro_overrides)
    unique_tasks = list(unique.values())

    print(f"R40A builder: {len(plan)} shape entries, {len(unique_tasks)} unique builds")
    print(f"  (R37_FIX_B=1 default; +R40A_PF_FENCE=1 on every shape; sched-strategy stripped)")
    if only_shapes:
        print(f"  Shape filter active: {sorted(only_shapes)}")

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
            for line in err.strip().split("\n")[-6:]:
                print(f"    {line}")

    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round": "R40A",
        "shapes_to_module": {
            f"{m}x{n}x{k}": module_name_for(n, k, parent_tag, macro_overrides)
            for (m, n, k, parent_tag, macro_overrides, _) in plan
        },
        "macro_overrides": {
            f"{m}x{n}x{k}": macro_overrides
            for (m, n, k, _, macro_overrides, _) in plan
        },
        "built": built,
        "cached": cached,
        "failed": failed,
    }
    with open(os.path.join(SCRIPT_DIR, "R40A_BUILD_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Manifest: R40A_BUILD_MANIFEST.json")


if __name__ == "__main__":
    main()
