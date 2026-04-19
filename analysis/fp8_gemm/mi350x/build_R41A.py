#!/usr/bin/env python3
"""R41 Opt A: Cluster C deep-K tail-pf-off SWEEP + extract_tile fence builder.

Targets the 5 catastrophic K=32768 shapes that under R40B are ~97% wrong cells
with ~10% finite. Hypothesis: R25C_TAIL_PF_OFF_ITERS=120 with K_DIM=32768
(k_byte_iters=128) means PF runs only on first 8 iters then the kernel rides on
extract_tile-staged registers for >100 iters; combined with FUSED_STEP34=1's
fewer iter-boundary fences, tile-register liveness across deep K is the
corruption vector.

Sweep: R41A_PFOFF_OVERRIDE in {0, 8, 16, 32, 64} x R41A_EXTRACT_TILE_FENCE in
{0, 1} = 10 cells per shape, 5 shapes = 50 builds.

Per-cell macro stack (over R40B base):
  -DFUSED_STEP34=1
  -DR25C_TAIL_PF_OFF_ITERS=<base>     (kept from parent variant; overridden inside kernel)
  -DR41A_DEEP_K_FIX=1
  -DR41A_PFOFF_OVERRIDE=<po>          (kernel #undef/#define R25C if po!=0 AND K_DIM>=16384)
  -DR41A_EXTRACT_TILE_FENCE=<ef>
  (memc sched-strategy stripped per R40B convention)

Module suffix: _R41A_po<N>_ef<0|1>
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
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R41A")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")),
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

# 5 catastrophic K=32768 cluster-C shapes
CATASTROPHIC_SHAPES = [
    (4096,   4096,  32768),
    (4096,   6144,  32768),
    (4096,  28672,  32768),
    (4096, 128256,  32768),
    (14336,  4096,  32768),
]

# All 5 shapes use the same parent variant (per R40B BEST_VARIANTS)
PARENT_VARIANT = "ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all"

# R41A sweep grid
PFOFF_OVERRIDES = [0, 8, 16, 32, 64]
EXTRACT_FENCES = [0, 1]

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


def module_name_for(n_dim, k_dim, parent_tag, po, ef):
    return (f"tk_mxfp4_gluon_cpp_n{n_dim}_k{k_dim}_"
            f"{safe_tag(parent_tag)}_R41A_po{po}_ef{ef}")


def build_one(n_dim, k_dim, parent_tag, base_cppflags, po, ef):
    module_name = module_name_for(n_dim, k_dim, parent_tag, po, ef)
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(BUILD_DIR, out_file)
    key = f"N={n_dim},K={k_dim},po={po},ef={ef}"

    if os.path.exists(out_path):
        return (key, "cached", 0.0, module_name)

    with open(KERNEL_SRC) as f:
        kernel_src_text = f.read()
    patched = kernel_src_text.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},",
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n_dim}_k{k_dim}_po{po}_ef{ef}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    extra_cppflags = strip_bad_sched(base_cppflags)
    # R41A macro stack: FUSED_STEP34=1 base + R41A gate + sweep params
    extra_macros = (
        f"-DFUSED_STEP34=1 "
        f"-DR41A_DEEP_K_FIX=1 "
        f"-DR41A_PFOFF_OVERRIDE={po} "
        f"-DR41A_EXTRACT_TILE_FENCE={ef}"
    )
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
    max_workers = int(args[0]) if args else 16

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

    plan = []
    for (m, n, k) in CATASTROPHIC_SHAPES:
        for po in PFOFF_OVERRIDES:
            for ef in EXTRACT_FENCES:
                plan.append((m, n, k, PARENT_VARIANT, base_flags, po, ef))

    # Unique builds keyed on (n,k,po,ef) — different M same (n,k) can share .so
    unique = {}
    for (m, n, k, parent_tag, flags, po, ef) in plan:
        unique[(n, k, po, ef)] = (n, k, parent_tag, flags, po, ef)
    unique_tasks = list(unique.values())

    print(f"R41A builder: {len(plan)} (M,N,K,po,ef) entries, "
          f"{len(unique_tasks)} unique builds")
    print(f"  Parent variant: {PARENT_VARIANT}")
    print(f"  Sweep: po in {PFOFF_OVERRIDES} x ef in {EXTRACT_FENCES}")
    print(f"  Forced macros: FUSED_STEP34=1, R41A_DEEP_K_FIX=1")
    print(f"  memc sched-strategy stripped from CPPFLAGS")

    t_start = time.time()
    completed = built = cached = failed = 0
    failures = []
    shapes_to_module = {}
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
                print(f"  [{completed:>3}/{len(unique_tasks)}] BUILT {key} ({el:.1f}s)",
                      flush=True)
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

    # Build per-shape per-cell module map
    shape_cell_modules = {}
    for (m, n, k, parent_tag, flags, po, ef) in plan:
        cell = f"po{po}_ef{ef}"
        shape_key = f"{m}x{n}x{k}"
        shape_cell_modules.setdefault(shape_key, {})[cell] = (
            module_name_for(n, k, parent_tag, po, ef)
        )

    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round": "R41A",
        "parent_variant": PARENT_VARIANT,
        "sweep_pfoff": PFOFF_OVERRIDES,
        "sweep_ef": EXTRACT_FENCES,
        "forced_macros": {
            "FUSED_STEP34": 1,
            "R41A_DEEP_K_FIX": 1,
        },
        "shapes_to_modules_per_cell": shape_cell_modules,
        "built": built,
        "cached": cached,
        "failed": failed,
    }
    out_manifest = "R41A_BUILD_MANIFEST.json"
    with open(os.path.join(SCRIPT_DIR, out_manifest), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {out_manifest}")


if __name__ == "__main__":
    main()
