#!/usr/bin/env python3
"""R41 Opt B: Cluster-B near-gate per-shape variant retune.

For each of 11 cluster-B shapes (wcf 0.3-8% under R40B baseline), build up to 4
variant retunes on top of the FUSED_STEP34=1, R25C_TAIL_PF_OFF_ITERS overridden
base (no -mllvm max-memory-clause):

  v0 = pfoff retune: R25C_TAIL_PF_OFF_ITERS in {0, k_iters/8, k_iters/4, k_iters/2}.
       v0 itself is implemented as a 4-cell sub-sweep (v0a, v0b, v0c, v0d).
  v1 = drop _btw_all (no BARRIER_TO_WAITCNT_ALL) for shapes that have it.
  v2 = swap _v12 -> _v32 (STEP3_BARRIER_VMCNT 12 -> 32) for shapes that have v12.
  v3 = swap _lgk2 -> _lgk1 (STEP12_BR_LGKMCNT 2 -> 1) for shapes that have lgk2.

NO kernel source edits. Build-flag fork only. Module suffix: _R41B_<variant_id>.
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
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R41B")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")),
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
SCHED_BAD = "-mllvm -amdgpu-sched-strategy=max-memory-clause"

# Forced macros for R41B safe path (same as R40B)
R41B_BASE_MACROS = {
    "FUSED_STEP34": 1,
}

# 11 cluster-B near-gate shapes (parent variant tag from BEST_VARIANTS_V3 / R40B)
# (M, N, K) -> (parent_variant_tag, current_wcf, has_btw_all, has_v12, has_lgk2)
CLUSTER_B = {
    (16384, 28672, 2048):  ("ts_lgk2_gm6_v12_memc_pfoff4",                0.0029, False, True, True),
    (32768, 28672, 2048):  ("ts_lgk2_gm6_v12_memc_pfoff4",                0.0060, False, True, True),
    (4096,  32768, 6144):  ("ts_v12_gm7_memc_pfoff19_kx6144_btw_all",     0.0172, True,  True, False),
    (4096,  32768, 14336): ("ts_v12_tv0_memc_btw_all",                    0.0214, True,  True, False),
    (4096,  32768,128256): ("ts_lgk2_v12_memc_btw_all",                   0.0808, True,  True, True),
    (14336, 32768, 4096):  ("ts_lgk2_gm7_v12_memc_pfoff14",               0.0056, False, True, True),
    (16384, 14336, 4096):  ("ts_gm7_v12_memc_dc_pfoff14",                 0.0374, False, True, False),
    (16384, 4096, 14336):  ("ts_gm8_v12_btw_all",                         0.0115, True,  True, False),
    (28672, 4096,  8192):  ("ts_v12_tv0_gm7_memc_pfoff28_kx8192_btw_all", 0.0039, True,  True, False),
    (28672, 4096, 16384):  ("ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",   0.0090, True,  False, True),
    (32768, 4096,  2048):  ("ts_lgk2_gm6_v12_memc_pfoff4",                0.0080, False, True, True),  # newly lost in R40B 5-run
}


def parse_variants_from_bench():
    """Parse the (suffix, flags) variant table from bench_all_42.py."""
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
    return re.sub(r"\s*-mllvm\s+-amdgpu-sched-strategy=max-memory-clause\b", "", flags).strip()


def safe_tag(parent_tag):
    out = re.sub(r"(?<![A-Za-z0-9])memc(?![A-Za-z0-9])", "", parent_tag)
    out = re.sub(r"_memc(?=_|$)", "", out)
    out = re.sub(r"__+", "_", out)
    return out.strip("_")


def macro_flags(macro_overrides):
    return " ".join(f"-D{k}={v}" for k, v in sorted(macro_overrides.items()))


def derive_variant_flags(parent_flags, variant_id, k_dim):
    """Derive (cppflags, extra_macros, valid) for a given variant_id.

    variant_id is a string like 'v0a', 'v0b', 'v0c', 'v0d', 'v1', 'v2', 'v3'.
    Returns (cppflags_string, extra_macros_dict, valid_bool, description).
    """
    flags = strip_bad_sched(parent_flags)
    # k_iters: compute from K_DIM. Each K-iter = 256 bytes-worth.
    # For matched-variants-by-K (kx2048, kx6144, kx14336, kx16384, etc),
    # k_iters = K // 256.
    k_iters = k_dim // 256

    extra_macros = dict(R41B_BASE_MACROS)

    if variant_id == "v0a":
        # pfoff = 0 (no tail-pf-off, full prefetch through all iters)
        extra_macros["R25C_TAIL_PF_OFF_ITERS"] = 0
        # Strip any existing R25C_TAIL_PF_OFF_ITERS / R25C_K_EXACT in parent flags
        # (we'll let our macro definition override; but parent may also set R25C_K_EXACT
        # which gates the override on a specific K — for k_dim that matches it, this is fine)
        return (flags, extra_macros, True, f"pfoff=0 (full PF)")
    if variant_id == "v0b":
        po = max(1, k_iters // 8)
        extra_macros["R25C_TAIL_PF_OFF_ITERS"] = po
        extra_macros["R25C_K_LIMIT"] = 32768
        extra_macros["R25C_K_EXACT"] = k_dim
        return (flags, extra_macros, True, f"pfoff=k/8={po}")
    if variant_id == "v0c":
        po = max(1, k_iters // 4)
        extra_macros["R25C_TAIL_PF_OFF_ITERS"] = po
        extra_macros["R25C_K_LIMIT"] = 32768
        extra_macros["R25C_K_EXACT"] = k_dim
        return (flags, extra_macros, True, f"pfoff=k/4={po}")
    if variant_id == "v0d":
        po = max(1, k_iters // 2)
        extra_macros["R25C_TAIL_PF_OFF_ITERS"] = po
        extra_macros["R25C_K_LIMIT"] = 32768
        extra_macros["R25C_K_EXACT"] = k_dim
        return (flags, extra_macros, True, f"pfoff=k/2={po}")
    if variant_id == "v1":
        # Drop BARRIER_TO_WAITCNT_ALL=1 (the _btw_all mechanism)
        if "-DBARRIER_TO_WAITCNT_ALL=1" not in flags:
            return (flags, extra_macros, False, "no _btw_all to drop")
        new_flags = flags.replace("-DBARRIER_TO_WAITCNT_ALL=1", "").strip()
        new_flags = re.sub(r"\s+", " ", new_flags)
        return (new_flags, extra_macros, True, "drop _btw_all")
    if variant_id == "v2":
        # Swap STEP3_BARRIER_VMCNT=12 -> =32
        if "-DSTEP3_BARRIER_VMCNT=12" not in flags:
            return (flags, extra_macros, False, "no v12 to swap")
        new_flags = flags.replace("-DSTEP3_BARRIER_VMCNT=12", "-DSTEP3_BARRIER_VMCNT=32")
        return (new_flags, extra_macros, True, "v12->v32")
    if variant_id == "v3":
        # Swap STEP12_BR_LGKMCNT=2 -> =1
        if "-DSTEP12_BR_LGKMCNT=2" not in flags:
            return (flags, extra_macros, False, "no lgk2 to swap")
        new_flags = flags.replace("-DSTEP12_BR_LGKMCNT=2", "-DSTEP12_BR_LGKMCNT=1")
        return (new_flags, extra_macros, True, "lgk2->lgk1")
    raise ValueError(f"unknown variant {variant_id}")


def module_name_for(n_dim, k_dim, parent_tag, variant_id):
    base = safe_tag(parent_tag)
    return f"tk_mxfp4_gluon_cpp_n{n_dim}_k{k_dim}_{base}_R41B_{variant_id}"


def build_one(m_dim, n_dim, k_dim, parent_tag, parent_flags, variant_id):
    """Build a single (n,k,parent_tag,variant_id). Returns (key, status, ...)."""
    cppflags, extra_macros, valid, desc = derive_variant_flags(parent_flags, variant_id, k_dim)
    if not valid:
        return (f"N={n_dim},K={k_dim},{parent_tag},{variant_id}", "skipped", 0.0, desc)

    module_name = module_name_for(n_dim, k_dim, parent_tag, variant_id)
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(BUILD_DIR, out_file)
    key = f"N={n_dim},K={k_dim},{parent_tag},{variant_id}({desc})"

    if os.path.exists(out_path):
        return (key, "cached", 0.0)

    with open(KERNEL_SRC) as f:
        kernel_src_text = f.read()
    patched = kernel_src_text.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},",
    )
    wrapper_src = os.path.join(
        BUILD_DIR, f"wrap_n{n_dim}_k{k_dim}_{safe_tag(parent_tag)}_R41B_{variant_id}.cpp"
    )
    with open(wrapper_src, "w") as f:
        f.write(patched)

    extra_macro_str = macro_flags(extra_macros)
    full_cppflags = (extra_macro_str + " " + cppflags).strip()

    t0 = time.time()
    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT
    cmd = (
        f'make -C {SCRIPT_DIR} TARGET={os.path.join(BUILD_DIR, module_name)} '
        f'SRC={wrapper_src} '
        f'CPPFLAGS="-DK_DIM={k_dim} -DN_DIM={n_dim} {full_cppflags}"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0 or not os.path.exists(out_path):
        return (key, "FAILED", elapsed, result.stderr[-1500:])

    return (key, "ok", elapsed)


SMOKE_SHAPES = {
    (16384, 28672, 2048),
    (4096, 32768, 128256),
    (28672, 4096, 8192),
}

VARIANT_IDS = ["v0a", "v0b", "v0c", "v0d", "v1", "v2", "v3"]


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

    plan = []  # (m,n,k,parent_tag,parent_flags,variant_id)
    for (m, n, k), (parent_tag, _wcf, _has_btw, _has_v12, _has_lgk2) in CLUSTER_B.items():
        if smoke and (m, n, k) not in SMOKE_SHAPES:
            continue
        if parent_tag not in suffix_to_flags:
            print(f"  ERROR: variant {parent_tag} not in bench_all_42.py")
            sys.exit(1)
        flags = suffix_to_flags[parent_tag]
        for vid in VARIANT_IDS:
            plan.append((m, n, k, parent_tag, flags, vid))

    mode = "SMOKE" if smoke else "FULL"
    print(f"R41B builder [{mode}]: {len(plan)} build attempts")
    print(f"  Forced base macros: {R41B_BASE_MACROS}")
    print(f"  memc sched-strategy stripped from CPPFLAGS")
    print(f"  Variants per shape: {VARIANT_IDS}")

    t_start = time.time()
    completed = built = cached = failed = skipped = 0
    failures = []
    skip_reasons = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(build_one, *t): t for t in plan}
        for fut in as_completed(futs):
            res = fut.result()
            completed += 1
            key, status = res[0], res[1]
            if status == "cached":
                cached += 1
                print(f"  [{completed:>3}/{len(plan)}] CACHED {key}", flush=True)
            elif status == "ok":
                built += 1
                el = res[2]
                print(f"  [{completed:>3}/{len(plan)}] BUILT {key} ({el:.1f}s)", flush=True)
            elif status == "skipped":
                skipped += 1
                desc = res[3] if len(res) > 3 else ""
                skip_reasons.append((key, desc))
                print(f"  [{completed:>3}/{len(plan)}] SKIP {key}: {desc}", flush=True)
            else:
                failed += 1
                err = res[3] if len(res) > 3 else ""
                failures.append((key, err))
                print(f"  [{completed:>3}/{len(plan)}] FAILED {key}", flush=True)

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s: built={built}, cached={cached}, skipped={skipped}, failed={failed}")
    if failures:
        print("\nFailures:")
        for key, err in failures[:10]:
            print(f"  {key}")
            for line in err.strip().split("\n")[-5:]:
                print(f"    {line}")

    # Build per-shape per-variant manifest: shape -> {variant_id: module_name or skip_reason}
    shape_to_variants = {}
    for (m, n, k, parent_tag, parent_flags, vid) in plan:
        shape_key = f"{m}x{n}x{k}"
        if shape_key not in shape_to_variants:
            shape_to_variants[shape_key] = {"parent_variant": parent_tag, "variants": {}}
        cppflags, extra_macros, valid, desc = derive_variant_flags(parent_flags, vid, k)
        if not valid:
            shape_to_variants[shape_key]["variants"][vid] = {"status": "skipped", "reason": desc}
        else:
            mod = module_name_for(n, k, parent_tag, vid)
            so_path = os.path.join(BUILD_DIR, f"{mod}{EXT_SUFFIX}")
            shape_to_variants[shape_key]["variants"][vid] = {
                "status": "built" if os.path.exists(so_path) else "missing",
                "module": mod,
                "extra_macros": extra_macros,
                "description": desc,
            }

    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round": "R41B",
        "mode": mode,
        "forced_base_macros": R41B_BASE_MACROS,
        "cluster_b_shapes": [f"{m}x{n}x{k}" for (m, n, k) in CLUSTER_B.keys()],
        "variant_ids": VARIANT_IDS,
        "shape_variants": shape_to_variants,
        "built": built,
        "cached": cached,
        "skipped": skipped,
        "failed": failed,
    }
    out_manifest = "R41B_BUILD_MANIFEST_SMOKE.json" if smoke else "R41B_BUILD_MANIFEST.json"
    with open(os.path.join(SCRIPT_DIR, out_manifest), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {out_manifest}")


if __name__ == "__main__":
    main()
