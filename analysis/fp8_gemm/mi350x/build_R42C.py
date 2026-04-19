#!/usr/bin/env python3
"""R42 Opt C: broaden R41A extract_tile vmcnt fence beyond deep-K only.

C1 (R42C_FENCE_NO_K_GUARD=1): drop the K_DIM>=16384 guard. Fence applies
  for any K when FUSED_STEP34=1.
C2 (R42C_FENCE_ANY_PATH=1): drop both FUSED_STEP34 AND K_DIM guards.
  Fence applies everywhere extract_tile is called.

Per the R41 integration manifest, each of the 42 shapes has a parent variant
(from R40B, R41A, R41B, or R40A). For C1 we rebuild each shape with its
parent variant flags + the macro stack:
  -DFUSED_STEP34=1
  -DR41A_DEEP_K_FIX=1
  -DR41A_EXTRACT_TILE_FENCE=1
  -DR42C_FENCE_NO_K_GUARD=1
  (memc sched-strategy stripped per R40B/R41A convention)

For shapes that already used R41A (5 deep-K cluster-C), the R42C_NO_K_GUARD
is a no-op since K_DIM>=16384 already triggers the fence. We rebuild anyway
so the .so embeds the new module name and we have a clean comparison.

For the 1 shape from R40A (4096x32768x128256), R40A used the R37_FIX_B
non-fused path (FUSED_STEP34=0). R42C C1 requires FUSED_STEP34, so we
SKIP this shape and reuse the R41 integration .so (preserves baseline).

Usage:
  python3 build_R42C.py --variant c1 [max_workers]
  python3 build_R42C.py --variant c2 [max_workers]   (only R40B-source shapes)
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

MANIFEST_PATH = os.path.join(SCRIPT_DIR, "R41_INTEGRATION_MANIFEST.json")


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
    return re.sub(r"\s*-mllvm\s+-amdgpu-sched-strategy=max-memory-clause\b", "",
                  flags).strip()


def safe_tag(parent_tag):
    out = re.sub(r"(?<![A-Za-z0-9])memc(?![A-Za-z0-9])", "", parent_tag)
    out = re.sub(r"_memc(?=_|$)", "", out)
    out = re.sub(r"__+", "_", out)
    return out.strip("_")


def parent_tag_from_so(so_path):
    """Recover parent variant tag from the .so file basename.

    Examples:
      tk_mxfp4_gluon_cpp_n4096_k8192_ts_v12_tv0_gm7_pfoff28_kx8192_btw_all_R40B_safe
        -> ts_v12_tv0_gm7_pfoff28_kx8192_btw_all   (safe form; need to recover memc)
      tk_mxfp4_gluon_cpp_n4096_k32768_ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all_R41A_po*_ef*
        -> ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all
      tk_mxfp4_gluon_cpp_n4096_k14336_ts_gm8_v12_btw_all_R41B_v0b
        -> ts_gm8_v12_btw_all
    """
    base = os.path.basename(so_path).split(".")[0]
    # strip prefix
    m = re.match(r"tk_mxfp4_gluon_cpp_n\d+_k\d+_(.*)$", base)
    if not m:
        raise RuntimeError(f"can't parse parent from {base}")
    rest = m.group(1)
    # strip suffix tag
    for suffix_pat in [
        r"_R40B_safe$",
        r"_R41A_po\d+_ef[01]$",
        r"_R41B_v\w+$",
        r"_R40A_PF_FENCE\d+_R40A$",
    ]:
        new = re.sub(suffix_pat, "", rest)
        if new != rest:
            return new
    raise RuntimeError(f"can't strip suffix from {rest}")


def find_full_variant_tag(safe_form, variants_dict):
    """Given a safe-form tag (e.g. 'ts_v12_tv0_gm7_pfoff28_kx8192_btw_all'),
    find the original variant tag in variants_dict (which may include `_memc`)
    that maps to the same safe form."""
    if safe_form in variants_dict:
        return safe_form
    for tag in variants_dict:
        if safe_tag(tag) == safe_form:
            return tag
    raise RuntimeError(f"no variant matches safe form '{safe_form}'")


def module_name_for(n_dim, k_dim, parent_tag, variant_id):
    return (f"tk_mxfp4_gluon_cpp_n{n_dim}_k{k_dim}_"
            f"{safe_tag(parent_tag)}_R42C_{variant_id}")


def build_one(n_dim, k_dim, parent_tag, base_cppflags, variant_id, build_dir,
              extra_macro_flags):
    module_name = module_name_for(n_dim, k_dim, parent_tag, variant_id)
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(build_dir, out_file)
    key = f"N={n_dim},K={k_dim},{safe_tag(parent_tag)}"

    if os.path.exists(out_path):
        return (key, "cached", 0.0, module_name, out_path)

    with open(KERNEL_SRC) as f:
        kernel_src_text = f.read()
    patched = kernel_src_text.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},",
    )
    wrapper_src = os.path.join(build_dir,
                               f"wrap_n{n_dim}_k{k_dim}_{safe_tag(parent_tag)}_{variant_id}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    extra_cppflags = strip_bad_sched(base_cppflags)
    extra_cppflags = (extra_macro_flags + " " + extra_cppflags).strip()

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
        return (key, "FAILED", elapsed, module_name, "", result.stderr[-1500:])
    return (key, "ok", elapsed, module_name, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["c1", "c2"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    variant_id = args.variant
    build_dir = os.path.join(SCRIPT_DIR, f"build_R42C_{variant_id}")
    os.makedirs(build_dir, exist_ok=True)

    if variant_id == "c1":
        # Drop K guard, keep FUSED_STEP34 requirement.
        # All shapes from R40B/R41A/R41B used FUSED_STEP34=1; only R40A did not.
        extra_macro_flags = (
            "-DFUSED_STEP34=1 "
            "-DR41A_DEEP_K_FIX=1 "
            "-DR41A_EXTRACT_TILE_FENCE=1 "
            "-DR42C_FENCE_NO_K_GUARD=1"
        )
    else:  # c2
        # Drop both guards.
        extra_macro_flags = (
            "-DFUSED_STEP34=1 "
            "-DR41A_DEEP_K_FIX=1 "
            "-DR41A_EXTRACT_TILE_FENCE=1 "
            "-DR42C_FENCE_ANY_PATH=1"
        )

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    shapes_to_so = manifest["shapes_to_so_path"]
    shapes_to_source = manifest["shapes_to_source"]

    variants_table = parse_variants_from_bench()
    variants_dict = {}
    for sfx, flags in variants_table:
        tag = sfx.lstrip("_") if sfx else "default"
        variants_dict[tag] = flags

    plan = []  # (m, n, k, parent_tag, base_flags, source, parent_so)
    skipped = {}  # shape -> reason (use baseline so)
    for shape, parent_so in shapes_to_so.items():
        m, n, k = (int(x) for x in shape.split("x"))
        source = shapes_to_source[shape]
        if source == "R40A":
            # R40A used the !FUSED_STEP34 path; C1 fence is no-op.
            # Skip the rebuild and use the baseline .so.
            skipped[shape] = ("R40A_non_fused_path",
                              "C1 fence requires FUSED_STEP34", parent_so)
            continue
        if variant_id == "c2" and source != "R40B":
            # C2 phase only targets R40B-source shapes per directive.
            skipped[shape] = ("non_R40B_skipped_for_c2",
                              "C2 only targets R40B shapes", parent_so)
            continue
        safe_form = parent_tag_from_so(parent_so)
        try:
            full_tag = find_full_variant_tag(safe_form, variants_dict)
        except RuntimeError as e:
            print(f"  WARN {shape}: {e}; using safe_form as-is", flush=True)
            full_tag = safe_form
            if full_tag not in variants_dict:
                # Fallback: build from safe_form's flags by trying to find any
                # variant whose safe_tag matches.
                print(f"  ERROR: no variant flags found for {safe_form}",
                      file=sys.stderr)
                sys.exit(1)
        flags = variants_dict[full_tag]
        plan.append((m, n, k, full_tag, flags, source, parent_so))

    # Unique builds keyed on (n, k, safe_tag) — same parent variant builds same .so.
    unique = {}
    plan_keys = []  # to map plan -> unique
    for (m, n, k, parent_tag, flags, source, parent_so) in plan:
        ukey = (n, k, safe_tag(parent_tag))
        if ukey not in unique:
            unique[ukey] = (n, k, parent_tag, flags)
        plan_keys.append((m, n, k, source, parent_so, ukey, parent_tag))
    unique_tasks = list(unique.values())

    print(f"R42C builder [{variant_id}]: {len(plan)} shapes, "
          f"{len(unique_tasks)} unique builds")
    print(f"  Build dir:   {build_dir}")
    print(f"  Macro stack: {extra_macro_flags}")
    print(f"  Skipped:     {len(skipped)} shapes ({list(skipped.keys())})")

    t_start = time.time()
    completed = built = cached = failed = 0
    failures = []
    unique_results = {}  # ukey -> (status, module_name, so_path)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for ukey, (n, k, parent_tag, flags) in unique.items():
            fut = ex.submit(build_one, n, k, parent_tag, flags, variant_id,
                            build_dir, extra_macro_flags)
            futs[fut] = ukey
        for fut in as_completed(futs):
            ukey = futs[fut]
            res = fut.result()
            completed += 1
            key, status = res[0], res[1]
            if status == "cached":
                cached += 1
                unique_results[ukey] = (status, res[3], res[4])
                print(f"  [{completed:>3}/{len(unique_tasks)}] CACHED {key}",
                      flush=True)
            elif status == "ok":
                built += 1
                unique_results[ukey] = (status, res[3], res[4])
                print(f"  [{completed:>3}/{len(unique_tasks)}] BUILT  {key} "
                      f"({res[2]:.1f}s)", flush=True)
            else:
                failed += 1
                err = res[5] if len(res) > 5 else ""
                failures.append((key, err))
                unique_results[ukey] = (status, res[3], "")
                print(f"  [{completed:>3}/{len(unique_tasks)}] FAILED {key}",
                      flush=True)

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s: built={built}, cached={cached}, failed={failed}")
    if failures:
        print("\nFailures:")
        for key, err in failures[:8]:
            print(f"  {key}")
            for line in err.strip().split("\n")[-8:]:
                print(f"    {line}")

    # Build per-shape .so map
    shapes_to_so_path = {}
    shapes_to_source = {}
    notes = {}
    for (m, n, k, source, parent_so, ukey, parent_tag) in plan_keys:
        shape = f"{m}x{n}x{k}"
        status, module_name, so_path = unique_results[ukey]
        if status in ("cached", "ok"):
            shapes_to_so_path[shape] = so_path
            shapes_to_source[shape] = f"R42C_{variant_id}_from_{source}"
            notes[shape] = (f"R42C {variant_id} on {source} parent "
                            f"{safe_tag(parent_tag)}; "
                            f"+R42C_FENCE_{'NO_K_GUARD' if variant_id=='c1' else 'ANY_PATH'}")
        else:
            shapes_to_so_path[shape] = parent_so  # fallback (won't be used)
            shapes_to_source[shape] = f"BUILD_FAILED_from_{source}"
            notes[shape] = "build failed; harness should mark MISSING"
    # Skipped shapes use baseline .so
    for shape, (reason, detail, parent_so) in skipped.items():
        shapes_to_so_path[shape] = parent_so
        shapes_to_source[shape] = f"BASELINE_kept_{reason}"
        notes[shape] = detail

    out = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round": f"R42C_{variant_id}",
        "variant_id": variant_id,
        "macro_stack": extra_macro_flags,
        "build_dir": build_dir,
        "n_planned_builds": len(unique_tasks),
        "built": built,
        "cached": cached,
        "failed": failed,
        "skipped_shapes": list(skipped.keys()),
        "shapes_to_so_path": shapes_to_so_path,
        "shapes_to_source": shapes_to_source,
        "notes": notes,
    }
    out_manifest = os.path.join(SCRIPT_DIR,
                                f"R42C_{variant_id.upper()}_BUILD_MANIFEST.json")
    with open(out_manifest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Manifest: {out_manifest}")


if __name__ == "__main__":
    main()
