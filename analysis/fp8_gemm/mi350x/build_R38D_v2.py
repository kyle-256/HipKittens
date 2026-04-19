#!/usr/bin/env python3
"""R38 Opt D v2: For shapes that R38D couldn't fix with R37_FIX_B path,
try alternate code paths:
  - FUSED34: append -DFUSED_STEP34=1 to each candidate
  - LEGACY: append -DR37_FIX_B=0 (route through pre-R37 path)

Modules tagged _R38Dv2_<axis> to keep distinct from R38D.
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
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R38Dv2")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..")),
)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def parse_variants_from_bench():
    bench_path = os.path.join(SCRIPT_DIR, "bench_all_42.py")
    with open(bench_path) as f:
        src = f.read()
    pat = re.compile(r"variants\s*=\s*\[(.*?)^\s*\]", re.DOTALL | re.MULTILINE)
    m = pat.search(src)
    src_list = "variants = [\n" + m.group(1) + "\n]"
    ns = {}
    exec(src_list, ns)
    return ns["variants"]


def strip_bad_sched(flags):
    out = re.sub(r"\s*-mllvm\s+-amdgpu-sched-strategy=max-memory-clause\b", "", flags)
    return out.strip()


def build_one(n_dim, k_dim, parent_tag, axis, base_cppflags):
    extra = {"f34": "-DFUSED_STEP34=1", "leg": "-DR37_FIX_B=0"}[axis]
    module_name = f"tk_mxfp4_gluon_cpp_n{n_dim}_k{k_dim}_{parent_tag}_R38Dv2_{axis}"
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(BUILD_DIR, out_file)
    key = f"N={n_dim},K={k_dim},{parent_tag}_R38Dv2_{axis}"

    if os.path.exists(out_path):
        return (key, "cached", 0.0)

    with open(KERNEL_SRC) as f:
        kernel_src_text = f.read()
    patched = kernel_src_text.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},",
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n_dim}_k{k_dim}_{parent_tag}_R38Dv2_{axis}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    extra_cppflags = strip_bad_sched(base_cppflags) + " " + extra

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
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16

    os.makedirs(BUILD_DIR, exist_ok=True)
    variants = parse_variants_from_bench()
    suffix_to_flags = {}
    for sfx, flags in variants:
        tag = sfx.lstrip("_") if sfx else "default"
        suffix_to_flags[tag] = flags

    # Load R38D candidates and R38D results to find the failing shapes.
    with open(os.path.join(SCRIPT_DIR, "R38D_candidates.json")) as f:
        cands = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "bench_all42_results_R38_optD.json")) as f:
        r38d = json.load(f)

    # For each shape that's still NOT WIN under R38D, build top-5 candidates × {f34, leg}.
    # Only NO_FIX or LOSE shapes need v2.
    plan = []
    tasks = []
    for shape_str, info in cands["shapes"].items():
        if info["status"] != "have_candidates":
            continue
        m, n, k = info["M"], info["N"], info["K"]
        comp = info["comp"]
        ps = r38d["per_shape"].get(shape_str, {})
        # If R38D found a WIN already (correct AND ≥comp), skip.
        if ps.get("best_variant") and ps.get("best_tflops", 0) >= comp:
            continue
        for c in info["candidates"][:5]:
            tag = c["variant"]
            if tag not in suffix_to_flags:
                continue
            flags = suffix_to_flags[tag]
            for axis in ("f34", "leg"):
                tasks.append((n, k, tag, axis, flags))
                plan.append((m, n, k, tag, axis))

    unique_tasks = list(set(tasks))
    print(f"R38Dv2 builder: {len(plan)} shape×variant×axis entries, {len(unique_tasks)} unique builds")

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
                if completed % 20 == 0 or completed >= len(unique_tasks)-5:
                    print(f"  [{completed:>3}/{len(unique_tasks)}] BUILT {key} ({res[2]:.1f}s)")
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
        "shapes_to_R38Dv2": [
            {"M": m, "N": n, "K": k, "tag": tag, "axis": axis,
             "module": f"tk_mxfp4_gluon_cpp_n{n}_k{k}_{tag}_R38Dv2_{axis}"}
            for (m, n, k, tag, axis) in plan
        ],
        "built": built, "cached": cached, "failed": failed,
    }
    with open(os.path.join(SCRIPT_DIR, "R38Dv2_BUILD_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Manifest: R38Dv2_BUILD_MANIFEST.json")


if __name__ == "__main__":
    main()
