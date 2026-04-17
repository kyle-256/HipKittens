#!/usr/bin/env python3
"""Round 19 Optimizer C: Stack R18A's BARRIER_TO_WAITCNT macros onto the
5 R10/R11 iterilp WINs (S1-S5). 5 shapes x 3 macro variants = 15 builds.

Hypothesis: barrier-removal (source-level cycle save) ⊥ iterative-ilp
(backend scheduler). They should compose. R16 falsified iterilp ⊥ regalloc
but source-rewrite is a different axis; worth a shot.

Parent macros (per R16A which we trust):
    S1 (M=14336,N=4096,K=32768)  parent=_lgk2_dc      iterilp
    S2 (M=16384,N=4096,K=28672)  parent=_u32          iterilp
    S3 (M=4096,N=32768,K=28672)  parent=_v20_memc     iterilp
    S4 (M=4096,N=28672,K=32768)  parent=_u16          iterilp
    S5 (M=4096,N=32768,K=14336)  parent=_ts_lgk2_memc iterilp

Output suffix pattern: <parent>_iterilp_btw_{step3,step12,all}
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

ITERILP = "-mllvm -amdgpu-sched-strategy=iterative-ilp"

# (tag, M, N, K, suffix, parent_macros)
JOBS = [
    ("S1", 14336, 4096, 32768,
     "_lgk2_dc",
     "-DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("S2", 16384, 4096, 28672,
     "_u32",
     "-DUNROLL_K=32"),
    ("S3",  4096, 32768, 28672,
     "_v20_memc",
     "-DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("S4",  4096, 28672, 32768,
     "_u16",
     "-DUNROLL_K=16"),
    ("S5",  4096, 32768, 14336,
     "_ts_lgk2_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
]

# (variant_tag, macro_define)
BTW_VARIANTS = [
    ("step3",  "-DBARRIER_TO_WAITCNT_STEP3=1"),
    ("step12", "-DBARRIER_TO_WAITCNT_STEP12=1"),
    ("all",    "-DBARRIER_TO_WAITCNT_ALL=1"),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(tag, M, N, K, parent_suffix, parent_macros, btw_tag, btw_macro):
    full_suffix = f"{parent_suffix}_r19c_iterilp_btw_{btw_tag}"
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (tag, full_suffix, "cached", 0.0, "")
    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{N}_k{K}{full_suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)
    # iterilp last (last-wins for sched-strategy family)
    flags = f"{parent_macros} {btw_macro} {ITERILP}"
    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DK_DIM={K} -DN_DIM={N} {flags} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (tag, full_suffix, "FAIL", dt, r.stderr[-600:])
    return (tag, full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    tasks = []
    for (tag, M, N, K, parent_suffix, parent_macros) in JOBS:
        for (btw_tag, btw_macro) in BTW_VARIANTS:
            tasks.append((tag, M, N, K, parent_suffix, parent_macros, btw_tag, btw_macro))
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    print(f"Total builds: {len(tasks)}  (workers={workers})")
    print("Compound: parent_macros + BARRIER_TO_WAITCNT_<v> + iterative-ilp (last)")
    t0 = time.time()
    fail_lines = []
    n_ok = n_cached = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, *t): t[0] for t in tasks}
        for fut in as_completed(futs):
            tag, full_suffix, status, dt, err = fut.result()
            print(f"  {tag:3s} {full_suffix:60s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                n_fail += 1
                fail_lines.append(f"--- {tag} {full_suffix} ---\n{err}\n")
            elif status == "cached":
                n_cached += 1
            else:
                n_ok += 1
    print(f"\nElapsed: {time.time()-t0:.1f}s   ok={n_ok} cached={n_cached} fail={n_fail}")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
    return 0 if not fail_lines else 1


if __name__ == "__main__":
    sys.exit(main())
