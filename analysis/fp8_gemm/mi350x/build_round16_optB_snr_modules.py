#!/usr/bin/env python3
"""Build N=256, K=4096 modules for SNR correctness checks of R16B compounds.

Per AGENT_PROMPT: "SNR check on small 256x256x4096 random fp4 first".
We need a small-shape build per (parent_macros, flag) combo so the K=4096
reduction stays within bf16 dynamic range and we get meaningful SNR.

Builds: 5 parent macro sets × (parent + iterilp + 4 R15B safe-DIFF flags) = 30 modules
        Each at N=256, K=4096.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

ITERILP = "-mllvm -amdgpu-sched-strategy=iterative-ilp"

# (lab, parent_suffix, parent_cppflags) -- N=256, K=4096 fixed
SHAPES = [
    ("S1", "_lgk2_dc",
     "-DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("S2", "_u32",          "-DUNROLL_K=32"),
    ("S3", "_v20_memc",
     "-DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("S4", "_u16",          "-DUNROLL_K=16"),
    ("S5", "_ts_lgk2_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
]

SAFE_FLAGS = [
    ("noemxpre",       "-mllvm -amdgpu-opt-exec-mask-pre-ra=false"),
    ("largeivf2",      "-mllvm -large-interval-freq-threshold=2"),
    ("nolicm",         "-mllvm -disable-machine-licm"),
    ("sinkavoidspill", "-mllvm -sink-insts-to-avoid-spills"),
]

R16B_INFIX = "_r16b_iterilp_"
ITERILP_SUFFIX_FOR_S1S2 = "_r10_iterilp"
ITERILP_SUFFIX_FOR_S3S5 = "_r11_iterilp"

N_SNR = 256
K_SNR = 4096

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_so(N, K, full_suffix, full_flags, label):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (label, full_suffix, "cached", 0.0, "")
    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{N}_k{K}{full_suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)
    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DK_DIM={K} -DN_DIM={N} {full_flags} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (label, full_suffix, "FAIL", dt, r.stderr[-500:])
    return (label, full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    jobs = []
    for (lab, ps, pf) in SHAPES:
        # parent only
        jobs.append((lab + "/parent", ps, pf))
        # iterilp only
        ilp_suf = (ITERILP_SUFFIX_FOR_S1S2 if lab in ("S1", "S2")
                   else ITERILP_SUFFIX_FOR_S3S5)
        jobs.append((lab + "/iterilp", ps + ilp_suf, f"{pf} {ITERILP}"))
        # compounds
        for (tag, ef) in SAFE_FLAGS:
            jobs.append((lab + "/" + tag, ps + R16B_INFIX + tag,
                         f"{pf} {ITERILP} {ef}"))
    print(f"Total SNR module builds: {len(jobs)}  (workers={workers})  N={N_SNR} K={K_SNR}")
    t0 = time.time()
    fails = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_so, N_SNR, K_SNR, suf, ff, lab): lab
                for (lab, suf, ff) in jobs}
        for fut in as_completed(futs):
            label, suf, status, dt, err = fut.result()
            print(f"  {label:25s}  {suf:55s}  {status:8s}  ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fails.append(f"--- {label} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fails:
        print("\nFAILURES:\n" + "\n".join(fails))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
