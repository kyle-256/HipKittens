#!/usr/bin/env python3
"""Round 22 Optimizer A — LDS sub-arbitration probes (LDS_RD_STAGGER_NOP).

R21-recon found TCP_TA_DATA_STALL = 167-294 % of GRBM with 0 % LDS bank
conflict on DLA1 / DLA2 / DLA7. That points at LDS port-side sub-arbitration:
multiple ds_read_b128 issued per cycle exceeding LDS port bandwidth, not bank
conflict.

R22A inserts `s_nop {0,1}` after each `ds_read_b128` line in the 4 hot-path
KPAIR functions (kpair_32mfma_with_lds_and_pf [_S2],
kpair_32mfma_with_lds_rowspread_pf [_S3],
kpair_32mfma_with_lds_and_pf_swapped_sel [_S4],
kpair_64mfma_step12_swapped_sel [_S12-fused]).

Targets: DLA2 (128256x32768x4096) and DLA7 (28672x32768x4096) — the two shapes
with the highest TCP stall (~292/294 % GRBM). DLA1 is bonus (167 %).

Per benchmark-rules.md: warmup=200 iters=500 trim=10%. Build defaults all-zero
so opt-out path is bit-identical to the parent .so.

Variants per shape:
  ""              parent baseline (rebuilt with R22A no-op hooks)
  "_r22a_nop1"    LDS_RD_STAGGER_NOP=1  (inserts `s_nop 0` => 1 idle cycle)
  "_r22a_nop2"    LDS_RD_STAGGER_NOP=2  (inserts `s_nop 1` => 2 idle cycles)
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

# (label, M_native, N, K, parent_suffix, parent_flags)
# DLA2 = 128256 x 32768 x 4096, parent best = ts_gm2_v12_memc_dc.
# DLA7 = 28672 x 32768 x 4096,  parent best = ts_lgk2_v12_memc_btw_all (R20A).
# DLA1 = 4096 x 32768 x 128256, parent best = ts_pf6_6_v12_memc.
SHAPES = [
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA7",  28672, 32768,  4096, "_ts_lgk2_v12_memc_btw_all",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -DBARRIER_TO_WAITCNT_ALL=1"),
    ("DLA1",   4096, 32768, 128256, "_ts_pf6_6_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
]

VARIANTS = [
    ("_r22a_baseline", ""),
    ("_r22a_nop1",     "-DLDS_RD_STAGGER_NOP=1"),
    ("_r22a_nop2",     "-DLDS_RD_STAGGER_NOP=2"),
]

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(N, K, full_suffix, parent_flags, extra_flags, force=False):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if force and os.path.exists(so_path):
        os.remove(so_path)
    if os.path.exists(so_path):
        return (full_suffix, "cached", 0.0, "")
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
        f"-DK_DIM={K} -DN_DIM={N} {parent_flags} {extra_flags} -o {so_path}"
    )
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (full_suffix, "FAIL", dt, r.stderr[-500:])
    return (full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    force = "--force" in sys.argv
    jobs = []
    for (lab, M, N, K, ps, pflags) in SHAPES:
        for (vs, ef) in VARIANTS:
            full = ps + vs
            jobs.append((lab, N, K, full, pflags, ef))

    print(f"R22A builds: {len(jobs)} ({len(SHAPES)} shapes x {len(VARIANTS)} variants)  force={force}")
    print("=" * 100)
    workers = int(os.environ.get("BUILD_WORKERS", "6"))
    t0 = time.time()
    fail_lines = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, N, K, full, pflags, ef, force): (lab, full)
                for (lab, N, K, full, pflags, ef) in jobs}
        for fut in as_completed(futs):
            lab, full = futs[fut]
            full_suffix, status, dt, err = fut.result()
            print(f"  {lab:5s} {full_suffix:65s} {status:8s} ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                fail_lines.append(f"--- {lab} {full_suffix} ---\n{err}\n")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    if fail_lines:
        print("\nFAILURES:\n" + "\n".join(fail_lines))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
