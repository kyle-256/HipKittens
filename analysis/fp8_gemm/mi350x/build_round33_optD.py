#!/usr/bin/env python3
"""R33-OptD-Build: aiter SRD config swap (FORK kernel kernel_mxfp4_gluon_cpp_aiterSRD.cpp).

Per R33_AITER_ARCHAEOLOGY Finding A — switching SRD to aiter's
ADD_TID_ENABLE + INDEX_STRIDE=01 pattern unlocks safe vmcnt(15) operation by
moving the OOB envelope from per-buffer-flat to per-lane-stripe.

Variants on L6 incumbent ts_lgk2_v12_memc_btw_all:
  V1 (sanity)  : +AITER_SRD_MODE=1 only — confirm SRD swap doesn't break correctness
  V2 (vm15)    : +AITER_SRD_MODE=1 +BARRIER_TO_WAITCNT_RELAXED_VMCNT=15 — the unlock
  V3 (vm25)    : +AITER_SRD_MODE=1 +BARRIER_TO_WAITCNT_RELAXED_VMCNT=25 — push further

Plus a baseline LEGACY-mode FORK build to confirm the fork kernel itself is identical
when AITER_SRD_MODE=0:
  V0 (legacy)  : fork kernel with AITER_SRD_MODE=0 (default) — should match incumbent
"""
import os, sys, sysconfig, subprocess, time, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp_aiterSRD.cpp")

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip -I{SCRIPT_DIR} "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-Rpass-analysis=kernel-resource-usage "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)

PARENT_FLAGS = (
    "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 "
    "-mllvm -amdgpu-sched-strategy=max-memory-clause "
    "-DBARRIER_TO_WAITCNT_ALL=1"
)

N = 32768
K = 128256

JOBS = [
    {
        "tag": "V0_legacy_fork",
        "suffix": "_R33D_v12_memc_btw_all_legacyfork",
        "flags": f"{PARENT_FLAGS}",  # fork without AITER_SRD_MODE -> default 0 (legacy)
    },
    {
        "tag": "V1_aitersrd",
        "suffix": "_R33D_v12_memc_btw_all_aitersrd",
        "flags": f"{PARENT_FLAGS} -DAITER_SRD_MODE=1",
    },
    {
        "tag": "V2_aitersrd_vm15",
        "suffix": "_R33D_v12_memc_btw_all_aitersrd_vm15",
        "flags": f"{PARENT_FLAGS} -DAITER_SRD_MODE=1 -DBARRIER_TO_WAITCNT_RELAXED_VMCNT=15",
    },
    {
        "tag": "V3_aitersrd_vm25",
        "suffix": "_R33D_v12_memc_btw_all_aitersrd_vm25",
        "flags": f"{PARENT_FLAGS} -DAITER_SRD_MODE=1 -DBARRIER_TO_WAITCNT_RELAXED_VMCNT=25",
    },
]


def build_one(job):
    suffix = job["suffix"]; tag = job["tag"]
    extra_flags = job["flags"]
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")

    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{N}_k{K}{suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DK_DIM={K} -DN_DIM={N} {extra_flags} -o {so_path}"
    )
    log_path = os.path.join(BUILD_DIR, f"compile_R33D_{tag}_n{N}_k{K}.log")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1500)
    dt = time.time() - t0
    with open(log_path, "w") as f:
        f.write("CMD:\n" + cmd + "\n\nSTDOUT:\n" + r.stdout + "\n\nSTDERR:\n" + r.stderr + "\n")

    result = {
        "tag": tag, "N": N, "K": K, "suffix": suffix,
        "module_name": module_name, "so_path": so_path,
        "compile_sec": round(dt, 1), "returncode": r.returncode,
        "log_path": log_path,
    }
    if r.returncode != 0 or not os.path.exists(so_path):
        result["status"] = "FAIL"
        result["stderr_tail"] = r.stderr[-3000:]
        return result

    result["so_size_kb"] = round(os.path.getsize(so_path) / 1024, 1)

    vgpr = agpr = sgpr = spills = occ = scratch = None
    for line in r.stderr.splitlines():
        m = re.search(r"VGPRs:\s*(\d+)", line)
        if m: vgpr = int(m.group(1))
        m = re.search(r"AGPRs:\s*(\d+)", line)
        if m: agpr = int(m.group(1))
        m = re.search(r"SGPRs:\s*(\d+)", line)
        if m: sgpr = int(m.group(1))
        m = re.search(r"VGPRSpill[s]?:\s*(\d+)", line)
        if m: spills = int(m.group(1))
        m = re.search(r"Occupancy:\s*(\d+)", line)
        if m: occ = int(m.group(1))
        m = re.search(r"ScratchSize\s*\[bytes/lane\]:\s*(\d+)", line)
        if m: scratch = int(m.group(1))

    result["vgpr"] = vgpr
    result["agpr"] = agpr
    result["sgpr"] = sgpr
    result["spills"] = spills
    result["occupancy"] = occ
    result["scratch"] = scratch
    result["status"] = "PASS" if os.path.getsize(so_path) > 200 * 1024 else "FAIL_SMALL"
    return result


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    print(f"R33-D builds: {len(JOBS)} (parallel)")
    t_global = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=len(JOBS)) as ex:
        futs = {ex.submit(build_one, job): job["tag"] for job in JOBS}
        for fut in as_completed(futs):
            tag = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"tag": tag, "status": "EXCEPTION", "error": str(e)}
            results.append(res)
            status = res.get("status", "?")
            dt = res.get("compile_sec", 0)
            v = res.get("vgpr", "?")
            sp = res.get("spills", "?")
            oc = res.get("occupancy", "?")
            sc = res.get("scratch", "?")
            print(f"  {tag:24s} {status:10s} compile={dt}s VGPR={v} spills={sp} scratch={sc} occ={oc}", flush=True)

    dt_global = time.time() - t_global
    out_json = os.path.join(SCRIPT_DIR, "R33_OPT_D_BUILD_RESULTS.json")
    with open(out_json, "w") as f:
        json.dump({"total_sec": round(dt_global, 1),
                   "results": sorted(results, key=lambda x: x.get("tag",""))}, f, indent=2)
    print(f"\nWall clock: {dt_global:.1f}s")
    print(f"Results: {out_json}")
    fails = [r for r in results if r.get("status") not in ("PASS",)]
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
