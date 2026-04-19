#!/usr/bin/env python3
"""R34-OptB-Build: VGPR-staged B-tile prefetch (FORK kernel kernel_mxfp4_gluon_cpp_vgprPF.cpp).

Per R34_DECIDER_VERDICT.md - aiter's vmcnt(15) safety comes from B-tile loads
targeting scratch VGPRs (no M0 update), reducing LDS-write-pointer pressure.
This fork mimics that pattern by routing the FIRST `VGPR_PF_N` Step4 B-tile
prefetches through scratch float4 VGPRs, with ds_write_b128 deposit at
function end (gated by VGPR_PF_DRAIN_VMCNT).

Variants on L6 incumbent ts_lgk2_v12_memc_btw_all:
  V0_legacyfork  : fork file with VGPR_PF_MODE=0 (sanity, must == incumbent)
  V0_vgprpf      : VGPR_PF_MODE=1, VGPR_PF_N=4, drain vmcnt(0) — the M0-relief sanity check
  V1_vmcnt15     : V0 + RELAXED_VMCNT=15 — THE hypothesis test
  V2_vmcnt15_snop: V1 + EXPLICIT_S_NOP=1 — composability with R33-A finding
  V3_vmcnt20     : V0 + RELAXED_VMCNT=20 — push toward aiter's full window (if V1 stable)
"""
import os, sys, sysconfig, subprocess, time, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp_vgprPF.cpp")

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
        "tag": "V0_legacyfork",
        "suffix": "_R34B_v12_memc_btw_all_legacyfork",
        "flags": f"{PARENT_FLAGS}",  # VGPR_PF_MODE=0 default
    },
    {
        "tag": "V0_vgprpf",
        "suffix": "_R34B_v12_memc_btw_all_vgprpf",
        "flags": f"{PARENT_FLAGS} -DVGPR_PF_MODE=1 -DVGPR_PF_N=4 -DVGPR_PF_DRAIN_VMCNT=0",
    },
    {
        "tag": "V1_vmcnt15",
        "suffix": "_R34B_v12_memc_btw_all_vmcnt15",
        "flags": f"{PARENT_FLAGS} -DVGPR_PF_MODE=1 -DVGPR_PF_N=4 -DVGPR_PF_DRAIN_VMCNT=0 -DBARRIER_TO_WAITCNT_RELAXED_VMCNT=15",
    },
    {
        "tag": "V2_vmcnt15_snop",
        "suffix": "_R34B_v12_memc_btw_all_vmcnt15_snop",
        "flags": f"{PARENT_FLAGS} -DVGPR_PF_MODE=1 -DVGPR_PF_N=4 -DVGPR_PF_DRAIN_VMCNT=0 -DBARRIER_TO_WAITCNT_RELAXED_VMCNT=15 -DEXPLICIT_S_NOP=1",
    },
    {
        "tag": "V3_vmcnt20",
        "suffix": "_R34B_v12_memc_btw_all_vmcnt20",
        "flags": f"{PARENT_FLAGS} -DVGPR_PF_MODE=1 -DVGPR_PF_N=4 -DVGPR_PF_DRAIN_VMCNT=0 -DBARRIER_TO_WAITCNT_RELAXED_VMCNT=20",
    },
    {
        "tag": "V4_vgprpf_n0",
        "suffix": "_R34B_v12_memc_btw_all_vgprpf_n0",
        "flags": f"{PARENT_FLAGS} -DVGPR_PF_MODE=1 -DVGPR_PF_N=0 -DVGPR_PF_DRAIN_VMCNT=0",
    },
    {
        "tag": "V5_vgprpf_n1",
        "suffix": "_R34B_v12_memc_btw_all_vgprpf_n1",
        "flags": f"{PARENT_FLAGS} -DVGPR_PF_MODE=1 -DVGPR_PF_N=1 -DVGPR_PF_DRAIN_VMCNT=0",
    },
    {
        "tag": "V6_discard_vgpr",
        "suffix": "_R34B_v12_memc_btw_all_discard_vgpr",
        "flags": f"{PARENT_FLAGS} -DVGPR_PF_MODE=1 -DVGPR_PF_N=4 -DVGPR_PF_DRAIN_VMCNT=0 -DVGPR_PF_DISCARD_VGPR=1",
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
    log_path = os.path.join(BUILD_DIR, f"compile_R34B_{tag}_n{N}_k{K}.log")
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
    print(f"R34-B builds: {len(JOBS)} (parallel)")
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
    out_json = os.path.join(SCRIPT_DIR, "R34_OPT_B_BUILD_RESULTS.json")
    with open(out_json, "w") as f:
        json.dump({"total_sec": round(dt_global, 1),
                   "results": sorted(results, key=lambda x: x.get("tag",""))}, f, indent=2)
    print(f"\nWall clock: {dt_global:.1f}s")
    print(f"Results: {out_json}")
    fails = [r for r in results if r.get("status") not in ("PASS",)]
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
