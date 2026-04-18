#!/usr/bin/env python3
"""R31-OptA-Build: UNROLL_K sweep on L6 (4096x32768x128256).

Parent: ts_lgk2_v12_memc_btw_all (incumbent best at 5353.9 TFLOPS, 92.6% of comp).
Sweep: UNROLL_K in {1, 2, 4, 16, 32}.

Builds for N_DIM=32768, K_DIM=128256, M_DIM compiled at runtime (M=4096 at bench).
Captures VGPR/AGPR/SGPR/spills/occupancy from compile remarks.
"""
import os, sys, sysconfig, subprocess, time, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip -I{SCRIPT_DIR} "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-Rpass-analysis=kernel-resource-usage "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)

# L6 best parent flags (from bench_all_42.py:480):
PARENT_FLAGS = (
    "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 "
    "-mllvm -amdgpu-sched-strategy=max-memory-clause "
    "-DBARRIER_TO_WAITCNT_ALL=1"
)

N = 32768
K = 128256

JOBS = []
for u in [1, 2, 4, 16, 32]:
    JOBS.append({
        "tag": f"u{u}",
        "N": N,
        "K": K,
        "suffix": f"_R31A_ts_lgk2_v12_memc_btw_all_u{u}",
        "flags": f"{PARENT_FLAGS} -DUNROLL_K={u}",
    })


def build_one(job):
    N = job["N"]; K = job["K"]
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
    log_path = os.path.join(BUILD_DIR, f"compile_R31A_{tag}_n{N}_k{K}.log")
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

    ru_lines = []
    for line in r.stderr.splitlines():
        if any(k in line for k in ("VGPRs", "AGPRs", "SGPRs", "Occupancy", "spill", "Spill", "ScratchSize")):
            ru_lines.append(line.strip())
    result["resource_usage"] = ru_lines[-30:]
    return result


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    print(f"R31-A builds: {len(JOBS)} (parallel)")
    print(f"Parent: ts_lgk2_v12_memc_btw_all on L6 (N=32768 K=128256)")
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
            print(f"  {tag:6s} {status:10s} compile={dt}s VGPR={v} spills={sp} occ={oc}", flush=True)

    dt_global = time.time() - t_global
    out_json = os.path.join(SCRIPT_DIR, "R31_OPT_A_BUILD_RESULTS.json")
    with open(out_json, "w") as f:
        json.dump({"total_sec": round(dt_global, 1), "results": sorted(results, key=lambda x: x.get("tag",""))}, f, indent=2)
    print(f"\nWall clock: {dt_global:.1f}s")
    print(f"Results: {out_json}")
    fails = [r for r in results if r.get("status") not in ("PASS",)]
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
