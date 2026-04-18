#!/usr/bin/env python3
"""R30 OPT A build: 5 variants × specific (M,N,K) — kernel needs M_DIM at compile time.

Builds:
  DLA2 (M=128256, N=32768, K=4096):
    - ts_lgk2_memc_btw_all
    - ts_lgk2_v12_memc_btw_all
    - v20_memc_btw_step3
  S5L (M=32768, N=14336, K=2048):
    - ts_lgk2_memc_btw_all
    - v20_memc_btw_step3

Output SOs use suffix _r30oa to avoid colliding with the existing default-M=8192 SOs.
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

VARIANT_FLAGS = {
    "ts_lgk2_memc_btw_all":     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -mllvm -amdgpu-sched-strategy=max-memory-clause -DBARRIER_TO_WAITCNT_ALL=1",
    "ts_lgk2_v12_memc_btw_all": "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -DBARRIER_TO_WAITCNT_ALL=1",
    "v20_memc_btw_step3":       "-DSTEP3_BARRIER_VMCNT=20 -mllvm -amdgpu-sched-strategy=max-memory-clause -DBARRIER_TO_WAITCNT_STEP3=1",
}

JOBS = [
    # (label, M, N, K, variant)
    ("DLA2_lgk2_memc_btw_all",     128256, 32768, 4096, "ts_lgk2_memc_btw_all"),
    ("DLA2_lgk2_v12_memc_btw_all", 128256, 32768, 4096, "ts_lgk2_v12_memc_btw_all"),
    ("DLA2_v20_memc_btw_step3",    128256, 32768, 4096, "v20_memc_btw_step3"),
    ("S5L_lgk2_memc_btw_all",       32768, 14336, 2048, "ts_lgk2_memc_btw_all"),
    ("S5L_v20_memc_btw_step3",      32768, 14336, 2048, "v20_memc_btw_step3"),
]


def build_one(args):
    label, M, N, K, variant = args
    suffix = f"_{variant}_r30oa_m{M}"
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")

    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace("PYBIND11_MODULE(tk_mxfp4_gluon_cpp,", f"PYBIND11_MODULE({module_name},")
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{N}_k{K}{suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    extra = VARIANT_FLAGS[variant]
    cmd = (
        f"/opt/rocm/bin/hipcc {wrapper_src} {BASE} "
        f"-DM_DIM={M} -DN_DIM={N} -DK_DIM={K} {extra} -o {so_path}"
    )
    log_path = os.path.join(BUILD_DIR, f"compile_R30OA_{label}.log")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1500)
    dt = time.time() - t0
    with open(log_path, "w") as f:
        f.write("CMD:\n" + cmd + "\n\nSTDOUT:\n" + r.stdout + "\n\nSTDERR:\n" + r.stderr + "\n")
    out = {
        "label": label, "M": M, "N": N, "K": K, "variant": variant,
        "module_name": module_name, "so_path": so_path,
        "compile_sec": round(dt, 1), "rc": r.returncode, "log": log_path,
    }
    if r.returncode != 0 or not os.path.exists(so_path):
        out["status"] = "FAIL"
        out["stderr_tail"] = r.stderr[-2000:]
        return out
    out["so_size_kb"] = round(os.path.getsize(so_path) / 1024, 1)

    vgpr = agpr = sgpr = spills = scratch = occ = None
    for line in r.stderr.splitlines():
        if "VGPRs" in line and "AGPRs" in line:
            m = re.search(r"VGPRs:\s*(\d+)", line); vgpr = int(m.group(1)) if m else vgpr
            m = re.search(r"AGPRs:\s*(\d+)", line); agpr = int(m.group(1)) if m else agpr
            m = re.search(r"SGPRs:\s*(\d+)", line); sgpr = int(m.group(1)) if m else sgpr
        if "Spill" in line:
            m = re.search(r"VGPRSpills?:\s*(\d+)", line); spills = int(m.group(1)) if m else spills
        if "ScratchSize" in line:
            m = re.search(r"ScratchSize.*?(\d+)", line); scratch = int(m.group(1)) if m else scratch
        if "Occupancy" in line:
            m = re.search(r"Occupancy.*?(\d+)", line); occ = int(m.group(1)) if m else occ
    out.update({"vgpr": vgpr, "agpr": agpr, "sgpr": sgpr, "spills": spills, "scratch": scratch, "occupancy": occ})
    out["status"] = "PASS"
    return out


def main():
    print(f"R30 OPT A builds: {len(JOBS)}")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=len(JOBS)) as ex:
        futs = {ex.submit(build_one, j): j[0] for j in JOBS}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"  {r['label']:32s} {r['status']:6s} M={r['M']} N={r['N']} K={r['K']} "
                  f"vgpr={r.get('vgpr','?')} agpr={r.get('agpr','?')} occ={r.get('occupancy','?')} "
                  f"spills={r.get('spills','?')}  ({r['compile_sec']}s)", flush=True)
            if r["status"] != "PASS":
                print("    STDERR:", r.get("stderr_tail", "")[-800:])
    dt = time.time() - t0
    out = os.path.join(SCRIPT_DIR, "R30_OPT_A_BUILD.json")
    with open(out, "w") as f:
        json.dump({"total_sec": round(dt, 1), "results": sorted(results, key=lambda r: r["label"])}, f, indent=2)
    print(f"\nWall: {dt:.1f}s. Wrote {out}")
    return 1 if any(r["status"] != "PASS" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
