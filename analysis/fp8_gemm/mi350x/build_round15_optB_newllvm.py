#!/usr/bin/env python3
"""Round 15 Optimizer B: untested LLVM flag families.

R14A tested 27 flags clustered around membound-threshold, regalloc-bias,
dce-in-ra, dpp-combine, loop-prefetch, sched-metric-bias, lst, sghazard,
etc. R15B explores DIFFERENT families per the AGENT_PROMPT:

  1. Coalescer: join-globalcopies/splitedges/liveintervals false, twoaddr-reschedule,
     large-interval-{freq,size}-threshold, late-remat-update-threshold
  2. AGPR/spill: enable-deferred-spilling, disable-spill-fusing,
     sink-insts-to-avoid-spills, amdgpu-prealloc-sgpr-spill-vgprs,
     amdgpu-opt-vgpr-liverange=false
  3. Machine sink/LICM: machine-sink-load-instrs-threshold, machine-sink-cycle-limit,
     machine-sink-bfi, sink-freq-percent-threshold, disable-machine-sink,
     disable-machine-licm, hoist-cheap-insts, avoid-speculation
  4. Post-RA scheduler: misched-postra=true, enable-post-misched,
     disable-post-ra, post-RA-scheduler, break-anti-dependencies
  5. iglp / igroup: amdgpu-igrouplp-exact-solver,
     amdgpu-igrouplp-exact-solver-cutoff
  6. Loop / prefetch: amdgpu-disable-loop-alignment, loop-rotate-multi,
     enable-loop-distribute, max-prefetch-iters-ahead
  7. AMDGPU specific: amdgpu-aa, amdgpu-use-aa-in-codegen, amdgpu-reassign-regs,
     amdgpu-opt-exec-mask-pre-ra, amdgpu-enable-pre-ra-optimizations,
     amdgpu-enable-delay-alu=false, amdgpu-enable-vopd, amdgpu-early-ifcvt,
     amdgpu-sgpr-hazard-boundary-cull=false, amdgpu-sgpr-hazard-mem-wait-cull=false,
     amdgpu-sgpr-hazard-wait=false

All flags verified to exist via `llc -mcpu=gfx950 -help-hidden`.
NONE overlap with R14A's FLAGS list.
"""
import os, sys, sysconfig, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")

# Same 4 broken deep-LOSE shapes as R14A
SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP3_PF_N=6 -DSTEP4_PF_N=6 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc",
     "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DSTEP3_BARRIER_VMCNT=12 -mllvm -amdgpu-sched-strategy=max-memory-clause"),
    ("P1",   28672,  4096, 16384, "_ts_gm8",
     "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8"),
]

# (tag, extra_flags) -- tag goes into the variant suffix.
# Categories from AGENT_PROMPT.
FLAGS = [
    # Family 1: COALESCER
    ("nojglc",   "-mllvm -join-globalcopies=false"),
    ("nojli",    "-mllvm -join-liveintervals=false"),
    ("nojse",    "-mllvm -join-splitedges=false"),
    ("notarsch", "-mllvm -twoaddr-reschedule=false"),
    ("largeivf2","-mllvm -large-interval-freq-threshold=2"),
    ("largeivs8","-mllvm -large-interval-size-threshold=8"),
    ("lateremat0","-mllvm -late-remat-update-threshold=0"),
    ("revlocal",  "-mllvm -greedy-reverse-local-assignment"),

    # Family 2: SPILL / AGPR
    ("defspill", "-mllvm -enable-deferred-spilling"),
    ("nospillfu","-mllvm -disable-spill-fusing"),
    ("sinkavoidspill", "-mllvm -sink-insts-to-avoid-spills"),
    ("preallocs","-mllvm -amdgpu-prealloc-sgpr-spill-vgprs"),
    ("noliveopt","-mllvm -amdgpu-opt-vgpr-liverange=false"),

    # Family 3: MACHINE SINK / LICM
    ("nosink",   "-mllvm -disable-machine-sink"),
    ("nolicm",   "-mllvm -disable-machine-licm"),
    ("hoistcheap","-mllvm -hoist-cheap-insts"),
    ("noavoidspec","-mllvm -avoid-speculation=false"),
    ("sinkbfi",  "-mllvm -machine-sink-bfi=false"),
    ("sinkcycle100","-mllvm -machine-sink-cycle-limit=100"),

    # Family 4: POST-RA SCHED
    ("postmi",   "-mllvm -misched-postra=true"),
    ("enpostmi", "-mllvm -enable-post-misched"),
    ("nopostra", "-mllvm -disable-post-ra"),
    ("postraN",  "-mllvm -post-RA-scheduler"),
    ("breakcrit","-mllvm -break-anti-dependencies=critical"),
    ("breakall", "-mllvm -break-anti-dependencies=all"),

    # Family 5: IGLP / igroup
    ("iglpcut0", "-mllvm -amdgpu-igrouplp-exact-solver-cutoff=0"),
    ("iglpexact","-mllvm -amdgpu-igrouplp-exact-solver"),

    # Family 6: LOOP / prefetch
    ("nolal",    "-mllvm -amdgpu-disable-loop-alignment"),
    ("looprmul", "-mllvm -loop-rotate-multi"),
    ("loopdist", "-mllvm -enable-loop-distribute"),

    # Family 7: AMDGPU specific
    ("aaaa",     "-mllvm -amdgpu-use-aa-in-codegen=true"),
    ("noaa",     "-mllvm -amdgpu-use-aa-in-codegen=false"),
    ("rerag",    "-mllvm -amdgpu-reassign-regs=true"),
    ("eppra",    "-mllvm -amdgpu-enable-pre-ra-optimizations=true"),
    ("noeppra",  "-mllvm -amdgpu-enable-pre-ra-optimizations=false"),
    ("noemxpre", "-mllvm -amdgpu-opt-exec-mask-pre-ra=false"),
    ("nodalu",   "-mllvm -amdgpu-enable-delay-alu=false"),
    ("vopd",     "-mllvm -amdgpu-enable-vopd"),
    ("eifcvt",   "-mllvm -amdgpu-early-ifcvt"),
    ("nosgpcb",  "-mllvm -amdgpu-sgpr-hazard-boundary-cull=false"),
    ("nosgpmwc", "-mllvm -amdgpu-sgpr-hazard-mem-wait-cull=false"),
    ("nosgpwa",  "-mllvm -amdgpu-sgpr-hazard-wait=false"),
]

NEW_PREFIX = "_r15b_"

BASE = (
    f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    f"-I/opt/rocm/include/hip "
    f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    f"-shared -fPIC -std=c++20 -w "
    f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm"
)


def build_one(label, N, K, parent_flags, parent_suffix, tag, extra_flags):
    full_suffix = parent_suffix + NEW_PREFIX + tag
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return (label, tag, full_suffix, "cached", 0.0, "")
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
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return (label, tag, full_suffix, "FAIL", dt, r.stderr[-400:])
    return (label, tag, full_suffix, "OK", dt, "")


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    workers = int(os.environ.get("BUILD_WORKERS", "8"))
    jobs = []
    for (lab, M, N, K, ps, pf) in SHAPES:
        for (tag, ef) in FLAGS:
            jobs.append((lab, M, N, K, ps, pf, tag, ef))
    print(f"Total builds: {len(jobs)}  (workers={workers}, {len(SHAPES)} shapes x {len(FLAGS)} flags)")
    t0 = time.time()
    fail_lines = []
    n_ok = n_cached = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, lab, N, K, pf, ps, tag, ef): (lab, tag)
                for (lab, M, N, K, ps, pf, tag, ef) in jobs}
        for fut in as_completed(futs):
            label, tag, full_suffix, status, dt, err = fut.result()
            print(f"  {label:6s} {tag:14s}  {status:8s}  ({dt:.1f}s)", flush=True)
            if status == "FAIL":
                n_fail += 1
                fail_lines.append(f"--- {label} {tag} ---\n{err}\n")
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
