#!/usr/bin/env python3
"""R26-D AUDIT: Build + test the missing K_EXACT variants for shape 27 (and other LOSE shapes).

Shape 27: M=14336, N=4096, K=32768
- Current FINAL best: ts_pf4_memc_btw_step3 = 4965.3 TFLOPS (94.7% of comp)
- MISSING from FINAL parallel: ts_lgk2_gm7_memc_pfoff124_kx32768_btw_all
  (R25-G WIN: SC +17.55% vs parent)

Other LOSE shapes from FINAL partial (12, 18, 21, 22, 27):
- Shape 12 (M=4096, N=14336, K=16384): missing _ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all
- Shape 18 (M=4096, N=28672, K=32768): missing _ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all
- Shape 21 (M=4096, N=32768, K=14336): missing _ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all
- Shape 22 (M=4096, N=32768, K=28672): missing R25-G/H K=28672 variant — NOT in bench_all_42 list
- Shape 27 (M=14336, N=4096, K=32768): missing _ts_lgk2_gm7_memc_pfoff124_kx32768_btw_all

We also expect to see other LOSEs as FINAL progresses (16384x4096x28672 etc).
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")
TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT",
                         "/shared_nfs/kyle/test/HipKittens")

# (suffix, cppflags) — all from bench_all_42.py R25-G/H block
VARIANTS = {
    "_ts_lgk2_gm7_memc_pfoff124_kx32768_btw_all":
        "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DGROUP_SIZE_M=7 "
        "-DR25C_TAIL_PF_OFF_ITERS=124 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=32768 "
        "-mllvm -amdgpu-sched-strategy=max-memory-clause "
        "-DBARRIER_TO_WAITCNT_ALL=1",
    "_ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all":
        "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DGROUP_SIZE_M=7 "
        "-DR25C_TAIL_PF_OFF_ITERS=56 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=16384 "
        "-mllvm -amdgpu-sched-strategy=max-memory-clause "
        "-DBARRIER_TO_WAITCNT_ALL=1",
    "_ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all":
        "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 "
        "-DR25C_TAIL_PF_OFF_ITERS=120 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=32768 "
        "-mllvm -amdgpu-sched-strategy=max-memory-clause "
        "-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule "
        "-DBARRIER_TO_WAITCNT_ALL=1",
    "_ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all":
        "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 -DTAIL_BARRIER_VMCNT=0 "
        "-DR25C_TAIL_PF_OFF_ITERS=54 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=14336 "
        "-mllvm -amdgpu-sched-strategy=max-memory-clause "
        "-mllvm -amdgpu-disable-clustered-low-occupancy-reschedule "
        "-DBARRIER_TO_WAITCNT_ALL=1",
}

# (M, N, K, suffix, current_best_tag, current_best_tflops, comp_tflops)
TARGETS = [
    # Shape 27
    (14336, 4096, 32768, "_ts_lgk2_gm7_memc_pfoff124_kx32768_btw_all",
     "ts_pf4_memc_btw_step3", 4965.3, 5245.4),
    # Shape 12
    (4096, 14336, 16384, "_ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",
     "ts_lgk2_memc_btw_all", 4870.3, 5013.0),
    # Shape 18
    (4096, 28672, 32768, "_ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all",
     "ts_pf4_memc_btw_step3", 5450.9, 5649.9),
    # Shape 21
    (4096, 32768, 14336, "_ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all",
     "ts_lgk2_memc_btw_all", 5240.0, 5296.1),
]


def module_name_for_nk(n, k):
    return f"tk_mxfp4_gluon_cpp_n{n}_k{k}"


def build_for_nk(n_dim, k_dim, suffix, cppflags):
    module_name = module_name_for_nk(n_dim, k_dim) + suffix
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(BUILD_DIR, out_file)
    if os.path.exists(out_path):
        print(f"  [cached] {module_name}")
        return out_path

    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n_dim}_k{k_dim}{suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    print(f"  Compiling {module_name} ...", end=" ", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT
    cmd = (
        f'make -C {SCRIPT_DIR} TARGET={os.path.join(BUILD_DIR, module_name)} '
        f'SRC={wrapper_src} '
        f'CPPFLAGS="-DK_DIM={k_dim} -DN_DIM={n_dim} {cppflags}"'
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0
    if r.returncode != 0 or not os.path.exists(out_path):
        print(f"FAILED ({elapsed:.1f}s)")
        print(r.stderr[-2000:])
        return None
    print(f"OK ({elapsed:.1f}s)")
    return out_path


def bench_variant(gpu_id, m, n, k, module_suffix, reps=5):
    """Returns list of trimmed-mean TFLOPS, one per rep."""
    module_name = module_name_for_nk(n, k) + module_suffix
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return None

    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS, TRIM = 200, 500, 0.10
M, N, K = {m}, {n}, {k}
def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')<<4)|torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')
def preshuffle(se):
    r,kb=se.shape; pr=math.ceil(r/64)*64; pk=math.ceil(kb/8)*8
    raw=torch.full((pr,pk),0x7F,dtype=torch.uint8,device=se.device)
    raw[:r,:kb]=(se.to(torch.int16)+127).to(torch.uint8)
    sh=raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh=sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)
spec=importlib.util.spec_from_file_location('{module_name}','{so_path}')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
A=gen_fp4(M,K); B=gen_fp4(N,K)
sc_a=torch.randint(-2,3,(M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.randint(-2,3,(N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
run=lambda:mod.gemm_rcr(A,B,A_sc,B_sc,C)
for _ in range(WARMUP): run()
torch.cuda.synchronize()
times=[]
for _ in range(ITERS):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort(); trim=int(len(times)*TRIM)
times=times[trim:-trim] if trim>0 else times
avg=sum(times)/len(times); t=2.0*M*N*K/(avg*1e-3)/1e12
print(json.dumps({{"tflops":round(t,2),"ms":round(avg,4)}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    results = []
    for rep in range(reps):
        try:
            r = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, timeout=600, env=env)
            if r.returncode != 0:
                print(f"    rep{rep+1}: FAILED rc={r.returncode}")
                print(r.stderr[-500:])
                continue
            d = json.loads(r.stdout.strip())
            results.append(d["tflops"])
            print(f"    rep{rep+1}: {d['tflops']:.2f} TFLOPS")
        except Exception as e:
            print(f"    rep{rep+1}: EXCEPTION {e}")
    return results


def stats(xs):
    if not xs: return (0, 0, 0)
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = math.sqrt(var)
    return (mean, std, std / mean * 100 if mean else 0)


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"=== R26-D AUDIT — GPU {gpu} ===")
    print(f"warmup=200 iters=500 trim=0.10 reps=5\n")

    # Phase 1: build all missing variants
    print("--- BUILD PHASE ---")
    built = {}
    for m, n, k, suffix, _, _, _ in TARGETS:
        flags = VARIANTS[suffix]
        out = build_for_nk(n, k, suffix, flags)
        built[(n, k, suffix)] = out
    print()

    # Phase 2: benchmark
    print("--- BENCH PHASE ---")
    findings = []
    for m, n, k, suffix, current_best_tag, current_best_tflops, comp in TARGETS:
        print(f"\n>>> Shape M={m} N={n} K={k} (comp={comp})")
        print(f"  Current best (FINAL): {current_best_tag} = {current_best_tflops:.1f} TFLOPS")
        print(f"  Testing R25-G K_EXACT: {suffix}")
        if built[(n, k, suffix)] is None:
            print("  BUILD FAILED — skip")
            continue
        runs = bench_variant(gpu, m, n, k, suffix, reps=5)
        if not runs:
            print("  ALL REPS FAILED")
            continue
        mean, std, std_pct = stats(runs)
        delta = mean - current_best_tflops
        delta_pp = delta / current_best_tflops * 100
        ratio_vs_comp = mean / comp * 100
        print(f"  R25-G K_EXACT mean: {mean:.2f} TFLOPS  std: {std:.2f} ({std_pct:.2f}%)")
        print(f"  Δ vs FINAL best: {delta:+.2f} TFLOPS ({delta_pp:+.2f}%)")
        print(f"  vs comp: {ratio_vs_comp:.1f}%")
        findings.append({
            "shape": (m, n, k),
            "current_best": (current_best_tag, current_best_tflops),
            "r25g_kexact_mean": round(mean, 2),
            "r25g_kexact_std_pct": round(std_pct, 2),
            "delta_pp": round(delta_pp, 2),
            "comp": comp,
            "ratio_vs_comp": round(ratio_vs_comp, 1),
            "would_win": mean >= comp,
            "reps": runs,
        })

    print("\n=== SUMMARY ===")
    for f in findings:
        m, n, k = f["shape"]
        print(f"  M={m} N={n} K={k}: R25-G K_EXACT {f['r25g_kexact_mean']:.1f} "
              f"vs FINAL {f['current_best'][1]:.1f} ({f['delta_pp']:+.2f}%)  "
              f"ratio_comp {f['ratio_vs_comp']}%  "
              f"WIN={f['would_win']}")

    out_path = os.path.join(SCRIPT_DIR, "r26d_audit_shape27_results.json")
    with open(out_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "findings": findings}, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
