#!/usr/bin/env python3
"""R26-D AUDIT round 2: test newly-discovered LOSE shapes against K_EXACT variants."""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

def module_name_for_nk(n, k):
    return f"tk_mxfp4_gluon_cpp_n{n}_k{k}"

# (M, N, K, suffix, current_best_tag, current_best_tflops, comp)
TARGETS = [
    # Shape 22: K=28672 — needs the K_EXACT variant just built
    (4096, 32768, 28672, "_ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all",
     "ts_gm8_v12_btw_step3", 5272.0, 5568.2),
    # Shape 20: K=6144 — has _ts_v12_gm7_memc_pfoff19_kx6144_btw_all built
    (4096, 32768, 6144, "_ts_v12_gm7_memc_pfoff19_kx6144_btw_all",
     "ts_gm2_v12_memc_btw_all", 4532.9, 4548.6),
    # Shape 32: K=14336 — needs _ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all? K=14336 not 16384
    # Actually for K=14336 there's _ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all
    (16384, 4096, 14336, "_ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all",
     "ts_u16", 5032.7, 5142.1),
]


def bench_variant(gpu_id, m, n, k, module_suffix, reps=5):
    module_name = module_name_for_nk(n, k) + module_suffix
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        print(f"  MISSING: {so_path}")
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
                print(f"    rep{rep+1}: FAIL rc={r.returncode}")
                print(r.stderr[-500:])
                continue
            d = json.loads(r.stdout.strip())
            results.append(d["tflops"])
            print(f"    rep{rep+1}: {d['tflops']:.2f} TFLOPS")
        except Exception as e:
            print(f"    rep{rep+1}: EXC {e}")
    return results


def stats(xs):
    if not xs: return (0, 0, 0)
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = math.sqrt(var)
    return (mean, std, std / mean * 100 if mean else 0)


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"=== R26-D AUDIT ROUND 2 — GPU {gpu} ===")
    findings = []
    for m, n, k, suffix, cb_tag, cb_tflops, comp in TARGETS:
        print(f"\n>>> M={m} N={n} K={k}")
        print(f"  Current FINAL best: {cb_tag} = {cb_tflops:.1f} TFLOPS (comp={comp})")
        runs = bench_variant(gpu, m, n, k, suffix, reps=5)
        if not runs:
            findings.append({"shape": (m,n,k), "status": "NO_BUILD"})
            continue
        mean, std, std_pct = stats(runs)
        delta_pp = (mean - cb_tflops) / cb_tflops * 100
        ratio = mean / comp * 100
        print(f"  R25-G K_EXACT: mean={mean:.2f} std={std:.2f} ({std_pct:.2f}%)")
        print(f"  Δ vs FINAL best: {delta_pp:+.2f}%, vs comp: {ratio:.1f}%")
        findings.append({
            "shape": (m,n,k), "suffix": suffix, "mean": round(mean,2),
            "std_pct": round(std_pct,2), "delta_pp": round(delta_pp,2),
            "ratio_comp": round(ratio,1),
            "would_win": mean >= comp, "reps": runs,
        })

    print("\n=== SUMMARY ===")
    for f in findings:
        print(f, flush=True)

    out = os.path.join(SCRIPT_DIR, "r26d_audit_more_results.json")
    with open(out, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "findings": findings}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
