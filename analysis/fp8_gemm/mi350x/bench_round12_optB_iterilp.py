#!/usr/bin/env python3
"""Round 12 Optimizer B single-shot bench: 3 NEAR-THRESHOLD + 5 MID-LOSE shapes.
For each shape, bench parent (re-bench on GPU 7) + parent_r12_iterilp candidate.
GPU 7, warmup=200, iters=500, trim=10%.
Also runs SNR finite-fraction check before timing the candidate.
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
GPU = 7
NEW_SUFFIX = "_r12_iterilp"

# (label, M, N, K, comp_tflops, parent_suffix)
SHAPES = [
    # NEAR-THRESHOLD LOSE (could flip to WIN)
    ("NT1_32768x4096x7168",     32768,   4096,   7168, 4666.8, "_ts_gm8_v12"),
    ("NT2_4096x14336x16384",     4096,  14336,  16384, 5013.0, "_ts_lgk2"),
    ("NT3_6144x4096x16384",      6144,   4096,  16384, 4428.1, "_ts_lgk2"),
    # MID-LOSE
    ("ML1_16384x28672x2048",    16384,  28672,   2048, 3482.3, "_ts_gm2_v12_memc_dc"),
    ("ML2_4096x32768x6144",      4096,  32768,   6144, 4548.6, "_ts_pf4_memc"),
    ("ML3_16384x28672x4096",    16384,  28672,   4096, 4411.7, "_ts_gm2_v12_memc"),
    ("ML4_28672x4096x8192",     28672,   4096,   8192, 4810.0, "_ts_lgk2_memc_dc"),
    ("ML5_14336x32768x4096",    14336,  32768,   4096, 4462.6, "_ts_v12_tv0_memc"),
]


def bench_variant(M, N, K, full_suffix, gpu_id, do_snr=False):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so", "path": so_path}
    snr_block = ""
    if do_snr:
        snr_block = """
# SNR sanity: small random inputs, check output finite
C2 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
run2 = lambda: mod.gemm_rcr(A, B, A_sc, B_sc, C2)
run2(); torch.cuda.synchronize()
finite_frac = float(torch.isfinite(C2.float()).float().mean().item())
import json
print('SNR_FINITE_FRAC=' + str(finite_frac))
"""
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
M, N, K = {M}, {N}, {K}
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
{snr_block}
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
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        finite = None
        for ln in r.stdout.splitlines():
            if ln.startswith("SNR_FINITE_FRAC="):
                finite = float(ln.split("=", 1)[1])
        last = r.stdout.strip().splitlines()[-1]
        d = json.loads(last)
        if finite is not None:
            d["finite_frac"] = finite
        return d
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"Round 12 OptB iterative-ilp single-shot bench. GPU={GPU} warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gpu": GPU, "shapes": []}
    for (label, M, N, K, comp, parent) in SHAPES:
        cand = parent + NEW_SUFFIX
        print(f"\n{label}  shape={M}x{N}x{K}  comp={comp}  parent={parent}")
        # Baseline rebench on GPU 7
        rb = bench_variant(M, N, K, parent, GPU)
        if "error" in rb:
            print(f"  baseline {parent:38s} ERROR: {rb}", flush=True)
            tb = None; rb_t = None
        else:
            tb = rb["tflops"]; rb_t = tb
            print(f"  baseline {parent:38s} {tb:7.2f} TFLOPS  ({tb/comp*100:.2f}%)", flush=True)
        # Candidate (with SNR finite check)
        rc = bench_variant(M, N, K, cand, GPU, do_snr=True)
        if "error" in rc:
            print(f"  cand     {cand:38s} ERROR: {rc}", flush=True)
            tc = None; rc_t = None
        else:
            tc = rc["tflops"]; rc_t = tc
            ff = rc.get("finite_frac")
            ff_str = "" if ff is None else f"  finite={ff:.3f}"
            print(f"  cand     {cand:38s} {tc:7.2f} TFLOPS  ({tc/comp*100:.2f}%){ff_str}", flush=True)
        if tb and tc:
            d_tflops = tc - tb
            d_pp = (tc - tb) / comp * 100
            cand_pct = tc / comp * 100
            base_pct = tb / comp * 100
            flip_marker = "  *** FLIPS TO WIN ***" if (cand_pct >= 100.0 and base_pct < 100.0) else ""
            print(f"  delta={d_tflops:+.2f} TFLOPS  ({d_pp:+.2f}pp){flip_marker}", flush=True)
        out["shapes"].append({
            "label": label, "M": M, "N": N, "K": K, "comp": comp,
            "parent": parent, "candidate": cand,
            "baseline_tflops": rb_t, "candidate_tflops": rc_t,
            "candidate_finite_frac": rc.get("finite_frac") if "error" not in rc else None,
            "candidate_error": rc.get("error") if "error" in rc else None,
            "delta_tflops": (rc_t - rb_t) if (rb_t and rc_t) else None,
            "delta_pp": ((rc_t - rb_t) / comp * 100) if (rb_t and rc_t) else None,
            "candidate_pct": (rc_t / comp * 100) if rc_t else None,
            "baseline_pct": (rb_t / comp * 100) if rb_t else None,
        })
    print("=" * 110)
    with open(os.path.join(SCRIPT_DIR, "bench_round12_optB_iterilp_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round12_optB_iterilp_results.json")


if __name__ == "__main__":
    main()
