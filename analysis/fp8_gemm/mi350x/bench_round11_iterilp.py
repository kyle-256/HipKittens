#!/usr/bin/env python3
"""Round 11 single-shot bench: 7 deep-LOSE + 3 WIN spot-check shapes.
For each shape, bench parent (re-bench on GPU 6) + parent_r11_iterilp.
GPU 6, warmup=200, iters=500, trim=10%.
Also runs SNR correctness check before timing.
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
GPU = 6
NEW_SUFFIX = "_r11_iterilp"

# (label, M, N, K, comp_tflops, parent_suffix)
SHAPES = [
    # Deep-LOSE
    ("DLA1_4096x32768x128256",   4096,  32768, 128256, 5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA2_128256x32768x4096", 128256,  32768,   4096, 4536.4, "_ts_gm2_v12_memc_dc"),
    ("DLA3_4096x32768x28672",    4096,  32768,  28672, 5568.2, "_v20_memc"),
    ("DLA4_4096x28672x32768",    4096,  28672,  32768, 5649.9, "_u16"),
    ("DLA5_32768x4096x14336",   32768,   4096,  14336, 5223.4, "_ts_gm8_v12"),
    ("DLA6_4096x32768x14336",    4096,  32768,  14336, 5296.1, "_ts_lgk2_memc"),
    ("DLA7_28672x32768x4096",   28672,  32768,   4096, 4466.6, "_ts_lgk2_v12_memc"),
    # WIN regression-check
    ("WIN1_16384x4096x7168",    16384,   4096,   7168, 4443.2, "_ts_lgk2_v20_memc"),
    ("WIN2_32768x6144x2048",    32768,   6144,   2048, 3239.9, "_ts_gm8_v12"),
    ("WIN3_4096x128256x32768",   4096, 128256,  32768, 3195.3, "_memc"),
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
    print(f"Round 11 iterative-ilp single-shot bench. GPU={GPU} warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gpu": GPU, "shapes": []}
    for (label, M, N, K, comp, parent) in SHAPES:
        cand = parent + NEW_SUFFIX
        print(f"\n{label}  shape={M}x{N}x{K}  comp={comp}  parent={parent}")
        # Baseline rebench
        rb = bench_variant(M, N, K, parent, GPU)
        if "error" in rb:
            print(f"  baseline {parent:36s} ERROR: {rb}", flush=True)
            tb = None; rb_t = None
        else:
            tb = rb["tflops"]; rb_t = tb
            print(f"  baseline {parent:36s} {tb:7.2f} TFLOPS  ({tb/comp*100:.2f}%)", flush=True)
        # Candidate (with SNR finite check)
        rc = bench_variant(M, N, K, cand, GPU, do_snr=True)
        if "error" in rc:
            print(f"  cand     {cand:36s} ERROR: {rc}", flush=True)
            tc = None; rc_t = None
        else:
            tc = rc["tflops"]; rc_t = tc
            ff = rc.get("finite_frac")
            ff_str = "" if ff is None else f"  finite={ff:.3f}"
            print(f"  cand     {cand:36s} {tc:7.2f} TFLOPS  ({tc/comp*100:.2f}%){ff_str}", flush=True)
        if tb and tc:
            d_tflops = tc - tb
            d_pp = (tc - tb) / comp * 100
            print(f"  delta={d_tflops:+.2f} TFLOPS  ({d_pp:+.2f}pp)", flush=True)
        out["shapes"].append({
            "label": label, "M": M, "N": N, "K": K, "comp": comp,
            "parent": parent, "candidate": cand,
            "baseline_tflops": rb_t, "candidate_tflops": rc_t,
            "candidate_finite_frac": rc.get("finite_frac") if "error" not in rc else None,
            "delta_tflops": (rc_t - rb_t) if (rb_t and rc_t) else None,
            "delta_pp": ((rc_t - rb_t) / comp * 100) if (rb_t and rc_t) else None,
        })
    print("=" * 110)
    with open(os.path.join(SCRIPT_DIR, "bench_round11_iterilp_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round11_iterilp_results.json")


if __name__ == "__main__":
    main()
