#!/usr/bin/env python3
"""Round 13 OptC single-shot bench. warmup=200 iters=500 trim=10%, GPU 7.
Re-benchmarks each shape's current best (iterilp WIN) AS WELL AS the surviving
candidates. Gate: candidate must be >= current_best + 0.5pp (using comp as denom).
Skips BROKEN candidates per smoke results.
Computes SNR (sanity) for each candidate.
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
GPU = 7

# (label, M, N, K, parent_suffix, current_best_suffix, comp_tflops)
SHAPES = [
    ("S1_14336x4096x32768", 14336, 4096, 32768, "_v16_wpe2",     "_v16_wpe2_r10a_iterilp",     5245.4),
    ("S2_16384x4096x28672", 16384, 4096, 28672, "_u8",           "_u8_r10a_iterilp",           5525.3),
    ("S3_4096x32768x28672", 4096, 32768, 28672, "_v20_memc",     "_v20_memc_r11_iterilp",      5568.2),
    ("S4_4096x28672x32768", 4096, 28672, 32768, "_u16",          "_u16_r11_iterilp",           5649.9),
    ("S5_4096x32768x14336", 4096, 32768, 14336, "_ts_lgk2_memc", "_ts_lgk2_memc_r11_iterilp",  5296.1),
]

# Surviving candidates per shape (from smoke). iterminreg BROKEN on 4/5 shapes
# — we run only the survivors per smoke to avoid wasted time.
SURVIVING = {
    "S1_14336x4096x32768": ["_r13c_itermaxocc", "_r13c_iterilp_brlgk4", "_r13c_iterilp_v24"],
    "S2_16384x4096x28672": ["_r13c_itermaxocc", "_r13c_iterilp_brlgk4", "_r13c_iterilp_v24"],
    "S3_4096x32768x28672": ["_r13c_itermaxocc", "_r13c_iterilp_brlgk4", "_r13c_iterilp_v24"],
    "S4_4096x28672x32768": ["_r13c_itermaxocc", "_r13c_iterilp_brlgk4", "_r13c_iterilp_v24"],
    "S5_4096x32768x14336": ["_r13c_iterminreg", "_r13c_itermaxocc",
                             "_r13c_iterilp_brlgk4", "_r13c_iterilp_v24"],
}


def bench(M, N, K, full_suffix, gpu_id, do_snr=False):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so", "path": so_path}
    snr_block = ""
    if do_snr:
        snr_block = """
finite_frac = float(torch.isfinite(C.float()).float().mean().item())
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
run(); torch.cuda.synchronize()
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
            tag = "APERTURE" if "APERTURE" in r.stderr else f"rc={r.returncode}"
            return {"error": tag, "stderr": r.stderr[-300:]}
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
    print(f"Round 13 OptC single-shot. GPU={GPU} warmup={WARMUP} iters={ITERS}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "gpu": GPU, "shapes": []}
    for (label, M, N, K, parent, cur_best, comp) in SHAPES:
        print(f"\n{label}  shape={M}x{N}x{K}  comp={comp}")
        # Re-bench current best fresh on this GPU
        r_best = bench(M, N, K, cur_best, GPU)
        if "error" in r_best:
            print(f"  current_best  {cur_best:42s} ERROR: {r_best.get('error')}", flush=True)
            best_t = None
        else:
            best_t = r_best["tflops"]
            print(f"  current_best  {cur_best:42s} {best_t:7.2f} TFLOPS  ({best_t/comp*100:.2f}%)", flush=True)
        shape_out = {"label": label, "M": M, "N": N, "K": K, "comp": comp,
                     "current_best_suffix": cur_best,
                     "current_best_tflops": best_t,
                     "candidates": []}
        # Each surviving candidate
        for suf in SURVIVING[label]:
            cand_full = parent + suf
            r = bench(M, N, K, cand_full, GPU, do_snr=True)
            if "error" in r:
                print(f"  cand          {cand_full:42s} ERROR: {r.get('error')}", flush=True)
                shape_out["candidates"].append({"suffix": cand_full, "error": r.get("error"),
                                                 "stderr": r.get("stderr","")[:200]})
                continue
            t = r["tflops"]; ff = r.get("finite_frac")
            d_pp = ((t - best_t) / comp * 100) if best_t else None
            ff_str = f"  finite={ff:.3f}" if ff is not None else ""
            d_str = f"  delta={t - best_t:+.2f} TFLOPS  ({d_pp:+.2f}pp)" if best_t else ""
            print(f"  cand          {cand_full:42s} {t:7.2f} TFLOPS  ({t/comp*100:.2f}%){d_str}{ff_str}", flush=True)
            shape_out["candidates"].append({
                "suffix": cand_full, "tflops": t, "finite_frac": ff,
                "delta_tflops": (t - best_t) if best_t else None,
                "delta_pp": d_pp,
                "ratio_pct": round(t/comp*100, 2),
            })
        out["shapes"].append(shape_out)
    print("=" * 110)
    with open(os.path.join(SCRIPT_DIR, "bench_round13_optC_singleshot.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round13_optC_singleshot.json")


if __name__ == "__main__":
    main()
