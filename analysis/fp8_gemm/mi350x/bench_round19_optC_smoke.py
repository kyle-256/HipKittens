#!/usr/bin/env python3
"""R19C smoke bench: only S5/all passed SNR (OK-MARGINAL @ 40.94 dB vs 59.88 dB noise floor).

Other 14 variants live at-or-below parent noise floor and are excluded per R18A
DLA1/step12 lesson (SNR-marginal can crash under random scales).

Bench: GPUs 6 and 7. warmup=200 iters=500 trim=10%.
For S5: M=4096 N=32768 K=14336, parent _ts_lgk2_memc_r11_iterilp,
candidate _ts_lgk2_memc_r19c_iterilp_btw_all.

Bench protocol:
  1. Pre-flight: run candidate kernel ONCE with FULL random scales [-2,2] to
     catch aperture violations (R18A DLA1/step12 lesson).
  2. If pre-flight OK, do parent rebench + candidate bench on same GPU.
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10

# (tag, M, N, K, comp_tflops, parent_suffix, cand_suffix, gpu)
JOBS = [
    ("S5_4096x32768x14336", 4096, 32768, 14336, 5296.1,
     "_ts_lgk2_memc_r11_iterilp",
     "_ts_lgk2_memc_r19c_iterilp_btw_all",
     6),
]


def aperture_probe(M, N, K, suffix, gpu_id):
    """Run candidate ONCE with full random scales to catch aperture violations."""
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so"}
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(7)
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
mod.gemm_rcr(A,B,A_sc,B_sc,C); torch.cuda.synchronize()
fin = float(torch.isfinite(C.float()).float().mean().item())
print(json.dumps({{"finite":fin}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def bench_one(M, N, K, suffix, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so"}
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
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"R19C smoke bench. warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "shapes": []}
    for (label, M, N, K, comp, parent, cand, gpu) in JOBS:
        print(f"\n{label}  shape={M}x{N}x{K}  comp={comp}  gpu={gpu}")
        print(f"  parent: {parent}")
        print(f"  cand  : {cand}")

        print(f"  [aperture probe with random scales]", flush=True)
        ap = aperture_probe(M, N, K, cand, gpu)
        print(f"    {ap}", flush=True)
        if "error" in ap:
            out["shapes"].append({"label": label, "aperture_error": ap})
            continue

        rb = bench_one(M, N, K, parent, gpu)
        if "error" in rb:
            print(f"  parent bench ERROR: {rb}", flush=True)
            out["shapes"].append({"label": label, "parent_error": rb})
            continue
        tb = rb["tflops"]
        print(f"  parent: {tb:7.2f} TFLOPS  ({tb/comp*100:.2f}%)", flush=True)

        rc = bench_one(M, N, K, cand, gpu)
        if "error" in rc:
            print(f"  cand   bench ERROR: {rc}", flush=True)
            out["shapes"].append({"label": label, "parent_tflops": tb, "cand_error": rc})
            continue
        tc = rc["tflops"]
        d_pp = (tc - tb) / comp * 100
        print(f"  cand  : {tc:7.2f} TFLOPS  ({tc/comp*100:.2f}%)  Δ={d_pp:+.2f}pp", flush=True)
        out["shapes"].append({
            "label": label, "M": M, "N": N, "K": K, "comp": comp, "gpu": gpu,
            "parent_suffix": parent, "cand_suffix": cand,
            "parent_tflops": tb, "cand_tflops": tc, "delta_pp": d_pp,
            "aperture_probe_finite": ap.get("finite"),
        })
    print("=" * 110)
    with open(os.path.join(SCRIPT_DIR, "bench_round19_optC_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round19_optC_smoke.json")


if __name__ == "__main__":
    main()
