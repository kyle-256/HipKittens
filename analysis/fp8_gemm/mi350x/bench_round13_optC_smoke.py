#!/usr/bin/env python3
"""Round 13 OptC aperture smoke. warmup=20 iters=50, 3 retries on failure.
Detects HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (the iterilp SGPR clobber bug).
Marks each variant: OK / FLAKY (failed once) / BROKEN (failed all 3).
GPU 7 only.
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 20
ITERS = 50
TRIM = 0.10
GPU = 7

# (label, M, N, K, parent_suffix, comp_tflops)
SHAPES = [
    ("S1_14336x4096x32768", 14336, 4096, 32768, "_v16_wpe2",     5245.4),
    ("S2_16384x4096x28672", 16384, 4096, 28672, "_u8",           5525.3),
    ("S3_4096x32768x28672", 4096, 32768, 28672, "_v20_memc",     5568.2),
    ("S4_4096x28672x32768", 4096, 28672, 32768, "_u16",          5649.9),
    ("S5_4096x32768x14336", 4096, 32768, 14336, "_ts_lgk2_memc", 5296.1),
]

STACK_SUFFIXES = [
    "_r13c_iterminreg",
    "_r13c_itermaxocc",
    "_r13c_iterilp_brlgk4",
    "_r13c_iterilp_v24",
]


def smoke_one(M, N, K, full_suffix, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so", "path": so_path}
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
finite_frac = float(torch.isfinite(C.float()).float().mean().item())
print(json.dumps({{"tflops":round(t,2),"ms":round(avg,4),"finite_frac":finite_frac}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            err = r.stderr[-400:]
            tag = "APERTURE" if "APERTURE" in r.stderr else ("RC" + str(r.returncode))
            return {"error": tag, "stderr": err}
        last = r.stdout.strip().splitlines()[-1]
        return json.loads(last)
    except Exception as e:
        return {"error": str(e)}


def smoke_with_retry(M, N, K, full_suffix, gpu_id, retries=3):
    fails = 0
    last = None
    for i in range(retries):
        r = smoke_one(M, N, K, full_suffix, gpu_id)
        last = r
        if "error" in r:
            fails += 1
            continue
        # success → return immediately with result + fails-so-far
        r["retries_used"] = i
        r["status"] = "OK"
        return r
    # all failed
    last["status"] = "BROKEN"
    last["fails"] = fails
    return last


def main():
    print(f"Round 13 OptC smoke. GPU={GPU} warmup={WARMUP} iters={ITERS}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "gpu": GPU, "results": []}
    for (label, M, N, K, parent, comp) in SHAPES:
        # First: re-bench parent's R10/R11 iterilp variant fresh (regression-protect baseline).
        # Map parent → existing iterilp suffix:
        if parent == "_v16_wpe2":
            current_best_suffix = parent + "_r10a_iterilp"
        elif parent == "_u8":
            current_best_suffix = parent + "_r10a_iterilp"
        else:
            current_best_suffix = parent + "_r11_iterilp"
        print(f"\n{label}  shape={M}x{N}x{K}  parent={parent}  current_best={current_best_suffix}")
        r = smoke_with_retry(M, N, K, current_best_suffix, GPU)
        if r.get("status") == "OK":
            print(f"  current_best  {current_best_suffix:42s} OK   tflops={r['tflops']:.2f}", flush=True)
        else:
            print(f"  current_best  {current_best_suffix:42s} {r.get('status','?')}  err={r.get('error')}", flush=True)
        out["results"].append({"label": label, "variant": current_best_suffix, "kind": "current_best", **r})
        # Then each new stack candidate
        for suf in STACK_SUFFIXES:
            full = parent + suf
            r = smoke_with_retry(M, N, K, full, GPU)
            if r.get("status") == "OK":
                print(f"  cand          {full:42s} OK   tflops={r['tflops']:.2f}  retries={r.get('retries_used',0)}", flush=True)
            else:
                print(f"  cand          {full:42s} BROKEN  fails={r.get('fails')} err={r.get('error')}", flush=True)
            out["results"].append({"label": label, "variant": full, "kind": "candidate", **r})
    print("=" * 110)
    with open(os.path.join(SCRIPT_DIR, "bench_round13_optC_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round13_optC_smoke.json")


if __name__ == "__main__":
    main()
