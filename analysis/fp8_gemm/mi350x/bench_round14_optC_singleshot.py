#!/usr/bin/env python3
"""Round 14 OptC DLA1 single-shot bench. warmup=200 iters=500, trim=10%.

Tests parent + smoke survivors that came within ~10 TFLOPS of parent or beat it.
GPU 3 only.
"""
import json, math, os, subprocess, sys, sysconfig, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
GPU = 3

M, N, K = 4096, 32768, 128256
PARENT_SUFFIX = "_ts_pf6_6_v12_memc"
COMP_TFLOPS = 5781.1

# Top smoke candidates (within ~30 TFLOPS of parent + parent)
CANDIDATES = [
    PARENT_SUFFIX,                              # parent baseline
    PARENT_SUFFIX + "_r14c_v3_brlgk0",          # 5145.04 smoke
    PARENT_SUFFIX + "_r14c_v2_v16",             # 5142.30
    PARENT_SUFFIX + "_r14c_v2_v20",             # 5139.80
    PARENT_SUFFIX + "_r14c_v2_v8",              # 5137.77
    PARENT_SUFFIX + "_r14c_v2_v24",             # 5135.68
    PARENT_SUFFIX + "_r14c_v4_tbv16",           # 5133.14
    PARENT_SUFFIX + "_r14c_v3_brlgk4",          # 5129.70
    PARENT_SUFFIX + "_r14c_cb_tbv16_brlgk4",    # 5129.44
    PARENT_SUFFIX + "_r14c_cb_gm8_tbv16",       # 5116.20 (lowest of the close set)
    PARENT_SUFFIX + "_r14c_v5_gm8",             # 5116.34
    PARENT_SUFFIX + "_r14c_v4_tbv0",            # 5114.02
    PARENT_SUFFIX + "_r14c_v2_v4",              # 5108.94
    PARENT_SUFFIX + "_r14c_v4_tbv4",            # 5102.44
    PARENT_SUFFIX + "_r14c_v10_static_xcd",     # 5099.72
]


def bench_one(full_suffix, gpu_id, warmup, iters, trim):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so", "path": so_path}
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS, TRIM = {warmup}, {iters}, {trim}
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
                           capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0:
            return {"error": "RC" + str(r.returncode), "stderr": r.stderr[-400:]}
        last = r.stdout.strip().splitlines()[-1]
        return json.loads(last)
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"Round 14 OptC DLA1 single-shot. GPU={GPU} warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print(f"Shape {M}x{N}x{K}  parent={PARENT_SUFFIX}  comp_tflops={COMP_TFLOPS}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gpu": GPU,
           "M": M, "N": N, "K": K, "parent_suffix": PARENT_SUFFIX,
           "comp_tflops": COMP_TFLOPS, "results": []}
    parent_tflops = None
    for full_suffix in CANDIDATES:
        r = bench_one(full_suffix, GPU, WARMUP, ITERS, TRIM)
        if "error" in r:
            print(f"  {full_suffix:50s} ERROR  err={r['error']}", flush=True)
            out["results"].append({"variant": full_suffix, **r})
            continue
        ratio = r["tflops"] / COMP_TFLOPS * 100
        if full_suffix == PARENT_SUFFIX:
            parent_tflops = r["tflops"]
            delta_str = "(parent)"
        else:
            d = r["tflops"] - (parent_tflops or 0)
            dpp = d / COMP_TFLOPS * 100
            delta_str = f"Δ={d:+.2f}T  {dpp:+.3f}pp"
        print(f"  {full_suffix:50s} tflops={r['tflops']:.2f}  ({ratio:.2f}%)  {delta_str}", flush=True)
        out["results"].append({"variant": full_suffix, "ratio": ratio,
                                "delta_vs_parent_tflops": (None if parent_tflops is None else r["tflops"] - parent_tflops),
                                **r})
    print("=" * 110)
    with open(os.path.join(SCRIPT_DIR, "bench_round14_optC_singleshot.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round14_optC_singleshot.json")


if __name__ == "__main__":
    main()
