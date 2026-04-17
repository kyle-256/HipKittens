#!/usr/bin/env python3
"""Round 14 OptC DLA1 aperture smoke. warmup=20 iters=50, 3 retries on failure.
GPU 3 only.

Tests only DIFF variants from asm_diff_probe_r14c.json (skips silent no-ops).
Detects HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (the iterilp SGPR clobber bug
class — Round 13A confirmed it generalizes to non-iterilp scheduler swaps).

Marks: OK / FLAKY / BROKEN / SKIP_NOOP.
"""
import json, math, os, subprocess, sys, sysconfig, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 20
ITERS = 50
TRIM = 0.10
GPU = 3

M, N, K = 4096, 32768, 128256
PARENT_SUFFIX = "_ts_pf6_6_v12_memc"
COMP_TFLOPS = 5781.1  # aiter

# Load probe results to skip NOOPs
PROBE_PATH = os.path.join(SCRIPT_DIR, "asm_diff_probe_r14c.json")
with open(PROBE_PATH) as f:
    PROBE = json.load(f)


def smoke_one(full_suffix, gpu_id):
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
                           capture_output=True, text=True, timeout=400, env=env)
        if r.returncode != 0:
            err = r.stderr[-500:]
            tag = "APERTURE" if "APERTURE" in r.stderr else ("RC" + str(r.returncode))
            return {"error": tag, "stderr": err}
        last = r.stdout.strip().splitlines()[-1]
        return json.loads(last)
    except Exception as e:
        return {"error": str(e)}


def smoke_with_retry(full_suffix, gpu_id, retries=3):
    fails = 0
    last = None
    for i in range(retries):
        r = smoke_one(full_suffix, gpu_id)
        last = r
        if "error" in r:
            fails += 1
            continue
        r["retries_used"] = i
        r["status"] = "OK"
        return r
    last["status"] = "BROKEN"
    last["fails"] = fails
    return last


def main():
    print(f"Round 14 OptC DLA1 smoke. GPU={GPU} warmup={WARMUP} iters={ITERS}")
    print(f"Shape {M}x{N}x{K}  parent={PARENT_SUFFIX}  comp_tflops={COMP_TFLOPS}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "gpu": GPU, "M": M, "N": N, "K": K,
           "parent_suffix": PARENT_SUFFIX, "comp_tflops": COMP_TFLOPS, "results": []}

    # Always re-bench parent fresh
    print(f"\nParent baseline:")
    r = smoke_with_retry(PARENT_SUFFIX, GPU)
    if r.get("status") == "OK":
        print(f"  {PARENT_SUFFIX:50s} OK   tflops={r['tflops']:.2f}  retries={r.get('retries_used',0)}", flush=True)
    else:
        print(f"  {PARENT_SUFFIX:50s} BROKEN err={r.get('error')}", flush=True)
    out["results"].append({"variant": PARENT_SUFFIX, "kind": "parent", **r})

    # Iterate variants in stable order
    print("\nCandidates (DIFF only):")
    for full_suffix, info in PROBE["variants"].items():
        suffix = full_suffix[len(PARENT_SUFFIX):]
        if info["verdict"] == "NOOP":
            print(f"  {full_suffix:50s} SKIP_NOOP", flush=True)
            out["results"].append({"variant": full_suffix, "kind": "candidate",
                                    "status": "SKIP_NOOP"})
            continue
        if info["verdict"] != "DIFF":
            print(f"  {full_suffix:50s} SKIP_{info['verdict']}", flush=True)
            out["results"].append({"variant": full_suffix, "kind": "candidate",
                                    "status": "SKIP_" + info["verdict"]})
            continue
        r = smoke_with_retry(full_suffix, GPU)
        if r.get("status") == "OK":
            ratio = r["tflops"] / COMP_TFLOPS * 100
            print(f"  {full_suffix:50s} OK   tflops={r['tflops']:.2f}  ({ratio:.2f}% aiter)  retries={r.get('retries_used',0)}", flush=True)
        else:
            print(f"  {full_suffix:50s} BROKEN  fails={r.get('fails')} err={r.get('error')}", flush=True)
        out["results"].append({"variant": full_suffix, "kind": "candidate", **r})

    print("=" * 110)
    with open(os.path.join(SCRIPT_DIR, "bench_round14_optC_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round14_optC_smoke.json")


if __name__ == "__main__":
    main()
