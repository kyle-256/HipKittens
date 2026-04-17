#!/usr/bin/env python3
"""R14 OptA single-shot bench at warmup=200 iters=500 trim=0.10.
Re-benches parent + each smoke survivor on GPU 1.
Threshold: candidate >= parent + 0.5pp of competitor TFLOPS to advance to verify.
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
SMOKE_JSON = os.path.join(SCRIPT_DIR, "bench_round14_optA_smoke.json")

WARMUP = 200
ITERS = 500
TRIM = 0.10
GPU = 1

# (label, M, N, K, comp_tflops, parent_suffix)
SHAPES = [
    ("DLA1", 4096,  32768, 128256, 5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA2", 128256, 32768,  4096, 4536.4, "_ts_gm2_v12_memc_dc"),
    ("DLA7", 28672, 32768,  4096, 4466.6, "_ts_lgk2_v12_memc"),
    ("P1",   28672,  4096, 16384, 5273.0, "_ts_gm8"),  # competitor estimated from 93.7% ratio + parent
]
SHAPE_BY_LABEL = {s[0]: s for s in SHAPES}


def bench_one(M, N, K, full_suffix, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
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
                           capture_output=True, text=True, timeout=1200, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def main():
    with open(SMOKE_JSON) as f:
        smoke = json.load(f)
    # Group survivors by label
    survivors = {}
    for r in smoke["results"]:
        if r["status"] == "OK":
            survivors.setdefault(r["label"], []).append((r["tag"], r["suffix"]))
    print(f"R14A singleshot. GPU={GPU} warmup={WARMUP} iters={ITERS}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gpu": GPU, "shapes": []}
    for (lab, M, N, K, comp, ps) in SHAPES:
        sl = survivors.get(lab, [])
        if not sl:
            continue
        print(f"\n{lab}  shape={M}x{N}x{K}  comp={comp}  parent={ps}  candidates={len(sl)}")
        # bench parent first
        rb = bench_one(M, N, K, ps, GPU)
        if "error" in rb:
            print(f"  PARENT {ps:36s} ERROR: {rb}", flush=True)
            tb = None
        else:
            tb = rb["tflops"]
            print(f"  PARENT {ps:36s} {tb:7.2f} TFLOPS  ({tb/comp*100:.2f}%)", flush=True)
        shape_results = {"label": lab, "M": M, "N": N, "K": K, "comp": comp,
                        "parent": ps, "parent_tflops": tb, "candidates": []}
        for (tag, fs) in sl:
            rc = bench_one(M, N, K, fs, GPU)
            if "error" in rc:
                print(f"  {tag:14s} ERROR: {rc}", flush=True)
                cand = {"tag": tag, "suffix": fs, "tflops": None, "error": rc}
            else:
                tc = rc["tflops"]
                if tb:
                    d_tflops = tc - tb
                    d_pp = (tc - tb) / comp * 100
                    advance = "★" if d_pp >= 0.5 else (" " if d_pp >= 0 else "")
                    print(f"  {tag:14s}  {tc:7.2f} TFLOPS  ({tc/comp*100:.2f}%)  "
                          f"d={d_tflops:+6.2f} ({d_pp:+.2f}pp) {advance}", flush=True)
                else:
                    print(f"  {tag:14s}  {tc:7.2f} TFLOPS", flush=True)
                cand = {"tag": tag, "suffix": fs, "tflops": tc,
                       "delta_tflops": (tc - tb) if tb else None,
                       "delta_pp": ((tc - tb) / comp * 100) if tb else None}
            shape_results["candidates"].append(cand)
        out["shapes"].append(shape_results)
    with open(os.path.join(SCRIPT_DIR, "bench_round14_optA_singleshot.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 110)
    # Emit advance list
    print("Advance list (>= +0.5pp single-shot):")
    for s in out["shapes"]:
        for c in s["candidates"]:
            if c.get("delta_pp") and c["delta_pp"] >= 0.5:
                print(f"  {s['label']:5s} {c['tag']:14s} +{c['delta_pp']:.2f}pp  ({c['tflops']:.2f} vs {s['parent_tflops']:.2f})")
    print("Saved bench_round14_optA_singleshot.json")


if __name__ == "__main__":
    main()
