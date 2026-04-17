#!/usr/bin/env python3
"""R17B single-shot smoke. P1 (28672x4096x16384) parent _ts_gm8 vs each
of 10 triple/quad-stack candidates. Side-by-side same GPU.

warmup=200 iters=500 trim=10%. Gate: candidate >= base + 0.5pp.
"""
import json, os, subprocess, sys, sysconfig
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10

# P1: 28672x4096x16384, comp baseline 5273.0 TFLOPS, parent _ts_gm8
SHAPE = ("P1", 28672, 4096, 16384, 5273.0, "_ts_gm8")

# Per agent prompt: GPUs 2,3,4 (R17A on rocprof + 8-GPU rebaseline; R17C on remaining)
GPU_POOL = [2, 3, 4]
NEW_PREFIX = "_r17b_"

COMPOUNDS = [
    ("rcg", "noemxpre", "tv16"),
    ("rcg", "noemxpre", "v20"),
    ("rcg", "noemxpre", "lgk2"),
    ("rcg", "noemxpre", "extbr"),
    ("rcg", "tv16", "v20"),
    ("rcg", "tv16", "lgk2"),
    ("rcg", "tv16", "extbr"),
    ("rcg", "noemxpre", "tv16", "v20"),
    ("rcg", "noemxpre", "tv16", "lgk2"),
    ("rcg", "noemxpre", "tv16", "extbr"),
]


def bench_one(M, N, K, full_suffix, gpu_id):
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
print(json.dumps({{"tflops":round(t,2),"ms":round(avg,4)}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=1500, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def task(args):
    tags, csuf, gpu = args
    lab, M, N, K, comp, ps = SHAPE
    base = bench_one(M, N, K, ps, gpu)
    cand = bench_one(M, N, K, ps + csuf, gpu)
    return tags, csuf, gpu, comp, base, cand


def main():
    pairs = []
    for i, tags in enumerate(COMPOUNDS):
        csuf = NEW_PREFIX + "X".join(tags)
        gpu = GPU_POOL[i % len(GPU_POOL)]
        pairs.append((tags, csuf, gpu))

    print(f"R17B SMOKE  warmup={WARMUP} iters={ITERS} trim={TRIM}  GPUs={GPU_POOL}")
    print(f"P1: 28672x4096x16384  parent=_ts_gm8  comp_baseline=5273.0 TFLOPS")
    print(f"Total candidates: {len(pairs)}")
    print("=" * 110)

    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gpus": GPU_POOL,
           "shape": SHAPE[0], "M_N_K": [SHAPE[1], SHAPE[2], SHAPE[3]],
           "comp_baseline": SHAPE[4], "parent_suffix": SHAPE[5],
           "pairs": []}
    passes = []
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(task, a): a for a in pairs}
        for fut in as_completed(futs):
            tags, csuf, gpu, comp, base, cand = fut.result()
            tag_label = "+".join(tags)
            if "error" in base or "error" in cand:
                print(f"  {tag_label:48s} gpu={gpu}  ERROR base={base} cand={cand}", flush=True)
                out["pairs"].append({"tag":tag_label,"csuf":csuf,"gpu":gpu,
                                     "base":base,"cand":cand})
                continue
            bt = base["tflops"]; ct = cand["tflops"]
            d = ct - bt
            pp = d / comp * 100
            gate = pp >= 0.5
            print(f"  {tag_label:48s} gpu={gpu}  base={bt:7.1f}  cand={ct:7.1f}  d={d:+6.1f}  ({pp:+.2f}pp)  {'PASS' if gate else 'fail'}", flush=True)
            out["pairs"].append({"tag":tag_label,"csuf":csuf,"gpu":gpu,
                                 "comp":comp,"base_tflops":bt,"cand_tflops":ct,
                                 "delta_tflops":round(d,2),"delta_pp":round(pp,3),
                                 "gate_pass":gate})
            if gate:
                passes.append((tag_label, csuf, pp))
    out["smoke_passes"] = [{"tag":t,"csuf":c,"delta_pp":round(p,3)}
                            for t,c,p in passes]
    with open(os.path.join(SCRIPT_DIR, "bench_round17_optB_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 110)
    print(f"Smoke PASS count: {len(passes)}")
    for t,c,p in passes:
        print(f"  {t}  +{p:.2f}pp")
    print("Saved bench_round17_optB_smoke.json")


if __name__ == "__main__":
    main()
