#!/usr/bin/env python3
"""R16C single-shot smoke. For each (shape, compound) that passed SNR safety,
re-bench parent + candidate side-by-side on the same GPU.

warmup=200 iters=500 trim=10%. Gate to advance to verify: candidate >= base + 0.5pp.
"""
import json, os, subprocess, sys, sysconfig
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10

# Shape -> (M, N, K, comp_baseline, parent_suffix)
SHAPE_INFO = {
    "DLA1": (4096,  32768, 128256, 5781.1, "_ts_pf6_6_v12_memc"),
    "DLA2": (128256, 32768,  4096, 4536.4, "_ts_gm2_v12_memc_dc"),
    "DLA7": (28672, 32768,  4096, 4466.6, "_ts_lgk2_v12_memc"),
    "P1":   (28672,  4096, 16384, 5273.0, "_ts_gm8"),
}

GPU_POOL = [5, 6, 7]
NEW_PREFIX = "_r16c_"


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
    shape, tag_label, csuf, gpu = args
    M, N, K, comp, ps = SHAPE_INFO[shape]
    parent_suffix = ps
    cand_suffix = ps + csuf
    base = bench_one(M, N, K, parent_suffix, gpu)
    cand = bench_one(M, N, K, cand_suffix, gpu)
    return shape, tag_label, csuf, gpu, comp, base, cand


def main():
    snr = json.load(open(os.path.join(SCRIPT_DIR, "snr_probe_r16c.json")))
    pairs = []
    for shape, info in snr["shapes"].items():
        for tag_label, ent in info["compounds"].items():
            if ent["verdict"] in ("OK", "OK-SUBSAMPLE"):
                csuf = NEW_PREFIX + "X".join(tag_label.split("+"))
                pairs.append((shape, tag_label, csuf))

    print(f"R16C SMOKE  warmup={WARMUP} iters={ITERS} trim={TRIM}  GPUs={GPU_POOL}")
    print(f"Total pairs to bench: {len(pairs)}")
    print("=" * 110)
    args_list = [(s, t, c, GPU_POOL[i % len(GPU_POOL)])
                 for i, (s, t, c) in enumerate(pairs)]
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gpus": GPU_POOL,
           "pairs": []}
    passes = []
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(task, a): a for a in args_list}
        for fut in as_completed(futs):
            shape, tag_label, csuf, gpu, comp, base, cand = fut.result()
            if "error" in base or "error" in cand:
                print(f"  {shape:6s} {tag_label:48s} gpu={gpu}  ERROR base={base} cand={cand}", flush=True)
                out["pairs"].append({"shape":shape,"tag":tag_label,"csuf":csuf,"gpu":gpu,
                                     "base":base,"cand":cand})
                continue
            bt = base["tflops"]; ct = cand["tflops"]
            d = ct - bt
            pp = d / comp * 100
            gate = pp >= 0.5
            print(f"  {shape:6s} {tag_label:48s} gpu={gpu}  base={bt:7.1f}  cand={ct:7.1f}  d={d:+6.1f}  ({pp:+.2f}pp)  {'PASS' if gate else 'fail'}", flush=True)
            out["pairs"].append({"shape":shape,"tag":tag_label,"csuf":csuf,"gpu":gpu,
                                 "comp":comp,"base_tflops":bt,"cand_tflops":ct,
                                 "delta_tflops":round(d,2),"delta_pp":round(pp,3),
                                 "gate_pass":gate})
            if gate:
                passes.append((shape, tag_label, csuf, pp))
    out["smoke_passes"] = [{"shape":s,"tag":t,"csuf":c,"delta_pp":round(p,3)}
                            for s,t,c,p in passes]
    with open(os.path.join(SCRIPT_DIR, "bench_round16_optC_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 110)
    print(f"Smoke PASS count: {len(passes)}")
    for s,t,c,p in passes:
        print(f"  {s} {t}  +{p:.2f}pp")
    print("Saved bench_round16_optC_smoke.json")


if __name__ == "__main__":
    main()
