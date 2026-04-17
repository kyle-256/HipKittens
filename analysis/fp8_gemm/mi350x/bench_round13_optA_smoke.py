#!/usr/bin/env python3
"""Smoke test (warmup=20, iters=50, 3 retries) for R13 OptA candidates.
For each (shape, strategy), launch in a fresh process 3 times. If any
launch produces an HSA aperture violation / page-fault / non-finite
output, mark that retry FAIL. Final verdict per (shape, strat):
  3/3 OK -> CANDIDATE
  2-/3   -> FLAKY (still report tflops if any OK, else BROKEN)
  0/3    -> BROKEN

Excludes the silent-noop strategies (maxocc, itermaxoccx) and any
(shape, strat) where text_hash matched parent.
"""
import json, os, subprocess, sys, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 20
ITERS = 50
RETRIES = 3
GPU = 5

# (label, M, N, K, comp, parent)
SHAPES = [
    ("DLA1", 4096,  32768, 128256, 5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA2", 128256, 32768,  4096, 4536.4, "_ts_gm2_v12_memc_dc"),
    ("DLA7", 28672, 32768,  4096, 4466.6, "_ts_lgk2_v12_memc"),
    ("WIN2", 32768,  6144,  2048, 3239.9, "_ts_gm8_v12"),
]

# Per ASM-diff probe: skip silent-noop or parent-equivalent variants.
# Load the probe results JSON to determine which (shape, strat) to skip.
def load_skip_set():
    p = os.path.join(SCRIPT_DIR, "asm_diff_probe_r13_results.json")
    if not os.path.exists(p):
        return set()
    d = json.load(open(p))
    skip = set()
    for lab, sd in d["shapes"].items():
        parent_h = sd["parent"]
        for tag, info in sd["strats"].items():
            if info["text_hash"] == parent_h:
                skip.add((lab, tag))
    return skip

# Always skip the universally-noop strategies
ALWAYS_SKIP = {"maxocc", "itermaxoccx"}
# Real candidates per probe: iterminreg, itermaxocc (DLA1 only), maxilp
STRATS = ["iterminreg", "itermaxocc", "maxilp"]


def smoke_one(M, N, K, full_suffix, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so"}
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS = {WARMUP}, {ITERS}
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
finite_frac = float(torch.isfinite(C.float()).float().mean().item())
for _ in range(WARMUP): run()
torch.cuda.synchronize()
times=[]
for _ in range(ITERS):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort(); trim=int(len(times)*0.10)
times=times[trim:-trim] if trim>0 else times
avg=sum(times)/len(times); t=2.0*M*N*K/(avg*1e-3)/1e12
print(json.dumps({{"tflops":round(t,2),"ms":round(avg,4),"finite_frac":finite_frac}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            err = r.stderr[-300:]
            kind = "CRASH"
            if "APERTURE_VIOLATION" in err or "Memory access fault" in err or "read-only" in err:
                kind = "APERTURE"
            return {"error": kind, "stderr": err}
        last = r.stdout.strip().splitlines()[-1]
        d = json.loads(last)
        return d
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT"}
    except Exception as e:
        return {"error": "EXC", "msg": str(e)}


def main():
    skip = load_skip_set()
    print(f"Skipping (parent-equivalent): {sorted(skip)}")
    print(f"Always-skip strategies: {ALWAYS_SKIP}")
    out = {"warmup": WARMUP, "iters": ITERS, "retries": RETRIES, "gpu": GPU,
           "results": []}
    for (lab, M, N, K, comp, ps) in SHAPES:
        for tag in STRATS:
            if tag in ALWAYS_SKIP:
                continue
            if (lab, tag) in skip:
                print(f"\n{lab} {tag}: SKIP (parent-equivalent)")
                out["results"].append({"label": lab, "strat": tag,
                    "verdict": "PARENT_EQUIV", "tflops_runs": []})
                continue
            full = ps + f"_r13_{tag}"
            print(f"\n{lab} {tag} ({full}):")
            tflops = []
            errors = []
            for i in range(RETRIES):
                r = smoke_one(M, N, K, full, GPU)
                if "error" in r:
                    print(f"  retry{i+1}: ERROR {r.get('error')}")
                    errors.append(r.get("error"))
                else:
                    ff = r.get("finite_frac", 1.0)
                    if ff < 0.99:
                        print(f"  retry{i+1}: NaN-output (finite={ff:.3f})")
                        errors.append(f"NAN({ff:.3f})")
                    else:
                        print(f"  retry{i+1}: {r['tflops']:.2f} TFLOPS  finite={ff:.3f}")
                        tflops.append(r["tflops"])
            ok = len(tflops)
            if ok == RETRIES:
                verdict = "OK"
            elif ok == 0:
                verdict = "BROKEN"
            else:
                verdict = "FLAKY"
            print(f"  -> {ok}/{RETRIES}  verdict={verdict}")
            out["results"].append({
                "label": lab, "strat": tag, "candidate": full,
                "comp": comp, "tflops_runs": tflops, "errors": errors,
                "verdict": verdict,
            })
    with open(os.path.join(SCRIPT_DIR, "bench_round13_optA_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved bench_round13_optA_smoke.json")


if __name__ == "__main__":
    main()
