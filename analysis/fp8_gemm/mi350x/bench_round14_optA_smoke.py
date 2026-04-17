#!/usr/bin/env python3
"""R14 OptA aperture smoke test.
For each non-NOOP, non-broken-build variant from asm_diff_probe_r14a.json,
run a quick warmup=20/iters=50 bench with up to 3 retries to detect:
  - HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (compiler-bug pattern from R12)
  - NaN/Inf output (silent miscompile)
  - Hard errors

Output: bench_round14_optA_smoke.json with (label, tag, status, retry_count)
where status in {OK, BROKEN_APERTURE, BROKEN_NAN, BROKEN_OTHER, MISSING}.

Survivors flow to bench_round14_optA_singleshot.py.
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
PROBE_JSON = os.path.join(SCRIPT_DIR, "asm_diff_probe_r14a.json")

WARMUP = 20
ITERS = 50
TRIM = 0.10
GPU = 1
N_RETRIES = 3

# (label, M, N, K, parent_suffix)
SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc"),
    ("P1",   28672,  4096, 16384, "_ts_gm8"),
]
SHAPE_BY_LABEL = {s[0]: s for s in SHAPES}

# pav* variants are known to fail to build (illegal VGPR-to-SGPR copy).
SKIP_TAGS = {"pav16", "pav32", "pav64", "nopav"}


def smoke_one(M, N, K, full_suffix, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"status": "MISSING", "path": so_path}
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
finite_frac = float(torch.isfinite(C.float()).float().mean().item())
print('FINITE='+str(finite_frac))
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
                           capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            stderr = r.stderr[-500:]
            if "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION" in stderr or \
               "Memory access fault" in stderr or "memory aperture" in stderr.lower():
                return {"status": "BROKEN_APERTURE", "stderr": stderr[-200:]}
            return {"status": "BROKEN_OTHER", "rc": r.returncode, "stderr": stderr[-200:]}
        finite = None
        for ln in r.stdout.splitlines():
            if ln.startswith("FINITE="):
                finite = float(ln.split("=", 1)[1])
        # NOTE: bf16 saturates with K=128256 even on the parent (~47% finite),
        # so we don't gate solely on finite>=0.999. Instead, hard NaN means
        # finite==0 or close. We compare to parent finite later in singleshot.
        if finite is not None and finite < 0.001:
            return {"status": "BROKEN_NAN", "finite": finite}
        last = r.stdout.strip().splitlines()[-1]
        d = json.loads(last)
        d["status"] = "OK"
        d["finite"] = finite
        return d
    except subprocess.TimeoutExpired:
        return {"status": "BROKEN_TIMEOUT"}
    except Exception as e:
        return {"status": "BROKEN_OTHER", "exception": str(e)}


def main():
    with open(PROBE_JSON) as f:
        probe = json.load(f)
    # Build candidates: (label, tag, full_suffix)
    cands = []
    for lab, sd in probe["shapes"].items():
        ps = SHAPE_BY_LABEL[lab][4]
        for tag, info in sd["flags"].items():
            if info["verdict"] != "DIFF":
                continue
            if tag in SKIP_TAGS:
                continue
            full_suffix = ps + "_r14a_" + tag
            cands.append((lab, tag, full_suffix))
    print(f"R14A smoke. GPU={GPU} warmup={WARMUP} iters={ITERS}  candidates={len(cands)}")
    print("=" * 90)
    out = {"warmup": WARMUP, "iters": ITERS, "trim": TRIM, "gpu": GPU, "results": []}
    for (lab, tag, fs) in cands:
        M, N, K = SHAPE_BY_LABEL[lab][1], SHAPE_BY_LABEL[lab][2], SHAPE_BY_LABEL[lab][3]
        last = None
        for retry in range(N_RETRIES):
            r = smoke_one(M, N, K, fs, GPU)
            r["retry"] = retry
            last = r
            if r["status"] == "OK":
                break
            # Retry only on BROKEN_APERTURE flake (per R12 finding) or TIMEOUT.
            if r["status"] not in ("BROKEN_APERTURE", "BROKEN_TIMEOUT"):
                break
        status = last["status"]
        tflops = last.get("tflops")
        finite = last.get("finite")
        extra = ""
        if status != "OK":
            extra = f"  (retry={last['retry']})"
        if tflops is not None:
            extra += f"  tflops={tflops}"
        if finite is not None and status == "OK":
            extra += f"  finite={finite:.3f}"
        print(f"  {lab:5s} {tag:14s}  {status:18s}{extra}", flush=True)
        out["results"].append({"label": lab, "tag": tag, "suffix": fs,
                              "status": status, "tflops": tflops,
                              "finite": finite, "retry": last["retry"]})
    print("=" * 90)
    counts = {}
    for r in out["results"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Status counts: {counts}")
    with open(os.path.join(SCRIPT_DIR, "bench_round14_optA_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round14_optA_smoke.json")


if __name__ == "__main__":
    main()
