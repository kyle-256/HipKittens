#!/usr/bin/env python3
"""Round 14 OptB smoke. warmup=20 iters=50, 3-retry on failure.
Detects HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION + SNR (NaN-fraction) check.
GPU 2 only. Pure macro-only variants (no iterilp), so aperture bug should NOT
appear, but we keep the same retry harness as Round 13.
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 20
ITERS = 50
TRIM = 0.10
GPU = 2

# (label, M, N, K, suffix, comp_tflops, parent_for_compare)
SHAPES_VARIANTS = [
    # DLA1 4096x32768x128256 comp=5781.1, parent _ts_pf6_6_v12_memc tflops 5213.6
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc_r14b_extbr",     5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc_r14b_noembed",   5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v20_memc",                5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v24_memc",                5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_lgk4_v12_memc",           5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc_tv0",            5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc_tv16",           5781.1, "_ts_pf6_6_v12_memc"),
    # DLA2 128256x32768x4096 comp=4536.4, parent _ts_gm2_v12_memc_dc tflops 4213.7
    ("DLA2", 128256, 32768, 4096, "_ts_gm1_v12_memc_dc",               4536.4, "_ts_gm2_v12_memc_dc"),
    ("DLA2", 128256, 32768, 4096, "_ts_gm1_lgk2_v12_memc_dc",          4536.4, "_ts_gm2_v12_memc_dc"),
    ("DLA2", 128256, 32768, 4096, "_ts_gm1_v20_memc_dc",               4536.4, "_ts_gm2_v12_memc_dc"),
    ("DLA2", 128256, 32768, 4096, "_ts_gm1_v12_memc_dc_extbr",         4536.4, "_ts_gm2_v12_memc_dc"),
    ("DLA2", 128256, 32768, 4096, "_ts_gm1_v12_memc_dc_tv0",           4536.4, "_ts_gm2_v12_memc_dc"),
    ("DLA2", 128256, 32768, 4096, "_ts_gm2_v12_memc_dc_noembed",       4536.4, "_ts_gm2_v12_memc_dc"),
    ("DLA2", 128256, 32768, 4096, "_ts_gm2_v12_memc_dc_extbr",         4536.4, "_ts_gm2_v12_memc_dc"),
    # DLA7 28672x32768x4096 comp=4466.6, parent _ts_lgk2_v12_memc tflops 4219.5
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc_tv0",              4466.6, "_ts_lgk2_v12_memc"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc_tv16",             4466.6, "_ts_lgk2_v12_memc"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc_extbr",            4466.6, "_ts_lgk2_v12_memc"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc_noembed",          4466.6, "_ts_lgk2_v12_memc"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk4_v12_memc",                  4466.6, "_ts_lgk2_v12_memc"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v20_memc",                  4466.6, "_ts_lgk2_v12_memc"),
    # P1 28672x4096x16384 comp=5350.6, parent _ts_gm8 tflops 5013.9
    ("P1", 28672, 4096, 16384, "_ts_gm8_memc",                         5350.6, "_ts_gm8"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_lgk2_memc",                    5350.6, "_ts_gm8"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_v12_memc",                     5350.6, "_ts_gm8"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_extbr",                        5350.6, "_ts_gm8"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_extbr_memc",                   5350.6, "_ts_gm8"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_noembed",                      5350.6, "_ts_gm8"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_noembed_memc",                 5350.6, "_ts_gm8"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_v20_memc",                     5350.6, "_ts_gm8"),
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
        r["retries_used"] = i
        r["status"] = "OK"
        return r
    last["status"] = "BROKEN"
    last["fails"] = fails
    return last


def main():
    print(f"Round 14 OptB smoke. GPU={GPU} warmup={WARMUP} iters={ITERS} variants={len(SHAPES_VARIANTS)}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "gpu": GPU, "results": []}
    benched_parents = set()
    for (label, M, N, K, suffix, comp, parent_suf) in SHAPES_VARIANTS:
        # Bench parent fresh once per (M,N,K,parent) combo
        parent_key = (M, N, K, parent_suf)
        if parent_key not in benched_parents:
            benched_parents.add(parent_key)
            r_parent = smoke_with_retry(M, N, K, parent_suf, GPU)
            print(f"\n[{label}] parent {parent_suf:42s} -> ", end="")
            if r_parent.get("status") == "OK":
                print(f"OK tflops={r_parent['tflops']:.2f} finite={r_parent.get('finite_frac',-1):.3f}")
            else:
                print(f"{r_parent.get('status','?')} err={r_parent.get('error')}")
            out["results"].append({"label": label, "M": M, "N": N, "K": K,
                                   "variant": parent_suf, "kind": "parent",
                                   "comp": comp, **r_parent})
        # Then candidate
        r = smoke_with_retry(M, N, K, suffix, GPU)
        if r.get("status") == "OK":
            print(f"  cand {suffix:48s} OK   tflops={r['tflops']:.2f}  finite={r.get('finite_frac',-1):.3f}  retries={r.get('retries_used',0)}", flush=True)
        else:
            print(f"  cand {suffix:48s} BROKEN  fails={r.get('fails')} err={r.get('error')}", flush=True)
        out["results"].append({"label": label, "M": M, "N": N, "K": K,
                               "variant": suffix, "kind": "candidate",
                               "comp": comp, "parent": parent_suf, **r})
    print("=" * 110)
    with open(os.path.join(SCRIPT_DIR, "bench_round14_optB_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round14_optB_smoke.json")


if __name__ == "__main__":
    main()
