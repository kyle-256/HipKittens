#!/usr/bin/env python3
"""Round 14 OptB single-shot. warmup=200 iters=500 trim=10%, GPU 2.
Re-bench parent + each survivor on the same GPU with full-rigor parameters.
Gate: candidate must be >= parent + 0.5pp (using comp as denom).
"""
import json, math, os, subprocess, sys, time, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 200
ITERS = 500
TRIM = 0.10
GPU = 2

# Survivors after smoke: drop variants that regressed >50 TFLOPS in smoke
# (smoke is noisy but huge regressions are real).
# Keep all DLA1 (within ~200 TFLOPS), all DLA7 (close to parent),
# DLA2 keep only the gm2-stack (gm1 family clearly regresses),
# P1 keep the close ones (memc, noembed_memc, v20_memc, noembed).
SHAPES_VARIANTS = [
    # DLA1: keep all 7 (smoke spread modest)
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc_r14b_extbr",     5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc_r14b_noembed",   5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v20_memc",                5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v24_memc",                5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_lgk4_v12_memc",           5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc_tv0",            5781.1, "_ts_pf6_6_v12_memc"),
    ("DLA1", 4096, 32768, 128256, "_ts_pf6_6_v12_memc_tv16",           5781.1, "_ts_pf6_6_v12_memc"),
    # DLA2: drop gm1 family (~-330 TFLOPS regression in smoke). Keep gm2 noembed/extbr.
    ("DLA2", 128256, 32768, 4096, "_ts_gm2_v12_memc_dc_noembed",       4536.4, "_ts_gm2_v12_memc_dc"),
    ("DLA2", 128256, 32768, 4096, "_ts_gm2_v12_memc_dc_extbr",         4536.4, "_ts_gm2_v12_memc_dc"),
    # DLA7: keep all 6 (smoke spread modest, several positive)
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc_tv0",              4466.6, "_ts_lgk2_v12_memc"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc_tv16",             4466.6, "_ts_lgk2_v12_memc"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc_extbr",            4466.6, "_ts_lgk2_v12_memc"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v12_memc_noembed",          4466.6, "_ts_lgk2_v12_memc"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk4_v12_memc",                  4466.6, "_ts_lgk2_v12_memc"),
    ("DLA7", 28672, 32768, 4096, "_ts_lgk2_v20_memc",                  4466.6, "_ts_lgk2_v12_memc"),
    # P1: keep close-to-parent (within ~25 TFLOPS in smoke)
    ("P1", 28672, 4096, 16384, "_ts_gm8_memc",                         5350.6, "_ts_gm8"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_noembed",                      5350.6, "_ts_gm8"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_noembed_memc",                 5350.6, "_ts_gm8"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_v20_memc",                     5350.6, "_ts_gm8"),
    ("P1", 28672, 4096, 16384, "_ts_gm8_v12_memc",                     5350.6, "_ts_gm8"),
]


def bench(M, N, K, full_suffix, gpu_id):
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
run(); torch.cuda.synchronize()
finite_frac = float(torch.isfinite(C.float()).float().mean().item())
print('SNR_FINITE_FRAC=' + str(finite_frac))
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
            tag = "APERTURE" if "APERTURE" in r.stderr else f"rc={r.returncode}"
            return {"error": tag, "stderr": r.stderr[-300:]}
        finite = None
        for ln in r.stdout.splitlines():
            if ln.startswith("SNR_FINITE_FRAC="):
                finite = float(ln.split("=", 1)[1])
        last = r.stdout.strip().splitlines()[-1]
        d = json.loads(last)
        if finite is not None:
            d["finite_frac"] = finite
        return d
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"Round 14 OptB single-shot. GPU={GPU} warmup={WARMUP} iters={ITERS} candidates={len(SHAPES_VARIANTS)}")
    print("=" * 110)
    out = {"warmup": WARMUP, "iters": ITERS, "gpu": GPU, "shapes": []}
    parent_t_cache = {}  # (M,N,K,parent) -> tflops
    for (label, M, N, K, suffix, comp, parent_suf) in SHAPES_VARIANTS:
        pkey = (M, N, K, parent_suf)
        if pkey not in parent_t_cache:
            r_p = bench(M, N, K, parent_suf, GPU)
            if "error" in r_p:
                print(f"\n[{label}] parent {parent_suf:42s} ERROR: {r_p.get('error')}")
                parent_t_cache[pkey] = None
            else:
                parent_t_cache[pkey] = r_p["tflops"]
                print(f"\n[{label}] parent {parent_suf:42s} {r_p['tflops']:7.2f} TFLOPS  ({r_p['tflops']/comp*100:.2f}%)  finite={r_p.get('finite_frac',-1):.3f}")
            out["shapes"].append({
                "label": label, "M": M, "N": N, "K": K, "comp": comp,
                "kind": "parent", "suffix": parent_suf,
                "tflops": r_p.get("tflops"),
                "finite_frac": r_p.get("finite_frac"),
                "error": r_p.get("error"),
            })
        best_t = parent_t_cache[pkey]
        r = bench(M, N, K, suffix, GPU)
        if "error" in r:
            print(f"  cand {suffix:48s} ERROR: {r.get('error')}")
            out["shapes"].append({"label": label, "M": M, "N": N, "K": K,
                                  "kind": "candidate", "suffix": suffix,
                                  "parent": parent_suf,
                                  "error": r.get("error"),
                                  "stderr": r.get("stderr","")[:200]})
            continue
        t = r["tflops"]; ff = r.get("finite_frac")
        d_pp = ((t - best_t) / comp * 100) if best_t else None
        ff_str = f"  finite={ff:.3f}" if ff is not None else ""
        d_str = f"  delta={t - best_t:+.2f}T  ({d_pp:+.2f}pp)" if best_t else ""
        ratio_pct = round(t/comp*100, 2)
        print(f"  cand {suffix:48s} {t:7.2f} TFLOPS  ({ratio_pct:5.2f}%){d_str}{ff_str}", flush=True)
        out["shapes"].append({
            "label": label, "M": M, "N": N, "K": K, "comp": comp,
            "kind": "candidate", "suffix": suffix, "parent": parent_suf,
            "tflops": t,
            "ratio_pct": ratio_pct,
            "delta_tflops": (t - best_t) if best_t else None,
            "delta_pp": d_pp,
            "finite_frac": ff,
            "parent_tflops": best_t,
        })
    print("=" * 110)
    with open(os.path.join(SCRIPT_DIR, "bench_round14_optB_singleshot.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round14_optB_singleshot.json")


if __name__ == "__main__":
    main()
