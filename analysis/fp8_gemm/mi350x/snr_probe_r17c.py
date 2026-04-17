#!/usr/bin/env python3
"""R17C SNR safety probe at small M=256.

Catches: BROKEN-NAN (catastrophic SGPR clobber), BROKEN-APERTURE (subprocess
crash). Note: the full aperture-violation bug only triggers at full M, so the
real safety check still happens in smoke-bench. This is just the cheap gate.
"""
import json, math, os, subprocess, sys, sysconfig
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc"),
    ("P1",   28672,  4096, 16384, "_ts_gm8"),
]

ATTRS = ["fwgs64_256", "fwgs256_256",
         "vgpr256", "vgpr224", "vgpr192",
         "sgpr96", "sgpr80", "mnwg8"]

NEW_PREFIX = "_r17c_"
GPU_POOL = [5, 6, 7]
SMALL_M = 256


def run_one(N, K, suffix, gpu_id, save_path):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so", "path": so_path}
    script = f"""
import sys, math, torch, importlib.util, json, numpy as np
torch.manual_seed(42)
M, N, K = {SMALL_M}, {N}, {K}
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
sc_a=torch.zeros((M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.zeros((N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
mod.gemm_rcr(A,B,A_sc,B_sc,C); torch.cuda.synchronize()
Cf = C.float()
finite = float(torch.isfinite(Cf).float().mean().item())
nz = float((Cf != 0).float().mean().item())
print(json.dumps({{"finite":finite,"nz":nz}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=180, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def task(args):
    lab, N, K, ps, tag, gpu = args
    cand_suffix = ps + NEW_PREFIX + tag
    rref = run_one(N, K, ps, gpu, "")
    rcnd = run_one(N, K, cand_suffix, gpu, "")
    verdict = "OK"
    if "error" in rcnd:
        verdict = "BROKEN-APERTURE"
    elif "error" in rref:
        verdict = "REF-FAIL"
    else:
        rf = rref.get("finite", 0.0); cf = rcnd.get("finite", 0.0)
        if cf < 0.01 and rf > 0.05:
            verdict = "BROKEN-NAN"
        elif rf > 0.05 and cf < 0.25 * rf:
            verdict = "BROKEN-DELTA"
    return lab, tag, gpu, rref, rcnd, verdict


def main():
    args_list = []
    i = 0
    for (lab, _, N, K, ps) in SHAPES:
        for tag in ATTRS:
            gpu = GPU_POOL[i % len(GPU_POOL)]
            args_list.append((lab, N, K, ps, tag, gpu))
            i += 1
    print(f"Total SNR checks: {len(args_list)}")
    out = {"shapes": {}}
    for (lab, _, _, _, _) in SHAPES:
        out["shapes"][lab] = {"attrs": {}}
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(task, a): a for a in args_list}
        for fut in as_completed(futs):
            lab, tag, gpu, rref, rcnd, verdict = fut.result()
            print(f"  {lab:6s} {tag:14s} gpu={gpu}  ref={rref} cand={rcnd}  {verdict}", flush=True)
            out["shapes"][lab]["attrs"][tag] = {"verdict": verdict, "gpu": gpu, "ref": rref, "cand": rcnd}
    json.dump(out, open(os.path.join(SCRIPT_DIR, "snr_probe_r17c.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
