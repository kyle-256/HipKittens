#!/usr/bin/env python3
"""R19A SNR + aperture safety probe.

Two-phase check, lifted from R18A snr_probe but extended:
  Phase A: noise-floor SNR with uniform-scale (fp4 nibbles 0..2, scale 2^-3).
           Variant must be within 6 dB of parent noise floor AND >= 20 dB absolute.
  Phase B: random-scale aperture probe — small bench (warmup=20, iters=20)
           with random fp4 + random scales in [-2, 2]. If subprocess crashes
           with HSA aperture violation, mark BROKEN-APERTURE (lesson from R18A
           DLA1/step12).

Per benchmark-rules: SNR-OK alone is necessary-but-not-sufficient; we MUST
also do the aperture probe before benching.
"""
import json, math, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

# (label, M_native, N, K, parent_suffix)  [matches build_round19_optA.py]
SHAPES = [
    ("S1",  14336,  4096, 32768, "_lgk2_dc"),
    ("S2",  16384,  4096, 28672, "_u32"),
    ("S3",   4096, 32768, 28672, "_v20_memc"),
    ("S4",   4096, 28672, 32768, "_u16"),
    ("S5",  32768,  4096, 14336, "_ts_gm8_v12"),
    ("S6",   4096, 32768, 14336, "_ts_lgk2_memc"),
    ("S7",  28672, 32768,  4096, "_ts_lgk2_v12_memc"),
    ("S8",   4096, 32768,  6144, "_ts_pf4_memc"),
    ("S9",  16384, 28672,  2048, "_ts_gm2_v12_memc_dc"),
    ("S10", 16384, 28672,  4096, "_ts_gm2_v12_memc"),
    ("S11", 28672,  4096,  8192, "_ts_lgk2_memc_dc"),
    ("S12", 14336, 32768,  4096, "_ts_v12_tv0_memc"),
    ("S13",  4096, 14336, 16384, "_ts_lgk2"),
    ("S14",  6144,  4096, 16384, "_ts_lgk2"),
    ("S15", 32768,  4096,  7168, "_ts_gm8_v12"),
]
VARIANTS = ["_r19a_step3", "_r19a_step12", "_r19a_all"]
GPU_POOL = [0, 1, 2, 3]


def make_uniform_script(M, N, K, suffix, save_path):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    return module_name, so_path, f"""
import sys, math, torch, importlib.util, json, numpy as np, os
torch.manual_seed(42)
M, N, K = {M}, {N}, {K}
def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0,3,(r,c),dtype=torch.uint8,device='cuda')<<4)|torch.randint(0,3,(r,c),dtype=torch.uint8,device='cuda')
def manual_pad_scales(rows, k_blocks, scale_byte=124):
    pr=math.ceil(rows/64)*64; pk=math.ceil(k_blocks/8)*8
    return torch.full((pr//64, pk*64), scale_byte, dtype=torch.uint8, device='cuda')
spec=importlib.util.spec_from_file_location('{module_name}','{so_path}')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
A=gen_fp4(M,K); B=gen_fp4(N,K)
A_sc=manual_pad_scales(M, K//32); B_sc=manual_pad_scales(N, K//32)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
mod.gemm_rcr(A,B,A_sc,B_sc,C); torch.cuda.synchronize()
Cf = C.float()
finite_mask = torch.isfinite(Cf)
finite = float(finite_mask.float().mean().item())
n_lo = max(0, N//2 - 128); n_hi = min(N, n_lo + 256)
m_lo = max(0, M//2 - 128); m_hi = min(M, m_lo + 256)
tile = Cf[m_lo:m_hi, n_lo:n_hi].cpu().numpy()
np.save('{save_path}', tile)
print(json.dumps({{"finite":finite, "shape":list(tile.shape)}}))
"""


def make_aperture_script(M, N, K, suffix):
    """Random scales in [-2,2] + random fp4. Smoke 20 iters to catch races."""
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    return module_name, so_path, f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
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
for _ in range(20): run()
torch.cuda.synchronize()
import time
t0=time.time()
for _ in range(20): run()
torch.cuda.synchronize()
ms=(time.time()-t0)*1000.0/20
tflops=2.0*M*N*K/(ms*1e-3)/1e12
print(json.dumps({{"tflops":round(tflops,1),"ms":round(ms,4)}}))
"""


def run_one(M, N, K, suffix, gpu_id, save_path):
    mn, sp, script = make_uniform_script(M, N, K, suffix, save_path)
    if not os.path.exists(sp):
        return {"error": "missing .so", "path": sp}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
            capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def run_aperture(M, N, K, suffix, gpu_id):
    mn, sp, script = make_aperture_script(M, N, K, suffix)
    if not os.path.exists(sp):
        return {"error": "missing .so"}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
            capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def compute_snr(ref_path, cand_path):
    import numpy as np
    if not (os.path.exists(ref_path) and os.path.exists(cand_path)):
        return None, "missing"
    ref = np.load(ref_path).astype(np.float64)
    cnd = np.load(cand_path).astype(np.float64)
    if ref.shape != cnd.shape:
        return None, "shape mismatch"
    mask = np.isfinite(ref) & np.isfinite(cnd)
    if mask.sum() < 100:
        return None, "no finite overlap"
    ref_m = ref[mask]; cnd_m = cnd[mask]
    diff = ref_m - cnd_m
    sig = float((ref_m * ref_m).mean()); err = float((diff * diff).mean())
    if sig <= 0: return None, "ref power 0"
    if err <= 0: return float("inf"), "perfect"
    return 10.0 * math.log10(sig / err), "ok"


def parent_phase(args):
    lab, M, N, K, ps, gpu, tmpdir, idx = args
    path = os.path.join(tmpdir, f"parent{idx}_{lab}.npy")
    if os.path.exists(path):
        return lab, idx, path, {"cached": True}
    r = run_one(M, N, K, ps, gpu, path)
    return lab, idx, path, r


def variant_phase(args):
    lab, M, N, K, ps, vs, gpu, tmpdir = args
    cand_suffix = ps + vs
    cand_path = os.path.join(tmpdir, f"cand_{lab}_{vs.lstrip('_')}.npy")
    # Phase A: uniform SNR
    rcand = run_one(M, N, K, cand_suffix, gpu, cand_path)
    if "error" in rcand:
        return lab, vs, gpu, {"unif": rcand, "verdict": "BROKEN-APERTURE-UNIFORM"}
    parent_path = os.path.join(tmpdir, f"parent1_{lab}.npy")
    snr, why = compute_snr(parent_path, cand_path)
    snr_db = (None if snr is None else (float("inf") if snr == float("inf") else round(snr, 2)))
    # Phase B: random-scale aperture probe
    rapr = run_aperture(M, N, K, cand_suffix, gpu)
    return lab, vs, gpu, {"unif": rcand, "snr_db": snr_db, "snr_why": why, "aperture": rapr}


def main():
    tmpdir = os.path.join(SCRIPT_DIR, "snr_probe_r19a_tiles")
    os.makedirs(tmpdir, exist_ok=True)

    print("=== Phase 1: parent reference (run 1) ===", flush=True)
    args1 = []
    for i, (lab, M, N, K, ps) in enumerate(SHAPES):
        gpu = GPU_POOL[i % len(GPU_POOL)]
        args1.append((lab, M, N, K, ps, gpu, tmpdir, 1))
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        for fut in as_completed([ex.submit(parent_phase, a) for a in args1]):
            lab, idx, path, r = fut.result()
            print(f"  parent1[{lab}]: {r}", flush=True)

    print("\n=== Phase 2: parent reference (run 2) — noise floor ===", flush=True)
    args2 = []
    for i, (lab, M, N, K, ps) in enumerate(SHAPES):
        gpu = GPU_POOL[i % len(GPU_POOL)]
        args2.append((lab, M, N, K, ps, gpu, tmpdir, 2))
    noise_floor = {}
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        for fut in as_completed([ex.submit(parent_phase, a) for a in args2]):
            lab, idx, path, r = fut.result()
            print(f"  parent2[{lab}]: {r}", flush=True)
    for (lab, M, N, K, ps) in SHAPES:
        p1 = os.path.join(tmpdir, f"parent1_{lab}.npy")
        p2 = os.path.join(tmpdir, f"parent2_{lab}.npy")
        snr, why = compute_snr(p1, p2)
        noise_floor[lab] = (None if snr is None else (float("inf") if snr == float("inf") else round(snr, 2)))
        print(f"  noise[{lab}] = {noise_floor[lab]} dB ({why})", flush=True)

    print("\n=== Phase 3: per-variant SNR + aperture ===", flush=True)
    args3 = []
    i = 0
    for (lab, M, N, K, ps) in SHAPES:
        for vs in VARIANTS:
            gpu = GPU_POOL[i % len(GPU_POOL)]
            args3.append((lab, M, N, K, ps, vs, gpu, tmpdir))
            i += 1

    out = {"shapes": {}, "noise_floor": noise_floor, "ok_pairs": [], "broken_pairs": []}
    for (lab, _, _, _, _) in SHAPES:
        out["shapes"][lab] = {"variants": {}, "noise_floor_db": noise_floor.get(lab)}

    print(f"Total checks: {len(args3)}")
    print("=" * 110)
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = [ex.submit(variant_phase, a) for a in args3]
        for fut in as_completed(futs):
            lab, vs, gpu, info = fut.result()
            entry = {"gpu": gpu, **info}
            # Verdict
            unif = info.get("unif", {})
            apr = info.get("aperture", {})
            snr_db = info.get("snr_db")
            nf = noise_floor.get(lab)
            if "error" in unif:
                verdict = "BROKEN-APERTURE-UNIFORM"
            elif "error" in apr:
                verdict = "BROKEN-APERTURE-RANDOM"
            elif snr_db is None:
                verdict = "BROKEN-NO-SNR"
            elif nf is None:
                verdict = "BROKEN-NO-NOISE-FLOOR"
            elif snr_db == float("inf"):
                verdict = "OK-PERFECT"
            elif snr_db >= nf - 6.0 and snr_db >= 20.0:
                verdict = "OK"
            elif snr_db >= 15.0:
                verdict = "OK-MARGINAL"
            else:
                verdict = "BROKEN-RACE"
            entry["verdict"] = verdict
            out["shapes"][lab]["variants"][vs] = entry
            print(f"  {lab:5s} {vs:18s} gpu={gpu}  snr={snr_db!s:>10s} dB  noise={nf!s:>10s} dB  apr_tflops={apr.get('tflops')!s:>8s}  {verdict}", flush=True)
            if verdict.startswith("OK"):
                out["ok_pairs"].append([lab, vs, verdict])
            else:
                out["broken_pairs"].append([lab, vs, verdict])

    print("\n" + "=" * 110)
    print(f"OK: {len(out['ok_pairs'])}    BROKEN: {len(out['broken_pairs'])}")
    print("OK entries:")
    for p in out["ok_pairs"]:
        print(f"  {p}")
    with open(os.path.join(SCRIPT_DIR, "snr_probe_r19a.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved snr_probe_r19a.json")


if __name__ == "__main__":
    main()
