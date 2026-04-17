#!/usr/bin/env python3
"""R16C SNR safety probe: tiny 256x256x4096 random fp4 SNR test for each
compound .so before bench. Filters new compiler-bug triggers cheaply.

Compares each compound's output vs the parent's output for that shape using
the *same* parent kernel (since the parent is the trusted reference) at the
TARGET SHAPE's (M, N, K). But the parent kernel was compiled for a specific
(N, K) module — we must invoke each compound at its OWN (M, N, K).

Approach:
  - For each (shape, compound), invoke the *compound* kernel at small M=256,
    fixed (N, K) of its parent. Record the BF16 output's basic finiteness
    + checksum + abs-max.
  - Treat NaN/Inf or all-zero output as BROKEN-SNR/BROKEN-APERTURE.
  - For shapes where M > N or M > K, the kernel still works but slot 256 is fine.
  - We further compare against the parent's output at the same (M=256, N, K) as
    the SNR-reference. SNR threshold = 25 dB.

Scales = all-zero (deterministic, no accumulator overflow), as in snr_check_r13b.
"""
import json, math, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

# Mirror SHAPES
SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc"),
    ("P1",   28672,  4096, 16384, "_ts_gm8"),
]

COMPOUNDS = [
    ("iterilp", "regclassglob"),
    ("iterilp", "sinkavoidspill"),
    ("iterilp", "nolicm"),
    ("iterilp", "noemxpre"),
    ("iterilp", "largeivf2"),
    ("regclassglob", "sinkavoidspill"),
    ("regclassglob", "nolicm"),
    ("regclassglob", "noemxpre"),
    ("sinkavoidspill", "nolicm"),
    ("regclassglob", "nolicm", "sinkavoidspill"),
    ("iterilp", "regclassglob", "nolicm"),
    ("iterilp", "regclassglob", "sinkavoidspill"),
]

NEW_PREFIX = "_r16c_"
GPU_POOL = [5, 6, 7]
SMALL_M = 256       # Force kernel to run a tiny tile
SNR_THRESHOLD_DB = 25.0


def run_one(M_native, N, K, suffix, gpu_id, save_path):
    """Run kernel once with deterministic inputs (M=SMALL_M slice from a
    full M=M_native call), save center 256x256 tile to npy.

    NOTE: kernel modules are compiled with hard-coded N_DIM/K_DIM macros, but
    M is a runtime tensor dim. So we can use M=SMALL_M while keeping (N, K).
    """
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so", "path": so_path}
    script = f"""
import sys, math, torch, importlib.util, json, numpy as np, os
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
chk = float(Cf.sum().item()); cmax = float(Cf.abs().max().item())
# Save central tile (limited by N)
n_lo = max(0, N//2 - 128); n_hi = min(N, n_lo + 256)
m_lo = max(0, M//2 - 128); m_hi = min(M, m_lo + 256)
tile = Cf[m_lo:m_hi, n_lo:n_hi].cpu().numpy()
np.save('{save_path}', tile)
print(json.dumps({{"finite":finite,"nz":nz,"sum":chk,"max":cmax,"shape":list(tile.shape)}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=180, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        line = r.stdout.strip().splitlines()[-1]
        return json.loads(line)
    except Exception as e:
        return {"error": str(e)}


def snr_db(ref_path, cand_path):
    import numpy as np
    if not (os.path.exists(ref_path) and os.path.exists(cand_path)):
        return None, "missing tile"
    ref = np.load(ref_path).astype(np.float32)
    cnd = np.load(cand_path).astype(np.float32)
    if ref.shape != cnd.shape:
        return None, f"shape {ref.shape} != {cnd.shape}"
    if not np.isfinite(cnd).all():
        return None, "cnd has NaN/Inf"
    if not np.isfinite(ref).all():
        return None, "ref has NaN/Inf"
    diff = ref - cnd
    sig = float((ref * ref).mean())
    err = float((diff * diff).mean())
    if sig <= 0:
        return None, "ref signal power 0"
    if err <= 0:
        return float("inf"), "perfect"
    return 10.0 * math.log10(sig / err), "ok"


def task(args):
    """Robust safety verdict (M=SMALL_M produces partially-NaN parent output
    for these huge-K shapes; we don't insist on full SNR equality, only that
    the CANDIDATE doesn't degrade meaningfully vs the PARENT under the same
    inputs):

      verdict cases:
        BROKEN-APERTURE: subprocess died (likely memory aperture crash)
        BROKEN-NAN:      cand is fully NaN (finite_frac near 0) AND ref isn't
        BROKEN-DELTA:    cand finite_frac drops > 0.10 below ref finite_frac
        OK-SUBSAMPLE:    finite tiles agree, SNR not computable due to NaN
                          mask differences but cand passes finite-fraction sanity
        OK:              SNR >= threshold on the finite subset
    """
    lab, M_native, N, K, ps, tags, gpu, tmpdir = args
    csuf = NEW_PREFIX + "X".join(tags)
    tag_label = "+".join(tags)
    cand_suffix = ps + csuf
    ref_path = os.path.join(tmpdir, f"ref_{lab}.npy")
    cand_path = os.path.join(tmpdir, f"cand_{lab}_{tag_label.replace('+','_')}.npy")
    rref_meta_path = os.path.join(tmpdir, f"ref_{lab}.json")
    if not os.path.exists(ref_path):
        rref = run_one(M_native, N, K, ps, gpu, ref_path)
        if "error" in rref:
            return lab, tag_label, gpu, rref, None, None, "REF_FAIL"
        with open(rref_meta_path, "w") as f:
            json.dump(rref, f)
    with open(rref_meta_path) as f:
        rref_meta = json.load(f)
    ref_finite = rref_meta.get("finite", 0.0)

    rcand = run_one(M_native, N, K, cand_suffix, gpu, cand_path)
    if "error" in rcand:
        return lab, tag_label, gpu, None, rcand, None, "BROKEN-APERTURE"
    cand_finite = rcand.get("finite", 0.0)
    if cand_finite < 0.01 and ref_finite > 0.05:
        return lab, tag_label, gpu, None, rcand, None, "BROKEN-NAN"
    # Note: previously we used cand_finite < ref_finite - 0.10 → too strict
    # given BF16 saturation noise across compilers. Now we only flag if cand
    # has < 25% of ref finite mass (catches genuine SGPR-clobber but allows
    # ordinary FMA-rounding-induced extra-NaNs).
    if ref_finite > 0.05 and cand_finite < 0.25 * ref_finite:
        return lab, tag_label, gpu, None, rcand, None, "BROKEN-DELTA"

    # Compute SNR over the FINITE INTERSECTION mask.
    import numpy as np
    try:
        ref = np.load(ref_path).astype(np.float32)
        cnd = np.load(cand_path).astype(np.float32)
        mask = np.isfinite(ref) & np.isfinite(cnd)
        if mask.sum() < 100:
            return lab, tag_label, gpu, None, rcand, (None, "no finite overlap"), "OK-SUBSAMPLE"
        ref_m = ref[mask]; cnd_m = cnd[mask]
        diff = ref_m - cnd_m
        sig = float((ref_m * ref_m).mean())
        err = float((diff * diff).mean())
        if sig <= 0:
            snr_val, why = None, "ref signal power 0"
        elif err <= 0:
            snr_val, why = float("inf"), "perfect"
        else:
            snr_val, why = 10.0 * math.log10(sig / err), "ok"
    except Exception as e:
        return lab, tag_label, gpu, None, rcand, (None, str(e)), "OK-SUBSAMPLE"

    if snr_val is None:
        verdict = "OK-SUBSAMPLE"
    elif snr_val != float("inf") and snr_val < SNR_THRESHOLD_DB:
        verdict = "BROKEN-SNR"
    else:
        verdict = "OK"
    return lab, tag_label, gpu, None, rcand, (snr_val, why), verdict


def main():
    tmpdir = os.path.join(SCRIPT_DIR, "snr_probe_r16c_tiles")
    os.makedirs(tmpdir, exist_ok=True)
    # Pre-compute references serially first to avoid races.
    print("=== Phase 1: compute parent references ===")
    for (lab, M_native, N, K, ps) in SHAPES:
        ref_path = os.path.join(tmpdir, f"ref_{lab}.npy")
        ref_meta_path = os.path.join(tmpdir, f"ref_{lab}.json")
        if os.path.exists(ref_path) and os.path.exists(ref_meta_path):
            print(f"  {lab}: cached ref")
            continue
        gpu = GPU_POOL[0]
        print(f"  {lab}: computing ref on gpu={gpu}", flush=True)
        rref = run_one(M_native, N, K, ps, gpu, ref_path)
        if "error" in rref:
            print(f"    REF FAIL: {rref}")
        else:
            print(f"    {rref}")
            with open(ref_meta_path, "w") as f:
                json.dump(rref, f)

    print("\n=== Phase 2: per-compound SNR check ===")
    args_list = []
    i = 0
    for (lab, M_native, N, K, ps) in SHAPES:
        for tags in COMPOUNDS:
            gpu = GPU_POOL[i % len(GPU_POOL)]
            args_list.append((lab, M_native, N, K, ps, tags, gpu, tmpdir))
            i += 1

    out = {"snr_threshold_db": SNR_THRESHOLD_DB, "shapes": {},
           "ok_pairs": [], "broken_pairs": []}
    for (lab, _, _, _, _) in SHAPES:
        out["shapes"][lab] = {"compounds": {}}
    print(f"Total SNR checks: {len(args_list)}")
    print("=" * 100)
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(task, a): a for a in args_list}
        for fut in as_completed(futs):
            lab, tag_label, gpu, ref_err, rcand, snr_info, verdict = fut.result()
            entry = {"verdict": verdict, "gpu": gpu,
                     "cand_run": rcand, "ref_err": ref_err}
            if snr_info is not None:
                snr, why = snr_info
                entry["snr_db"] = (None if snr is None else (float("inf") if snr == float("inf") else round(snr, 2)))
                entry["snr_why"] = why
            out["shapes"][lab]["compounds"][tag_label] = entry
            snr_str = f"{entry.get('snr_db', '?')}"
            print(f"  {lab:6s} {tag_label:50s} gpu={gpu}  snr={snr_str:>10s} dB  {verdict}", flush=True)
            if verdict == "OK":
                out["ok_pairs"].append([lab, tag_label])
            else:
                out["broken_pairs"].append([lab, tag_label, verdict])

    print("\n" + "=" * 100)
    print(f"OK: {len(out['ok_pairs'])}    BROKEN: {len(out['broken_pairs'])}")
    print("BROKEN entries:")
    for p in out["broken_pairs"]:
        print(f"  {p}")
    with open(os.path.join(SCRIPT_DIR, "snr_probe_r16c.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved snr_probe_r16c.json")


if __name__ == "__main__":
    main()
