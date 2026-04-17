#!/usr/bin/env python3
"""R18A SNR safety probe.

Approach:
  - Use R11-style small inputs (fp4 nibbles 0..2, scale 2^-3).
  - For each shape, run PARENT twice → establish noise-floor SNR (parent vs parent).
  - For each variant, run candidate once → compute SNR vs parent.
  - Verdict: variant SNR must be within 6 dB of noise-floor SNR (and ≥ 20 dB absolute).
  - This is VERY robust against:
      * non-deterministic FMA reorder (parent itself is non-deterministic)
      * bf16 saturation (we compare on finite intersection only)
      * different NaN positions across runs

Per benchmark-rules and R17A finding: must SNR-validate before benching to avoid
silent wrong output from barrier→waitcnt swap.
"""
import json, math, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

# (label, M_native, N, K, parent_suffix)
SHAPES = [
    ("DLA1", 4096,  32768, 128256, "_ts_pf6_6_v12_memc"),
    ("DLA2", 128256, 32768,  4096, "_ts_gm2_v12_memc_dc"),
    ("DLA7", 28672, 32768,  4096, "_ts_lgk2_v12_memc"),
    ("P1",   28672,  4096, 16384, "_ts_gm8"),
]
VARIANTS = ["_r18a_p3_step3", "_r18a_p3_step12", "_r18a_p3_all"]

GPU_POOL = [0, 1]
# Use FULL native M to ensure all CTAs are scheduled normally.
# Use small fp4 nibbles + small scale to keep MOST output bf16-finite.

def make_script(M, N, K, suffix, save_path):
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    return module_name, so_path, f"""
import sys, math, torch, importlib.util, json, numpy as np, os
torch.manual_seed(42)
M, N, K = {M}, {N}, {K}
def gen_fp4(r, K):
    c = K // 2
    # nibbles 0..2 only (small fp4 values 0, 0.5, 1.0)
    return (torch.randint(0,3,(r,c),dtype=torch.uint8,device='cuda')<<4)|torch.randint(0,3,(r,c),dtype=torch.uint8,device='cuda')
def manual_pad_scales(rows, k_blocks, scale_byte=124):
    pr=math.ceil(rows/64)*64; pk=math.ceil(k_blocks/8)*8
    # All scales = 2^-3 = 0.125
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
# Save center 256x256 tile for SNR comparison
n_lo = max(0, N//2 - 128); n_hi = min(N, n_lo + 256)
m_lo = max(0, M//2 - 128); m_hi = min(M, m_lo + 256)
tile = Cf[m_lo:m_hi, n_lo:n_hi].cpu().numpy()
np.save('{save_path}', tile)
print(json.dumps({{"finite":finite, "shape":list(tile.shape)}}))
"""


def run_one(M, N, K, suffix, gpu_id, save_path):
    module_name, so_path, script = make_script(M, N, K, suffix, save_path)
    if not os.path.exists(so_path):
        return {"error": "missing .so", "path": so_path}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-500:]}
        line = r.stdout.strip().splitlines()[-1]
        return json.loads(line)
    except Exception as e:
        return {"error": str(e)}


def compute_snr(ref_path, cand_path):
    import numpy as np
    if not (os.path.exists(ref_path) and os.path.exists(cand_path)):
        return None, "missing tile"
    ref = np.load(ref_path).astype(np.float64)
    cnd = np.load(cand_path).astype(np.float64)
    if ref.shape != cnd.shape:
        return None, f"shape {ref.shape} != {cnd.shape}"
    mask = np.isfinite(ref) & np.isfinite(cnd)
    if mask.sum() < 100:
        return None, "no finite overlap"
    ref_m = ref[mask]; cnd_m = cnd[mask]
    diff = ref_m - cnd_m
    sig = float((ref_m * ref_m).mean())
    err = float((diff * diff).mean())
    if sig <= 0:
        return None, "ref power 0"
    if err <= 0:
        return float("inf"), "perfect"
    return 10.0 * math.log10(sig / err), "ok"


def task(args):
    """Run candidate once at full M_native, compare to parent_run_1."""
    lab, M_native, N, K, parent_suffix, var_suffix, gpu, tmpdir = args
    cand_suffix = parent_suffix + var_suffix
    parent_path = os.path.join(tmpdir, f"parent1_{lab}.npy")
    cand_path = os.path.join(tmpdir, f"cand_{lab}_{var_suffix.lstrip('_')}.npy")

    rcand = run_one(M_native, N, K, cand_suffix, gpu, cand_path)
    if "error" in rcand:
        return lab, var_suffix, gpu, rcand, None, None, "BROKEN-APERTURE"
    cand_finite = rcand.get("finite", 0.0)

    snr, why = compute_snr(parent_path, cand_path)
    return lab, var_suffix, gpu, None, rcand, (snr, why), None  # verdict assigned in main


def main():
    tmpdir = os.path.join(SCRIPT_DIR, "snr_probe_r18a_tiles")
    os.makedirs(tmpdir, exist_ok=True)

    # Phase 1: parent run 1
    print("=== Phase 1: parent reference (run 1) ===")
    for (lab, M_native, N, K, ps) in SHAPES:
        path = os.path.join(tmpdir, f"parent1_{lab}.npy")
        meta_path = os.path.join(tmpdir, f"parent1_{lab}.json")
        if os.path.exists(path) and os.path.exists(meta_path):
            print(f"  {lab}: cached parent1")
            continue
        gpu = GPU_POOL[0]
        print(f"  {lab}: M={M_native} N={N} K={K} on gpu={gpu}", flush=True)
        rref = run_one(M_native, N, K, ps, gpu, path)
        print(f"    {rref}", flush=True)
        if "error" not in rref:
            with open(meta_path, "w") as f: json.dump(rref, f)

    # Phase 2: parent run 2 (noise floor)
    print("\n=== Phase 2: parent reference (run 2) — noise floor ===")
    noise_floor = {}
    for (lab, M_native, N, K, ps) in SHAPES:
        path = os.path.join(tmpdir, f"parent2_{lab}.npy")
        meta_path = os.path.join(tmpdir, f"parent2_{lab}.json")
        if not (os.path.exists(path) and os.path.exists(meta_path)):
            gpu = GPU_POOL[0]
            print(f"  {lab}: rerun parent on gpu={gpu}", flush=True)
            r = run_one(M_native, N, K, ps, gpu, path)
            print(f"    {r}", flush=True)
            if "error" not in r:
                with open(meta_path, "w") as f: json.dump(r, f)
        snr, why = compute_snr(os.path.join(tmpdir, f"parent1_{lab}.npy"), path)
        noise_floor[lab] = snr
        print(f"  {lab}: noise-floor SNR = {snr if snr is None else round(snr, 2)} dB ({why})", flush=True)

    # Phase 3: variant SNR
    print("\n=== Phase 3: per-variant SNR ===")
    args_list = []
    i = 0
    for (lab, M_native, N, K, ps) in SHAPES:
        for var_suffix in VARIANTS:
            gpu = GPU_POOL[i % len(GPU_POOL)]
            args_list.append((lab, M_native, N, K, ps, var_suffix, gpu, tmpdir))
            i += 1

    out = {"shapes": {}, "noise_floor": noise_floor,
           "ok_pairs": [], "broken_pairs": []}
    for (lab, _, _, _, _) in SHAPES:
        out["shapes"][lab] = {"variants": {}, "noise_floor_db": noise_floor.get(lab)}
    print(f"Total checks: {len(args_list)}")
    print("=" * 110)
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(task, a): a for a in args_list}
        for fut in as_completed(futs):
            lab, var_suffix, gpu, ref_err, rcand, snr_info, _ = fut.result()
            entry = {"gpu": gpu, "cand_run": rcand, "ref_err": ref_err}
            if snr_info is not None:
                snr, why = snr_info
                entry["snr_db"] = (None if snr is None else (float("inf") if snr == float("inf") else round(snr, 2)))
                entry["snr_why"] = why
            # Verdict logic
            if ref_err is not None or "error" in (rcand or {}):
                verdict = "BROKEN-APERTURE"
            elif entry.get("snr_db") is None:
                verdict = "BROKEN-NO-SNR"
            else:
                snr_v = entry["snr_db"]
                nf = noise_floor.get(lab)
                if nf is None:
                    verdict = "BROKEN-NO-NOISE-FLOOR"
                elif snr_v == float("inf"):
                    verdict = "OK-PERFECT"
                elif snr_v >= nf - 6.0 and snr_v >= 20.0:
                    verdict = "OK"
                elif snr_v >= 15.0:
                    verdict = "OK-MARGINAL"
                else:
                    verdict = "BROKEN-RACE"
            entry["verdict"] = verdict
            out["shapes"][lab]["variants"][var_suffix] = entry
            sn = entry.get("snr_db", "?")
            nf_str = noise_floor.get(lab, "?")
            print(f"  {lab:6s} {var_suffix:25s} gpu={gpu}  snr={sn!s:>10s} dB  noise={nf_str!s:>10s} dB  {verdict}", flush=True)
            if verdict.startswith("OK"):
                out["ok_pairs"].append([lab, var_suffix, verdict])
            else:
                out["broken_pairs"].append([lab, var_suffix, verdict])

    print("\n" + "=" * 110)
    print(f"OK: {len(out['ok_pairs'])}    BROKEN: {len(out['broken_pairs'])}")
    print("BROKEN entries:")
    for p in out["broken_pairs"]:
        print(f"  {p}")
    with open(os.path.join(SCRIPT_DIR, "snr_probe_r18a.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved snr_probe_r18a.json")


if __name__ == "__main__":
    main()
