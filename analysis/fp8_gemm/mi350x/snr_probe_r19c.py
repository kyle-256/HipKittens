#!/usr/bin/env python3
"""R19C SNR probe — same noise-floor methodology as R18A.

Phase 1: parent run #1 (small fp4 inputs, uniform 2^-3 scales) → save tile.
Phase 2: parent run #2 → noise-floor SNR vs run #1.
Phase 3: each variant run once → SNR vs parent #1.

Verdict (per R18A):
  - inf SNR              ⇒ OK-PERFECT
  - SNR >= noise-6dB AND >= 20dB absolute ⇒ OK
  - SNR >= 15dB          ⇒ OK-MARGINAL
  - else                 ⇒ BROKEN-RACE
  - aperture violation   ⇒ BROKEN-APERTURE

GPUs: 6 and 7 (per task constraints; do NOT touch 0-5).
"""
import json, math, os, subprocess, sys, sysconfig, time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

# (tag, M, N, K, parent_iterilp_suffix)
SHAPES = [
    ("S1", 14336, 4096, 32768, "_lgk2_dc_r10_iterilp"),
    ("S2", 16384, 4096, 28672, "_u32_r10_iterilp"),
    ("S3",  4096, 32768, 28672, "_v20_memc_r11_iterilp"),
    ("S4",  4096, 28672, 32768, "_u16_r11_iterilp"),
    ("S5",  4096, 32768, 14336, "_ts_lgk2_memc_r11_iterilp"),
]

# (tag, parent_macro_suffix)  -- the suffix used in build_round19_optC.py
PARENT_BUILD_SUFFIX = {
    "S1": "_lgk2_dc",
    "S2": "_u32",
    "S3": "_v20_memc",
    "S4": "_u16",
    "S5": "_ts_lgk2_memc",
}

VARIANTS = ["step3", "step12", "all"]
GPU_POOL = [6, 7]


def make_script(M, N, K, suffix, save_path):
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
    tag, M, N, K, parent_build_suffix, btw_tag, gpu, tmpdir = args
    cand_suffix = f"{parent_build_suffix}_r19c_iterilp_btw_{btw_tag}"
    parent_path = os.path.join(tmpdir, f"parent1_{tag}.npy")
    cand_path = os.path.join(tmpdir, f"cand_{tag}_{btw_tag}.npy")

    rcand = run_one(M, N, K, cand_suffix, gpu, cand_path)
    if "error" in rcand:
        return tag, btw_tag, gpu, rcand, None, None
    snr, why = compute_snr(parent_path, cand_path)
    return tag, btw_tag, gpu, None, rcand, (snr, why)


def main():
    tmpdir = os.path.join(SCRIPT_DIR, "snr_probe_r19c_tiles")
    os.makedirs(tmpdir, exist_ok=True)

    print("=== Phase 1: parent reference (run 1) on the iterilp parent ===")
    for (tag, M, N, K, ps) in SHAPES:
        path = os.path.join(tmpdir, f"parent1_{tag}.npy")
        meta_path = os.path.join(tmpdir, f"parent1_{tag}.json")
        if os.path.exists(path) and os.path.exists(meta_path):
            print(f"  {tag}: cached parent1")
            continue
        gpu = GPU_POOL[0]
        print(f"  {tag}: M={M} N={N} K={K} parent={ps} on gpu={gpu}", flush=True)
        rref = run_one(M, N, K, ps, gpu, path)
        print(f"    {rref}", flush=True)
        if "error" not in rref:
            with open(meta_path, "w") as f: json.dump(rref, f)

    print("\n=== Phase 2: parent reference (run 2) — noise floor ===")
    noise_floor = {}
    for (tag, M, N, K, ps) in SHAPES:
        path = os.path.join(tmpdir, f"parent2_{tag}.npy")
        meta_path = os.path.join(tmpdir, f"parent2_{tag}.json")
        if not (os.path.exists(path) and os.path.exists(meta_path)):
            gpu = GPU_POOL[0]
            print(f"  {tag}: rerun parent on gpu={gpu}", flush=True)
            r = run_one(M, N, K, ps, gpu, path)
            print(f"    {r}", flush=True)
            if "error" not in r:
                with open(meta_path, "w") as f: json.dump(r, f)
        snr, why = compute_snr(os.path.join(tmpdir, f"parent1_{tag}.npy"), path)
        noise_floor[tag] = snr
        print(f"  {tag}: noise-floor SNR = {snr if snr is None else round(snr, 2)} dB ({why})", flush=True)

    print("\n=== Phase 3: per-variant SNR ===")
    args_list = []
    i = 0
    for (tag, M, N, K, ps) in SHAPES:
        for btw_tag in VARIANTS:
            gpu = GPU_POOL[i % len(GPU_POOL)]
            args_list.append((tag, M, N, K, PARENT_BUILD_SUFFIX[tag], btw_tag, gpu, tmpdir))
            i += 1

    out = {"shapes": {}, "noise_floor": noise_floor,
           "ok_pairs": [], "broken_pairs": []}
    for (tag, _, _, _, _) in SHAPES:
        out["shapes"][tag] = {"variants": {}, "noise_floor_db": noise_floor.get(tag)}
    print(f"Total checks: {len(args_list)}")
    print("=" * 110)
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(task, a): a for a in args_list}
        for fut in as_completed(futs):
            tag, btw_tag, gpu, ref_err, rcand, snr_info = fut.result()
            entry = {"gpu": gpu, "cand_run": rcand, "ref_err": ref_err}
            if snr_info is not None:
                snr, why = snr_info
                entry["snr_db"] = (None if snr is None else (float("inf") if snr == float("inf") else round(snr, 2)))
                entry["snr_why"] = why
            if ref_err is not None or "error" in (rcand or {}):
                verdict = "BROKEN-APERTURE"
            elif entry.get("snr_db") is None:
                verdict = "BROKEN-NO-SNR"
            else:
                snr_v = entry["snr_db"]
                nf = noise_floor.get(tag)
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
            out["shapes"][tag]["variants"][btw_tag] = entry
            sn = entry.get("snr_db", "?")
            nf_str = noise_floor.get(tag, "?")
            sn_disp = sn if sn != float("inf") else "inf"
            print(f"  {tag:3s} {btw_tag:7s}  gpu={gpu}  snr={sn_disp!s:>10s} dB  noise={nf_str!s:>10s} dB  {verdict}", flush=True)
            if verdict.startswith("OK"):
                out["ok_pairs"].append([tag, btw_tag, verdict])
            else:
                out["broken_pairs"].append([tag, btw_tag, verdict])

    print("\n" + "=" * 110)
    print(f"OK: {len(out['ok_pairs'])}    BROKEN: {len(out['broken_pairs'])}")
    print("OK entries:")
    for p in out["ok_pairs"]:
        print(f"  {p}")
    print("BROKEN entries:")
    for p in out["broken_pairs"]:
        print(f"  {p}")
    with open(os.path.join(SCRIPT_DIR, "snr_probe_r19c.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved snr_probe_r19c.json")


if __name__ == "__main__":
    main()
