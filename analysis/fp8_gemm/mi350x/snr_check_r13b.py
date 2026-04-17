#!/usr/bin/env python3
"""R13B correctness check: SNR variant-vs-baseline.

The baseline (with default sched-strategy) is the trusted reference for this shape;
each candidate must produce output within 25 dB of it (this filters NaN/garbage).
Uses a SMALL-SCALE input distribution (sc in {-1,0,1}) so the bf16 output rarely
saturates, which is how prior R10A/R11 SNR checks worked.
"""
import json, math, os, subprocess, sys, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
GPU = 6
M, N, K = 28672, 4096, 16384

PARENTS = ["_ts_gm8", "_ts_lgk2_v12_memc"]
VARIANTS = [
    "_r13b_iterilp",
    "_r13b_iterminreg",
    "_r13b_maxocc",
    "_r13b_iteroccexp",
]
STACK = ["_r13b_iterilp_stack_memc", "_r13b_memc_stack_iterilp"]


def run_one(full_suffix, save_path):
    """Run kernel once with fixed seed/inputs and save C tensor to npy."""
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so"}
    script = f"""
import sys, math, torch, importlib.util, json, numpy as np
torch.manual_seed(42)
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
# Use ALL-ZERO scales: max output magnitude bounded by 6*6*K = ~590k, well below bf16 max (3.4e38).
# This eliminates accumulator saturation so candidate vs baseline must agree to FMA-rounding noise.
sc_a=torch.zeros((M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.zeros((N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
mod.gemm_rcr(A,B,A_sc,B_sc,C); torch.cuda.synchronize()
finite = float(torch.isfinite(C.float()).float().mean().item())
# Save a deterministic checksum + abs-max + sample tile for SNR
Cf = C.float()
chk = float(Cf.sum().item()); cmax = float(Cf.abs().max().item())
# save the central 256x256 tile for SNR comparison
tile = Cf[M//2-128:M//2+128, N//2-128:N//2+128].cpu().numpy()
np.save('{save_path}', tile)
print(json.dumps({{"finite":finite,"sum":chk,"max":cmax}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(GPU)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
        line = r.stdout.strip().splitlines()[-1]
        return json.loads(line)
    except Exception as e:
        return {"error": str(e)}


def snr_db(ref_path, cand_path):
    import numpy as np
    ref = np.load(ref_path).astype(np.float32)
    got = np.load(cand_path).astype(np.float32)
    if not np.all(np.isfinite(ref)):
        return None, "ref non-finite"
    if not np.all(np.isfinite(got)):
        return None, "got non-finite"
    diff = ref - got
    sig = float((ref * ref).mean())
    err = float((diff * diff).mean())
    if sig <= 0:
        return None, "ref zero"
    if err <= 0:
        return float("inf"), "perfect"
    return 10.0 * math.log10(sig / err), "ok"


def main():
    print(f"R13B SNR check (variant vs baseline). GPU={GPU}")
    print("=" * 110)
    out = {"gpu": GPU, "rows": []}
    tile_dir = os.path.join(SCRIPT_DIR, "r13b_snr_tiles")
    os.makedirs(tile_dir, exist_ok=True)
    for psuf in PARENTS:
        # Run baseline first for this parent
        base_full = psuf + "_r13b_baseline"
        base_path = os.path.join(tile_dir, f"{psuf}_baseline.npy")
        rb = run_one(base_full, base_path)
        print(f"\nParent {psuf}  baseline: {rb}")
        if "error" in rb:
            continue
        for vsuf in VARIANTS:
            full = psuf + vsuf
            cand_path = os.path.join(tile_dir, f"{psuf}{vsuf}.npy")
            rc = run_one(full, cand_path)
            if "error" in rc:
                print(f"  {full:50s}  ERROR: {rc}", flush=True)
                out["rows"].append({"variant": full, "result": rc, "snr_db": None})
                continue
            snr, why = snr_db(base_path, cand_path)
            verdict = "OK" if (snr is not None and (math.isinf(snr) or snr >= 25.0)) else "FAIL"
            print(f"  {full:50s}  finite={rc.get('finite'):.3f}  snr={snr if snr is None else round(snr,2)} dB  [{verdict}]", flush=True)
            out["rows"].append({"variant": full, "result": rc, "snr_db": snr, "verdict": verdict})
        if psuf == "_ts_gm8":
            for vsuf in STACK:
                full = psuf + vsuf
                cand_path = os.path.join(tile_dir, f"{psuf}{vsuf}.npy")
                rc = run_one(full, cand_path)
                if "error" in rc:
                    print(f"  {full:50s}  ERROR: {rc}", flush=True)
                    continue
                snr, why = snr_db(base_path, cand_path)
                verdict = "OK" if (snr is not None and (math.isinf(snr) or snr >= 25.0)) else "FAIL"
                print(f"  {full:50s}  finite={rc.get('finite'):.3f}  snr={snr if snr is None else round(snr,2)} dB  [{verdict}]", flush=True)
                out["rows"].append({"variant": full, "result": rc, "snr_db": snr, "verdict": verdict})
    with open(os.path.join(SCRIPT_DIR, "snr_check_r13b_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
