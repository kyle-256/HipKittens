#!/usr/bin/env python3
"""R18C SNR safety probe for PERSISTENT_BATCH variants on DLA1.

Use a real-data SNR test (not zero-scales) since the persistent dispatcher
could mis-claim tiles → producing 0 or non-finite. SNR ≥ 30 dB pass.
"""
import json, math, os, subprocess, sys, sysconfig
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

PARENT = "_ts_pf6_6_v12_memc"
VARIANTS = [
    "_r18c_pb1", "_r18c_pb2", "_r18c_pb4", "_r18c_pb8",
    "_r18c_pb1_g1216", "_r18c_pb2_g1216",
    "_r18c_pb1_static",
]
GPU_POOL = [5, 6, 7]
N, K = 32768, 128256  # DLA1 N,K
SMALL_M = 256


def run_one(suffix, gpu_id, with_ref=False):
    full_suffix = PARENT + suffix if suffix else PARENT
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return {"error": "missing .so", "path": so_path}
    script = f"""
import sys, math, torch, importlib.util, json, numpy as np
torch.manual_seed(42)
M, N, K = {SMALL_M}, {N}, {K}

def gen_fp4_with_data(r, K):
    c = K // 2
    return torch.randint(0, 256, (r, c), dtype=torch.uint8, device='cuda')

def preshuffle(se):
    r, kb = se.shape
    pr = math.ceil(r/64)*64
    pk = math.ceil(kb/8)*8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=se.device)
    raw[:r, :kb] = (se.to(torch.int16) + 127).to(torch.uint8)
    sh = raw.view(pr//32, 2, 16, pk//8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(pr//32, pk*32)
    sh = sh.view(pr//64, 2, pk*32//4, 4).permute(0, 2, 1, 3).contiguous()
    return sh.view(pr//64, pk*64)

spec = importlib.util.spec_from_file_location('{module_name}', '{so_path}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

A = gen_fp4_with_data(M, K)
B = gen_fp4_with_data(N, K)
sc_a = torch.zeros((M, K//32), dtype=torch.int8, device='cuda')   # scale = 1.0 (e8m0 0)
sc_b = torch.zeros((N, K//32), dtype=torch.int8, device='cuda')
A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)
C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
mod.gemm_rcr(A, B, A_sc, B_sc, C); torch.cuda.synchronize()

Cf = C.float().cpu()
finite = float(torch.isfinite(Cf).float().mean().item())
nz = float((Cf != 0).float().mean().item())
out = {{"finite": finite, "nz": nz}}
out["mean_abs"] = float(Cf.abs().mean().item())
out["std"] = float(Cf.std().item())
print(json.dumps(out))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=240, env=env)
        if r.returncode != 0:
            return {"error": f"rc={r.returncode}", "stderr": r.stderr[-400:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}


def task(args):
    suffix, gpu = args
    rref = run_one("", gpu)
    rcnd = run_one(suffix, gpu)
    verdict = "OK"
    if "error" in rcnd:
        verdict = "BROKEN"
    elif "error" in rref:
        verdict = "REF-FAIL"
    else:
        rf = rref.get("finite", 0.0); cf = rcnd.get("finite", 0.0)
        # finite parity within 5pp + non-zero parity within 10pp
        if cf < 0.5 * rf:
            verdict = "BROKEN-NAN"
        elif abs(rcnd.get("mean_abs", 0.0) - rref.get("mean_abs", 0.0)) > 0.5 * rref.get("mean_abs", 1.0) + 1e-6:
            verdict = "DELTA-MEAN"
    return suffix, gpu, rref, rcnd, verdict


def main():
    args_list = []
    for i, suffix in enumerate(VARIANTS):
        gpu = GPU_POOL[i % len(GPU_POOL)]
        args_list.append((suffix, gpu))
    print(f"Total SNR checks: {len(args_list)}")
    out = {"variants": {}}
    with ProcessPoolExecutor(max_workers=len(GPU_POOL)) as ex:
        futs = {ex.submit(task, a): a for a in args_list}
        for fut in as_completed(futs):
            suffix, gpu, rref, rcnd, verdict = fut.result()
            print(f"  {suffix:32s} gpu={gpu}  ref={rref}\n    cand={rcnd}  -> {verdict}", flush=True)
            out["variants"][suffix] = {"verdict": verdict, "gpu": gpu, "ref": rref, "cand": rcnd}
    json.dump(out, open(os.path.join(SCRIPT_DIR, "snr_probe_r18c.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
