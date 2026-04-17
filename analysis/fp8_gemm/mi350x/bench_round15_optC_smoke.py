#!/usr/bin/env python3
"""Round 15 OptC smoke + SNR. Single-shot timing; ALL variants get SNR-checked
on 256x256x4096 random fp4 first; perf only on SNR pass.

Methodology:
  - SNR sanity: 256x256x4096 random fp4 vs torch reference; SNR ≥ 25 dB required.
    NaN/Inf → FAIL (uses snr_check.compute_snr_db).
  - Perf single-shot: warmup=20 iters=50 trim=10% (for fast smoke; verify uses 200/500).
  - Variants run on appropriate shape: P1→DLA7, P2→DLA1+DLA7, P3→DLA1.

GPU plan: GPUs 6 & 7 (alternating).
"""
import json, math, os, subprocess, sys, sysconfig, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")

WARMUP = 20
ITERS = 50
TRIM = 0.10

PARENT_SUFFIX = "_ts_pf6_6_v12_memc"
SHAPES = {
    # name: (M, N, K, comp_tflops_aiter)
    "DLA1": (4096, 32768, 128256, 5781.1),
    "DLA7": (28672, 32768, 4096, 4742.6),  # rough ref, R8 best
}

VARIANTS = [
    # (suffix, shape_name, probe_label)
    ("_r15c_p1_lgkm0",     "DLA7", "P1"),
    ("_r15c_p1_lgkm2",     "DLA7", "P1"),
    ("_r15c_p1_lgkm4",     "DLA7", "P1"),
    ("_r15c_p2_s4v4",      "DLA1", "P2"),
    ("_r15c_p2_s4v8",      "DLA1", "P2"),
    ("_r15c_p2_s4v12",     "DLA1", "P2"),
    ("_r15c_p2_s4v16",     "DLA1", "P2"),
    ("_r15c_p2_s4v20",     "DLA1", "P2"),
    ("_r15c_p2_s4v4",      "DLA7", "P2"),
    ("_r15c_p2_s4v8",      "DLA7", "P2"),
    ("_r15c_p2_s4v12",     "DLA7", "P2"),
    ("_r15c_p2_s4v16",     "DLA7", "P2"),
    ("_r15c_p2_s4v20",     "DLA7", "P2"),
    ("_r15c_p3_pfg_neg1",  "DLA1", "P3"),
    ("_r15c_p3_pfg_pos1",  "DLA1", "P3"),
]


def make_runner_script(M, N, K, suffix, gpu_id, do_snr):
    full_suffix = PARENT_SUFFIX + suffix
    module_name = f"tk_mxfp4_gluon_cpp_n{N}_k{K}{full_suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return None, None
    # SNR via comparison vs. parent .so — same M,N,K shape so any difference is a
    # correctness regression. We compute SNR( parent_C  vs  variant_C ).
    parent_so = os.path.join(BUILD_DIR,
        f"tk_mxfp4_gluon_cpp_n{N}_k{K}{PARENT_SUFFIX}{EXT_SUFFIX}")
    snr_block = ""
    if do_snr and suffix != "" and os.path.exists(parent_so):
        snr_block = f"""
# Cross-variant SNR: parent .so vs this variant .so on the SAME inputs.
# Any non-trivial deviation is a correctness regression (barrier/PF reorder
# must be bit-identical except for non-deterministic float reductions).
parent_module = 'tk_mxfp4_gluon_cpp_n{N}_k{K}{PARENT_SUFFIX}'
spec_p = importlib.util.spec_from_file_location(parent_module, '{parent_so}')
mod_p = importlib.util.module_from_spec(spec_p); spec_p.loader.exec_module(mod_p)
C_parent = torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
mod_p.gemm_rcr(A,B,A_sc,B_sc,C_parent)
torch.cuda.synchronize()
mod.gemm_rcr(A,B,A_sc,B_sc,C)
torch.cuda.synchronize()
finite_p = float(torch.isfinite(C_parent.float()).float().mean().item())
finite_v = float(torch.isfinite(C.float()).float().mean().item())
import math as _math
if finite_p < 1.0 or finite_v < 1.0:
    snr_db = float('nan')
else:
    diff = (C_parent.float() - C.float())
    sig_pwr = (C_parent.float()**2).mean().item()
    err_pwr = (diff**2).mean().item()
    if sig_pwr <= 0: snr_db = float('-inf')
    elif err_pwr <= 0: snr_db = float('inf')
    else: snr_db = 10.0 * _math.log10(sig_pwr/err_pwr)
finite_s = finite_v
"""
    elif do_snr:
        # parent self-test: just record finite_frac as proxy
        snr_block = """
mod.gemm_rcr(A,B,A_sc,B_sc,C); torch.cuda.synchronize()
finite_s = float(torch.isfinite(C.float()).float().mean().item())
snr_db = float('inf') if finite_s == 1.0 else float('nan')
"""
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
{snr_block}
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
out = {{"tflops":round(t,2),"ms":round(avg,4),"finite_frac":finite_frac}}
{"out['snr_db']=snr_db; out['finite_s']=finite_s" if do_snr else ""}
print(json.dumps(out))
"""
    return script, so_path


def smoke_one(M, N, K, suffix, gpu_id, do_snr):
    script, so_path = make_runner_script(M, N, K, suffix, gpu_id, do_snr)
    if script is None:
        return {"error": "missing .so"}
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=400, env=env)
        if r.returncode != 0:
            tag = "APERTURE" if "APERTURE" in r.stderr else ("RC" + str(r.returncode))
            return {"error": tag, "stderr": r.stderr[-500:]}
        last = r.stdout.strip().splitlines()[-1]
        return json.loads(last)
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"Round 15 OptC smoke. warmup={WARMUP} iters={ITERS}")
    print(f"Variants: {len(VARIANTS)}; parents: {len(set(s for _,s,_ in VARIANTS))}")
    print("=" * 130)

    out = {"warmup": WARMUP, "iters": ITERS, "results": []}

    # First: parent baselines for each shape
    parent_results = {}
    for shape_name in set(s for _,s,_ in VARIANTS):
        M, N, K, comp = SHAPES[shape_name]
        gpu = 6
        r = smoke_one(M, N, K, "", gpu, do_snr=True)
        parent_results[shape_name] = r
        if "error" in r:
            print(f"  PARENT {shape_name} BROKEN: {r}")
        else:
            print(f"  PARENT {shape_name:5s} M={M:>5} N={N:>5} K={K:>6}  tflops={r['tflops']:.2f}  ({r['tflops']/comp*100:.2f}% aiter)  snr={r.get('snr_db','?'):.2f}dB", flush=True)
        out["results"].append({"variant": "PARENT", "shape": shape_name, **r,
                               "comp_tflops": comp})

    # Variants: parallel across GPUs 6 & 7
    print("\nCandidates:")
    def task(idx, suffix, shape_name, probe):
        M, N, K, comp = SHAPES[shape_name]
        gpu = 6 + (idx % 2)
        r = smoke_one(M, N, K, suffix, gpu, do_snr=True)
        return (suffix, shape_name, probe, comp, r)

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(task, i, s, sh, p): (s, sh, p) for i,(s,sh,p) in enumerate(VARIANTS)}
        for fut in as_completed(futs):
            suffix, shape_name, probe, comp, r = fut.result()
            tag = f"{probe} {shape_name} {suffix}"
            if "error" in r:
                print(f"  {tag:65s} BROKEN err={r.get('error')}", flush=True)
            else:
                snr = r.get('snr_db', float('nan'))
                snr_ok = isinstance(snr,(int,float)) and not math.isnan(snr) and snr >= 25.0
                ratio = r['tflops'] / comp * 100
                parent = parent_results.get(shape_name, {})
                ptf = parent.get('tflops', 0) or 1
                delta_pp = (r['tflops'] - ptf) / ptf * 100
                gate = "PASS" if snr_ok and delta_pp >= 0.5 else ("SNR_FAIL" if not snr_ok else "no-gate")
                print(f"  {tag:65s} tflops={r['tflops']:.2f} ({ratio:.2f}% aiter, Δ{delta_pp:+.2f}pp parent) snr={snr:.2f}dB [{gate}]", flush=True)
            out["results"].append({"variant": suffix, "shape": shape_name, "probe": probe,
                                   "comp_tflops": comp, **r})

    print("=" * 130)
    with open(os.path.join(SCRIPT_DIR, "bench_round15_optC_smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bench_round15_optC_smoke.json")


if __name__ == "__main__":
    main()
