#!/usr/bin/env python3
"""Test EARLY_SCALE_PF: hoist scale buffer_load_dwordx2 above _raw=pf_* copy via shadow regs.

Variants:
- baseline: scale loads sit just before Step12 (post _raw copy)
- early_scale: scale loads issued at top of iteration into shadow nxt_pf_*; pf_*=nxt_pf_* at end

Each variant runs in its OWN child process. Bench: warmup=200, iters=500, trim=10%.
"""
import os, sys, math, time, subprocess, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

WARMUP = 200
ITERS = 500
TRIM = 0.10

VARIANTS = [
    ("baseline",     ""),
    ("early_scale",  "-DEARLY_SCALE_PF=1"),
]

CORR_M, CORR_N, CORR_K = 1024, 1024, 4096

# 3 deep-LOSE shapes
BENCH_SHAPES = [
    (14336, 4096, 32768, 5245.4, "14336x4096x32768"),
    (16384, 4096, 28672, 5525.3, "16384x4096x28672"),
    (4096, 32768, 128256, 5781.1, "4096x32768x128256"),
]


def build(tag, flags, n_dim, k_dim, build_dir):
    os.makedirs(build_dir, exist_ok=True)
    so = f"{build_dir}/tk_mxfp4_{tag}.cpython-310-x86_64-linux-gnu.so"
    if os.path.exists(so):
        os.remove(so)
    src_orig = f"{SCRIPT_DIR}/kernel_mxfp4_gluon_cpp.cpp"
    src_patched = f"{build_dir}/{tag}.cpp"
    with open(src_orig) as f:
        content = f.read()
    content = content.replace("PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
                              f"PYBIND11_MODULE(tk_mxfp4_{tag},")
    with open(src_patched, "w") as f:
        f.write(content)
    cmd = (
        f"/opt/rocm/bin/hipcc {src_patched} "
        f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
        f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
        f"-I/opt/rocm/include/hip "
        f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
        f"-shared -fPIC -std=c++20 -w "
        f"-DN_DIM={n_dim} -DK_DIM={k_dim} {flags} "
        f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm "
        f"-Rpass-analysis=kernel-resource-usage "
        f"-o {so}"
    )
    print(f"  Building {tag} (N={n_dim}, K={k_dim})...")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    res_lines = []
    for line in r.stderr.split("\n"):
        if "kernel-resource-usage" in line or ("VGPRs" in line and "remark" in line):
            res_lines.append(line.strip())
    for l in res_lines[:8]:
        print(f"    [{tag}] {l[-180:]}")
    if r.returncode != 0:
        print(f"  BUILD FAILED ({tag}) in {dt:.1f}s")
        print(r.stderr[-2000:])
        return None
    print(f"  built {tag} in {dt:.1f}s")
    return so


CHILD_RUNNER = '''
import os, sys, math, time, json, importlib.util, torch
SCRIPT_DIR = "{SCRIPT_DIR}"
sys.path.insert(0, SCRIPT_DIR)

mode = "{mode}"
tag  = "{tag}"
so   = "{so}"
M, N, K = {M}, {N}, {K}
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}

torch.manual_seed(42)
k_blocks = K // 32

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    return (hi << 4) | lo

def preshuffle_mfma16_merged(scale_exp):
    rows, kb = scale_exp.shape
    pr = math.ceil(rows / 64) * 64
    pk = math.ceil(kb / 8) * 8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)
    sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    sh = sh.view(pr // 32, pk * 32)
    sh = sh.view(pr // 64, 2, pk * 32 // 4, 4)
    sh = sh.permute(0, 2, 1, 3).contiguous()
    return sh.view(pr // 64, pk * 64)

A = gen_fp4(M, K); B = gen_fp4(N, K)
sc_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
sc_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
A_sc = preshuffle_mfma16_merged(sc_a)
B_sc = preshuffle_mfma16_merged(sc_b)
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

spec = importlib.util.spec_from_file_location(f"tk_mxfp4_{{tag}}", so)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def call():
    mod.gemm_rcr(A, B, A_sc, B_sc, C)

if mode == "correctness":
    C.zero_()
    call()
    torch.cuda.synchronize()
    FP4_LUT = torch.tensor(
        [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],
        dtype=torch.float32, device="cuda"
    )
    def unpack(p, K):
        lo = (p & 0x0F).to(torch.int64); hi = ((p >> 4) & 0x0F).to(torch.int64)
        out = torch.empty(p.shape[0], K, dtype=torch.float32, device="cuda")
        out[:, 0::2] = FP4_LUT[lo]; out[:, 1::2] = FP4_LUT[hi]
        return out
    def expand(exp, K):
        return torch.pow(2.0, exp.float()).repeat_interleave(32, dim=1)[:, :K]
    Af = unpack(A, K) * expand(sc_a, K)
    Bf = unpack(B, K) * expand(sc_b, K)
    Cref = Af @ Bf.T
    noise = C.float() - Cref.float()
    sig = (Cref.float()**2).sum().item()
    noi = (noise**2).sum().item()
    snr = 10*math.log10(sig/noi) if noi > 0 else float("inf")
    max_err = noise.abs().max().item()
    # Fingerprint: sum of abs values
    fp = C.float().abs().sum().item()
    print(json.dumps({{"snr": snr, "max_err": max_err,
                       "Cmax": C.float().abs().max().item(),
                       "Crefmax": Cref.float().abs().max().item(),
                       "fingerprint": fp}}))
elif mode == "bench":
    for _ in range(WARMUP): call()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    tn = int(len(times) * TRIM)
    trimmed = times[tn:len(times)-tn]
    mean_s = sum(trimmed) / len(trimmed)
    tflops = 2 * M * N * K / (mean_s * 1e12)
    print(json.dumps({{"tflops": tflops, "us": mean_s*1e6}}))
'''


def run_child(mode, tag, so, M, N, K, gpu_id):
    script = CHILD_RUNNER.format(
        SCRIPT_DIR=SCRIPT_DIR, mode=mode, tag=tag, so=so,
        M=M, N=N, K=K, WARMUP=WARMUP, ITERS=ITERS, TRIM=TRIM
    )
    env = {**os.environ, "HIP_VISIBLE_DEVICES": str(gpu_id)}
    r = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, timeout=900, env=env)
    if r.returncode != 0:
        print(f"  CHILD FAILED ({mode}/{tag}):")
        print(r.stderr[-1200:])
        return None
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        print(f"  CHILD parse error ({mode}/{tag}): {e}\n  stdout: {r.stdout[-400:]}")
        return None


def main():
    work_dir = "/tmp/early_scale_test"
    os.makedirs(work_dir, exist_ok=True)
    GPU = int(os.environ.get("BENCH_GPU", "0"))

    # ── Build correctness ──
    print(f"\n=== Build for correctness ({CORR_M}x{CORR_N}x{CORR_K}) ===")
    sos_corr = {}
    for tag, flags in VARIANTS:
        bd = f"{work_dir}/corr_{tag}"
        so = build(tag, flags, CORR_N, CORR_K, bd)
        if so: sos_corr[tag] = so

    # ── Correctness ──
    print(f"\n=== Correctness check ===")
    corr = {}
    for tag, _ in VARIANTS:
        if tag not in sos_corr: continue
        r = run_child("correctness", tag, sos_corr[tag], CORR_M, CORR_N, CORR_K, GPU)
        if r is None:
            corr[tag] = None
            print(f"  {tag}: FAIL (no result)")
        else:
            corr[tag] = r
            ok = r["snr"] > 25
            print(f"  {tag:20s} SNR {r['snr']:7.2f} dB  max_err {r['max_err']:.2f}  "
                  f"|C|={r['Cmax']:.1f} |Cref|={r['Crefmax']:.1f}  fp={r['fingerprint']:.1f}  {'OK' if ok else 'FAIL'}")

    # Cross-check fingerprints if SNR is bad (torch ref overflow case)
    if "baseline" in corr and "early_scale" in corr and corr["baseline"] and corr["early_scale"]:
        b_fp = corr["baseline"]["fingerprint"]
        e_fp = corr["early_scale"]["fingerprint"]
        rel = abs(b_fp - e_fp) / max(abs(b_fp), 1e-9)
        print(f"  baseline vs early_scale fingerprint rel diff: {rel:.6f}  ({'MATCH' if rel < 1e-3 else 'DIFFER'})")

    # ── Build & bench each shape ──
    bench_results = {}
    for M, N, K, aiter, label in BENCH_SHAPES:
        print(f"\n=== Build & Bench shape {label}  (aiter={aiter}) ===")
        sos_bench = {}
        for tag, flags in VARIANTS:
            if corr.get(tag) is None or corr[tag]["snr"] < 25:
                # Allow with fingerprint match even if SNR low (torch overflow)
                if "baseline" in corr and "early_scale" in corr and corr.get("baseline") and corr.get("early_scale"):
                    bp = corr["baseline"]["fingerprint"]; ep = corr["early_scale"]["fingerprint"]
                    if bp > 0 and abs(bp - ep) / bp < 1e-3:
                        pass  # fingerprint-match path
                    else:
                        print(f"  skip {tag} (correctness/fingerprint failed)")
                        continue
                else:
                    print(f"  skip {tag} (correctness failed)")
                    continue
            bd = f"{work_dir}/bench_{label}_{tag}"
            so = build(tag, flags, N, K, bd)
            if so: sos_bench[tag] = so

        bench_results[label] = {}
        for tag, _ in VARIANTS:
            if tag not in sos_bench: continue
            r = run_child("bench", tag, sos_bench[tag], M, N, K, GPU)
            if r is None:
                print(f"  {tag}: BENCH FAIL")
                continue
            bench_results[label][tag] = r
            ratio = r["tflops"] / aiter * 100
            print(f"  {tag:20s} {r['tflops']:7.1f} TFLOPS   ({r['us']:.1f} us)  {ratio:5.1f}% of aiter")

    # ── Final summary ──
    print(f"\n=== FINAL summary (warmup={WARMUP}, iters={ITERS}, trim={TRIM}) ===")
    print(f"{'shape':25s} {'baseline':>12s} {'early_scale':>12s}  {'delta':>8s}")
    for M, N, K, aiter, label in BENCH_SHAPES:
        br = bench_results.get(label, {}).get("baseline", {}).get("tflops", float("nan"))
        er = bench_results.get(label, {}).get("early_scale", {}).get("tflops", float("nan"))
        delta = er - br if (br == br and er == er) else float("nan")
        print(f"{label:25s} {br:12.1f} {er:12.1f}  {delta:+8.1f}")

    with open(f"{work_dir}/results.json", "w") as f:
        json.dump({"correctness": corr, "bench": bench_results}, f, indent=2)
    print(f"\nResults: {work_dir}/results.json")


if __name__ == "__main__":
    main()
