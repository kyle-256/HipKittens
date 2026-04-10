"""
SNR + Determinism test for FP8 JIT GEMM kernel (RCR / RRR / CRR layouts).

Reference: A.float() @ B.float() (FP32 accumulation, matches hardware accumulation).
SNR threshold: 48 dB (FP8 precision floor: 20*log10(2^3) ≈ 18 dB per mantissa bit).
Determinism: DET_RUNS consecutive calls must produce bitwise-identical results.
"""
import subprocess, sys, os, math

DIR = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")

# FP8 precision: eps = 2^-3 (3 mantissa bits); max SNR = 20*log10(1/eps) = 18.1 dB per element.
# In practice, with K-dimension accumulation the SNR is much higher.
# We observe ~49.6 dB for 4096x4096x4096 compared to float32 reference.
# Threshold set to 48 dB, consistent with existing FP8_SNR_THRESHOLD_DB default.
SNR_THRESHOLD = 48.0
DET_RUNS = 10
LAYOUTS = ["rcr", "rrr", "crr"]

# Representative shapes covering small, medium, large, and rectangular cases.
# All multiples of 256 (M,N) and 128 (K), K >= 256.
TEST_SHAPES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 2048, 4096),
    (2048, 4096, 4096),
]

# Per-shape subprocess script — runs correctness + determinism for one shape × layout.
_SCRIPT = r'''
import torch, sys, os, math
sys.path.insert(0, "{sodir}")
import tk_fp8_layouts as m

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

M, N, K = {M}, {N}, {K}
layout = "{lay}"

def gen(rows, cols):
    return (torch.randn(rows, cols, device="cuda") * 0.1).to(torch.float8_e4m3fn)

if layout == "rcr":
    A = gen(M, K)
    B = gen(N, K)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    ref = A.float() @ B.float().T
    def run(): m.gemm_rcr(A, B, C, 1.0, 1.0, 4)
elif layout == "rrr":
    A = gen(M, K)
    B = gen(K, N)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    ref = A.float() @ B.float()
    def run(): m.gemm_rrr(A, B, C, 1.0, 1.0, 4)
else:  # crr
    A = gen(K, M)
    B = gen(K, N)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    ref = A.float().T @ B.float()
    def run(): m.gemm_crr(A, B, C, 1.0, 1.0, 4)

torch.cuda.synchronize()

# SNR check
run()
torch.cuda.synchronize()
ref_f32 = ref.float()
C_f32   = C.float()
signal_power = (ref_f32 ** 2).sum().item()
noise_power  = ((C_f32 - ref_f32) ** 2).sum().item()
snr = 10.0 * math.log10(signal_power / (noise_power + 1e-30))

# Determinism: {det_runs} consecutive calls must be bitwise identical
runs = []
for _ in range({det_runs}):
    C.zero_()
    run()
    torch.cuda.synchronize()
    runs.append(C.clone())
det = all(torch.equal(runs[0], runs[i]) for i in range(1, {det_runs}))

print(f"{{snr:.3f}} {{int(det)}}")
'''


def run_script(script: str) -> str | None:
    so_dir = DIR
    env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": so_dir}
    try:
        r = subprocess.run(
            ["python3", "-c", script],
            capture_output=True, text=True, env=env, cwd=DIR, timeout=120,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        if r.stderr:
            print(f"    [stderr] {r.stderr[-300:]}", flush=True)
    except Exception as e:
        print(f"    [exception] {e}", flush=True)
    return None


def main():
    print(f"[Phase 1] Checking tk_fp8_layouts.so in {DIR} ...")
    import importlib, sys
    try:
        sys.path.insert(0, DIR)
        import tk_fp8_layouts  # noqa: F401
        print(f"  tk_fp8_layouts loaded OK")
    except ImportError as e:
        print(f"  ERROR: cannot import tk_fp8_layouts: {e}")
        sys.exit(1)

    print(f"\n[Phase 2] SNR + Determinism (SNR threshold={SNR_THRESHOLD} dB, runs={DET_RUNS})")
    print(f"{'M':>5} {'N':>5} {'K':>5}", end="")
    for lay in LAYOUTS:
        print(f" | {lay.upper():>3}  {'SNR':>7} {'Det':>3}", end="")
    print()
    print("-" * 75)

    all_pass = True

    for M, N, K in TEST_SHAPES:
        line = f"{M:>5} {N:>5} {K:>5}"
        row_pass = True

        for lay in LAYOUTS:
            script = _SCRIPT.format(
                sodir=DIR, M=M, N=N, K=K,
                lay=lay, det_runs=DET_RUNS,
            )
            out = run_script(script)

            if out is None:
                line += f" | {lay.upper():>3}  {'ERROR':>7} {'?':>3}"
                row_pass = False
                continue

            parts = out.split()
            snr = float(parts[0])
            det = int(parts[1])

            snr_ok = snr >= SNR_THRESHOLD
            det_ok = det == 1
            ok = snr_ok and det_ok

            snr_str = f"{snr:7.2f}"
            det_str = "OK" if det_ok else "FAIL"
            flag = "" if ok else " !"
            line += f" | {lay.upper():>3}  {snr_str} {det_str:>3}{flag}"

            if not ok:
                row_pass = False

        all_pass = all_pass and row_pass
        print(line, flush=True)

    print("=" * 75)
    if all_pass:
        print("ALL PASS")
    else:
        print("SOME FAILURES — see rows marked with !")
        sys.exit(1)


if __name__ == "__main__":
    main()
