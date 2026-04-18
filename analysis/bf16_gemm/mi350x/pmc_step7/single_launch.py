"""Single-launch script for rocprofv3 PMC capture (P23 Step 7).

Profiles ONE BF16 kernel call on a chosen (M, N, K, layout) so rocprofv3
captures exactly one main GEMM dispatch. Uses the existing autotuned
(group_m, num_xcds) from `bench_bf16_no_jit_final.json`.

Usage:
  python3 single_launch.py <layout> <shape_tag>
    layout    = rcr | crr | rrr
    shape_tag = m4096n28672k4096 | m8192n22016k4096 | m8192n28672k4096

Reused from /tmp/p21_dev_c/single_launch.py and /tmp/p21_dev_e/single_launch.py
(P21 Dev C / Dev E PMC characterizations).
"""
import sys, os, torch

torch.manual_seed(42)

# Use the package directly from the repo's analysis dir (must be pre-built `make`).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import tk_bf16_layouts  # noqa: E402

SHAPES = {
    "m4096n28672k4096": (4096, 28672, 4096),
    "m8192n22016k4096": (8192, 22016, 4096),
    "m8192n28672k4096": (8192, 28672, 4096),
}

# (group_m, num_xcds) from bench_bf16_no_jit_final.json autotune
CFG = {
    ("rcr", "m4096n28672k4096"): (4, 16),
    ("rcr", "m8192n22016k4096"): (2, 32),
    ("rcr", "m8192n28672k4096"): (16, 4),
    ("crr", "m4096n28672k4096"): (4, 16),
    ("crr", "m8192n22016k4096"): (2, 32),
    ("crr", "m8192n28672k4096"): (24, 2),
    ("rrr", "m4096n28672k4096"): (4, 16),
    ("rrr", "m8192n22016k4096"): (2, 32),
    ("rrr", "m8192n28672k4096"): (16, 4),
}

if len(sys.argv) != 3:
    raise SystemExit(__doc__)

layout, shape_tag = sys.argv[1], sys.argv[2]
M, N, K = SHAPES[shape_tag]
gm, xcd = CFG[(layout, shape_tag)]

if layout == "rcr":
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    fn = tk_bf16_layouts.gemm_rcr
elif layout == "rrr":
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    fn = tk_bf16_layouts.gemm_rrr
elif layout == "crr":
    A = torch.randn(K, M, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    fn = tk_bf16_layouts.gemm_crr
else:
    raise SystemExit(f"unknown layout {layout}")

C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()
fn(A, B, C, gm, xcd)
torch.cuda.synchronize()
print(f"done {layout} {shape_tag} M={M} N={N} K={K} gm={gm} xcd={xcd}")
