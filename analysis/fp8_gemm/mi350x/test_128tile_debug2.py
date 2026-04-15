#!/usr/bin/env python3
"""Debug: compare with small K to isolate the iteration causing errors."""
import math, sys, os, torch
torch.manual_seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# We need separate builds for different K values
import subprocess, sysconfig

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/shared_nfs/kyle/test/HipKittens")
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

def build_kernel(src_file, module_name, n_dim, k_dim, extra=""):
    """Build a kernel with specific N,K dims."""
    out = os.path.join(SCRIPT_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(out):
        return
    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT
    cmd = (f'make -C {SCRIPT_DIR} TARGET={os.path.join(SCRIPT_DIR, module_name)} '
           f'SRC={os.path.join(SCRIPT_DIR, src_file)} '
           f'CPPFLAGS="-DK_DIM={k_dim} -DN_DIM={n_dim} {extra}"')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(f"BUILD FAILED: {r.stderr[-500:]}")
        sys.exit(1)

# Build both kernels for K=256 (minimum: BK=128 bytes = 256 FP4 elements, 1 iteration)
N = 32768

# The minimum K for 1 iteration is BK*2 = 256 bytes = 512 FP4 elements (K_DIM=512 does K_BYTES=256, k_byte_iters=2)
# Actually: K_BYTES = K_DIM/2 = K/2 bytes, k_byte_iters = K_BYTES/BK = K/(2*128) = K/256
# For k_byte_iters=1: K = 256
# For k_byte_iters=2: K = 512

for K in [256, 512, 1024]:
    print(f"\n{'='*60}")
    print(f"Testing K={K} (k_byte_iters={K//256})")
    print(f"{'='*60}")

    # Build with patched module names
    ref_module = f"tk_ref_k{K}"
    tile_module = f"tk_128_k{K}"

    # Create patched source for ref
    ref_src = os.path.join(SCRIPT_DIR, f"_ref_k{K}.cpp")
    tile_src = os.path.join(SCRIPT_DIR, f"_128_k{K}.cpp")

    for src_in, src_out, mod_old, mod_new in [
        ("kernel_mxfp4_gluon_cpp.cpp", ref_src, "tk_mxfp4_gluon_cpp", ref_module),
        ("kernel_mxfp4_128tile.cpp", tile_src, "tk_mxfp4_128tile", tile_module),
    ]:
        with open(os.path.join(SCRIPT_DIR, src_in)) as f:
            code = f.read()
        code = code.replace(f"PYBIND11_MODULE({mod_old},", f"PYBIND11_MODULE({mod_new},")
        with open(src_out, "w") as f:
            f.write(code)

    build_kernel(f"_ref_k{K}.cpp", ref_module, N, K)
    build_kernel(f"_128_k{K}.cpp", tile_module, N, K)

    # Import
    if ref_module in sys.modules:
        del sys.modules[ref_module]
    if tile_module in sys.modules:
        del sys.modules[tile_module]

    ref_mod = __import__(ref_module)
    tile_mod = __import__(tile_module)

    M = 256
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

    A = gen_fp4(M, K)
    B = gen_fp4(N, K)
    sc_exp_a = torch.zeros(M, k_blocks, dtype=torch.int8, device="cuda")  # unit scales
    sc_exp_b = torch.zeros(N, k_blocks, dtype=torch.int8, device="cuda")
    A_sc = preshuffle_mfma16_merged(sc_exp_a)
    B_sc = preshuffle_mfma16_merged(sc_exp_b)

    C_ref = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    C_128 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    ref_mod.gemm_rcr(A, B, A_sc, B_sc, C_ref)
    tile_mod.gemm_rcr(A, B, A_sc, B_sc, C_128)
    torch.cuda.synchronize()

    diff = (C_ref.float() - C_128.float()).abs()
    max_diff = diff.max().item()
    exact = (C_ref == C_128).all().item()
    print(f"  Unit scales: max_diff={max_diff:.4f}, exact={exact}")

    if max_diff > 0.5:
        idx = diff.argmax()
        row = idx.item() // N
        col = idx.item() % N
        print(f"  Max diff at [{row}, {col}]")

        # Check per-block (128x128 tile, then 64x64 half, then 32x32 warp)
        for mh in range(2):
            for nh in range(2):
                r0 = mh * 128
                c0 = nh * (N // 2)
                r1 = min(r0 + 128, M)
                c1 = c0 + (N // 2)
                bd = diff[r0:r1, c0:c1].max().item()
                print(f"    mh={mh},nh={nh}: max_diff={bd:.4f}")

print("\nDone!")
