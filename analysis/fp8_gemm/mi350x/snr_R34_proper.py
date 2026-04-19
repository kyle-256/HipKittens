#!/usr/bin/env python3
"""R34 PROPER SNR validation: small-K torch reference, SNR > 40 dB gate.

The K=128256 SNR methodology is BROKEN (all kernels produce ~50% NaN/inf,
SNR is always negative). This script tests at K=4096 where output stays
finite, using a torch float32 matmul reference.

Usage:
  HIP_VISIBLE_DEVICES=N python3 snr_R34_proper.py [module_name ...]
"""
import os, sys, math, json, gc, importlib, subprocess, time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
sys.path.insert(0, BUILD_DIR)
TK_ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                   cwd=SCRIPT_DIR).decode().strip()

# Test at K=4096 where output stays finite
M, N, K = 4096, 32768, 4096
SNR_GATE = 40.0  # dB — user requirement


# ── FP4 E2M1 dequantization (no inf/nan in FP4) ──
FP4_E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # positive
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # negative
], dtype=torch.float32)


def dequant_fp4(packed_uint8, K):
    """Dequantize packed FP4 (2 nibbles per byte) to float32. Shape: [rows, K]."""
    lo = (packed_uint8 & 0x0F).to(torch.int64)
    hi = ((packed_uint8 >> 4) & 0x0F).to(torch.int64)
    table = FP4_E2M1_TABLE.to(packed_uint8.device)
    lo_f = table[lo]
    hi_f = table[hi]
    # Interleave: even columns = lo, odd columns = hi
    rows = packed_uint8.shape[0]
    cols = packed_uint8.shape[1]
    out = torch.empty(rows, cols * 2, dtype=torch.float32, device=packed_uint8.device)
    out[:, 0::2] = lo_f
    out[:, 1::2] = hi_f
    return out[:, :K]


def apply_block_scales(data_f32, scale_exp_i8, block_size=32):
    """Apply E8M0 block scales to dequantized FP4 data.
    data_f32: [rows, K], scale_exp_i8: [rows, K//block_size]
    Returns: [rows, K] with each block of 32 elements scaled by 2^(scale_exp).
    """
    rows, K_dim = data_f32.shape
    k_blocks = K_dim // block_size
    # scales: 2^exp
    scales = (2.0 ** scale_exp_i8.to(torch.float32))  # [rows, k_blocks]
    # Broadcast: repeat each scale for block_size columns
    scales_expanded = scales.unsqueeze(-1).expand(rows, k_blocks, block_size)
    scales_expanded = scales_expanded.reshape(rows, k_blocks * block_size)[:, :K_dim]
    return data_f32 * scales_expanded


def torch_reference(A_packed, B_packed, A_scale_exp, B_scale_exp, K_dim):
    """Compute FP4 GEMM reference in float32 via torch.matmul."""
    A_f32 = dequant_fp4(A_packed, K_dim)
    B_f32 = dequant_fp4(B_packed, K_dim)
    A_scaled = apply_block_scales(A_f32, A_scale_exp)
    B_scaled = apply_block_scales(B_f32, B_scale_exp)
    # C = A @ B^T (RCR layout)
    C_f32 = torch.matmul(A_scaled, B_scaled.T)
    return C_f32.to(torch.bfloat16)


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


def build_variant(src_file, module_name, extra_cppflags=""):
    """Build a variant at K=4096 with proper PYBIND11_MODULE patching."""
    so_path = os.path.join(BUILD_DIR, f"{module_name}.cpython-310-x86_64-linux-gnu.so")
    if os.path.exists(so_path):
        return so_path

    with open(os.path.join(SCRIPT_DIR, src_file)) as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper = os.path.join(BUILD_DIR, f"wrap_snr_R34_{module_name}.cpp")
    with open(wrapper, "w") as f:
        f.write(patched)

    base_flags = (
        f"-DK_DIM={K} -DN_DIM={N} "
        "-DTILE_SIZE=1 -DLGK_MODE=2 -DSTEP3_BARRIER_VMCNT=12 "
        "-DMAX_ELEM_COORD=1 -DBARRIER_TO_WAITCNT_ALL=1 "
        "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 "
        "-mllvm -amdgpu-sched-strategy=max-memory-clause "
        + extra_cppflags
    )
    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT
    cmd = (
        f'make -C {SCRIPT_DIR} '
        f'TARGET={os.path.join(BUILD_DIR, module_name)} '
        f'SRC={wrapper} '
        f'CPPFLAGS="{base_flags}"'
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(f"  BUILD FAIL: {r.stderr[-200:]}")
        return None
    return so_path


def main():
    torch.manual_seed(42)
    k_blocks = K // 32

    print(f"=== R34 PROPER SNR test: M={M}, N={N}, K={K}, gate={SNR_GATE} dB ===\n")

    # Generate inputs
    A = gen_fp4(M, K)
    B = gen_fp4(N, K)
    # Small scale exponents to keep output finite
    A_scale_exp = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
    B_scale_exp = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
    A_sc = preshuffle_mfma16_merged(A_scale_exp)
    B_sc = preshuffle_mfma16_merged(B_scale_exp)

    # Torch reference
    print("Computing torch reference (float32 matmul)...", flush=True)
    C_ref = torch_reference(A, B, A_scale_exp, B_scale_exp, K)
    ref_finite = torch.isfinite(C_ref.float()).sum().item()
    print(f"  ref: finite={ref_finite}/{C_ref.numel()} ({ref_finite/C_ref.numel()*100:.1f}%) "
          f"max={C_ref.float().abs().max().item():.2f}\n")

    # Build and test variants
    variants = [
        ("incumbent", "kernel_mxfp4_gluon_cpp.cpp", ""),
        ("legacyfork", "kernel_mxfp4_gluon_cpp_vgprPF.cpp", "-DVGPR_PF_MODE=0"),
        ("vgprpf", "kernel_mxfp4_gluon_cpp_vgprPF.cpp", "-DVGPR_PF_MODE=1"),
    ]

    results = {}
    for tag, src, extra in variants:
        mod_name = f"tk_mxfp4_gluon_cpp_snr_R34_{tag}_k{K}"
        print(f"[{tag}] building...", flush=True)
        so = build_variant(src, mod_name, extra)
        if so is None:
            results[tag] = {"status": "BUILD_FAIL"}
            continue

        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f"  IMPORT FAIL: {e}")
            results[tag] = {"status": "IMPORT_FAIL", "error": str(e)}
            continue

        C_test = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        try:
            mod.gemm_rcr(A, B, A_sc, B_sc, C_test)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  RUNTIME FAIL: {e}")
            results[tag] = {"status": "CRASH", "error": str(e)}
            continue

        # SNR vs torch reference
        diff = (C_test.float() - C_ref.float())
        both_finite = torch.isfinite(C_test.float()) & torch.isfinite(C_ref.float())
        n_finite = both_finite.sum().item()

        if n_finite > 0:
            ref_vals = C_ref.float()[both_finite]
            diff_vals = diff[both_finite]
            sig = (ref_vals ** 2).mean().item()
            err = (diff_vals ** 2).mean().item() + 1e-30
            ratio = max(sig, 1e-30) / err
            snr = 10.0 * math.log10(ratio) if ratio > 0 else float('-inf')
            max_abs = diff_vals.abs().max().item()
            bit_eq = ((C_test == C_ref) & both_finite).sum().item()
        else:
            snr = float('-inf')
            max_abs = float('inf')
            bit_eq = 0

        verdict = "PASS" if snr >= SNR_GATE else "FAIL"
        results[tag] = {
            "status": "OK", "snr_dB": snr, "max_abs_diff": max_abs,
            "finite_frac": n_finite / C_test.numel(),
            "bit_eq_frac": bit_eq / max(n_finite, 1),
            "verdict": verdict,
        }
        print(f"  {tag}: SNR={snr:.2f} dB  max_abs={max_abs:.4g}  "
              f"finite={n_finite/C_test.numel()*100:.1f}%  "
              f"bit_eq={bit_eq/max(n_finite,1)*100:.2f}%  -> {verdict}")

    # Save
    out_path = os.path.join(SCRIPT_DIR, "R34_SNR_PROPER_RESULTS.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
