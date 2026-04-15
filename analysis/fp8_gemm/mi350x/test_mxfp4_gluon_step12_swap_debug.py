"""Isolate fused swapped step12 vs nonfused swapped behavior."""

import math

import torch

torch.manual_seed(0)

import tk_mxfp4_gluon_cpp


M = 256
N = 256
K = 256
NUM_THREADS = 256
WARP_THREADS = 64
FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
    device="cuda",
)


def gen_fp4(rows: int, k_dim: int) -> torch.Tensor:
    cols = k_dim // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    return (hi << 4) | lo


def preshuffle_mfma16_merged(scale_exp: torch.Tensor) -> torch.Tensor:
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


def unpack(packed: torch.Tensor, k_dim: int) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    out = torch.empty(packed.shape[0], k_dim, dtype=torch.float32, device="cuda")
    out[:, 0::2] = FP4_LUT[lo]
    out[:, 1::2] = FP4_LUT[hi]
    return out


def expand(exp: torch.Tensor, k_dim: int) -> torch.Tensor:
    return torch.pow(2.0, exp.float()).repeat_interleave(32, dim=1)[:, :k_dim]


def build_inputs():
    k_blocks = K // 32
    a = gen_fp4(M, K)
    b = gen_fp4(N, K)

    # Match the active-block harness: only the first 128-K slice is live.
    a[:, K // 4 :] = 0
    b[:, K // 4 :] = 0

    sc_exp_a = torch.zeros((M, k_blocks), dtype=torch.int8, device="cuda")
    sc_exp_b = torch.zeros((N, k_blocks), dtype=torch.int8, device="cuda")
    sc_exp_a[:, : k_blocks // 2] = torch.randint(-2, 3, (M, k_blocks // 2), dtype=torch.int8, device="cuda")
    sc_exp_b[:, : k_blocks // 2] = torch.randint(-2, 3, (N, k_blocks // 2), dtype=torch.int8, device="cuda")

    a_sc = preshuffle_mfma16_merged(sc_exp_a)
    b_sc = preshuffle_mfma16_merged(sc_exp_b)
    return a, b, a_sc, b_sc, sc_exp_a, sc_exp_b


def main() -> None:
    a, b, a_sc, b_sc, sc_exp_a, sc_exp_b = build_inputs()

    layout_out = torch.empty((4 * NUM_THREADS * 4, 8), dtype=torch.float32, device="cuda")
    step12_out = torch.empty((4 * WARP_THREADS * 16, 4), dtype=torch.float32, device="cuda")
    full_nonfused = torch.empty((M, N), dtype=torch.float32, device="cuda")
    full_nonfused_permlane = torch.empty((M, N), dtype=torch.float32, device="cuda")
    full_fused = torch.empty((M, N), dtype=torch.float32, device="cuda")
    full_fused_permlane = torch.empty((M, N), dtype=torch.float32, device="cuda")

    tk_mxfp4_gluon_cpp.debug_step12_layout(a, b, a_sc, b_sc, layout_out)
    tk_mxfp4_gluon_cpp.debug_step12_swap_compare(a, b, a_sc, b_sc, step12_out)
    tk_mxfp4_gluon_cpp.debug_active_block_swap_all_nonfused(a, b, a_sc, b_sc, full_nonfused)
    tk_mxfp4_gluon_cpp.debug_active_block_swap_all_nonfused_permlane(a, b, a_sc, b_sc, full_nonfused_permlane)
    tk_mxfp4_gluon_cpp.debug_active_block_swap_all_fused(a, b, a_sc, b_sc, full_fused)
    tk_mxfp4_gluon_cpp.debug_active_block_swap_all_fused_permlane(a, b, a_sc, b_sc, full_fused_permlane)
    torch.cuda.synchronize()

    ref = (unpack(a, K) * expand(sc_exp_a, K)) @ (unpack(b, K) * expand(sc_exp_b, K)).T

    layout = layout_out.view(4, NUM_THREADS, 4, 8)
    br_ref, br_ds, a1_ref, a1_ds = layout.unbind(0)
    br_layout_err = (br_ref - br_ds).abs().max().item()
    a1_layout_err = (a1_ref - a1_ds).abs().max().item()

    step12 = step12_out.view(4, WARP_THREADS, 16, 4)
    ref_bl, fused_bl, ref_br, fused_br = step12.unbind(0)
    bl_acc_err = (ref_bl - fused_bl).abs().max().item()
    br_acc_err = (ref_br - fused_br).abs().max().item()

    full_block_err = (full_nonfused - full_fused).abs().max().item()
    full_block_nonfused_permlane_err = (full_nonfused - full_nonfused_permlane).abs().max().item()
    full_block_fused_permlane_err = (full_fused - full_fused_permlane).abs().max().item()
    full_block_permlane_err = (full_nonfused_permlane - full_fused_permlane).abs().max().item()
    full_nonfused_ref_err = (full_nonfused - ref).abs().max().item()
    full_nonfused_permlane_ref_err = (full_nonfused_permlane - ref).abs().max().item()
    full_fused_ref_err = (full_fused - ref).abs().max().item()
    full_fused_permlane_ref_err = (full_fused_permlane - ref).abs().max().item()

    print(f"Br layout max_err: {br_layout_err:.6f}, equal={torch.equal(br_ref, br_ds)}")
    print(f"A1 layout max_err: {a1_layout_err:.6f}, equal={torch.equal(a1_ref, a1_ds)}")
    print(f"Step12 A0xBl fused-vs-ref max_err: {bl_acc_err:.6f}")
    print(f"Step12 A0xBr fused-vs-ref max_err: {br_acc_err:.6f}")
    print(f"Full block fused-vs-nonfused max_err: {full_block_err:.6f}")
    print(f"Full block nonfused permlane-vs-standard max_err: {full_block_nonfused_permlane_err:.6f}")
    print(f"Full block fused permlane-vs-standard max_err: {full_block_fused_permlane_err:.6f}")
    print(f"Full block permlane fused-vs-nonfused max_err: {full_block_permlane_err:.6f}")
    print(f"Full block nonfused-vs-math max_err: {full_nonfused_ref_err:.6f}")
    print(f"Full block nonfused permlane-vs-math max_err: {full_nonfused_permlane_ref_err:.6f}")
    print(f"Full block fused-vs-math max_err: {full_fused_ref_err:.6f}")
    print(f"Full block fused permlane-vs-math max_err: {full_fused_permlane_ref_err:.6f}")

    if (
        br_layout_err == 0.0
        and a1_layout_err == 0.0
        and bl_acc_err == 0.0
        and br_acc_err == 0.0
        and full_block_nonfused_permlane_err == 0.0
        and full_block_fused_permlane_err == 0.0
        and full_block_permlane_err == 0.0
        and full_nonfused_ref_err == 0.0
        and full_nonfused_permlane_ref_err == 0.0
        and full_fused_ref_err == 0.0
        and full_fused_permlane_ref_err == 0.0
    ):
        print("Conclusion: fused swapped step12 is numerically equivalent to the nonfused swap path, and the permlane row-store matches both standard fused/nonfused stores and the math reference.")


if __name__ == "__main__":
    main()
