"""Validate the active-block harness and operand-swap block POC."""

import math

import torch

torch.manual_seed(0)

import tk_mxfp4_gluon_cpp


FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
    device="cuda",
)

M = 256
N = 256
K = 256


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


def main() -> None:
    k_blocks = K // 32
    a = gen_fp4(M, K)
    b = gen_fp4(N, K)

    # Zero the back half so the harness isolates the first 128-K slice exactly.
    a[:, K // 4 :] = 0
    b[:, K // 4 :] = 0

    sc_exp_a = torch.zeros((M, k_blocks), dtype=torch.int8, device="cuda")
    sc_exp_b = torch.zeros((N, k_blocks), dtype=torch.int8, device="cuda")
    sc_exp_a[:, : k_blocks // 2] = torch.randint(-2, 3, (M, k_blocks // 2), dtype=torch.int8, device="cuda")
    sc_exp_b[:, : k_blocks // 2] = torch.randint(-2, 3, (N, k_blocks // 2), dtype=torch.int8, device="cuda")

    a_sc = preshuffle_mfma16_merged(sc_exp_a)
    b_sc = preshuffle_mfma16_merged(sc_exp_b)
    active_out = torch.empty((M, N), dtype=torch.float32, device="cuda")
    active_swap_step34_out = torch.empty((M, N), dtype=torch.float32, device="cuda")
    active_swap_step34_with_lds_out = torch.empty((M, N), dtype=torch.float32, device="cuda")
    active_swap_step34_mainlike_out = torch.empty((M, N), dtype=torch.float32, device="cuda")
    active_swap_all_nonfused_out = torch.empty((M, N), dtype=torch.float32, device="cuda")
    swap_out = torch.empty((128, 64), dtype=torch.float32, device="cuda")

    tk_mxfp4_gluon_cpp.debug_active_block(a, b, a_sc, b_sc, active_out)
    tk_mxfp4_gluon_cpp.debug_active_block_swap_step34(a, b, a_sc, b_sc, active_swap_step34_out)
    tk_mxfp4_gluon_cpp.debug_active_block_swap_step34_with_lds(a, b, a_sc, b_sc, active_swap_step34_with_lds_out)
    tk_mxfp4_gluon_cpp.debug_active_block_swap_step34_mainlike(a, b, a_sc, b_sc, active_swap_step34_mainlike_out)
    tk_mxfp4_gluon_cpp.debug_active_block_swap_all_nonfused(a, b, a_sc, b_sc, active_swap_all_nonfused_out)
    tk_mxfp4_gluon_cpp.debug_operand_swap_block(a, b, a_sc, b_sc, swap_out)
    torch.cuda.synchronize()

    ref = (unpack(a, K) * expand(sc_exp_a, K)) @ (unpack(b, K) * expand(sc_exp_b, K)).T
    ref00 = ref[:64, :64]

    active_err = (active_out - ref).abs().max().item()
    active_swap_step34_err = (active_swap_step34_out - ref).abs().max().item()
    active_swap_step34_with_lds_err = (active_swap_step34_with_lds_out - ref).abs().max().item()
    active_swap_step34_mainlike_err = (active_swap_step34_mainlike_out - ref).abs().max().item()
    active_swap_all_nonfused_err = (active_swap_all_nonfused_out - ref).abs().max().item()
    orig_block_err = (swap_out[:64, :] - ref00).abs().max().item()
    swap_block_err = (swap_out[64:, :] - ref00).abs().max().item()

    print(f"debug_active_block max_err: {active_err:.6f}")
    print(f"debug_active_block_swap_step34 max_err: {active_swap_step34_err:.6f}")
    print(f"debug_active_block_swap_step34_with_lds max_err: {active_swap_step34_with_lds_err:.6f}")
    print(f"debug_active_block_swap_step34_mainlike max_err: {active_swap_step34_mainlike_err:.6f}")
    print(f"debug_active_block_swap_all_nonfused max_err: {active_swap_all_nonfused_err:.6f}")
    print(f"debug_operand_swap_block orig max_err: {orig_block_err:.6f}")
    print(f"debug_operand_swap_block swap_sel max_err: {swap_block_err:.6f}")

    ok = (
        active_err == 0.0
        and active_swap_step34_err == 0.0
        and active_swap_step34_with_lds_err == 0.0
        and active_swap_step34_mainlike_err == 0.0
        and active_swap_all_nonfused_err == 0.0
        and orig_block_err == 0.0
        and swap_block_err == 0.0
    )
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
