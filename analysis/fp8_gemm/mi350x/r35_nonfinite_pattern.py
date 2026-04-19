#!/usr/bin/env python3
"""R35 follow-up: probe the LARGE non-finite tier (~12%) for positional pattern.

Earlier: det_wrong=0.16% (the inf cells that happen to be deterministic).
But total non-finite is 1.88M cells = 11.24% of output. The bulk is non-deterministic.

This script counts non-finite cells per 256x256 tile, per 16x16 wave-tile, per 32x32 mfma block.
"""
import os, sys, math, importlib.util, json
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR  = os.path.join(SCRIPT_DIR, "build_all42")
sys.path.insert(0, BUILD_DIR)

FP4_E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0,
], dtype=torch.float32)

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
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

def load_module(mod_name):
    so_path = os.path.join(BUILD_DIR, f"{mod_name}.cpython-310-x86_64-linux-gnu.so")
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    M, N, K = 4096, 4096, 2048
    mod_name = "tk_mxfp4_gluon_cpp_n4096_k2048_ext_br"
    mod = load_module(mod_name)

    k_blocks = K // 32
    torch.manual_seed(42)
    A = gen_fp4(M, K); B = gen_fp4(N, K)
    sa = torch.full((M, k_blocks), -4, dtype=torch.int8, device='cuda')
    sb = torch.full((N, k_blocks), -4, dtype=torch.int8, device='cuda')
    A_sc = preshuffle_mfma16_merged(sa)
    B_sc = preshuffle_mfma16_merged(sb)

    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    mod.gemm_rcr(A, B, A_sc, B_sc, C)
    torch.cuda.synchronize()
    nonfinite = ~torch.isfinite(C.float())
    print(f"Non-finite total: {nonfinite.sum().item()} ({nonfinite.float().mean().item()*100:.2f}%)")

    # Per 256x256 tile heat (count of non-finite cells)
    M_tiles = M // 256
    N_tiles = N // 256
    nf_view = nonfinite.view(M_tiles, 256, N_tiles, 256).permute(0, 2, 1, 3).contiguous()
    tile_counts = nf_view.sum(dim=(2, 3))
    print(f"\n256x256 tile heat (per-tile non-finite count, max={256*256}=65536):")
    for r in range(M_tiles):
        row_str = ' '.join(f"{tile_counts[r,c].item():>5d}" for c in range(N_tiles))
        print(f"  M{r:2d}: {row_str}")

    # Within a 256x256 tile, where are the non-finite cells?  Average over all tiles.
    nf_within = nf_view.float().mean(dim=(0,1))  # 256x256 average
    print(f"\nMean non-finite rate within 256x256 (downsampled to 16x16 grid of 16-cell blocks):")
    nf_ds = nf_within.view(16, 16, 16, 16).mean(dim=(1,3))  # 16x16
    for r in range(16):
        row_str = ' '.join(f"{nf_ds[r,c].item()*100:>5.1f}" for c in range(16))
        print(f"  rblk{r:2d}: {row_str}")

    # Within each 16x16 sub-block, average rate per (subrow, subcol)
    print(f"\nMean non-finite rate within 16x16 (over all 256x256 sub-blocks):")
    nf_per_16 = nf_within.view(16, 16, 16, 16).mean(dim=(0,2))  # 16x16
    for r in range(16):
        row_str = ' '.join(f"{nf_per_16[r,c].item()*100:>5.1f}" for c in range(16))
        print(f"  sub{r:2d}: {row_str}")

    # Look at within-256x256 tile per-row rate
    nf_per_row_in_tile = nf_view.float().mean(dim=(0,1,3))  # 256
    print(f"\nMean non-finite rate per row offset within 256x256 tile (256 rows):")
    for r in range(0, 256, 16):
        block_avg = nf_per_row_in_tile[r:r+16].mean().item() * 100
        print(f"  rows {r:3d}..{r+16:3d}: {block_avg:6.2f}%   per-row: " +
              ' '.join(f"{nf_per_row_in_tile[r+i].item()*100:5.1f}" for i in range(16)))

    nf_per_col_in_tile = nf_view.float().mean(dim=(0,1,2))  # 256
    print(f"\nMean non-finite rate per col offset within 256x256 tile (256 cols):")
    for c in range(0, 256, 16):
        block_avg = nf_per_col_in_tile[c:c+16].mean().item() * 100
        print(f"  cols {c:3d}..{c+16:3d}: {block_avg:6.2f}%   per-col: " +
              ' '.join(f"{nf_per_col_in_tile[c+i].item()*100:5.1f}" for i in range(16)))

    # Sanity: which exact rows/cols within a tile are 100% non-finite?
    full_nf_rows = (nf_per_row_in_tile > 0.95).nonzero().squeeze(-1).tolist()
    full_nf_cols = (nf_per_col_in_tile > 0.95).nonzero().squeeze(-1).tolist()
    print(f"\nRows in tile with >95% non-finite rate: {len(full_nf_rows)}")
    print(f"  {full_nf_rows[:64]}")
    print(f"\nCols in tile with >95% non-finite rate: {len(full_nf_cols)}")
    print(f"  {full_nf_cols[:64]}")

    full_finite_rows = (nf_per_row_in_tile < 0.01).nonzero().squeeze(-1).tolist()
    full_finite_cols = (nf_per_col_in_tile < 0.01).nonzero().squeeze(-1).tolist()
    print(f"\nRows in tile with <1% non-finite rate: {len(full_finite_rows)}")
    print(f"  {full_finite_rows[:64]}")
    print(f"\nCols in tile with <1% non-finite rate: {len(full_finite_cols)}")
    print(f"  {full_finite_cols[:64]}")

if __name__ == '__main__':
    main()
