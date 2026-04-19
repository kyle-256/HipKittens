#!/usr/bin/env python3
"""R35: Locate the ~17% deterministic-wrong cells and find positional pattern.

Setup (from R34 findings):
  M=N=4096, K=2048
  module: tk_mxfp4_gluon_cpp_n4096_k2048_ext_br
  n_runs=5, scale_mode=const_-4

Output:
  - count, percentage of det-wrong cells
  - histograms by (row mod 16, col mod 16, row mod 32, ...)
  - 256x256 tile heatmap
  - per-row / per-col aggregate
  - dump first 50 wrong-cell coordinates
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

def dequant_fp4(packed_uint8, K):
    lo = (packed_uint8 & 0x0F).to(torch.int64)
    hi = ((packed_uint8 >> 4) & 0x0F).to(torch.int64)
    table = FP4_E2M1_TABLE.to(packed_uint8.device)
    rows, cols = packed_uint8.shape
    out = torch.empty(rows, cols * 2, dtype=torch.float32, device=packed_uint8.device)
    out[:, 0::2] = table[lo]
    out[:, 1::2] = table[hi]
    return out[:, :K]

def apply_block_scales(data_f32, scale_exp_i8, block_size=32):
    rows, K_dim = data_f32.shape
    k_blocks = K_dim // block_size
    scales = (2.0 ** scale_exp_i8.to(torch.float32))
    scales_expanded = scales.unsqueeze(-1).expand(rows, k_blocks, block_size)
    return scales_expanded.reshape(rows, -1)[:, :K_dim] * data_f32

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

def torch_ref(A, B, sa, sb, K):
    A_f = dequant_fp4(A, K)
    B_f = dequant_fp4(B, K)
    A_s = apply_block_scales(A_f, sa)
    B_s = apply_block_scales(B_f, sb)
    return torch.matmul(A_s, B_s.T).to(torch.bfloat16)

def main():
    M, N, K = 4096, 4096, 2048
    n_runs = 5
    mod_name = "tk_mxfp4_gluon_cpp_n4096_k2048_ext_br"
    print(f"R35 wrong-cell locator: M={M} N={N} K={K} module={mod_name}", flush=True)

    mod = load_module(mod_name)

    k_blocks = K // 32
    torch.manual_seed(42)
    A = gen_fp4(M, K); B = gen_fp4(N, K)
    sa = torch.full((M, k_blocks), -4, dtype=torch.int8, device='cuda')
    sb = torch.full((N, k_blocks), -4, dtype=torch.int8, device='cuda')
    A_sc = preshuffle_mfma16_merged(sa)
    B_sc = preshuffle_mfma16_merged(sb)

    C_ref = torch_ref(A, B, sa, sb, K)
    print(f"  ref computed: shape={tuple(C_ref.shape)}, ref_finite={torch.isfinite(C_ref.float()).float().mean().item()*100:.1f}%", flush=True)

    runs_i16 = []
    for _ in range(n_runs):
        C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
        mod.gemm_rcr(A, B, A_sc, B_sc, C)
        torch.cuda.synchronize()
        runs_i16.append(C.view(torch.int16).clone())
    stack = torch.stack(runs_i16, 0)
    agree = (stack == stack[0:1]).all(dim=0)  # bool [M,N]
    det_frac = agree.float().mean().item()

    C0 = runs_i16[0].view(torch.bfloat16)
    finite_mask = torch.isfinite(C0.float())
    finite_frac = finite_mask.float().mean().item()
    print(f"  det_frac={det_frac*100:.2f}%  kernel_finite={finite_frac*100:.2f}%", flush=True)

    diff = (C0.float() - C_ref.float()).abs()
    det_wrong = agree & (diff > 1e10)  # huge errors (overflow magnitudes)
    det_correct_mask = agree & (diff < 100)
    print(f"  det_wrong (|diff|>1e10) cells: {det_wrong.sum().item()} ({det_wrong.float().mean().item()*100:.2f}%)", flush=True)
    print(f"  det_correct (|diff|<100) cells: {det_correct_mask.sum().item()} ({det_correct_mask.float().mean().item()*100:.2f}%)", flush=True)
    nondet = ~agree
    print(f"  non-det cells: {nondet.sum().item()} ({nondet.float().mean().item()*100:.2f}%)", flush=True)

    # Are det_wrong identical to non-finite?
    nonfinite = ~finite_mask
    overlap_nf_dw = (nonfinite & det_wrong).sum().item()
    print(f"  overlap(non-finite, det-wrong) = {overlap_nf_dw} / non-finite={nonfinite.sum().item()} / det-wrong={det_wrong.sum().item()}", flush=True)

    # Position pattern analysis
    rows, cols = torch.where(det_wrong)
    print(f"\n  Sample wrong-cell coordinates (first 30):")
    for i in range(min(30, rows.numel())):
        r = rows[i].item(); c = cols[i].item()
        print(f"    ({r:4d}, {c:4d})  ref={C_ref[r,c].item():.6g}  ker={C0[r,c].item():.6g}")

    print("\n  Histogram by row mod 16:")
    rmod16 = (rows % 16).cpu()
    for v in range(16):
        cnt = (rmod16 == v).sum().item()
        bar = '#' * int(cnt * 50 / max(1, rmod16.numel() // 16))
        print(f"    mod16={v:2d}: {cnt:8d} {bar}")

    print("\n  Histogram by col mod 16:")
    cmod16 = (cols % 16).cpu()
    for v in range(16):
        cnt = (cmod16 == v).sum().item()
        bar = '#' * int(cnt * 50 / max(1, cmod16.numel() // 16))
        print(f"    mod16={v:2d}: {cnt:8d} {bar}")

    print("\n  Histogram by row mod 32 (mfma16 boundary):")
    rmod32 = (rows % 32).cpu()
    for v in range(32):
        cnt = (rmod32 == v).sum().item()
        print(f"    mod32={v:2d}: {cnt:8d}")

    print("\n  Histogram by col mod 32:")
    cmod32 = (cols % 32).cpu()
    for v in range(32):
        cnt = (cmod32 == v).sum().item()
        print(f"    mod32={v:2d}: {cnt:8d}")

    print("\n  Histogram by row mod 64 (wave boundary):")
    rmod64 = (rows % 64).cpu()
    for v in range(64):
        cnt = (rmod64 == v).sum().item()
        if cnt > 0 or v < 4 or (v % 16 == 0):
            print(f"    mod64={v:2d}: {cnt:8d}")

    print("\n  Histogram by row mod 256 (BLOCK_M tile boundary):")
    rmod256 = (rows % 256).cpu()
    for v in range(0, 256, 16):
        cnt = ((rmod256 >= v) & (rmod256 < v+16)).sum().item()
        print(f"    mod256[{v:3d}..{v+16:3d}]: {cnt:8d}")

    print("\n  Histogram by col mod 256 (BLOCK_N tile boundary):")
    cmod256 = (cols % 256).cpu()
    for v in range(0, 256, 16):
        cnt = ((cmod256 >= v) & (cmod256 < v+16)).sum().item()
        print(f"    mod256[{v:3d}..{v+16:3d}]: {cnt:8d}")

    # 256x256 tile heatmap (16x16 tiles -> 16x16 grid)
    print("\n  256x256 tile heatmap (rows = M_tile_idx, cols = N_tile_idx, value = wrong cells in tile):")
    M_tiles = M // 256
    N_tiles = N // 256
    tile_r = (rows // 256).cpu()
    tile_c = (cols // 256).cpu()
    heat = torch.zeros(M_tiles, N_tiles, dtype=torch.int64)
    for r, c in zip(tile_r.tolist(), tile_c.tolist()):
        heat[r, c] += 1
    for r in range(M_tiles):
        row_str = ' '.join(f"{heat[r,c].item():>5d}" for c in range(N_tiles))
        print(f"    M{r:2d}: {row_str}")

    # Per-row counts: are entire rows wrong?
    rows_with_wrong = torch.zeros(M, dtype=torch.int64)
    for r in rows.cpu().tolist():
        rows_with_wrong[r] += 1
    print(f"\n  Rows with any wrong cells: {(rows_with_wrong > 0).sum().item()} / {M}")
    print(f"  Rows with >100 wrong cells: {(rows_with_wrong > 100).sum().item()}")
    print(f"  Rows with >1000 wrong cells: {(rows_with_wrong > 1000).sum().item()}")
    print(f"  Max wrong cells in any row: {rows_with_wrong.max().item()}")

    cols_with_wrong = torch.zeros(N, dtype=torch.int64)
    for c in cols.cpu().tolist():
        cols_with_wrong[c] += 1
    print(f"  Cols with any wrong cells: {(cols_with_wrong > 0).sum().item()} / {N}")
    print(f"  Cols with >100 wrong cells: {(cols_with_wrong > 100).sum().item()}")
    print(f"  Cols with >1000 wrong cells: {(cols_with_wrong > 1000).sum().item()}")
    print(f"  Max wrong cells in any col: {cols_with_wrong.max().item()}")

    # First 30 rows that have ANY wrong cells, and how many
    nonzero_rows = torch.where(rows_with_wrong > 0)[0]
    print(f"\n  First 30 rows with wrong cells (row, count):")
    for i in range(min(30, nonzero_rows.numel())):
        r = nonzero_rows[i].item()
        print(f"    row {r:4d}: {rows_with_wrong[r].item()} wrong")

    nonzero_cols = torch.where(cols_with_wrong > 0)[0]
    print(f"\n  First 30 cols with wrong cells (col, count):")
    for i in range(min(30, nonzero_cols.numel())):
        c = nonzero_cols[i].item()
        print(f"    col {c:4d}: {cols_with_wrong[c].item()} wrong")

    # Additional: what about non-finite cells per row/col
    nf_rows, nf_cols = torch.where(nonfinite)
    print(f"\n  Non-finite cells: {nonfinite.sum().item()}")
    print(f"  Non-finite rows with any: {(nonfinite.any(dim=1)).sum().item()}")
    print(f"  Non-finite cols with any: {(nonfinite.any(dim=0)).sum().item()}")

    # Mod 16 distribution in NON-finite specifically
    nf_rmod16 = (nf_rows % 16).cpu()
    nf_cmod16 = (nf_cols % 16).cpu()
    print(f"\n  Non-finite row mod 16:")
    for v in range(16):
        print(f"    mod16={v:2d}: {(nf_rmod16 == v).sum().item():8d}")
    print(f"  Non-finite col mod 16:")
    for v in range(16):
        print(f"    mod16={v:2d}: {(nf_cmod16 == v).sum().item():8d}")

    # Save coords to JSON for later analysis
    coords = list(zip(rows[:1000].cpu().tolist(), cols[:1000].cpu().tolist()))
    with open(os.path.join(SCRIPT_DIR, "R35_WRONG_CELL_COORDS.json"), 'w') as f:
        json.dump({
            "M": M, "N": N, "K": K, "module": mod_name,
            "n_det_wrong": int(det_wrong.sum().item()),
            "n_nondet": int(nondet.sum().item()),
            "n_nonfinite": int(nonfinite.sum().item()),
            "first_1000_coords": coords,
        }, f, indent=2)
    print(f"\nWrote coords -> R35_WRONG_CELL_COORDS.json")

if __name__ == '__main__':
    main()
