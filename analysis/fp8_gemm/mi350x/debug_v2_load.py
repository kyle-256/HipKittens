import torch
import sys
torch.manual_seed(42)

import tk_fp8_layouts

N = 8192

def gen_fp8(rows, cols):
    return (torch.randn(rows, cols, dtype=torch.float32, device="cuda") * 0.1).to(torch.float8_e4m3fn)

At = gen_fp8(N, N)
B  = gen_fp8(N, N)
C  = torch.zeros(N, N, dtype=torch.bfloat16, device="cuda")

tk_fp8_layouts.gemm_crr(At, B, C)
torch.cuda.synchronize()

C_ref = (At.float().T @ B.float()).bfloat16()

diff = (C.float() - C_ref.float()).abs()
max_err = diff.max().item()
mean_err = diff.mean().item()
print(f"Max error: {max_err:.4f}, Mean error: {mean_err:.4f}")

if max_err < 0.5:
    print("PASS - v2 A-tile load is correct!")
    sys.exit(0)

print(f"\nAnalyzing error pattern...")

BLK = 256
n_blocks = N // BLK

bad_threshold = 1.0

for br in range(min(4, n_blocks)):
    for bc in range(min(4, n_blocks)):
        block = diff[br*BLK:(br+1)*BLK, bc*BLK:(bc+1)*BLK]
        block_max = block.max().item()
        if block_max > bad_threshold:
            print(f"\nBlock ({br},{bc}): max_err={block_max:.4f}")
            for sub_r in range(2):
                for sub_c in range(2):
                    sub = block[sub_r*128:(sub_r+1)*128, sub_c*128:(sub_c+1)*128]
                    sub_max = sub.max().item()
                    if sub_max > bad_threshold:
                        top_idx = torch.nonzero(sub > bad_threshold)
                        print(f"  Sub({sub_r},{sub_c}): max={sub_max:.4f}, n_bad={len(top_idx)}")
                        if len(top_idx) > 0:
                            for i in range(min(5, len(top_idx))):
                                r, c = top_idx[i]
                                gr, gc = br*BLK + sub_r*128 + r.item(), bc*BLK + sub_c*128 + c.item()
                                print(f"    [{gr},{gc}] kernel={C[gr,gc].float().item():.4f} ref={C_ref[gr,gc].float().item():.4f} diff={diff[gr,gc].item():.4f}")

row_max = diff.max(dim=1).values
col_max = diff.max(dim=0).values

bad_rows = torch.nonzero(row_max > bad_threshold).squeeze()
bad_cols = torch.nonzero(col_max > bad_threshold).squeeze()

print(f"\nRows with error > {bad_threshold}: {bad_rows.numel()} out of {N}")
if bad_rows.numel() > 0 and bad_rows.numel() <= 20:
    print(f"  Rows: {bad_rows.tolist()}")
elif bad_rows.numel() > 20:
    print(f"  First 20 rows: {bad_rows[:20].tolist()}")
    row_mod_128 = bad_rows % 128
    row_mod_16 = bad_rows % 16
    print(f"  Row % 128 values: {torch.unique(row_mod_128).tolist()}")
    print(f"  Row % 16 values: {torch.unique(row_mod_16).tolist()}")

print(f"\nCols with error > {bad_threshold}: {bad_cols.numel()} out of {N}")
if bad_cols.numel() > 0 and bad_cols.numel() <= 20:
    print(f"  Cols: {bad_cols.tolist()}")
elif bad_cols.numel() > 20:
    print(f"  First 20 cols: {bad_cols[:20].tolist()}")
    col_mod_128 = bad_cols % 128
    col_mod_16 = bad_cols % 16
    print(f"  Col % 128 values: {torch.unique(col_mod_128).tolist()}")
    print(f"  Col % 16 values: {torch.unique(col_mod_16).tolist()}")
