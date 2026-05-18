"""HK grouped FP8 blockwise wgrad (CRR/TN, K-contig col-T inputs) — persistent.

Per group g ∈ [0, G):
  dW[g, :, :] = sum_{m in [m_start_g, m_end_g)} dY[m, n] * X[m, k]
              = g_T[:, m_start_g:m_end_g] @ a_T[:, m_start_g:m_end_g]^T
where g_T = dY^T [N, M_total] and a_T = X^T [K, M_total] (caller pre-transposes).

Per-tile k_iters varies (= M_g / BLOCK_K). Persistent kernel maps tile_id
to (group_idx, n_block, k_block) with fixed N×K tiles per group.

Layout (for HK kernel):
  A         [N, M_total]    fp8 e4m3fnuz   (g_T)
  B         [K, M_total]    fp8 e4m3fnuz   (a_T)
  C         [G, N, K]       bf16           (dW)
  A_scale   [Mb, N]         fp32           (per-N-element, g_scale_kc)
  B_scale   [Mb, K]         fp32           (per-K-element, a_scale_kc)
  group_offs[G+1]           int32 prefix sum of M lengths
"""
import math
import os
import sys

import torch
import importlib.util
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, ext_so):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, ext_so))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# Per-shape wgrad registry: (G, MPG, N, K) → (BM, BN, BK, NW).
# Small MPG benefits from BM=128/W=4 (more N-tiles, better CU coverage on
# short K-loop). Big MPG prefers default BM=256/W=8 (heavier per-tile work
# amortizes pipeline overhead).
WGRAD_REGISTRY = {
    # (G, MPG, N, K) : (BM, BN, BK, NW)
    (8,  1024, 4096, 4096): (128, 128, 128, 4),
    (16, 512,  4096, 4096): (128, 128, 128, 4),
}


def _load_wgrad_so(bm, bn, bk, nw):
    if (bm, bn, bk, nw) == (256, 128, 128, 8):
        return _load("tk_wgrad", "tk_wgrad.cpython-312-x86_64-linux-gnu.so")
    name = f"tk_wgrad_BM{bm}_BN{bn}_BK{bk}_W{nw}"
    so = name + ".cpython-312-x86_64-linux-gnu.so"
    if not os.path.exists(os.path.join(HERE, so)):
        subprocess.run(["make", "wgrad-tuned",
                        f"BLOCK_M={bm}", f"BLOCK_N={bn}", f"BLOCK_K={bk}", f"NUM_WARPS={nw}"],
                       cwd=HERE, check=True)
    return _load(name, so)

torch.manual_seed(0)

BLOCK_M, BLOCK_N = 256, 128       # fwd's BM/BN — wgrad uses these for output (N, K)
BK = 128                            # BLOCK_K

G_groups   = int(os.environ.get("BW_G", "4"))
M_per_g    = int(os.environ.get("BW_MPG", "2048"))   # reduction (varies per group); must be %128 (BLOCK_K)
N          = int(os.environ.get("BW_N", "4096"))     # output rows; must be %256 (fwd BM)
K          = int(os.environ.get("BW_K", "4096"))     # output cols; must be %128 (fwd BN)
unaligned  = os.environ.get("BW_UNALIGNED", "0") != "0"
num_warmup = int(os.environ.get("BW_WARMUP", "20"))
num_iters  = int(os.environ.get("BW_ITERS", "50"))
check      = os.environ.get("BW_CHECK", "1") != "0"

# Per-shape config from registry (else default BM=256/W=8).
_cfg = WGRAD_REGISTRY.get((G_groups, M_per_g, N, K), (256, 128, 128, 8))
BLOCK_M, BLOCK_N, _BLOCK_K, _NW = _cfg
tk_wgrad = _load_wgrad_so(*_cfg)

assert N % BLOCK_M == 0, f"N={N} must be %{BLOCK_M} (fwd BM)"
assert K % BLOCK_N == 0, f"K={K} must be %{BLOCK_N} (fwd BN)"

# Per-group M must be %BK so K-loop iterates an integer count.
def round_up(x, m): return ((x + m - 1) // m) * m

if unaligned:
    torch.manual_seed(123)
    pert = torch.randint(-M_per_g // 4, M_per_g // 4 + 1, (G_groups,))
    pert = (pert // BK) * BK    # round perturbation to BK-multiple (no padding needed)
    group_lens = (M_per_g + pert).clamp(min=BK).tolist()
else:
    assert M_per_g % BK == 0
    group_lens = [M_per_g] * G_groups

# Wgrad: M is reduction axis. m_start_g must also be BK-multiple (so each
# group's K-loop starts at a clean boundary). For balanced/random with
# BK-multiple lens, m_start_g = sum of prior BK-multiples → BK-multiple ✓.
M_total = sum(group_lens)
group_offs = [0]
for L in group_lens:
    group_offs.append(group_offs[-1] + L)

Mb_total = M_total // BK
Mb_per_g = [L // BK for L in group_lens]
Nb_out = N // BK
Kb_out = K // BK


def gen_fp8(*shape):
    return (torch.randn(*shape, device="cuda") * 0.1).to(torch.float8_e4m3fnuz)


def gen_scales(*shape):
    return torch.rand(*shape, device="cuda", dtype=torch.float32) * 0.5 + 0.5


# ─── User-facing tensors ───────────────────────────────────────────────
# Natural orientation: dY [M_total, N], X [M_total, K]. Caller pre-transposes
# to col-T so M is the contiguous (cols) axis for the kernel.
dY = gen_fp8(M_total, N)
X  = gen_fp8(M_total, K)
g_T = dY.T.contiguous()   # [N, M_total]
a_T = X.T.contiguous()    # [K, M_total]
dW  = torch.zeros(G_groups, N, K, dtype=torch.bfloat16, device="cuda")

# Per-N-element + per-K-element scales (Triton's K-contig wgrad layout).
g_scale = gen_scales(Mb_total, N)   # [Mb, N]
a_scale = gen_scales(Mb_total, K)   # [Mb, K]

group_offs_t = torch.tensor(group_offs, device="cuda", dtype=torch.int32)

# Persistent tile mapping: each group has (N/BM_fwd) × (K/BN_fwd) tiles, FIXED.
n_tiles_per_g = N // BLOCK_M
k_tiles_per_g = K // BLOCK_N
tiles_per_group = n_tiles_per_g * k_tiles_per_g
total_tiles = G_groups * tiles_per_group


def ref_wgrad():
    """Per-group wgrad reference in fp32."""
    out = torch.zeros(G_groups, N, K, dtype=torch.float32, device="cuda")
    for g in range(G_groups):
        m0, m1 = group_offs[g], group_offs[g+1]
        g_T_g = g_T[:, m0:m1].float()
        a_T_g = a_T[:, m0:m1].float()
        out_g = torch.zeros(N, K, dtype=torch.float32, device="cuda")
        for mb in range(m0 // BK, m1 // BK):
            mb_lo, mb_hi = (mb - m0 // BK) * BK, ((mb - m0 // BK) + 1) * BK
            partial = g_T_g[:, mb_lo:mb_hi] @ a_T_g[:, mb_lo:mb_hi].T   # [N, K]
            g_s = g_scale[mb, :].unsqueeze(1)   # [N, 1] per-N
            a_s = a_scale[mb, :].unsqueeze(0)   # [1, K] per-K
            out_g += partial * g_s * a_s
        out[g] = out_g
    return out


def snr_db(test, ref_):
    t, r = test.float(), ref_.float()
    sig = (r * r).sum().item()
    noise = ((t - r) ** 2).sum().item()
    if noise == 0:
        return float("inf")
    return 10.0 * math.log10(sig / max(noise, 1e-45))


def dispatch():
    tk_wgrad.dispatch_grouped_wgrad(g_T, a_T, dW, g_scale, a_scale,
                                     group_offs_t.view(1, 1, 1, -1),
                                     G_groups, total_tiles)


print(f"=== HK Grouped FP8 Blockwise WGRAD (persistent) ===")
print(f"G={G_groups}  unaligned={unaligned}  N={N}  K={K}")
print(f"group_lens = {group_lens}")
print(f"M_total={M_total}  total_tiles={total_tiles}\n")

dispatch()
torch.cuda.synchronize()

if check:
    dW_ref = ref_wgrad()
    snr = snr_db(dW, dW_ref)
    diff = (dW.float() - dW_ref).abs()
    print(f"  SNR:  {snr:.2f} dB")
    print(f"  Max err: {diff.max().item():.4f}   Mean err: {diff.mean().item():.6f}")
    print(f"  Large errs (>0.5): {(diff > 0.5).sum().item()}")
    ok = snr > 48.0
    print(f"  Correctness: {'PASS' if ok else 'FAIL'}\n")
    if not ok:
        sys.exit(1)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for _ in range(num_warmup):
    dispatch()
times = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start.record()
    dispatch()
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

avg = sum(times) / len(times)
flops = 2 * M_total * N * K
tflops = flops / (avg * 1e9)
print(f"Avg time: {avg:.4f} ms")
print(f"Performance: {tflops:.2f} TFLOPS")
