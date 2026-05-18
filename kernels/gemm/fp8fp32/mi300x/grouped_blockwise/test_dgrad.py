"""HK grouped FP8 blockwise dgrad (NN/RRR) test — fwd-routing.

Per group g ∈ [0, G):
  dX[offs[g]:offs[g+1], :] = dY[offs[g]:offs[g+1], :] @ W[g, :, :]

Routes through the fwd kernel: pre-transpose W per group so the kernel's
NT contract (C = A @ B^T) computes A=dY @ B^T=W ⇒ dY @ W.

Layout:
  dY        [M_total, N]    fp8 e4m3fnuz
  W         [G, N, K]       fp8 e4m3fnuz (caller pre-transposes per group)
  dX        [M_total, K]    bf16
  dY_scale  [M_total, Nb]   fp32 (per-N-block)
  W_scale   [G, Nb, Kb]     fp32

Reduction axis = N (not K). Kernel's A_scale_T is [Nb, M_total] here.
"""
import math
import os
import sys

import torch
import tk_kernel

torch.manual_seed(0)

BLOCK_M, BLOCK_N = 256, 128
BK = 128

G_groups   = int(os.environ.get("BW_G", "4"))
M_per_g    = int(os.environ.get("BW_MPG", "2048"))
N          = int(os.environ.get("BW_N", "4096"))     # reduction axis, must be %128
K          = int(os.environ.get("BW_K", "4096"))     # output cols, must be %128
unaligned  = os.environ.get("BW_UNALIGNED", "0") != "0"
num_warmup = int(os.environ.get("BW_WARMUP", "20"))
num_iters  = int(os.environ.get("BW_ITERS", "50"))
check      = os.environ.get("BW_CHECK", "1") != "0"

# Kernel constraints (with fwd-routing, reduction=N → fwd's K=N must %128):
assert N % 128 == 0, f"N={N} must be %128"
assert K % 128 == 0, f"K={K} must be %128 (output cols → fwd's BLOCK_N)"


def round_up(x, m): return ((x + m - 1) // m) * m


# ─── Group lengths ─────────────────────────────────────────────────────
if unaligned:
    torch.manual_seed(123)
    pert = torch.randint(-M_per_g // 4, M_per_g // 4 + 1, (G_groups,))
    pert = (pert // 16) * 16
    group_lens = (M_per_g + pert).clamp(min=16).tolist()
else:
    assert M_per_g % BLOCK_M == 0
    group_lens = [M_per_g] * G_groups

M_total_real = sum(group_lens)
group_offs_real = [0]
for L in group_lens:
    group_offs_real.append(group_offs_real[-1] + L)

group_lens_padded = [round_up(L, BLOCK_M) for L in group_lens]
group_offs_padded = [0]
for L in group_lens_padded:
    group_offs_padded.append(group_offs_padded[-1] + L)
M_total_padded = group_offs_padded[-1]

Nb = N // BK
Kb = K // BK


def gen_fp8(*shape):
    return (torch.randn(*shape, device="cuda") * 0.1).to(torch.float8_e4m3fnuz)


def gen_scales(*shape):
    return torch.rand(*shape, device="cuda", dtype=torch.float32) * 0.5 + 0.5


# ─── User-facing tensors ───────────────────────────────────────────────
dY_real        = gen_fp8(M_total_real, N)               # [M_total_real, N]
W              = gen_fp8(G_groups, N, K)                # [G, N, K] (original)
dY_scale_real  = gen_scales(M_total_real, Nb)
W_scale        = gen_scales(G_groups, Nb, Kb)

# ─── Padded views ──────────────────────────────────────────────────────
dY_padded       = torch.zeros(M_total_padded, N, dtype=dY_real.dtype, device="cuda")
dY_scale_padded = torch.zeros(M_total_padded, Nb, dtype=torch.float32, device="cuda")
dX_padded       = torch.zeros(M_total_padded, K, dtype=torch.bfloat16, device="cuda")
for g in range(G_groups):
    sL, sH = group_offs_real[g], group_offs_real[g+1]
    dL     = group_offs_padded[g]
    dY_padded[dL:dL + sH - sL]       = dY_real[sL:sH]
    dY_scale_padded[dL:dL + sH - sL] = dY_scale_real[sL:sH]

# ─── Pre-transpose W per group → Wp [G, K, N] ──────────────────────────
# Then fwd kernel sees A=dY_padded (plays M, K=N), B=Wp (plays N=K, K=N).
Wp = W.transpose(1, 2).contiguous()   # [G, K, N]

# ─── Scale layouts for fwd kernel ──────────────────────────────────────
# Fwd expects A_scale_T [Kb_red, M], B_scale_T [G, Kb_red, Nb_out].
# For dgrad: Kb_red = Nb (reduction = N), Nb_out = Kb.
A_scale_T = dY_scale_padded.T.contiguous()   # [Nb, M_total_padded]
# W_scale [G, Nb, Kb] is already in [G, Kb_red=Nb, Nb_out=Kb] form.
B_scale_T = W_scale.contiguous()              # [G, Nb, Kb]

group_offs_t = torch.tensor(group_offs_padded, device="cuda", dtype=torch.int32)
num_pid_n_out = K // BLOCK_N    # K plays N_out → BLOCK_N tiles
tiles_per_g = [(L // BLOCK_M) * num_pid_n_out for L in group_lens_padded]
cum_tiles_t = torch.tensor(
    [0] + list(torch.cumsum(torch.tensor(tiles_per_g), 0).tolist()),
    device="cuda", dtype=torch.int32,
)
total_tiles = int(cum_tiles_t[-1].item())


def ref_unpadded():
    """Per-group dgrad reference in fp32."""
    out = torch.zeros(M_total_real, K, dtype=torch.float32, device="cuda")
    for g in range(G_groups):
        m0, m1 = group_offs_real[g], group_offs_real[g+1]
        dY_g = dY_real[m0:m1].float()
        W_g = W[g].float()
        out_g = torch.zeros(m1 - m0, K, dtype=torch.float32, device="cuda")
        for ni in range(Nb):
            n0, n1 = ni * BK, (ni + 1) * BK
            partial = dY_g[:, n0:n1] @ W_g[n0:n1, :]
            a_s = dY_scale_real[m0:m1, ni:ni+1]
            b_s = W_scale[g, ni, :].repeat_interleave(BK)[:K].unsqueeze(0)
            out_g += partial * a_s * b_s
        out[m0:m1] = out_g
    return out


def snr_db(test, ref_):
    t, r = test.float(), ref_.float()
    sig = (r * r).sum().item()
    noise = ((t - r) ** 2).sum().item()
    if noise == 0:
        return float("inf")
    return 10.0 * math.log10(sig / max(noise, 1e-45))


def dispatch():
    tk_kernel.dispatch_grouped(dY_padded, Wp, dX_padded, A_scale_T, B_scale_T,
                                group_offs_t.view(1, 1, 1, -1),
                                cum_tiles_t.view(1, 1, 1, -1),
                                G_groups, total_tiles)


def extract_unpadded():
    out = torch.empty(M_total_real, K, dtype=torch.bfloat16, device="cuda")
    for g in range(G_groups):
        sL = group_offs_padded[g]
        L_real = group_offs_real[g+1] - group_offs_real[g]
        dL = group_offs_real[g]
        out[dL:dL+L_real] = dX_padded[sL:sL+L_real]
    return out


print(f"=== HK Grouped FP8 Blockwise DGRAD (fwd-routing) ===")
print(f"G={G_groups}  unaligned={unaligned}  N={N}  K={K}")
print(f"group_lens = {group_lens}")
print(f"M_total real={M_total_real}  padded={M_total_padded}  total_tiles={total_tiles}\n")

dispatch()
torch.cuda.synchronize()

if check:
    C_ref = ref_unpadded()
    C_hk  = extract_unpadded()
    snr = snr_db(C_hk, C_ref)
    diff = (C_hk.float() - C_ref).abs()
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
flops = 2 * M_total_real * N * K
tflops = flops / (avg * 1e9)
overhead_pct = (M_total_padded / max(M_total_real, 1) - 1) * 100
print(f"Avg time: {avg:.4f} ms")
print(f"Performance: {tflops:.2f} TFLOPS  (real FLOPs; pad overhead {overhead_pct:.1f}%)")
