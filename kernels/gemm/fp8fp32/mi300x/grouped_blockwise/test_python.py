"""HK grouped FP8 blockwise GEMM (unaligned-friendly fwd) — persistent kernel.

Supports arbitrary per-group M lengths (no requirement that M_g % BLOCK_M == 0).
The kernel itself is aligned-only; this harness pads each group's A/A_scale
rows up to the next BM=256 multiple before launch, then extracts the valid
rows from the padded output. Pad rows are zero (so accumulator contributes 0).

API:
  group_lens [G] int   per-group M values (any positive ints)
  A          [M_total, K]    fp8 row-major (M_total = sum(group_lens))
  B          [G, N, K]       fp8 row-major
  A_scale    [M_total, Kb]   fp32
  B_scale    [G, Nb, Kb]     fp32

Internally:
  M_padded[g] = ceil(M_g / BM) * BM
  M_total_padded = sum(M_padded)
  Pad A, A_scale per-group to M_padded[g]; pad C accordingly.
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


# Per-shape fwd registry: (G, MPG, N, K) → chunk override (BM/W kept default).
# Per-shape chunk sweep showed different configs prefer different chunks.
FWD_CHUNK_REGISTRY = {
    (2,  4096, 4096, 4096): 8,
    (4,  4096, 4096, 4096): 8,
    (8,  1024, 4096, 4096): 2,
    (16, 512,  4096, 4096): 16,
}


def _load_fwd_so(chunk):
    if chunk == 4:
        return _load("tk_kernel", "tk_kernel.cpython-312-x86_64-linux-gnu.so")
    name = f"tk_kernel_BM256_BN128_BK128_W8_CHK{chunk}"
    so = name + ".cpython-312-x86_64-linux-gnu.so"
    if not os.path.exists(os.path.join(HERE, so)):
        subprocess.run(["make", "tuned", f"CHUNK={chunk}"],
                       cwd=HERE, check=True)
    return _load(name, so)

torch.manual_seed(0)

BLOCK_M, BLOCK_N = 256, 128
BK = 128

G_groups   = int(os.environ.get("BW_G", "4"))
M_per_g    = int(os.environ.get("BW_MPG", "2048"))
N          = int(os.environ.get("BW_N", "4096"))     # must be %128
K          = int(os.environ.get("BW_K", "4096"))     # must be %128
unaligned  = os.environ.get("BW_UNALIGNED", "0") != "0"
num_warmup = int(os.environ.get("BW_WARMUP", "20"))
num_iters  = int(os.environ.get("BW_ITERS", "50"))
check      = os.environ.get("BW_CHECK", "1") != "0"

assert N % 128 == 0
assert K % 128 == 0

# Per-shape fwd config (chunk).
_chunk = FWD_CHUNK_REGISTRY.get((G_groups, M_per_g, N, K), 4)
tk_kernel = _load_fwd_so(_chunk)


# ─── Generate group_lens ───────────────────────────────────────────────
if unaligned:
    # Imbalanced: M_per_g is the *mean*, lengths vary ±25%, all multiples of 16.
    torch.manual_seed(123)
    pert = torch.randint(-M_per_g // 4, M_per_g // 4 + 1, (G_groups,))
    pert = (pert // 16) * 16
    group_lens = (M_per_g + pert).clamp(min=16).tolist()
    # Ensure at least one valid m-tile per group: round each up to >= 16.
else:
    assert M_per_g % BLOCK_M == 0, f"M_per_g={M_per_g} must be %{BLOCK_M} (aligned mode)"
    group_lens = [M_per_g] * G_groups

M_total_real = sum(group_lens)
group_offs_real = [0]
for L in group_lens:
    group_offs_real.append(group_offs_real[-1] + L)


# ─── Padded layout ─────────────────────────────────────────────────────
def round_up(x, m): return ((x + m - 1) // m) * m

group_lens_padded = [round_up(L, BLOCK_M) for L in group_lens]
group_offs_padded = [0]
for L in group_lens_padded:
    group_offs_padded.append(group_offs_padded[-1] + L)
M_total_padded = group_offs_padded[-1]

Kb = K // BK
Nb = N // BK


def gen_fp8(*shape):
    return (torch.randn(*shape, device="cuda") * 0.1).to(torch.float8_e4m3fnuz)


def gen_scales(*shape):
    return torch.rand(*shape, device="cuda", dtype=torch.float32) * 0.5 + 0.5


# ─── User-facing tensors (real, unpadded) ──────────────────────────────
A_real        = gen_fp8(M_total_real, K)                 # [M_total_real, K]
B             = gen_fp8(G_groups, N, K)                  # [G, N, K]
A_scale_real  = gen_scales(M_total_real, Kb)             # [M_total_real, Kb]
B_scale_nat   = gen_scales(G_groups, Nb, Kb)             # [G, Nb, Kb]


# ─── Build padded views (zero pad rows between groups) ─────────────────
A_padded       = torch.zeros(M_total_padded, K, dtype=A_real.dtype, device="cuda")
A_scale_padded = torch.zeros(M_total_padded, Kb, dtype=torch.float32, device="cuda")
C_padded       = torch.zeros(M_total_padded, N, dtype=torch.bfloat16, device="cuda")
for g in range(G_groups):
    src_lo, src_hi = group_offs_real[g],   group_offs_real[g+1]
    dst_lo         = group_offs_padded[g]
    dst_hi         = dst_lo + (src_hi - src_lo)
    A_padded[dst_lo:dst_hi]       = A_real[src_lo:src_hi]
    A_scale_padded[dst_lo:dst_hi] = A_scale_real[src_lo:src_hi]
    # Padding rows [dst_hi : dst_lo + group_lens_padded[g]] stay at 0.

A_scale_T = A_scale_padded.T.contiguous()                  # [Kb, M_total_padded]
B_scale_T = B_scale_nat.transpose(1, 2).contiguous()       # [G, Kb, Nb]

group_offs_padded_t = torch.tensor(group_offs_padded, device="cuda", dtype=torch.int32)

num_pid_n = N // BLOCK_N
tiles_per_g = [(L // BLOCK_M) * num_pid_n for L in group_lens_padded]
cum_tiles_t = torch.tensor([0] + list(torch.cumsum(torch.tensor(tiles_per_g), 0).tolist()),
                            device="cuda", dtype=torch.int32)
total_tiles = int(cum_tiles_t[-1].item())


def ref_unpadded():
    """Reference using user-facing UNPADDED A/A_scale per group."""
    out = torch.zeros(M_total_real, N, dtype=torch.float32, device="cuda")
    for g in range(G_groups):
        m0, m1 = group_offs_real[g], group_offs_real[g+1]
        A_g = A_real[m0:m1].float()
        B_g = B[g].float()
        out_g = torch.zeros(m1 - m0, N, dtype=torch.float32, device="cuda")
        for ki in range(Kb):
            k0, k1 = ki * BK, (ki + 1) * BK
            partial = A_g[:, k0:k1] @ B_g[:, k0:k1].T
            a_s = A_scale_real[m0:m1, ki:ki+1]
            b_s = B_scale_nat[g, :, ki].repeat_interleave(BK)[:N].unsqueeze(0)
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
    tk_kernel.dispatch_grouped(A_padded, B, C_padded, A_scale_T, B_scale_T,
                                group_offs_padded_t.view(1, 1, 1, -1),
                                cum_tiles_t.view(1, 1, 1, -1),
                                G_groups, total_tiles)


def extract_unpadded():
    """Extract user-facing rows from C_padded."""
    out = torch.empty(M_total_real, N, dtype=torch.bfloat16, device="cuda")
    for g in range(G_groups):
        src_lo  = group_offs_padded[g]
        L_real  = group_offs_real[g+1] - group_offs_real[g]
        dst_lo  = group_offs_real[g]
        out[dst_lo:dst_lo+L_real] = C_padded[src_lo:src_lo+L_real]
    return out


print(f"=== HK Grouped FP8 Blockwise (persistent, unaligned-friendly) ===")
print(f"G={G_groups}  unaligned={unaligned}  N={N}  K={K}")
print(f"group_lens = {group_lens}")
print(f"group_lens_padded = {group_lens_padded}")
print(f"M_total real={M_total_real}  padded={M_total_padded}  total_tiles={total_tiles}")
print(f"warmup={num_warmup} iters={num_iters} check={check}\n")

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
# FLOPs counted on REAL (unpadded) shape: 2 * sum(M_g * N * K)
flops = 2 * M_total_real * N * K
tflops = flops / (avg * 1e9)
overhead_pct = (M_total_padded / max(M_total_real, 1) - 1) * 100
print(f"Avg time: {avg:.4f} ms")
print(f"Performance: {tflops:.2f} TFLOPS  (real FLOPs; pad overhead {overhead_pct:.1f}%)")
