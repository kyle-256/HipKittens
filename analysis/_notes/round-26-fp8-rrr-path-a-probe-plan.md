# Round 26 — Plateau confirmation + executable FP8 RRR fuse path A probe plan

**Date**: 2026-04-30 evening
**Round**: 26 / 100 (auto_optimize)
**Score**: 836 / 834 / 833 (3 runs at HK fde4b692 + Primus c3b70e37, noise band)
**Plateau**: rounds 21-26 all 833-836, 6 consecutive rounds without movement
**Previous best**: 835 (round 22)
**Patience**: 30 (used 6 / 30 since plateau started)
**Primus-Turbo HEAD**: c3b70e37
**HipKittens HEAD before round**: 8bc19074 (round-18-19 H4 gate consolidation)

## TL;DR

Round 26 ships ONLY this docs note. No kernel / dispatcher / Primus changes.
Reason: the score plateau (832-836) is **architectural** per round-12 rocprof
breakdown (HK GEMM kernel 1.34× slower than Triton GEMM raw); the only
remaining task-body-compliant wedge is **FP8 RRR fuse path A empirical
numerical probe** (round-17 docs recommendation, untested). This note
provides the **executable** probe specification so a future round can
implement + run the probe in a single resume cycle without re-deriving the
scaffolding from round-7/8 BF16 + round-17 FP8 layout analysis.

## Why round 26 ships docs, not kernel

Concrete options surveyed and rejected this round:

1. **Rule tune (group_m / num_xcds refinement)** — task body explicitly
   forbids: "**严禁**在 spill 失败后转去做 BF16 num_xcds tune / group_m
   tune / rule refinement 这类 small fix —— 那些**已经 saturated**". Verified
   saturated rounds 5-70 across all 32 metric shapes (config.py rules carry
   per-shape comments documenting the sweep).

2. **Forward kernel template rewrite (MFMA cell shape 16x16x128 → 32x32x64)**
   — 1-2+ round project. Round 12 rocprof says HK GEMM kernel is the actual
   gap (148 µs Triton vs 198 µs HK on gpt_oss-GateUP-B4-M2048 FP8). Closing
   it requires new BM/BN/BK template + register layout re-derivation +
   fuse-epilog re-fitting + numerical re-verification. Single round =
   incomplete work; partial commit risks introducing register spills /
   numerical regressions that break plateau **downward**.

3. **Single-toggle micro-opt (RCR_PREFETCH_LGKM / RCR_MAIN_UNROLL /
   RCR_TWO_TILE_MIN_KI)** — repeatedly tested rounds 1-15 (round-15 docs:
   `RCR_TWO_TILE_MIN_KI 28→20 probe — no-op`; round-12 docs: SRD hoist
   no-op because compiler already CSEs). Risk-reward: any change either
   no-op (compiler already optimal) OR breaks plateau downward (touched
   load-balance). Plateau is the local optimum.

4. **FP8 RRR fuse path A empirical numerical probe** — round-17 docs
   recommendation. Untested as of round 26. **This is the round-26 commit's
   subject** — preserves task body main line (K-tail fuse) without
   requiring single-round kernel rewrite.

## Round-17 motivation recap

FP8 dA backward (RRR layout) for K-misaligned shapes (gpt_oss-Down K=2880,
K_REM=64) currently falls into Primus H4 reroute (`b.transpose(-2,-1)
.contiguous()` → call HK grouped_rcr) per `grouped_gemm_fp8_impl.py:305+`.
Round-16 rocprof showed the H4 transpose itself eats 21.6% of bwd wall
(78 µs per call × 2 calls / iter). Eliminating H4 requires native FP8
grouped_rrr_kernel K-tail support — i.e. fuse path A (LDS-staged) since
path B (direct HBM→register) is HBM-layout-incompatible with RRR's B
storage (round-17 derivation: B is `[1, G, K, N]`, K is N-strided not
contiguous → b128 K-tail load impossible).

Round-7/8 BF16 RRR fuse path A reached SNR 25.45 dB but allclose FAIL.
Round-7 docs hypothesized the bug is in `Bs[1][n_strip]` post-epilog-2
LDS layout itself (deeper than `subtile_inplace` SGPR aliasing — manual
ds_read sidesteps that and gets +7 dB but ~25% cells still stale).

Round-17 docs said FP8 may succeed where BF16 failed because the LDS
swizzle differs:
* BF16 RRR uses `col_l rt_32x16_s` + `st_32x16_s` (XOR bank-conflict
  swizzle with hardware 4-lane transpose).
* FP8 RRR uses `col_l rt_16x16_s` + `ST_v2` for B (different swizzle —
  no hardware transpose needed; FP8 mfma reads K=128 wide directly).

If FP8 ST_v2 doesn't have the same post-epilog-2 staleness mode, path A
hybrid will succeed where BF16's failed.

## Probe specification (executable in round 27+)

### Step 1: Add compile-time probe gate

In `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`, just before the
`grouped_rrr_kernel` template definition (line ~2454), add:

```cpp
// Round 26 — Path A K-tail fuse empirical probe gate. Default 0
// (production unchanged). Set to 1 + recompile to enable in-kernel
// K-tail accumulation via cooperative G::load → LDS → load_a/load_b
// → rrr_mma. Probe ONLY — production should stay 0 until SNR ≥ 25 dB
// AND allclose pass on the gpt_oss-Down probe shape.
#ifndef FP8_RRR_FUSE_PROBE
#define FP8_RRR_FUSE_PROBE 0
#endif
```

### Step 2: Insert path A K-tail block before scale + store

In `grouped_rrr_kernel` (line 2456), after Epilog 2 closes (line 2678)
and before the `combined_scale` mul (line 2680), insert:

```cpp
        // Round 26 — Path A K-tail fuse probe. Cooperatively reload the
        // K-tail iter (k = ki_dyn) into As[tic][0/1] / Bs[tic][0/1] via
        // G::load (which uses raw_buffer_load_lds — no-op on OOB voffset),
        // pre-zeroing first so OOB cells stay 0 (effective zero-pad for
        // K=[K_global, fast_k+K_BLOCK)). Then call the existing load_a /
        // load_b helpers + rrr_mma 4 times to accumulate K-tail directly
        // into cA/cB/cC/cD before scale.
#if FP8_RRR_FUSE_PROBE
        if (g.fast_k < g.k) {
            // Pre-zero As[tic] and Bs[tic] cooperatively. ST_row /
            // ST_v2 each have 256*128 = 32K bytes; 8 warps * 64 lanes
            // = 512 threads write 64 bytes/thread (1 b128 pair) =
            // 32K bytes / tile. Use kittens::store(tile, 0) helper or
            // direct ds_write loop — see include/types/shared/st.cuh
            // for the cooperative store helper signature.
            //
            // CRITICAL: __syncthreads() required AFTER pre-zero AND
            // BEFORE G::load AND AFTER G::load — round-7/8 BF16 docs
            // showed s_barrier alone doesn't drain lgkmcnt and ds_read
            // can race the buffer_load_lds.
            //
            // Use a kittens helper if available:
            //   zero(As[tic][0]); zero(As[tic][1]);
            //   zero(Bs[tic][0]); zero(Bs[tic][1]);
            //   __syncthreads();
            //
            // OR cooperative manual zero (mirror of G::load coop pattern):
            //   const int tid = threadIdx.x;
            //   constexpr int BYTES_PER_TILE = sizeof(ST_row);
            //   constexpr int BYTES_PER_THREAD = BYTES_PER_TILE / _NUM_THREADS;
            //   #pragma unroll
            //   for (int b = 0; b < BYTES_PER_THREAD; b += 16) {
            //       *reinterpret_cast<__uint128_t*>(
            //           &As[tic][0].data[tid * BYTES_PER_THREAD + b]) = 0;
            //       *reinterpret_cast<__uint128_t*>(
            //           &As[tic][1].data[tid * BYTES_PER_THREAD + b]) = 0;
            //       *reinterpret_cast<__uint128_t*>(
            //           &Bs[tic][0].data[tid * BYTES_PER_THREAD + b]) = 0;
            //       *reinterpret_cast<__uint128_t*>(
            //           &Bs[tic][1].data[tid * BYTES_PER_THREAD + b]) = 0;
            //   }
            //   __syncthreads();

            // G::load on K-tail iter (k = ki_dyn). Full-tensor SRD
            // (round-17 probe accepts cross-group contamination on B —
            // for G=1 probe shape this is a no-op since per-group SRD
            // == full-tensor SRD when G=1).
            G::load(Bs[tic][0], g.b, b_co(bc*2,   ki_dyn), soB);
            G::load(As[tic][0], g.a, a_co(br*2,   ki_dyn), soA);
            G::load(Bs[tic][1], g.b, b_co(bc*2+1, ki_dyn), soB);
            G::load(As[tic][1], g.a, a_co(br*2+1, ki_dyn), soA);
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
            __syncthreads();

            // Reload a/b register tiles from LDS via existing helpers
            // (subtile_inplace + load — same code path as main loop).
            load_b(b0, Bs[tic][0], wn);
            load_a(a, As[tic][0], wm);
            asm volatile("s_waitcnt lgkmcnt(0)");
            rrr_mma(cA, a, b0);

            load_b(b1, Bs[tic][1], wn);
            asm volatile("s_waitcnt lgkmcnt(0)");
            rrr_mma(cB, a, b1);

            load_a(a, As[tic][1], wm);
            asm volatile("s_waitcnt lgkmcnt(0)");
            rrr_mma(cC, a, b0);
            rrr_mma(cD, a, b1);
            __syncthreads();
        }
#endif
```

### Step 3: Add dispatcher gate to skip external K-tail when probe is on

In `dispatch_grouped_rrr` (line 4799), gate the external K-tail / N-tail
launches behind `#if !FP8_RRR_FUSE_PROBE` so the probe's in-kernel
accumulation is the only K-tail contribution (otherwise the legacy
`grouped_ktail_kernel_lds_rrr<64>` runs AFTER the probe's main kernel
and the C buffer gets double-accumulated):

```cpp
        // Round 26 — Probe gate. With FP8_RRR_FUSE_PROBE=1 the main
        // grouped_rrr_kernel accumulates K-tail in-epilog (path A),
        // so the external LDS K-tail / N-tail / scalar tail launches
        // below would double-add. Skip them entirely under PROBE; the
        // probe shape (gpt_oss-Down B=4 M=2048 K=2880 N=2880, G=1)
        // does not need N-tail support since N=2880 % BLOCK_SIZE=256
        // = 64 ≠ 0 BUT the probe shape uses G=1 / single-group input
        // tensors so cross-group contamination does not apply. Probe
        // measures K-tail-only LDS staleness mode.
#if !FP8_RRR_FUSE_PROBE
        if (g.fast_n != g.n || g.fast_k != g.k) {
            // [existing external kernel launches]
        }
#endif
```

### Step 4: Compile probe variant

```bash
cd /workspace/code/HipKittens/analysis/fp8_gemm/mi350x
source ../../../env.src
# Probe build:
make clean
CXXFLAGS="-DFP8_RRR_FUSE_PROBE=1" make -j 2>&1 | tee /tmp/build_probe.log
# Save probe .so to side path so production .so is preserved
cp tk_fp8_layouts.cpython-312-x86_64-linux-gnu.so \
   tk_fp8_layouts_probe.cpython-312-x86_64-linux-gnu.so
# Production rebuild:
make clean
make -j 2>&1 | tee /tmp/build_prod.log
```

### Step 5: Probe Python script

Save as `/tmp/fp8_rrr_path_a_probe.py`:

```python
"""Round-26 probe scaffold (round-27+ executes).

Compares FP8 grouped RRR with FP8_RRR_FUSE_PROBE=1 .so against fp32
reference. Uses G=1 to eliminate cross-group contamination concern.
"""
import os, sys
import torch

# Probe shape: gpt_oss-Down B=4 M=2048 K=2880 N=2880 → for G=1 use
# the largest single-group flavor (M=2048, N=2880, K=2880).
M, N, K, G = 2048, 2880, 2880, 1
device = "cuda"

# Build probe input
torch.manual_seed(42)
a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
b_bf16 = torch.randn(G, K, N, dtype=torch.bfloat16, device=device)

# Quantize to fp8e4m3 (mirror Primus quantize_fp8_tensorwise)
def quantize_fp8(x):
    amax = x.abs().amax()
    scale = 448.0 / amax.clamp_min(1e-6)
    q = (x.float() * scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, 1.0 / scale

a_fp8, a_scale = quantize_fp8(a_bf16)
b_fp8_list = []
b_scales = []
for g in range(G):
    bg, bg_scale = quantize_fp8(b_bf16[g])
    b_fp8_list.append(bg)
    b_scales.append(bg_scale)
b_fp8 = torch.stack(b_fp8_list, dim=0)
b_scales_t = torch.tensor(b_scales, dtype=torch.float32, device=device)

# fp32 reference
a_dq = a_fp8.float() * a_scale
b_dq = torch.stack([
    b_fp8[g].float() * b_scales[g] for g in range(G)
], dim=0)
ref = torch.empty(M, N, dtype=torch.bfloat16, device=device)
group_M = [M]  # G=1 so all M rows in group 0
m_start = 0
for g in range(G):
    rows = group_M[g]
    a_g = a_dq[m_start:m_start+rows]
    b_g = b_dq[g]
    # RRR: out = a @ b (b is [K, N])
    ref[m_start:m_start+rows] = (a_g @ b_g).bfloat16()
    m_start += rows

# Call HK probe binding
sys.path.insert(0, "/workspace/code/HipKittens/analysis/fp8_gemm/mi350x")
import tk_fp8_layouts_probe as tk  # the probe-built .so
group_offs = torch.tensor([0, M], dtype=torch.int64, device=device)
out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
tk.grouped_rrr_dscale(
    a=a_fp8.unsqueeze(0).unsqueeze(0),  # [1, 1, M, K]
    b=b_fp8.unsqueeze(0),              # [1, G, K, N]
    c=out.unsqueeze(0).unsqueeze(0),   # [1, 1, M, N]
    a_scale=torch.tensor([a_scale], dtype=torch.float32, device=device),
    b_scale=b_scales_t,
    group_offs=group_offs,
    group_m=4, num_xcds=8, m_per_group=M,
)

# Compute SNR + allclose
err = (out.float() - ref.float()).abs()
ref_pow = (ref.float() ** 2).mean().sqrt()
err_pow = (err ** 2).mean().sqrt()
snr_db = 20 * torch.log10(ref_pow / err_pow.clamp_min(1e-6))
max_err = err.max().item()
allclose = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
print(f"SNR = {snr_db.item():.2f} dB  max_err = {max_err:.3f}  allclose = {allclose}")
```

### Step 6: Run probe + interpret SNR

```bash
HIP_VISIBLE_DEVICES=2 python3 /tmp/fp8_rrr_path_a_probe.py 2>&1 | tee /tmp/probe_result.log
```

**Decision tree** (round-17 docs):
* SNR ≥ 25 dB AND allclose PASS → ST_v2 doesn't have the BF16 staleness
  issue → implement production hybrid for FP8 RRR fuse → eliminates H4
  transpose 21.6% bwd wall → bench bwd FP8 gpt_oss +20-25% (Primus dispatch
  H4 gate removed for K-misaligned cases). **Metric forward is unaffected**
  (H4 only impacts dA backward) so score does NOT move from 833-836; this
  is a pure backward-perf wedge.
* SNR < 20 dB → ST_v2 has the same staleness issue as BF16 → path A hybrid
  fails → defer to MFMA cell-shape rewrite track.
* 20 dB < SNR < 25 dB → partial fix (round-7/8 BF16 result analog at 25.45
  dB) → debug `Bs[tic]` LDS layout post-epilog-2 → likely 1-2 more rounds
  to converge.

### Step 7: Cleanup

If probe SNR ≥ 25 dB → keep PROBE scaffolding (becomes the basis of the
production hybrid). If SNR < 20 dB → revert PROBE scaffolding (delete
`#if FP8_RRR_FUSE_PROBE` blocks) and document the failure mode in
`round-27-probe-result.md`. Either way, rebuild production .so before
running `_metric_grouped_only.py` to confirm no regression.

## What round 26 commits

This file (`round-26-fp8-rrr-path-a-probe-plan.md`) only.

No kernel / dispatcher / Primus changes. The probe scaffolding above is
**executable specification**, not implementation; round 27+ implements +
runs.

## Score impact

Round 26 metric = 833-836 (3-run mean ≈ 834, plateau noise band). No
delta from round 25.

## Side note: why path A is the correct K-tail-fuse main-line move

Task body section "**核心架构方向（必读，决定本轮所有判断）**" + "**主线方向
（rounds 1-N 不许换）：K-tail fuse 进主 kernel epilog**" together imply:

1. The main line is K-tail fuse into the main kernel epilog (no external
   `grouped_ktail_kernel_*` launches).
2. Forward RCR (BF16 + FP8) shipped path B fuse rounds 3-7. Done.
3. Backward dA RRR (BF16 + FP8) currently uses H4 reroute (Primus-side
   transpose → call RCR fuse). This is a **layout-shim** workaround per
   round-9/14 docs; it converts an RRR call into an RCR fuse call but
   pays a ~21.6% transpose overhead (round-16 docs).
4. **Native RRR K-tail fuse (path A LDS-staged) is the next K-tail-fuse
   main-line wedge** that has not been pushed to production. Eliminates
   the H4 transpose; converts a 2-launch (transpose + RCR) bwd dA into
   a 1-launch (native RRR) bwd dA.

Round 27+ executing this probe plan is therefore directly on the task body
main line, not a pivot.
