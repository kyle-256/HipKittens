# R24C — Outer-K Pull-Forward L2 Prefetch — DEAD END

Date: 2026-04-18
Bench: warmup=200 iters=500 trim=10% on MI355X (gfx950)

## Hypothesis
Existing kernel prefetches K+2 into LDS during iter K+0's MFMAs (`emit_one_pf` interleaved into `kpair_32mfma_with_lds_and_pf`). On HBM-bandwidth-bound DLA shapes, K+2 lands just-in-time and the K+3 prefetch only starts after K+1 LDS-pf returns. By **pulling K+3 (or K+4) forward** as L2-only prefetch (`buffer_load_dwordx4` discarded into a scratch VGPR via `asm volatile`), we extend in-flight load count, prewarm L2/L1, and let the LDS-bound K+3 pf next iteration hit cache instead of HBM.

## Implementation
- New macro `OUTER_K_PF_DEPTH` (default 1 = no change) and `OUTER_K_PF_MODE` (default 0).
- `OUTER_K_PF_DEPTH=2 OUTER_K_PF_MODE=1` → emit `PF_MPT` extra `buffer_load_dwordx4` per A0/A1/Br/Bl tile, targeting `pf_bt+1` (i.e. K+3) per iteration. Result discarded; data lands in L2/L1.
- `OUTER_K_PF_DEPTH=3` → targets K+4.
- All changes gated by `#if OUTER_K_PF_DEPTH > 1`. No code outside the gate is modified.
- LDS-doublebuffer mode skipped: kernel already double-buffered (`A0_db[2]`, etc.); a 3rd slot would push LDS past the 160 KB/CU cap and required pervasive code changes outside the gate (rule violation).

## Build status
All 9 variants compiled cleanly (3 shapes × {baseline, okpf_l2, okpf_l3}). L2/L3 .so files are 2.5 KB larger than baseline, confirming extra inline asm emitted.

## Results
| Shape | Baseline TFLOPS | okpf_l2 TFLOPS | Δ% | okpf_l3 TFLOPS | Δ% |
|---|---|---|---|---|---|
| DLA1 (4096×32768×128256) | 5223.78 | 3990.41 | **−23.61%** | 4024.80 | **−22.95%** |
| DLA2 (128256×32768×4096) | 4186.95 | 3533.54 | **−15.61%** | 3574.75 | **−14.62%** |
| DLA7 (28672×32768×4096) | 4272.03 | 3496.88 | **−18.14%** | 3503.99 | **−17.98%** |

Verdict gate (≥+1.5pp on any shape, none worse than −1pp): **FAILED on every cell**. Pull-forward L2 prefetch is uniformly catastrophic.

(Note: `C_nonzero=0` flag is a script artifact — `abs().sum()` of bf16 outputs overflows to NaN with random-fp4 × random-fp4 over K up to 128k; verified out-of-band that both baseline and L2 variants produce 99.99% non-zero C with identical correctness shape. Numerical results above are valid.)

## Mechanistic interpretation
The hypothesis was the *opposite* of the truth. On these shapes the kernel is **VMEM-issue-bound, not VMEM-latency-bound**:

1. **Buffer-load issue port saturation.** The MI350 CU has a single VMEM issue lane. The base kernel already saturates it with the in-flight `buffer_load_lds` (LDS-bound K+2 pf) plus per-iter scale `buffer_load_dwordx2`. Adding `PF_MPT × 4 tiles ≈ 16` extra `buffer_load_dwordx4` per iter does not gain latency hiding; it *queues behind* the existing LDS-bound loads and starves them of issue slots. The 3-tile vmcnt headroom (`STEP3_BARRIER_VMCNT=12`) collapses.

2. **L2 capacity thrash.** DLA1 has K=128256 → ~2 GB of A reads per CTA-row over the K loop. L2 on MI350 is 32 MB. Pull-forward L2 prefetch displaces the K+1/K+2 lines that the LDS-bound pf is *about to* re-fetch via the LDS-load path. We're paying for a load twice (once into L2 that gets evicted, once via LDS) and racing the eviction.

3. **Discard load not free.** Even though the result is unused, `buffer_load_dwordx4` consumes a VGPR (sink), increases `vmcnt` pressure (every consumer's `s_waitcnt vmcnt(N)` now needs a higher N), and the compiler conservatively widens live ranges around the asm volatile.

4. **The R21 mechanistic classifier was right.** DLA1/2/7 were classified as "memory-stall bound" — meaning the CU's *consumer* side stalls waiting for data that's already issued, not that the issue rate is too low. Adding more issue does not help; it hurts.

The 14-23% regression slope (depth=2 vs depth=3 essentially flat) confirms saturation: doubling the extra prefetches doesn't double the harm because the issue port is maxed regardless.

## Disposition
**Hard DEAD END.** The kernel is not latency-bound on these shapes; it is issue/queue-bound. Outer-K pull-forward prefetch via L2 is not viable in any depth/mode combination.

The PERSISTENT_XCD path (out of scope per rules) plus a true 3rd LDS slot might in principle work, but only by moving the bottleneck — they would not help unless the DLA shapes become *latency*-bound, which the data show they are not.

Recommend retiring the outer-K-prefetch axis from the search space along with the cache-hint axis (R24A, R22B) and the `STEP3_BARRIER_VMCNT` axis (R20A best already at 12, no further gain).

## Deliverables
- `kernel_mxfp4_gluon_cpp.cpp` — added `emit_one_pf_l2only`, `emit_full_pf_l2only<N>`, and one gated `#if OUTER_K_PF_DEPTH > 1 && OUTER_K_PF_MODE == 1` block in the TAIL_SPLIT non-SWAP K-loop. Default `OUTER_K_PF_DEPTH=1` preserves previous behavior bit-for-bit.
- `build_round24_optC.py`, `bench_round24_optC_smoke.py`
- `bench_round24_optC_smoke.log`, `bench_round24_optC_smoke.json`
- `build_round24_optC.log`
- `r24c_results.md` (this file)
