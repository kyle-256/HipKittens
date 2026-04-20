# R57 Dev D: FP8-Style Interleaved Pipeline for MXFP8 CRR

## Verdict: DIAGNOSTIC-SB-INTERLEAVE-POSITIVE-VS-SB-BASELINE

The FP8 CRR `do_cluster` interleaved-MMA pipeline pattern, adapted for MXFP8 CRR
as `MXFP8_CRR_SB_PIPELINE=4`, delivers a consistent +18-19% improvement over the
naive single-buffer baseline (`SB_PIPELINE=0`) across all tested shapes. However,
this is a **single-buffer-only** improvement; the production double-buffer path
remains 22-27% faster because it overlaps global loads across K-iterations via
double-buffered LDS, which the single-buffer constraint fundamentally prevents.

The pipeline variant is functional, correct, and deterministic. It is **not a SHIP**
because it does not improve over the production double-buffer path — it improves the
single-buffer path within its structural constraints. The finding is classified as
DIAGNOSTIC because it proves the interleaved-load-between-MMA pattern works for
MXFP8 CRR and can inform future double-buffer pipeline optimizations.

## What was done

Ported the FP8 CRR `do_cluster` concept to MXFP8 CRR as `MXFP8_CRR_SB_PIPELINE=4`
in `crr_mxfp8_exact_8wave_fastpath.inc`. The key structural change vs SB_PIPELINE=3:

```
SB3: barrier -> all 4 global_load -> cC -> cD -> vmcnt(0)
SB4: barrier -> global_load_b x2 -> cC -> global_load_a x2 -> cD -> vmcnt(0)
```

The interleaving places B-side VMEM loads before cC and A-side VMEM loads between
cC and cD, giving each load batch more time in flight before the next iteration's
LDS reads need them.

### Design iterations

1. **V1 (fine-grained barriers)**: Broke each `crr_mma_scaled_from_packs_fixed_phase`
   into 8 individual `crr_mma_scaled_base` calls with `sched_barrier(0)` around every
   2 MMAs. Result: 29 VGPR spills on V2 CRR path (256 VGPRs + scratch), but ~1949
   TFLOPS — the spills were overlapped by the pipeline.

2. **V2 (coarse barriers, split cC)**: Kept MMA chains intact via
   `crr_mma_scaled_from_packs_fixed_phase` but split cC in half with
   `sched_barrier(0)` between halves to insert A-side loads. Result: still 29 VGPR
   spills (the sched_barrier forced register pressure up).

3. **V3 (sched_barrier around loads/MMAs)**: Kept MMA chains intact, added
   `sched_barrier(0)` before and after each global_load + MMA boundary. Result: 0
   spills but SLOWER (1617 TFLOPS) — the barriers prevented LLVM from overlapping
   loads with adjacent MMA tails/heads.

4. **V4 (final: structural interleave, no forced barriers)**: Kept MMA chains intact,
   placed global_loads between cC and cD without sched_barrier fences, relying on
   the source-level placement. Result: 0 VGPR spills, ~1965 TFLOPS. This is the
   committed variant.

## Performance results

### 70B Down CRR (M=4096, N=8192, K=28672) — 5-run bench

| Variant | Median TFLOPS | VGPRs (V2) | Spill |
|---------|--------------|------------|-------|
| SB_PIPELINE=0 (baseline) | 1640 | 250 | 0 |
| SB_PIPELINE=4 (this) | 1965 | 250 | 0 |
| **Delta** | **+19.8%** | **same** | **same** |
| Production DB (reference) | 2490 | 227 | 0 |

### 70B GateUp CRR (M=4096, N=28672, K=8192) — 5-run bench

| Variant | Median TFLOPS | VGPRs (V2) | Spill |
|---------|--------------|------------|-------|
| SB_PIPELINE=0 (baseline) | 1557 | 250 | 0 |
| SB_PIPELINE=4 (this) | 1903 | 250 | 0 |
| **Delta** | **+22.2%** | **same** | **same** |

### 8192^3 cross-shape check

| Variant | TFLOPS | VGPRs (V2) | Spill |
|---------|--------|------------|-------|
| SB_PIPELINE=0 | 1743 | 250 | 0 |
| SB_PIPELINE=4 | 2066 | 250 | 0 |
| **Delta** | **+18.5%** | **same** | **same** |

### Quality gates

| Cell | SNR (dB) | Det 3/3 | Pass rate |
|------|----------|---------|-----------|
| 70B Down CRR | 49.60 | PASS | 100% |
| 70B GateUp CRR | 49.60 | PASS | 100% |
| 8192^3 | 49.60 | PASS | 100% |

### Default-path byte-identity

Production double-buffer build (MXFP8_CRR_LDS_SINGLE_BUFFER=0, default) is
unaffected. The `#elif MXFP8_CRR_SB_PIPELINE == 4` block is inside the
`#if MXFP8_CRR_LDS_SINGLE_BUFFER` guard and compiles to dead code when the
macro is 0 (default).

## Why not SHIP

The single-buffer SB_PIPELINE=4 at 1965 TFLOPS is still 21% below the production
double-buffer at 2490 TFLOPS. The fundamental constraint is that single-buffer LDS
cannot overlap global loads for iteration N+1 with LDS reads for iteration N — the
single buffer must be fully read before it can be overwritten. Double-buffer avoids
this by reading from buffer[tic] while writing to buffer[toc].

The interleaved pipeline improves the single-buffer path by overlapping the global
writes (which happen AFTER the barrier, when LDS reads are already complete) with
the second half of the MMA chain. But the first half of the MMA chain (cA, cB) still
runs without any VMEM overlap, limiting the achievable utilization.

## Key insight for future work

The +19% improvement from load interleaving confirms that the WAIT-DOMINANT
bottleneck identified in R56B is real and addressable. The same interleaving
principle applied to the double-buffer path — interleaving the N+2 prefetch loads
between MMA pairs rather than issuing them in a burst — may yield an additional
+5-10% over the current production path. This would require modifying the
double-buffer K-loop in `kernel_mxfp8_layouts.cpp` lines 5414-5464.
