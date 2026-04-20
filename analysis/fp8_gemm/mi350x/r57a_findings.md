# R57 Dev A: CRR_STEADY_VMCNT sweep — REFUTED-EMPIRICAL-VMCNT-FLAT

## Verdict: REFUTED

CRR_STEADY_VMCNT sweep across {4, 6, 8, 12} produces no meaningful performance
delta on either 70B Down CRR or 70B GateUp CRR. Best variant (vmcnt=6) shows
+0.20% on 70B Down vs baseline vmcnt=4 — within the 0.084% noise stdev,
well below the +1% SHIP gate. VMCNT overlap is NOT the bottleneck on these cells.

## Hypothesis (from R56B WAIT-DOMINANT class)

R56B PMC showed SQ_WAIT_ANY ratio = 2.389 (+139%) on 70B Down CRR. Hypothesis:
production vmcnt=4 is too aggressive (drains too much), causing MMA stalls while
waiting for VMEM completions that could overlap with compute. Larger vmcnt values
would allow more VMEM ops in-flight during MMA, reducing wait stalls.

## Build note

**CRITICAL**: The Makefile does not process `CXXFLAGS_EXTRA`. Must use `CPPFLAGS`
for compile-time defines:
```bash
make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp \
  CPPFLAGS="-DM_DIM=4096 -DN_DIM=8192 -DK_DIM=28672 -DCRR_STEADY_VMCNT=<N>"
```

Initial Phase 1 ISA inspection revealed the `CXXFLAGS_EXTRA` path produced
byte-identical GPU code objects across all vmcnt values. Switching to `CPPFLAGS`
confirmed binary diffs and ISA changes (5 `s_waitcnt vmcnt(N)` sites in the
CRR `<true,2>` kernel change with the define).

## Phase 0: VGPR check — ALL PASS

| vmcnt | VGPRs (CRR <true,2>) | Spill | Occupancy |
|-------|----------------------|-------|-----------|
| 4     | 227                  | 0     | 2         |
| 6     | 227                  | 0     | 2         |
| 8     | 227                  | 0     | 2         |
| 12    | 227                  | 0     | 2         |

## Phase 1: ISA inspection — LEVER CONFIRMED

5 `s_waitcnt vmcnt(N)` sites in the CRR kernel change value with the define.
Binary diff between vmcnt=4 and vmcnt=12 GPU code objects confirmed DIFFERENT
(char 20926). Other kernels (RCR, RRR) are unaffected.

## Phase 2: 5-run bench — ALL WITHIN NOISE

### 70B Down CRR (4096x8192x28672) — primary target

| vmcnt | Run 1   | Run 2   | Run 3   | Run 4   | Run 5   | Median  | Delta % |
|-------|---------|---------|---------|---------|---------|---------|---------|
| 4     | 843.08  | 844.85  | 842.74  | 845.00  | 843.45  | 843.45  | baseline|
| 6     | 845.35  | 844.02  | 845.23  | 844.20  | 845.15  | 845.15  | +0.20%  |
| 8     | 844.45  | 844.72  | 844.49  | 844.09  | 845.11  | 844.49  | +0.12%  |
| 12    | 844.18  | 844.19  | 844.70  | 844.29  | 845.04  | 844.29  | +0.10%  |

Global mean: 844.42 TFLOPS, global stdev: 0.71 TFLOPS (0.084%).
All deltas within noise band.

### 70B GateUp CRR (4096x28672x8192) — cross-shape check

| vmcnt | Run 1   | Run 2   | Run 3   | Run 4   | Run 5   | Median  | Delta % |
|-------|---------|---------|---------|---------|---------|---------|---------|
| 4     | 836.80  | 836.51  | 836.28  | 836.21  | 836.29  | 836.29  | baseline|
| 6     | 836.18  | 837.30  | 836.68  | 837.34  | 837.17  | 837.17  | +0.11%  |
| 8     | 836.27  | 836.65  | 836.62  | 836.80  | 836.62  | 836.62  | +0.04%  |
| 12    | 836.65  | 835.78  | 836.65  | 836.63  | 836.55  | 836.63  | +0.04%  |

No regression on GateUp — all within +/-0.5% noise band.

### FP8 Baseline (for ratio reference)

| Cell         | Median TFLOPS |
|--------------|---------------|
| 70B Down CRR | 3059.92       |
| 70B GateUp   | 2842.85       |

MX/FP8 ratio: 27.6% (70B Down), 29.4% (70B GateUp).

## SHIP gate evaluation

- Required: >=+1% median TFLOPS vs baseline on 70B Down CRR
- Best: vmcnt=6 at +0.20%
- **FAIL** — 5x below the SHIP threshold

## Macro disposition

REFUTED — keep production default `CRR_STEADY_VMCNT=4`. No macro added.

## Implications for WAIT-DOMINANT bottleneck

The R56B SQ_WAIT_ANY +139% observation is real, but relaxing the steady-state
vmcnt drain in the K-loop body does not help. Possible explanations:

1. **AMDGPU scheduler ignores the relaxed vmcnt**: The compiler may re-tighten
   the s_waitcnt below the programmer's requested value if it determines the
   subsequent asm needs the data. This appears NOT to be the case (ISA confirmed
   the vmcnt value changes in 5 sites).

2. **VMEM latency is not the dominant wait source**: The SQ_WAIT_ANY counter
   may be dominated by LDS/barrier waits rather than VMEM waits. The CRR kernel
   has `s_barrier` immediately after each `s_waitcnt vmcnt(N)`, so the barrier
   synchronization itself may be the actual wait source.

3. **Scale-fetch is already fully overlapped**: At vmcnt=4, the 4 remaining
   in-flight loads may already be the new iteration's prefetches, meaning the
   previous iteration's data is fully resolved. Increasing vmcnt just adds
   slack that the hardware doesn't use.

4. **Memory bandwidth limited**: The CRR cell may be hitting memory bandwidth
   limits, not compute/wait limits. Relaxing vmcnt doesn't help if the
   bottleneck is upstream at the memory controller.

## Bench setup

- GPU: AMD Instinct MI355X (gfx950), HIP_VISIBLE_DEVICES=0
- Warmup: 100 iters, Bench: 200 iters per run
- 30s cooldown between runs, 60s cooldown between rebuilds
- 5 runs per variant per cell
- Build: `CPPFLAGS="-DM_DIM=... -DN_DIM=... -DK_DIM=... -DCRR_STEADY_VMCNT=N"`
