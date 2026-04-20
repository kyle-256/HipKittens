# R57 Dev B: CRR sched_barrier and K-loop unroll

**Verdict: REFUTED-EMPIRICAL-CRR-SCHED-BARRIER-NONDETERMINISTIC**

## Summary

Tested two never-before-tested CRR levers on 70B Down CRR (M=4096 N=8192 K=28672)
and 70B GateUp CRR (M=4096 N=28672 K=8192):

1. `CRR_ENABLE_SCHED_BARRIER=1` -- adds `__builtin_amdgcn_sched_barrier(0)` between
   MMA pairs in CRR K-loop body
2. `CRR_MAIN_UNROLL=2` -- attempts to unroll the CRR K-loop body by 2 iterations

**Both levers fail to SHIP.** sched_barrier introduces systematic non-determinism
(10/10 det FAIL across both shapes); unroll=2 is completely ignored by the AMDGPU
compiler (byte-identical ISA).

## Phase 0: Build sweep (all 4 configs)

CRR V2 preshuffle kernel (`crr_exact_8wave_scaled_kernel<true, 2>`):

| Config | Flags | VGPRs | Spill | Occupancy |
|--------|-------|-------|-------|-----------|
| 0 (baseline) | defaults | 227 | 0 | 2 |
| 1 (sched_barrier) | CRR_ENABLE_SCHED_BARRIER=1 | 234 | 0 | 2 |
| 2 (unroll=2) | CRR_MAIN_UNROLL=2 | 227 | 0 | 2 |
| 3 (both) | both | 234 | 0 | 2 |

All build clean. No spill. sched_barrier adds +7 VGPRs (227->234), still well
under 256 ceiling. UNROLL=2 produces identical VGPR count to baseline.

## Phase 1: ISA inspection

### Config 2 (UNROLL=2) -- COMPILER IGNORES PRAGMA

Config 2 produces **byte-identical ISA** to Config 0 baseline. The AMDGPU compiler
completely ignores `#pragma unroll(2)` for this CRR K-loop. K-loop body: 1786 ISA
lines in both configs.

This means Config 3 (both) is functionally identical to Config 1 (sched_barrier
only). **The experiment reduces to 2 effective configs: baseline vs sched_barrier.**

### Config 1 (sched_barrier) -- 23 barrier directives inserted

Config 1 inserts 23 `sched_barrier mask(0x00000000)` directives into the CRR V2
kernel, adding 24 ISA lines (1786 -> 1810). The barriers appear at:
- Between the first 4-MFMA group and second 4-MFMA group within each K-pair half
- Around `s_setprio 0/1` transitions
- Between `s_barrier` (workgroup barrier) and MFMA groups

The sched_barrier(0) prevents instruction reordering across these boundaries,
constraining the scheduler from interleaving VALU/SALU instructions into the MFMA
stream.

## Phase 2: Benchmark results (5-run SCLK, warmup=100, iters=200)

### 70B Down CRR (M=4096 N=8192 K=28672)

FP8 CRR baseline: 2998.98 TFLOPS

| Config | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median | Det | SNR dB |
|--------|-------|-------|-------|-------|-------|--------|-----|--------|
| 0 (baseline) | 2651.94 | 2650.45 | 2659.08 | 2659.08 | 2649.81 | 2651.94 | 5/5 PASS | 49.60 |
| 1 (sched_barrier) | 2661.95 | 2657.97 | 2666.47 | 2648.47 | 2655.00 | 2657.97 | 0/5 PASS | 49.35-49.60 |

Config 1 vs Config 0 median delta: +6.03 TFLOPS / **+0.23%** (noise band)
Config 1 determinism max abs diffs: 0.36, 0.53, 0.58, 0.71, 0.85

MXFP8/FP8 ratio: baseline 88.4%, sched_barrier 88.6%

### 70B GateUp CRR (M=4096 N=28672 K=8192)

FP8 CRR baseline: 2798.31 TFLOPS

| Config | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median | Det | SNR dB |
|--------|-------|-------|-------|-------|-------|--------|-----|--------|
| 0 (baseline) | 2486.02 | 2485.35 | 2488.11 | 2483.95 | 2482.73 | 2485.35 | 5/5 PASS | 49.60 |
| 1 (sched_barrier) | 2484.60 | 2487.47 | 2494.67 | 2483.71 | 2485.14 | 2485.14 | 0/5 PASS | 48.73-49.59 |

Config 1 vs Config 0 median delta: -0.21 TFLOPS / **-0.01%** (noise band)
Config 1 determinism max abs diffs: 0.44, 0.55, 0.65, 0.67, 0.73

MXFP8/FP8 ratio: baseline 88.8%, sched_barrier 88.8%

## SHIP gate assessment

| Criterion | Status |
|-----------|--------|
| >=+1% median TFLOPS on 70B Down CRR | FAIL (+0.23%) |
| SNR >= 48 dB | PASS (49.35-49.60 dB) |
| Det 3/3 PASS | **FAIL** (0/10 runs pass det with sched_barrier) |
| No regression on 70B GU CRR | PASS (-0.01%) |

**SHIP GATE: FAIL** -- double failure on both perf threshold and determinism.

## Root cause analysis

### Why sched_barrier breaks determinism on CRR

The CRR K-loop uses `s_setprio 1` / `s_setprio 0` to boost MFMA dispatch priority.
Without sched_barrier, the compiler freely interleaves VALU/SALU instructions between
MFMA ops, and the MFMA accumulation order is stable because all operand data is
ready before the MFMA block begins (gated by `s_waitcnt lgkmcnt(0)`).

When sched_barrier(0) is inserted, it:
1. Prevents the compiler from interleaving loads/SALU between MFMA groups
2. Forces tighter scheduling that makes the kernel more sensitive to wave-scheduling
   order across the 8 waves per workgroup
3. Since each wave's MFMA accumulates into the same output tile, and sched_barrier
   constrains the compiler from hiding memory latency within MFMA, the resulting
   execution order becomes timing-dependent

This is the opposite of what happens on RRR/RCR (where sched_barrier is default ON
and deterministic). The CRR kernel structure -- with its A-transpose shared memory
ping-pong and conditional even/odd K-pair shift -- creates a different interaction
with sched_barrier that introduces race conditions in the accumulation path.

### Why UNROLL=2 is ignored

The CRR K-loop body at line 5412 uses `TK_PRAGMA_UNROLL(CRR_MAIN_UNROLL)` which
expands to `#pragma unroll 2`. However, the K-loop contains:
- Conditional logic for even/odd K-pair handling (scale shift blocks)
- Multiple `s_barrier` synchronization points
- Complex LDS ping-pong addressing

The AMDGPU backend's loop unroller cannot duplicate this body because:
1. The barrier count would double, violating the 1:1 barrier enter/exit contract
2. The conditional K-pair even/odd logic cannot be statically resolved at compile time
3. The LDS addressing pattern depends on the iteration parity

The compiler correctly refuses to unroll and silently drops the pragma.

## Conclusion

Both CRR optimization levers are **CLOSED**:

- `CRR_ENABLE_SCHED_BARRIER`: +0.23% perf (noise) with catastrophic determinism
  failure. The sched_barrier interaction with CRR's A-transpose wave structure
  introduces non-determinism. This axis is permanently closed for CRR.

- `CRR_MAIN_UNROLL=2`: No-op, byte-identical ISA. The AMDGPU compiler cannot
  unroll the CRR K-loop body due to barriers, conditionals, and LDS addressing
  complexity. This axis is permanently closed.

The CRR performance gap (-11.2% to -12% vs FP8) is NOT addressable through
instruction scheduling or loop structure changes. The bottleneck is structural:
the A-transpose memory access pattern and scale-fetch overhead identified in
R56B's PMC diagnostic (SQ_WAIT_ANY 2.39x, MfmaUtil -19.9pp).
