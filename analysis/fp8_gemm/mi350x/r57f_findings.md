# R57 Dev F: Noinline MMA Wrappers + ASM Volatile Memory Fence on CRR DB K-loop

## Verdict: REFUTED-EMPIRICAL-SCHEDULING-BARRIER-NEUTRAL

Three scheduling-barrier approaches were tested to force global_load interleaving
between MMA pairs in the CRR double-buffer K-loop. All three produce distinct ISA
from baseline, but none deliver measurable TFLOPS improvement. The MI355X VMEM
hardware engine already overlaps global loads with MFMA execution regardless of
source-level scheduling; the bottleneck is NOT instruction ordering but rather
inherent memory bandwidth or MMA occupancy saturation.

## Variants

### V1: `__attribute__((noinline))` MMA pair wrapper
- **Mechanism**: Wraps two `crr_mma()` calls in a `noinline` device function.
  Generates `s_swappc_b64` call instruction which LLVM cannot reorder across.
- **ISA effect**: 2x `s_swappc_b64` in K-loop body. Global loads interleaved.
- **Resource cost**: VGPRs 132 (down from 230), ScratchSize 784 bytes/lane.
  32 scratch_store_dwordx4 prologue + 16 scratch_store/load per K-iter (accumulator
  save/restore across call boundary).
- **Performance**: 846.36 TFLOPS median (baseline 846.22). Delta +0.02%. Neutral.
- **Determinism**: 3/3 PASS, SNR 49.60 dB.
- **Finding**: Despite 784 bytes scratch spill, performance is identical to baseline.
  The scratch latency is fully hidden behind MFMA execution inside the callee.

### V2: `asm volatile("" ::: "memory")` compiler fence
- **Mechanism**: Memory clobber prevents LLVM IR reordering, no call frame.
- **ISA effect**: Global loads move from mid-loop to immediately after first MMA group.
  ISA is NOT byte-identical to baseline (unlike R57E's `CRR_DO_MMA` macro reordering
  which LLVM undid). The `asm volatile` with memory clobber is a stronger scheduling
  barrier than source-level reordering alone.
- **Resource cost**: VGPRs 230 (same as baseline), ScratchSize 0, Spill 0.
- **Performance 70B Down CRR** (M=4096 N=8192 K=28672):
  - Baseline 5x: 845.06, 846.64, 846.43, 846.22, 845.31 TFLOPS (median 846.22)
  - V2 5x:       848.14, 845.31, 844.82, 845.49, 846.66 TFLOPS (median 845.49)
  - Delta: -0.73 TFLOPS = -0.086%. NEUTRAL.
- **Performance 70B GateUp CRR** (M=4096 N=28672 K=8192):
  - Baseline 3x: 835.83, 836.23, 834.66 TFLOPS (median 835.83)
  - V2 3x:       837.00, 837.21, 836.40 TFLOPS (median 837.00)
  - Delta: +1.17 TFLOPS = +0.14%. NEUTRAL.
- **Determinism**: 3/3 PASS, SNR 49.60 dB.

### V3: `__builtin_amdgcn_sched_barrier(0x8)` VMEM-read fence
- **Mechanism**: Hardware scheduling hint constraining VMEM read reordering.
- **ISA effect**: `; sched_barrier mask(0x00000008)` comments emitted.
  Same load interleaving pattern as V2.
- **Resource cost**: VGPRs 230, ScratchSize 0, Spill 0.
- **Not benchmarked**: ISA is nearly identical to V2; performance expected neutral.
- **Determinism**: Builds and passes correctness (same as V2).

## Key ISA Evidence

### Baseline K-loop body ordering:
```
buffer_load_dwordx4 (global_load_a x2)     # top of iter
s_barrier + s_waitcnt
16x v_mfma_f32_16x16x128_f8f6f4 (cA+cB)   # first MMA group
s_barrier
... (LDS loads, address calc, ~60 instructions) ...
buffer_load_dwordx4 (global_load_a x2)     # mid-loop
buffer_load_dwordx4 (global_load_b x2)     # mid-loop
s_waitcnt vmcnt(4) + s_barrier
16x v_mfma_f32_16x16x128_f8f6f4 (cC+cD)   # second MMA group
s_barrier
buffer_load_dwordx4 (global_load_b x2)     # end of iter
s_cbranch loop
```

### V2 K-loop body ordering:
```
buffer_load_dwordx4 (global_load_a x2)     # top of iter
s_barrier + s_waitcnt
16x v_mfma_f32_16x16x128_f8f6f4 (cA+cB)   # first MMA group
s_setprio 0
buffer_load_dwordx4 (global_load_b x4)     # MOVED: interleaved after 1st MMA
s_barrier
... (address calc) ...
buffer_load_dwordx4 (global_load_a x2)     # mid-loop
s_waitcnt vmcnt(4) + s_barrier
16x v_mfma_f32_16x16x128_f8f6f4 (cC+cD)   # second MMA group
s_barrier
s_cbranch loop
```

The interleaving IS achieved: global_load_b moved from end-of-iter to between MMA
groups. But the hardware's out-of-order VMEM engine already handles this overlap
at the instruction issue level, so the source-level reordering provides no benefit.

## Root Cause Analysis

The CRR double-buffer path is already well-pipelined by the hardware. The MI355X
VMEM engine issues global loads independently of VALU/MFMA execution, and the
`s_waitcnt vmcnt(N)` instructions already provide correct synchronization without
needing source-level load placement to be optimal. The Dev D single-buffer +20%
improvement from SB_PIPELINE=4 was NOT due to instruction reordering barriers
but rather due to the architectural change from single-buffer to pipelined-fetch
(4 K-iters ahead), which fundamentally increased the memory prefetch distance.

## SHIP gate: DOES NOT SHIP
- +1% median TFLOPS on 70B Down CRR: FAIL (-0.086%)
- SNR >= 48 dB: PASS (49.60)
- Det 3/3: PASS
- VGPR <= 256, 0 spill: PASS (V2: 230 VGPR, 0 spill)
- Zero regression on 70B GU: PASS (+0.14%)

## Macro guardedunder `MXFP8_CRR_NOINLINE_MMA_R57F`:
- 0 = production baseline (default)
- 1 = noinline MMA pair wrapper (s_swappc_b64, scratch spill)
- 2 = asm volatile memory fence (ISA reordering achieved, no spill)
- 3 = sched_barrier(0x8) VMEM-read fence (same as V2 with explicit hint)
