# R55 Dev C — 70B Down RCR P3 VALU dependency-chain break

## Mission

Attack the **70B Down RCR HEADROOM gap** (-3.4pp; baseline 91.6% MX/FP8 vs 95% gate)
identified by R54 Dev B PMC diagnostic
(`SQ_WAIT_ANY +71.6%`, `MfmaUtil −16pp`, `SQ_ACTIVE_INST_VALU +76.7%`,
`SQ_INSTS_VMEM_RD +12.5%`).

P3 lever: **VALU dependency chain break** between scale-broadcast and MFMA.
Hypothesis: LLVM scheduler is interleaving VALU instructions
(`v_lshrrev_b32`, address arithmetic, scale remap) *into* the per-cell 8-MFMA
burst sandwiched between `s_setprio(1)` and `s_setprio(0)`, expanding the
in-window VALU floor and stretching the time MFMA waits on dependent VALU.

## Treatment

Add `MXFP8_RCR_VALU_DEP_BREAK` macro (default 0) to
`analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`:

```c
#if MXFP8_RCR_VALU_DEP_BREAK >= 1
#define MXFP8_RCR_VALU_DEP_BREAK_PRE() do { __builtin_amdgcn_sched_barrier(0); } while (0)
#else
#define MXFP8_RCR_VALU_DEP_BREAK_PRE() do {} while (0)
#endif
```

Insert `MXFP8_RCR_VALU_DEP_BREAK_PRE();` immediately before each of the
**11 `s_setprio(1)`** sites in the SQUARE
`rcr_exact_8wave_scaled_kernel<true,2>` body (steady, pre-tail, tail-A,
tail-B loops). Sites: lines 3177, 3234, 3290, 3345, 3615, 3641, 3667, 3692,
3741, 3767, 3792 of `r55c_workspace/kernel_mxfp8_layouts.cpp` (after macro
addition).

Default OFF preserves baseline byte-identity.

## Phase 0 — ISA inspection

Disassembled the SQUARE RCR scaled kernel
(`_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv`) for the
70B Down cell (M=4096, N=8192, K=28672). Key findings on the **K-loop body**
(763-line window between `.LBB2_7` and the first non-loop label):

| metric | baseline |
|---|---|
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 128 |
| `v_mov_b32` (scale broadcasts) | **0** |
| `v_lshrrev_b32` (only in tails) | 6 |
| `s_mov_b32` | 22 |
| `v_readfirstlane_b32` | 18 |
| `s_waitcnt` | 21 |
| `s_barrier` | 31 |
| `buffer_load_*` | 22 |
| `ds_read_*` | 96 |

**Critical finding:** scales (`v146/v147/v148/v149` A-side, `v158/v159` B-side)
feed MFMAs *directly* from `buffer_load` result registers — there are **zero
`v_mov_b32` broadcast instructions** in the K-loop body. The LLVM scheduler
already eliminates the canonical VALU broadcast → MFMA dependency that the
"break" lever was nominally targeting.

The actual VALU pressure attributable to the +76.7% `SQ_ACTIVE_INST_VALU`
PMC must come from a different source (likely the 96 `ds_read` lane-shuffle
chains, or the 18 `v_readfirstlane` SGPR materialisations driving address
arithmetic, or accumulator move sequences in the epilogue).

Therefore the lever was reframed: instead of restructuring a non-existent
broadcast sequence, the macro inserts `__builtin_amdgcn_sched_barrier(0)`
immediately before each `s_setprio(1)`. This forces the LLVM scheduler to
push any free-floating non-MFMA instructions *upstream* of the high-priority
window so the 8-MFMA cell burst executes back-to-back without VALU
interleaving.

## Phase 1 — implementation + rebuild

Resource summaries (`-Rpass-analysis=kernel-resource-usage`) for the SQUARE
RCR scaled kernel:

| instantiation | baseline | treatment |
|---|---|---|
| `<true, 2>` (PRESHUFFLE_QUANT V2 — primary) | VGPRs **254** spill 0 | VGPRs **250** spill 0 |
| `<true, 1>` (PRESHUFFLE_QUANT V1) | VGPRs 252 spill 0 | VGPRs 252 spill 0 |
| `<false, 1>` (legacy V1) | VGPRs 256 spill 0 | VGPRs 256 spill 41 |

**Side effect:** the `<false, 1>` legacy V1 instantiation regresses to 41
VGPR spills. This instantiation is *not* used in our test path
(`MXFP8_PRESHUFFLE_QUANT=1` always selects the `<true, *>` family); the
production tree only ever launches `<true, 2>` (V2 preshuffled) for any cell
where R52/R54 selected V2. The macro must therefore stay default-OFF; if a
production caller ever launches `<false, 1>`, the macro must remain disabled
for that path. Acceptable.

For the **primary** `<true, 2>` instantiation register pressure DROPS from
254 → 250 VGPR (still occ 2, no spill, LDS 131072 unchanged). The
sched_barriers indirectly relieved register pressure by reducing the
inter-MFMA register live range.

ISA spot-check: total opcode counts in the kernel are **identical**
(mfma=128, v_mov=128, v_lshrrev=9, s_waitcnt=27, s_barrier=34, s_setprio=30)
between baseline and treatment. The *layout* shifted (treatment kernel is
1562 lines vs 1547 baseline, register colouring rearranged
e.g. A data-tile registers moved from `v[214:221]` → `v[192:199]` and the
B-side scale receivers re-clustered). No new opcodes inserted.

Smoke (single 20-warmup, 50-iter run on 70B Down RCR): **2895.10 TFLOPS,
SNR 49.61 dB, det 3/3 PASS**.

## Phase 2 — bench (5×, MXFP8_WARMUP=100, MXFP8_ITERS=200, GPU 2)

`r55c_workspace/r55c_phase2_bench.sh` — 5 runs/cell, 30s cooldown between
runs, 60s rebuild_cool between gate switches, `HIP_VISIBLE_DEVICES=2`.
Median (rank-3-of-5) is the score.

| Cell | Baseline (TFLOPS) | Treatment (TFLOPS) | Δ%      | SNR base/treat | Det base/treat |
|------|-------------------|--------------------|---------|---------------:|---------------:|
| **70B_Down_RCR** (M=4096 N=8192 K=28672) | **2930.69** | **2949.67** | **+0.65%** | 49.61 / 49.61 | 5/5 / 5/5 |
| 70B_QO_RCR     (M=4096 N=8192 K=8192)  | 2874.34     | 2878.50     | +0.14%   | 49.60 / 49.60 | 5/5 / 5/5 |
| 8B_Down_RCR    (M=4096 N=4096 K=14336) | 2928.28     | 2929.43     | +0.04%   | 49.60 / 49.60 | 5/5 / 5/5 |

Full TFLOPS distributions (sorted ascending across 5 runs):

```
70B_Down_RCR  baseline:  2896.04 2927.85 2930.69 2939.43 2943.23
70B_Down_RCR  treatment: 2943.43 2944.92 2949.67 2955.02 2955.78
70B_QO_RCR    baseline:  2862.67 2867.21 2874.34 2874.53 2877.96
70B_QO_RCR    treatment: 2867.48 2877.97 2878.50 2879.45 2886.83
8B_Down_RCR   baseline:  2880.25 2920.23 2928.28 2928.43 2929.91
8B_Down_RCR   treatment: 2920.67 2923.40 2929.43 2932.91 2933.40
```

**The 70B Down RCR effect is monotone-clean: the worst treatment run
(2943.43) is on par with the best baseline run (2943.23).** The signal
is real but small — well below the +1% pass threshold.

## Verdict

The treatment produces a **consistent, monotonic-clean +0.65% gain on the
70B Down RCR primary cell**, with no SNR or determinism regressions, and
preserves correctness on both co-cells (8B Down RCR +0.04%, 70B Q/O RCR
+0.14% — both within run-to-run noise). VGPR pressure on the primary
`<true,2>` instantiation drops 254 → 250.

But +0.65% **does not meet the R55C pass criterion of ≥+1%** on the primary
cell. The Phase 0 ISA inspection had already identified that the lever as
specified (broadcast → MFMA chain break) was structurally vacuous — the
compiler emits zero `v_mov_b32` broadcasts in the K-loop body, scales feed
MFMAs directly. The fallback `sched_barrier(0)` directive does shift the
schedule (lower VGPR pressure, +0.65% TFLOPS) but the underlying VALU
pressure source (likely `ds_read` lane-shuffle chains, scale-shift epilogue,
or address-materialisation `v_readfirstlane`) is not addressed by this
intervention.

Side-effect: legacy `<false,1>` instantiation regresses to 41 VGPR spills
when macro=1. Production safety requires macro stays default-OFF; only the
preshuffled paths benefit.

### Final line

R55 Dev C: 70B Down RCR P3 VALU dependency break — REFUTED-PASS-CRITERION-EMPIRICAL — sched_barrier(0) before each of 11 setprio(1) sites in SQUARE rcr_exact_8wave_scaled_kernel<true,2> yielded a clean monotonic +0.65% on 70B Down RCR (median 2930.69 → 2949.67 TFLOPS, every treatment run ≥ best baseline run) with VGPRs 254 → 250 and SNR/det fully preserved on all 3 cells, but +0.65% < +1% pass threshold; Phase 0 ISA proved the canonical "broadcast → MFMA" chain does not exist (0 v_mov_b32 broadcasts in K-loop; scales feed MFMAs directly from buffer_load result registers), so the +76.7% SQ_ACTIVE_INST_VALU floor must come from ds_read lane-shuffles / readfirstlane address arith / accumulator moves not from the targeted dep chain; legacy `<false,1>` instantiation regresses to 41 VGPR spills (production unaffected since MXFP8_PRESHUFFLE_QUANT=1 path always selects `<true,*>`). Macro left default-OFF.
