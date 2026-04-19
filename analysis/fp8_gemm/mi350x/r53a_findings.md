# R53 Dev A — CRR scale-cache layout V2 (opsel-keyed K-phase MMA dispatch) — REFUTED

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ bd59941f
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=0
**Hypothesis:** Eliminate the 6 `v_lshrrev_b32` shifts/K-pair that R48D
identified as the structural CRR ~92% ceiling by switching the steady
K-loop from `crr_mma_scaled_from_packs_fixed_phase` (always opsel<0..3>,
preceded by shifts that move bytes 2/3 → bytes 0/1 on odd iters) to
`crr_mma_scaled_phase<K_PHASE>` (opsel<0..3> on even iters, opsel<2,2..3,3>
on odd iters — bytes 2/3 of the scale-pack VGPR consumed directly via
opsel without any intermediate shift). Gated by
`MXFP8_CRR_SCALE_LAYOUT_V2=1` (default OFF); baseline kept as `#else` arm.

## TL;DR — VERDICT: REFUTED

One-line summary: *CRR opsel-keyed K-phase MMA dispatch eliminates the 6
`v_lshrrev_b32`/K-pair as designed (ISA-confirmed), but inlining both
phase<0> and phase<1> opsel-templated MMA chains inside the runtime-branched
body forces a 70-VGPR spill / 144 byte/lane scratch / 56 scratch_load+store
inner-loop traffic, regressing the three target cells by -63 to -69%.*

| Cell                   | Baseline TFLOPS | V2 TFLOPS | Δ%      | SNR (both) |
|------------------------|-----------------|-----------|--------:|------------|
| 70B Gate/Up 4096×28672×8192 | 2509       | 797       | **-68.2%** | 49.60 dB PASS |
| 70B Q/O     4096×8192×8192  | 2572       | 938       | -63.5%  | 49.59 dB PASS |
| 8192³        8192×8192×8192 | 2702       | 824       | -69.5%  | 49.60 dB PASS |

Geomean ~-67%. SHIP gate (geomean ≥ +1.5%, worst ≥ -2%) FAILS by ~50× on
both axes. Numerics correct (det 3/3 PASS, SNR identical to baseline) — the
regression is purely a performance pessimization from the body-duplication
spill, not a correctness bug.

## Lever choice and rationale

R48D ceiling analysis pinpointed 6 `v_lshrrev_b32`/K-pair as the dominant
delta vs RCR/RRR (CRR ~92% vs RCR ~95% vs RRR ~94% on the FP8 per-tensor
scale). R49A's failure mode (host-side scale re-pack → -14.16% from
compiler-inserted `v_perm_b32`) ruled out the layout-side approach. R50A's
failure mode (scale prefetch lead-distance → 5 VGPR spill / 24 byte scratch)
ruled out the live-range-extension approach.

R53A picks a third axis: **leave the scale-pack BIT-IDENTICAL on the host,
leave live ranges unchanged, and instead change the MMA opsel template**
to consume bytes 2/3 of the scale-pack VGPR directly (`opsel<2,2>..<3,3>`
on odd K-iters) instead of shifting them down to bytes 0/1 (`opsel<0,0>..<1,1>`).
The infrastructure for this was already present in `kernel_mxfp8_layouts.cpp`
as `crr_mma_scaled_phase<K_PHASE>` (used by RRR's similar pattern); R53A
just routes CRR through it on odd K-iters.

This was lever #1 from the R53 prompt's CRR candidate list. It is genuinely
distinct from R49A (no host change, no v_perm) and R50A (no live-range
extension), so the analytical bail conditions for those cycles do not
trigger here. The bail condition that DOES trigger is body-duplication
VGPR pressure, which was not previously enumerated.

## Source diff summary

`crr_mxfp8_exact_8wave_fastpath.inc`:

1. New macro `MXFP8_CRR_SCALE_LAYOUT_V2` (default 0) with 30+ line
   doc block describing the lever, its predicted upside (~5pp), and the
   failure mode actually observed.
2. Compatibility guard:
   ```cpp
   #if MXFP8_CRR_SCALE_LAYOUT_V2 && MXFP8_CRR_SCALE_LEAD
   #error "MXFP8_CRR_SCALE_LAYOUT_V2 and MXFP8_CRR_SCALE_LEAD are mutually exclusive"
   #endif
   ```
3. Steady K-loop body (lines ~968–1093): on every k iteration, runtime
   branch on `(k & 1)`:
   - `LAYOUT_V2=0` (baseline): even iter → MMA at fixed_phase opsel<0..3>;
     odd iter → 6× `v_lshrrev_b32` to shift bytes 2/3 down to bytes 0/1,
     then MMA at fixed_phase opsel<0..3>.
   - `LAYOUT_V2=1` (treatment): even iter → `crr_mma_scaled_phase<0>`
     (opsel<0,0>..<1,1>, consumes bytes 0/1); odd iter → `crr_mma_scaled_phase<1>`
     (opsel<2,2>..<3,3>, consumes bytes 2/3 directly, no shift).

Both `crr_exact_cA_with_b1_interleave_raw_phase<K_PHASE>` and
`crr_mma_scaled_phase<K_PHASE>` already exist in `kernel_mxfp8_layouts.cpp`
— no new helpers were introduced.

The `LAYOUT_V2=0` arm is byte-identical to the prior baseline (validated
by build remark: 227 VGPR / 0 spill / 0 scratch, matches pre-patch
exactly across all 3 cells).

## ISA before/after (70B Gate/Up, `_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv`)

| Metric                | Baseline | V2     | Δ                                              |
|-----------------------|---------:|-------:|------------------------------------------------|
| Total kernel lines    | 3447     | 3767   | +320 (+9.3%)                                   |
| `v_lshrrev_b32` count | 13       | 7      | **-6** — lever WORKED, 6 shifts gone as predicted |
| `v_perm_b32` count    | 0        | 0      | 0 — distinct from R49A, no lane perms forced   |
| `v_mfma_*` count      | 192      | 224    | **+32** — body duplicated for both opsel phases |
| `scratch_load` count  | 0        | 26     | **+26** — NEW spill traffic in inner hot loop  |
| `scratch_store` count | 0        | 30     | **+30** — NEW spill traffic in inner hot loop  |

Build resource A/B (consistent across all 3 cells):

| Cell                       | Baseline VGPR/Scratch/Spill | V2 VGPR/Scratch/Spill |
|----------------------------|-----------------------------|-----------------------|
| 70B Gate/Up 4096×28672×8192 | 227 / 0 / 0                | 256 / 144 / **70**    |
| 70B Q/O     4096×8192×8192  | 227 / 0 / 0                | 256 / 144 / **70**    |
| 8192³        8192×8192×8192 | 227 / 0 / 0                | 256 / 144 / **70**    |

CRR_MAIN_UNROLL=2 and =4 builds were exercised to test whether unroll
factor mediated the spill: same 256 VGPR / 70 spill / 144 byte scratch
at unroll=1, 2, AND 4. Spill is structural to inlining both opsel phases
in the body, not a function of unroll factor.

## Strict-SCLK A/B numbers

A quick A/B (warmup=20, iters=50, 1 run/cell) was used to MEASURE the
regression magnitude after the build evidence (70 VGPR spill, +56
scratch_load/store) made the verdict unambiguous. Per R50 protocol, when
the bail signal is this strong (~9× cost inflation per K-pair from
scratch traffic in the inner loop, on a kernel already in HEADROOM
territory), the full 5-runs/30s-cooldown/60s-rebuild bench is not
required to reach the REFUTED verdict — even with worst-case SCLK slip,
the lever cannot recover from -65% mean.

The full bench harness `r53a_bench.sh` (3 runs/arm × 3 cells × 30s
cooldown × 60s rebuild) is committed for reproducibility but was not
executed.

| Cell                       | Baseline TFLOPS | V2 TFLOPS | Δ%      |
|----------------------------|----------------:|----------:|--------:|
| 70B Gate/Up 4096×28672×8192 | 2509           | 797       | -68.2%  |
| 70B Q/O     4096×8192×8192  | 2572           | 938       | -63.5%  |
| 8192³        8192×8192×8192 | 2702           | 824       | -69.5%  |

Geomean ~-67%.

## SNR + det results

All three V2 cells: SNR 49.59–49.60 dB (PASS, ≥ 45 dB gate), det 3/3 PASS,
identical to baseline SNR. The kernel is functionally correct under V2 —
the regression is pure throughput, not numerics.

## Diagnosis

At unroll=1, the steady K-loop body contains a runtime branch on `(k & 1)`.
The compiler must materialize BOTH phase<0> and phase<1> MMA chains in the
inlined body because the MFMA opsel is encoded as a 2-bit immediate in the
instruction (not a runtime input):

- phase<0>: 8 MFMAs at opsel<0,0>..<1,1> (cA-with-b1-interleave + cB) +
  8 more MFMAs for cC + cD = 16 distinct MFMA instructions
- phase<1>: 8 MFMAs at opsel<2,2>..<3,3> + 8 more = 16 distinct MFMAs
- Total: 32 MFMA instructions in body (vs 16 in baseline) — confirmed
  by `v_mfma_*` count delta (+32) in the ISA dump.

Each MFMA opcode has the opsel encoded in the instruction immediate, so
the compiler cannot share code between the two phases. The duplicated
body saturates the 256-VGPR cap (occupancy 2 retained but 70 VGPRs
spilled to per-lane scratch totaling 144 bytes/lane).

Baseline trades 6 `v_lshrrev_b32` (3 shifts × 2 packs/cluster) per K-pair
for ZERO spill traffic. R53A trades 6 `v_lshrrev_b32` for 56
`scratch_load`/`scratch_store` inner-loop instructions — a ~9× cost
inflation per K-pair.

## Why this is distinct from R49A and R50A (and what it teaches)

- **R49A (CRR opsel re-pack at host preshuffle):** repacked scales at the
  HOST side so ALL K-iters could use opsel<0..3>. Failure mode: extra
  `v_perm_b32` lane permutations + occupancy 2→1 collapse at 8192³.
- **R50A (CRR scale prefetch lead-distance):** pre-issued scales 1 BK
  earlier, extending live range. Failure mode: 5 VGPR spill / 24 byte
  scratch.
- **R53A (this cycle):** NO host change, NO v_perm, NO live-range
  extension. Failure mode: 70 VGPR spill / 144 byte scratch from body
  duplication of opsel-templated MMA chains — **~14× worse spill than R50A**,
  ~5× worse throughput regression than R49A.

The lesson: the 256-VGPR / 8-wave / occupancy-2 budget for CRR has zero
slack for ANY kernel-side approach that increases the count of distinct
MMA opsel template instantiations inside the inlined body. Every viable
CRR lever from now must either (a) keep MMA opsel a compile-time constant
across all K-iters (which is what R49A tried, and it forced v_perm), or
(b) move the work entirely outside the body (LDS-side staging — lever #4
from the R53 prompt — untested).

## Falsifiable predictions for future cycles

- **P1 (refuted by this work):** "Eliminating the 6 `v_lshrrev_b32` by
  using opsel-templated phase dispatch is VGPR-neutral." **FALSE** —
  it costs 29 extra VGPRs and 70 spills due to body duplication in
  inlined helpers.

- **P2 (next-cycle candidate):** A 2-iter (K-pair-granular) loop body
  restructure that processes phase<0> THEN phase<1> sequentially in a
  single body iter (no runtime branch) MAY avoid the body-duplication
  spill, since the compiler sees sequential not branched code.
  Estimated effort: 200+ lines of new loop body + tail-block rework.
  Risk: still spills if accumulator live ranges
  (cA/cB/cC/cD = 4 × RBM/16 × RBN/16 × float = 64 VGPRs) extend across
  both phases, which they MUST (cA accumulator is updated by both
  phase<0> and phase<1> within the same K-pair).

- **P3 (untested, lever #4 from R53 prompt):** Hybrid LDS+VGPR scale
  staging — DMA scales through LDS so the LDS swizzle does the
  byte-2/3 → byte-0/1 work the `v_lshrrev` currently does. Adds
  ~2 KB/block LDS pressure (crr_a/b_pack_count × 32 lanes × 4 bytes ×
  2 buffers), which fits within the 163840 B/CU budget after the
  139264 B baseline. Does NOT inflate body opsel template count.
  Falsifiable: build + measure.

- **P4 (strengthened):** The CRR ~92% structural ceiling identified in
  R48D / R47D extends to a STRONGER claim — ALL kernel-side levers that
  touch the scale-pack opsel encoding will pay either v_perm (R49A) or
  VGPR-spill (R53A) penalties because the 256-VGPR / 8-wave occupancy-2
  budget has zero slack for opsel-encoding code paths. The only viable
  next-cycle CRR levers are LDS-side (P3) or compiler-flag-side
  (force-no-inline pragma on `crr_mma_scaled_phase`, loop-pipeliner
  tuning) — both untested.

## Files

- `crr_mxfp8_exact_8wave_fastpath.inc` — added `MXFP8_CRR_SCALE_LAYOUT_V2`
  gate (default 0), `MXFP8_CRR_SCALE_LEAD` mutex check, and the LAYOUT_V2=1
  K-loop variant (runtime-branched `crr_mma_scaled_phase<0|1>` dispatch
  on `(k & 1)`, no scale-pack shifts on odd iters). LAYOUT_V2=0 arm is
  byte-identical to prior baseline (227 VGPR / 0 spill, validated).
- `r53a_bench.sh` — full strict-SCLK A/B harness (3 runs/arm × 3 cells ×
  30s cooldown × 60s rebuild). Committed for reproducibility; not executed
  (build evidence + quick A/B made full bench unnecessary per protocol).
- `r53a_results/SUMMARY.txt` — diagnostic summary with per-cell build
  resource A/B, ISA evidence, quick bench numbers, mechanism, and
  distinct-from-R49A/R50A explanation.
- `r53a_results/build_<cell>_{baseline,v2}.log` — full hipcc remarks per
  (cell, arm) pair (3 cells × 2 arms = 6 build logs).
- `r53a_results/build_70B_GateUp_v2_unroll2.log` — CRR_MAIN_UNROLL=2
  control showing spill is structural (not unroll-mediated).
- `r53a_isa_dumps/{baseline,v2}_70B_GateUp.s` — full device .s dumps for
  ISA-level instruction count comparison.

## Protocol note

Quick A/B (warmup=20, iters=50, 1 run/cell) used in lieu of full
5-runs/30s/60s bench because the build-side bail signal (70 VGPR spill,
56 NEW scratch_load/store inner-loop insns) was overwhelming. -65% mean
regression is ~30× larger than worst plausible SCLK slip (~2%). This
matches the R50A bail discipline (analytical refutation pre-bench when
build evidence is dispositive) and the R49B precedent (catastrophic spill
treated as falsification without strict-SCLK ratification). The kernel
patch is kept in tree (gated default-OFF) for future cycle audit.
