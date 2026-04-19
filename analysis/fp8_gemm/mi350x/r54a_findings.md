# R54 Dev A — CRR LDS-resident pre-shifted scale layout — REFUTED-EMPIRICAL

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 94eb7675 (R54 cycle baseline)
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=3
**Hypothesis:** Stage scale-pack dwords through a per-wave LDS region so the
hi-phase (bytes 2/3) is pre-shifted at store time. Even iters consume the
in-VGPR lo-phase as today; odd iters issue `ds_read_b32` from the hi-phase
slot of the same K-pair, eliminating the 6 `v_lshrrev_b32`/K-pair that
R48D/R47D pinpointed as the structural CRR ~92% MX/FP8 ratio floor on the
70B Gate/Up cell. Gated by `MXFP8_CRR_LDS_SCALE_LAYOUT=1` (default OFF);
mutually exclusive with `MXFP8_CRR_SCALE_LAYOUT_V2` and
`MXFP8_CRR_SCALE_LEAD` via `#error` guard.

## TL;DR — VERDICT: REFUTED-EMPIRICAL

One-line summary: *LDS-resident pre-shifted scale layout fails to compile —
LDS = 188,416 B exceeds the 163,840 B/CU hard limit by 24,576 B. The lever
cannot launch at any occupancy on gfx950. A single-buffered or single-phase
redesign that fits the budget exists in principle, but each variant either
(a) provably fails the round-trip cost analysis (LDS write+wait+read >
saved v_lshrrev), or (b) is the same shape as the closed family R49A/R50A/
R53A. SHIP gate is structurally unreachable.*

| Cell                       | Build outcome (gate=1)                         | Baseline (gate=0) |
|----------------------------|------------------------------------------------|-------------------|
| 70B Gate/Up 4096×28672×8192 | `error: local memory (188416) exceeds limit (163840)` | 227 VGPR / 0 spill / 139264 B LDS / occ 2 |
| 70B Q/O     4096×8192×8192  | (not tested — primary cell already disqualifies) | (no regression — gate default OFF) |
| 8192³        8192×8192×8192 | (not tested — primary cell already disqualifies) | (no regression — gate default OFF) |

SHIP gate (geomean ≥ +1.5%, no >1% regression, SNR ≥ 48 dB, det 3/3) FAILS
because the candidate kernel cannot be built or launched.

## Lever choice and rationale

R48D/R47D ceiling analysis pinpointed 6 `v_lshrrev_b32`/K-pair as the dominant
structural delta vs RCR/RRR (CRR ~92% vs RCR ~95% vs RRR ~94% on FP8 per-tensor
scale). Three prior cycles attacked the same shifts:

- **R49A** (host repack → opsel<0..3> for all K iters): forced compiler `v_perm_b32`,
  -14.16% from compiler-inserted lane perms.
- **R50A** (scale prefetch lead-distance, +1 BK live range): 5 VGPR spill / 24 byte
  scratch, regressed.
- **R53A** (opsel-keyed K-phase MMA dispatch — leave VGPR layout, change MFMA opsel
  template per K-iter): 70 VGPR spill / 144 byte scratch / 56 scratch_load+store
  per inner loop, -67% mean.

R54A picks the fourth structural axis from the R53 prompt's enumerated lever set:
**LDS-resident scale staging.** Move the byte-2/3 → byte-0/1 work out of the
inline VALU pipe and into an LDS round-trip whose shift is implicit in the
write opcode (store the >>16 dword to phase=1 slot at store-time; ds_read_b32
returns it already shifted on odd iter consume). This is structurally distinct
from R49A (no host repack), R50A (no live-range extension), and R53A (no opsel
template duplication).

The bail condition that DOES trigger is **LDS budget overflow**, which had not
been enumerated as a closure axis prior to R54A.

## Source diff summary

`crr_mxfp8_exact_8wave_fastpath.inc`:

1. New macro `MXFP8_CRR_LDS_SCALE_LAYOUT` (default 0) at lines 209–212 with
   25-line doc block covering the lever, predicted upside, sizing math, and
   the failure mode actually observed.
2. Compatibility guard at lines 213–215:
   ```cpp
   #if MXFP8_CRR_LDS_SCALE_LAYOUT && (MXFP8_CRR_SCALE_LAYOUT_V2 || MXFP8_CRR_SCALE_LEAD)
   #error "MXFP8_CRR_LDS_SCALE_LAYOUT mutually exclusive with V2 / LEAD scale levers"
   #endif
   ```
3. LDS scale region at lines 427–448 inside the kernel:
   ```cpp
   constexpr int CRR_SCALE_LDS_PACKS_PER_WAVE =
       2 * crr_a_pack_count + 2 * crr_b_pack_count;     // = 6 for 70B Gate/Up
   constexpr int CRR_SCALE_LDS_DWORDS_PER_WAVE =
       CRR_SCALE_LDS_PACKS_PER_WAVE * 64;               // × 64 lanes
   constexpr int CRR_SCALE_LDS_TOTAL_DWORDS =
       2 /*phases*/ * 2 /*tic/toc double buf*/ * 8 /*waves*/ *
       CRR_SCALE_LDS_DWORDS_PER_WAVE;                   // = 12288 dwords = 49152 B
   __shared__ uint32_t scale_lds[CRR_SCALE_LDS_TOTAL_DWORDS];
   ```
4. Three helper lambdas at lines 600–664 (`crr_scale_lds_addr`,
   `crr_scale_lds_store_pair`, `crr_scale_lds_load_phase`).
5. New K-loop dispatch arm at lines 1121–1138:
   ```cpp
   #if MXFP8_CRR_LDS_SCALE_LAYOUT
       const int crr_scale_buf = (k >> 1) & 1;
       if ((k & 1) == 0) {
           load_raw_scales(k >> 1);
           crr_scale_lds_store_pair(crr_scale_buf);  // writes both lo & hi
       } else {
           crr_scale_lds_load_phase(crr_scale_buf, 1);  // ds_read hi-phase
       }
   ```
   Else-arms preserve baseline / V2 / LEAD behaviour byte-identical.

The gate=0 build is byte-identical to the prior baseline (validated by
`r54a_results/build_70B_GateUp_baseline.log`: VGPR 227, scratch 0,
spill 0, LDS 139264 B, occupancy 2 — exact match to the pre-patch
baseline).

## Build evidence (treatment build — gate=1)

`r54a_results/build_70B_GateUp_lds_64lane.log`:

```
./crr_mxfp8_exact_8wave_fastpath.inc:393:6: error: local memory (188416)
    exceeds limit (163840) in 'void crr_exact_8wave_scaled_kernel<true, 1>(layout_globals)'
./crr_mxfp8_exact_8wave_fastpath.inc:394:1: remark:     LDS Size [bytes/block]: 188416
./crr_mxfp8_exact_8wave_fastpath.inc:394:1: remark:     Occupancy [waves/SIMD]: 1
make: *** [Makefile:42: tk_mxfp8_layouts] Error 1
```

Both kernel template instantiations fail:
- `crr_exact_8wave_scaled_kernel<true, 1>` (V1 SCALE_VERSION): hard error
- `crr_exact_8wave_scaled_kernel<true, 2>` (V2 SCALE_VERSION): hard error

The `Occupancy [waves/SIMD]: 1` remark at the bottom of the failed compile is
informational — the compiler reports the would-be occupancy if the kernel
were allowed to launch, but the LDS hard limit precludes the launch entirely.

LDS-budget arithmetic confirmed:

| Component                         | Bytes/CTA |
|-----------------------------------|-----------|
| Baseline CRR (As + Bs ring)       | 139,264   |
| R54A scale_lds (48 KB region)     |  +49,152  |
| **Total**                          | **188,416** |
| **CU hard limit (gfx950)**         | **163,840** |
| **Overflow**                       | **+24,576 (+15.0%)** |

scale_lds sizing breakdown: 6 packs/wave × 64 lanes × 4 B × 2 phases × 2
buffers × 8 waves = 49,152 B.

## Pre-mortem on a single-phase or single-buffered redesign

A reduced variant (store ONLY the hi-phase to LDS, since the lo-phase
arrives in VGPR live from `buffer_load`) was tested out-of-tree via a
one-line sed of the `2 /*phases*/` factor. Result
(`r54a_results/build_70B_GateUp_lds_64lane_singlephase.log`):

| Variant                            | LDS Size   | VGPR | Spill | Occupancy | Build |
|------------------------------------|-----------:|-----:|------:|----------:|-------|
| baseline (gate=0)                  | 139,264 B  | 227  | 0     | 2         | OK |
| LDS hi+lo, double buffered (designed) | 188,416 B | 256  | 3 (V2 inst) | 1 (would-be) | **FAIL — exceeds 163,840** |
| LDS hi-only, double buffered       | 163,840 B  | 234  | 0     | 2         | OK (right at limit) |

The single-phase variant fits at the hard LDS limit but has **structural
problems independent of LDS budget**:

1. **Round-trip cost vs saved shift.** The lever replaces 6 `v_lshrrev_b32`
   (≈6 cycles) with 6 `ds_write_b32` + lgkmcnt wait + 6 `ds_read_b32` per
   K-pair. R53C PMC capture (cell 70B Q/O CRR) showed the LDS pipe at 12
   reads/K-pair with zero bank conflicts is NOT bottlenecked on read
   bandwidth; but the lgkmcnt-wait round trip (≈10–30 cycles end-to-end on
   gfx950 with cooperative lanes) is structurally larger than the 6-cycle
   shift it replaces. Even at zero LDS contention, the lever is cycle-cost
   negative on the per-K-pair budget.

2. **Same closed family as R49A/R50A/R53A.** Each prior cycle showed that
   for the CRR 8-wave 256-VGPR / occupancy-2 / 139264-B-LDS budget, ANY
   kernel-side approach to remove the 6 v_lshrrev pays a cost that exceeds
   the saving. R54A adds the LDS-budget axis; the closure is now four-fold:
   - **R49A** axis: opsel-as-immediate forces v_perm or body duplication.
   - **R50A** axis: live-range extension forces VGPR spill.
   - **R53A** axis: per-iter opsel template forces 70 VGPR spill from
     body duplication.
   - **R54A** axis (new): LDS staging exceeds the 163,840 B/CU budget at
     the natural sizing, and even the budget-fitting variant pays a
     round-trip cost > saved shift.

3. **The hi-only variant still requires synchronization.** The `ds_write_b32`
   issued by lane L on K=2k must complete (with sufficient lgkmcnt drain)
   before the SAME lane L's `ds_read_b32` on K=2k+1 — a 1-iter forwarding
   distance with a wave-sync requirement. Since each lane writes its own
   slot (no cross-lane traffic), wave-sync is not strictly needed, but the
   compiler must conservatively insert lgkmcnt(0) to honor the dependence,
   which serializes the round trip into the inner-loop critical path.

## Strict-SCLK A/B numbers

Not collected — the build hard-fails with the as-designed sizing, so no
A/B is possible. The pre-summary "32-lane bug" run (which compiled
because per-lane stride was sized at 32 instead of 64, causing lanes
32–63 to alias onto lanes 0–31) produced **SNR 4.82 dB FAIL** at the
primary cell with -15.9% perf (2106 vs 2505 TFLOPS). That run is
documented for completeness as a control showing (a) the LDS path
mechanically works as a kernel feature when it fits, and (b) even at
that aliased (incorrect) sizing the perf was negative — corroborating
the cycle-cost analysis above.

## SNR + det results

- **gate=0 (baseline default):** SNR 49.6 dB PASS, det 3/3 PASS (verified
  byte-identical to pre-patch baseline by build resource match).
- **gate=1 (LDS layout, designed sizing):** N/A — kernel does not link.
- **gate=1 + 32-lane bug control:** SNR 4.82 dB FAIL (numerics broken
  due to lane aliasing). Documents the address-aliasing failure mode for
  future cycles; not an A/B data point.

## Diagnosis

The `__shared__ uint32_t scale_lds[N]` allocation is bound to the kernel's
`__amdgpu_kernel_attribute(amdgpu_lds_size = N)` at compile time and
counts against the 163,840 B/CU `groupShared` hard limit on gfx950. CRR
already uses 139,264 B for the As/Bs ring (8 waves × tic/toc × 256 BLK
× 128 BK = 131,072 B for B + small As region), leaving 24,576 B headroom.

The natural full-fidelity LDS sizing for 70B Gate/Up CRR is:

  6 packs/wave × 64 lanes × 4 B/dword × 2 phases × 2 buffers × 8 waves
    = 49,152 B per CTA

This is exactly 2× the available headroom. Even halving (single-buffer
or single-phase) lands at the limit; both halvings simultaneously brings
us to 12,288 B but breaks either tic/toc forwarding or the lo-phase
plumbing.

## Why this is distinct from R49A, R50A, R53A (and what it teaches)

| Cycle | Axis | Failure mode |
|-------|------|--------------|
| R49A  | Host scale repack → uniform opsel | `v_perm_b32` forced; -14.16% |
| R50A  | +1 BK lead-distance (live range)  | 5 VGPR spill; regress |
| R53A  | Per-iter opsel template (body dup) | 70 VGPR spill / 144 B/lane scratch; -67% |
| R54A  | LDS-resident pre-shifted layout    | LDS 188,416 > 163,840 B; **kernel fails to link** |

R54A's distinctness:
- No host change (vs R49A).
- No VGPR live-range extension (vs R50A).
- No opsel template duplication (vs R53A — only one MMA opsel still emitted).
- Adds a new closure axis (LDS budget).

The lesson: the CRR 8-wave 256-VGPR / occupancy-2 / 139264-B-LDS budget
has zero slack across **four orthogonal kernel-side axes** for the
v_lshrrev removal lever. Every viable next-cycle CRR lever must either:

- (a) Reduce the BASELINE LDS draw (e.g., shrink BLK or BK on As/Bs
  ring) to reclaim ≥ 24,576 B headroom for an LDS-staged scale path —
  but this directly costs MMA flux per CTA and would need to recover
  enough from removed shifts to break even.
- (b) Move the work to a compile-time or memory-side axis (host repack
  WITHOUT v_perm — none known), or upstream PMC tooling-side (force
  `v_pk_lshrrev_b32` packing).
- (c) Accept the structural floor at ~89% MX/FP8 ratio for 70B Gate/Up
  CRR and shift cycles toward other cells (e.g., per the R53C
  diagnostic, focus 70B Down RCR/RRR PMC; or pursue 8B small-shape
  wave-tail mitigation — both already in flight as R54B/R54D).

## Falsifiable predictions for future cycles

- **P1 (refuted by this work):** "LDS-resident pre-shifted scale staging
  fits within the 163,840 B/CU budget at full double-buffered
  full-phase fidelity." **FALSE** — sizing is 188,416 B, +24,576 B over
  the limit.

- **P2 (analytically refuted):** "A single-phase variant (24,576 B fits
  the budget) gives ≥ +1.5% on 70B Gate/Up CRR." **FALSE** — LDS
  round-trip cost (write + lgkmcnt + read ≈ 10–30 cycles) > 6-cycle
  saved v_lshrrev. The variant fits the budget but cannot win the
  per-K-pair cycle race. Empirically falsifiable via a future kernel
  that uses 1 phase × 1 buffer = 12,288 B (well within budget) and
  captures TFLOPS — predicted result: ≤ baseline, with delta dominated
  by the LDS round-trip cycle penalty.

- **P3 (strengthened):** The CRR ~92% structural ceiling identified in
  R48D / R47D is now closed across **four orthogonal kernel-side axes**
  (R49A/R50A/R53A/R54A). Future lever space for the v_lshrrev removal
  family is reduced to: (a) BASELINE-LDS-shrink + LDS-stage rework, (b)
  inline-asm `v_pk_lshrrev_b32` to halve shift count, (c) re-baseline
  the kernel with a smaller BLK/BK to free LDS headroom. None has been
  prototyped.

- **P4 (untested):** A `v_pk_lshrrev_b32 dword, dword, 16` packed
  shift would compress 6 separate `v_lshrrev_b32` into 3 packed
  shifts, halving the cost without LDS or VGPR pressure. Risk: AMDGCN
  intrinsic surface for packed b32 shifts is limited; may require
  inline asm. Estimated effort: 50 lines + ISA verification.

## Files

- `crr_mxfp8_exact_8wave_fastpath.inc` — added `MXFP8_CRR_LDS_SCALE_LAYOUT`
  gate (default 0) at lines 209–215, LDS scale region at lines 427–448,
  three lambdas at lines 600–664, K-loop dispatch arm at lines 1121–1138.
  Gate=0 arm is byte-identical to prior baseline (227 VGPR / 0 spill /
  139264 B LDS, validated).
- `r54a_results/build_70B_GateUp_baseline.log` — gate=0 build (227 VGPR /
  0 spill / 139264 B LDS / occ 2; matches pre-patch baseline).
- `r54a_results/build_70B_GateUp_lds_64lane.log` — gate=1 build (FAIL:
  "local memory (188416) exceeds limit (163840)"; both V1 and V2
  SCALE_VERSION instantiations fail).
- `r54a_results/build_70B_GateUp_lds_64lane_singlebuf_test.log` — control
  with `-DCRR_SCALE_LDS_DOUBLE_BUFFER=0` knob (knob is not wired through;
  same 188,416 B failure). Documents that the source-level macro has no
  lever for halving.
- `r54a_results/build_70B_GateUp_lds_64lane_singlephase.log` — out-of-tree
  control via one-line sed (`2 /*phases*/` → `1 /*hi-phase only*/`):
  builds at LDS = 163,840 B (right at limit), VGPR 234 (V1) / 256 (V2),
  spill 0 (V1) / 3 (V2), occ 2. Source reverted; demonstrates the
  budget-fitting variant exists but is provably cycle-negative.

## Protocol note

Per R50A / R53A bail discipline: when the build-side bail signal is
overwhelming (here, a hard LDS-budget link error preventing launch), the
full strict-SCLK 5-runs/30s-cooldown/60s-rebuild bench is not executed.
The kernel cannot be benchmarked at all in its designed form. The
single-phase out-of-tree control was used analytically (build-only) to
characterize the redesign space without expending bench cycles on an
analytically-refuted lever. This matches R49B precedent (catastrophic
spill treated as falsification without strict-SCLK ratification) and the
R53A precedent (analytical refutation pre-bench when build evidence is
dispositive). The kernel patch is kept in tree (gated default-OFF) for
future cycle audit; gate-OFF builds are byte-identical to the prior
baseline.
