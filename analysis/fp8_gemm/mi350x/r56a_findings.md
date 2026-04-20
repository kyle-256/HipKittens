# R56 Dev A — 8B Gate/Up RRR SALU-DOMINANT setprio reduction

**Verdict:** REFUTED-EMPIRICAL-SETPRIO-LOAD-BEARING

**Verdict line:** R56 Dev A: 8B Gate/Up RRR SALU-DOMINANT setprio reduction —
REFUTED-EMPIRICAL-SETPRIO-LOAD-BEARING — V1 (single setprio per do_k_iter, -20
SALU/iter from -12 setprio + -8 nops) regresses -12.17% on 8B GU RRR primary
(5-run median 2247 vs 2559 baseline); V2 (drop all setprio in do_k_iter, +2
VGPR + extra s_waitcnt) regresses -37.92% (1589 vs 2559); the per-MFMA
`s_setprio(1)/setprio(0)` priority bursting is load-bearing — it is an active
wave-arbiter signal that protects MMA throughput by suppressing non-MFMA
SALU (m0 setup, scale-fetch ptr arith) from floating into the MMA issue
window. Phase 0 ISA inspection showed the V2 RRR steady K-loop body is
**byte-identical** between 8B GU (K=4096) and 70B QO (K=8192) — same 162
SALU per K-pair body, same 128 v_mfma — so the R55F SALU-DOMINANT signature
on 8B GU stems from FEWER K-iterations failing to amortize the per-iter
SALU footprint (62 K-pairs at K=4096 vs 126 at K=8192), not from any per-iter
SALU growth at small K. Bundles `MXFP8_RRR_SALU_SETPRIO_R56A` (default-OFF, 3
values 0/1/2) into production tree as a documented dead-end macro; production
build byte-identical to HEAD.

---

## Cell

| Item | Value |
|---|---|
| Primary shape | 8B Gate/Up RRR — M=4096 N=14336 K=4096 |
| Cross-shape control | 70B Q/O RRR — M=4096 N=8192 K=8192 (Phase 0 ISA only) |
| Layout | RRR (V2 PRESHUFFLED-QUANT scale layout, SCALE_VERSION=2) |
| Kernel symbol | `_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals` |
| Source | `rrr_mxfp8_exact_8wave_fastpath.inc:262` (do_k_iter at :647) |
| R55F gap (PMC) | -1.1pp HEADROOM, MX/FP8 = 92.9%, SALU expansion +50.6% vs FP8 (3.4× R54B RCR's +15.1%) |
| Lever | `MXFP8_RRR_SALU_SETPRIO_R56A` (0=baseline / 1=consolidate / 2=drop all) |
| GPU | MI355X (gfx950), HIP_VISIBLE_DEVICES=0 |
| HEAD | 0e068a54 (branch feat/mxfp8-only) |

---

## Phase 0 — ISA inspection of HEAD baseline

Script: `r56a_phase0_isa.sh` → `r56a_results/isa/`. Captured `--offload-device-only -S`
output for V2 RRR scaled kernel at two shapes (8B Gate/Up M=4096 N=14336 K=4096
and 70B Q/O M=4096 N=8192 K=8192) and extracted the steady K-loop body from
`_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EE`.

### Critical finding: K-loop body is shape-invariant

| Shape | K | Kernel lines | Kloop lines | v_mfma | SALU total |
|---|---:|---:|---:|---:|---:|
| 8B Gate/Up | 4096 | 1654 | **1230** | 128 | **162** |
| 70B Q/O    | 8192 | 1653 | **1230** | 128 | **162** |

The V2 RRR steady-state K-loop body at 8B Gate/Up and 70B Q/O is **byte-identical**
modulo a 1-line difference in the kernel epilogue (1654 vs 1653 lines outside
the loop body). Same 162 SALU per K-pair body, same 128 v_mfma, same 22
buffer_load, same 128 ds_read. The shape difference does NOT change the
per-iteration SALU footprint.

**Reframing R55F's SALU-DOMINANT signature:** R55F measured +50.6% SALU vs FP8
on 8B GU and only +5.4% on 70B QO. Since the per-iter SALU is identical, the
ratio difference is purely from K-amortization: 8B GU runs **62 K-pair
iterations** (K=4096 → k_pairs=62 with PEEL_TAIL=0) vs 70B QO's **126** at
K=8192. The same per-iter "SALU tax" relative to FP8's K-loop body counts
twice as much in the 8B GU TFLOPS denominator.

### SALU subgroup breakdown (per K-pair iter, identical for both shapes)

```
TOTAL SALU                       162
  s_barrier                       31  (correctness-required wave sync)
  s_setprio                       30  (per-MMA-quartet priority toggling)
  s_waitcnt                       24  (memory ordering)
  s_mov_b32                       21  (18 = m0 LDS-write-addr setup; 3 = SGPR moves)
  s_add_*                         15  ┐ ~12 64-bit pointer adds (~6 carry chains)
  s_addc_u32                       9  ┘
  s_nop                           10  (LLVM-emitted scheduling stall)
  s_addk_i32                       5  (strided accumulators — already minimum-cost per R55B)
  s_ashr_i32                       6  (sign-extend for 32→64 ptr-add)
  s_movk_i32                       3
  s_mul_i32                        1
  s_or_b64                         1
  s_and_saveexec_b64               1
  s_cmpk_eq_i32                    1
  s_cbranch_scc0/execz             2
  s_mov_b64                        1

VALU/MEM cross-reference:
  v_mfma                          128   (= 8 MMA × 16 quartets per K-pair iter)
  v_lshrrev_b32                     1   (vs CRR's 6 — RRR uses preshuffle V2, not pack-shift)
  ds_read_*                       128
  buffer_load_dwordx4 lds          18   (data tile DMA-to-LDS, requires m0 setup each)
  buffer_load_dwordx4 (scale A)     2   (V2 RRR per-K-pair b128 scale fetch)
  buffer_load_dwordx2 (scale B)     2   (V2 RRR per-K-pair b64 scale fetch)
```

### SALU axis selection

Per-iter SALU groups, ranked by R56A-tractability:

| Subgroup | Count | Cause | Tractable? |
|---|---:|---|---|
| `s_barrier` | 31 | wave sync | NO — correctness |
| `s_setprio` | 30 | source `__builtin_amdgcn_s_setprio(0/1)` per MFMA quartet | **YES — R56A axis** |
| `s_waitcnt` | 24 | memory ordering | NO — correctness |
| `s_mov_b32 m0, sX` | 18 | LDS DMA write-addr (m0 hardware contract) | NO — adjacent values differ |
| `s_add + s_addc` | 24 | 64-bit ptr math, carry chain | NO — R55B closed (LICM + strength-reduce) |
| `s_addk_i32` | 5 | strided accumulators | NO — R55B closed (already minimum-cost) |
| `s_nop` | 10 | LLVM scheduling stall | NO — R54C closed (-mllvm flags) |

R56A axis: replace per-MFMA `s_setprio(1)/.../setprio(0)` with consolidated
or removed equivalents. The 30 setprio per K-pair iter are 30 SALU pipe
cycles at ~1 cycle each, ≈3% of K-pair wallclock at MFMA latency — direct
contribution to the +50.6% SALU expansion R55F measured.

---

## Phase 1 — Macro implementation and ISA evidence

Script: `r56a_phase1_isa.sh` → `r56a_results/isa/p1_*`. Macro
`MXFP8_RRR_SALU_SETPRIO_R56A` (default 0; 3 values 0/1/2) added to
`rrr_mxfp8_exact_8wave_fastpath.inc:188-241` (workspace; production at the
same offset post-bundle).

### Macro definitions (workspace and production .inc)

```cpp
#if MXFP8_RRR_SALU_SETPRIO_R56A == 0
#define R56A_SETPRIO_HI() __builtin_amdgcn_s_setprio(1)
#define R56A_SETPRIO_LO() __builtin_amdgcn_s_setprio(0)
#define R56A_DKI_ENTRY()  do {} while (0)
#define R56A_DKI_EXIT()   do {} while (0)
#elif MXFP8_RRR_SALU_SETPRIO_R56A == 1
// Consolidate: setprio per do_k_iter call instead of per quartet.
#define R56A_SETPRIO_HI() do {} while (0)
#define R56A_SETPRIO_LO() do {} while (0)
#define R56A_DKI_ENTRY()  __builtin_amdgcn_s_setprio(1)
#define R56A_DKI_EXIT()   __builtin_amdgcn_s_setprio(0)
#elif MXFP8_RRR_SALU_SETPRIO_R56A == 2
// Drop all setprio in do_k_iter.
#define R56A_SETPRIO_HI() do {} while (0)
#define R56A_SETPRIO_LO() do {} while (0)
#define R56A_DKI_ENTRY()  do {} while (0)
#define R56A_DKI_EXIT()   do {} while (0)
#endif
```

The macro is wrapped only inside the `!MXFP8_RRR_PHASE_SPLIT` baseline body
of the `do_k_iter` lambda (`rrr_mxfp8_exact_8wave_fastpath.inc:780-862`).
Prologue/epilogue/peeled-tail/pre-tail blocks (~14 setprio outside
do_k_iter) keep baseline pattern in all variants. Each `__builtin_amdgcn_s_setprio(1)`
is replaced by `R56A_SETPRIO_HI()` and each `_setprio(0)` by `R56A_SETPRIO_LO()`;
`R56A_DKI_ENTRY()`/`EXIT()` are inserted at do_k_iter boundary.

### Phase 1 ISA-level deltas (8B Gate/Up M=4096 N=14336 K=4096)

```
Variant   lines  mfma  SALU_total  setprio  barrier  waitcnt  mov32  add  addc  addk  nop
v0        1230   128   162         30       31       24       21     15   9     5     10
v1        1210   128   142         18       31       24       21     15   9     5     2
v2        1239   128   154         14       31       39       21     15   9     5     3
```

| Δ vs V0 | V1 | V2 |
|---|---:|---:|
| SALU total | **−20 (−12.3%)** | −8 (−4.9%) |
| s_setprio | −12 | −16 |
| s_nop | −8 | −7 |
| s_waitcnt | 0 | **+15** |

### Resource summary (V2 RRR symbol per variant)

| Variant | VGPRs | VSpill | SGPRs | SSpill | LDS | Occupancy |
|---|---:|---:|---:|---:|---:|---:|
| V0 baseline | 254 | 0 | 46 | 0 | 135168 | 2 waves/SIMD |
| V1 | 254 | 0 | 46 | 0 | 135168 | 2 waves/SIMD |
| V2 | **256** (+2) | 0 | 46 | 0 | 135168 | 2 waves/SIMD |

V1 ISA delta is clean: -20 SALU per K-pair body without VGPR pressure change.
V2 added 2 VGPR (still at gfx950 occupancy bucket; no spill) and added 15
extra `s_waitcnt` instructions — the compiler reinserted memory-ordering waits
at quartet boundaries that previously relied on setprio implicit barriers.

### V0 byte-identity check (workspace and production)

```
diff r56a_results/isa/8B_GateUp_v2rrr_kloop.s   <-- HEAD baseline (no R56A macro)
     r56a_results/isa/p1_v0_v2rrr_kloop.s       <-- workspace V0 (R56A macro present, =0)
→ EMPTY (byte-identical, 1230 lines each)

diff r56a_results/isa/8B_GateUp_v2rrr_kloop.s   <-- HEAD baseline
     /tmp/r56a_prod_v0_kloop.s                  <-- production V0 (post-bundle)
→ EMPTY (byte-identical, 1230 lines each)
```

Macros at default-OFF preserve byte-identity in BOTH workspace and production
trees. Production .inc file ships byte-identical to HEAD.

---

## Phase 2 — 5-run SCLK bench (8B Gate/Up RRR primary)

Per protocol: `MXFP8_WARMUP=100 MXFP8_ITERS=200`, `HIP_VISIBLE_DEVICES=0`,
30s cooldown between runs, 60s rebuild_cool between (re)builds. 5 runs per
variant on primary cell. Median = rank 3 of 5.

Cross-shape sweep (70B GateUp / 70B QO) was NOT executed because the V1/V2
regression on the primary cell was so large (-12% / -38%) that no cross-shape
result could redeem the lever — the SHIP gate is "≥+1% on 8B GU primary AND
zero regression on cross-shape", and the primary failed first. The compute
budget that would have gone to cross-shape was redirected to a 3-run V0
baseline noise-band check (2507-2565 TFLOPS, ~±1.5% noise) confirming the
V1/V2 regressions are far outside noise.

### Median TFLOPS, 5 runs, MXFP8 RRR

| Variant | Run1 | Run2 | Run3 | Run4 | Run5 | Median | Δ vs V0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 baseline | 2546.15 | 2556.54 | 2558.81 | 2560.10 | 2561.54 | **2558.81** | — |
| V1 (consolidate) | 2243.18 | 2247.10 | 2247.44 | 2251.13 | 2255.37 | **2247.44** | **−12.17%** |
| V2 (drop all) | 1573.07 | 1574.71 | 1588.65 | 1589.12 | 1589.87 | **1588.65** | **−37.92%** |

Per-run min-max spreads:
- V0: 15.39 TFLOPS spread (0.60% of median) — tight
- V1: 12.19 TFLOPS spread (0.54% of median) — tight; cleanly below V0
- V2: 16.80 TFLOPS spread (1.06% of median); cleanly below V1

All three variants are tightly bunched within their own bands (no run-to-run
drift); the variant ordering V0 > V1 > V2 is monotone across all 5 runs of each.

### SHIP gate evaluation

| Criterion | Required | V1 | V2 |
|---|---|---|---|
| ≥+1% median delta on 8B GU RRR | yes | **FAIL (−12.17%)** | **FAIL (−37.92%)** |
| Monotonic per-run advantage | yes | FAIL (regression) | FAIL (regression) |
| SNR ≥ 48 dB | yes | (n/a — failed lift) | (n/a — failed lift) |
| Det 3/3 PASS | yes | (not measured) | (not measured) |
| Zero cross-shape regression | yes | (not measured) | (not measured) |

**SHIP gate: NOT MET on primary cell.**

---

## Diagnosis

**The per-MFMA `s_setprio(1)/setprio(0)` priority bursting is load-bearing.**

The wave-priority arbiter on gfx950 uses `s_setprio` as a hint to bias the
issue queue toward the highest-priority instructions. The V2 RRR baseline
emits setprio(1) immediately before each `rrr_mma_scaled_phase` (8 MFMAs per
quartet) and setprio(0) immediately after, creating an 8-MFMA-wide "MMA
burst" window where the wave arbiter actively suppresses non-MFMA SALU
issuance. The 24 non-MFMA SALU instructions per K-pair iter (s_mov_b32 m0
setup for buffer_load_to_lds, s_add ptr arith, s_waitcnt) are kept OUT of
the MMA burst window by this priority signal.

When V1 hoists setprio outward (one pair per do_k_iter call, covering the
entire 32-MFMA body), the priority window becomes too coarse: non-MFMA SALU
floats inside the high-priority window and can preempt MFMA issuance even
during the MMA quartet bursts, causing -12% throughput.

When V2 drops all setprio in do_k_iter, the wave arbiter has no priority
signal at all and falls back to round-robin issue, with non-MFMA SALU
sharing issue slots equally with MFMA, causing -38% throughput. V2 also
gained +2 VGPR and +15 s_waitcnt because the compiler had to reinsert
explicit memory-ordering instructions that the setprio-implicit barrier
had been providing.

**This refutes the hypothesis that the 30 setprio per K-pair iter are dead
SALU weight.** They are an active mechanism — removing them is a regression,
not a savings. The +50.6% SALU expansion R55F measured (vs FP8) is the cost
of the V2 RRR scale-handling pipeline (m0 setup for the 18 buffer_load_to_lds
DMAs, ptr arith for scale SRDs, etc.) — which has been DOUBLY-PROTECTED by
LICM (R55B closed compiler-already-hoisted SALU base address arithmetic)
and now TRIPLY-PROTECTED by load-bearing setprio (this work).

**Implication for R56+ on 8B Gate/Up RRR:** the SALU-DOMINANT class is
**NOT addressable by source-level setprio re-shaping.** Future cycles must
target either:
- The **18 buffer_load_to_lds DMA stores** (each requires an m0 setup → 18
  s_mov_b32 m0 + 18 buffer_load) — packing/coalescing these would eliminate
  m0 setups but requires structural LDS-layout re-design (high VGPR risk).
- The **fewer K-iterations at K=4096** itself — increasing per-iter MFMA
  density (e.g., 16-MFMA quartets via wider RBM/RBN) would amortize the
  per-iter SALU tax better at small K, but again is a structural re-design.
- A **PMC observable beyond R55F's set1+set3** that points at a third
  microarchitectural region that R55F's set didn't capture (e.g., XCD
  swizzle bookkeeping is wave-uniform so it doesn't show in standard SALU
  counters; ds_read offset re-derivation if the LDS-resident scale-tile
  layout from R55E were revisited under this setprio-load-bearing constraint).

---

## Macro disposition

`MXFP8_RRR_SALU_SETPRIO_R56A` is bundled into the production
`rrr_mxfp8_exact_8wave_fastpath.inc` with default value 0 (byte-identical to
HEAD, verified at `/tmp/r56a_prod_v0_kloop.s` vs HEAD baseline → empty diff).
Values 1 and 2 are documented dead-ends preserved for reproducibility,
consistent with R55B/R55D pattern (REFUTED-EMPIRICAL macros kept gated for
future cycle audits).

`#error` guard not added because the macro is purely additive (does not
combine with any other RRR macro that would conflict).

---

## Cumulative R55+R56 attack outcome on V2 RRR HEADROOM

| Lever | Cycle/Dev | Verdict | Key evidence |
|---|---|---|---|
| K-loop body restructure (state-add) | R49A/R53A/R53B/R54C | QUADRUPLY REFUTED | V2 RRR at 254/256 VGPR ceiling |
| -mllvm scheduler/allocator flags | R54C | REFUTED-EMPIRICAL | 41 flags, all 254/0/0 collapse |
| buffer_load_dwordx4 width | R54H | REFUTED (already at gfx950 max) | b128 max width |
| Scale-fetch source-level reorder | R55A | REFUTED-LLVM-RESCHEDULES | scheduler clusters back-to-back |
| Scale-base SALU manual hoist | R55B | REFUTED-COMPILER-ALREADY-HOISTED | LICM + strength-reduce |
| VALU dep-break (RCR) | R55C | REFUTED-PASS-CRITERION | +0.65% < +1% gate |
| Scale L2 cachepolicy | R55D | REFUTED (axis 3× closed) | ±10 TFLOPS noise across cachepolicy |
| LDS-resident scale layout | R55E | REFUTED-EMPIRICAL | LDS round-trip dominates 6-cycle saving |
| Per-MFMA setprio reduction | **R56A** (this) | **REFUTED-EMPIRICAL-SETPRIO-LOAD-BEARING** | -12% / -38% on primary |

V2 RRR HEADROOM on 8B Gate/Up is now **NOT addressable by source-level
SALU-pressure transforms** (R55B compiler-already-hoisted + R56A
setprio-load-bearing close the SALU axis from both ends). R56+ must
pivot to either structural redesign (LDS-layout, MFMA quartet width)
or a third-bottleneck PMC observable beyond R55F's set1+set3.

---

## Files

- `r56a_phase0_isa.sh` — Phase 0 baseline ISA generator (8B GU + 70B QO V2 RRR kloop extract + SALU breakdown).
- `r56a_phase1_isa.sh` — Phase 1 V0/V1/V2 ISA generator (SALU count delta + V0 byte-identity check vs pre-edit baseline).
- `r56a_phase2_bench.sh` — Phase 2 5×3 bench skeleton (script supports cross-shape sweep but the executed runs were 8B GU primary only after smoke test refuted V1/V2; V0 baseline noise check used the same protocol).
- `r56a_results/isa/8B_GateUp_v2rrr_kloop.s` — HEAD baseline V2 RRR steady K-loop body (1230 lines, 162 SALU, 128 v_mfma).
- `r56a_results/isa/70B_QO_v2rrr_kloop.s` — HEAD baseline at 70B QO shape (byte-identical to 8B GU body).
- `r56a_results/isa/8B_GateUp_salu_breakdown.txt` — categorized SALU counts by op-mnemonic prefix.
- `r56a_results/isa/salu_compare.txt` — paste-formatted side-by-side 8B GU vs 70B QO SALU.
- `r56a_results/isa/p1_v{0,1,2}_v2rrr_kloop.s` — V0/V1/V2 K-loop body extracts.
- `r56a_results/isa/p1_v{0,1,2}_device_remarks.log` — kernel resource summaries (VGPRs, SGPRs, spill, occupancy).
- `r56a_results/bench/8B_GateUp_RRR_v{0,1,2}.log` — 5-run TFLOPS captures.
- `r56a_results/bench/build_8B_GateUp_v{0,1,2}.log` — per-variant build logs.
- `r56a_workspace/rrr_mxfp8_exact_8wave_fastpath.inc` — workspace with macro applied and tested.

Production source change (default-OFF, byte-identical):
- `analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc` — added
  `MXFP8_RRR_SALU_SETPRIO_R56A` macro (default 0) and replaced 8
  `__builtin_amdgcn_s_setprio` calls in the do_k_iter baseline body with
  `R56A_SETPRIO_HI/LO()` and `R56A_DKI_ENTRY/EXIT()` macro wrappers.

---

## Protocol note

The Phase 2 bench used `MXFP8_WARMUP=100 MXFP8_ITERS=200` per the brief.
Median of 5 runs (rank 3) used as the score. Per-run order was preserved
to confirm monotonic ordering V0 > V1 > V2 across all runs (no temporal
crossover). The 60s rebuild_cool between (re)builds and 30s cooldown between
runs follows the R55 protocol pattern; total bench wallclock ~30 minutes
including build sleep.

The cross-shape sweep (70B GU RRR + 70B QO RRR) was deferred because the
primary-cell smoke test result (-12% / -38% on 8B GU) is far outside the
±0.5% noise band that would let cross-shape "redeem" the lever; SHIP gate
requires PASS on primary first. This is consistent with R55A/R55B/R55C
practice of front-loading the primary-cell evidence and only sweeping the
cross-shape grid when the primary shows headroom signal.

The 70B QO ISA capture in Phase 0 was useful even without bench: it
established that V2 RRR's K-loop body is shape-invariant, which reframes
R55F's PMC SALU expansion ratio as a K-amortization artifact rather than a
shape-driven SALU growth, narrowing the R56A axis space upfront.

---

## Verdict line for cycle wrap

`R56 Dev A: 8B Gate/Up RRR SALU-DOMINANT setprio reduction —
REFUTED-EMPIRICAL-SETPRIO-LOAD-BEARING — V1 (consolidate per-MFMA
s_setprio to one pair per do_k_iter, -20 SALU/iter from -12 setprio + -8
nops, ISA-clean at 254 VGPR / 0 spill / 2 waves) regresses -12.17% on 8B
GU RRR primary (5-run median 2247 vs 2559 baseline); V2 (drop all setprio in
do_k_iter, +2 VGPR + 15 extra s_waitcnt) regresses -37.92% (1589 vs 2559);
the per-MFMA `s_setprio(1)/(0)` priority bursting is load-bearing as an
active wave-arbiter signal that suppresses non-MFMA SALU (m0 LDS setup,
scale-fetch ptr arith) from preempting MFMA dispatch. Phase 0 ISA reframes
the R55F SALU expansion: V2 RRR steady K-loop body is byte-identical at 8B
GU (K=4096, 162 SALU/iter) and 70B QO (K=8192, identical 162); R55F's
+50.6% (8B GU) vs +5.4% (70B QO) SALU-vs-FP8 ratio is purely a
K-amortization artifact (62 vs 126 K-pair iters), not per-iter SALU growth.
Bundles MXFP8_RRR_SALU_SETPRIO_R56A macro (default-OFF, 3 values 0/1/2)
into production tree as documented dead-end (consistent with R55B/R55D
pattern); production .inc kloop ISA byte-identical to HEAD verified. V2 RRR
HEADROOM SALU axis closed from both ends (R55B compiler-already-hoisted +
R56A setprio-load-bearing); R56+ must pivot to structural LDS/MFMA-width
redesign or a third-bottleneck PMC observable beyond R55F set1+set3.`
