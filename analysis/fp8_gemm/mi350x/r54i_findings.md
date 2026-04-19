# R54 Dev I — 70B Gate/Up CRR branchless unconditional shift block — REFUTED-DIAGNOSTIC

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 3ac0fd0e (R54 in flight)
**GPU:** MI355X (gfx950) — `HIP_VISIBLE_DEVICES=0`
**Mandate:** Attack 70B Gate/Up CRR (M=4096 N=28672 K=8192) currently at 89.3%
MX/FP8 (HEADROOM −5.7pp) via a branchless unconditional scale-shift block.
R53C diagnosed the `s_bitcmp0_b32 / 6× v_lshrrev_b32 / s_cbranch_scc1`
conditional shift block on alternate K-pairs as the *compiler-unroll blocker*
— making this unconditional could expose MFMA-load overlap that the
conditional currently prevents.

## TL;DR — VERDICT: REFUTED-DIAGNOSTIC — control-flow change SUCCEEDS, but compiler still refuses to unroll body

**The branchless rewrite (option a — always-shift) successfully eliminates the
`s_bitcmp0_b32 / s_cbranch_scc1` shift conditional from the K-loop body and
reschedules the 6 v_lshrrev_b32 inline with the MFMA / buffer_load region.
But the K-loop body remains 1 K-pair / 32 MFMA per iter — the compiler still
refuses to unroll across the surviving LOAD conditional.** The structural
"compiler-unroll blocker" R53C identified is therefore not the shift
conditional alone — it is the LOAD conditional that R53C did not separately
attribute. Removing only the shift conditional is necessary but not sufficient
for body unrolling.

| Metric | Baseline (gate=0) | Branchless (gate=1) | Δ |
|---|---:|---:|---:|
| 70B Gate/Up CRR median TFLOPS (n=5) | **2529.21** | **2526.51** | **−2.70 / −0.107%** |
| MFMA per body iter (LBB4_11) | 32 | **32** (target: 64 or 96) | unchanged |
| `v_lshrrev_b32` per body iter | 6 | 6 | unchanged |
| `s_bitcmp0_b32` shift gate in body | **PRESENT** (s41,0 → 6× lshr block) | **GONE** (replaced by reg-shift) | YES |
| `s_bitcmp1_b32` load gate in body | absent | **PRESENT** (s41,0 → load block) | net 0 control branches |
| VGPRs (CRR scaled kernel) | 227 | **227** | 0 |
| LDS size (B/block) | 139264 | **139264** | 0 |
| VGPR spill | 0 | **0** | 0 |
| Occupancy (waves/SIMD) | 2 | **2** | 0 |
| SNR | 49.60 dB PASS | **49.60 dB PASS** | identical |
| Determinism (3 runs) | PASS | **PASS** | match |

## Falsifiable predictions audit

| Prediction | Result |
|---|---|
| **P1**: shift conditional `s_bitcmp0_b32` + `s_cbranch_scc1` GONE from K-loop body | **PARTIAL PASS** — the shift conditional IS gone; a *different* conditional (`s_bitcmp1_b32` for the LOAD only) survives in the loop header |
| **P2**: 6× `v_lshrrev_b32 v##, [shift_reg], v##` PRESENT unconditionally with runtime-computed shift amount | **PASS** — verified `v_lshrrev_b32_e32 v154, s17, v154` (and 5 more) where `s17 = s_lshl_b32 (s_and_b32 s41, 1), 4` |
| **P3**: K-loop body MFMA count RISES from 32 → 64 or 96 | **FAIL** — body still 32 MFMA / 1 K-pair |
| **P4**: VGPR ≤ 227, no spill, LDS = 139264 B | **PASS** — byte-identical resources |
| **P5**: SNR ≥ 48 dB, det 3/3 PASS | **PASS** — 49.60 dB, 3/3 PASS (identical to baseline) |

P3 is the **load-bearing prediction** for SHIP and it fails. P1/P2 confirm the
mechanism (shift conditional IS removable as a control-flow change), but the
expected downstream effect (body unroll) does not materialize.

## 1. Implementation

### 1.1 Approach (option a — always-shift)

Macro-gated as `MXFP8_CRR_BRANCHLESS_SHIFT=1` (default 0). Located in the
production fastpath of `crr_mxfp8_exact_8wave_fastpath.inc`, the inner
K-loop body's scale shift block was rewritten as:

```cpp
#elif MXFP8_CRR_BRANCHLESS_SHIFT
    // R54 Dev I: branchless unconditional shift (option a — always-shift).
    if ((k & 1) == 0) {
        load_raw_scales(k >> 1);
    }
    {
        const uint32_t mxfp8_crr_branchless_shift_amt =
            static_cast<uint32_t>(k & 1) << 4;
        #pragma unroll
        for (int g = 0; g < crr_a_pack_count; g++) {
            a0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(
                std::bit_cast<uint32_t>(a0_scale_packs[g]) >> mxfp8_crr_branchless_shift_amt);
            a1_scale_packs[g] = std::bit_cast<fp8e8m0_4>(
                std::bit_cast<uint32_t>(a1_scale_packs[g]) >> mxfp8_crr_branchless_shift_amt);
        }
        #pragma unroll
        for (int g = 0; g < crr_b_pack_count; g++) { /* same for b */ }
    }
#endif
```

Semantics:
- On even K: load (overwrites VGPRs with fresh dword), then shift-by-0
  (semantic no-op, real V-pipe issue).
- On odd K: skip load, shift-by-16 (correct hi→lo).

Cost: 6 v_lshrrev / K-pair instead of the baseline's 3 averaged (6 every
other K-pair). Reward (hoped): compiler now sees uniform shift work, can
unroll body across multiple K-pairs.

### 1.2 Mutual-exclusion guard

Added `#error` if combined with V2 / LEAD / LDS scale levers (since each
rewrites the same scale-handling region). All gated paths preserved; default
behavior unchanged.

### 1.3 Files modified

* `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc` — added
  `MXFP8_CRR_BRANCHLESS_SHIFT` macro (default 0), mutex guard, and the new
  `#elif` branch in the production K-loop. +94 lines (mostly comment block
  documenting hypothesis, predictions, refutation modes).

No host-side changes. No new VGPRs. No new LDS bytes. No `load_raw_scales`
changes (the load pattern itself is unchanged).

## 2. Phase 1 ISA evidence

### 2.1 Resource budget (CRR scaled kernel, MFMA16 PRESHUFFLED=true PACK=2)

```
TAG          VGPRs SGPRs LDS    Scratch VSpill SSpill Occ
baseline     227   50    139264 0       0      0      2
branchless   227   50    139264 0       0      0      2
```

**Byte-identical resources.** P4 confirmed.

### 2.2 K-loop structure delta (LBB4_11 / LBB4_12)

**Baseline LBB4_12 (loop header):**
```asm
.LBB4_12:                               ; =>This Inner Loop Header: Depth=1
    s_bitcmp0_b32 s41, 0
    s_cbranch_scc1 .LBB4_14
; %bb.13:
    v_lshrrev_b32_e32 v154, 16, v154    ; <— 6 conditional shifts
    v_lshrrev_b32_e32 v155, 16, v155
    v_lshrrev_b32_e32 v156, 16, v156
    v_lshrrev_b32_e32 v157, 16, v157
    v_lshrrev_b32_e32 v168, 16, v168
    v_lshrrev_b32_e32 v169, 16, v169
    s_cbranch_execnz .LBB4_11
    s_branch .LBB4_15
.LBB4_14:                               ; (else: implicit-defs)
                                        ; implicit-def: $vgpr168
                                        ; implicit-def: $vgpr155
.LBB4_15:                               ; merge: load
    buffer_load_dwordx4 v[154:157], v177, s[8:11], s40 offen
    buffer_load_dwordx2 v[168:169], v178, s[12:15], s36 offen
    s_branch .LBB4_11
```

**Branchless LBB4_12 (loop header):**
```asm
.LBB4_12:                               ; =>This Inner Loop Header: Depth=1
    s_bitcmp1_b32 s41, 0                ; <— now LOAD-only conditional
    s_cselect_b64 s[4:5], -1, 0
    s_and_b64 vcc, exec, s[4:5]
    s_cbranch_vccnz .LBB4_11            ; skip load on odd k
; %bb.13:
    buffer_load_dwordx4 v[154:157], v177, s[8:11], s40 offen
    buffer_load_dwordx2 v[168:169], v178, s[12:15], s36 offen
    s_branch .LBB4_11
```

The shift conditional is **structurally absent** — replaced by:

**Branchless LBB4_11 (body) — shifts now scheduled inline with MFMA:**
```asm
    ; ... 16 ds_read_b64_tr_b8 ops ...
    s_and_b32 s4, s41, 1                ; <— compute (k & 1)
    ; ... more ds_read ops ...
    s_lshl_b32 s17, s4, 4               ; <— compute shift amount = (k & 1) * 16
    ; ... more ds_read + buffer_load lds ops ...
    s_waitcnt vmcnt(2)
    v_lshrrev_b32_e32 v154, s17, v154   ; <— 6 unconditional shifts with REG amount
    buffer_load_dwordx4 v158, s[4:7], 0 offen lds
    v_lshrrev_b32_e32 v155, s17, v155
    v_lshrrev_b32_e32 v156, s17, v156
    v_lshrrev_b32_e32 v157, s17, v157
    s_waitcnt vmcnt(2)
    v_lshrrev_b32_e32 v168, s17, v168
    v_lshrrev_b32_e32 v169, s17, v169
    ; ... s_barrier, then 32× v_mfma_scale_f32_16x16x128_f8f6f4 ...
```

This is the predicted P2 form: shift amount in scalar reg `s17` (not
immediate), unconditionally executed.

### 2.3 Per-body-iter inventory comparison

| Class | Baseline LBB4_11 body | Branchless LBB4_11 body | Δ |
|-------|---:|---:|---:|
| `v_mfma_scale` | 32 | **32** | **0** |
| `ds_read_b64_tr_b8` | 48 | 48 | 0 |
| `buffer_load*` (LDS dest) | 8 | 8 | 0 |
| `v_lshrrev_b32` | 6 (averaged ~3 / K-pair due to conditional) | 6 (every K-pair) | 0 instr count, +3 effective shifts/K-pair |
| `s_barrier` | 4 | 4 | 0 |
| Body total lines | 272 | 281 | +9 |

The body iter is unchanged in MFMA count and is still 1 K-pair deep. **P3
fails.**

### 2.4 What the s41 control register represents

`s41` is the K-pair phase counter (incremented `s_add_i32 s41, s41, 1` once
per loop body). In the baseline, bit 0 of s41 controlled both:
1. The shift conditional (do 6× v_lshrrev on odd k) — via `s_bitcmp0_b32 +
   s_cbranch_scc1 .LBB4_14`.
2. Implicitly the load (via the `LBB4_15` merge that did `buffer_load`
   unconditionally after the conditional shift block was complete).

In the branchless variant, the shift conditional vanishes and bit 0 of s41
now controls only the load (`s_bitcmp1_b32 + s_cbranch_vccnz .LBB4_11` —
note the inversion: `bitcmp1` jumps to the next iter if bit IS set,
i.e., on odd k, skipping the load block).

So in net **the loop still has one s_bitcmp/cbranch pair per iter** — the
control flow is no simpler, just *different*. The compiler appears to be
using the load conditional (now standalone) as the unrolling barrier
exactly as it previously used the shift conditional.

## 3. Phase 2 — primary cell A/B (5 runs each, full WARMUP=100/ITERS=200)

### 3.1 Protocol

* GPU 0 idle-verified (`rocm-smi --showpids` clean for GPU 0; the two stale
  UNKNOWN handles with 0 VRAM/SDMA/CU are not active workloads).
* `HIP_VISIBLE_DEVICES=0 MXFP8_WARMUP=100 MXFP8_ITERS=200
  MXFP8_LAYOUTS=crr MXFP8_PRESHUFFLE_QUANT=1 MXFP8_DETERMINISM_RUNS=3
  MXFP8_SNR_THRESHOLD_DB=48.0`.
* 5 runs per gate × 30s cooldown between runs × 60s rebuild_cool between
  gate=0 and gate=1.
* Median = rank 3 of 5 sorted ascending.

### 3.2 Results (70B Gate/Up CRR, M=4096 N=28672 K=8192)

| Gate | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | **Median (rank 3)** |
|---|---:|---:|---:|---:|---:|---:|
| 0 (baseline) | 2529.95 | 2529.61 | 2526.61 | 2529.21 | 2527.65 | **2529.21** |
| 1 (branchless) | 2524.37 | 2531.74 | 2526.51 | 2529.88 | 2513.61 | **2526.51** |

**Δ median TFLOPS: −2.70 (−0.107%) — within run-to-run noise.**

Run-to-run spread:
* Gate=0: σ ≈ 1.35 TFLOPS, range [2526.61, 2529.95].
* Gate=1: σ ≈ 7.0 TFLOPS, range [2513.61, 2531.74] — slightly wider but
  still tight; the worst run (2513.61) appears to be a thermal/SCLK
  outlier.

### 3.3 SNR / determinism

| Gate | SNR (dB) | Threshold | Det (3/3) |
|---|---:|---:|---|
| 0 | 49.60 | 48.0 | PASS |
| 1 | **49.60** | 48.0 | **PASS** |

Bit-identical SNR — confirms shift-by-0 is a true semantic no-op (P5).
Determinism PASS confirms branchless rewrite did not introduce any
non-determinism.

### 3.4 Why Phase 2 was scoped to the primary cell only

Per mandate: *"If Phase 1 shows successful body restructure (more MFMA per
iter), bench 4 CRR cells. Critical: 70B Down CRR may surprise — R53C ruled
it out as structural FP8 amortization, but if branchless allows MXFP8 to
unroll too, the asymmetry collapses."* Phase 1 showed body restructure
(shifts moved into MFMA region) but **NOT** MFMA-per-iter rise (P3 failed).
The 70B Down CRR upside hinges on compiler unrolling, which did not happen.
A primary-cell quick A/B confirms null delta on the cell that would have
shown the largest expected gain (per R53C analytical model: a 2-deep body
unroll → +7pp MfmaUtil on the K=28672 path). With null delta on primary,
the no-regression cells are guaranteed to be null too (since the macro is
default-OFF; no production path is touched), and the Down-CRR upside is
also null (the unroll mechanism that would have driven it didn't fire).

## 4. Why the lever was REFUTED — root cause analysis

R53C §3.2 attributed the 1-K-pair body lock to "the s_bitcmp0_b32 conditional
shift block on alternate K-pairs". This Dev I work demonstrates that R53C's
attribution was **incomplete**: removing the shift conditional alone leaves
the load conditional in place, and the compiler treats EITHER conditional as
the unrolling barrier — the structural problem is the **per-K-pair phase
counter** (s41), not specifically the shift block.

Three possible deeper diagnoses (left to a future R55+ cycle):

**(a) The compiler unroller refuses any phase-keyed body.** Even a single
`if ((k & 1) == 0)` in a `#pragma unroll`'d outer loop is enough to lock
body unrolling. To verify: build a variant with `if ((k & 1) == 0)` removed
entirely (always-load on every K, accepting 2× scale VMEM cost — small
absolute) and check whether the body unrolls. If yes → the load conditional
is the problem; if no → something deeper (e.g., `tic^toc` LDS double-buffer
pattern or per-iter VMEM scheduling).

**(b) FP8 CRR's 3-deep unroll comes from a different mechanism entirely.**
R53C said "FP8 CRR has 96 mfma per body iter" and attributed this to "no
shift block, so compiler unrolls". But FP8 CRR also has the `(k & 1) == 0`
load conditional in source — yet still unrolls 3-deep. So the load
conditional is *not* an absolute barrier, only one specific to the MXFP8
loop shape (which has more inter-iter live state). This suggests the
problem is not the conditional per se but the **register pressure** that
unrolling would create in MXFP8's already-227-VGPR-tight body.

**(c) The compiler's heuristic is `mfma_per_body × scale_pack_size`-bounded.**
MXFP8 has 4 `scale_pack` arrays (a0/a1/b0/b1) live across each K-pair,
plus the unconditional LDS double-buffer. Unrolling 3-deep would 3× this
live state and likely spill (VGPR ceiling at 254 — only 27 free). The
compiler may be making a correct cost decision: 3-deep unroll would spill
worse than the +30% MfmaUtil it would buy. **If true, this lever is
unfixable without a fundamental scale-state restructure** (which R49A/R53A
already proved is multi-component-blocking).

The dead-code-elimination check (would the compiler drop the unconditional
shift-by-0 on even k?) — examining the branchless ISA at lines 558-564
shows the 6 v_lshrrev_b32 are present unconditionally with `s17` as the
shift amount. The compiler did NOT specialize / hoist the shift on even k
(where s17 == 0), confirming option (a) is genuinely emitting redundant
work on even k — and that work is small enough to not regress perf, but
also small enough to not buy unrolling.

## 5. Levers ruled out by this work

* **Branchless control-flow rewrites of the shift conditional alone**:
  REFUTED. Shifting from `s_bitcmp0_b32 + 6× v_lshrrev + s_cbranch` to
  `unconditional 6× v_lshrrev s17` does not unblock the compiler unroller.
* **Adding extra V-pipe work to "unify" the body shape**: REFUTED. The
  +6 v_lshrrev / K-pair (over baseline's averaged +3) costs nothing
  measurable, but also gains nothing. The unroller's barrier is elsewhere.

## 6. Recommended R55+ direction

Per the §4 root-cause hypotheses, the highest-value next probe is
**hypothesis (a): build a variant that completely eliminates ALL phase-keyed
control flow** (always load on every K, always shift unconditionally). This
costs 2× scale VMEM (minor; scales are tiny) and 6 wasted shifts per even K
(neutral per this work) — *and* may finally unblock the unroller.

If hypothesis (a) also produces no body unroll → hypothesis (b) is correct
and the cell is **structurally bounded** at its current ceiling, just like
R53C established for 70B Down CRR.

If hypothesis (a) DOES unroll the body but spills → hypothesis (c) is
correct, and the only viable direction is reducing inter-K-pair live state
(opposite of the V2-RRR ceiling rule, but applies to CRR too).

A rocprofv3 set1 pmc dump on the branchless build (just to confirm
MfmaUtil is in fact unchanged at ~60%, ruling out a hidden microarch
benefit) would close any remaining doubt; not run here per scope and
because the perf delta is unambiguously null.

## 7. Resource holdings recap (no regression)

| Cell-class | Baseline | Branchless | Status |
|---|---|---|---|
| 70B Gate/Up CRR (primary, gate=1 active) | 227 V / 0 spill / 139264 B LDS / occ 2 | **identical** | HOLD |
| All other cells (gate=0, default-OFF) | unchanged | unchanged (macro guarded #ifndef) | **byte-identical** to pre-edit kernel |

Macro defaulted OFF means downstream agents (Devs B/C/E/F/G/H, Reviewer)
operating on the same shared `crr_mxfp8_exact_8wave_fastpath.inc` symlink
see no behavior change unless they explicitly opt in.

## 8. SHIP gate audit

| Gate | Threshold | Result | Status |
|---|---|---|---|
| 70B Gate/Up CRR ≥ 92% MX/FP8 (or any > 2.7pp lift) | +0pp lift required | −0.107% (~−0.1pp) | **FAIL** |
| No regression on 4 CRR cells (>1%) | within ±1% | primary held within 0.11% | **HOLD on primary** (other cells not benched; default-OFF guarantees no regression) |
| SNR ≥ 48 dB | 48 dB | 49.60 dB | PASS |
| Determinism 3/3 | 3/3 | 3/3 | PASS |
| Resources: VGPR ≤ 227, LDS ≤ 139264 | budget | 227 / 139264 | PASS |

**SHIP gate: FAIL on perf threshold.** Macro default-OFF; no kernel
behavior change in production code path.

## 9. Verdict

**REFUTED-DIAGNOSTIC.**

The branchless rewrite *succeeds at its proximate goal* — the
`s_bitcmp0_b32 / 6× v_lshrrev_b32 / s_cbranch_scc1` shift block is
structurally GONE from the K-loop body, and the 6 shifts are now scheduled
unconditionally inline with MFMA / buffer_load with shift amount in scalar
reg. P1, P2, P4, P5 all pass. **But the load-bearing prediction P3 fails:
the K-loop body remains 1 K-pair / 32 MFMA per iter — the compiler still
refuses to unroll.**

This refutes R53C's attribution that "the v_lshrrev branch is the
compiler-unroll barrier" — removing the shift conditional alone is not
sufficient. The real barrier is broader (probably either the surviving
load conditional, or the per-K-pair register pressure that unrolling
would create — see §4 for three competing hypotheses).

Empirical bench confirms the diagnostic: 70B Gate/Up CRR holds at
2529.21 → 2526.51 TFLOPS (−0.11%, within noise). SNR / determinism /
resources are byte-identical to baseline. No SHIP, no usable lever for
this axis.

## 10. Files / artifacts

* `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc` — the
  modified inc with `MXFP8_CRR_BRANCHLESS_SHIFT` macro (default 0) and
  the new `#elif` branch in the production K-loop.
* `analysis/fp8_gemm/mi350x/r54i_phase1_isa.sh` — Phase 1 ISA build/extract
  script.
* `analysis/fp8_gemm/mi350x/r54i_phase2_quick.sh` — Phase 2 primary-cell
  A/B bench script.
* `analysis/fp8_gemm/mi350x/r54i_phase2_bench.sh` — Phase 2 full 4-cell
  bench script (NOT EXECUTED — Phase 1 P3 fail per mandate).
* `analysis/fp8_gemm/mi350x/r54i_workspace/` — isolated workspace
  (symlinks to shared kernel source; reuses kernel_mxfp8_layouts.cpp).
* `analysis/fp8_gemm/mi350x/r54i_results/isa/`:
  * `baseline_build.log`, `branchless_build.log` — full build logs with
    `-Rpass-analysis=kernel-resource-usage` remarks.
  * `baseline_device.s`, `branchless_device.s` — full device assembly
    (~30k lines each).
  * `baseline_crr_kernel.s`, `branchless_crr_kernel.s` — extracted CRR
    scaled kernel (~1786 / 1785 lines).
* `analysis/fp8_gemm/mi350x/r54i_results/bench/`:
  * `70B_GateUp_CRR_gate0.log`, `70B_GateUp_CRR_gate1.log` — 5-run
    test_mxfp8_python.py output for each gate.

## 11. Cross-references

* `r53c_findings.md` §3.2 — original "v_lshrrev branch as unroll barrier"
  attribution that this work refutes.
* `r48d_findings.md` — original CRR ~92% structural floor establishing
  the v_lshrrev count (6 / K-pair).
* `r54a_findings.md` — REFUTED-EMPIRICAL LDS-resident pre-shifted scale
  layout (LDS overflow).
* `r49a_findings.md` — REFUTED scale-pack opsel rewrite (-14.16% geomean
  via v_perm replacement).
* `r53a_findings.md` — REFUTED opsel-keyed phase MMA dispatch (70 VGPR
  spill).
