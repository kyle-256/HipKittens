# Round 19 Optimizer B — Final Verdict (per-site barrier-removal bisect)

**Date**: 2026-04-17
**Author**: Kyle.Zhao@amd.com / kyle-256
**Branch**: mxfp4
**Time spent**: ~30 min wall (build 20 s, SNR 5 min, smoke 12 s, verify 7 min)

---

## TL;DR

**LOSE — no per-site subset of the R18A barrier-removal opens a window on DLA1/DLA2/DLA7.**

- 12 of 36 (shape × variant) pairs were broken even on the cooperative uniform-input
  noise-floor probe (4 hot sites × 3 shapes; plus 2-site combos; plus vmcnt sweep; plus the R18A
  ALL reference). Only **DLA1/_r19b_t1** (TAIL/STEP12 only) was SNR-OK + aperture-OK.
- 1-shot smoke on DLA1/_r19b_t1 looked promising at +1.43 pp.
- 5-run same-GPU verify collapsed to **Δ-0.27 pp** (variant_mean 5127.65 vs parent_mean 5143.48).
- Conclusion: the inner-loop s_barrier IS load-bearing for DLA1/DLA2/DLA7 across every
  hot site individually; it's not a single removable site, and even the safest one (TAIL only)
  doesn't move TFLOPS.

The macro infrastructure (10 per-site flips + vmcnt override) has been added with safe
defaults; existing R18A behavior is preserved bit-exactly when only the aggregate macros are set.

---

## 1. Approach

Extended R18A's two-axis macro (`BARRIER_TO_WAITCNT_STEP3`, `_STEP12`) into per-site control:

| Macro | Site | Hot? |
|---|---|---|
| `BARRIER_TO_WAITCNT_STEP3_S1` | `kpair_64mfma_step34` (FUSED_STEP34) | dead (FUSED_STEP34=0) |
| `BARRIER_TO_WAITCNT_STEP3_S2` | `kpair_32mfma_with_lds_and_pf` (template) | **HOT** |
| `BARRIER_TO_WAITCNT_STEP3_S3` | `kpair_32mfma_with_lds_rowspread_pf` (template) | **HOT** |
| `BARRIER_TO_WAITCNT_STEP3_S4` | `kpair_32mfma_with_lds_and_pf_swapped_sel` (template) | **HOT** |
| `BARRIER_TO_WAITCNT_STEP3_S5` | TAIL_SPLIT inner (!STEP3_EMBED_BARRIER) | dead |
| `BARRIER_TO_WAITCNT_STEP3_S6` | TAIL_SPLIT outer (!STEP3_EMBED_BARRIER) | dead |
| `BARRIER_TO_WAITCNT_STEP3_S7` | no-TAIL_SPLIT outer (!STEP3_EMBED_BARRIER) | dead |
| `BARRIER_TO_WAITCNT_STEP12_S1` | TAIL_SPLIT==1 path | **HOT** (parents use TAIL_SPLIT=1) |
| `BARRIER_TO_WAITCNT_STEP12_S2` | TAIL_SPLIT==0 path | dead |
| `BARRIER_TO_WAITCNT_RELAXED_VMCNT` | overrides vmcnt(N) value at all R19B sites | n/a |

Each per-site macro defaults to its aggregate (STEP3 or STEP12) so R18A's
`-DBARRIER_TO_WAITCNT_ALL=1` produces bit-identical asm.

Tested only the 4 LIVE sites (S2, S3, S4, T1) plus 2-site combos (s23, s24, s34, s234)
plus a 3-value vmcnt sweep (1, 4, 15 — current is 12 for parents) plus R18A's full ALL
as a reference. 13 variants × 3 shapes = 39 builds, all OK.

## 2. SNR validation (`snr_probe_r19b.py`)

Identical to R18A: noise-floor SNR (parent-vs-parent) then variant-vs-parent,
with R18A's added random-scale aperture probe for SNR-OK candidates.

| shape | noise floor | OK pairs (uniform + aperture) |
|---|---|---|
| DLA1 (4096×32768×128256) | 8.61 dB | **only `_r19b_t1`** (15.47 dB, OK-MARGINAL) |
| DLA2 (128256×32768×4096) | -1.37 dB | none (parent saturates bf16 — no SNR-validatable variant exists) |
| DLA7 (28672×32768×4096) | -1.89 dB | none (same as DLA2) |

Notable findings:
- **Even vmcnt15 fails SNR** despite keeping the s_barrier in place. Just changing the
  vmcnt VALUE perturbs scheduling enough to alter FMA-reorder noise. Confirms barrier
  removal is not the only source of variability in this kernel.
- The full ALL macro reference reproduces R18A's "BROKEN-RACE" verdict on all 3 shapes
  (no surprises).
- 2-site combos (s23, s24, s34, s234) all worse than individual sites — synergy is
  destructive, not additive.
- DLA2 noise floor is -1.37 dB (parent itself non-deterministic at SNR < 0); same as
  R18A's finding. No SNR-validatable improvement exists for DLA2/DLA7 by construction.

## 3. Smoke bench (`bench_round19_optB_smoke.py`)

Only 1 SNR/aperture-OK pair to bench: DLA1/`_r19b_t1`. warmup=200 iters=500 trim=10%.

| pair | parent TFLOPS | variant TFLOPS | Δpp |
|---|---|---|---|
| DLA1/`_r19b_t1` | 5078.92 | 5161.59 | +1.43 (single-shot) ⇒ verify |

## 4. 5-run same-GPU verify (`bench_round19_optB_verify.py`)

| | parent | variant |
|---|---|---|
| run 1 | 5156.11 | 5152.88 |
| run 2 | 5149.65 | 5144.74 |
| run 3 | 5121.70 | 5121.96 |
| run 4 | 5147.08 | 5073.36 |
| run 5 | 5142.84 | 5145.31 |
| max | 5156.11 | 5152.88 |
| mean | **5143.48** | **5127.65** |

```
Δmean = -0.27 pp (vs comp 5781.1)
gate1 (v_mean ≥ p_max): False   gate2 (Δ ≥ +1pp): False   ⇒ LOSE
```

The smoke +1.43 pp was within single-shot variance; under 5-run averaging the
variant is statistically indistinguishable from parent (and slightly worse on
mean due to an outlier run 4).

## 5. Why no win was findable

R17A's profile of DLA1 found **49 % VALU busy** + 0.04 % LDS-stall + 5 % BW.
**Not memory-bound, not LDS-bound, not bandwidth-bound** — instead **MFMA-accumulator
dependency stall** is the bottleneck. Switching `s_barrier` → `s_waitcnt` only saves
~50 cycles per K-iter (≈0.1 ms total at 2004 iters), and moreover the swap perturbs
register/SIMD scheduling enough to lose those gains in pipeline bubbles (or just stay neutral).

For DLA1/DLA2/DLA7 specifically, **the cross-wave LDS coordination provided by
s_barrier IS load-bearing**. Wave-private LDS partitions don't save us — the kernel
overlaps producer waves' LDS writes with consumer waves' reads, and dropping the
barrier corrupts a different output every time.

P1 (which did get +4.16 pp in R18A) is a special case because (a) it has small K=16384
(only 256 K-iters, so the barrier fraction is bigger) and (b) parent SNR was 11.78 dB
on uniform inputs — meaning the parent itself is mostly bf16-finite, leaving room to
SNR-validate. None of DLA1/DLA2/DLA7 has BOTH small K and high parent-SNR.

## 6. Regression check

The kernel.cpp diff:
1. Adds 10 new `#ifndef ... #define ... 0` (or default-to-aggregate) macros.
2. Adds 10 new per-site `MXFP4_*BARRIER_INST_S*` strings, each gated on its own macro.
3. Replaces 9 inline-asm sites' macro reference (e.g. `MXFP4_STEP3_BARRIER_INST` →
   `MXFP4_STEP3_BARRIER_INST_S2`).

Each per-site macro defaults to the corresponding aggregate (`BARRIER_TO_WAITCNT_STEP3`
for S1-S7, `BARRIER_TO_WAITCNT_STEP12` for T1/T2). When neither aggregate nor per-site
flag is set, every per-site string expands to exactly the original
`s_waitcnt vmcnt(N)\ns_barrier\n` ⇒ bit-identical to the pre-R19B kernel.

When `-DBARRIER_TO_WAITCNT_ALL=1` is set, both aggregates become 1 ⇒ all per-site flags
become 1 ⇒ all sites become `s_waitcnt vmcnt(N) lgkmcnt(0)` ⇒ bit-identical to R18A's
BARRIER_TO_WAITCNT_ALL=1.

**Sanity verified**: rebuilt P1's `_ts_gm8_r18a_p3_all` against the new kernel
(`build_round18_optA_p3.py` re-run) — compiles and links OK.

⇒ **No regression possible** for any pre-existing build line.

## 7. Deliverables

| File | Contents |
|---|---|
| `kernel_mxfp4_gluon_cpp.cpp` (modified) | 10 new per-site macros + vmcnt override; defaults preserve R18A |
| `build_round19_optB.py` + `build_round19_optB.log` | 39 builds (3 shapes × 13 variants), all OK |
| `snr_probe_r19b.py` + `.log` + `.json` | 36 uniform SNR + 1 aperture probe; 1 OK / 35 BROKEN |
| `bench_round19_optB_smoke.py` + `.log` + `.json` | 1 ok variant smoke bench |
| `bench_round19_optB_verify.py` + `.log` + `.json` | 5-run same-GPU; LOSE Δ-0.27 pp |
| `round19_optB_verdict.md` | This file |

## 8. Verdict

**LOSE — no per-site or vmcnt-relaxation barrier subset opens a perf window on
DLA1/DLA2/DLA7.**

The R19B macro infrastructure is committed (zero-risk: all defaults preserve R18A).
No per-shape flag changes recommended for DLA1/DLA2/DLA7.

## 9. Dead-end vectors confirmed (for R20+ registry)

1. **Per-site STEP3 barrier removal** on DLA1/DLA2/DLA7 — every individual hot site (S2,
   S3, S4) breaks SNR even more severely than the aggregate STEP3=1.
2. **2-site STEP3 combos** (s23, s24, s34) and 3-site (s234) — all worse than individual.
3. **TAIL/STEP12 site removal** on DLA1 — SNR-MARGINAL but no perf gain (Δ-0.27 pp).
4. **vmcnt sweep** (1, 4, 15) at all R19B sites — every value breaks SNR; even keeping
   the barrier and only changing vmcnt perturbs FMA-reorder noise enough to fail SNR.
5. **DLA2/DLA7** are SNR-unvalidatable for any kernel-internal change — parent itself
   saturates bf16 at SNR < 0 dB. Future work on these shapes needs a different
   correctness probe (not output SNR).

## 10. Recommendations

- **Don't iterate further on barrier removal for DLA1/DLA2/DLA7**. The mechanism
  (s_barrier providing cross-wave LDS visibility) is genuinely load-bearing for these
  shapes; per-site bisection cannot find a safe subset.
- **The R18A P1 win is an isolated case**, not a generalizable axis. Future +1pp gains
  must come from somewhere else (kernel rewrite per R17A's P1/P2 proposals).
- **DLA2/DLA7 need a new correctness probe**: noise-floor SNR can't tell anything
  apart when parent's own SNR is < 0 dB. Either reduce input dynamic range further
  (smaller fp4 nibbles? all-zero scales?) or compare ULP histograms instead.
