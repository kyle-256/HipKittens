# R41 Integration Verdict — full 42-shape 5-run consensus

**Date**: 2026-04-19
**Reviewer**: R41 Final Integration
**Manifest**: `R41_INTEGRATION_MANIFEST.json` (5 R41A + 2 R41B + 1 R40A + 34 R40B)
**Bench**: `bench_all_42_R41_INTEGRATION.py`, 5 runs × 42 shapes, 8 GPUs parallel
**Bench rules**: warmup=200, iters=500, trim=0.10, R39B random-scale gate
(seed=42, wcf<2%, snr_med>=10dB, finite>=0.99)
**Verified-correct**: `n_OK >= 3 AND wcf_max < 2% AND wcf_std < 1% AND fin_min >= 0.99`

---

## Headline

**Verified-correct: 20/42** (target was 30/42 — **MISSED by 10**).
**WIN (pct_comp >= 100%): 6/42**.
**Total CRASH: 2/42**.

The R41 round did not hit the 30/42 target. It locked in **all 8 per-shape
overrides** as expected (5 R41A + 2 R41B + 1 R40A all PASS_5/5 in integration),
but the underlying R40B baseline regressed under this independent 5-run probe:
12 of the 34 shapes that R40B was supposed to carry are now FLAKE/WRONG.

---

## Source-distribution table

| Source | shapes | verified-correct | WIN | regression vs component |
|---|---|---|---|---|
| R41A (cluster-C deep-K fence) | 5 | **5/5** | 0 | none (drift -1.0% to +0.1%) |
| R41B (cluster-B variant retune) | 2 | **0/2** | 0 | **2/2 verdict-regression** (PASS_3/5 → FLAKE_2/5) |
| R40A (K=128256 PF fence) | 1 | **1/1** | 0 | drift -2.3% (acceptable) |
| R40B (base) | 34 | **14/34** | 6 | 7/34 verdict regression vs R41D 5-run |
| **Total** | **42** | **20** | **6** | 9 regressions |

---

## Full leaderboard (sorted by source then shape)

| shape | src | verdict | tflops | comp | pct | wcf_max | wcf_std | fin_min | n_OK | VC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| 4096x4096x32768 | R41A | PASS_5/5 | 3987.4 | 5152.8 | 77.4% | 0.0000 | 0.0000 | 1.0000 | 5 | Y |
| 4096x6144x32768 | R41A | PASS_5/5 | 3095.7 | 3784.2 | 81.8% | 0.0000 | 0.0000 | 1.0000 | 5 | Y |
| 4096x28672x32768 | R41A | PASS_5/5 | 3499.0 | 5649.9 | 61.9% | 0.0149 | 0.0059 | 0.9990 | 5 | Y |
| 4096x128256x32768 | R41A | PASS_5/5 | 3116.9 | 3195.3 | 97.5% | 0.0015 | 0.0006 | 0.9998 | 5 | Y |
| 14336x4096x32768 | R41A | PASS_5/5 | 3142.4 | 5245.4 | 59.9% | 0.0012 | 0.0005 | 1.0000 | 5 | Y |
| 16384x4096x14336 | R41B | FLAKE_2/5 | 4487.8 | 5142.1 | 87.3% | 0.0104 | 0.0011 | 0.9856 | 2 | . |
| 32768x4096x2048 | R41B | FLAKE_2/5 | 3141.4 | 3131.8 | 100.3% | 0.0112 | 0.0017 | 0.9887 | 2 | . |
| 4096x32768x128256 | R40A | PASS_5/5 | 4141.5 | 5781.1 | 71.6% | 0.0017 | 0.0007 | 0.9977 | 5 | Y |
| 4096x4096x8192 | R40B | PASS_5/5 | 3790.8 | 3959.9 | 95.7% | 0.0132 | 0.0028 | 1.0000 | 5 | Y |
| 4096x4096x16384 | R40B | PASS_5/5 | 4242.0 | 4642.1 | 91.4% | 0.0178 | 0.0047 | 0.9956 | 5 | Y |
| 4096x14336x8192 | R40B | PASS_5/5 | 3986.2 | 4345.8 | 91.7% | 0.0049 | 0.0009 | 0.9910 | 5 | Y |
| 4096x14336x16384 | R40B | PASS_5/5 | 4201.0 | 5013.0 | 83.8% | 0.0073 | 0.0018 | 0.9904 | 5 | Y |
| 4096x32768x4096 | R40B | FLAKE_1/5 | 3737.1 | 4166.5 | 89.7% | 0.0086 | 0.0010 | 0.9831 | 1 | . |
| 4096x32768x6144 | R40B | FLAKE_1/5 | 4160.3 | 4548.6 | 91.5% | 0.0085 | 0.0009 | 0.9867 | 1 | . |
| 4096x32768x14336 | R40B | WRONG_5/5 | — | 5296.1 | — | 0.0268 | 0.0043 | 0.9749 | 0 | . |
| 4096x32768x28672 | R40B | FAIL_CRASH | — | 5568.2 | — | — | — | — | 0 | . |
| 6144x4096x8192 | R40B | PASS_5/5 | 3357.0 | 3822.0 | 87.8% | 0.0152 | 0.0034 | 0.9947 | 5 | Y |
| 6144x4096x16384 | R40B | PASS_5/5 | 3670.0 | 4428.1 | 82.9% | 0.0086 | 0.0023 | 0.9958 | 5 | Y |
| 6144x32768x4096 | R40B | PASS_3/5 | 4035.2 | 4291.0 | 94.0% | 0.0219 | 0.0065 | 0.9842 | 3 | . |
| 16384x4096x2048 | R40B | PASS_3/5 | 3073.6 | 2995.0 | 102.6% | 0.0005 | 0.0000 | 0.9853 | 3 | . |
| 16384x4096x3072 | R40B | PASS_5/5 | 3478.2 | 3492.3 | 99.6% | 0.0133 | 0.0011 | 0.9959 | 5 | Y |
| 16384x4096x4096 | R40B | PASS_5/5 | 3892.8 | 3951.8 | 98.5% | 0.0126 | 0.0017 | 0.9956 | 5 | Y |
| 16384x4096x6144 | R40B | PASS_5/5 | 4147.8 | 4259.9 | 97.4% | 0.0128 | 0.0024 | 0.9953 | 5 | Y |
| 16384x4096x7168 | R40B | PASS_5/5 | 4182.5 | 4443.2 | 94.1% | 0.0123 | 0.0023 | 0.9958 | 5 | Y |
| 16384x4096x28672 | R40B | FAIL_CRASH | — | 5525.3 | — | — | — | — | 0 | . |
| 16384x6144x2048 | R40B | PASS_4/5 | 3207.8 | 3047.6 | 105.3% | 0.0007 | 0.0000 | 0.9894 | 4 | . |
| 16384x6144x4096 | R40B | FLAKE_2/5 | 4092.7 | 4042.5 | 101.2% | 0.0288 | 0.0049 | 0.9898 | 2 | . |
| 16384x14336x2048 | R40B | PASS_5/5 | 3231.4 | 3301.3 | 97.9% | 0.0013 | 0.0002 | 0.9919 | 5 | Y |
| 16384x14336x4096 | R40B | FLAKE_2/5 | 4047.8 | 4255.8 | 95.1% | 0.0557 | 0.0168 | 0.9914 | 2 | . |
| 16384x28672x2048 | R40B | PASS_3/5 | 3366.2 | 3482.3 | 96.7% | 0.0107 | 0.0014 | 0.9789 | 3 | . |
| 16384x28672x4096 | R40B | PASS_4/5 | 4038.4 | 4411.7 | 91.5% | 0.0166 | 0.0048 | 0.9886 | 4 | . |
| 28672x4096x8192 | R40B | FLAKE_1/5 | 4248.0 | 4810.0 | 88.3% | 0.0126 | 0.0029 | 0.9825 | 1 | . |
| 28672x4096x16384 | R40B | FLAKE_1/5 | 4407.9 | 5350.6 | 82.4% | 0.0221 | 0.0054 | 0.9875 | 1 | . |
| 28672x32768x4096 | R40B | PASS_5/5 | 3896.2 | 4466.6 | 87.2% | 0.0119 | 0.0032 | 0.9902 | 5 | Y |
| 32768x4096x3072 | R40B | PASS_3/5 | 3613.3 | 3630.6 | 99.5% | 0.0176 | 0.0045 | 0.9834 | 3 | . |
| 32768x4096x7168 | R40B | PASS_5/5 | 4128.1 | 4666.8 | 88.5% | 0.0148 | 0.0036 | 0.9909 | 5 | Y |
| 32768x4096x14336 | R40B | PASS_3/5 | 4427.8 | 5223.4 | 84.8% | 0.0451 | 0.0109 | 0.9855 | 3 | . |
| 32768x6144x2048 | R40B | PASS_4/5 | 3318.8 | 3239.9 | 102.4% | 0.0098 | 0.0015 | 0.9909 | 4 | Y |
| 32768x14336x2048 | R40B | FLAKE_2/5 | 3435.0 | 3351.4 | 102.5% | 0.0120 | 0.0019 | 0.9883 | 2 | . |
| 32768x28672x2048 | R40B | WRONG_5/5 | — | 3353.4 | — | 0.0119 | 0.0028 | 0.9644 | 0 | . |
| 14336x32768x4096 | R40B | WRONG_5/5 | — | 4462.6 | — | 0.0206 | 0.0056 | 0.9786 | 0 | . |
| 128256x32768x4096 | R40B | PASS_3/5 | 3896.7 | 4536.4 | 85.9% | 0.0306 | 0.0099 | 0.9957 | 3 | . |

---

## Regression table — integration vs per-component 5-run baseline

A regression is recorded if the integration `verified_correct` flips from
true (per-component) to false, OR tflops drift > 5%.

| shape | src | base_tflops | base_verdict | int_tflops | int_verdict | drift_% | flag |
|---|---|---:|---|---:|---|---:|---|
| 128256x32768x4096 | R40B | 3898.6 | PASS_4/5 | 3896.7 | PASS_3/5 | -0.0% | REGRESS_VERDICT |
| 16384x4096x2048 | R40B | 3159.9 | PASS_5/5 | 3073.6 | PASS_3/5 | -2.7% | REGRESS_VERDICT |
| 16384x6144x2048 | R40B | 3193.6 | PASS_4/5 | 3207.8 | PASS_4/5 | +0.4% | REGRESS_VC (fin straddles) |
| 32768x14336x2048 | R40B | 3438.7 | PASS_3/5 | 3435.0 | FLAKE_2/5 | -0.1% | REGRESS_VERDICT |
| 32768x4096x3072 | R40B | 3673.5 | PASS_3/5 | 3613.3 | PASS_3/5 | -1.6% | REGRESS_VC (wcf flake) |
| 4096x32768x4096 | R40B | 3817.8 | PASS_3/5 | 3737.1 | FLAKE_1/5 | -2.1% | REGRESS_VERDICT |
| 6144x32768x4096 | R40B | 4059.4 | PASS_3/5 | 4035.2 | PASS_3/5 | -0.6% | REGRESS_VC |
| 16384x4096x14336 | R41B | 4482.0 | PASS_3/5 | 4487.8 | FLAKE_2/5 | +0.1% | REGRESS_VERDICT |
| 32768x4096x2048 | R41B | 3270.2 | PASS_4/5 | 3141.4 | FLAKE_2/5 | -3.9% | REGRESS_VERDICT |

**No catastrophic perf drift** (max |drift| = 3.9%); all regressions are verdict
regressions caused by the same near-gate finite/wcf flake mechanism documented
in R41D (the kernel's residual cell-corruption rate straddles the 0.99/2%
gates run-to-run).

### Notably, no regression on:
- All 5 R41A picks (drift -1.0% to +0.1%, all PASS_5/5)
- The R40A K=128256 pick (drift -2.3%, PASS_5/5)
- 22 R40B shapes (PASS_3/5 or better preserved)

---

## Improvement table — integration RECOVERED shapes vs R41D R40B baseline

These shapes WERE WRONG in R41D's R40B 5-run but the integration found them
PASS / partial in this 5-run, suggesting they're at the gate boundary and
not deterministically broken:

| shape | R41D baseline | integration | tflops |
|---|---|---|---:|
| 16384x14336x4096 | WRONG_3/5 | FLAKE_2/5 | 4047.8 |
| 16384x28672x2048 | WRONG_5/5 | PASS_3/5 | 3366.2 |
| 16384x28672x4096 | WRONG_3/5 | PASS_4/5 | 4038.4 |
| 16384x6144x4096 | WRONG_3/5 | FLAKE_2/5 | 4092.7 |
| 28672x32768x4096 | WRONG_3/5 | PASS_5/5 | 3896.2 |
| 28672x4096x16384 | WRONG_4/5 | FLAKE_1/5 | 4407.9 |
| 28672x4096x8192 | WRONG_5/5 | FLAKE_1/5 | 4248.0 |
| 32768x4096x14336 | WRONG_3/5 | PASS_3/5 | 4427.8 |
| 4096x32768x6144 | WRONG_5/5 | FLAKE_1/5 | 4160.3 |

7 of these (the 4 PASS_3+/5 and the 3 FLAKE) are recovery signals showing the
R40B kernel is on the gate boundary, not deterministically wrong. Net of
recoveries vs regressions: roughly even.

---

## Honest assessment

### Did we hit 30/42?
**No.** Result: **20/42 verified-correct** (R41 plan target was 30/42).

### What fell out?
1. **Both R41B promotes regressed** under independent 5-run reviewer probe.
   `(16384,4096,14336)` was PASS_3/5 in R41B's own 5-run → FLAKE_2/5 here.
   `(32768,4096,2048)` was PASS_4/5 → FLAKE_2/5. Per the R41B verdict's own
   warning ("dominant blocker is finite < 0.99, not wcf"), these were always
   right at the gate edge and a fresh 5-run sample dropped them below.

2. **R40B base shows persistent flake on cluster-B / cluster-D shapes.**
   12 of 34 R40B shapes did not get the strict 5-run gate (PASS_<3 or
   `wcf_std >= 1%`). This matches the R41D audit's observation that ~17%
   of R40B shapes are gate-boundary flake-only rather than deterministic-PASS.

3. **2 CRASHes carried over**: `4096x32768x28672` and `16384x4096x28672`
   crash at K=28672 — same as in earlier rounds. Not addressed by R41A/B.

4. **R41A is the lone unequivocal win**: 5/5 cluster-C deep-K shapes
   recovered cleanly under integration 5-run probe. Drift -1.0% to +0.1%.
   The `extract_tile vmcnt fence` is the most durable finding of the round.

### What is the R41 round actually worth?
- **+5 cluster-C shapes** verified-correct (pure win — these were
  catastrophic-WRONG before R41A).
- **+1 R40A K=128256 shape** verified-correct (R41D promotion confirmed).
- **R41B**: net 0 verified-correct (the 2 promotes regress; both shapes
  still flake but no worse than R40B baseline).
- **Total durable contribution: +6 verified-correct** vs. raw R40B at 14/42.

### Comparison summary
| Round | Locked-in verified-correct | Note |
|---|---|---|
| R40B raw (this 5-run) | 14/42 | 12 R40B shapes flake out, 2 CRASH |
| R40B + R40A K=128256 + 5x R41A | 20/42 | **R41 INTEGRATION: 20** |
| R40B + R40A + 5x R41A + 2x R41B | (would be) 20-22 | R41B promotes regress to FLAKE here |
| R41 plan TARGET | 30/42 | **MISSED by 10** |

---

## Files

- `R41_INTEGRATION_MANIFEST.json` — per-shape source + .so absolute paths
- `bench_all_42_R41_INTEGRATION.py` — bench harness (manifest-driven, 8-GPU parallel, 5-run consensus)
- `R41_INTEGRATION_5RUN.json` — full per-shape × per-run results + consensus
- `R41_INTEGRATION_5RUN.log` — bench progress log
- `build_R41_integration_manifest.py` — manifest builder script
- `R41_INTEGRATION_VERDICT.md` — this document

---

## GO/NO-GO recommendation for R41 commit

**WEAK GO** — commit R41A integration only (+5 cluster-C recoveries are
durable and unambiguous); **DO NOT commit R41B promotes** (both regress
to FLAKE_2/5 under independent 5-run); keep R40A K=128256 (PASS_5/5 stable).

This locks in **20/42 verified-correct** — the R41B work was directionally
correct (the 2 picks DID flip during R41B's bench) but the underlying flake
margin was too thin to survive a fresh 5-run probe. Next round should focus
on the finite-gate root cause (R35 quadrant / MFMA vgpr cohort race) rather
than more pfoff/variant axis sweeps that don't address the dominant blocker.
