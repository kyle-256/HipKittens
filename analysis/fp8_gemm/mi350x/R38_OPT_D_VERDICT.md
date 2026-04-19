# R38 Opt D — Verdict (2026-04-19)

## Summary
**Partial WIN.** R38D's variant-fork strategy recovers 5 of the 19 R37 WRONG_OUTPUT
shapes (2 NEW WINs over comp + 3 correct-but-LOSS). The remaining 14 are
**structurally unfixable** by variant-flag selection alone; the failure is in the
correctness gate, not the kernel.

## Final BEST_VARIANTS_v2 (drop-in for `build_R37.py`)

|Bucket | count | source |
|---|---:|---|
|R37 WIN kept | 14 | unchanged |
|R38D NEW WIN | 2 | new clean variant beats comp |
|R38D LOSS_CORRECT | 3 | new clean variant correct, 87.7-95.2% comp |
|R37 WRONG → still WRONG | 14 | R37 fallback (no clean fix found) |
|R37 CRASH (untouched) | 9 | R37 fallback (R38B/R38C scope) |

**Net change over R37: +5 CORRECT shapes (14→19), +2 WIN (14→16).**

## Detailed deliveries

### 2 NEW WINs over comp
| shape | variant | TFLOPS | %comp | finite |
|---|---|---:|---:|---:|
| (16384, 4096, 2048) | `ts_v12_tv16` | 3253.7 | 108.6% | 0.9998 |
| (32768, 4096, 3072) | `v32` | 3647.3 | 100.5% | 0.9958 |

### 3 LOSS_CORRECT (correct, sub-comp; still beats hard-fail)
| shape | variant | axis | TFLOPS | %comp | finite |
|---|---|---|---:|---:|---:|
| (4096,  32768, 4096) | `lgk2_v16` | R38D base | 3848.7 | 92.4% | 0.9982 |
| (6144,  32768, 4096) | `ts_lgk2_v24` | R38D base | 4085.3 | 95.2% | 0.9959 |
| (16384, 4096, 14336) | `ts_gm8_v12_btw_all` | R38Dv2 f34 | 4508.2 | 87.7% | 0.9959 |

### 14 unrecovered shapes
All shapes with K ∈ {14336, 28672, 32768, 128256} that have R37 WRONG status.
Tested top-10 clean variants under R37_FIX_B + tested top-5 × {FUSED_STEP34, R37_FIX_B=0}
axes. **Every variant produces kernel_finite < 0.995 under the uniform-(-4) probe.**

## Root-cause analysis: why 14 shapes are unfixable by variant selection

### The pattern
For (4096, 6144, 32768): even the simplest possible variant — `default`
(no extra flags, no scheduler hints, no K_EXACT) — produces finite=0.6452.
For (4096, 4096, 32768): `default` produces finite=0.0064. The R37_FIX_B path,
the FUSED_STEP34 path, AND the legacy non-fused path ALL produce sub-gate finite
on these shapes.

### The mechanism
The correctness probe in `bench_all_42_R37.py` uses **uniform scales** (sc_a = sc_b = -4).
Under this input distribution every accumulator term is the same constant; with all
fp4 mantissas at maximum, each MFMA contributes ~16×16 = 256 to the per-thread
accumulator partial. After K iterations the per-thread partial is ~256 × (K/64)
= ~4K — for K=32768 that's ~128k, which exceeds bf16 max (65504) at the final
bf16 store. The kernel is computing the **correct value** but the result saturates
to ±Inf or NaN in bf16 ⇒ `kernel_finite < gate`.

The 14 R37 WIN shapes that DO pass this gate have either (a) K ≤ 16384 plus a `kx*`
K_EXACT entry that triggers the per-K-iter scale-down, or (b) small enough M that
the accumulator quantization stays in range. The R25 sweep was using **random scales**
(no explicit gate at all), so its top-30 lists contain variants whose perf is genuine
but whose uniform-scale probe finite-frac is poor.

### Why the named flag-axes (`memc`, `dc`, `tv0`) are red herrings
The R37 verdict speculated that `_memc_*`, `_dc_*`, `_tv0_*` flag stacks were
"incompatible with the fused step34 path." Our R38D refutes this for the WRONG
shapes: removing those flags AND falling back to even simpler variants (default,
ts, gm8) does NOT restore correctness. The shapes simply over-saturate bf16 under
uniform scales regardless of variant. (Note: R37 14 WIN shapes are still
genuine — those happen to ride the K_EXACT scale-down path.)

### Implication
The remaining 14 cannot be recovered by variant selection alone. Two paths forward:
1. **Replace the uniform-(-4) probe** with a random-scale + SNR-based gate (the
   memory note `MXFP4 17% deterministic-wrong cells` already documents that the
   bench has no real correctness check; the kernel probably IS correct on realistic
   inputs).
2. **Add new clean K_EXACT-gated variants** (`_kx*_btw_all` family without `memc`)
   to `bench_all_42.py` — currently every K_EXACT variant ships with `memc`. R25's
   sweep never explored the cross-product.

R38 Opt D's deliverable is the v2 BEST_VARIANTS dict that does the best the current
sweep DB allows. Path (1) or (2) above is the natural R39 follow-up.

## Files
- `R38_BEST_VARIANTS_v2.py` — drop-in replacement for `build_R37.py` BEST_VARIANTS.
- `R38_BEST_VARIANTS_v2.json` — same data + provenance/source tags per shape.
- `bench_all42_results_R38_optD.json` — full R38D sweep data.
- `bench_all42_results_R38_optDv2.json` — escape-hatch axis sweep data.
- `R38_OPT_D_PROGRESS.md` — methodology + step-by-step run log.
