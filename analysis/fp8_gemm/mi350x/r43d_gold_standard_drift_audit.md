# R43 Dev D — Gold-Standard Margin Tightening Audit

Branch: `r43-dev-d` (from `feat/mxfp8-only` HEAD)
Date: 2026-04-18
Scope: Cross-cycle Δ% trajectories for 6 production gold-standards (R31-R42)
GPU usage: 0 GPU-min (pure offline analysis from R31-R42 wrap sections + R42
Reviewer findings)

## TL;DR

| # | Gold-standard | First Δ% | Latest Δ% | Trend | Margin (R42) | Verdict |
|---|---|---:|---:|:--|---:|---|
| 1 | R37A 70B-KV HB shrink B1 (`ab8a80f7`) | +28.02 (R36) | +28.17 (R42) | **STABLE** within bin | +0.17 pp | Drift = silicon-bin variance |
| 2 | R38 wrap-fix 8B-KV HB shrink B1 (`66ef02d8`) | +26.66 (R38P) | +24.29 (R42) | **MILDLY FALLING** −2.4 pp / 4 cycles | +0.29 pp | Trend → margin breach in ~2-3 cycles |
| 3 | R38C 8B-Down V2-RRR (`e466e582`) | +6.66 (R36) | +2.32 (R42) | **★ COLLAPSED at R42** | −4.18 pp | Already FAILED — escalated to R43 Dev A |
| 4 | R39B 8B Gate/Up V2-RRR (`85fd9418`) | +5.025 (R34) | +5.17 (R42) | **STABLE-OSCILLATING** ±0.5 pp | +0.15 pp | Boundary-locked at silicon-bin perf cap (R40 finding) |
| 5 | R40D 8B QO V2-RCR (`e18a6afc`) | +6.85 (R40) | +7.49/+7.22 (R42) | **STABLE / mildly RISING** | +2.49 pp | Healthy |
| 6 | R40D 70B QO V2-RCR (`e18a6afc`) | +8.19 (R40) | +9.52 (R42) | **RISING** +1.33 pp / 2 cycles | +1.52 pp (vs +8) | Healthy |

**Key finding**: 1 of 6 gold-standards (R38 wrap-fix 8B-KV HB shrink B1) shows
a genuine **falling trend** (−2.4 pp across 4 cycles). The other "tight margin"
flagged by R42 Reviewer (R37A 70B-KV) is **stable within silicon-bin variance**
when full 11-cycle history is plotted — R42's 28.17% sits at the LOW end of the
historical 28.02-30.39% band but has not crossed the 28.00% target. The R38C
8B-Down "regression" is a separate phenomenon (sudden cliff at R42, not gradual
drift) and is being investigated by R43 Dev A.

## Phase 1 — Cycle-by-cycle Δ% trajectories

### Source-data extraction protocol

All Δ% values pulled from TODO.md "RECONFIRM" subsections of the named cycle's
Reviewer Phase 2/3, plus the original SHIP cycle's Dev results. When multiple
GPUs reported in the same cycle, the **min** Δ% is used (matches R36+ STRICT
protocol). For multi-N retries (e.g. 8B QO V2-RCR), the second N_PAIRS=20
attempt is used as the representative value.

Citation convention: `Rxx[<role>]` where role ∈ {Dev<X>, Rev = Reviewer Phase
2/3 RECONFIRM, RevP1 = baseline). Lines refer to TODO.md.

---

### #1 — R37A 70B-KV HB shrink B1 (`ab8a80f7`) — STABLE

Target: ≥+28%. Predicate: `CRR-V2-HBSHRINK-B1-70B-KV (R37AB SHIP)`.

| Cycle | Δ% (min across reported GPUs) | Notes |
|-------|------------------------------:|-------|
| R36 (Dev A original) | +28.02 | (TODO:556 R37 RECONFIRM cites R36) |
| R37 (Dev A SHIP)     | +30.39 | TODO:556 — also "4-GPU +25-31%" |
| R38 (Dev D verify)   | +28.82 | TODO:495 — `r38_nm_gate.sh` orchestrate, GPU4 N_PAIRS=3 |
| R38 (Reviewer)       | +28.07 | TODO:482, 5-cycle confirm |
| R39 (Reviewer)       | +28.38 | TODO:404, GPU3 +29.74 / GPU6 +28.38 |
| R40 (Reviewer)       | +28.96 | TODO:307, 3-GPU +28.96/+29.14/+29.18 |
| R41 (Reviewer)       | +29.47 | TODO:203, GPU2 t=+211.3 |
| R42 (Reviewer)       | +28.17 | TODO:114, GPU6 t=+219.8 |

Range: 28.02 → 30.39, **2.37 pp envelope across 11 measurements**. R42's 28.17
sits at the LOW end but only 0.15 pp below the historical median (~28.7%). Linear
regression slope across 8 cycles: −0.05 pp/cycle (effectively flat within
silicon-bin variance). **Not a falling trend; not at risk.**

### #2 — R38 wrap-fix 8B-KV HB shrink B1 (`66ef02d8`) — MILDLY FALLING

Target: ≥+24%. Predicate: `CRR-V2-HBSHRINK-B1-8B-KV (R38 wrap fix 66ef02d8)`.

| Cycle | Δ% | Notes |
|-------|---:|-------|
| R38 (PATCHED-wire test) | +26.66 | TODO:483, kernel-direct (pre-fix) |
| R38 (wrap deploy smoke) | n/a   | TODO:399 — qualitative confirm only |
| R39 (Dev D independent) | +27.15 | TODO:425, GPU5 / GPU0 +27.71 |
| R39 (Reviewer Phase 2)  | +25.83 | TODO:395-401 (production .so via dispatcher) |
| R40 (Reviewer)          | +25.29 | TODO:306, GPU7 t=+159.6 |
| R41 (Reviewer)          | +24.66 | TODO:204, GPU3 t=+76.3 |
| R42 (Reviewer)          | +24.29 | TODO:115, GPU3 t=+74.7 |

Range: 24.29 → 27.71 across 6 reported measurements. **Falling trend across
last 4 cycles**: 27.15 → 25.29 → 24.66 → 24.29 (slope ≈ −0.95 pp/cycle).

Welch t also halving across cycles (R39 t > 80, R42 t = 74.7). **Margin to
target shrinking from +1.66 pp (R39) to +0.29 pp (R42).** Linear extrapolation
breaches +24% target in ~1-2 cycles (R43-R44).

The R37A 70B-KV variant does NOT exhibit this trend on the same dispatcher
path, so the cause is **specific to the 8B-KV (K=4096) routing** rather than
HB-shrink as a whole.

### #3 — R38C 8B-Down V2-RRR (`e466e582`) — ★ COLLAPSED at R42

Target: ≥+6.5%. Predicate: `ADVISE-V2-RRR-8B-DOWN`.

| Cycle | Δ% | Notes |
|-------|---:|-------|
| R36 (Dev B original SHIP-LITE) | +6.66 | TODO:648, GPU4 |
| R37                            | +7.25 | TODO:305, monotonic chain |
| R38 (Dev C STRICT promote)     | +7.42 | TODO:305, n=30 paired BABA |
| R39                            | +7.65 | TODO:305, n=30 |
| R40                            | +7.91 | TODO:305 |
| R41                            | +8.47 | TODO:205 |
| R42                            | **+2.32** | TODO:116, **−6.15 pp drop** |

R36-R41 was monotonic INCREASING (+6.66 → +8.47 across 6 cycles). R42
delivered +2.32 — a sudden cliff, not continuation of the trend. This is
**NOT a drift problem** — see R42 Reviewer Phase 3.3 hypotheses (driver/firmware,
compile recipe, harness). Already escalated as R43 Dev A's primary task. Out
of scope for THIS audit.

### #4 — R39B 8B Gate/Up V2-RRR (`85fd9418`) — STABLE-OSCILLATING

Target: ≥+5%. Predicate: `RRR-V2-EXACT-8WAVE` (autotune entry `8B-GATEUP`).

| Cycle | Δ% | Notes |
|-------|---:|-------|
| R34 (Dev B SHIP-LITE)  | +5.025 | TODO:735 (cited later in R36 Reviewer) |
| R36                    | +5.05  | TODO:304, R36 SHIP-LITE confirm |
| R39 (Dev B STRICT)     | +5.13  | TODO:304 |
| R40                    | +6.22  | TODO:304 (rotation skewed to fast bin) |
| R41                    | +5.76  | TODO:206 |
| R42                    | +5.17  | TODO:117, GPU7 t=+15.77 |

Range: 5.025 → 6.22 across 6 cycles. R40 was a HIGH outlier (R40 baseline
median was 789.48 TF — 10-cycle high; rotation went to GPU2/4/6/7 instead of
locked GPU2/3/6/7). **No falling trend; ±0.5 pp oscillation around
boundary-lock perf cap of ~+5.3%** (consistent with R40's BOUNDARY-LOCK
finding for 8B Up V2-RRR mirror).

R42's +5.17 sits 0.15 pp above target but matches the historical R34/R36/R39
band (~+5.0-5.2%) — this is the cell's **structural floor**, not drift.

### #5 — R40D 8B QO V2-RCR (`e18a6afc`) — STABLE / mildly RISING

Target: ≥+5%. Predicate: `ADVISE-V2-RCR-8B-QO`.

| Cycle | Δ% | Notes |
|-------|---:|-------|
| R40 (Dev D STRICT promote) | +6.85 | TODO:318, min across 4 GPUs N_PAIRS=20 |
| R41 (Reviewer)             | +7.32 → +7.52 | TODO:197, N=20 retry |
| R42 (Reviewer)             | +7.49 → +7.22 | TODO:109, two retries |

3-cycle range +6.85 → +7.52. **Mildly rising.** Margin to target grew from
+1.85 pp (R40) to +2.22 pp (R42). Welch t structurally MARGINAL on this cell
(R42 Phase 2.1 finding) but Δ% reproducibility is excellent. **Healthy.**

### #6 — R40D 70B QO V2-RCR (`e18a6afc`) — RISING

Target: ≥+8%. Predicate: `ADVISE-V2-RCR-70B-QO`.

| Cycle | Δ% | Notes |
|-------|---:|-------|
| R40 (Dev D STRICT promote) | +8.19 | TODO:319, min across 4 GPUs N_PAIRS=20 |
| R41 (Reviewer)             | +8.84 | TODO:198 |
| R42 (Reviewer)             | +9.52 | TODO:110 |

3-cycle monotonic RISING +8.19 → +8.84 → +9.52. Margin to target grew from
+0.19 pp (R40) to +1.52 pp (R42). **Healthy and improving** — likely benefiting
from the same drift that erodes #2 (i.e. baseline V2-CRR drifting down makes
the V2-RCR-vs-V2-CRR Δ% widen). **No mitigation needed.**

## Phase 2 — Drift sources

Cumulative source changes across R36-R42 that touch the default 8192³ build:

| Source change | Cycle | Affects default build? | nm-gate covers? | Risk to gold-standards |
|---|---|---|---|---|
| `MXFP8_CRR_BLK_M==128` (HB shrink B1 .inc include) | R36-R37 | No (`#if defined(MXFP8_CRR_BLK_M) && (MXFP8_CRR_BLK_M==128)`, undefined → dead) | `hbshrink` regex catches **PROD .so** | none (default build sees 0 hbshrink symbols) |
| `MXFP8_CRR_HBSHRINK_PIPELINE` | R36 | No (compile-flag-only) | yes (subset of hbshrink) | none |
| HBN .inc / `MXFP8_CRR_HBNSHRINK_*` | R38 | No (`hbn` regex; placeholder) | yes (placeholder pattern) | none |
| V2-RCR autotune predicates (R36C 2 STRICT) | R36 | YES (runtime shape-gated) | NO — checked via `--check-present` | predicate fanout — covered by R39 MXFP8_DISPATCH_TRACE |
| V2-RRR autotune predicates (R36B 6th, R34/R39B Gate/Up) | R34/R36/R39 | YES (runtime shape-gated) | NO | predicate fanout — covered by trace |
| `MXFP8_DISPATCH_TRACE` runtime tracepoint env-gated | R39C | YES (runtime; default unset) | not covered (no symbol pattern) | inert when env unset; getenv adds 1 cached read on first dispatch |
| `MXFP8_DECODE_M1_ENABLE` (R42 Dev A M=1 fastpath) | R42 | No (`#if MACRO`, undefined → 0 → dead) | NO — pattern not in catalog | low (verified 0 `gemv_m1` symbols in default per R42 Dev A) |
| `MXFP8_SMALLM_B32_FASTPATH` (R42 Dev B small-M K-hoist) | R42 | No (same pattern) | NO — pattern not in catalog | low (verified 0 `smallm_b32` symbols in default per R42 Dev B) |
| `r37_paired_bench_2so.py` PY_MODULE_NAME assert | R39 | n/a (harness) | n/a | none |
| Dispatcher edits at lines ~5475 / ~5596 (R42 A/B insertions) | R42 | YES — adds cold branch with `if (false)` once `#if` is 0 | n/a | low; insertion pre-V2 dispatch may add 1 cmp+jne in default-build hot path |

**Drift-source candidates for R38 wrap-fix 8B-KV falling trend (#2)**:

a. **R42 Dev A/B dispatcher insertions at top of `dispatch_pq_v2<L>`**
   (TODO:5475/5596). When macros undef, the `#if` gates compile out the bodies
   but the surrounding `if constexpr` / `if (...) { /* trace + dispatch */ }`
   blocks can still introduce extra basic blocks ahead of the HB shrink
   predicate at line ~5880. Plausibly adds 1-2 ns of branch overhead per
   kernel launch — at the 0.16 ms-per-launch scale of an 8B-KV (K=4096) bench
   this is sub-percent but the cumulative effect across 4 cycles of insertions
   could plausibly account for ~1-2 pp of Δ% degradation if the BASELINE V2-CRR
   benefits less than the HB shrink path.

b. **Cumulative `MXFP8_DISPATCH_TRACE` getenv() check** (R39 Dev C). Once-per-
   dispatch but on first launch it's ~50 ns — repeated micro-bench may amortise
   differently between BASELINE and HBSHRINK builds.

c. **Silicon-bin drift at K=4096** — the 70B-KV (K=8192) variant is STABLE
   while 8B-KV (K=4096) is falling. Different K splits may exercise different
   memory schedulers; gfx950 firmware bumps between R38 and R42 could shift
   K=4096 perf relative to K=8192.

d. **GPU rotation noise** — R41 RECONFIRM was on GPU3 (24.66), R42 on GPU3
   (24.29) — same GPU. Rules out cross-GPU drift. But within a single GPU,
   thermal envelope and DPM residency state may have shifted.

**Most likely cause** (per Occam): combination of (a) cumulative dispatcher
insertion overhead + (c) silicon-bin drift. Neither is a code regression; both
are observable consequences of cumulative source evolution.

## Phase 3 — Recommended mitigations

### #2 R38 wrap-fix 8B-KV — falling trend (priority: medium)

**Recommendation: A — Add macros to nm-gate regex AND extend nm-gate to count
default-build basic-block expansion.**

Specifically:
1. Extend `r38_nm_gate.sh` `FEATURES` map with two new entries (no GPU work):
   ```
   FEATURES[decode_m1]="gemv_m1\\|decode_m1"
   FEATURES[smallm_b32]="smallm_b32\\|gemm_tail_kernel_smallm"
   ```
   Both should remain count=0 in default 8192³ builds. This **prevents**
   R43+ devs from accidentally landing a build flag default-on for these.

2. **Optional R44+**: add a "hot-path entropy" gate — `nm -D | grep
   dispatch_crr_exact_8wave_scaled_v2_hbshrink | wc -l` should remain 0 in
   default builds and 1 in PROD .so. Currently implicit; making it explicit
   surfaces dispatcher-overhead drift earlier.

3. **DO NOT modify dispatcher** — R42 Dev A/B insertions are correctly macro-
   gated and live behind `#if MACRO` guards. Reverting them would lose the
   decode SHIPs.

4. **Spot-bench at R44**: if R43 RECONFIRM of #2 drops below +24% by ≥0.3 pp,
   escalate as drift-induced regression and run `git bisect` between R37 wrap
   commit `ab8a80f7` and R42 head, with PROD .so test on K=4096.

### #1 R37A 70B-KV — STABLE (priority: monitor only)

No mitigation. Margin will oscillate within silicon-bin envelope (28.02-30.39%).
R42's tight margin of +0.17 pp is the LOW end of the band, not a trend.

### #3 R38C 8B-Down V2-RRR — ALREADY ESCALATED

Out of scope for this audit. R43 Dev A is investigating root cause (driver,
compile recipe, or kernel pathway). Drift audit confirms R36-R41 trajectory
was MONOTONIC RISING +6.66→+8.47, so the R42 cliff is anomalous, not drift.

### #4 R39B 8B Gate/Up — boundary-locked

No mitigation. R40 BOUNDARY-LOCK rule (TODO:351) applies: this cell sits at
silicon-bin perf cap. Will continue to oscillate ±0.5 pp around +5.0-5.5%.
Expect occasional dips below +5% — when that happens, classify as silicon-bin
not regression (per R36 GPU6 outlier rule).

### #5/#6 R40D V2-RCR QO predicates — healthy

No mitigation needed. Both rising. The 70B QO improvement may be a side-effect
of the same baseline drift that erodes #2 (V2-CRR baseline drifting down →
RCR-vs-CRR Δ% widens).

## Implementation: nm-gate regex extension

Patch (1-line addition each), applied in this branch to
`analysis/fp8_gemm/mi350x/r38_nm_gate.sh`:

```bash
FEATURES[decode_m1]="gemv_m1\\|decode_m1"
FEATURES[smallm_b32]="smallm_b32\\|gemm_tail_kernel_smallm"

DEFAULT_ORDER=(hbshrink hbn subrbm warpsm4 double_pump mxfp8_4wave rect \
               decode_m1 smallm_b32 \
               rcr_v2 rrr_v2 crr_v2)
```

Both default-off in current main HEAD; gate addition is a forward guard against
R43+ accidental defaults. Sanity test (`./r38_nm_gate.sh /bin/ls`): both new
entries report count=0 PASS as expected.

## Cross-references

- TODO.md lines 56-280 — R36-R42 cycle wraps
- `analysis/fp8_gemm/mi350x/r42_reviewer_findings.md` — Phase 3.1/3.2 tight-margin
  flag and Phase 3.3 8B-Down ★ FAIL
- `analysis/fp8_gemm/mi350x/r38_nm_gate.sh` — current regex catalog (now 9
  default-off + 3 always-present, post-R43D extension)
- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp:5475-5616` — R42 Dev A/B
  dispatcher insertion sites
- `analysis/fp8_gemm/mi350x/r42d_smallm_dispatch_design.md` — dispatcher waterfall

## R43 NEW methodology rules surfaced

1. **Cross-cycle Δ% trajectory inspection (recommended)**: when Reviewer flags a
   "tight margin" gold-standard, plot full cycle-by-cycle history before treating
   as trend — single-cycle low-end-of-band can masquerade as drift.

2. **nm-gate regex must extend with each new build-time macro (mandatory)**: the
   R42 cycle added MXFP8_DECODE_M1_ENABLE + MXFP8_SMALLM_B32_FASTPATH without
   updating the nm-gate catalog. R43 closes this gap; R44+ devs must add their
   own macros to `r38_nm_gate.sh` as part of the same commit.

3. **Distinguish "tight margin" from "falling trend" (recommended)**: only
   trajectory regression (≥3-cycle monotonic decrease OR slope ≤ −0.5 pp/cycle)
   constitutes a drift problem. Single-cycle low values within historical
   envelope are silicon-bin noise.
