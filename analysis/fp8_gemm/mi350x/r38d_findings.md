# R38 Dev D — Methodology hardening: G1' fallback orchestrate + nm-based dead-code gate

## Verdict: **METHODOLOGY SHIP** (R38+ priority items 3 + 4 both delivered as drop-in replacements)

## Summary

Two methodology improvements delivered as drop-in replacements for R36
Dev D's `r36_reviewer_orchestrate.sh` and the implicit md5 build-hygiene
rule:

1. **`r38_orchestrate.sh`** — adds the R37 NEW G1' fallback gate
   (`bench_mhz ≥ SCLK_GATE_MHZ AND median > G1P_MIN_MEDIAN_TF`) on top of
   the R36 G1+G2a+G2b ladder. Each accepted sample emits a JSON record
   with `gate_path` ∈ `{G1+G2a+G2b, G1'_fallback, EXHAUSTED}` so reviewers
   can audit which gate path each sample took.

2. **`r38_nm_gate.sh`** — verifies build hygiene structurally via
   `nm -D <so> | grep <feature_symbol> | wc -l == 0` for compile-flag-gated
   features in the default build. Replaces the R29 Dev C md5-based hygiene
   check, which R37 Dev C documented as unreliable in this env (consecutive
   identical hipcc → different md5).

## Task 1 deliverable: `r38_orchestrate.sh`

### Gate ladder (per (tag, rep) attempt)

```
1. Run bench. Compute G1, G2a, G2b, G1'.
2. If G1 AND G2a AND G2b      → ACCEPT  gate_path="G1+G2a+G2b"  (STRICT)
3. Else                        → RETRY up to MAX_RETRIES (default 3)
   - Track best G1' candidate (G2a AND G2b AND median > 500 TF)
4. After MAX_RETRIES:
   - If best G1' candidate exists → ACCEPT gate_path="G1'_fallback"
   - Else                          → EMIT  gate_path="EXHAUSTED"
```

This is strictly more permissive than R36 (every R36-acceptable sample is
R38-acceptable), but adds the G1' fallback path that R37 Dev A's recipe
proved necessary under 4-concurrent-agent host contention.

### G1' fallback validation

| scenario              | host          | samples taken | G1+G2a+G2b path | G1' fallback | EXHAUSTED |
|-----------------------|---------------|---------------|-----------------|---------------|-----------|
| Idle GPU4, no contention | clean       | 6 (3 prod, 3 base) | **6** | 0 | 0 |
| GPU4 same-GPU heavy contention (16K hgemm) | hostile | 8 attempts × 4 reps | 1 | 0 | 3 |
| Idle GPU4, raised SCLK_GATE_MHZ=2350 (synthetic G1 over-rejection) | demo | 2 reps | 0 | **2** | 0 |

Interpretation:
- **Clean idle**: every sample passes G1 primary. G1' is never needed. Same
  behavior as R36/R37.
- **Hostile contention**: G2b (CV ≤ 1%) is load-bearing — under same-GPU
  contention CV regularly exceeds 1% and G1' refuses to accept those
  samples (correct conservative behavior, not a regression vs R36).
- **Synthetic G1 over-rejection**: when bench actually runs at full sclk
  (≥ 2350 MHz post-bench) but pre-bench MHz reading happens to land below
  threshold (here forced via SCLK_GATE_MHZ=2350), G1' accepts the run.
  This recovers the "50% sample loss" pathology R37 Dev A observed under
  4-concurrent-agent host contention.

### Verification re-bench (R37 Dev A SHIP equivalence)

Re-benched R37 Dev A's HB shrink B1 70B-KV STRICT SHIP (commit `ab8a80f7`)
through the new orchestrate on GPU4. Used the exact same .so artifacts
that R37 Dev A used (md5 `ac1b12be9c499444ebbe988ca0bb1676` for prod,
`b912c51d878e7621d0d92f5de239db29` for baseline).

| GPU | prod n | prod median (TF) | base n | base median (TF) | Δ%      | Welch t |
|-----|--------|------------------|--------|------------------|---------|---------|
| 4   | 3      | **987.78**       | 3      | **766.82**       | **+28.82%** | **+21.78** |

Every sample took the **G1+G2a+G2b primary path** (no G1' fallback needed
on idle GPU4). Δ% +28.82% sits inside R37 Dev A's reported per-GPU spread
(GPU0/1/3/6: +25.13% to +31.12%, median-of-medians +30.39%) — the new
orchestrate produces equivalent numerics to R37 Dev A's orchestrate.
**SHIP equivalence: CONFIRMED.**

## Task 2 deliverable: `r38_nm_gate.sh`

### Symbol catalog (R38 NEW baseline)

Established on the existing R37 Dev A artifacts; the default 8192³ build
matches main HEAD by md5 `33b17d2c7e5990e559bc267f352c016b`.

| feature       | grep pattern                             | default 8192³ | 70B-KV baseline (no flags) | 70B-KV PROD (BLK_M=128 PIPE=1) |
|---------------|------------------------------------------|---------------|----------------------------|-----------------------------------|
| hbshrink (B1) | `hbshrink`                               | **0**         | 0                          | **4**                             |
| hbn (R38 A/B) | `\bhbn\b\|HBN_`                          | 0             | 0                          | 0                                 |
| subrbm        | `subrbm`                                 | 0             | 0                          | 0                                 |
| warpsm4       | `warpsm4`                                | 0             | 0                          | 0                                 |
| double_pump   | `double_pump`                            | 0             | 0                          | 0                                 |
| mxfp8_4wave   | `_4wave`                                 | 0             | 0                          | 0                                 |
| rect          | `_8wave_rect`                            | 0             | 0                          | 0                                 |
| rcr_v2 (rt-gated) | `dispatch_rcr_exact_8wave_scaled_v2` | 1 (present)   | 1 (present)                | 1 (present)                       |
| rrr_v2 (rt-gated) | `dispatch_rrr_exact_8wave_scaled_v2` | 1 (present)   | 1 (present)                | 1 (present)                       |
| crr_v2 (rt-gated) | `dispatch_crr_exact_8wave_scaled_v2` | 1 (present)   | 1 (present)                | 2 (default + hbshrink stub)       |

OVERALL gate result for default 8192³: **PASS** (every default-off feature
shows 0 symbols; every always-present runtime-gated dispatch shows ≥ 1).

### Why nm beats md5

R37 Dev C documented two consecutive identical `hipcc kernel_mxfp8_layouts.cpp`
invocations producing different md5 hashes. Likely cause: hipcc embeds
build IDs or non-deterministic codegen-pass ordering. Md5 therefore can
flag a "regression" when the source has not changed at all (false
positive) and conversely cannot prove dead-code elimination structurally.

The nm-gate is the *correct* invariant: a compile-flag-gated feature is
provably not invokable from runtime iff its kernel symbol is absent from
the dynamic symbol table. The check is also tractable to extend — adding
a new feature requires one entry in the `FEATURES` associative array.

### V2-RCR / V2-RRR predicate audit

R36 Dev C's V2-RCR predicates and R36 Dev B's V2-RRR predicate are
**runtime shape-gated, not compile-flag-gated**. Their kernel symbols are
always present in every build; the nm-gate verifies presence rather than
absence (`expected="present"`). This is a methodology distinction
documented in the script header — runtime predicates do not get an
"absent on default" guarantee.

### Usage

```bash
# Basic catalog check on default build:
./r38_nm_gate.sh /path/to/default.so

# Verify that hbshrink is ACTIVE in PROD build:
./r38_nm_gate.sh --expect-active hbshrink /path/to/prod_70bkv.so

# JSON output for CI integration:
./r38_nm_gate.sh --json /path/to/default.so
```

Exit code 0 if every catalog feature passes its `absent`/`present` expectation;
1 otherwise.

## Proposed methodology rule diff for `TODO.md`

Replace the R37+ priority list item 6 ("methodology — R38+ rules") block
with this expanded version. **NOT applied to main TODO.md per instructions
— diff documented here only.**

```diff
 6. **【methodology — R38+ rules, MUST follow】**:
    - All R29-R36 rules carry forward.
-   - **R37 NEW (recommended)**: orchestrate fallback filter `bench_mhz ≥ 2200 AND median > 500 TF` for high-contention host runs (Dev A's recipe, used to recover Dev A's 4-GPU triangulation when R36 G1 dropped 50% of samples).
-   - **R37 NEW (recommended)**: `nm`-based dead-code gate for compile-flag-gated features when md5 build hygiene is unreliable.
+   - **R38 NEW (mandatory)**: use `r38_orchestrate.sh` — drop-in replacement for `r36_reviewer_orchestrate.sh` / `r37a_orchestrate.sh`. Implements R36 G1+G2a+G2b primary ladder PLUS R37 NEW G1' fallback (`bench_mhz ≥ 2200 AND median > 500 TF`). Each accepted sample emits a `gate_path` JSON record (`{G1+G2a+G2b, G1'_fallback, EXHAUSTED}`) for reviewer audit.
+   - **R38 NEW (mandatory)**: use `r38_nm_gate.sh` — drop-in replacement for md5 build-hygiene check. Verifies `nm -D <so> | grep <feature> == 0` for default-off compile-flag-gated features (hbshrink, hbn, subrbm, warpsm4, double_pump, _4wave, _8wave_rect) and presence for runtime-gated dispatches (rcr_v2, rrr_v2, crr_v2). Md5 still logged for traceability but no longer load-bearing.
+   - SHIP claims must include both the orchestrate's gate-path histogram AND the nm-gate result for the prod .so.
```

## Followups / R39 candidates

1. Wire `r38_nm_gate.sh` into `r38_orchestrate.sh` build step (post-build
   hook). Currently the two scripts are independent — the orchestrate
   doesn't currently rebuild the .so, so nm-gate isn't wired in
   automatically. R39 candidate: a unified driver that builds, nm-gates,
   then benches.
2. The G1' fallback is currently hard-coded to `median > 500 TF`. For
   shapes where 500 TF is not the right floor (e.g. small shapes that
   only achieve 200-400 TF), `G1P_MIN_MEDIAN_TF` is parameterized but
   per-cell defaults would be cleaner.
3. The `hbn` catalog entry is a placeholder for R38 Dev A/B's HBN-shrink
   variant (in flight). When that lands, the catalog auto-detects the
   feature; no script change needed.

## Artifacts

- New orchestrate: `analysis/fp8_gemm/mi350x/r38_orchestrate.sh`
- New nm-gate:     `analysis/fp8_gemm/mi350x/r38_nm_gate.sh`
- Verify re-bench (clean): `analysis/fp8_gemm/mi350x/r38d_verify/`
- Contention probe:        `analysis/fp8_gemm/mi350x/r38d_verify_contended/`
- G1' fallback demo:       `analysis/fp8_gemm/mi350x/r38d_g1p_demo/`
- This file:               `analysis/fp8_gemm/mi350x/r38d_findings.md`
