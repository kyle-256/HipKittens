# R44 Dev D — R36 3-gate retry harness migration audit (R44+ priority #6)

Date: 2026-04-18
Branch base: HEAD = `51a7c759` (R43 cycle wrap)
Author: R44 Dev D

## Summary

R43 NEW methodology rule 1 (MANDATORY) requires every paired-bench harness
to enforce the R36 3-gate retry pattern:
- **G1**  `sclk-post-preheat` ≥ 2200 MHz (R34) with up to 3 retries
- **G2a** `sclk-post-bench`   ≥ 2200 MHz (R35)
- **G2b** per-run `stdev/mean` ≤ 1%      (R35)

The reference compliant implementation is
`analysis/fp8_gemm/mi350x/r38c_8bdown_orchestrate.sh:79-140`.

**Audit scope**: every invocation of `r37_paired_bench_2so.py` (the
two-`.so` paired-BABA harness referenced in the rule).  In addition I
spot-checked the gold-standard single-`.so` paired-bench orchestrate
scripts (which use `r33c_paired_bench.py`) since they share the same
power-throttle failure mode.

**Result**: 6 invocation sites of `r37_paired_bench_2so.py`; all 6 are
**non-compliant** in the source tree at HEAD `51a7c759`.  3 are active
reviewer harnesses → patched in this commit.  3 are historical
single-cycle scripts → marked DEPRECATED with header notes pointing
callers to the patched R43 reviewer phase script.

The 5 active gold-standard orchestrate scripts that drive 70B-KV /
8B-KV / 8B-Down / 8B-Gate / 8B-Up / QO already implement the 3-gate
retry correctly (verified by source inspection).

## Audit table — `r37_paired_bench_2so.py` invocation sites

| # | File | Line | Has G1? | Has G2a? | Has G2b? | Verdict | R44 Action |
|---|------|------|---------|----------|----------|---------|------------|
| 1 | `r37_reviewer_ship_verify.sh` | 142 | NO | NO | NO | NONE | DEPRECATED — header note added; pre-R38 workflow, not on the active reviewer path |
| 2 | `r38_reviewer_ship_verify.sh` | 158 | NO | NO | NO | NONE | DEPRECATED — header note added; pre-R40 workflow, replaced by `r40_reviewer_phase2_pair.sh` |
| 3 | `r39_reviewer_ship_verify.sh` | 158 | NO | NO | NO | NONE | DEPRECATED — header note added; **R39 findings.md misleadingly claims "R36 NEW 3-gate" but the script body has no retry / no gate parsing** (audit finding) |
| 4 | `r40_reviewer_phase2_pair.sh`  |  44 | NO | NO | NO | NONE | **PATCH-NOW** — likely re-used by R44+ reviewer phases under similar name; wrapped in retry loop |
| 5 | `r42_reviewer_phase23.sh`      |  41 | NO | NO | NO | NONE | **PATCH-NOW** — directly responsible for R42 P3.3 false-positive (~1 day escalation) |
| 6 | `r43_reviewer_phase23.sh`      |  41 | NO | NO | NO | NONE | **PATCH-NOW** — most-recent reviewer template; will be base for R44+ reviewer phases |

### Verdict count
- HAS-3-GATE-RETRY: 0/6
- MISSING-G1+G2a+G2b (NONE): 6/6
- PATCH-NOW: 3 (r40 phase2, r42 phase23, r43 phase23)
- DEPRECATED: 3 (r37/r38/r39 ship_verify)
- OUT-OF-SCOPE: 0

## Cross-check — gold-standard single-.so orchestrate scripts

These call `r33c_paired_bench.py` (the single-`.so` layout-flip paired
bench).  Same power-throttle failure mode applies, so the 3-gate retry
should also be present.  All 5 are compliant per source inspection
(grep for `MAX_RETRIES`, `sclk-post-preheat`, `sclk-post-bench`,
`STDEV_MEAN_GATE`, `while.*attempt`):

| File | 3-gate retry present? | Notes |
|------|-----------------------|-------|
| `r38c_8bdown_orchestrate.sh`         | YES | Reference impl; 8B-Down V2-RRR gold-standard |
| `r37d_8bdown_orchestrate.sh`         | YES | Predecessor of r38c |
| `r39_reviewer_8bdown_orchestrate.sh` | YES | 8B-Down reviewer re-bench |
| `r39b_8bgate_orchestrate.sh`         | YES | 8B Gate/Up V2-RRR gold-standard |
| `r40b_8bup_orchestrate.sh`           | YES | 8B Up V2-RRR gold-standard |
| `r40d_qo_orchestrate.sh`             | YES | QO V2-RCR gold-standard |

No patches needed for gold-standard runners.

## Patches applied

### Patch 1 — `r43_reviewer_phase23.sh`
Wrapped both `BENCH_KIND=two_so` (`r37_paired_bench_2so.py`) and
`BENCH_KIND=one_so_layout` (`r33c_paired_bench.py`) invocations in a
shared `run_bench_attempt` shell helper that:
1. Runs the bench with attempt-suffixed output paths.
2. Parses `[sclk-post-preheat]` / `[sclk-post-bench]` lines (Mhz captured
   via `sed -nE 's/.*\(([0-9]+)Mhz\).*/\1/p'`).
3. Parses median/mean/stdev for both bench arms via permissive regex
   (`^[A-Za-z_]+ +median=`) — handles both `r37_paired_bench_2so.py`'s
   `CRR_DEFAULT/CRR_HBSHRINK` labels and `r33c_paired_bench.py`'s
   user-supplied `LAYOUT_A/LAYOUT_B` labels.
4. Computes `PASS = (rc==0) AND (G1_pass) AND (G2a_pass OR G2b_pass)`,
   matching the r38c reference logic.
5. Outer `while [ $attempt -le $MAX_RETRIES ]` loop with 5s sleep on fail.
6. On PASS, copies the attempt's output to both `_clean.txt` and the
   legacy `_bench.txt` paths so downstream parsers don't break.
7. On exhaustion, exits 2 (visible failure) so reviewer scripts cannot
   accidentally consume throttled data.

Defaults match r38c: `SCLK_GATE_MHZ=2200`, `SCLK_POSTBENCH_GATE_MHZ=2200`,
`STDEV_MEAN_GATE=0.01`, `MAX_RETRIES=3`.

### Patch 2 — `r42_reviewer_phase23.sh`
Same patch as #1 (this script was the source of the R42 P3.3 false
positive — patched retroactively for any future re-runs).

### Patch 3 — `r40_reviewer_phase2_pair.sh`
Same patch as #1, with output filename pattern `${CELL}_${LABEL}_…`.

### Patch 4–6 — DEPRECATED header notes
Added a 4-line comment block above the `python3 r37_paired_bench_2so.py`
invocation in each of:
- `r37_reviewer_ship_verify.sh:142`
- `r38_reviewer_ship_verify.sh:158`
- `r39_reviewer_ship_verify.sh:158`
pointing future callers to `r43_reviewer_phase23.sh`.

For the r39 script, the comment also flags the
**audit finding**: `r39_reviewer_findings.md:177` claims this script
implements "R36 NEW 3-gate" but the source has no retry loop / no gate
parsing.  The historical R39 reviewer log appears to have relied on the
underlying bench script printing the gate values without enforcing them.

## Validation

- All 6 patched/edited files: `bash -n` syntax check PASS.
- 0 GPU consumed (pure source-edit task).
- No build artifacts touched.
- Default 8192³ build invariance: N/A — only shell scripts touched.

## R44+ recommendations

1. **Reviewer-template lock**: any R44/R45+ reviewer phase script should
   start as a copy of the patched `r43_reviewer_phase23.sh`.  Inheriting
   the `run_bench_attempt` helper guarantees rule-1 compliance.
2. **CI gate (suggested, not implemented this cycle)**: a one-line
   `grep -L "while.*attempt.*MAX_RETRIES" analysis/fp8_gemm/mi350x/*.sh`
   could be added to `r38_nm_gate.sh` or a new pre-commit hook to flag
   any new harness without retry.
3. **r37_paired_bench_2so.py output format alignment**: consider
   renaming `CRR_DEFAULT` / `CRR_HBSHRINK` to also use the
   `LAYOUT_A LAYOUT_B`-style template-able labels so the gate parsers
   in patched orchestrate scripts can use the strict
   `^${LAYOUT_A} median=` regex instead of the more permissive
   `^[A-Za-z_]+ +median=`.  Defer to R45+.
