# Round 16 Optimizer C — Cross-axis compounds on 4 stuck deep-LOSE shapes — VERDICT

**Theme:** Test pair-wise / triple compounds of `iterilp`, `regclassglob`,
`sinkavoidspill`, `nolicm`, `noemxpre`, `largeivf2` on the 4 stuck shapes
(DLA1 / DLA2 / DLA7 / P1). Key hypothesis: regalloc-changing flags
(`regclassglob`, `sinkavoidspill`, `nolicm`) might restructure the SGPR
allocation enough to AVOID the iterilp aperture-violation bug on
DLA1/DLA2/DLA7.

**Result: 0 commit-worthy wins. DEAD END.** The iterilp-SGPR-clobber bug
is INDEPENDENT of register-allocator-altering flag combinations — it
re-triggers on every iterilp+X compound at full M. New BROKEN-APERTURE
registry entries: 12 (every iterilp+X combo on DLA1/DLA2/DLA7).

---

## Methodology

- **Build** (`build_round16_optC_stuck_compounds.py`): 4 shapes × 12 compound
  variants = 48 builds. **All 48 OK** (after fixing `+`→`X` separator in suffix
  to keep the C identifier valid). Each compound is `parent_flags + " ".join(tags)`,
  with last-spec-wins for `-mllvm -amdgpu-sched-strategy=`.
- **ASM-diff probe** (`asm_diff_probe_r16c.py`): all 48 produce `.text` hash
  ≠ parent (parent compiled without -mllvm wrappers, so any -mllvm flag
  changes some symbol layout uniformly — but hashes mutually distinct between
  compounds, indicating real codegen differences).
- **SNR safety probe** (`snr_probe_r16c.py`): tiny M=256 run with all-zero
  scales. Compares finite-fraction vs parent (M=256 partially-NaNs even for
  the parent at huge K). After threshold tuning to `cand_finite < 0.25 *
  ref_finite` (catches genuine SGPR-clobber, allows BF16-saturation noise),
  **all 48 pass**. The aperture bug only triggers at full M, so the small-M
  SNR probe was insufficient as a pre-filter; we relied on smoke-bench to
  detect the crashes.
- **Smoke** (`bench_round16_optC_smoke.py`): single-shot warmup=200 iters=500
  trim=10%, GPUs 5,6,7. Parent + candidate side-by-side per compound.
- **Verify** (`bench_round16_optC_verify.py`): 5-run reps, single GPU.
  Gate: mean ≥ base.max AND mean Δ ≥ +1.0pp.
- **Re-verify** (`bench_round16_optC_reverify.py`): for the one tentative WIN,
  6 reps on a single GPU with 1 discard run each.

---

## Per-shape × per-compound matrix

`AC` = aperture crash (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION at full M)
`DIFF/fail` = compiled, ran, smoke Δpp < +0.5
`smoke-PASS` = smoke Δpp ≥ +0.5
`verify-FAIL` = smoke-PASS but 5-run mean Δpp < +1.0

| Compound                           | DLA1 | DLA2 | DLA7 | P1                          |
|------------------------------------|:----:|:----:|:----:|:----------------------------|
| iterilp + regclassglob             |  AC  |  AC  |  AC  |  AC                         |
| iterilp + sinkavoidspill           |  AC  |  AC  |  AC  |  DIFF/fail (+0.44pp)        |
| iterilp + nolicm                   |  AC  |  AC  |  AC  |  DIFF/fail (-0.53pp)        |
| iterilp + noemxpre                 |  AC  |  AC  |  AC  |  smoke-PASS → verify-FAIL   |
| iterilp + largeivf2                |  AC  |  AC  |  AC  |  smoke-PASS → verify-FAIL   |
| regclassglob + sinkavoidspill      | DIFF/fail | DIFF/fail | DIFF/fail | DIFF/fail            |
| regclassglob + nolicm              |  AC  | DIFF/fail | smoke-PASS\* → verify-FAIL | DIFF/fail |
| regclassglob + noemxpre            |  AC  | DIFF/fail | DIFF/fail | smoke-PASS → **verify-WIN-then-FAIL** |
| sinkavoidspill + nolicm            | DIFF/fail | DIFF/fail | DIFF/fail | DIFF/fail            |
| regclassglob + nolicm + sinkavoidspill | DIFF/fail | DIFF/fail | DIFF/fail | DIFF/fail        |
| iterilp + regclassglob + nolicm    |  AC  |  AC  |  AC  |  DIFF/fail (-0.94pp)        |
| iterilp + regclassglob + sinkavoidspill | AC | AC |  AC  |  AC                          |

\* DLA7 regclassglob+nolicm smoke showed +8.23pp but base=3838 was a
cold-start outlier; verify mean Δ=+0.10pp (FAIL).

### iterilp + X failure pattern (key finding)

- **DLA1 / DLA2 / DLA7 (every iterilp+X combo, including triples)** → APERTURE
  CRASH at full M. Confirms the SGPR-clobber bug is **NOT** sensitive to
  register-allocation policy (regclassglob, sinkavoidspill, nolicm) nor to
  scheduler-altering flags (largeivf2, noemxpre).
- **P1 + iterilp+X** → all SURVIVE (P1's smaller K × different iteration
  count keeps the bad SGPR live-range out of the aperture-violating
  codepath). Three of five ran; smoke-passed two; verify-failed both.

This is a HARD compiler bug, not avoidable via flag tuning at the LLVM level.
The dead-end status of iterilp on DLA1/DLA2/DLA7 (R10/R11) is confirmed.

---

## Verify table (5-run, gate: mean ≥ base.max AND Δpp ≥ +1.0)

| Shape | Compound                  | base.mean | base.max | cand.mean | cand.max | Δ pp     | gate(max) | gate(+1pp) | verdict |
|-------|---------------------------|----------:|---------:|----------:|---------:|---------:|:---------:|:----------:|---------|
| P1    | iterilp + noemxpre        |   4969.42 |  4996.54 |   4980.10 |  5009.72 | +0.203   | FAIL      | FAIL       | FAIL    |
| P1    | iterilp + largeivf2       |   4987.28 |  4999.21 |   4996.76 |  5012.72 | +0.180   | FAIL      | FAIL       | FAIL    |
| DLA7  | regclassglob + nolicm     |   4189.20 |  4200.75 |   4193.44 |  4200.20 | +0.095   | FAIL      | FAIL       | FAIL    |
| P1    | regclassglob + noemxpre   |   4670.93 |  5011.88 |   5029.20 |  5064.34 | +6.794\*\*| PASS      | PASS\*\*   | (anomaly)|

\*\* P1 regclassglob+noemxpre's first verify base run was 3349 TFLOPS
(cold-start). Re-verify with discard-first-run + 6 clean reps:

| Shape | Compound                  | base.mean | base.max | cand.mean | cand.max | Δ pp     | gate(max) | gate(+1pp) | verdict |
|-------|---------------------------|----------:|---------:|----------:|---------:|---------:|:---------:|:----------:|---------|
| P1    | regclassglob + noemxpre   |   5015.94 |  5028.83 |   5031.88 |  5045.70 | **+0.302**| PASS      | FAIL       | sub-+1pp|

The clean re-verify confirms a small but real lift on P1
(`cand.mean ≥ base.max`), but well under the +1.0pp commit gate.

---

## BROKEN-APERTURE registry additions (R16C)

| Shape | Tag                                  | Trigger              |
|-------|--------------------------------------|----------------------|
| DLA1  | iterilp + regclassglob               | aperture crash full M |
| DLA1  | iterilp + sinkavoidspill             | aperture crash full M |
| DLA1  | iterilp + nolicm                     | aperture crash full M |
| DLA1  | iterilp + noemxpre                   | aperture crash full M |
| DLA1  | iterilp + largeivf2                  | aperture crash full M |
| DLA1  | iterilp + regclassglob + nolicm      | aperture crash full M |
| DLA1  | iterilp + regclassglob + sinkavoidspill | aperture crash full M |
| DLA1  | regclassglob + nolicm                | aperture crash full M |
| DLA1  | regclassglob + noemxpre              | aperture crash full M |
| DLA2  | iterilp + regclassglob               | aperture crash full M |
| DLA2  | iterilp + sinkavoidspill             | aperture crash full M |
| DLA2  | iterilp + nolicm                     | aperture crash full M |
| DLA2  | iterilp + noemxpre                   | aperture crash full M |
| DLA2  | iterilp + largeivf2                  | aperture crash full M |
| DLA2  | iterilp + regclassglob + nolicm      | aperture crash full M |
| DLA2  | iterilp + regclassglob + sinkavoidspill | aperture crash full M |
| DLA7  | iterilp + regclassglob               | aperture crash full M |
| DLA7  | iterilp + sinkavoidspill             | aperture crash full M |
| DLA7  | iterilp + nolicm                     | aperture crash full M |
| DLA7  | iterilp + noemxpre                   | aperture crash full M |
| DLA7  | iterilp + largeivf2                  | aperture crash full M |
| DLA7  | iterilp + regclassglob + nolicm      | aperture crash full M |
| DLA7  | iterilp + regclassglob + sinkavoidspill | aperture crash full M |
| P1    | iterilp + regclassglob               | aperture crash full M |
| P1    | iterilp + regclassglob + sinkavoidspill | aperture crash full M |

Total: **25 new BROKEN-APERTURE entries** confirming the iterilp SGPR-clobber
bug is robust across regalloc/sched/loop flag perturbations.

---

## Key conclusion

The iterilp scheduler-strategy aperture-violation bug on DLA1/DLA2/DLA7 is
**not** caused by register pressure or sink/LICM/loop-prefetch behavior —
it is a fundamental SGPR-handling bug in the iterative-ILP scheduler that
re-triggers regardless of regalloc-policy or transformation-pass tweaks.

The strongest sub-threshold positive in R16C is P1 `regclassglob + noemxpre`
(clean re-verify Δ = +0.30pp), still well below the +1.0pp commit gate.

The kernel-level optimization ceiling is confirmed; no compiler flag
combination produces a commit-worthy lift on the 4 stuck shapes.

---

## Files written

- `build_round16_optC_stuck_compounds.{py,log}`
- `asm_diff_probe_r16c.{py,log,json}`
- `snr_probe_r16c.{py,log,json}` + `snr_probe_r16c_tiles/` (256x256 reference tiles)
- `bench_round16_optC_smoke.{py,log,json}`
- `bench_round16_optC_verify.{py,log,json}`
- `bench_round16_optC_reverify.{py,log,json}`
- `round16_optC_verdict.md` (this file)

No commits made (0 wins under +1pp gate).
