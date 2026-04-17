# Round 15 Optimizer A — `regclassglob` cross-shape probe — VERDICT

**Flag tested:** `-mllvm -greedy-regclass-priority-trumps-globalness=true`
(R14A discovered a sub-threshold +0.24pp signal on P1 with this flag.)

**Method:** rebuild + fresh asm-diff hash + warmup=200 / iters=500 / trim=10%
single-shot smoke + 5-run verify on all surviving candidates. GPU isolation via
`HIP_VISIBLE_DEVICES`. Cross-GPU bias avoided by running parent + candidate
back-to-back on the same GPU within each thread.

## Phase A — `regclassglob` alone on the 4 deep-LOSE parents

ASM-diff (fresh hashes): all 4 shapes produce **DIFF** vs parent.
This contradicts the original R14A summary in the prompt that suggested DLA2/DLA7
were NOOP — the R14A `asm_diff_probe_r14a.json` itself shows DIFF too, so the
prompt note was misremembering. R15A confirms DIFF on all 4.

| Shape | M×N×K               | Parent suffix              | Smoke parent | Smoke cand | Δpp     | Gate (+0.5pp) |
|-------|---------------------|----------------------------|-------------:|-----------:|--------:|---------------|
| DLA1  | 4096×32768×128256   | `_ts_pf6_6_v12_memc`       |       5223.79|     3904.39| -22.823 | **fail**\*    |
| DLA2  | 128256×32768×4096   | `_ts_gm2_v12_memc_dc`      |       4211.73|     4208.91|  -0.062 | fail          |
| DLA7  | 28672×32768×4096    | `_ts_lgk2_v12_memc`        |       4226.53|     4225.66|  -0.019 | fail          |
| P1    | 28672×4096×16384    | `_ts_gm8`                  |        776.82|     5091.97| +81.835 | PASS\*        |

\*DLA1 candidate's smoke value of 3904 was a transient first-load artifact;
re-bench on a fresh GPU produced parent=5107.45 / cand=5114.02 (Δ+6.57 TFLOPS,
+0.11pp). Sub-threshold positive, no win.

\*P1 parent's smoke value of 776 was the initial-load artifact; the reading is
unreliable. Re-bench produced parent=3236.75 / cand=4956.65 (still huge first-run
parent variance — the P1 `_ts_gm8` parent is unusually unstable on first
measurements). 5-run verify (below) is the source of truth.

## Phase B — `regclassglob` × macro-tweak compounds on P1

ASM-diff: all 7 compounds DIFF vs `_ts_gm8` parent. `_noembed` was NOOP vs the
bare `regclassglob` variant (so the noembed macro itself is a no-op on P1 once
regclassglob is on); skipped from smoke.

| Compound suffix                             | Smoke parent | Smoke cand | Δpp     | Gate (+0.5pp) |
|---------------------------------------------|-------------:|-----------:|--------:|---------------|
| `_ts_gm8_r15a_regclassglob_v20`             |      4965.20 |    5013.78 | +0.921  | **PASS**      |
| `_ts_gm8_r15a_regclassglob_tv16`            |      5051.52 |    5090.42 | +0.738  | **PASS**      |
| `_ts_gm8_r15a_regclassglob_lgk2`            |      5061.28 |    5081.52 | +0.384  | fail (close)  |
| `_ts_gm8_r15a_regclassglob_tv0`             |      4959.34 |    4970.59 | +0.213  | fail          |
| `_ts_gm8_r15a_regclassglob_v24`             |      4959.08 |    4967.67 | +0.163  | fail          |
| `_ts_gm8_r15a_regclassglob_extbr`           |      4978.31 |    4896.97 | -1.543  | fail          |
| `_ts_gm8_r15a_regclassglob_noembed`         | (NOOP vs bare rcg — skipped)                    |

## 5-run verify (commit-gate)

Same GPU (1), warmup=200, iters=500, 5 baseline runs + 5 candidate runs after
discarding 1 warmup run each. Gate = (mean ≥ base.max) AND (mean Δ ≥ +1.0pp).

| Variant                         | Base mean | Cand mean | Δ TFLOPS | Δ pp    | mean≥max | ≥+1.0pp |
|---------------------------------|----------:|----------:|---------:|--------:|----------|---------|
| P1 `regclassglob` (bare)        |   4993.47 |   5005.89 |  +12.42  | +0.236  | FAIL     | FAIL    |
| P1c `regclassglob_v20`          |   4991.71 |   4993.34 |   +1.63  | +0.031  | FAIL     | FAIL    |
| P1c `regclassglob_tv16`         |   4993.75 |   5005.60 |  +11.85  | +0.225  | **PASS** | FAIL    |
| P1c `regclassglob_lgk2`         |   4996.10 |   4987.57 |   -8.53  | -0.162  | FAIL     | FAIL    |

## Conclusion

**0 / 11 candidates pass commit gates.** Negative finding — no kernel/binary
changes committed.

Sub-threshold positive movements worth recording for future kernel-source
work (matched R14A's signal but did not cross +1.0pp):
- **P1 bare `regclassglob`**: +0.236pp (R14A measured +0.24pp — replicates
  cleanly across rounds; reproducible signal, value too small for the gate).
- **P1c `regclassglob_tv16`**: +0.225pp with `mean ≥ base.max` PASS — close to a
  win in the noise, but still below the +1.0pp commit threshold. Potential
  micro-tweak target for an eventual kernel rewrite.

Notable single-shot smoke peaks that did NOT replicate in 5-run:
- `regclassglob_v20`: +0.92pp smoke → +0.03pp 5-run (smoke was noise)
- `regclassglob_lgk2`: +0.38pp smoke → -0.16pp 5-run (regression in 5-run)
- `regclassglob_extbr`: -1.54pp smoke (regression — already gated out)

The flag continues to register as a real-but-tiny perturbation on P1 (consistent
+0.2pp). No combination with the macro axis amplifies it past the noise.
This validates the prior MXFP4 24/42 ceiling finding: flag tuning is exhausted;
breakthrough requires kernel rewrite.

## Files committed (logs/probes only — no code/binary)

- `build_round15_optA_regclassglob.{py,log}`
- `asm_diff_probe_r15a.{py,log,json}`
- `bench_round15_optA_smoke.{py,log,json}`
- `bench_round15_optA_verify.{py,log,json}`
- `round15_optA_verdict.md`
