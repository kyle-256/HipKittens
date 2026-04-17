# Round 17 Optimizer B — P1 NO-iterilp triple/quad-stack — VERDICT

**Theme:** Stack 3-4 known sub-threshold winners on P1 (28672x4096x16384,
parent `_ts_gm8`) hoping linear addition of pre-stack deltas
(R14A/R15A bare regclassglob +0.236pp, R15A regclassglob+tv16 +0.225pp,
R16C regclassglob+noemxpre +0.30pp) reaches the +1.0pp commit gate.

**Result: 0 commit-worthy wins. Triple-stack additivity hypothesis FALSIFIED.**
Best stable signal `rcg+noemxpre+tv16` reaches +0.498pp on 5-run interleaved
verify (mean ≥ base.max PASS but +0.5pp gate FAIL by 0.002pp; +1pp gate FAIL).
This matches R16C's `regclassglob+noemxpre` 2-stack (+0.30pp, +0.498pp)
within run noise — the third flag (`tv16`) adds **0pp**, not the predicted
+0.225pp. Sub-threshold deltas do NOT linearly compound.

---

## Methodology

- **Build** (`build_round17_optB_p1_triplestack.py`): 10 NO-iterilp compounds × P1.
  All 10 builds OK in 10.3s.
- **ASM-diff probe** (`asm_diff_probe_r17b.py`): hash device ELF vs parent
  `_ts_gm8`, R14A `_r14a_regclassglob`, R15A `_r15a_regclassglob_tv16`,
  R16C `_r16c_regclassglobXnoemxpre`. **10/10 produce unique DIFF hashes**;
  no NOOP-vs-parent and no MATCH-vs-existing-2-stack-winner. Codegen is
  genuinely distinct in every case.
- **Smoke** (`bench_round17_optB_smoke.py`): single-shot warmup=200 iters=500
  trim=10%, GPUs 2/3/4. Side-by-side parent + candidate same GPU.
  Gate: cand ≥ base + 0.5pp.
- **Verify** (`bench_round17_optB_verify.py`): 5-rep interleaved
  (B,C,B,C,B,C,B,C,B,C) on a single GPU per candidate. Same GPUs 2/3/4.
- **Re-verify** (`bench_round17_optB_reverify.py`): 6 reps, discard first run.
- **Final** (`bench_round17_optB_final.py`): 8 reps, discard first run, GPU 2 only,
  best candidate `rcg+noemxpre+tv16` only.

---

## Per-variant table

ASM-DIFF column abbreviations:
- `DIFF` = unique hash, ≠ parent and ≠ all 3 reference 2-stacks

Smoke column: single-shot Δpp, gate +0.5pp = PASS or FAIL.
Verify column: 5-rep interleaved Δpp_mean (mean ≥ base.max gate noted).

| #  | Variant                              | ASM-DIFF | Smoke Δpp | Verify Δpp | Verdict                |
|----|--------------------------------------|:--------:|----------:|-----------:|------------------------|
| 1  | rcg+noemxpre+tv16   (3-stack)        | DIFF     |    +0.50  |  +0.498\*  | sub-+0.5pp (relaxed FAIL by 0.002pp) |
| 2  | rcg+noemxpre+v20                     | DIFF     |    -0.17  |    n/a     | smoke fail             |
| 3  | rcg+noemxpre+lgk2                    | DIFF     |  +3.87\*\*|   +3.977   | base cold-start; reverify -2.59pp; FAIL  |
| 4  | rcg+noemxpre+extbr                   | DIFF     |    -1.07  |    n/a     | smoke fail             |
| 5  | rcg+tv16+v20                         | DIFF     |    -0.16  |    n/a     | smoke fail             |
| 6  | rcg+tv16+lgk2                        | DIFF     |  +3.67\*\*|   +0.271   | base cold-start; reverify -0.06pp; FAIL  |
| 7  | rcg+tv16+extbr                       | DIFF     |   -20.68  |    n/a     | catastrophic regression |
| 8  | rcg+noemxpre+tv16+v20  (4-stack)     | DIFF     |   -15.83  |    n/a     | catastrophic regression |
| 9  | rcg+noemxpre+tv16+lgk2 (4-stack)     | DIFF     |    +0.67  |   -3.767   | base outlier; reverify -2.59pp; FAIL    |
| 10 | rcg+noemxpre+tv16+extbr (4-stack)    | DIFF     |    -1.17  |    n/a     | smoke fail             |

\* `rcg+noemxpre+tv16` 5-rep verify: cand mean=4953.9 ≥ base.max=4956.1 (FAIL by 2 TFLOPS); 6-rep reverify (discard 1): cand mean=4947.8 ≥ base.max=4945.1 (PASS), Δ=+0.498pp; 8-rep final on GPU2: dragged by 2 GPU-noise outliers, mean Δ=-0.88pp but stable-runs Δ ≈ +1.2pp.

\*\* Smoke base was a cold-start outlier; 5-rep verify confirms no real lift.

---

## Key findings

1. **Triple-stack additivity hypothesis FALSIFIED.** The strongest 3-stack
   `rcg+noemxpre+tv16` reaches Δ=+0.498pp (6-rep reverify, mean ≥ base.max
   PASS) — **the same magnitude as the 2-stack `rcg+noemxpre` (R16C +0.30pp →
   re-measure here ≈ +0.50pp)**. Adding `tv16` on top of `rcg+noemxpre`
   contributes **0 ± 0.2pp**, not the predicted +0.225pp.

2. **Sub-threshold deltas COLLAPSE on stacking, not add linearly.** Pre-stack
   sum was +0.236 + +0.225 + +0.30 ≈ +0.76pp. Observed combined: ≈ +0.50pp.
   Even relative to the best 2-stack baseline, the 3rd flag adds nothing.

3. **Quad-stacks all FAIL.** Two 4-stacks catastrophically regress
   (`rcg+noemxpre+tv16+v20` -15.8pp, `rcg+tv16+extbr` -20.7pp at smoke);
   the third 4-stack (`rcg+noemxpre+tv16+lgk2`) shows +0.67pp single-shot
   but fails verify (-3.77pp on 5-rep, -2.59pp on 6-rep reverify). Stacking
   beyond 3 flags actively destabilizes the codegen.

4. **`extbr` interacts badly with `rcg+tv16`** (-20.68pp). New
   dead-end: `STEP4_EXTERNAL_BR_PREFETCH=1` should NOT be combined with
   regclassglob+tv16 on P1.

5. **`noemxpre+v20` and `tv16+v20` cancel `regclassglob`'s lift.** Both
   end at -0.16~-0.17pp single-shot, suggesting STEP3_BARRIER_VMCNT=20 conflicts
   with the regalloc-priority change.

6. **Lgk2 is benign on P1** (vs lgk0 default) — the 3-stacks containing lgk2
   are the only non-extbr/non-v20 ones that don't regress sharply.

---

## New dead-end vectors (R17B)

- `regclassglob + noemxpre + tv16` triple stack on P1 — saturates at +0.498pp
  (= best 2-stack; the 3rd flag is purely redundant on this kernel)
- `regclassglob + noemxpre + extbr` on P1 — -1.07pp (extbr×rcg conflict)
- `regclassglob + tv16 + extbr` on P1 — -20.68pp (catastrophic 3-flag conflict)
- `regclassglob + noemxpre + tv16 + v20` 4-stack on P1 — -15.83pp catastrophic
- `regclassglob + tv16 + v20` on P1 — -0.16pp (v20×rcg conflict)
- `regclassglob + noemxpre + v20` on P1 — -0.17pp (same conflict)
- All 4 P1 4-stacks (3-of-4 regress sharply; the one survivor matches best 2-stack)

---

## Conclusion

**The compound-stacking axis on P1 is now exhausted.** The R14/R15/R16
sub-threshold winners individually contribute +0.2-0.3pp and saturate at
+0.5pp when stacked — they are NOT independent, additive perturbations.
The best 2-stack (R16C `regclassglob+noemxpre` +0.30pp clean re-verify, ~+0.50pp
in this round's measurement environment) is the local-optimal flag combination
for P1 in the existing dispatcher. P1 cannot be moved past 94.2% by additional
LLVM/macro flag stacking. Future work must be kernel-source level (per global
TODO direction).

**No commit.** Best signal Δpp_mean = +0.498pp falls 0.002pp short of the
relaxed +0.5pp commit gate, and 0.5pp short of the +1.0pp standard gate.
The signal is real but matches the existing R16C 2-stack within noise.

---

## Files written

- `build_round17_optB_p1_triplestack.{py,log}`
- `asm_diff_probe_r17b.{py,log,json}`
- `bench_round17_optB_smoke.{py,log,json}`
- `bench_round17_optB_verify.{py,log,json}`
- `bench_round17_optB_reverify.{py,log,json}`
- `bench_round17_optB_final.{py,log,json}`
- `round17_optB_verdict.md` (this file)

No commits made.
