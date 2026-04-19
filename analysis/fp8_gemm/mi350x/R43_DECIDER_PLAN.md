# R43 Decider Plan — push 27/42 → 30+/42 verified-correct

**Date**: 2026-04-19
**Branch**: `mxfp4` @ `835a8f01`
**Baseline**: R42 Opt A1 (FINITE_GATE=0.98), 27/42 verified-correct, 10/42 WIN.
**Stretch goal**: 30/42 verified-correct (need +3 net), no regressions in the 27 VC.
**Ground-truth oracle**: `R43_DECIDER_PER_SHAPE.json` (built from `R42_OPT_A_PHASE2_A1_RELAX_GATE.json`).

---

## 1. Per-shape bucket assignment (42 rows; full table in JSON)

| Bucket | n | Shapes (M×N×K) |
|---|---|---|
| **VC** (verified-correct, ≥95% comp) | 13 | small-K + R41A K=32768 well-tuned |
| **VC_CLAWBACK** (verified-correct but <95% comp) | 14 | R43 Opt C target |
| **CRASH** | 2 | `4096x32768x28672`, `16384x4096x28672` (R43 Opt A target) |
| **WRONG_5/5** | 1 | `16384x28672x4096` (wcf=0.042, fin=0.988 — bordering WCF_BOUND) |
| **FIN_BOUND** (wcf<2%, fin∈[0.96, 0.98)) | 3 | `32768x4096x2048`, `16384x14336x2048`, `16384x28672x2048` |
| **WCF_BOUND** (wcf_max ≥2%) | 9 | `{4096x32768x{6144,14336}, 16384x6144x4096, 16384x14336x4096, 28672x4096x{8192,16384}, 32768x4096x{7168,14336}, 128256x32768x4096}` (R43 Opt B target) |

The 9 WCF_BOUND shapes are the cohort-race residual after gate relax. The 3 FIN_BOUND are statistically very close — `16384x14336x2048` has wcf<0.001 but fin=0.92 from one outlier run; the other two are barely below the 0.98 floor. The single WRONG_5/5 (`16384x28672x4096`) is structurally part of the WCF_BOUND cluster (wcf=0.042) and we treat it that way for Opt B.

---

## 2. Worker assignments

Each worker uses Opus, isolates GPU via `HIP_VISIBLE_DEVICES`, runs warmup=200/iters=500/trim=0.10, and reports through 5-run consensus at GATE=0.98.

### R43 Opt A — CRASH structural fix (GPUs 0–1)

**Targets**: `4096x32768x28672`, `16384x4096x28672` (both 5/5 FAIL_CRASH).

**Mechanism (from R42 Opt B)**: deterministic CRASH iff `FUSED_STEP34=1 AND TAIL_SPLIT=1` at K_DIM=28672 (k_byte_iters=112). Suspect: `kernel_mxfp4_gluon_cpp.cpp:~3222` emits unconditional `emit_pf_tail<0>(pf_a0_p, pf_a1_p)` in the FUSED_STEP34 path with no R25C gate; `make_pf_params` likely produces an OOB voff at the k=112 boundary.

**Two falsifiable fix candidates** (test both serially on GPU 0; reuse 1 for build pool):

1. **A.fix1 — `R43A_GATE_PF_TAIL_KBOUND`** (default OFF macro): wrap the unconditional `emit_pf_tail<0>` in the FUSED+TAIL_SPLIT branch with `if constexpr (k_byte_iters % some_div == 0 || pf_bt < k_byte_iters)` — i.e., skip the prefetch when the tail params would land OOB.
2. **A.fix2 — `R43A_WIDEN_SRD_NUM_RECORDS`** (default OFF): widen `num_records` for B-tile SRD by `+(k_byte_iters * stride_B)` for the tail prefetch SRD only, so OOB voff still maps to a legal in-range record. Compare against the R33 SRD-swap memo (refuted for vmcnt(15) but here we're widening for legality, not for vmcnt freedom).

**Falsifiable predictions** (each must be verifiable in ≤30 min bench):
- **P-A.1**: A.fix1 closes both 28672-CRASH shapes → `n_OK ≥ 4/5` under 5-run; tflops ≥ 3500 (≥63% comp) on both. If A.fix1 fails on either shape, fall to A.fix2.
- **P-A.2**: Either fix introduces ZERO regressions on the other 40 shapes (run integration probe at GPU 0 over the 40, single rep). If any pre-VC shape drops ≥5% perf or flips to FAIL, REJECT and gate the macro per-shape only.
- **P-A.3** (stretch): Same fix recovers `4096x32768x14336` to PASS_VC (it's currently PASS_3/5, wcf=0.022, sister K=14336 with FUSED_STEP34 may hit the same boundary). Only test if P-A.1 succeeds.

**Stopping criteria**:
- WIN: both CRASH shapes flip to PASS_VC, no regressions → +2 to leaderboard, COMMIT and update integration manifest.
- PARTIAL: one CRASH shape passes — commit per-shape macro for that shape only, leave the other in CRASH state.
- DEAD: both fixes fail on both shapes after ≤4 hours → write VERDICT, defer to R44 (try R38B-fork-with-FUSED_STEP34=0 + a dedicated 28672 build variant).

### R43 Opt B — MFMA cohort race fix via R34 VGPR-PF + `+v` keepalive (GPUs 2–4)

**Targets**: 9 WCF_BOUND shapes (and the 1 WRONG_5/5 sister). All have wcf_max ≥2% but fin_min ≥0.97. Per R42 Opt A diagnostic (`R42_OPT_A_PHASE1_DIAGNOSTIC.json`): bad-cell positions have median Jaccard ≈ 0.06 across 5 fixed-input runs → MFMA accumulator race.

**Mechanism prior**: see `project_mxfp4_vgprpf_compiler_bug.md` — the R34 VGPR-PF fork builds at 219 VGPR/0 spills and sustained vmcnt(15) without GPU faults, but compiler dropped scratch VGPR contents between adjacent `asm volatile` blocks. The remediation (never tried) is `asm volatile("" : "+v"(b_scratch[i]))` keepalive barriers between buffer_load and ds_write.

**Approach** (kernel-level):
1. Resurrect `kernel_mxfp4_gluon_cpp_vgprPF.cpp` (R34 fork on disk).
2. Insert `asm volatile("" : "+v"(<each scratch VGPR>))` keepalive immediately after each `buffer_load` and before each `ds_write` consumer.
3. Build behind `R43B_VGPRPF_KEEPALIVE=1` (default OFF).

**Ground-truth verification protocol** (the falsifiable hook):
- Re-run **the exact same 5-probe `INPUT_REUSE=True` Phase-1 oracle** from `R42_OPT_A_PHASE1_DIAGNOSTIC.{py,json}` on the new build for **3 representative WCF_BOUND shapes**:
  - `16384x14336x4096` (mid-K, mid-wcf 0.023)
  - `28672x4096x8192` (mid-K, wcf 0.032)
  - `4096x32768x14336` (large-K, wcf 0.022)
- Compute Jaccard across the 5 probes per shape.

**Falsifiable predictions** (each verifiable in ≤30 min):
- **P-B.1 (mechanism)**: If the race is closed, Jaccard ≥ 0.7 on all 3 shapes (vs current ≈ 0.06). If Jaccard stays <0.3 → race is NOT in the prefetch-VGPR cohort → KILL and report.
- **P-B.2 (leaderboard)**: ≥2 of the 3 representative shapes flip to PASS_VC under 5-run @ GATE=0.98. Stretch: ≥4 of all 9 WCF_BOUND shapes flip.
- **P-B.3 (perf)**: VGPR-PF perf does NOT regress >3% on the 13 already-VC shapes (run smoke 1-rep across all 42 shapes after kernel resurrects).

**Stopping criteria**:
- WIN: P-B.1 + P-B.2 both hold → 5-run on the 9 WCF_BOUND + 13 already-VC shapes, commit if ≥+2 net.
- PARTIAL: race closed (P-B.1) but only 1 shape flips → diagnostic complete, commit kernel as default OFF, schedule per-shape integration in next round.
- DEAD: Jaccard stays low (P-B.1 fails) OR keepalive barriers don't compile clean OR perf regresses universally → write VERDICT, abandon VGPR-PF axis.
- Hard timeout: 6 hours from start (kernel work is heaviest of the round).

### R43 Opt C — perf claw-back on 14 VC_CLAWBACK shapes (GPUs 5–7)

**Targets** (verified-correct under GATE=0.98, but pct_comp < 95%):
- 0–80% comp: `14336x4096x32768` (60.5%), `4096x28672x32768` (61.9%), `4096x32768x128256` (73.5%), `4096x4096x32768` (77.4%)
- 80–95%: `4096x6144x32768`, `6144x4096x16384`, `4096x14336x16384`, `14336x32768x4096`, `16384x4096x14336`, `28672x32768x4096`, `6144x4096x8192`, `4096x14336x8192`, `4096x32768x4096`, `6144x32768x4096`

**Approach**:
1. Per-shape variant sweep over the existing R40B/R41A/R41B variant table BUT now under FINITE_GATE=0.98 instead of 0.99. R41B's coarse retune REJECTED 2 shapes under 0.99-gate; some may pass under 0.98.
2. Strictly per-shape: for each of the 14 shapes, build the top 8 candidate variants from the existing table (no NEW kernel macros), run 1-rep smoke, then 5-run consensus on any that beat the current per-shape p50 by ≥3% AND PASS the gate.
3. Reuse `bench_all_42_R42A1.py` as the gate harness (FINITE_GATE=0.98 already set).

**Falsifiable predictions** (verifiable in ≤30 min/shape):
- **P-C.1**: ≥3 of the 14 shapes recover to ≥100% comp (WIN_VC) under 5-run, no PASS_VC regressions.
- **P-C.2**: ≥6 of the 14 shapes improve perf by ≥5% while remaining PASS_VC.
- **P-C.3** (per-shape commit gate): ANY proposed variant must hold ALL of: `n_OK_5 ≥ 4`, `wcf_std < 0.005`, `fin_min ≥ 0.985`, `tflops_p50 ≥ 1.05 × current_p50`. If a variant only meets `n_OK_5 ≥ 3`, it's a candidate but does NOT replace current.

**Stopping criteria**:
- WIN: ≥3 shapes flip to WIN_VC → commit per-shape manifest update.
- PARTIAL: only 1–2 flip but ≥4 improve perf without losing VC → commit those.
- DEAD: zero improvements after ≤6 hours of sweep across the 14 → write VERDICT, defer to a kernel-axis round.

---

## 3. Conflict points & tie-breaking

- **Opt A vs Opt C overlap**: Opt A introduces a new macro (`R43A_GATE_PF_TAIL_KBOUND` or `R43A_WIDEN_SRD_NUM_RECORDS`); Opt C re-tunes existing variant flags. They could both want to claim `(K_DIM=14336)` shapes (Opt A speculative recovery vs Opt C re-tune). Tie-breaker: **Opt A wins on CRASH shapes only**; for any K=14336 shape, Opt C's per-shape variant takes priority unless Opt A demonstrates strict dominance under 5-run reviewer.
- **Opt B vs Opt C overlap**: VGPR-PF kernel is a separate `.so` (different file, like the R34 fork). It will not collide with Opt C's variant builds. If Opt B promotes for some WCF_BOUND shape AND Opt C also has a variant candidate for the same shape (none currently overlap), the per-shape integration manifest takes the higher VC + higher tflops_p50 candidate (`n_OK ≥ 4` mandatory).
- **Opt A vs Opt B**: no overlap — Opt A targets CRASH (K=28672), Opt B targets WCF_BOUND (K∈{4096..16384}, none at K=28672).

**Hard rule**: integration manifest only swaps in a R43 variant if it strictly dominates current under 5-run (`n_OK_R43 ≥ n_OK_baseline AND tflops_R43 ≥ 0.97 × tflops_baseline AND wcf_max < 0.02 AND fin_min ≥ 0.98`). No variant flips on the 27 already-VC shapes unless this dominance is met.

---

## 4. Integration plan

After all 3 workers report:

1. **Build new manifest**: copy `R41_INTEGRATION_MANIFEST.json` → `R43_INTEGRATION_MANIFEST.json`; per worker verdict, swap in new `.so` paths for any approved shape.
2. **5-run reviewer**: run `bench_all_42_R42A1.py` (FINITE_GATE=0.98) with `--manifest=R43_INTEGRATION_MANIFEST.json` across all 42 shapes, 5 runs, 8-GPU parallel.
3. **Acceptance**:
   - Floor: ≥27/42 (no net regression) AND ≥1 net new VC shape → COMMIT round.
   - Stretch: ≥30/42 → COMMIT + write `R43_INTEGRATION_VERDICT.md` + update `TODO.md` headline.
4. **Write-back**: per-worker VERDICT.md, `R43_DECIDER_PLAN.md` (this file), `R43_INTEGRATION_5RUN.{json,log}`, `R43_INTEGRATION_VERDICT.md`, update `TODO.md` and `AGENT_PROMPT.md`. Update durable memos:
   - `project_mxfp4_R43A_crash_fix.md` if Opt A wins
   - update `project_mxfp4_vgprpf_compiler_bug.md` with Opt B keepalive findings
   - new `project_mxfp4_R43_finite_gate_clawback.md` if Opt C wins ≥3 WIN_VC

---

## 5. Hard rules (recap)

- All bench: warmup=200, iters=500, trim_frac=0.10, GATE=0.98, 5-run consensus.
- Every new macro: default OFF, per-shape gated through integration manifest.
- 8 GPUs (HIP_VISIBLE_DEVICES=0..7); workers must NOT all queue on GPU 0. Bench harness uses 8-GPU pool.
- Use Opus for all sub-agent dispatches. No `sleep`+poll — use sub-agents to monitor background runs.
- Worker may reject its own assignment if pre-flight refutes the prediction.
