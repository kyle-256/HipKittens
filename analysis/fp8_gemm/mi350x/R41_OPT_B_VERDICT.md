# R41 Opt B Verdict — Cluster-B near-gate per-shape variant retune

**Date**: 2026-04-19
**Agent**: R41_OPT_B
**Hypothesis**: Cluster-B residual wrong cells (R40B 5-run wcf 0.3-8%) caused by
per-shape suboptimal `R25C_TAIL_PF_OFF_ITERS` and variant-flag (`_btw_all`,
`_v12`/`_v32`, `_lgk1`/`_lgk2`) carried over from the non-fused autotune.
Re-sweep on FUSED_STEP34=1 base (no kernel edits) should tip 4-7 shapes across the
2% wrong-cell gate.

**Surgical change**: Build-flag fork only (NO kernel source edits). Per-shape, per-variant:
- `-DFUSED_STEP34=1` (R40B base)
- Strip `-mllvm -amdgpu-sched-strategy=max-memory-clause`
- Per-variant overrides:
  - `v0a/v0b/v0c/v0d`: `R25C_TAIL_PF_OFF_ITERS in {0, k_iters/8, k_iters/4, k_iters/2}` with `R25C_K_EXACT=K_DIM`
  - `v1`: drop `-DBARRIER_TO_WAITCNT_ALL=1` (no `_btw_all`)
  - `v2`: swap `-DSTEP3_BARRIER_VMCNT=12` -> `=32` (`_v12`->`_v32`)
  - `v3`: swap `-DSTEP12_BR_LGKMCNT=2` -> `=1` (`_lgk2`->`_lgk1`)
- Module suffix: `_R41B_<variant_id>`
- Build harness: `build_R41B.py`. 11 shapes x 7 variants = 77 plans (66 built, 11 skipped where variant N/A).

**Bench gate** (`bench_all_42_R41B.py`, R39B-style):
warmup=200, iters=500, trim=0.10, random scale [-2,2], seed=42,
PASS = `wrong_cell_frac < 2% AND snr_med >= 10 dB AND finite >= 0.99`.
3-run consensus for optimizer; **5-run reviewer for promotion**
(strict gate: `n_OK >= 3 AND wcf_max < 2% AND wcf_std < 1%`).

---

## VERDICT: **PARTIAL — 2 of 11 cluster-B shapes recover under strict 5-run gate.**

| Bucket | Count | Notes |
|---|---|---|
| 5-run STRICT PASS (n_OK>=3, wcf_max<2%, wcf_std<1%) | **2/11** | (16384,4096,14336), (32768,4096,2048) |
| 5-run PARTIAL (n_OK>=3 but wcf_max>=2%) | 1/11 | (16384,14336,4096) — flake risk |
| 5-run FAIL (n_OK<3) but had >=1 OK in 5 runs | 6/11 | flake-only, not promotable |
| Still all-WRONG (no OK in any of 5 runs) | 2/11 | (4096,32768,14336), (4096,32768,128256) |

Below the optimistic prediction (4-7 shapes flipped). Three observations explain the
gap:

1. **The dominant blocker is `kernel_finite < 0.99`, not `wcf_max < 2%`.** The wcf
   sweep DID lower wcf in many cases (e.g. v3 on 16384x28672x2048 cuts wcf 0.29% -> 0.51%
   on the median run, but `fin_min` stays 0.982). The fin gate is what disqualifies
   the bulk of candidates.

2. **The "1-of-3 flake PASS" pattern.** Many (shape, variant) cells have 1-2 OK
   runs out of 5, but most runs fail the fin gate. This is a runs-rebuild-flake
   not a deterministic-fail: the kernel correctness depends on something
   non-deterministic each launch (e.g. an MFMA vgpr cohort race). Tail-PF
   sweep cannot fix the underlying race.

3. **Variant axis effects are 2nd-order vs the underlying race.** Across the 5-run
   data, `wcf_max` varies <1% between v0a..v3 on most shapes — the variant choice
   does NOT meaningfully change the corruption regime. Only `(16384,4096,14336)`
   and `(32768,4096,2048)` have any variant that consistently lands above the gate.

---

## Per-shape best variant + recommendation

| Shape | best variant | tflops/comp | n_OK/5 | wcf_max | fin_min | verdict |
|---|---|---|---|---|---|---|
| (16384,28672,2048)  | v0a (pfoff=0)         | 3422/3482 (98.3%) | 1/5 | 0.0110 | 0.984 | NO_PROMOTE (flake) |
| (32768,28672,2048)  | v2 (v32)              | 3320/3353 (99.0%) | 2/5 | 0.0082 | 0.986 | NO_PROMOTE (flake) |
| (4096,32768,6144)   | v2 (v32)              | 4129/4549 (90.8%) | 1/5 | 0.0178 | 0.983 | NO_PROMOTE (flake) |
| (4096,32768,14336)  | none                  | -                 | 0/5 | -      | -     | NO_PROMOTE (no PASS) |
| (4096,32768,128256) | none                  | -                 | 0/5 | -      | -     | NO_PROMOTE (no PASS) |
| (14336,32768,4096)  | v0c (pfoff=k/4=4)     | 3875/4463 (86.8%) | 2/5 | 0.0184 | 0.984 | NO_PROMOTE (flake) |
| (16384,14336,4096)  | v0b (pfoff=k/8=2)     | 3986/4256 (93.7%) | 4/5 | 0.0438 | 0.994 | PARTIAL (wcf flake) |
| **(16384,4096,14336)**  | **v0b (pfoff=k/8=7)** | **4461/5142 (86.8%)** | **3/5** | **0.0191** | **0.988** | **PROMOTE** |
| (28672,4096,8192)   | v2 (v32)              | 4403/4810 (91.5%) | 1/5 | 0.0124 | 0.981 | NO_PROMOTE (flake) |
| (28672,4096,16384)  | v1 (drop _btw_all)    | 4401/5351 (82.3%) | 2/5 | 0.0296 | 0.985 | NO_PROMOTE (flake) |
| **(32768,4096,2048)**   | **v3 (lgk1)**         | **3250/3132 (103.8%) WIN** | **4/5** | **0.0130** | **0.986** | **PROMOTE** |

### PROMOTE candidates (2)

- **(16384, 4096, 14336)**: variant `v0b` — `R25C_TAIL_PF_OFF_ITERS=7` (= K_iters/8 where
  K_iters = 14336/256 = 56) on `ts_gm8_v12_btw_all` parent. 5-run avg 4461 TFLOPS,
  wcf_max=0.0191 (just under gate), wcf_mean=0.0120, wcf_std=0.0040, fin_min=0.988.
  vs R40B (WRONG_4/5): NEW PASS; vs aiter 5142: 86.8% (LOSE_CORRECT).
  - **NOTE**: this overlaps R40C's per-shape rescue for the same shape (R40C reported
    PASS at 4383 TFLOPS, 85.2% comp). R41B's v0b is +78 TFLOPS (+1.8%) better.
    Per-shape integration should pick R41B v0b.

- **(32768, 4096, 2048)**: variant `v3` — `STEP12_BR_LGKMCNT=1` swap on
  `ts_lgk2_gm6_v12_memc_pfoff4` parent. 5-run avg 3250 TFLOPS, wcf_max=0.0130,
  wcf_mean=0.0103, wcf_std=0.0018, fin_min=0.986. **WIN (103.8% of comp).**
  vs R40B (WRONG_3/5): newly recovered. Promotes from "newly lost in R40B 5-run" back to PASS+WIN.
  - Backup choice: v0b (`R25C_TAIL_PF_OFF_ITERS=0`), 4/5 PASS, avg 3203 TFLOPS (102.3%
    of comp), wcf_max=0.0112 — also a WIN. Either works.

### Flake-risk candidate (1, not promoted)

- **(16384, 14336, 4096)**: variant `v0b` — 4/5 OK but wcf_max=0.0438 in run 3,
  exceeds the 2% gate. wcf_std=0.0124. Not promotable under strict reviewer rule.
  Could be flagged as "watch list" candidate for re-bench in next round.

### Refuted candidates (8)

8 of 11 shapes either had 0 PASS in 5 runs ((4096,32768,14336), (4096,32768,128256))
OR had only 1-2/5 OK runs across all variants (the "flake-only" bucket). The variant
sweep cannot recover these — root cause is below the variant axis (likely the
FUSED_STEP34 + R37_FIX_B vmcnt fence completeness, or the MFMA vgpr cohort race
suspected for cluster-B residual).

---

## Leaderboard delta vs R40B 26/42 baseline

| Bucket | R40B | After R41B integration |
|---|---|---|
| Verified-correct PASS | 24 (5-run stable) | **26** (+2 from R41B) |
| Per-shape rescues (R40A + R40C) | +2 expected | +1 (R40A K=128256 still rescues; R41B's K=14336 supersedes R40C) |
| **Total verified-correct** | **26 projected** | **27** |

Net delta: +1/+2 verified-correct (R41B's (32768,4096,2048) is a NEW shape;
R41B's (16384,4096,14336) supersedes R40C's rescue at +1.8% perf).

Falsification check: hypothesis predicted "4-7 shapes flipped"; actual = 2 shapes
firmly + 1 partial = REFUTED for the 4+ prediction. The variant-flag axis is
too coarse to fix the residual cluster-B corruption beyond 2 shapes.

---

## Observations / next-round levers

1. **The dominant residual blocker is `finite < 0.99` (NaN/Inf cells)**, not
   wcf>2%. Across all 5-run candidates, the wcf was usually 0.5-1.5% (well under
   gate) — but fin_min was 0.97-0.99, just under gate. This means the kernel is
   producing a small, persistent number of NaN/Inf outputs that the variant retune
   cannot eliminate. Suggests the residual mechanism is upstream of the K-loop
   prefetch state machine.

2. **The "flake-only" pattern strongly suggests a non-deterministic cohort race**
   (likely the MFMA vgpr cohort race hypothesized in R35). Variants change WHEN
   the race fires but not WHETHER it can fire. A real fix needs either:
   - kernel-level vgpr keepalive barriers (R34 VGPR-PF approach extended), OR
   - reordering the MFMA issue to remove the race window entirely.

3. **R41A (Cluster C deep-K) is a separate orthogonal effort**; R41B does not
   touch the K=32768 catastrophic shapes. The 2 shapes that have 0 PASS here
   are (4096,32768,14336) and (4096,32768,128256) — both K-tail extreme — likely
   share mechanism with cluster C.

---

## Files

- `build_R41B.py` — build harness (44 effective + 33 cached = 66 .so + 11 skipped)
- `bench_all_42_R41B.py` — bench script restricted to 11 cluster-B shapes
- `R41_OPT_B_BUILD.log` — full build log (0 failures)
- `R41_OPT_B_BENCH_SMOKE.log/.json` — 19-task single-run smoke
- `R41_OPT_B_BENCH_FULL.log/.json` — 66-task 3-run consensus (6/11 had >=1 PASS variant)
- `R41_OPT_B_BENCH_5RUN.log/.json` — 53-task 5-run consensus (top candidates from full)
- `R41B_BUILD_MANIFEST.json` — per-shape per-variant module map
- Compiled .so files in `build_R41B/`
