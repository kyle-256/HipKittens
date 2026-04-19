# R42 Opt C — Broaden R41A extract_tile vmcnt fence — VERDICT

## Headline: **REFUTED** — broadening the R41A vmcnt fence beyond `K_DIM>=16384` causes catastrophic prefetch-overlap loss; net 5-run gain is only +1 verified-correct shape but at −9% mean perf cost across the leaderboard.

## Setup
- **C1** (`R42C_FENCE_NO_K_GUARD=1`): drop the `K_DIM>=16384` guard; fence at every `extract_tile` for any K when `FUSED_STEP34=1`.
- **C2** (`R42C_FENCE_ANY_PATH=1`): drop both `FUSED_STEP34` AND `K_DIM` guards. **SKIPPED** (see below).
- Kernel modification: opt-in macros added at `kernel_mxfp4_gluon_cpp.cpp:306-350`. Default behavior is bit-identical to R41A. Original R41A gating preserved as comment.
- Baseline: `R41_INTEGRATION_5RUN.json` (20/42 verified-correct).
- Build: 32 unique builds for 41 shapes (1 skipped: `4096x32768x128256` uses R40A non-fused path → C1 fence is a no-op there).
- 3-run smoke (GPU 4,5) → 13 shapes flipped verdict → 5-run reviewer on those 13.

## 5-run reviewer on the 13 flipped shapes

| Shape | Baseline | C1 5-run | b_p50 | c1_p50 | Drift | Final |
|-------|----------|----------|-------|--------|-------|-------|
| 16384x4096x2048 | PASS_3/5 | PASS_5/5 | 3073.6 | 2893.4 | −5.9% | **REAL_FLIP_TO_PASS** |
| 16384x6144x2048 | PASS_4/5 | PASS_5/5 | 3207.8 | 3007.0 | −6.3% | **REAL_FLIP_TO_PASS** |
| 16384x4096x14336 | FLAKE_2/5 | PASS_5/5 | 4487.8 | 3692.6 | −17.7% | **REAL_FLIP_TO_PASS** |
| 32768x4096x2048 | FLAKE_2/5 | WRONG_5/5 | 3141.4 | — | — | still_fail (got worse) |
| 32768x6144x2048 | PASS_4/5 | PASS_3/5 | 3318.8 | 3098.7 | −6.6% | **REAL_FLIP_TO_FAIL** (now <verified) |
| 28672x32768x4096 | PASS_5/5 | WRONG_5/5 | 3896.2 | — | — | **REAL_FLIP_TO_FAIL** |
| 32768x14336x2048 | FLAKE_2/5 | FLAKE_2/5 | 3435.0 | 3117.1 | −9.3% | still_fail |
| 16384x6144x4096 | FLAKE_2/5 | FLAKE_2/5 | 4092.7 | 3618.8 | −11.6% | still_fail |
| 16384x14336x4096 | FLAKE_2/5 | PASS_4/5 | 4047.8 | 3590.2 | −11.3% | still_fail (PASS_4/5 ≠ verified per gate) |
| 16384x28672x4096 | PASS_4/5 | PASS_4/5 | 4038.4 | 3679.4 | −8.9% | still_fail |
| 4096x4096x16384 | PASS_5/5 | PASS_5/5 | 4242.0 | 3765.4 | −11.2% | still_pass (regressed perf) |
| 6144x4096x16384 | PASS_5/5 | PASS_5/5 | 3670.0 | 3034.9 | −17.3% | still_pass (regressed perf) |
| 6144x4096x8192 | PASS_5/5 | PASS_5/5 | 3357.0 | 2814.2 | −16.2% | still_pass (regressed perf) |

**Net 5-run flips: +3 to PASS, −2 to FAIL → +1 net verified-correct.**

## Perf drift on shapes that already passed (3-run smoke, n=31)
- mean: −9.1%
- min: −18.1% (16384x4096x14336)
- max: +0.6% (4096x32768x128256, which used baseline .so anyway)
- 5/31 shapes drifted ≤ 1% (the 5 R41A cluster-C deep-K shapes — fence already active there, so no change).

## Mechanism — why broadening fails
Per the R41A win note: the deep-K (K=32768) cluster-C shapes are *not* prefetch-overlap-bound. With `tail_pf_off=120` and only 8 useful PF iters out of 128, the kernel rides on already-staged registers; one extra `s_waitcnt vmcnt(0)` in the loop body costs nothing because the load queue is naturally empty by then.

At smaller K (and any healthy prefetch state machine), the fence drains all in-flight `buffer_load_dwordx4` before *every* extract_tile, killing the LDS-staged-prefetch / VMEM-prefetch overlap that 30+ R-rounds of tuning bought. Result: −10-18% perf almost everywhere. Some near-gate cluster-B shapes "stabilize" their flakiness because the artificial drain hides the same load-extract race that flakes them in baseline — but the cure is far worse than the disease.

## C2 (drop FUSED_STEP34 guard too) — **SKIPPED**
Per the mission directive ("IF C1 has wins, try C2 on the 14 shapes whose R41 integration source is R40B (FUSED_STEP34=0 in some cases). Otherwise skip C2."): C1's net +1 verified-correct comes paired with system-wide −9% perf. Pursuing C2 — which fires the same fence on the non-fused path too — would extend the same perf cliff to the R40A-source shape with no plausible upside; C2 is REFUTED by extension.

Also: all R40B/R41A/R41B-source shapes already use `FUSED_STEP34=1`, so C2 only newly affects 1 shape (the R40A `4096x32768x128256`), which already passes 5/5 in baseline. No mechanism for C2 to add wins.

## Recommendation
**REFUTED — do NOT promote C1 default-on.** Per-shape selective use is also rejected: the 3 real flip-to-PASS shapes all came at −6% to −18% perf cost, and 2 baseline-PASS shapes regressed to FAIL. The R41A gate (`FUSED_STEP34 && K_DIM>=16384`) is mechanism-correct: the load-extract race only manifests catastrophically at deep K where prefetch overlap is already lost. At lower K, the fence is pure perf damage.

Lesson: the R41A win was a tight match between mechanism and gating — broadening it without re-tuning prefetch-overlap state machines is a strict perf loss. Future work for cluster-B near-gate flakiness should target the actual race (probably scale/data tile alignment per `project_mxfp4_scale_data_misalign.md`) rather than VMEM drains.

## Deliverables (all in `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/`)
- `R42C_BUILD_MANIFEST.json` (= `R42C_C1_BUILD_MANIFEST.json`) — per-shape C1 module map
- `bench_all_42_R42C1.py` — smoke harness
- `bench_R42C1_flips_5run.py` — 5-run reviewer (flipped shapes only)
- `R42_OPT_C_C1_SMOKE.{json,log}` — 3-run smoke
- `R42_OPT_C_C1_5RUN.{json,log}` — 5-run reviewer
- `kernel_mxfp4_gluon_cpp.cpp:306-350` — non-default macro additions (R42C_FENCE_NO_K_GUARD, R42C_FENCE_ANY_PATH; both default 0; original R41A gate preserved as comment).
