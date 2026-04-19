# R41 Decider Plan — From R40B 24/42 toward 42/42 verified-correct

**Date**: 2026-04-19
**Author**: R41 decider agent
**Inputs**: R40B 5-run consensus (`R40B_5RUN_CONSENSUS.json`), R40 verdicts
(A/B/C/D), R40 reviewer report, R37 leaderboard, R35 wrong-cells diagnosis,
commit `7f37a8dd`.
**Bench gate**: `bench_all_42_R39B.py` (random scales [-2,2]),
`wrong_cell_frac < 2% AND snr_med >= 10 dB AND finite >= 0.99`,
warmup=200, iters=500, trim=0.10, GPUs 0-7.

---

## 0. Headline state going into R41

| Bucket | Count | Notes |
|---|---|---|
| R40B PASS (5-run stable, wcf_max < 2%) | 22 | new BEST_VARIANTS baseline |
| R40B PASS (flake-risk, wcf_max in [2%, 3%]) | 2 | `(16384,6144,4096)`, `(32768,4096,14336)` |
| R40A per-shape rescue | +1 | `(4096, 32768, 128256)` |
| R40C per-shape rescue | +1 | `(16384, 4096, 14336)` |
| **Projected post-integration baseline** | **26/42** | before R41 work |
| BROKEN — Cluster C catastrophic (K=32768, fin~10%, wcf~97%) | 5 | highest leverage |
| BROKEN — Cluster B near-gate (wcf 0.3-8%) | 9 | easiest to flip |
| CRASH — aperture violation | 2 | structural; lower priority |

**R41 ceiling target**: 33-37/42 verified-correct (best case 38/42 if Cluster C
+ 5 of B-near-gate land; CRASH-2 deferred).

---

## 1. Updated root-cause synthesis per cluster

### Cluster C — 5 catastrophic K=32768 shapes
Shapes: `4096x4096x32768`, `4096x6144x32768`, `4096x28672x32768`,
`4096x128256x32768`, `14336x4096x32768`. All use the same module suffix
`ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all_R40B_safe`.

R40D (no-prefetch) **shrunk** wcf on 4 of these 5 (e.g.
`4096x4096x32768`: 15.7% → 2.9%, `4096x128256x32768`: 16.0% → 2.85%) —
the only direction with a uniformly-positive sign in the R40D data. R40C drain
did NOT fix the upper-left 128×128 quadrant. R40A fence had no effect.

Three live hypotheses, ranked by posterior:

- **(a) MOST LIKELY — Tail-drift in `tA0/tBl` register-buffered tile across
  120-iter unroll** (`R25C_TAIL_PF_OFF_ITERS = 120` for these shapes). The
  K-loop unrolls to `k_byte_iters = 128` iters and `R25C_TAIL_PF_OFF_ITERS=120`
  means PF is suppressed across the LAST 120 iters — i.e. PF runs only on the
  first 8 iters and then the kernel rides on `extract_tile`-staged registers
  for >100 iters of stale-pointer reads. Combined with R37_FIX_B's incomplete
  fence at the step12 boundary, the tile-register liveness across deep K is
  the dominant corruption vector. R40D's improvement (less prefetch =
  more deterministic = less random race garbage) is consistent with this.

- **(b) PLAUSIBLE — `pfoff120` variant flag interacts with the FUSED_STEP34
  emission path differently than non-fused.** The `_pfoff120` variant was
  selected by autotune against the *garbage-fast* non-fused path. Under
  R40B (FUSED_STEP34=1), `pfoff120` is no longer the right tail-pf offset —
  the fused step34 issues fewer iter-boundary fences, and 120-of-128 PF-off
  may be over-clamping the prefetch issue rate.

- **(c) SECONDARY — `make_pf_params` 16-bit immediate overflow at K_iter > 64**:
  pf-param construction uses `bt * stride` as a scalar immediate; at deep K
  this may silently truncate. Less likely (would crash, not corrupt cells)
  but worth a 5-minute audit.

### Cluster B — 9 near-gate shapes (wcf 0.3-4%, finite > 0.97)
Shapes: `(16384,28672,2048)` 0.29%, `(32768,28672,2048)` 0.60%,
`(4096,32768,6144)` 1.72%, `(4096,32768,14336)` 2.14%, `(4096,32768,128256)`
8.08%, `(14336,32768,4096)` 0.56%, `(16384,14336,4096)` 3.74%,
`(16384,4096,14336)` 1.15%, `(28672,4096,8192)` 0.39%, `(28672,4096,16384)`
0.90%, `(32768,4096,2048)` (newly lost in 5-run).

Common pattern: `finite ≥ 0.97`, `wcf` straddles the 2% gate. R37
recovered most of this cluster by sheer FUSED_STEP34=1 effect; the residual
fraction-of-a-percent wrong cells localize to the **tail iters** of these
shapes (boundary of the K loop where `pfoff*` shifts from PF-on to PF-off
and the `extract_tile` register cohort flips). R40A's per-shape success on
`(4096,32768,128256)` and R40C's per-shape success on `(16384,4096,14336)`
prove that **per-shape variant retune + fence-style microsurgery** can tip
several of these.

Hypothesis: residual wrong cells are caused by **per-shape suboptimal
`R25C_TAIL_PF_OFF_ITERS` and `_pfoff*` variant choice carried over from the
non-fused autotune**. Re-sweeping a small grid of `pfoff` values
({0, 4, 8, 12, k_iters/4}) on the FUSED_STEP34=1 base will likely flip 4-7
shapes across the gate.

### CRASH cluster — 2 shapes
`(16384,4096,28672)`, `(4096,32768,28672)`. Both K=28672, both use
`ts_lgk2_gm7_pfoff104_kx28672_btw_all_R40B_safe`. R37 already had these as
WRONG_OUTPUT under non-fused; FUSED_STEP34=1 promoted them to CRASH (HSA
aperture violation). Likely a tile-stride `lgk2` interaction with the fused
step34 SRD computation at K=28672 (a number with awkward factorization vs the
256-byte tile). Lower priority — structural and would need a different tile
decomposition.

---

## 2. Ranked R41 attacks (4 candidates)

> **All R41 source edits MUST be macros, default OFF, gated under
> `FUSED_STEP34 && R37_FIX_B == 0` OR `FUSED_STEP34` to preserve R40B
> behavior when the macro is off.** R40B baseline must remain bit-identical
> when no R41 macro is set.

### R41 OPT A — Cluster C: deep-K tail-pf-off SWEEP + `extract_tile` fence
**Priority: HIGHEST (5 catastrophic shapes target).**

**Hypothesis**: The 5 K=32768 shapes are catastrophically wrong because
`R25C_TAIL_PF_OFF_ITERS=120` (clamping PF across iters 8..127) interacts with
FUSED_STEP34's fewer iter-boundary scoreboard fences, leaving `tA0/tBl`
register-buffered tile values stale for ~100 iters. R40D's data (4 of 5
shapes IMPROVED with PF entirely off) supports the prefetch path being a
**partial** culprit; combined with a register-fence at the tail boundary,
correctness should recover.

**Surgical change** (kernel_mxfp4_gluon_cpp.cpp):
1. New macro `R41A_DEEP_K_FIX` (default 0). Two sub-flags:
   - `R41A_PFOFF_OVERRIDE` (int, default 0): if non-zero, OVERRIDES
     `R25C_TAIL_PF_OFF_ITERS` for K_DIM >= 16384.
   - `R41A_EXTRACT_TILE_FENCE` (bool, default 0): inserts
     `asm volatile("s_waitcnt vmcnt(0)" ::: "memory")` immediately
     BEFORE every `extract_tile(nxt_a0_d, tA0)` and
     `extract_tile(nxt_bl_d, tBl)` call in the K-loop body of the
     `FUSED_STEP34 && R37_FIX_B && K_DIM >= 16384` branch.
2. Build harness `build_R41A.py` runs a 5×2 sweep:
   `R41A_PFOFF_OVERRIDE ∈ {0, 8, 16, 32, 64} × R41A_EXTRACT_TILE_FENCE ∈ {0,1}` =
   10 builds × 5 catastrophic shapes = 50 .so files (~10 min on 8 GPUs).

**Predicted outcome**: 3-5 of 5 catastrophic shapes recovered. Perf cost
0-8% per shape (the fence is once-per-iter).

**Falsification**: If wcf stays > 5% on ALL 5 shapes for ALL 10 sweep cells,
the K=32768 corruption is NOT in the tail-pf-off + extract_tile path.
Hypothesis (a) refuted; promote (b) or (c).

**Test protocol**:
- Smoke (per build): 1 shape `m4096_n4096_k32768`, single-run, gate
  `wcf < 5%` to advance to full bench (loose gate to keep candidates).
- Full: `bench_all_42_R39B.py` 3-run consensus over all 5 catastrophic
  shapes for top-3 sweep cells; reviewer 5-run on any PASS.
- Check `r35_nonfinite_pattern.py`-style upper-left 128×128 quadrant map
  on ANY new PASS to confirm we fixed the R35 signature, not just averaged
  it down.

**Effort**: 4 agent-hours (1h kernel edit, 1h build sweep, 2h analysis +
5-run reviewer).

---

### R41 OPT B — Cluster B-near-gate: per-shape `pfoff` + variant retune
**Priority: HIGH (cheapest +5/+9 shapes).**

**Hypothesis**: Cluster B residual corruption is per-shape autotune mismatch:
the `_pfoff*` and `_lgk2/_v12/_gm6_gm7` flags were chosen against the
garbage-fast non-fused path. With FUSED_STEP34=1 base, a small re-sweep
will tip many across the 2% gate.

**Surgical change** — **NO kernel edits**. Build-flag fork only.
1. Build harness `build_R41B.py`:
   - For each of the 11 cluster-B shapes (including `(32768,4096,2048)`
     newly-lost), enumerate 4 variant retunes per shape:
     - PFOFF cycling: `R25C_TAIL_PF_OFF_ITERS ∈ {0, k_iters/8, k_iters/4, k_iters/2}`
     - Drop `_btw_all` for shapes that have it
     - Try `_v32` swap on currently-`_v12` variants
     - Try `_lgk1` swap on currently-`_lgk2` variants
   - All combine with `FUSED_STEP34=1`, no `-mllvm max-memory-clause`.
   - Total: 11 shapes × 4 variants = 44 builds (~6 min).
2. Suffix `_R41B_<sweep_id>`.

**Predicted outcome**: 4-7 of the 11 cluster-B shapes flipped to PASS.
Per-shape best-of-N tflops chosen; perf delta vs R40B should be neutral or
positive (we're searching the same axis space autotune already explored,
just at the FUSED_STEP34=1 base).

**Falsification**: If 0 of 11 shapes flip across the 2% gate after the full
sweep, the residual cells are NOT autotune-axis-tunable; cluster B requires
a kernel-level fix (likely a sibling of R41A's extract_tile fence).

**Test protocol**:
- Smoke: 3 shapes `m16384_n28672_k2048`, `m4096_n32768_k128256`,
  `m28672_n4096_k8192`, single-run.
- Full: `bench_all_42_R39B.py` 3-run for each variant; reviewer 5-run
  consensus on the 4-7 candidate winners. Pick best per-shape.
- Confirm no regression on the 22 stable R40B PASSes (sweep is per-shape
  scoped to the 11 cluster-B shapes; other 31 shapes unchanged).

**Effort**: 3 agent-hours (cheapest attack; pure build-flag enumeration).

---

### R41 OPT C — Cluster C alternate: `_pfoff120` → fork to non-`btw_all` variant
**Priority: MEDIUM (parallel insurance against R41A failure).**

**Hypothesis**: The K=32768 shapes share variant
`ts_v12_tv0_dc_gm7_pfoff120_kx32768_btw_all_R40B_safe`. The `_btw_all`
suffix routes through a swizzle-tile-write path; combined with deep-K and
FUSED_STEP34=1, it may be writing C-tiles into accumulators that are still
mid-MFMA. Drop `_btw_all` and `_dc` from the K=32768 variants; rebuild on
FUSED_STEP34=1 base. This is a sibling of R41B but specifically targeted at
cluster C.

**Surgical change** — NO kernel edits.
1. Build harness `build_R41C.py`:
   - For each of the 5 K=32768 shapes, build 3 variants:
     - Drop `_btw_all` (use plain `_pfoff120_kx32768`)
     - Drop `_dc` (no double-cover prefetch)
     - Drop both
   - Total: 5 × 3 = 15 builds.
2. Suffix `_R41C_<variant_id>`.

**Predicted outcome**: 1-3 of 5 catastrophic shapes recovered. Some perf
cost (5-10%) since `_btw_all` and `_dc` were perf-positive on the
non-fused path. Insurance: if R41A targets the wrong mechanism, R41C
covers a different axis.

**Falsification**: If 0 of 5 shapes improve, both `_btw_all` and `_dc` are
not the cluster-C culprit; R41A path is the only remaining lever.

**Test protocol**: Same as R41A but only on the 5 catastrophic shapes.

**Effort**: 2 agent-hours.

---

### R41 OPT D — Reviewer-only: 5-run re-verify R40A/R40C per-shape candidates + flake-risk audit
**Priority: BLOCKER (gates the integration to 26/42 baseline).**

**Hypothesis**: Before any R41 attack lands, we need 5-run consensus on
the R40A and R40C per-shape rescues that the reviewer projected. R40A and
R40C 3-run consensus may have been gate-flake; need 5-run before promoting.

**Action** — no kernel edits, no new builds.
1. Re-bench `(4096, 32768, 128256)` under R40A `R40A_PF_FENCE=1` build,
   5 runs.
2. Re-bench `(16384, 4096, 14336)` under R40C `R40C_LDS_DRAIN=1` build,
   5 runs.
3. Re-bench `(16384, 6144, 4096)` and `(32768, 4096, 14336)` (R40B
   flake-risk), 5 runs each, decide PASS/FAIL deterministically.

**Predicted outcome**: confirms or denies the projected 26/42 baseline.
If R40A/R40C rescues hold, +2 shapes lock in. If R40B flake-risks fall
either way, baseline shifts ±2.

**Effort**: 1 agent-hour.

---

## 3. Reviewer protocol (post-R41)

For each R41_OPT_X candidate that the optimizer flags as PASS:

1. **5-run consensus mandatory** (per `R40_REVIEWER_REPORT.md` decision —
   3-run inflated R40B by +1). Use `bench_all_42_R39B.py` 5x with
   warmup=200, iters=500, trim=0.10. Compute `wcf_mean`, `wcf_std`,
   `wcf_max` across 5 runs. Promote ONLY if `wcf_max < 2.0%` AND
   `wcf_std < 1.0%`.

2. **R40B baseline preservation check**: re-bench all 22 stable R40B
   PASSes under the new candidate (if it touches their variant). Any
   regression (PASS → WRONG, or > 5% perf drop) is a STOP.

3. **R35-signature audit**: for any new PASS shape, dump per-cell wrong
   map and verify the upper-left 128×128 quadrant of every 256×256 tile is
   < 1% wrong. If the quadrant pattern persists, the R35 hypothesis-3
   register-file race is still active and the candidate is masking, not
   fixing.

4. **Perf-vs-aiter sanity**: `tflops/comp` ratio reported per new PASS.
   - `< 80%`: serious — flag for follow-up but accept if correctness gain.
   - `80-90%`: log as LOSE_CORRECT (accept).
   - `≥ 90%`: log as healthy.
   - `≥ 100%`: WIN. Update leaderboard.

5. **Flake-risk policy**: any shape with `wcf_max in [1.5%, 2.0%]` across
   5 runs is flagged for **per-rebuild re-verification**, not promoted to
   stable.

---

## 4. Recommended dispatch order

**Wave 1 (parallel, immediate launch)**:
- **R41 OPT A** (Cluster C deep-K) — 1 worktree, 1 optimizer agent.
- **R41 OPT B** (Cluster B variant retune) — 1 worktree, 1 optimizer agent.
- **R41 OPT D** (reviewer 5-run verify R40A/R40C/flake-risk) — 1 reviewer
  agent, no worktree (uses existing R40A/R40C builds).

These are all independent (R41A touches kernel macro, R41B is build-flag
only on a different module set, R41D is bench-only).

**Wave 2 (gated on R41A result)**:
- If R41A recovers ≥ 3 of 5 catastrophic shapes → skip R41C.
- If R41A recovers ≤ 2 → launch R41 OPT C as insurance on remaining
  catastrophic shapes.

**Wall-clock budget**: ~6h (R41A is the slowest at 4h; B and D run in
parallel under it).

---

## 5. Per-shape integration plan (post-R41)

The kernel will need a per-shape **build manifest** that records, for each of
the 42 shapes, which macro stack to compile against. The current R40B
manifest (`R40B_BUILD_MANIFEST.json`) is the starting point.

### Macro stack legend
- `B` = R40B base (`FUSED_STEP34=1`, `R25C_TAIL_PF_OFF_ITERS=0`, no memc-flag)
- `B+A` = R40B base + `R40A_PF_FENCE=1`
- `B+C` = R40B base + `R40C_LDS_DRAIN=1`
- `B+R41A(po=N, ef=B)` = R40B + `R41A_DEEP_K_FIX=1` with override params
- `B+R41B(<variant>)` = R40B with overridden BEST_VARIANTS variant for that shape
- `B+R41C(<no_btw_all|no_dc|both>)` = R40B with stripped suffix

### Integration table (provisional — fill in as R41 results arrive)

| Cluster | Shapes | Default stack | Per-shape overrides |
|---|---|---|---|
| 22 R40B stable | (see R40_REVIEWER_REPORT.md table) | `B` | none |
| 2 R40B flake-risk | (16384,6144,4096), (32768,4096,14336) | `B` | re-verify each rebuild |
| R40A rescue | (4096, 32768, 128256) | `B+A` | promote on R41D 5-run PASS |
| R40C rescue | (16384, 4096, 14336) | `B+C` | promote on R41D 5-run PASS |
| Cluster C target | 5 K=32768 shapes | `B+R41A(...)` or `B+R41C(...)` | per-shape from R41A/C sweep winner |
| Cluster B target | 11 near-gate shapes | `B+R41B(<best>)` per shape | per-shape from R41B sweep |
| CRASH | (16384,4096,28672), (4096,32768,28672) | TBD | defer to R42 |

### Conflict policy
If a shape qualifies for multiple stacks (e.g. R41B retune AND R41A fence
both pass for the same shape), pick the one with **higher tflops at equal
correctness**. Tie-break by lower `wcf_mean`.

### Build pipeline change
`build_R41.py` (new) should:
1. Read `R41_INTEGRATION_MANIFEST.json` (this round's per-shape stack).
2. Generate one `.so` per shape with that shape's exact macro set.
3. Refuse to build if a shape has macros for `B+A AND B+C` simultaneously
   (incompatible — both touch step12 boundary).
4. Output `R41_BUILD_MANIFEST.json` for the bench harness.

---

## 6. Bench discipline (UNCHANGED from R40)

- `bench_all_42_R39B.py`, warmup=200, iters=500, trim=0.10, random
  scales [-2,2], seed=42, `wrong_cell_frac < 2% AND snr_med >= 10 dB AND
  finite >= 0.99`.
- 3-run consensus for optimizer-side verdicts; **5-run mandatory for
  reviewer promotion**.
- 8 GPUs available; check `rocm-smi` first.
- Do NOT use `bench_all_42_R37.py` (uniform-scale gate inflates).

---

## 7. R41 ceiling forecast

| Scenario | R41A | R41B | R41C | R41D | Total |
|---|---|---|---|---|---|
| Baseline (R40B + per-shape A+C) | 0 | 0 | 0 | +2 | **26** |
| R41B half-hits (5 of 11) | 0 | +5 | 0 | +2 | **31** |
| R41B half + R41A two-of-five | +2 | +5 | 0 | +2 | **33** |
| R41B half + R41A four-of-five | +4 | +5 | 0 | +2 | **35** |
| R41B all + R41A five + R41C insurance | +5 | +9 | 0 | +2 | **40** |
| Optimistic ceiling (only CRASH-2 left) | +5 | +9 | 0 | +2 | **40** |

Most-likely outcome (median across cells): **31-35/42 verified-correct after R41**.

CRASH-2 deferred. Closing those structural failures requires a different
tile decomposition (tracked separately as R42 work).

---

## 8. Files this plan references

- `R40B_5RUN_CONSENSUS.json` — 24/42 + 2 flake-risk verdict
- `R40_REVIEWER_REPORT.md` — integration policy and projected 26/42
- `R40_OPT_B_VERDICT.md` — FUSED_STEP34 mechanism
- `R40_OPT_A_VERDICT.md` — pf-fence rescues 1, regresses 1 broadly
- `R40_OPT_C_VERDICT.md` — lgkmcnt drain rescues 1, regresses 1 broadly,
  R35 quadrant pattern persists (key data for cluster C diagnosis)
- `R40_OPT_D_VERDICT.md` — no-prefetch IMPROVES 4 of 5 cluster-C shapes
  (key data for R41A hypothesis (a))
- `R35_WRONG_CELLS_DIAGNOSIS.md` — upper-left 128×128 quadrant signature
- `R37_LEADERBOARD.md` — first valid 14/42 baseline
- `R40B_BUILD_MANIFEST.json` — current per-shape variant map
- `R38_BEST_VARIANTS_v3.py` — variant axis enumeration source for R41B sweep
- `bench_all_42_R39B.py` — required bench harness for all R41 measurements
- `build_R40B.py` — template for R41A/B/C build harnesses
