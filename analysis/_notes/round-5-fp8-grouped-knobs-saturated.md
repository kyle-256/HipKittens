# Round 5 — FP8 grouped RCR: K-tail full hoist FAILS, all simple knobs saturated

**Date**: 2026-05-01
**HK SHA (pre)**: 333074d6 (round-4 sched_barrier removal)
**HK SHA (post)**: this commit (notes-only; code unchanged)
**Primus-Turbo SHA**: c3b70e374 (unchanged)
**Round-5 baseline metric**: 791 (round-4 ship was 793, noise band ±2)
**Round-5 result metric**: 791 (no code change; falsifications only)

## 1. Falsified strategies

### 1.1 Strategy A — full K-tail load HOIST before Epilog 2 (CATASTROPHIC FAIL)

**Hypothesis**: hoist all 12 K-tail `raw_buffer_load_b128` before Epilog
2's `s_waitcnt vmcnt(0)`. K-tail HBM round-trip (~150 cyc) overlaps
with Epilog 2's 4 mfma + barriers (~120-180 cyc). Epilog 2's vmcnt(0)
drains both main-loop and K-tail vmem in one wait. Post-Epilog 2 K-tail
degenerates to 4 mfma only.

**Implementation**: added `A_row_reg a_kt0; B_row_reg b0_kt, b1_kt;`
function-scope (round-3 already had `a_kt1`). Restructured K-tail to
issue all loads via these dedicated tiles BEFORE Epilog 2.

**Build resources stayed identical** (VGPRs 256 / Spill 67 / LDS 139796
B / Scratch 272 / Occupancy 2) — compiler "fitted" the new tiles in.

**Probe (gpt_oss-GateUP-B32-M4096)**: HK 1252 → **1087 TF (-13.2 %)**.

**Metric (focus=gpt_oss)**: score **791 → 736 (-55)**. grp_FP8 geomean
**0.867 → 0.753 (-11.4 pp)**. DSV3 [watch] below_1.0 went 5/16 → 6/16
(no correctness FAIL — pure perf regression).

**Root-cause analysis**: even though the `VGPRs Spill: 67` count was
unchanged, the spill DISTRIBUTION shifted. Adding 3 more named register
tiles forced the regalloc to choose DIFFERENT live ranges to spill —
the new spill victims landed on main-loop hot-path variables. Hot-path
spills ≈ 4-8 cyc per access × 22 K-iters × hundreds of references =
catastrophic. The build remark only counts STATIC spill slots, not
runtime spill-frequency. **REVERTED via `git checkout --`**.

**Lesson**: fn-scope register tile additions in this kernel are NOT
free even when the static spill count looks unchanged. Future K-tail
hoist attempts must (a) gate the new tiles via `if constexpr` template
specialisation (so dense kernels don't pay), or (b) restructure the
K-tail block to use main-loop tiles after they're dead, eliminating
the need for new declarations.

### 1.2 RCR_STEADY_VMCNT sweep {4, 8, 12} — INSENSITIVE

`RCR_STEADY_VMCNT` controls the main-loop wait threshold (wait until
≤N outstanding vmem). Default=8. Sweep on focus shape:

| value | HK TFLOPS |
|---|---|
| 4  | 1253 |
| 8  | 1252 (baseline) |
| 12 | 1253 |

All within ±1 TF (0.08 %). Knob is saturated/insensitive at the
current main-loop instruction mix. **REVERTED**.

### 1.3 RCR_PREFETCH_LGKM sweep {2, 4, 8} — INSENSITIVE

`RCR_PREFETCH_LGKM` controls main-loop wait for LDS visibility (wait
until ≤N outstanding LDS). Default=4. Sweep on focus shape:

| value | HK TFLOPS |
|---|---|
| 2 | 1252 |
| 4 | 1252 (baseline) |
| 8 | 1254 |

All within ±2 TF (0.16 %). Knob is saturated. **REVERTED**.

### 1.4 Down-B4-M4096 config sweep — round-69 rule already optimal

Round-5's worst-ratio gpt_oss FP8 shape is grpFP8-Down-B4-M4096 (ratio
0.831). Tried changing the round-69 rule from `(group_m=4, num_xcds=4)`
to `(group_m=4, num_xcds=8)` (revert to default xcds):

| variant | Down-B4-M4096 ratio | grp_FP8 geomean |
|---|---|---|
| (4, 4) baseline | 0.831 | 0.867 |
| (4, 8) | 0.827, 0.828 | 0.867, 0.867 |

xcds=8 slightly regresses (-0.4 pp on the focus shape, 0 on geomean).
Round-69's xcds=4 is at the local optimum. **REVERTED**.

Wider sweep `(group_m, xcds) ∈ {1,2,4,8,16} × {1,2,4,8}` via
in-process monkeypatch was misleading (see §3 below) — the metric
dispatch caches the config function reference at import time, so the
monkeypatch only takes effect on the first call which serves as
warm-up. Direct rule edits + full metric runs are the only reliable
sweep mechanism.

## 2. Why all the small levers are saturated

After round-3 (K-tail single-wait, +0.7 pp) and round-4 (sched_barrier
removal, +0.45 pp), the FP8 grouped RCR main loop has ~8 SALU
instructions per K-iter (8 s_barrier + 8 s_setprio + 4 s_waitcnt + 0
sched_barrier). Per-K-iter is now bounded by:

* **MFMA latency** — 4 `mma_ABt` × ~32 cyc throughput = 128 cyc
* **Cross-warp s_barrier** — 8 × ~8 cyc = 64 cyc (load-bearing for LDS
  visibility; cannot be removed without a structural change)
* **HBM/LDS visibility waits** — ~50-80 cyc, mostly amortised

The non-amortisable component (mfma + barrier) is ~190 cyc per K-iter,
compared to Triton's ~150 cyc per equivalent K-iter (42.7 % MFMA Util
vs HK's 35.2 %). The remaining ~40 cyc gap per K-iter × 22 K-iter ×
46 tiles × 256 CUs / 2 GHz ≈ 470 ms of "lost" work per kernel call —
all of which lives in the cross-warp barrier + LDS staging path.

**No knob can recover this**; the gap is structural.

## 3. Methodology note — config sweep gotcha

In-process monkeypatching `cfg_mod.select_default_config` does NOT
work for sweeping HK config rules. The dispatch path imports the
function at module init and may bind it as a method reference; later
patching of the module attribute is not seen by callers.

Reliable sweep mechanism = direct edit of the rule in `config.py` +
`python3 scripts/_metric_grouped_only.py`. Each variant takes ~13 sec.
Budget 5-10 variants per round.

## 4. What's left for next rounds

Per round-3/4 roadmaps (still valid):

* **(b) Port dense 2-tile main loop to grouped_rcr_kernel** — dense
  uses 2-tile main when `ki ≥ RCR_TWO_TILE_MIN_KI=28` (line 1256). It
  halves per-K-iter SALU overhead (1 barrier set per 2 K-iter instead
  of per K-iter). For gpt_oss ki=22 even, lowering MIN_KI=22 + porting
  the 2-tile body is a 2-4 hr structural change. Estimated yield: 35
  → 38-40 % MFMA Util on focus shape, +3-5 pp ratio.

* **(c) Skip A-tile LDS staging** — Triton-style direct HBM → reg main
  loop. Removes `buffer_load_lds` + `s_barrier` + `ds_read` for A-tile
  (B-tile keeps LDS staging since it's reused across M-slabs). 4-8 hr
  structural rewrite. Estimated yield: closes most of the MFMA Util
  gap (35 → 42 %), +5-7 pp ratio.

* **(d) BF16 grouped K-tail single-wait port** — BF16 RCR K-tail (line
  842-853 in `kernel_bf16_dynamic.cpp`) still uses 2 vmcnt(0) waits.
  Porting round-3 FP8 single-wait pattern requires adding `A_tile_kt1`
  (analogous to FP8 `a_kt1`). Same risk profile as FP8 round-3 (vgpr
  add). Estimated yield: +0.3-0.5 pp on grpBF16 geomean.

* **(e) Consider WHEN to add new register tiles**. Round-3 added
  `a_kt1` and got +0.7 pp. Round-5 added 3 more tiles and got -55
  score. Hypothesis: 1 new tile is recoverable by regalloc; 3+ tiles
  cause cascading hot-path spills. Future hoist attempts should add
  at most ONE new tile per round.

Strategy A is **closed** (full K-tail hoist via 3 register tiles).
Future K-tail hoists must be incremental (one tile at a time, with
metric verification per tile).

## 5. Files touched

- `analysis/_notes/round-5-fp8-grouped-knobs-saturated.md` (this file)
- (No kernel or config code changes shipped this round.)
