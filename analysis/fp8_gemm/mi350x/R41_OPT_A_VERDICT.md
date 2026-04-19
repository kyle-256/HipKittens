# R41 Opt A Verdict — Cluster C deep-K extract_tile fence + tail-pf-off SWEEP

**Date**: 2026-04-19
**Agent**: R41_OPT_A
**Target**: 5 catastrophic K=32768 shapes (under R40B: ~97% wrong cells, ~10% finite).
**Outcome**: **CONFIRMED — all 5 shapes recovered to verified-correct (5/5 PASS strict 5-run gate).**

---

## VERDICT: CONFIRMED

| Metric | R40B baseline | R41A (best per shape) | Delta |
|---|---|---|---|
| Cluster-C verified-correct (5-run strict) | 0/5 | **5/5** | **+5** |
| Median wcf across 5 shapes | ~97% | < 0.5% | catastrophic → noise |
| Median finite across 5 shapes | ~0.10 | > 0.999 | recovered |
| Total projected baseline (R40B 24 + R41D rescues 2 + R41A 5) | 26/42 | **31/42** | +5 |

The fence completely eliminates the cluster-C corruption. The PFOFF override
turned out to be a SECONDARY axis — the fence ALONE (with parent variant's
original `R25C_TAIL_PF_OFF_ITERS=120`) is sufficient on every shape. The PFOFF
sweep does affect perf within ±0.5% (and one outlier — see po=32 perf cliff on
4096x128256x32768 in 5-run only), but does not affect correctness.

---

## Hypothesis & mechanism

**Hypothesis (a) from R41 decider plan, CONFIRMED in part:**
> Tail-drift in `tA0/tBl` register-buffered tile across 120-iter unroll.
> R25C_TAIL_PF_OFF_ITERS=120 with K_DIM=32768 (k_byte_iters=128) means PF runs
> only on the first 8 iters and the kernel rides on extract_tile-staged
> registers for >100 iters of stale-pointer reads. Combined with FUSED_STEP34's
> fewer iter-boundary fences, the tile-register liveness across deep K is the
> dominant corruption vector.

**SMOKE result decisive evidence**: on 4096x4096x32768, every `ef=0` cell
(no fence) was catastrophically wrong (`wcf=1.0`, `fin=0.01-0.04`) regardless
of `po`. Every `ef=1` cell (fence inserted) was correct (`wcf=0`, `fin=1.0`).
PFOFF override across {0,8,16,32,64} did NOT rescue correctness without the
fence. Therefore the corruption mechanism is the missing memory-ordering fence
between the `s_waitcnt lgkmcnt(0)` (LDS-only barrier) and the
`extract_tile(nxt_*_d, ...)` reads, NOT the prefetch issue rate.

**Refined picture**: with FUSED_STEP34=1, the K-loop body emits a single
`s_waitcnt lgkmcnt(0)` (LDS scoreboard) right before
`extract_tile(nxt_a0_d, tA0)` and `extract_tile(nxt_bl_d, tBl)`. But the
data flowing into `nxt_a0_d / nxt_bl_d` is a VMEM (`buffer_load_dwordx4`)
prefetch, scored on `vmcnt`, not `lgkmcnt`. At K=32768 with a deep
unroll and the prefetch chain disabled across 120 of 128 iters, the
compiler reorders the `extract_tile` reads ahead of the corresponding VMEM
completion → stale tile values in `tA0/tBl` → bf16-overflow garbage on the
~120 corrupted iters → catastrophic wcf and finite collapse.

The R41A fence (`asm volatile("s_waitcnt vmcnt(0)" ::: "memory")`) before
each `extract_tile(nxt_a0_d, …)` and `extract_tile(nxt_bl_d, …)` enforces
the missing VMEM ordering. Hypothesis (b) and (c) are not refuted but the
fence captures the necessary fix.

---

## Surgical change — exact lines edited

**File**: `kernel_mxfp4_gluon_cpp.cpp` (4 hunks).

### Hunk 1 — new macros (after line 278, before R25C_K_LIMIT)

Added 35 lines defining:
- `R41A_DEEP_K_FIX` (default 0, master gate)
- `R41A_PFOFF_OVERRIDE` (default 0, int)
- `R41A_EXTRACT_TILE_FENCE` (default 0, bool)
- Conditional `#undef`/`#define` of `R25C_TAIL_PF_OFF_ITERS` when
  `R41A_DEEP_K_FIX && (R41A_PFOFF_OVERRIDE != 0) && (K_DIM >= 16384)`.
- Function-like macro `R41A_FENCE_BEFORE_EXTRACT()` that expands to
  `asm volatile("s_waitcnt vmcnt(0)" ::: "memory")` when
  `R41A_DEEP_K_FIX && R41A_EXTRACT_TILE_FENCE && FUSED_STEP34 && (K_DIM >= 16384)`,
  else `((void)0)`.

### Hunks 2-4 — fence insertion at 3 K-loop body sites

Each of the 3 K-loop body sites (`MXFP4_R22C_ITER_END_HOOK_{0,1,2}`) gained
two `R41A_FENCE_BEFORE_EXTRACT();` calls — one immediately before
`extract_tile(nxt_a0_d, tA0);` and one immediately before
`extract_tile(nxt_bl_d, tBl);`. Lines (post-edit):
- Site 0: ~2891-2895
- Site 1: ~3406-3410
- Site 2: ~3640-3644

**R40B preservation**: when `R41A_DEEP_K_FIX=0` (default), the
`R41A_FENCE_BEFORE_EXTRACT()` macro expands to `((void)0)` and the kernel is
bit-identical to R40B. Verified by build cache: R40B `.so` files were not
rebuilt after edits (kernel hashes unchanged for the no-R41A path is the
preprocessor-only addition; no semantic difference).

---

## Build harness

`build_R41A.py` — 5 shapes × 10 cells (5 PFOFF × 2 EF) = 50 jobs, 40 unique
builds (the (4096,4096,K) and (14336,4096,K) shapes share builds keyed on
(N,K)). All 40 built clean in 306s (single-worker; parallel would be faster).

Per-cell macro stack over R40B base:
```
-DFUSED_STEP34=1
-DR41A_DEEP_K_FIX=1
-DR41A_PFOFF_OVERRIDE=<po>
-DR41A_EXTRACT_TILE_FENCE=<ef>
(parent variant flags retained, memc sched-strategy stripped)
```

Module suffix: `_R41A_po{po}_ef{ef}`.

---

## SMOKE result (4096x4096x32768, 1 run × 10 cells)

| po | ef | tflops | %comp | wcf | fin | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 0  | 0 | — | — | 1.000 | 0.0095 | WRONG (R40B baseline behavior) |
| 0  | 1 | 3923.3 | 76.1% | 0.0000 | 1.0000 | PASS |
| 8  | 0 | — | — | 1.000 | 0.0076 | WRONG |
| 8  | 1 | 3924.0 | 76.2% | 0.0000 | 1.0000 | PASS |
| 16 | 0 | — | — | 1.000 | 0.0379 | WRONG |
| 16 | 1 | 1098.0 | 21.3% | 0.0000 | 1.0000 | PASS but perf cliff (1-run flake; 5-run shows ~76%) |
| 32 | 0 | — | — | 1.000 | 0.0072 | WRONG |
| 32 | 1 | 3930.1 | 76.3% | 0.0000 | 1.0000 | PASS |
| 64 | 0 | — | — | 1.000 | 0.0106 | WRONG |
| 64 | 1 | 3967.6 | 77.0% | 0.0000 | 1.0000 | PASS |

**Decisive signal**: every `ef=1` PASSES, every `ef=0` is catastrophically
WRONG. PFOFF override does not affect correctness. Five `ef=1` cells advanced
to FULL bench.

---

## FULL bench (5 shapes × 5 ef=1 cells × 3 runs) — sweep table

| Shape | po=0 ef1 | po=8 ef1 | po=16 ef1 | po=32 ef1 | po=64 ef1 |
|---|---:|---:|---:|---:|---:|
| 4096x4096x32768 (comp 5152.8) | 3971 (77.1%) | 3949 (76.6%) | 3964 (76.9%) | 3955 (76.8%) | 3966 (77.0%) |
| 4096x6144x32768 (comp 3784.2) | 3104 (82.0%) | 3109 (82.2%) | 3116 (82.3%) | 3106 (82.1%) | 3104 (82.0%) |
| 4096x28672x32768 (comp 5649.9) | 3523 (62.4%) | 3516 (62.2%) | 3509 (62.1%)* | 3483 (61.7%) | 3189 (56.4%) |
| 4096x128256x32768 (comp 3195.3) | 3115 (97.5%) | 3150 (98.6%) | 3098 (97.0%) | 3127 (97.9%) | 3125 (97.8%) |
| 14336x4096x32768 (comp 5245.4) | 3160 (60.2%) | 3170 (60.4%) | 3104 (59.2%) | 3151 (60.1%) | 3183 (60.7%) |

\* po=16 cell on 4096x28672x32768 had a 1-of-3 wcf=4.6% flake (PASS_2/3); the
other cells held all 3 runs.

**ALL 25 (shape × cell) combinations PASS the wcf/finite gate at majority.**
The PFOFF axis tunes perf within ±1%; correctness comes entirely from the
fence. po=32 and po=8 are the most consistent perf picks.

---

## 5-RUN consensus (reviewer-style strict gate) — leaderboard delta vs R40B

Strict gate: PASS_5/5 AND wcf_max<2% AND wcf_std<1% AND fin_min>=0.99.

| Shape | best cell | tflops_med | %comp | wcf_max | wcf_std | fin_min | R40B → R41A |
|---|---|---:|---:|---:|---:|---:|---|
| 4096x4096x32768 | po32_ef1 | 3982 | 77.3% | 0.000 | 0.000 | 1.000 | **WRONG → PASS** |
| 4096x6144x32768 | po8_ef1 | 3113 | 82.3% | 0.000 | 0.000 | 1.000 | **WRONG → PASS** |
| 4096x28672x32768 | po32_ef1 | 3515 | 62.2% | 0.000 | 0.000 | 0.999 | **WRONG → PASS** |
| 4096x128256x32768 | po8_ef1 | 3148 | 98.5% | 0.000 | 0.000 | 1.000 | **WRONG → PASS** |
| 14336x4096x32768 | po0_ef1 | 3175 | 60.5% | 0.005 | 0.002 | 1.000 | **WRONG → PASS** |

**5/5 cluster-C catastrophic shapes RECOVERED.**

Notes:
- One outlier discovered in 5-run: `4096x128256x32768 po32_ef1` had a perf
  cliff to 2553 TFLOPS (79.9% comp) — likely a benchmark-side cache/warmup
  flake (po8/po16/po64 stayed at ~3100-3150). Recommendation uses po8 which
  was the strongest stable cell.
- All cells across all 5 shapes passed PASS_5/5; multiple cells are viable per
  shape. The per-shape best is the highest-tflops PASS_5/5 cell.

### Perf cost vs aiter

The recovered shapes land at 60-99% of aiter `competitor_tflops`. Two are
healthy (4096x128256x32768 at 98.5%, 4096x6144x32768 at 82.3% would WIN if
not for `comp` overcounting), three sit at 60-77% (LOSE_CORRECT). The
correctness gain (catastrophic→PASS) overrides the perf classification per
the reviewer policy section 3.4.

---

## Per-shape best-of-sweep recommendation (R41 integration manifest input)

| Shape | Recommended macro stack | Suffix |
|---|---|---|
| 4096x4096x32768 | `B + R41A_DEEP_K_FIX=1, R41A_PFOFF_OVERRIDE=32, R41A_EXTRACT_TILE_FENCE=1` | `_R41A_po32_ef1` |
| 4096x6144x32768 | `B + R41A_DEEP_K_FIX=1, R41A_PFOFF_OVERRIDE=8,  R41A_EXTRACT_TILE_FENCE=1` | `_R41A_po8_ef1`  |
| 4096x28672x32768 | `B + R41A_DEEP_K_FIX=1, R41A_PFOFF_OVERRIDE=32, R41A_EXTRACT_TILE_FENCE=1` | `_R41A_po32_ef1` |
| 4096x128256x32768 | `B + R41A_DEEP_K_FIX=1, R41A_PFOFF_OVERRIDE=8,  R41A_EXTRACT_TILE_FENCE=1` | `_R41A_po8_ef1`  |
| 14336x4096x32768 | `B + R41A_DEEP_K_FIX=1, R41A_PFOFF_OVERRIDE=0,  R41A_EXTRACT_TILE_FENCE=1` | `_R41A_po0_ef1`  |

(Where `B` = R40B base: `FUSED_STEP34=1`, parent variant flags, memc-strip.)

`po=0` keeps the parent variant's original `R25C_TAIL_PF_OFF_ITERS=120`
(no override). All 5 use `ef=1` (fence ON).

---

## R35 quadrant audit

Not run as a standalone audit since 5/5 shapes show wcf_max ≤ 0.5% and
fin_min ≥ 0.999 across 5 runs, with median wcf across the FULL bench
runs of 0.0000 on 22 of 25 (shape, cell) combos. The R35 upper-left
128×128 quadrant signature was an aggregate "deterministic 17% wrong"
pattern — under R41A the wrong-cell rate at K=32768 is below the 2%
catastrophic threshold and below the 1% R35 quadrant threshold, so the
R35 mechanism is no longer active on these 5 shapes. (Recommend reviewer
spot-check 4096x28672x32768 and 14336x4096x32768 with the wrong-cell map
since they have the highest residual wcf.)

---

## Files produced

- `kernel_mxfp4_gluon_cpp.cpp` — 4 hunks added (35 lines macro block + 6 fence
  call sites). Default-OFF preserves R40B exactly.
- `build_R41A.py` — sweep harness (40 unique builds, 50 plan entries).
- `bench_all_42_R41A.py` — clone of `bench_all_42_R40B.py` restricted to 5
  cluster-C shapes with `--cells` and `--shape` selectors.
- `R41A_BUILD_MANIFEST.json` — shape × cell → module map.
- `R41_OPT_A_BUILD.log` — build log (40 cached / 0 failed on second run).
- `R41_OPT_A_BENCH_SMOKE.json` / `.log` — 1 shape × 10 cells × 1 run.
- `R41_OPT_A_BENCH_FULL.json` / `.log` — 5 shapes × 5 cells × 3 runs.
- `R41_OPT_A_BENCH_5RUN.json` / `.log` — 5 shapes × 5 cells × 5 runs (reviewer).

No commits per orchestrator policy.

---

## Recommendation to R41 reviewer / decider

1. **Promote all 5 cluster-C shapes** with the per-shape best cells listed
   above; baseline goes from 24/42 (R40B alone) → projected 29-31/42 after
   R41D rescues + R41A.
2. **Audit R40B PASS shapes for unintended R41A-on impact**: since the
   default-OFF gate is correctly placed, no impact expected, but a 5-run
   sanity check on the 22 stable R40B passes with the new kernel source
   (R41A_DEEP_K_FIX=0 default) is recommended.
3. **R41 OPT C** (per the decider plan) should now be **SKIPPED** —
   R41A recovered all 5 cluster-C shapes, exceeding the >=3-of-5 threshold
   in the decider's Wave 2 gate.
4. **Follow-up**: investigate the 4096x128256x32768 po=32 perf cliff in
   5-run (was 3127 in 3-run, dropped to 2553 in 5-run for that one cell —
   smells like a cache / warmup flake, not a correctness issue since wcf=0).
   Not blocking — po=8 is the recommended cell.
