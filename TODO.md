# MXFP4 Optimization TODO

**Last update:** 2026-04-20 (R64)
**Status:** 12/42 WIN, mean ~93.6% of aiter (+0.5pp from R63 via expanded autotune; **no NEW WIN**)
**Bench harness:** `analysis/fp8_gemm/mi350x/bench_all_42.py` (HipKittens-only, **11 variants**, parallel-GPU)

---

## Hard rules (do not violate)

1. **No aiter binary substitution.** Every reported number must come from a HipKittens kernel built from `kernel_mxfp4_gluon_cpp.cpp`. Past incident: R50–R61 used aiter `.co` to fake 42/42; user is intolerant of repeats.
2. **Bench protocol:** warmup=200, iters=500, trim_frac=0.10, GPU isolation via `HIP_VISIBLE_DEVICES`.
3. **Compete against `competitor_tflops`** (aiter ASM via Python dispatcher) embedded in `bench_all_42.py`.
4. **Commit only when there is measurable effect.** Inert scaffolding stays out of git.

## Current standing

| Cluster | Shapes | Status |
|---|---:|---|
| Easy WINs (M-large, K-small) | 12 | already winning (incl. 16384×6144×4096 R63 unlock via gm8) |
| Boundary (~95% to ~99.5%) | 6 | needs +0.5pp to +5pp; knob-tuning largely exhausted |
| LOSE: M=4096 K-large | 8 | -7% to -24% gap (root cause: inner-loop scheduling) |
| LOSE: K-large general | 16 | -10% to -25% gap |

Worst losers (4096×*×K, K≥16384): 67–82% of aiter. These dominate the headline gap.

R63 boundary residue (still LOSE):
  16384x14336x4096   95.6%   (no variant tried crosses 100%)
  4096x32768x4096    96.8%   (gm6 best; +0.5pp from R62)
  16384x28672x2048   96.1%   (gm6 best; -0.2pp noise from R62)
  4096x4096x8192     98.8%   (gm8 best; +2.4pp from R62 default)
  16384x4096x6144    98.1%   (unchanged; gb best now)
  32768x28672x2048   99.1%   (gm6 best; -0.3pp noise from R62)

---

## Productive axes (not yet exhausted)

### A. Inner-loop rewrite (HIGH priority)
- The current `kpair_64mfma_step34` emits 8 ds_reads then 24 pure MFMAs per Step.
- aiter emits an evenly-spread MFMA:ds_read:buffer_load 4:1:1 pattern.
- R50A tried to spread inside the existing asm volatile body but only addressed cohort-race; no perf win.
- **Next try:** rewrite the entire 64-MFMA inner kernel from scratch using aiter's emission template (see `project_mxfp4_aiter_disasm_findings.md`). Do NOT try to splice — the schedule is fundamental, not a knob.

### B. Different MFMA shape
- Current uses 16×16×128. UNTRIED: 32×32×64 with `mfma_scale` cbsz/blgp variants.
- Structurally different AGPR forwarding chain — would either close cohort race or open a new performance regime.

### C. Per-shape tile dimensions
- aiter uses 224×256 / 192×256 for the worst clusters.
- HipKittens is locked to BLK_M=BLK_N=256.
- Adding a 192×256 tile path may unlock 4096×K-large shapes specifically.

### D. K-pair count tuning
- Currently 2 KPairs per K iteration. aiter uses different KPair counts per shape.
- Cheap to try: extend `bench_all_42.py` variants with `-DKPAIRS_PER_ITER={1,2,4}`.

---

## Closed axes (don't reopen)

- **Fence positioning** in/around step34 (R45B, R47A, R49A, R49C, R50A — 5 closures)
- **MFMA↔ds_read 1:3/1:4 interleaving** inside existing `kpair_64mfma_step34` (R50A — ISA-verified emit but no win)
- **STEP34_INTERLEAVED scaffolding** (R62, 2026-04-20): added flag + 4 call-site gates, but `kpair_64mfma_step34_interleaved` function (already in tree at line 1432) does not compile when actually used — emits `ds_read_b128` with operands that resolve to AGPRs, producing 20+ "invalid operand for instruction" errors. Don't enable until that function is rewritten.
- **`UNROLL_K=2/16` per-shape variants** (R62): added to bench but unlock 0 NEW WINs (only re-rank already-winning shapes by ~0.5pp).
- **`asm_inline` "5084 TFLOPS" reference** — REVOKED 2026-04-17, that kernel is numerically incorrect (SNR -1.31 dB).
- **`STEP3_PF_N` / `STEP4_PF_N` > 8** (R63): static_assert `PF_N <= 2*PF_MPT=8` — pf=12,16 unbuildable.
- **`STEP3_BARRIER_VMCNT` ∈ {4, 12}** (R63): ≤0.5pp delta on boundary shapes; mostly regression. Default 8 stands.
- **`R25C_TAIL_PF_OFF_ITERS` ∈ {1, 2, 4}** (R63): macro is dead at FUSED_STEP34=1 (R63's `_BASE`). At FUSED_STEP34=0 it activates but causes HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION crashes at K=14336. Bug in tail-PF-off path; don't reopen without first fixing the in-flight buffer_load tracking.
- **`STEP4_EXTERNAL_BR_PREFETCH`** (R63): macro doesn't exist anymore (removed in cleanup commit a70e4a15 / 5405bdb4). Stale comment at line 2516.
- **`192×256` tile path** (R63 Opt-3 feasibility): NO-GO for now. Architectural rewrite — `BLK=256` is conflated as both M and N at ~30 sites; 10 `kpair_*` asm functions have 4×4 MFMA grid hard-numbered into operand slots; would need ~500-800 LOC parallel asm bodies. Must follow inner-loop rewrite (axis A) so we don't duplicate code we plan to throw away. `224×256` is strictly less feasible (224 is not a multiple of 64 for scale super-group).
- **`R50A_AITER_INTERLEAVE` macro** (R64): macro was removed in cleanup commit a70e4a15 along with ~250 lines of asm that resurrecting now would be wasted work — the schedule it implemented is the very same one the inner-loop rewrite (axis A) needs to redo from scratch. Don't resurrect; rewrite.
- **`AGPR_REGS_HINT_192` macro** (R64): neutral/weak across all sweeps — never sole best on any shape. Don't enable as default.
- **`WAVES_PER_EU_1`+`GM=8` and `WAVES_PER_EU_1`+`AGPR_REGS_HINT_192` combos** (R64): regress on most shapes; the individual `we1` variant is in autotune but the combos add nothing.
- **R64 isolated boundary "WINs"** (R64): three shapes (4096x4096x8192, 16384x4096x6144, 32768x28672x2048) crossed 100% on isolated re-bench but reverted to LOSE in the full sweep — boundary noise, not algorithmic progress. Don't claim future WINs without ≥3-run isolated confirmation AND a full-sweep confirmation.

---

## R64+ priorities

1. **Inner-loop rewrite** (Axis A) — biggest unrealized lever; do this before micro-knobs or tile-size variants. The existing `kpair_64mfma_step34` (line 863, 64 MFMAs in one asm body) needs full re-emit using aiter's 4:1:1 MFMA:buffer_load:ds_read schedule. Reference: `project_mxfp4_aiter_disasm_findings.md`.
2. **MFMA 32×32×64** (Axis B) — exploratory; only if Axis A stalls. Different AGPR forwarding chain.
3. **192×256 tile path** (Axis C) — defer to R65+ AFTER axis A makes the kpair body template-friendly.

## Bench script (R64 working set)
`bench_all_42.py` variants (11 total):
  `default | gm6 | gb | gb_gm6 | unr2 | unr16 | gm8 | we1 | tbv16 | unr2_gm6 | gb_unr2`
Run: `BENCH_GPUS=0,1,2,3,4,5,6,7 python3 bench_all_42.py` (~10 min for build+full sweep).
Single-shape autotune: `python3 bench_all_42.py M N K` picks best variant.

## Standing GPU/timing protocol
- GPUs 0-7 all available on this MI355X box; check with `rocm-smi --showuse` first.
- Bench result variance: ±2pp on borderline shapes is normal. Boundary WINs (within 1pp of 100%) need re-bench on isolated GPU to confirm.
- The 16384×4096×28672 shape showed a 43% reading in one full sweep that re-benched at 74.5% — interpret single-cell anomalies as contention noise, not regression.
- **R64 lesson**: full-sweep contention systematically depresses borderline TFLOPS by 2-4pp vs isolated runs. A WIN must show in BOTH isolated re-bench AND full sweep before being claimed — see R64 closures for three "WINs" that didn't survive.
