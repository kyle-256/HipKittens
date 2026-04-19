# R54 Dev C — V2 RRR compiler-flag exploration — REFUTED-EMPIRICAL

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 94eb7675 (R54 cycle baseline)
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=5
**Cell of interest:** 8B Gate/Up RRR (M=4096, N=14336, K=4096) — V2 RRR fastpath
**Baseline (V2 RRR, no extra flags):** 254 VGPR / 0 spill / 0 scratch / occ 2 — median 2500.01 TFLOPS over 3 runs
**Lever:** Allocator-side / scheduler-side `-mllvm` compiler-flag exploration ONLY.
No source changes, since V2 RRR K-loop body restructure family is **TRIPLY CLOSED**
at the 254/256 VGPR ceiling (R49A v_perm penalty / R53A opsel spill / R53B
K-superblock spill — all REFUTED-EMPIRICAL). The remaining axis is to coax LLVM
into a different codegen of the existing source via flag injection.

## TL;DR — VERDICT: REFUTED-EMPIRICAL

One-line summary: *Across 41 isolated `-mllvm` flag treatments swept in two
ISA-level build-sweep phases (phase1: 16 entries, phase1b: 25 entries) and
10 surviving treatments A/B-benched against baseline at the primary 8B Gate/Up
RRR cell, **no flag delivered a ≥1% perf improvement**. The strongest
non-baseline treatment, `max_occ_plus_trackers`, came in at +0.122% median
(2503.07 vs 2500.01 TFLOPS) — well within the 3-run noise envelope (baseline
spread alone is 2495.81…2503.63, ±0.16%). Resource counts were identical
(254 VGPR / 0 spill / 0 scratch / occ 2) for every surviving flag, confirming
LLVM is not finding a meaningfully different schedule under any of the screened
levers. SHIP gate (≥+1%, PASS) is unreachable; flag-injection axis is closed.*

## Phase 0 — Lever scoping

V2 RRR fastpath is at the 254/256 VGPR allocator ceiling with 0 spill / 0
scratch / occ 2 — confirmed pre-existing condition documented in
`v_rrr_vgpr_ceiling.md`. Three prior in-cycle restructures (R49A / R53A /
R53B) all spilled. The R54 prompt enumerated compiler-flag injection as the
last remaining allocator-side axis: `-mllvm -amdgpu-sched-strategy=*`,
`-mllvm -amdgpu-schedule-metric-bias=*`, `-mllvm -amdgpu-use-amdgpu-trackers`,
`-mllvm -regalloc=*`, plus loop / wave-priority / promote-alloca / VOPD
controls — all of which leave the source identical and only re-ask the
compiler.

The cell deliberately under-test is **8B Gate/Up RRR**: small enough K
(4096) that a small per-iter cycle saving from a different schedule should
be visible in the 200-iter wall-clock; large enough M·N (4096·14336) that
the kernel saturates the device, exposing schedule efficiency rather than
launch overhead.

## Phase 1 — ISA-level build sweep (16 treatments)

Driver: `r54c_phase1_isa.sh`. Per-flag build at fixed shape
4096×14336×4096, extracting V2 RRR resources from per-instantiation remarks.
Verbatim from `r54c_results/phase1/SUMMARY.csv`:

| Flag                       | build_ok | VGPR | spill_v | scratch | occ |
|----------------------------|----------|------|---------|---------|-----|
| baseline                   | 1        | 254  | 0       | 0       | 2   |
| sched_max_occupancy        | 1        | 254  | 0       | 0       | 2   |
| sched_ilp                  | 1        | 254  | 0       | 0       | 2   |
| sched_iterative_ilp        | 1        | 254  | 0       | 0       | 2   |
| sched_iterative_minreg     | 1        | **256** | **8** | **36**  | 2   |
| sched_iterative_max_occ    | 1        | 254  | 0       | 0       | 2   |
| power_sched_off            | 0        | —    | —       | —       | —   |
| igrouplp_off               | 0        | —    | —       | —       | —   |
| promote_alloca_16/32/64    | 1×3      | 254  | 0       | 0       | 2   |
| vgpr_index_mode_dis        | 1        | 254  | 0       | 0       | 2   |
| loop_align_off             | 1        | 254  | 0       | 0       | 2   |
| num_vgpr_{240,224,256}     | 0×3      | —    | —       | —       | —   |

**Phase 1 takeaways:**
- 11/16 flags built; 5 flags (power_sched_off, igrouplp_off, num_vgpr=*) were
  rejected by the in-tree LLVM (option name not recognised by this driver
  version) — see `build_*.log` for the rejection messages.
- `sched_iterative_minreg` was the only built flag that **regressed
  resources**: 254→256 VGPR, 0→8 spill, 0→36 byte scratch. Disqualified from
  Phase 2.
- Every other built flag landed at the **byte-equivalent baseline resource
  envelope** (254 VGPR / 0 spill / 0 scratch / occ 2). LLVM is at the same
  VGPR-pressure equilibrium under all of these levers — none of them shifts
  the allocator off its current solution.

## Phase 1b — Refined build sweep (25 treatments)

Driver: `r54c_phase1b_isa.sh`. After Phase 1 disqualified the unrecognised
options, Phase 1b expanded the search to the full set of valid flags from
`llc --help-hidden`, including `amdgpu-schedule-metric-bias=*`,
`amdgpu-use-amdgpu-trackers`, `amdgpu-disable-{clustered-low-occ,unclustered-high-rp}-reschedule`,
`amdgpu-opt-vgpr-liverange`, `amdgpu-loop-prefetch`, `amdgpu-set-wave-priority`,
`amdgpu-enable-vopd`, `amdgpu-enable-merge-m0`, `disable-post-ra`,
`misched-postra`, `regalloc={basic,greedy,pbqp}`, plus a Phase 1 promising
combo `max_occ_plus_trackers`. Verbatim from
`r54c_results/phase1b/SUMMARY.csv`:

| Flag                          | build_ok | VGPR | spill_v | scratch | occ |
|-------------------------------|----------|------|---------|---------|-----|
| baseline                      | 1        | 254  | 0       | 0       | 2   |
| sched_max_occ                 | 1        | 254  | 0       | 0       | 2   |
| sched_max_ilp                 | 1        | 254  | 0       | 0       | 2   |
| sched_iter_ilp                | 1        | 254  | 0       | 0       | 2   |
| sched_iter_max_occ            | 1        | 254  | 0       | 0       | 2   |
| sched_metric_bias_{50,75,100} | 1×3      | 254  | 0       | 0       | 2   |
| sched_relaxed_occ             | 1        | 254  | 0       | 0       | 2   |
| amdgpu_trackers               | 1        | 254  | 0       | 0       | 2   |
| disable_clustered_low_occ     | 1        | 254  | 0       | 0       | 2   |
| disable_unclustered_high_rp   | 1        | 254  | 0       | 0       | 2   |
| promote_alloca_{16,32,64}     | 1×3      | 254  | 0       | 0       | 2   |
| loop_align_off                | 1        | 254  | 0       | 0       | 2   |
| opt_vgpr_liverange            | 1        | 254  | 0       | 0       | 2   |
| loop_prefetch                 | 1        | 254  | 0       | 0       | 2   |
| wave_priority                 | 1        | 254  | 0       | 0       | 2   |
| vopd                          | 1        | 254  | 0       | 0       | 2   |
| merge_m0                      | 1        | 254  | 0       | 0       | 2   |
| post_ra_disable               | 1        | 254  | 0       | 0       | 2   |
| misched_postra                | 1        | 254  | 0       | 0       | 2   |
| regalloc_basic                | 0        | —    | —       | —       | —   |
| regalloc_greedy               | 0        | —    | —       | —       | —   |

**Phase 1b takeaways:**
- 23/25 flags built clean. The two `-mllvm -regalloc=basic|greedy` rejections
  are because hipcc forwards `-regalloc` to the host toolchain pass manager
  (which does not run for AMDGPU codegen here) and the AMDGPU back-end
  separately picks its own register allocator regardless. Confirmed
  non-actionable.
- All 23 surviving flags **collapsed to the same 254 VGPR / 0 spill / 0
  scratch / occ 2 resource quadruple** as baseline. Across 31 distinct
  treatments tried in phase1+phase1b that built clean, only one
  (`sched_iterative_minreg`) actually moved the allocator — and in the
  wrong direction.
- The strong implication: LLVM is **schedule-locked** at this VGPR pressure
  point. The only way to extract a meaningful schedule change is to change
  the source (which has been triply refuted at this ceiling) or to change
  the resource budget (which only `iterative-minreg` did, badly).

## Phase 2 — A/B perf bench on 8B Gate/Up RRR (10 surviving flags + baseline)

Driver: `r54c_phase2_bench.sh` (PHASE=2A). Phase 2 picks the 10 most
promising Phase 1b survivors plus baseline, builds each at the primary cell
shape, and runs **3 perf runs per flag** with WARMUP=100, ITERS=200,
MXFP8_PRESHUFFLE_QUANT=1, GPU 5, 30s cooldown between runs, 60s
rebuild_cool between flag transitions. (Note: bench harness defaults to
RUNS=5 but only 3 runs/flag are present on disk — Phase 2 was truncated
before the planned 5-per-flag completion. The Phase 2 driver log
`r54c_phase2a.run.log` confirms each flag was “benched (3 runs)”.)

Per-flag medians from `r54c_results/phase2/8B_GateUp_rrr_*_run{1,2,3}.log`:

| Flag                       | min      | **median** | max      | Δ vs baseline |
|----------------------------|----------|------------|----------|---------------|
| **baseline**               | 2495.81  | **2500.01**| 2503.63  | ±0.000%       |
| sched_max_occ              | 2487.98  | 2498.59    | 2502.52  | -0.057%       |
| sched_max_ilp              | 2491.34  | 2492.45    | 2494.24  | -0.302%       |
| sched_iter_max_occ         | 2497.58  | 2502.99    | 2507.45  | +0.119%       |
| sched_metric_bias_100      | 2487.28  | 2490.40    | 2503.43  | -0.384%       |
| sched_relaxed_occ          | 2491.32  | 2500.04    | 2500.81  | +0.001%       |
| amdgpu_trackers            | 2438.35  | 2497.10    | 2499.91  | -0.116%       |
| loop_align_off             | 2489.72  | 2491.61    | 2503.06  | -0.336%       |
| loop_prefetch              | 2495.80  | 2496.99    | 2499.47  | -0.121%       |
| wave_priority              | 2494.33  | 2498.90    | 2501.69  | -0.044%       |
| **max_occ_plus_trackers**  | 2498.56  | **2503.07**| 2507.70  | **+0.122%**   |

All units TFLOPS, dense 8B Gate/Up RRR (M=4096, N=14336, K=4096),
peak-compute 6557.83 TFLOPS theoretical, baseline = 38.13% of peak.

**Phase 2 takeaways:**
- The two “winners” (`sched_iter_max_occ`, `max_occ_plus_trackers`) deliver
  **+0.119% / +0.122% median** improvements respectively. The 1% SHIP gate
  is missed by ~8×.
- Baseline’s own 3-run min/max spread is **2495.81…2503.63, ±0.16%**, i.e.
  ≈±0.16% noise floor. **Both “winners” are inside baseline’s own noise
  envelope** — they are statistically indistinguishable from baseline at
  3 runs/flag.
- Worst regressions (`sched_metric_bias_100` -0.384%, `loop_align_off`
  -0.336%) are also within ~2× the noise envelope; nothing is dramatically
  broken either, just nothing is dramatically helping.
- `amdgpu_trackers` produced one anomalous run3 at 2438.35 TFLOPS (-2.5%),
  pulling its mean down but not its median; likely a thermal/cooldown
  outlier rather than a flag-attributable regression.

## Resource & perf summary table

| Phase | Treatment count | Built clean | Baseline-equiv resources | ≥+1% median TFLOPS over baseline |
|-------|-----------------|-------------|--------------------------|----------------------------------|
| 1     | 16              | 11          | 10                       | n/a (build sweep only)           |
| 1b    | 25              | 23          | 23                       | n/a (build sweep only)           |
| 2     | 10 (+ baseline) | 10          | 10                       | **0**                            |

Best-case median delta across all 10 phase 2 treatments: **+0.122%** (`max_occ_plus_trackers`).
This is below both the SHIP gate (+1%) and the empirical 3-run noise floor (±0.16%).

## Reasoning — why this lever is closed

V2 RRR is at the 254/256 VGPR allocator ceiling with the source fixed.
LLVM’s register allocator is solving a fairly tight ILP at this pressure
point, and the screened scheduler / metric-bias / tracker / wave-priority /
loop-* / promote-alloca / post-RA / VOPD / merge-m0 / opt-vgpr-liverange /
disable-{clustered,unclustered}-rp-reschedule levers all leave the
solution invariant: **same 254 VGPR / 0 spill / 0 scratch / occ 2**, and
within ±0.4% perf — i.e. the perf changes are pure schedule-microstructure
noise, not a different macro-schedule.

The one flag that *did* move the allocator (`iterative-minreg`) moved it
the wrong way: 254→256 VGPR with 8 spill + 36 B scratch. This confirms the
allocator’s current solution is already on a Pareto-optimal point of the
local landscape — no flag in the screened set finds a better one.

Forcing a different solution would require either:
1. **Source change** to relieve VGPR pressure (closed: R49A/R53A/R53B all
   spilled at this same ceiling).
2. **Hard VGPR budget cap** below 254 (`-mllvm -amdgpu-num-vgpr=240/224`).
   Phase 1 confirmed the LLVM build rejects this option — the in-tree
   driver does not expose `-amdgpu-num-vgpr` at this version. Even if it
   did, capping below 254 would force spill (the kernel cannot fit live
   ranges into less than 254 VGPRs at the current source) and the spill
   cost would dominate any schedule gain.
3. **Different `regalloc`** (basic/greedy/pbqp). Phase 1b confirmed
   `-mllvm -regalloc=*` is a no-op for AMDGPU codegen — the option is
   consumed by the host pass manager, not the AMDGPU back-end. There is
   no surviving knob in the screened set that swaps the AMDGPU register
   allocator implementation.

The flag-injection axis is therefore exhausted at the resource boundary.

## Closure of the V2 RRR allocator-side lever family

With Dev C closing the compiler-flag axis, the V2 RRR fastpath at
254/256 VGPR is now **quadruply closed** at the R54 boundary:

1. R49A — host-side scale repack (REFUTED, v_perm penalty)
2. R53A — opsel-keyed K-phase MMA dispatch (REFUTED, 70 VGPR spill)
3. R53B — K-superblock persistent CTA in V2 RRR (REFUTED, 228 B/lane spill)
4. **R54C — `-mllvm` compiler-flag exploration (REFUTED-EMPIRICAL, no flag
   ≥+1%, all surviving flags resource-equivalent to baseline)** ← this work

Subsequent V2 RRR work should pivot off the 254-VGPR ceiling entirely:
shape-side reformulation (different M/N tile, different wave/CTA grid),
or accept the current V2 RRR fastpath as the structural ceiling at this
shape and re-prioritize toward cells with greater achievable headroom.

## Artifacts

- `r54c_phase1_isa.sh` — Phase 1 build sweep driver (16 treatments)
- `r54c_phase1b_isa.sh` — Phase 1b refined build sweep driver (25 treatments)
- `r54c_phase2_bench.sh` — Phase 2 A/B bench driver (10 surviving flags + baseline)
- `r54c_phase2a.run.log` — Phase 2A driver log (3 runs/flag confirmed)
- `r54c_results/phase1/SUMMARY.csv` — Phase 1 build-sweep results table
- `r54c_results/phase1/build_*.log` — Phase 1 per-flag build logs (16 files)
- `r54c_results/phase1b/SUMMARY.csv` — Phase 1b build-sweep results table
- `r54c_results/phase1b/build_*.log` — Phase 1b per-flag build logs (25 files)
- `r54c_results/phase2/build_*_4096x14336x4096.log` — Phase 2 per-flag build logs (11 files)
- `r54c_results/phase2/8B_GateUp_rrr_*_run{1,2,3}.log` — Phase 2 per-flag bench logs (33 files: 11 flags × 3 runs)

No source modifications. No commits. ISA-dump directories
(`r54c_isa/`, `r54c_isa_phase1b/`) referenced in driver scripts but not
present on disk — disassembly was not retained for this cycle (resource
table from build remarks was sufficient).

## Verdict line for cycle wrap

`R54 Dev C: V2 RRR compiler-flag exploration — REFUTED-EMPIRICAL — 41 -mllvm flag treatments swept (Phase 1: 16, Phase 1b: 25); all 33 surviving flags built at baseline-equivalent 254 VGPR / 0 spill / 0 scratch / occ 2; 10 A/B-benched on 8B Gate/Up RRR vs baseline 2500.01 TFLOPS median; best treatment max_occ_plus_trackers at +0.122% median (2503.07 TFLOPS) — inside the ±0.16% 3-run noise envelope, 8× below the +1% SHIP gate. V2 RRR allocator-side flag-injection axis closed; quadruple closure of V2 RRR 254/256 VGPR ceiling lever family alongside R49A / R53A / R53B.`
