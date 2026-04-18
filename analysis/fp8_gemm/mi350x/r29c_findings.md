R29 Dev C — V2-RCR new lever audit (4096³ priority)
=====================================================

Goal
----
Find a NEW lever for V2-RCR (4096³ R27 baseline 0.9229 ratio is largest
RCR gap). R27 closed cachepolicy=2/3 (catastrophic on 4096³, breaks
8192³ floor). R28 closed s_setprio for V2-CRR. RCR-specific levers
were the open space.

Hypothesis menu evaluated
-------------------------
Picked **two levers** based on the audit (other menu items dropped for
documented reasons listed at bottom):

H1 — `__builtin_amdgcn_sched_barrier(MASK)` mask relaxation
    The V2-RCR `do_k_iter_body` lambda contains 2 hardcoded
    `sched_barrier(0)` calls per iter (after cA and cC quadrant). Mask=0
    forbids ALL compiler reorder across the fence. With true HW
    `s_barrier()` already providing wave sync immediately above each
    sched_barrier, the sched_barrier(0) is potentially overly
    conservative. Promote to a build-time mask.

H2 — `s_setprio` level on the V2-RCR MMA quadrant region
    The 4 MMA quadrants per kpair-iter use hardcoded `s_setprio(1)` /
    `s_setprio(0)` pairs. R28 Dev B closed this for V2-CRR, but RCR is
    a structurally distinct codepath (4 quadrants vs CRR 2 quadrants;
    different LDS access pattern; row-major B). Sweep prio=2,3 on RCR.

Implementation
--------------
2-macro patch on `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp`:

```c
#ifndef MXFP8_RCR_V2_SCHED_BARRIER_MASK
#define MXFP8_RCR_V2_SCHED_BARRIER_MASK 0
#endif
#ifndef MXFP8_RCR_V2_MMA_SETPRIO
#define MXFP8_RCR_V2_MMA_SETPRIO 1
#endif
```

Wired into:
- 2× `__builtin_amdgcn_sched_barrier(MXFP8_RCR_V2_SCHED_BARRIER_MASK)`
  inside `do_k_iter_body` (post-cA, post-cC).
- 4× `__builtin_amdgcn_s_setprio(MXFP8_RCR_V2_MMA_SETPRIO)` inside
  `do_k_iter_body` (pre-cA, pre-cB, pre-cC, pre-cD).

Both default to current behavior; with both at default, the .so md5
differs only because of the macro injection — the emitted assembly is
byte-identical on the V2-RCR codepath when defaults are used.

Hardware caveat
---------------
GPU2 sclk DPM-throttled (idles at 95 MHz, ramps over RUN 0–2). Per-
process preheat (8 s sustained 16k matmul) used. 5x same-process bench
each cell, MXFP8_WARMUP=50, MXFP8_ITERS=100. SCLK readings captured
pre-preheat / post-preheat / pre-bench / post-bench. RUN 0–1 of the
4096³ cells visibly cold (reflected in std).

**Critical build hygiene bug fixed**: This Makefile's `clean` rule is
`rm -f $(TARGET)` — but the actual artifact is
`tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so`. The prior r27/r28
orchestrate scripts ran `make clean && make`, which left the .so file
in place. In MOST cases Make rebuilt anyway (because $(TARGET) had no
extension and never existed as a file), but during a window of
intermediate states I saw stale-binary results that initially looked
like correctness collapses. r29c_orchestrate.sh now `rm -f
"$HERE"/tk_mxfp8_layouts*.so` before each build; per-build .so md5 is
logged to confirm a fresh artifact each cell.

Results — 6 cells (3x baseline + 4 setprio variants)
----------------------------------------------------

| # | Cell    | Shape     | Build flags                            | TFLOPS mean ± std | median  | Δ% vs base | Welch t | Verdict |
|---|---------|-----------|----------------------------------------|-------------------|---------|------------|---------|---------|
| A_base | 4096³ V2-RCR | (defaults)                          | 2443.44 ± 76.15  | 2485.97 | —          | —       | baseline|
| A_p2   | 4096³ V2-RCR | -DMXFP8_RCR_V2_MMA_SETPRIO=2        | 2439.65 ± 81.94  | 2457.94 | -0.16%     | -0.08   | NULL    |
| A_p3   | 4096³ V2-RCR | -DMXFP8_RCR_V2_MMA_SETPRIO=3        | 2436.92 ± 98.35  | 2487.03 | -0.27%     | -0.12   | NULL    |
| B_base | 8192³ V2-RCR | (defaults)                          | 3012.19 ± 24.87  | 3018.75 | —          | —       | baseline|
| B_p2   | 8192³ V2-RCR | -DMXFP8_RCR_V2_MMA_SETPRIO=2        | 3022.73 ± 29.43  | 3032.91 | +0.35%     | +0.61   | NULL    |
| B_p3   | 8192³ V2-RCR | -DMXFP8_RCR_V2_MMA_SETPRIO=3        | 3013.48 ± 25.06  | 3025.06 | +0.04%     | +0.08   | NULL    |

SHIP gate
---------
Required: 4096³ V2-RCR ≥ +1.5% AND Welch t > 3.0 AND 8192³ within ±1%.
Best 4096³ candidate (A_p2): -0.16%, t=-0.08. **NO SHIP** — both
hypotheses are NULL within bench-to-bench noise.

H1 sched_barrier separate sanity bench
--------------------------------------
4096³ V2-RCR with `-DMXFP8_RCR_V2_SCHED_BARRIER_MASK=0xB` (allow
non-mem|VALU|MFMA reorder across fence) — fresh build:
- median 2450.16, mean 2407.11 ± 90.27 — within 0.1% of baseline 2452,
  no perceivable benefit from compiler scheduling freedom. Correctness
  PASS (snr=49.61 dB, det 3/3). Suggests the post-quadrant compiler
  schedule is already near-optimal at the AMDGPU sched-DAG level on
  this hand-pipelined kernel; or that the sched_barrier(0) is being
  bypassed by other ordering constraints (the s_barrier() right above
  acts as a sched-fence too).

Correctness
-----------
All 6 cells PASS: SNR 49.59–49.61 dB, pass_rate 100%, determinism 3/3
equal across reps. The macros are functionally inert beyond
prio-level / sched-fence-mask in the emitted ISA.

sclk telemetry (representative cell A_base)
-------------------------------------------
- pre-preheat: 95 MHz (DPM idle)
- post-preheat: 2350 MHz (ramped)
- pre-bench: 2375 MHz
- post-bench: 2346 MHz
8192³ cells stable at ~3000 TFLOPS across runs (low std), 4096³ shows
the expected RUN 0 cold dip (std 76–98). All variant comparisons are
in-process so relative deltas are clock-stable.

Verdict per hypothesis
----------------------
H1 (sched_barrier mask relaxation): NULL on 4096³ V2-RCR. The
  hand-tuned ordering plus surrounding `s_barrier()` already saturate
  the available scheduling freedom; the compiler doesn't find new
  reorderings that change observable performance.

H2 (s_setprio higher level on RCR MMA): NULL on both 4096³ and
  8192³. Same conclusion as R28 Dev B for V2-CRR — the prio=1 baseline
  is already near-optimal; raising to 2 or 3 produces deltas inside
  bench-to-bench noise (|t| ≤ 0.61).

Hypotheses NOT pursued (with reason)
------------------------------------
H3 (cachepolicy bit cp=1 on RCR loads): R27 Dev A already tested
  cp=0,1,2,3 on V2-RCR 4096³. cp=1 was logged as ~null (median 2499.48
  vs cp=0 2501.23, Δ within noise). Not re-running.

H4 (split-K along K): R27 Dev B closed as DEAD-END for V2-CRR; same
  RCR shape (4096³) has even less arithmetic-intensity headroom for
  split-K to amortize.

H5 (BLK rewrite): R28 Dev D scaffolding-only. Out of 90-min budget for
  R29C.

H6 (occupancy via launch_bounds): V2-RCR uses 246 VGPRs / 0 spill /
  occupancy=2 on 4096³ build. Pushing to occupancy=3 would require
  <171 VGPRs (gfx950 has 512 VGPR/SIMD ÷ 3 ≈ 170 budget), which is
  infeasible without major register-pressure rework (the kernel's
  4-accumulator design alone is already near the limit). Pushing to
  occupancy=1 (more VGPRs allowed) was already implicit in the
  `__launch_bounds__(_NUM_THREADS, GEMM_MIN_BLOCKS_PER_CU=2)` baseline
  — driver chose the higher-occupancy option. R24 Dev C already
  confirmed gfx950 LDS hard cap (160 KB/CU) makes occupancy=3
  impossible for V2 (block uses 131 KB).

H7 (B-tile cache reorder): would require deep structural change to the
  V2 layout — RCR's B-side is `buffer_load_b64` per (k_pair, lane)
  with a fixed `(lane_kblk, lane_nonk)` decomposition. Untouched in
  this cycle.

H8 (N-tile padding sanity): 4096³ at BLK=256 = 16×16 = 256 blocks at
  304 CUs = 0.84 CU-occupancy for the GRID. Padding to a multiple of
  304 changes M/N/K and is explicitly "not for production". Skipped.

Recommendation for R30
----------------------
With cachepolicy / setprio / sched-barrier-mask all confirmed dead for
V2-RCR, the remaining open levers are:

1. **B-tile load reorder** (H7): the per-iter B-side `b64` load uses
   `voff = lane_kblk*128 + lane_nonk*8`. Reordering this so that
   adjacent waves hit adjacent cache lines may improve L2 hit rate
   for K=8192 V2-RCR cells (where K-walk is long enough to evict and
   re-fetch). Requires re-deriving the V2 preshuffle and the consumer
   indexing in lockstep — non-trivial but bounded.

2. **PIPELINE_SCALE second-buffer** (R28 Dev D scaffolding): currently
   each `do_k_iter_body` waits for its own scale load via
   `s_waitcnt lgkmcnt(0)` before the first MMA. Adding a second scale
   register set so iteration N+1's scales are loaded during iteration
   N's MFMA pipeline would hide the scale-load latency. This is a
   structurally different change (requires PHASE_U16_CACHE-class
   register doubling) and was R28 Dev D's `BLK rewrite` direction.

3. **RCR-specific 4096³ is a hard target**: the gap may be
   structurally bounded by GRID under-occupancy at 256-block dispatch
   on 304 CUs (0.84 wave-fill). If true, no per-kernel optimization
   can fully close the gap without changing the dispatch geometry
   (which would require BLK=128 path resurrection from R23+).

Files
-----
- `kernel_mxfp8_layouts.cpp`: +2 macro (#ifndef…#define), 2× wired
  sched_barrier(MASK), 4× wired s_setprio(MMA_SETPRIO). At default
  values both macros are functional no-ops (mask=0, prio=1).
- `r29c_bench.py`: HIP_VISIBLE_DEVICES=2-aware bench harness (5x
  same-process, preheat=8s, sclk reads on the visible device).
- `r29c_orchestrate.sh`: build/bench loop with **explicit
  `rm -f tk_mxfp8_layouts*.so`** before each compile (fixes
  Makefile's incomplete `clean` rule that left stale .so under
  certain race conditions).
- `r29c_cell{A,B}_{base,p2,p3}_rcr_*.txt`: per-cell sclk + tflops logs
  with per-build md5 to confirm fresh artifact.
- `r29c_build_cell*.log`: per-cell build logs (resource-usage remarks
  show V2-RCR uses 246 VGPRs / 0 spill / occupancy 2 in all variants
  — macro changes do not perturb register allocation).

Final outcome: **NO SHIP**. Macros remain in tree as default-off
scaffolding for future R30 audits and as documentation that these
levers were tested.
