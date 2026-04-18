R30 Dev D — V2-RCR B-tile load reorder (H7) audit
====================================================

Goal
----
R29 Dev C closed multiple V2-RCR levers (s_setprio, sched_barrier mask,
cachepolicy bits) — all NULL. He flagged H7 (B-tile load reorder) as the
"most promising R30 lever" for the 4096³ V2-RCR cell (R27 Reviewer ratio
0.9229, the largest open RCR gap). R30 Dev D goal: enumerate
correctness-safe B-tile load orderings in the V2-RCR `do_k_iter_body`
lambda and bench each.

Background and structural constraints
-------------------------------------
R29 Dev C also raised a dispatch-geometry ceiling: at BLK=256, the 4096³
GRID is 16×16 = 256 blocks across 304 CUs, leaving 16% of CUs idle.
This 16% wave-fill gap cannot be closed by any per-kernel optimization
without changing dispatch geometry (which would require BLK=128 path
resurrection, R31 territory).

H7 was scoped to: can a B-tile load reorder *within* the existing
do_k_iter_body close any of the *non-occupancy* portion of the gap?

V2-RCR do_k_iter_body topology (kernel_mxfp8_layouts.cpp:2780–3041)
--------------------------------------------------------------------
One body call performs 4 MMA quadrants (cA, cB, cC, cD) with 1 A-tile
LDS-pre-load, 2 LDS→reg loads (b0 already done at top of iter; b1 at
pre-cB), 1 A-tile pre-load (pre-cC), and 2 B-tile VMEM pre-loads:
- pre-cB: `G::load(Bs[tic][0], g.b, ..., k+2)` — next-iter B[0]
- pre-cD: `G::load(Bs[tic][1], g.b, ..., k+2)` followed by
          `TK_WAIT_VMCNT(6)` — next-iter B[1]

LDS lifetime constraint
-----------------------
The two B-tile slots are read by `rcr_exact_load_st_to_rt`:
- `b0 ← Bs[tic][0]` happens at the very top of the body (line 2807).
- `b1 ← Bs[tic][1]` happens at pre-cB (line 2893).

For a next-iter `G::load(Bs[tic][N])` to be safely hoisted earlier in
the body, ALL waves must finish reading the corresponding LDS slot
first. Concretely:
- `G::load(Bs[tic][0])` is safe at any slot at-or-after line 2807.
- `G::load(Bs[tic][1])` is safe ONLY at slots at-or-after line 2893
  (post-cB pre-cC, post-cC pre-cD, or post-cD).

This means the B[1] load **cannot legally be hoisted above pre-cB**
without inter-wave races that corrupt b1 reg contents.

H7 variant menu (build-time macro `MXFP8_RCR_V2_BLOAD_REORDER`)
---------------------------------------------------------------
Added to `kernel_mxfp8_layouts.cpp`:

```c
// 0 = baseline (B[0] pre-cB, B[1] pre-cD)
// 1 = both B loads hoisted to top-of-body (with first A load)
// 2 = both B loads deferred to pre-cD slot
// 3 = SWAP pairing (B[1] pre-cB, B[0] pre-cD)
// 4 = both B loads at pre-cB slot (batch early)
#ifndef MXFP8_RCR_V2_BLOAD_REORDER
#define MXFP8_RCR_V2_BLOAD_REORDER 0
#endif
```

Wired with `#if MXFP8_RCR_V2_BLOAD_REORDER == N` blocks at the three
candidate slots (top-of-body adjacent to A-load; pre-cB; pre-cD).

**Default-off byte-identical sanity**: head `kernel_mxfp8_layouts.cpp`
build artifact md5 = `d49dc0b77df8b24d751519b854a08490`; this branch's
default build (no `-D`) md5 = `d49dc0b77df8b24d751519b854a08490`.
Confirmed: the macro is functionally inert at default.

Hardware note
-------------
GPU3 (HIP_VISIBLE_DEVICES=3 → physical /dev/dri card 3, rocm-smi
identifies as GPU[3]). DPM idles at 95–158 MHz; ramps to 2300–2400 MHz
within the 8s preheat. All cells used 5x same-process bench,
MXFP8_WARMUP=50, MXFP8_ITERS=100. Per-build .so md5 logged to
guarantee fresh artifact (`rm -f tk_mxfp8_layouts*.so` before each
compile — same fix as R29C).

Results — 5 cells (4096³ V2-RCR baseline + 4 H7 variants) + 1 8192³
-----------------------------------------------------------------

| Cell      | Variant         | snr_db  | det  | TFLOPS mean ± std  | median  | Δ% vs base | Welch t | Verdict |
|-----------|-----------------|---------|------|--------------------|---------|------------|---------|---------|
| A0_base   | baseline        | 49.61   | True | 2486.42 ± 89.01    | 2521.34 | —          | —       | ref     |
| A1_v1     | hoist-both-top  |  7.82   | False| 2322.19 ± 122.07   | 2357.24 | −6.60%     | −2.42   | CORRUPT |
| A2_v2     | defer-both-cD   | 49.61   | True | 2473.18 ± 80.55    | 2482.58 | −0.53%     | −0.247  | NULL    |
| A3_v3     | swap (B[1]@cB)  | 19.33   | False| 2438.32 ± 86.91    | 2454.17 | −1.93%     | −0.86   | CORRUPT |
| A4_v4     | both@cB batch   | 24.40   | False| 2458.71 ± 89.37    | 2487.16 | −1.11%     | −0.49   | CORRUPT |
| B0_base   | 8192³ baseline  | 49.59   | True | 3003.27 ± 29.49    | 3016.02 | —          | —       | ref     |

Correctness analysis — why 3 of 4 reorders broke
------------------------------------------------
A1 (hoist-both-top), A3 (swap), and A4 (both@cB) all violate the LDS
lifetime constraint above: they issue `G::load(Bs[tic][1])` at a slot
ABOVE the `b1 ← Bs[tic][1]` LDS read (line 2893), so on at least some
wave schedules the VMEM write to LDS lands while another wave is still
draining the b1 read. This produces non-deterministic register
contents (det=False) and SNR collapse (8–24 dB vs the 49.6 dB clean
baseline). pass_rate=100% is misleading — the Hopper-style "≤3.0 abs
or ≤10% rel" gate tolerates large per-element drift but the population
SNR exposes it.

A2 (defer-both-cD) is the only correctness-safe reorder: both
`Bs[tic][0]` and `Bs[tic][1]` LDS reads have completed by pre-cD
(line 2998). It produces snr=49.61 dB and det=True — bit-identical
output across reps.

Performance — A2 (only correctness-safe reorder)
------------------------------------------------
Mean delta vs baseline: **−13.24 TFLOPS (−0.53%)**, Welch t = −0.247.
This is well within the 4096³ bench-to-bench noise band (RUN-0 cold
dip dominates std; tails of base and v2 are statistically equal).

The intuition for why A2 doesn't help: deferring both B-loads to the
last slot (pre-cD) **shrinks the VMEM-issue window for those B-tiles
to one MMA quadrant** (the cD MFMA). The baseline split (B[0]@cB,
B[1]@cD) gives B[0] **three MMA quadrants** of latency hiding (cB +
cC + cD) and B[1] one quadrant. Aggregating both into one slot
trades latency-hiding-window for a smaller VGPR-liveness footprint —
on a 246-VGPR/0-spill kernel the latter is not the bottleneck, so the
trade is net-zero.

The reverse (hoist-both-top, A1) IS what would maximally help latency
hiding (4 MMA quadrants of hide window for B[0]+B[1]), but it
violates the LDS lifetime → corrupt.

H7 verdict: **NO SHIP** — closed for V2-RCR
-------------------------------------------
The V2-RCR do_k_iter_body has effectively only ONE correctness-safe
B-tile reorder (defer-both to pre-cD), and that reorder is NULL within
noise. The original baseline (B[0]@cB, B[1]@cD) is already at the
maximum-latency-hiding ordering allowed by the LDS lifetime
constraint.

H7 is closed for V2-RCR per this audit.

Recommendation for R31 (audit-only carryover)
---------------------------------------------
The remaining 4096³ V2-RCR gap (R27 Reviewer ratio 0.9229) is most
plausibly bounded by the structural GRID under-occupancy ceiling
(304 CUs vs 256 BLK=256 blocks → 16% CUs idle). Closing this gap
requires dispatch-geometry change, not per-kernel optimization:

1. **BLK=128 path resurrection** (R23+ legacy): would dispatch 1024
   blocks on 304 CUs — full wave-fill, but 4× more LDS pressure per
   block (32 KB vs 131 KB at BLK=256 → may break LDS budget) and
   different scale layout (all V2 preshuffle code assumes BLK=256).
   Bounded but invasive.

2. **Persistent-CU dispatch** (one block per CU, internal K-loop over
   multiple block coordinates): keeps BLK=256 LDS budget; the 16
   "extra" CUs each take a second block — reduces idle from 16% to
   0% but requires kernel rewrite. Out of R30 scope.

3. **Accept the 0.92 ratio as structural floor** for the 4096³
   V2-RCR cell at BLK=256 dispatch — and shift R31 effort to other
   open levers (V2-CRR small shapes, V2-RRR mid-K headroom).

Files
-----
- `kernel_mxfp8_layouts.cpp`: +1 macro (`MXFP8_RCR_V2_BLOAD_REORDER`)
  with `#if`-guarded variants at 3 candidate slots in
  `do_k_iter_body`. Default-off, byte-identical to head.
- `r30d_bench.py`: HIP_VISIBLE_DEVICES=3-aware bench harness
  (R30D_PHYS_GPU env for rocm-smi sclk read; otherwise mirrors
  r28a_bench5x.py).
- `r30d_orchestrate.sh`: build/bench loop for 4 variants + 8192³ ref.
- `r30d_cell{A0_base,A1_v1,A2_v2,A3_v3,A4_v4}_rcr_4096x4096x4096.txt`:
  per-cell CORRECTNESS+DETERMINISM+SUMMARY logs.
- `r30d_cellB0_base_rcr_8192x8192x8192.txt`: 8192³ baseline.
- `r30d_build_*.log`: per-cell build logs (V2-RCR resource usage:
  246 VGPRs / 0 spill / occupancy=2 — unchanged across variants).

Final outcome
-------------
**NO SHIP** — H7 closed for V2-RCR.
**AUDIT-ONLY carryover for R31**: dispatch-geometry change (BLK=128
or persistent-CU) is the remaining lever for the 4096³ V2-RCR gap.

The `MXFP8_RCR_V2_BLOAD_REORDER` macro is left in tree as
default-off scaffolding documenting that the legal reorder space is
exhausted at the current dispatch geometry.
