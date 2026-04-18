# R45 Dev B — 8B Gate/Up V2-RRR BLK_N=256 exploration

## Verdict: **REFUTED-AT-DESIGN**

Cycle: R45 Dev B (R45+ priority #5 from R44 wrap, scoped from R44D survey).
Branch: `worktree-agent-a75633e9` (off `37bb9162` R44 cycle wrap).
Date: 2026-04-18.
GPU usage: **0 GPU-min** (refuted by static inspection — no prototype built,
no bench burned).

## TL;DR

The R44D survey (`r44d_margin_tightening_survey.md` §"Optional R45+
micro-optimization scope") proposed a "BLK_N=256 variant (RBN=64 ×
WARPS_N=4 OR RBN=32 × WARPS_N=8)" for V2-RRR Gate/Up, on the premise
that the **current** kernel covers BLK_N=128 per WG. **This premise is
false by a factor of 2x.**

Static inspection of `kernel_mxfp8_layouts.cpp:335-341` and
`rrr_mxfp8_exact_8wave_fastpath.inc:14-19, 41-44, 551-554` proves the
default V2-RRR kernel **already covers BLK_N=256 per WG today** by
emitting two N-direction sub-tiles (`cB`/`cD`) per warp.

What the brief actually asks for ("doubling N-tile width per workgroup
→ halving N-grid count") is a **2x expansion BEYOND what already
ships**, i.e. BLK_N=512. A real BLK_N=512 variant would push the
accumulator from 4×(64×32) = 8192 fp32-lanes to 4×(64×64) = 16384
fp32-lanes per warp = +128 VGPR/wave for accumulator alone, on top of
the **256-VGPR ceiling that the current kernel already saturates with
1-50 VGPR spills** (per `r31a_build_cell{1,2,3,4}.log:42-50`,
`r31c_*_build.log`).

The fallback "WARPS_N=2 + RBN=64" yields the same accumulator footprint
(per-warp share of N doubles) and additionally halves per-WG wave
count, losing 50% of intra-WG parallelism. Same ceiling, same
refutation.

The R44D survey's "0.5-1.5pp expected lift" estimate is based on the
underlying off-by-2x error (it expected to halve a WG-grid that is
already at the proposed final count of 56 N-tiles) and does not
survive correction.

## Detailed analysis

### Current V2-RRR per-WG output coverage

`kernel_mxfp8_layouts.cpp:335-341`:
```cpp
constexpr int BLK = GEMM_BLOCK_SIZE;       // 256
constexpr int HB  = BLK / 2;               // 128
constexpr int WARPS_M = GEMM_WARPS_M;      // 2
constexpr int WARPS_N = GEMM_WARPS_N;      // 4
constexpr int RBM = BLK / WARPS_M / 2;     // 64
constexpr int RBN = BLK / WARPS_N / 2;     // 32
```

`rrr_mxfp8_exact_8wave_fastpath.inc:41-44`:
```cpp
A_row_reg a;
B_col_reg b0, b1;                          // TWO B-tile regs (top + bottom HB)
rt_fl<RBM, RBN, col_l, rt_16x16_s> cA, cB, cC, cD;   // FOUR sub-acc tiles
zero(cA); zero(cB); zero(cC); zero(cD);
```

`rrr_mxfp8_exact_8wave_fastpath.inc:551-554` (epilogue store coords —
load-bearing for per-WG output extent):
```cpp
store(g.c, cA, {0, 0, br*WARPS_M*2 + wm,            bc*WARPS_N*2 + wn});
store(g.c, cB, {0, 0, br*WARPS_M*2 + wm,            bc*WARPS_N*2 + WARPS_N + wn});
store(g.c, cC, {0, 0, br*WARPS_M*2 + WARPS_M + wm,  bc*WARPS_N*2 + wn});
store(g.c, cD, {0, 0, br*WARPS_M*2 + WARPS_M + wm,  bc*WARPS_N*2 + WARPS_N + wn});
```

The N-direction extent per WG is:
```
WARPS_N * 2 (subtile multiplier for cB/cD) * RBN
  = 4 * 2 * 32  = 256 cols  =  BLK_N  =  BLK
```

Identically, the M-direction per-WG extent is `WARPS_M * 2 * RBM = 2 * 2
* 64 = 256` rows. So **BLK_M = BLK_N = BLK = 256 already**.

R39 Dev A's findings doc (`r39a_findings.md:22-32`) confirms this same
geometry independently:
> `BLK = 256`, `HB = 128`, `WARPS_M = 2`, `WARPS_N = 4`, `RBM = 64`,
> `RBN = 32`, accumulator `cA, cB, cC, cD` at 4 × 64×32.

### Grid count for 8B Gate/Up (M=4096, N=14336, K=4096)

```
N-tiles  = N / BLK_N = 14336 / 256 = 56
M-tiles  = M / BLK_M =  4096 / 256 = 16
Total WG = 56 × 16 = 896
CU count = 304
Saturation = 896 / 304 ≈ 2.95 waves of WGs per CU
```

The brief's claim "14336/128=112" is the **doubled** count that would
result from HALVING BLK_N (i.e., HB-N shrink). That is the R38 Dev A /
R39 Dev A failed paradigm, refuted at -33% to -45% across V2-CRR and
V2-RCR rect kernels.

The brief's proposed "BLK_N=256" with N-grid count of 56 IS THE
CURRENT STATE.

### What a real BLK_N expansion would look like

A genuine 2x N-tile expansion is BLK_N=512. Options:

**Option A: WARPS_N=4, RBN=64** (preserve wave count, double per-warp
N-coverage):
- Accumulator: 4 × (RBM × RBN) = 4 × (64 × 64) = 16384 fp32-lanes/warp
  = 256 VGPR for accumulator alone.
- Current accumulator at RBN=32: 4 × 64 × 32 = 8192 lanes = 128 VGPR.
- Delta: +128 VGPR/wave for accumulator.
- B-register pressure: `B_col_reg` doubles in N-cols, so b0+b1 = 4 × 64
  × 32 = 8192 lanes vs current 4 × 64 × 16 = 4096 lanes. Delta: +64 VGPR.
- A-register pressure: unchanged (RBM, BK same).
- Scale-pack VGPR: doubled for B side.

**Current kernel VGPR ceiling (measured)**: 256 VGPR/wave with 1-50
spills depending on shape (`r31a_build_cell{1..4}.log:48`,
`r31c_baseline_8k_build.log:48`). **256 is the architectural ceiling
for occ=2 (2 wave/CU) on gfx950 launch_bounds=GEMM_MIN_BLOCKS_PER_CU=2
in the kernel preamble**.

Adding +128 VGPR for accumulator alone would force occ=1 (1 wave/CU),
halving wave-level concurrency — independently of any spill behavior.
Plus the +64 VGPR B-register growth, the spill count would explode.

**Option B: WARPS_N=2, RBN=64** (halve wave count, double per-warp
N-coverage):
- Same per-warp accumulator footprint as Option A: 4 × 64 × 64 lanes =
  +128 VGPR/wave for accumulator.
- Per-WG wave count drops from 8 to 4 → halves intra-WG parallelism.
- LDS A-tile geometry: WARPS_M still 2, A unchanged. B LDS tile
  geometry doubles in N because each warp now consumes RBN=64 cols.
- Same VGPR ceiling violation as Option A; additionally loses wave-level
  parallelism that V2-RRR depends on for its dual-buffer DB pipeline.

**Option C: Add a new sub-tile pair (cE, cF, cG, cH) = WARPS_N=4,
RBN=32, but emit 4 N-subtiles per warp instead of 2**:
- Accumulator quadruples: 8 × 64 × 32 = 16384 lanes = 256 VGPR/wave for
  acc alone. Same as Option A. Same refutation.

All three "honest" BLK_N=512 designs hit the same VGPR ceiling.

### Why the +5% margin really is a floor

R44D classified +5% as a **floor-limit** (silicon-bin cap) per
`kernel_mxfp8_layouts.cpp:5978` comment "R34B +5.025% min" and R40
BOUNDARY-LOCK rule. R39 Dev A independently said V2-RRR is **MORE**
B-bandwidth-bound than V2-CRR on this shape (closer to its ceiling)
because it ships at higher absolute TFLOPS:

> "V2-RRR's higher absolute TFLOPS (~2538 TF on c5 vs CRR's ~2415)
> means V2-RRR is *more* bandwidth-bound on wide-N than V2-CRR, so
> HB-N shrink — which trades VGPR for WG-grid overhead — would
> produce an even worse delta."
> — `r39a_findings.md:67-71`

The reverse direction (BLK_N expansion) does NOT relieve the
B-bandwidth ceiling either; it just amortizes per-tile epilogue/setup
overhead. Per R33B's rect-V2 RCR refutation, that overhead is small
enough at the current 2.95× CU saturation that further amortization
does not move the needle:

> "Per-tile overhead doubles: rect halves per-tile work (1 BLK_N=128
> ctile does 4 quadrant MMAs vs square's 8 for BLK_N=256). The
> cA/cB/cC/cD epilogue, prologue load fences (TK_WAIT_VMCNT), and
> s_barrier count are roughly constant per tile. Doubling tile count
> doubles overhead, undoing the wave-fill gain."
> — `r33b_findings.md:158-162`

By symmetry, halving tile count (BLK_N expansion) only halves the
epilogue/prologue overhead for an already-saturated grid, but loses
much more if we have to drop occupancy to fit registers.

### Why "no spills allowed" guard automatically blocks all designs

The brief states: "verify register pressure remains at-or-below
current (no spills allowed)". The current default already has 1-50
spills (varies with -DM_DIM/-DN_DIM/-DK_DIM at build time). Any
BLK_N expansion design above strictly increases register pressure
(accumulator + B-register footprint), so the no-spills constraint
forecloses every option before benching.

If the constraint is interpreted as "no NEW spills above the
current default's 1-50", then per the +128 VGPR delta on the
accumulator alone, the prototype would necessarily spill more — and
the time-box mandate requires REFUTED-AT-DESIGN here.

## Pre-art chain (paradigm-equivalent precedent)

The "BLK_N grid manipulation for wide-N" hypothesis has been tested in
both directions and refuted:

| Cycle | Direction | Variant | Result | Source |
|-------|-----------|---------|--------|--------|
| R32D / R33B | HALVE BLK_N (rect-V2 RCR BLK_N=128) | 4096³ (square not wide-N, but same paradigm) | -33% to -36% | r33b_findings.md:140-143 |
| R38 Dev A | HALVE BLK_N (HB-N shrink V2-CRR PIPE=0/1) | 8B Gate/Up | -43% | r38 wrap, TODO:570 |
| R38 Dev A | HALVE BLK_N (HB-N shrink V2-CRR) | 70B Gate/Up | -45% | r38 wrap |
| R39 Dev A | HALVE BLK_N (HB-N shrink V2-RRR) | 8B/70B Gate/Up | REFUTED-BY-INSPECTION | r39a_findings.md:1-134 |
| R35 Dev C | DOUBLE WARPS_N (WARPS_N=8 V2-CRR) | wide-N | REFUTED | TODO:363, r44d:271 |
| R45 Dev B (THIS) | DOUBLE BLK_N (BLK_N=512 RBN=64 OR WARPS_N=2 RBN=64) | 8B Gate/Up V2-RRR | REFUTED-AT-DESIGN | this doc |

Both directions of BLK_N grid manipulation, AND both directions of
WARPS_N count manipulation, have now been exhausted on the V2-RRR /
V2-CRR / V2-RCR family at the wide-N geometry. The +5% Gate/Up V2-RRR
floor is structural for this kernel-template class.

## What might still work (R46+ scope, NOT R45)

These are mentioned ONLY as forward-pointers; none are recommended for
R45 follow-up by this dev — they're 3-5 day rewrites with weak EV.

1. **B-side LDS prefetch / cache-policy bits for V2-RRR wide-N**.
   R39A:113-116 cited as the only direction that targets the
   B-bandwidth ceiling directly (the actual binding constraint).
   Touches `MXFP8_RRR_V2_SCALE_CACHEPOLICY` (currently 0), B-tile
   prefetch ordering in `do_k_iter`. Speculative.

2. **K-tile expansion (BK=256) for V2-RRR Gate/Up K=4096**. Halves
   K-loop iterations. Doubles A LDS tile size (RBM × BK = 64 × 256 =
   16384 fp8 = 16 KB/wave; ×8 waves × 2 buffers ≈ 256 KB) — exceeds
   160 KB/CU LDS limit, so requires occ=1 anyway. Not promising.

3. **Tiny WARPS_M=4 V2-RRR wide-N variant** (mirror of R35C
   WARPS_M=4 V2-CRR which was negative on its target shape, suggesting
   asymmetric tilings are not generically winning). Speculative
   compound bet.

4. **Accept floor**: the R44D survey's primary recommendation
   ("**FLOOR-LIMIT NO-CHANGE**") was correct. The "Optional R45+
   scope" line in the survey was based on the off-by-2x error
   identified here; it should be retracted.

## Recommendation

1. **No code changes.** No prototype built. No bench burned.
2. **Retract the R44D §"Optional R45+ scope" suggestion** for V2-RRR
   Gate/Up wide-N tile expansion. The survey's BLK_N=128 claim was an
   off-by-2x error against the kernel's actual per-WG output extent
   (BLK_N=256 today); the survey-inferred "1-2 days kernel + 1 day
   bench, expected 0.5-1.5pp" estimate does not survive correction.
3. **Promote a lightweight rule to the methodology**: when a survey
   proposes a tile-config exploration, the proposing dev should derive
   per-WG output extent from the kernel's epilogue store coords (the
   load-bearing source of truth for BLK_M / BLK_N), not from the
   `BLK / WARPS_N / 2` declared constants alone — the latter ignores
   the cB/cD sub-tile multiplier that doubles per-WG N coverage on
   8-wave fastpaths.
4. **R44D Part 2 verdict on cell #2 stands**: 8B Gate/Up V2-RRR at
   +5.0% is FLOOR-LIMIT. R44 reviewer's +0.37pp margin (R44 wrap line
   2) is a sample within the silicon-bin envelope, not a tightening
   trend.

## GPU-min: 0

Pure offline analysis. No prototype scaffold built; no bench cycles
burned. The R45+ time-box rule (refute at design when the load-bearing
arithmetic is wrong) is honored.

## Files

- `analysis/fp8_gemm/mi350x/r45b_findings.md` — this file.
- Source-of-truth references:
  - `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp:335-341` (BLK,
    WARPS_*, RBM, RBN constexpr definitions)
  - `analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc:14-19`
    (BLK=256, WARPS_M=2, WARPS_N=4 static_asserts)
  - `analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc:41-44`
    (cA/cB/cC/cD declaration — 4 sub-tiles per warp)
  - `analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc:551-554`
    (epilogue store coords showing cB/cD use `WARPS_N + wn` offset →
    BLK_N coverage = `WARPS_N * 2 * RBN = 256`)
  - `analysis/fp8_gemm/mi350x/r31a_build_cell1.log:42-50` and
    `r31c_baseline_8k_build.log:42-50` (V2-RRR VGPR=256, spills 1-50)
- Pre-art:
  - `analysis/fp8_gemm/mi350x/r39a_findings.md` (V2-RRR HB-N shrink
    REFUTED-BY-INSPECTION; identical tile-config logic)
  - `analysis/fp8_gemm/mi350x/r33b_findings.md:140-181` (rect-V2 RCR
    -33% to -36% per-tile overhead refutation)
  - `analysis/fp8_gemm/mi350x/r44d_margin_tightening_survey.md:240-255`
    (the R44D suggestion this doc refutes; off-by-2x source)
</content>
</invoke>