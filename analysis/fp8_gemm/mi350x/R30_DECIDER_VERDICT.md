# R30 DECIDER VERDICT — MXFP4 GEMM, post-R29 (41/42 WIN, only L6 LOSE)

**Date:** 2026-04-18  Decider: R30 scout (Opus 4.7), no-build pure analysis.

Bench source: `bench_all42_results_R25_FINAL_v2.json` (round R25_FINAL,
warmup=200, iters=500, trim=0.10). L6 = 4096×32768×128256 at 92.6% of comp
(5354 / 5781 TFLOPS) — only residual.

---

## Q1 — K HARD-GATE audit (`kernel_mxfp4_gluon_cpp.cpp:80-110`)

**Finding: the "K_DIM ≤ 32768 HARD-GATE" is a misnomer.** The gate is
`R25C_K_LIMIT` (default 32768, line 96), and it gates **only** the R25C
*tail-prefetch-off* optimisation, not the kernel's K capability. Definition
at line 106-107:

```cpp
#define R25C_ACTIVE ((R25C_TAIL_PF_OFF_ITERS > 0) && (K_DIM <= R25C_K_LIMIT) \
                     && (R25C_K_EXACT == 0 || K_DIM == R25C_K_EXACT))
```

**What it actually guards:** the runtime branch `_r25c_tail_no_pf = (bt >=
k_byte_iters - 1 - R25C_TAIL_PF_OFF_ITERS)` at line 2810 inside the K-loop.
For `K_DIM ≤ 32768` (`k_byte_iters ≤ 128`) the loop is `#pragma unroll`
(line 2443-2448) and the branch folds to constants per iter. For
`K_DIM=128256` (`k_byte_iters=501`) only `#pragma unroll 8` applies, and the
runtime check inside the hot loop doubles code size → catastrophic regression
(comment lines 85-91; empirically confirmed in `R27C_FIX_VERDICT.md` — pfoff
∈ {16,32,64,128,200} regressed 55-60%).

**K_DIM-dependent macros that already adapt to K=128256 fine:**
- `K_BYTES = K_DIM/2` (line 538) — scales linearly, no overflow at K=128256.
- `k_byte_iters = K_BYTES / BK` = 501 — used as loop bound only.
- `static_assert(K_BYTES % BK == 0 ...)` (line 2172) — passes (128256 % 256 == 0).
- `BL_N0_STRIDE = 16 * K_BYTES` (line 719) — fits in `uint32_t` until K ≈ 0.5M.
- Unroll fall-through to `#pragma unroll 8` (line 2448) — already in effect at K=128256.

**There is NO LDS-overflow, VGPR-pressure, scale-table or buffer-math limit at
K=128256.** The R27-C `pfoff=1` build with `R25C_K_LIMIT=131072` ran cleanly
to 5050 TFLOPS — no fault, no spill explosion (`R27C_FIX_VERDICT.md` §4).

**Verdict: LIFT-VIABLE for the K_LIMIT macro. LIFT-DEAD for the underlying
optimisation at K=128256.** Lifting `R25C_K_LIMIT` past 128256 is a 1-line
change with zero correctness risk — but R25C itself does not pay at
K_iters=501 because the runtime branch can't fold under `unroll 8`. The
correct R30 path is the **static peel** (V8 / R25E), not a `K_LIMIT` bump.
Estimated effort to relax the gate safely for static-peel use: **<30 min**
(macro tweak only); but the V8 peel macro itself is a separate ~1.5 hr
project (kernel:80-110 has been rewritten 3× since R25-E, see
`R29_L6_DECIDER_VERDICT.md` §2).

---

## Q2 — Cross-shape variant transplant audit (free wins)

Each shape's `per_variant` dict has 153-160 of the 169 unique variant tags
already populated (computed by union over all 42 results). The R29 L4 win
came from **building a new variant** (pfoff=48 K_EXACT=14336), not
transplanting an existing per_variant entry. For Q2 I filtered out
`kxNNNN`-locked tags (only valid when target K matches) and ranked remaining
gaps by source TFLOPS.

| Target shape | Cur best (TF) | Ratio | Source variant (untested here) | Source shape | Source TF | Rationale |
|---|---:|---:|---|---|---:|---|
| 128256×32768×4096 | 4943 | 109.0% | `ts_lgk2_memc_btw_all` | 4096×28672×32768 | 5432 | small-ish K (4096); lgk2+btw_all parent dominates other large-K-untouched shapes |
| 128256×32768×4096 | 4943 | 109.0% | `ts_lgk2_v12_memc_btw_all` | 4096×28672×32768 | 5431 | as above with v12 step3 axis variant |
| 128256×32768×4096 | 4943 | 109.0% | `v20_memc_btw_step3` | 4096×28672×32768 | 5400 | v20 K-loop sync coarsening — never tried at this MN |
| 32768×14336×2048 | 3726 | 111.2% | `ts_lgk2_memc_btw_all` | 4096×28672×32768 | 5432 | shallow-K LOSE; lgk2 parent unsampled here |
| 16384×4096×4096 | 4462 | 112.9% | `v20_memc_btw_step3` | 4096×32768×28672 | 5335 | shallow-K + small-MN; v20 sync coarsening axis untouched |

**Caveat:** "source TFLOPS = 5432" does NOT predict 5432 on the target — it
predicts that this **parent macro stack** is competitive at *some* shape.
Because each shape is at <0.5% spread between 1st-place and 5th-place
(intra-shape `per_variant` data), a transplant is only likely to move the
needle if it brings a **structurally different parent** (lgk2 vs v12_tv0
vs v20). Realistic gain: **0-3pp p50, 5pp p90.** Total expected new wins
from Q2: 1-2 shapes at most.

**The R29 L4 success was NOT a per_variant transplant; it was a NEW build
with K_EXACT=14336 + pfoff=48.** Q2's "free win" framing is overstated —
the per_variant tables are already saturated to <1% intra-shape.

---

## Q3 — V6 split-K feasibility for L6

For 4096×32768×128256: K_iters=501 (BK=256). N_split=2 → K_iters=250 per
chunk (still > 128 = R25C unroll-fold limit; would need K_LIMIT=64128
gate); N_split=4 → K_iters=125 per chunk (under 128, pragma-unroll fires;
but 32064 is NOT in the K_EXACT enumeration so per-shape pfoff tuning needs
extension).

**Required work:**
1. Host-side workspace alloc: FP32 `[M,N]` accumulator (~1 GiB at 4096×32768).
2. Atomic store path in epilogue: replace `store_bf16_val` with `atomicAdd`
   on FP32 output, then a 2nd-pass bf16 cast kernel.
3. Kernel signature change: extra arg for split index + workspace pointer.
4. Host launch loop: spawn N_split×grid blocks instead of 1×grid.
5. SNR validation on full DLA1.

This is `R29_L6_DECIDER_VERDICT.md` §1 V6, scoped at **≥3 days** by that
decider, with **HIGH** regression risk (atomic contention nullifies gains;
output buffer doubles memory pressure). Expected gain: 3-8pp p50 on L6
**only**, no benefit to other 41 already-WIN shapes.

**Verdict: NO-GO this round.** Split-K is days, not hours. V8 (R25E
static-peel revival) is the cheaper alternative for the same predicted gain
(`R29_L6_DECIDER_VERDICT.md` §3).

---

## Final ranked recommendation

**Optimizer A — V8 R25E static-peel revival on L6** (highest EV per hour):
- Pull `r25e-kpeel` worktree macro `R25E_K_LOOP_PEEL_ITERS` (default 0,
  `R25E_K_LIMIT_LO=65536` so only DLA1 activates).
- Port to current kernel (kernel:80-110 area).
- ISA-audit the compiled `.s` to confirm the compiler emits two distinct
  `#pragma unroll 8` loops, not a re-merged runtime-branched body
  (this is *the* check R26-A skipped — see `R27C_FIX_VERDICT.md`).
- Sweep peel ∈ {4, 6, 8, 12} × 5-rep on idle GPU.
- Expected gain: 1-4pp on L6. Effort: ~1.5-2 hr if macro still applies.

**Optimizer B — Q2 transplant scout on the 3 most promising candidates**
(parallelisable, low risk):
- Build the 3 missing `lgk2_memc_btw_all` / `lgk2_v12_memc_btw_all` /
  `v20_memc_btw_step3` variants for `128256×32768×4096` (the shape with the
  cleanest gap to next-best parent class).
- Build the same 3 for `32768×14336×2048` (also missing lgk2 parent).
- 5-rep bench each on a separate GPU.
- Expected gain: 1 of 6 lands a 1-3pp improvement.
- Effort: ~45 min build + ~30 min bench.

**Defer:**
- **V6 split-K** — multi-day rewrite; postpone to dedicated sprint after V8
  outcome.
- **V7 Stream-K** — 2-week rewrite, weeks of regression risk; unjustified
  while only L6 remains.
- **V5 MFMA32×32×64** — 1.5-2 weeks per `R27_V5_MFMA32_SCOUT.md`; defer.
- **R25C K_LIMIT bump alone** — DEAD at K=128256 (R27-C confirmed); only
  useful as a precondition to V8.

**Stop condition:** if both A and B return zero gain, L6 is structurally
saturated at 92.6% under the current 16×16 MFMA × KPAIR pipeline. The next
viable lever is V5 MFMA32 (multi-week) or a kernel-wide redesign — neither
is an R30 move.

(word count: ~770)
