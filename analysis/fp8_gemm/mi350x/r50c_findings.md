# R50 Dev C — occupancy=1 LDS-boost at 4096³ family: hypothesis REFUTED before prototype

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 03c94ab3 (R49 wrap)
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=3
**Scope:** Test r49c §7.3's promising lever — that at 4096³-family shapes
where `total_tiles == nCU` the kernel runs at occupancy=1 and the second
"occupancy slot" of LDS (~80 KB) can be reclaimed for an A-prefetch
buffer or an extra scale-stage. Build a 4096³-specific variant, gate by
`MXFP8_OCC1_LDS_BOOST=1`, validate correctness, A/B vs baseline.

---

## TL;DR — VERDICT: REFUTED (analytical, prompt §"Bail conditions")

**The premise that 4096³ has a wasted second-occupancy LDS slot is
structurally false. The kernel ALREADY runs at 1 CTA/CU on every shape
because it ALREADY uses 128–136 KB of the 160 KB per-CU LDS budget — the
second CTA cannot fit on any shape, including the dispatch-headroom
shapes (8192³, 8B Gate/Up, 70B Gate/Up). There is no shape-specific
lever to unlock.**

| Layout | Per-CTA LDS | 2 CTAs require | Per-CU LDS limit | Max CTAs/CU (LDS) | "Free" LDS at occ=1 |
|---|---|---|---|---|---|
| RCR (V2, SCALE_VERSION=2)  | 131072 B = **128 KB** | 256 KB | 160 KB | **1** | 32 KB |
| RRR (V2, SCALE_VERSION=2)  | 135168 B = **132 KB** | 264 KB | 160 KB | **1** | 28 KB |
| CRR (V2, SCALE_VERSION=2)  | 139264 B = **136 KB** | 272 KB | 160 KB | **1** | 24 KB |

The prompt's bail clause:

> If after reading kernel + computing LDS budget you find that the
> existing kernel ALREADY uses >80 KB LDS per CU (no slack for a second
> buffer), document and REFUTE without prototyping.

128–136 KB > 80 KB on all 3 layouts → **bail clause triggered**. No
kernel changes; no prototype.

The "Occupancy [waves/SIMD]: 2" remark in build logs that r49c §7.3
quoted is the **per-CTA wave count per SIMD** (8 wavefronts × 1 CTA / 4
SIMDs = 2 waves/SIMD), NOT the CTAs/CU count. The compiler reports the
maximum compatible occupancy from the VGPR/SGPR budget (212–254 VGPR ≤
512 budget), but the actual deployed CTAs/CU is gated by the more
restrictive resource — here, LDS — to 1 CTA/CU on every shape. r49c §7.3
misread this metric.

---

## 1. Hardware verification (independent of prompt and r49c)

`hipDeviceProp_t` query on HIP_VISIBLE_DEVICES=3 (gfx950):

```
device: AMD Instinct MI355X
multiProcessorCount (CUs):                            256
maxThreadsPerMultiProcessor:                         2048
sharedMemPerBlock:                                 163840 B = 160.0 KB
sharedMemPerMultiprocessor:                     41943040 B  (40 MB — L2 cache, NOT addressable LDS)
hipDeviceAttributeMaxSharedMemoryPerMultiprocessor: 163840 B = 160.0 KB  (per-CU LDS, what matters)
regsPerBlock:                                      131072
regsPerMultiprocessor:                             131072
```

CU count agrees with r49c (256, NOT 304). Per-CU LDS is **160 KB** —
this is the constraint for 2-CTA-per-CU occupancy.

The `sharedMemPerMultiprocessor` value of 40 MB is the *L2 cache* on
this query, not addressable LDS. The correct addressable LDS attribute
is `hipDeviceAttributeMaxSharedMemoryPerMultiprocessor`, which reports
160 KB.

Source: /shared_nfs/kyle/test/Hipkittens2/.claude/worktrees/agent-a8573652
device verified live (binary at /tmp/lds_query, output captured).

---

## 2. LDS budget computation per kernel variant

Build at M=N=K=4096 with the production defaults (PRESHUFFLED_QUANT=1,
SCALE_VERSION=2, swizzles ON, SCALE_LDS_ENABLE=0). Build remarks per
fastpath kernel symbol from the `-Rpass-analysis=kernel-resource-usage`
output:

```
kernel_mxfp8_layouts.cpp:2415  rcr_exact_8wave_scaled_kernel<true, 2>:
    TotalSGPRs: 53     VGPRs: 254     Occupancy [waves/SIMD]: 2
    LDS Size [bytes/block]: 131072      Spill: 0/0

rrr_mxfp8_exact_8wave_fastpath.inc:145  rrr_exact_8wave_scaled_kernel<true, 2>:
    TotalSGPRs: 46     VGPRs: 254     Occupancy [waves/SIMD]: 2
    LDS Size [bytes/block]: 135168      Spill: 0/0  (ScratchSize=68/lane)

crr_mxfp8_exact_8wave_fastpath.inc:273  crr_exact_8wave_scaled_kernel<true, 2>:
    TotalSGPRs: 50     VGPRs: 227     Occupancy [waves/SIMD]: 2
    LDS Size [bytes/block]: 139264      Spill: 0/0
```

These match r31_4gpu_runs/build_mxfp8_*.log (recorded ~R31 cycle, same
kernel structure). Per-CTA LDS has not changed across cycles.

**Geometric breakdown** for V2-RCR (Layout 0):
- `__shared__ ST_A As[2][2]` — 4 × `st_fp8e4m3<128, 128, ...>` = 4 × 16384 = 65536 B
- `__shared__ ST_B Bs[2][2]` — 4 × `st_fp8e4m3<128, 128, ...>` = 4 × 16384 = 65536 B
- Subtotal A+B double-buffer = **131072 B** (matches reported)

For V2-RRR add ~4096 B for `ST_rrr_b` swizzle metadata = 135168 B.
For V2-CRR add ~8192 B for the col-layout A reencoding stage = 139264 B.

**Conclusion**: A+B double-buffer alone consumes 128 KB. The second
occupancy slot would require 128 KB more for a redundant A+B buffer set
— total 256 KB > 160 KB per-CU budget. **No shape can run at 2 CTAs/CU
without restructuring the A/B staging itself.**

---

## 3. Why the r49c §7.3 reading is incorrect

r49c §7.3 wrote:

> The current build reports `Occupancy [waves/SIMD]: 2`, meaning the SPI
> *could* place 2 CTAs per CU if it had 512+ tiles to dispatch — but at
> 4096³ it has exactly 256, so the second occupancy slot is empty.

The AMD `Occupancy [waves/SIMD]` metric is the **per-CTA wavefront
density** the kernel supports given its register budget. With 8 wavefronts
per CTA (`_NUM_WARPS=8`) and 4 SIMDs per CU, one CTA places 2
wavefronts on each SIMD → `2 waves/SIMD`. The compiler is reporting
that this register footprint *would not require* dropping to fewer
CTAs/CU — but the per-CU multiplexing (CTAs/CU) is a separate quantity
that has to be computed from the *minimum* of the LDS bound, the VGPR
bound, the SGPR bound, and the threads bound.

For this kernel:

| Constraint | Per-CTA cost | Per-CU budget | Max CTAs/CU |
|---|---|---|---|
| LDS                     | 131–139 KB  | 160 KB     | **1** ← binding |
| VGPRs (per wave)        | 254         | 1024 / SIMD × 4 = 4096 | 4 |
| SGPRs                   | 53          | many       | many |
| Threads                 | 512         | 2048       | 4 |

LDS is the binding constraint → 1 CTA/CU on every shape, every K, every
M, every N. The dispatch grid size (256 vs 512 vs 1024) only changes
how many waves are spawned over time, not the per-CU steady-state
multiplexing.

Concrete check: at 8192³ (1024 tiles), the kernel processes 1024 / 256
= 4 successive waves. Each wave is a fresh CTA per CU; only 1 CTA per
CU is resident at any instant due to LDS. The "occ=2 fills the device"
intuition would only hold if each CTA fit in ≤ 80 KB LDS, which it does
not.

---

## 4. Empirical baseline anchor (5 runs/cell, GPU 3, strict SCLK)

Confirms the published R48 baselines and provides ground-truth for the
findings doc. 8B Q/O (4096³) on all 3 layouts, 5 runs each, 30s
cooldown, MXFP8_WARMUP=100, MXFP8_ITERS=200, MXFP8_PRESHUFFLE_QUANT=1.

| Shape  | Layout | Median TFLOPS | Range (min..max) | Median Avg time (ms) |
|---|---|---|---|---|
| 4096³  | RCR    | **2477** | 2339..2488 | 0.0555 |
| 4096³  | RRR    | **2453** | 2444..2465 | 0.0561 |
| 4096³  | CRR    | **2283** | 2263..2286 | 0.0602 |

Correctness (MXFP8_CHECK=1, det_runs=3):

| Layout | SNR (dB) | Pass rate | Result |
|---|---|---|---|
| RCR | 49.61 | 16777216/16777216 (100.00%) | PASS |
| RRR | 49.61 | 16777216/16777216 (100.00%) | PASS |
| CRR | 49.61 | 16777216/16777216 (100.00%) | PASS |

These TFLOPS values are within ±1% of R48 wrap-baseline numbers (anchor
verified). The CRR ~8% gap vs RCR is the same structural CRR-at-4096³
issue documented in r49c §7.2.

Files: `r50c_results/baseline_8B_QO_{rcr,rrr,crr}_run{1..5}.log`,
`r50c_results/correctness_8B_QO_{rcr,rrr,crr}.log`.

No comparable treatment column exists because the bail clause prevents
prototyping a kernel change.

---

## 5. Could we use the 24–32 KB free LDS for a small scale-prefetch?

In principle: yes, 24 KB free at occ=1 is enough for a small B-scale
prefetch stage (~4 KB). But:

1. **The lever is not shape-specific.** All shapes already run at 1
   CTA/CU due to LDS, so any small scale-prefetch we add would apply
   uniformly to all shapes — there is no `total_tiles == nCU` gate to
   key off, contrary to r49c §7.3's framing. The hypothesis as stated
   (4096³-specific LDS boost) is unfounded.

2. **The 24 KB free at occ=1 was ALREADY the budget for prior cycles'
   scale-prefetch work and was NULL'd:**
   - `MXFP8_RCR_V2_SCALE_PREFETCH` (gate at line 432 of
     kernel_mxfp8_layouts.cpp) — defaulted OFF after R31C scale-stage
     work, never shipped.
   - `MXFP8_RCR_EXACT_PQ_SCALE_LDS_ENABLE` (line 372) — defaulted OFF.
   - `MXFP8_RCR_V2_SCALE_CACHEPOLICY` / `MXFP8_RCR_ASCALE_CACHEPOLICY`
     (lines 389–406) — R47C swept and shipped a partial change for
     CRR only; no LDS-stage win.
   - `MXFP8_RCR_EXACT_PQ_PREFETCH_NEXT_PAIR_ENABLE` (line 378) — OFF.

   These exhaustive sweeps over the same 24 KB headroom mean the
   "spend free LDS on scale prefetch" idea has been tested at least
   four times across cycles R31–R47, all with NULL or NEAR-NOISE
   outcomes.

3. **The actual bottleneck at 4096³ is K-loop interior.** R49c §7.2
   already identified this — at 4096³ the 256-tile wave fits the device
   exactly so the gap must come from the K-loop body (B-tile LDS
   bank conflicts at N=4096 specifically, or the CRR pre-fetch
   lead-distance issue), NOT from buffer depth.

The promising-lever framing in r49c §7.3 was based on a misread of the
occupancy metric. Once corrected, the lever evaporates.

---

## 6. SHIP gate decision

**REFUTED — no `MXFP8_OCC1_LDS_BOOST` macro added. No kernel changes
beyond this findings doc + a results dir + the 5-run anchor bench.**

Per prompt §"Bail conditions":
> If after reading kernel + computing LDS budget you find that the
> existing kernel ALREADY uses >80 KB LDS per CU (no slack for a second
> buffer), document and REFUTE without prototyping.

128–136 KB per CTA > 80 KB threshold → bail clause triggered on all 3
layouts. The hypothesis is structurally invalid because the kernel does
not have, and never had, a wasted second occupancy slot at any shape.

---

## 7. Alternative levers for the 4096³-family HEADROOM cells

The R49 cycle wrap (commit f0efd077) refuted 5 of the most plausible
mechanisms, leaving the following untouched:

1. **B-tile LDS bank-conflict pessimism at N=4096** (r49c §7.2 first
   bullet) — never directly probed at this N (most RRR R&D was N=14336
   / N=28672). Proposal: compile a 4096³-RRR variant with the inverse
   `st_16x128_v3_s` swizzle and bench. Estimated 1-day, low-risk.

2. **CRR scale-prefetch lead-distance** (r49c §7.2 second bullet) —
   still untouched. The CRR cA pre-fetch lgkmcnt edge could be moved
   one BK earlier. R45 reviewer + R47D both flagged the CRR-at-4096³
   ~8% gap vs RCR/RRR as structurally consistent — this is the only
   K-loop interior lever not yet attempted at this specific shape.

3. **CRR partial-unroll sweep at K=4096** — analogous to R48G's RRR
   sweep, but on CRR and at the smaller shapes. R48G found bimodal
   spill at N≥2 for RRR; CRR may have a different sweet spot. This is
   the most promising remaining mechanism per the R49 wrap analysis.

None of these involve LDS depth. The LDS depth lever is closed.

---

## 8. Files

- `analysis/fp8_gemm/mi350x/r50c_findings.md` — this document
- `analysis/fp8_gemm/mi350x/r50c_results/run_baseline_5x.sh` — 5-run anchor
- `analysis/fp8_gemm/mi350x/r50c_results/build_baseline_4096cube.log` — build remarks
- `analysis/fp8_gemm/mi350x/r50c_results/baseline_8B_QO_{rcr,rrr,crr}_run{1..5}.log` — 5×3=15 runs
- `analysis/fp8_gemm/mi350x/r50c_results/correctness_8B_QO_{rcr,rrr,crr}.log` — det+SNR

No source modifications. The hypothesized `MXFP8_OCC1_LDS_BOOST` macro
is not introduced because the prototype is REFUTED before introduction
per the prompt's bail clause.
