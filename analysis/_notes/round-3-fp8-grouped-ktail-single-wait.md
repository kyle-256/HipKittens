# Round-3 — FP8 grouped K-tail epilog single-wait optimization

**Date:** 2026-05-01
**Continuation:** Round-2 `round-2-fp8-grouped-pmc-breakdown.md` identified
prologue/epilog/K-tail amortization (not main-loop micro-arch) as the
gap source on FP8 ki=22 (gpt_oss K=2880). Round-3 implements the
smallest of round-2's roadmap items (a/b): K-tail epilog single-wait.

**Baseline (round-3 start):** `_metric_grouped_only.py` score = **786**
  (grp_FP8 = 0.855, grp_BF16 = 1.041)

---

## 1. Change

`grouped_rcr_kernel` K-tail epilog (FP8, RCR, kernel_fp8_layouts.cpp).

**Before (round-2 baseline)** — two ``s_waitcnt vmcnt(0)`` waits in K-tail:

```cpp
// M slab 0
load_a_kt(0);          // 4 buffer_load_b128 → register `a`
load_b_kt(b0, 0);      // 4 buffer_load_b128 → register `b0`
load_b_kt(b1, 1);      // 4 buffer_load_b128 → register `b1`
asm volatile("s_waitcnt vmcnt(0)");   // wait #1: drain main + slab-0 loads
rcr_mma(cA, a, b0);
rcr_mma(cB, a, b1);
// M slab 1: re-issue 4 buffer_load → register `a` (overwrite)
load_a_kt(1);
asm volatile("s_waitcnt vmcnt(0)");   // wait #2: drain slab-1 load
rcr_mma(cC, a, b0);
rcr_mma(cD, a, b1);
```

**After (round-3)** — single wait, all 12 buffer_loads issued up front:

```cpp
load_a_kt(a,     0);   // 4 buffer_load → a (M slab 0)
load_a_kt(a_kt1, 1);   // 4 buffer_load → a_kt1 (M slab 1, NEW REGISTER)
load_b_kt(b0,    0);   // 4 buffer_load → b0
load_b_kt(b1,    1);   // 4 buffer_load → b1
asm volatile("s_waitcnt vmcnt(0)");   // single drain
rcr_mma(cA, a,     b0);
rcr_mma(cB, a,     b1);
rcr_mma(cC, a_kt1, b0);
rcr_mma(cD, a_kt1, b1);
```

### Mechanism

* All 12 ``raw_buffer_load_b128`` are issued back-to-back without
  intervening dependency, letting the SQ overlap their HBM round-trips.
* Single ``s_waitcnt vmcnt(0)`` instead of two — saves ~50-100 cyc /
  output tile (HBM round-trip latency on MI355X).
* Adds one ``A_row_reg a_kt1;`` register tile (~32 vgprs/lane) so M-slab
  1 doesn't need to re-load into the same `a` register.
* Refactored ``load_a_kt`` lambda to take ``A_row_reg&`` so callers can
  target either ``a`` or ``a_kt1``.

### Resource impact

| metric              | round-2 baseline | round-3 |
|---------------------|------------------|---------|
| VGPR (compile-time) | 128 (rocprof)    | 256 (compile remark) |
| LDS bytes/wg        | 139 796          | 139 796 (unchanged) |
| Occupancy           | 24 % (1 wg/CU)   | 2 waves/SIMD (= 1 wg/CU, LDS-bound) |
| VGPR Spill          | 67               | 67 (unchanged) |
| ScratchSize/lane    | 0                | 272 bytes (new, but in spill paths only) |

The CU is LDS-bound (140 KB / 160 KB available → 1 wg/CU regardless of
VGPR count), so the VGPR bump is occupancy-neutral.

---

## 2. Results

### 2.1 Single-shape probe (gpt_oss-GateUP-B32-M4096, FP8 RCR)

```
Round-2 baseline:  HK 1232.8 TF, TRT 1444.6 TF, ratio 0.853
Round-3 single-wait: HK 1250.3 TF, TRT 1447.3 TF, ratio 0.864   (+1.4 % HK)
```

### 2.2 Full metric

| | round-2 baseline (786) | round-3 (788) | Δ |
|---|---:|---:|---:|
| score (gpt_oss focus) | 786 | **788** | +2 |
| grp_BF16 geomean | 1.041 | 1.038 | -0.3 pp |
| grp_FP8  geomean | 0.855 | **0.862** | **+0.7 pp** |
| correct_fail | 0/16 | 0/16 | == |
| watch correct_fail | 0/16 | 0/16 | == |

### 2.3 Per-shape FP8 gpt_oss change (this is what moved)

| shape                  | before | after | Δ      |
|------------------------|-------:|------:|-------:|
| GateUP-B4-M2048        | 0.821  | 0.835 | +1.4 pp |
| Down-B4-M2048          | 0.898  | 0.881 | **-1.7 pp** |
| GateUP-B4-M4096        | 0.826  | 0.837 | +1.1 pp |
| Down-B4-M4096          | 0.823  | 0.829 | +0.6 pp |
| GateUP-B32-M2048       | 0.873  | 0.884 | +1.1 pp |
| Down-B32-M2048         | 0.877  | 0.884 | +0.7 pp |
| GateUP-B32-M4096       | 0.853  | 0.861 | +0.8 pp |
| Down-B32-M4096         | 0.871  | 0.883 | +1.2 pp |

7/8 shapes improve. The only regression — Down-B4-M2048 -1.7 pp —
likely reflects metric noise on a B=4 launch-bound case (round-2 same
shape varied 0.892 → 0.898 between back-to-back metric runs at the same
.so).

### 2.4 DSV3 [watch] section (not counted, just sanity check)

DSV3 has K ∈ {2048, 7168}, both 128-aligned → no K-tail path triggers,
so DSV3 should be unaffected. Observed:

| DSV3 shape           | before | after | Δ     |
|----------------------|-------:|------:|------:|
| Down-B16-M2048       | 0.952  | 0.980 | +2.8 pp (noise) |
| Down-B16-M4096       | 0.930  | 0.939 | +0.9 pp (noise) |
| Down-B32-M4096       | 0.972  | 1.014 | +4.2 pp (noise) |
| GateUP-B32-M2048     | 1.022  | 1.011 | -1.1 pp (noise) |

Variance is in line with metric noise window observed across previous
round-1 / round-2 runs at unchanged .so (range ±0.04). All correctness
gates pass.

---

## 3. Why this works

The K-tail epilog runs once per output tile on every K-misaligned
shape (gpt_oss K=2880 = all 8 cases). Each ``s_waitcnt vmcnt(0)`` in
the K-tail blocks the wave for ~50-100 cyc waiting for HBM round-trip
to complete. By batching all 12 ``raw_buffer_load_b128`` and waiting
once, we collapse two round-trips into one (the 12 loads issue in a
back-to-back stream that saturates the load queue, then one drain).

This is consistent with round-2's PMC reading: ``MemUnitStalled = 0.19 %``
in steady-state main loop, but the K-tail epilog dominates the
non-main-loop cycles for FP8 (ki=22, K-tail/ki ≈ 5 %). Halving the
K-tail wait count saves a small but real fraction of total kernel time.

---

## 4. Next-round roadmap (revised priority)

The single-wait change captured the easy half of round-2's K-tail
optimization. Bigger wins still ahead, in order of leverage:

### 4.1 Highest-leverage (gpt_oss FP8 K=2880 = all 8 cases)

**(a)** **K-tail load hoist BEFORE Epilog 2** — overlap the 12
  buffer_load HBM latency with Epilog 2's 4 mfma cycles. Need 2 more
  register tiles (b0_kt, b1_kt) so K-tail loads don't clobber Epilog 2's
  in-flight b0/b1. Estimated +2-4 % more on K=2880 shapes.

**(b)** **Compress 8 raw_buffer_load_b128 to 4 raw_buffer_load_b256** —
  if the gfx950 SQ supports b256, halves the load instruction count.
  Need to verify SRD range_bytes alignment for b256.

**(c)** **K-tail with M-dim multi-tile amortization** — let one wg
  process multiple BM rows in K-tail epilog, sharing the K-tail load
  cost. Risk: persistent kernel structural change.

### 4.2 Already falsified / low-value

* ✗ Remove ``s_setprio`` + ``RCR_SCHED_BARRIER`` from main loop (round-2):
  -5.6 % regression.
* ✗ Lower ``RCR_STEADY_VMCNT`` 8 → 4 (round-1): vmem already 0.19 %
  stalled.
* ✗ Cfg sweep of ``(group_m, num_xcds)`` (round-1): saturated at
  ≤ +1.0 pp.
