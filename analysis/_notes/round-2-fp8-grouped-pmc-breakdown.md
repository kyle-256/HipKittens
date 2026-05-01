# Round-2 — FP8 grouped GEMM PMC breakdown + s_setprio falsification

**Date:** 2026-05-01
**Focus shape:** `gpt_oss-GateUP-B32-M4096`
  (B=32, M_per_group=4096, N=5760, K=2880, fp8_e4m3 tensorwise, layout=NT/RCR)
**Baseline (HEAD `8622e4bb` → `_metric_grouped_only.py` round-2):**
  grpFP8 segment geomean = 0.851 (HK 1235 TF / TRT 1446 TF, ratio 0.856)
**Continuation:** Round-1 `round-1-gpt-oss-focus-fp8-grouped-baseline.md`
  identified main-loop 1-tile schedule as the gap source. Round-2 verifies
  via PMC and falsifies the "remove scheduler hints" sub-hypothesis.

---

## 1. Hypothesis under test (round-1 next-step #1)

> "rocprof breakdown of `grouped_rcr_kernel<0,true,true>` will show vmem
>  stall % vs lds stall % vs mfma busy % to localize the bottleneck."

We add a side hypothesis from inspecting the source:

> "Main-loop has 8 `__builtin_amdgcn_s_setprio` + 2 `RCR_SCHED_BARRIER` per
>  K-iteration around 4 MFMA. With LDS=140 KB → 1 wg/CU → no other wave to
>  deprio, these scheduler hints are dead weight. Removing them should
>  recover ~2-4 pp."

---

## 2. Method

### 2.1 PMC counter probe (rocprofv3 -i pmc_main.txt)

```text
pmc: MfmaUtil OccupancyPercent
pmc: MemUnitStalled LdsUtil
pmc: LdsBankConflict MfmaFlopsBF16
```

Probe script: `/tmp/probe_fp8_gateup_b32_m4096.py` (HK + Triton, 20 warmup
+ 50 timed iter on the focus shape, both backends in same process via
`force_grouped_gemm_backend`).

Same PMC suite re-run on **BF16 grouped** at the identical shape via
`/tmp/probe_bf16_gateup_b32_m4096.py` for cross-precision contrast.

### 2.2 Falsification prototype (s_setprio + sched_barrier removal)

Edited `kernel_fp8_layouts.cpp` `grouped_rcr_kernel` **main loop only**
(lines 2078-2105), removed 8× `__builtin_amdgcn_s_setprio(1)/(0)` and
2× `RCR_SCHED_BARRIER()`. **Retained** all `__builtin_amdgcn_s_barrier()`
and `s_waitcnt` — they are required for cross-warp LDS visibility.
Recompiled `tk_fp8_layouts.so`, ran probe, compared TFLOPS, reverted.

---

## 3. Results

### 3.1 PMC: HK FP8 grouped vs Triton FP8 grouped (same shape)

| metric            | HK `grouped_rcr_kernel` | Triton `_grouped_fp8_persistent_gemm_kernel` | Δ        |
|-------------------|------------------------:|--------------------------------------------:|---------:|
| MfmaUtil          | **35.2 %**              | **42.7 %**                                  | TRT +7.5 pp |
| LdsUtil           | 12.8 %                  | 21.9 %                                      | TRT +9.1 pp |
| MemUnitStalled    | 0.19 %                  | 0.43 %                                      | tiny     |
| LdsBankConflict   | 0                       | 0                                           | ==       |
| OccupancyPercent  | 24.4 %                  | 24.1 %                                      | ==       |
| VGPR              | 128                     | 124                                         | ==       |
| LDS bytes/wg      | 140 288                 | 0 (static `__shared__`)                     | n/a      |
| Workgroup size    | 512                     | 512                                         | ==       |
| Grid size         | 131 072                 | 131 072                                     | ==       |

### 3.2 PMC: HK BF16 grouped vs Triton BF16 grouped (same shape)

| metric            | HK `grouped_kernel`     | Triton `_grouped_bf16_persistent_gemm_kernel` | Δ        |
|-------------------|------------------------:|----------------------------------------------:|---------:|
| MfmaUtil          | **56.0 %**              | **54.6 %**                                    | HK +1.4 pp |
| LdsUtil           | 20.2 %                  | 20.6 %                                        | ==       |
| MemUnitStalled    | 0.12 %                  | 0.28 %                                        | tiny     |
| LdsBankConflict   | 0                       | 0                                             | ==       |
| OccupancyPercent  | 23.9 %                  | 23.6 %                                        | ==       |

HK BF16 ≈ Triton BF16 in every PMC dimension; the kernel template is
healthy on the same shape with a longer ki.

### 3.3 Per-kernel timing (kernel-trace, FP8 path, 70 launches each)

```
HK  grouped_rcr_kernel               70  total=192.9 ms  avg=2755.6 µs
TRT _grouped_fp8_persistent_gemm     70  total=156.8 ms  avg=2240.4 µs
unary_kernel (quantize cast)        280  total= 63.9 ms  avg= 228.3 µs
reduce_row_kernel<AbsMaxOp>         840  total= 47.3 ms  avg=  56.3 µs
compute_scale_from_amax_kernel      280  total=  1.3 ms  avg=   4.6 µs
compute_group_offs_device           140  total=  0.7 ms  avg=   4.8 µs
```

Quantize overhead (~112 ms across both backends) is identical for HK and
Triton — both go through the same `turbo.ops.grouped_gemm_fp8` wrapper.
The entire HK→Triton ratio gap (516 µs / 2756 µs = **18.7 %**) lives in
the main GEMM kernel, NOT in dispatch / quantize / `group_offs` setup.

### 3.4 Falsification: remove `s_setprio` + `RCR_SCHED_BARRIER` from main loop

| variant                                                    | HK TF/s | TRT TF/s | ratio  | Δ vs base |
|------------------------------------------------------------|--------:|---------:|-------:|----------:|
| Baseline (HEAD)                                            | 1232.8  | 1444.6   | 0.853  | —         |
| Main loop without `s_setprio` + `RCR_SCHED_BARRIER`        | 1164.6  | 1445.0   | 0.806  | **-5.6 %** |

Removing the scheduler hints **HURT** performance by 5.6 %. The hints
are doing real work — `__builtin_amdgcn_s_setprio` raises wave priority
to bias issue toward MFMA, and `__builtin_amdgcn_sched_barrier(0)`
forbids compiler reordering across the barrier. Even with a single
wave-group per CU (LDS=140 KB → no co-resident wg), the priority bump
biases issue to MFMA over its co-located VALU/SALU instructions inside
the same wave's instruction stream. The sched barrier prevents the
compiler from inadvertently scheduling non-MFMA instructions inside the
MFMA's hidden-latency window.

**The HK schedule is NOT over-engineered; the hints earn their cost.**

→ `kernel_fp8_layouts.cpp` reverted (md5 back to `be641b8c`).

---

## 4. Interpretation

The two PMC tables together resolve the contradiction:

* **HK BF16 grouped** matches Triton on the same shape (MFMA Util 56 % vs
  54 %, LDS Util 20 % vs 21 %). Same template, same scheduler hints.
  **Template is fine.**
* **HK FP8 grouped** is 7.5 pp behind Triton on MFMA Util AND 9.1 pp
  behind on LDS Util on the same shape. Same template, same hints.

What changes between BF16 and FP8 on this shape is purely:

| dim               | BF16 grouped    | FP8 grouped     |
|-------------------|----------------:|----------------:|
| BK                | 64              | 128             |
| K                 | 2880            | 2880            |
| **ki = K/BK**     | **45**          | **22**          |
| MFMA latency      | ~16 cyc         | ~8 cyc          |

**FP8 has half the K-iterations** (22 vs 45) — so the prologue (~2 iter
worth of cycles) + epilog 1 (~1 iter) + epilog 2 (~1 iter) +
**fused K-tail** (1 iter, FP8-only because gpt_oss K=2880 % 128 = 64 ≠ 0
but K % 64 = 0 so BF16 has no K-tail) account for **~5 / 22 = 23 %** of
total kernel cycles in FP8 vs **~4 / 45 = 9 %** in BF16. Each non-main-
loop phase has lower MFMA density than the main loop (more vmem-wait,
more reordering, last-tile awkward register pressure).

The same `__builtin_amdgcn_s_setprio` schedule that hits 56 % MFMA Util
in the BF16 long main loop only achieves 35 % when averaged across an
FP8 short main loop + heavy prologue/epilog/K-tail tail.

**Conclusion:** the bottleneck is **prologue/epilog/K-tail amortization,
not main-loop micro-schedule.** Round-1's "main loop micro-arch rewrite"
hypothesis is **partially refuted** — main-loop is fine; what's needed
is to reduce the *fraction* of cycles spent in non-main phases.

---

## 5. Next-round roadmap (revised, by leverage)

### 5.1 Highest-leverage (shape-specific, gpt_oss FP8 K=2880 only)

**(a) Fuse K-tail epilog into Epilog 2.** Currently:
  - Epilog 2: load_a + load_b + s_waitcnt(vmcnt=0) + 4 mfma (~80 cyc)
  - K-tail:   8× `raw_buffer_load_b128` + s_waitcnt(vmcnt=0) + 4 mfma (~120 cyc)

  Total tail = 200 cyc. If we issue the K-tail's 8 buffer_load *before*
  Epilog 2's mfma, the K-tail's HBM latency overlaps with Epilog 2's
  MFMA → ~120 cyc saved per output tile, **~5-7 %** speedup on K=2880
  shapes (8 of 8 gpt_oss). Risk: medium (need to keep two register
  banks free during overlap; A_row_reg / B_row_reg overlap). Effort:
  2-3 hr (write + SNR verify on all 8 K=2880 shapes + metric).

**(b) Combine K-tail "M slab 0" + "M slab 1" into single load batch.**
  Current code does 4 buffer_load (a slab 0 + b0 + b1) → wait → 2 mfma
  → 4 buffer_load (a slab 1) → wait → 2 mfma. Two HBM round-trips with
  the same waited-vmcnt latency. If we batch all 8 buffer_loads up front
  → 1 wait → 4 mfma, we save 1 HBM round-trip latency (~50-100 cyc).

### 5.2 Medium-leverage (BF16-style schedule extension)

**(c) Port BF16 grouped's `Bs[3]` 3-stage LDS rotation to FP8.** BF16
  uses deeper buffer; FP8 uses 2-stage (`Bs[2][2]`). Adding a stage
  costs 16 KB LDS (already at 140 KB → may force occupancy drop).
  Test: synthetic 3-stage prototype, measure LDS over 156 KB blocks
  occupancy from 24 % to 12 % (if so, abandon).

### 5.3 Lower-leverage but worth doing

**(d) Port dense 2-tile main loop into `grouped_rcr_kernel`.** Already
  present in dense for `ki >= RCR_TWO_TILE_MIN_KI = 28` (currently
  even-ki only). gpt_oss FP8 ki=22 won't trigger, BUT DSV3 FP8 K=7168
  ki=56 would. Marginal for current focus (gpt_oss-only score).

### 5.4 Falsified / deprioritized

* ✗ Lower `RCR_STEADY_VMCNT` 8 → 4 (round-1's #2): MemUnitStalled = 0.19 %
  → vmem already not the wait. Would only add risk of vmem stall
  appearing.
* ✗ Remove `s_setprio` + `RCR_SCHED_BARRIER` (this round's prototype):
  -5.6 % regression measured.
* ✗ Tune `(group_m, num_xcds)` cfg (round-1 sweep): saturated at ≤ +1.0 pp.
* ✗ Lower `RCR_TWO_TILE_MIN_KI` 28 → 20 (round-1 dense probe): only
  +1.4 % on `ki=22` dense, and grouped ki=22 path is 1-tile only, so
  the 2-tile path isn't the bottleneck.

---

## 6. Verification of revert

```
$ md5sum analysis/fp8_gemm/mi350x/tk_fp8_layouts.cpython-312-x86_64-linux-gnu.so
be641b8cf16dc2160c49b0496957ba5c   # matches HEAD baseline
```

Probe re-run after revert:
```
HK  FP8 GateUP-B32-M4096: 1232.8 TF, 3.5275 ms / iter   # = baseline
TRT FP8 GateUP-B32-M4096: 1444.6 TF, 3.0102 ms / iter
```

No code change committed to either repo this round. Only this
analysis-notes file added.

---

## 7. Open data files (untracked, for round-3 self-resume)

* `/tmp/probe_fp8_gateup_b32_m4096.py` — single-shape FP8 grouped probe
* `/tmp/probe_bf16_gateup_b32_m4096.py` — single-shape BF16 grouped probe
* `/tmp/pmc_main.txt` — rocprofv3 PMC counter set
* `/tmp/rocprof_pmc/` — FP8 PMC output dir
* `/tmp/rocprof_pmc_bf16/` — BF16 PMC output dir
* `/tmp/rocprof_out/` — FP8 kernel-trace dir
* `/tmp/metric_round_2.log` — round-2 baseline metric stderr
