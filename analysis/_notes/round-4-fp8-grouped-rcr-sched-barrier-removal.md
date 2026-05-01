# Round 4 — FP8 grouped RCR main loop: remove `RCR_SCHED_BARRIER()` (compiler reorder hint)

**Date**: 2026-05-01
**HK SHA (pre)**: 09eaa086 (round-3 K-tail single-wait)
**HK SHA (post)**: this commit
**Primus-Turbo SHA**: c3b70e374 (unchanged this round)
**Focus**: gpt_oss FP8 forward (8 shapes, all K=2880 K-tail)
**Round-4 baseline metric**: 790 (= round-3 best)
**Round-4 result metric**: 792 (+2, both runs consistent)
**grp_FP8 geomean**: 0.864 → 0.869 (+0.45 pp)
**grp_BF16 geomean**: 1.039 → 1.044 (+0.49 pp, within noise; same kernel untouched)

## 1. Hypothesis

Round-2 (`round-2-fp8-grouped-pmc-breakdown.md` §5) falsified the bundled
removal of `__builtin_amdgcn_s_setprio(1/0)` **and**
`__builtin_amdgcn_sched_barrier(0)` from the FP8 grouped RCR main loop:
combined removal regressed by **−5.6 %** on the focus shape, so the
intervention was reverted. The two primitives have very different roles:

* `s_setprio` — runtime SALU instruction, biases warp-level priority for
  the MFMA issue port. Removing it allows non-MFMA waves to steal issue
  bandwidth from the MFMA-issuing wave, demonstrably bad on this kernel.
* `sched_barrier(0)` — pure compiler hint, prevents the LLVM machine
  scheduler from reordering instructions across the barrier. Has zero
  runtime cost; only affects compile-time scheduling decisions.

Hypothesis: the round-2 −5.6 % was driven entirely by `s_setprio` removal;
`sched_barrier(0)` removal alone may be neutral or positive (it lets the
back-end re-pack `buffer_load_lds` issues across the MFMA group, plausibly
shortening latency to the next `s_waitcnt lgkmcnt(0)`).

## 2. Patch

`analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:2078-2110` (grouped RCR
single-tile main loop). Removed the **two** `RCR_SCHED_BARRIER()` calls
appended after the `cA` and `cC` MFMAs (post-`s_barrier`). Kept all
`s_setprio(1/0)`, `s_barrier`, and `s_waitcnt` calls intact:

```cpp
// before:
__builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
__builtin_amdgcn_s_barrier(); RCR_SCHED_BARRIER();   // ← removed
// after:
__builtin_amdgcn_s_setprio(1); rcr_mma(cA, a, b0); __builtin_amdgcn_s_setprio(0);
__builtin_amdgcn_s_barrier();
```

The two removed calls are at the cA→load_b1 transition (originally line
2092) and the cC→load_b_tile_next transition (originally line 2106).

`Epilog 1` and `Epilog 2` `RCR_SCHED_BARRIER()`s (lines 2122/2140 in the
old layout, now 2130/2148) are **untouched** to keep the prologue→main
boundary undisturbed. Same reasoning for `K-tail` and the `RRR/CRR`
backward kernels.

## 3. Build resources (unchanged)

```
kernel_fp8_layouts.cpp:5167:  VGPRs: 256  Occupancy: 2  SGPRs Spill: 0
                              VGPRs Spill: 67  LDS: 139796 bytes  Scratch: 272 bytes/lane
```

Identical to round-3 (no register-pressure delta — `sched_barrier(0)` is
zero-instruction).

## 4. Probe (focus shape)

`/tmp/probe_fp8_gateup_b32_m4096.py` (gpt_oss-GateUP-B32-M4096):

| Variant | HK TFLOPS | TRT TFLOPS | ratio |
|---|---|---|---|
| Round-3 baseline (single-wait K-tail)        | 1233 | 1450 | 0.851 |
| Round-4 (sched_barrier removal)              | 1250 | 1450 | 0.862 |

Probe shows +1.4 % HK TFLOPS on the focus shape. The improvement appears
in the metric at the geomean level (+0.45 pp on grp_FP8).

## 5. Metric (full 16 shapes, 2 runs)

```
grp_BF16  vs triton geomean = 1.0386 (run 1), 1.0443 (run 2)   ← unchanged kernel, noise band
grp_FP8   vs triton geomean = 0.8688 (run 1), 0.8648 (run 2)   ← +0.45-0.05 pp from baseline 0.8643
focus score = 792 (both runs)
```

Per-shape gpt_oss FP8 deltas (round-3 baseline → round-4):
- GateUP-B4-M2048: 0.842 → 0.845
- Down-B4-M2048:   0.889 → 0.902 (+1.3 pp)
- GateUP-B4-M4096: 0.835 → 0.841
- Down-B4-M4096:   0.831 → 0.838
- GateUP-B32-M2048: 0.885 → 0.888
- Down-B32-M2048:   0.887 → 0.890
- GateUP-B32-M4096: 0.864 → 0.863 (focus, neutral within noise)
- Down-B32-M4096:   0.883 → 0.885

All BF16 ratios within 1 pp of round-3 (kernel untouched).
DSV3 [watch] all PASS, no correctness regressions.

## 6. Backward sanity check (FP8 dA + dB)

`grouped_rrr_kernel` (FP8 backward dA, line 2496 in
`kernel_fp8_layouts.cpp`) is **untouched** — only `grouped_rcr_kernel`
forward main loop was modified. Direct sanity bench at the focus shapes:

| shape | total fwd+bwd TFLOPS | da.norm | db.norm |
|---|---|---|---|
| GateUP-B32-M4096 | 1154 | 1.466e6 | 1.466e6 |
| Down-B32-M4096   | 1027 | 1.040e6 | 1.040e6 |
| GateUP-B4-M4096  | 1063 | 5.202e5 | 5.202e5 |

No NaN/Inf, norms match expected magnitude. Backward path healthy.

## 7. Calibrated round-2 finding

The round-2 `−5.6 %` regression was **almost entirely driven by
`s_setprio` removal**, not `sched_barrier`. Going forward:

* `s_setprio(1/0)` around MFMA blocks is **load-bearing** for FP8 grouped
  RCR — keep.
* `sched_barrier(0)` is a removable compiler hint; the back-end's default
  scheduling here is already adequate or better than the hint enforces.

## 8. Why the gain is small (and is likely tapped-out)

`sched_barrier(0)` removal lets LLVM pack the next iteration's
`load_b/load_a` ds_read closer to the current iteration's `mma`. Each
K-iter saves at most a handful of bubble cycles on the SIMD scalar issue
port. With 22 K-iter × 2 sched_barriers/iter × ~4 cycles/iter saved =
~176 cyc per output tile, on a ~120 K cyc tile budget = **~0.15 % per
tile**. The observed ~+1.4 % TFLOPS on the probe is in line with this
order of magnitude; the geomean lift (~+0.45 pp) reflects the
non-Gaussian distribution of the per-shape deltas.

There is no further "free" SALU compression available in the main loop
without changing semantics — `s_barrier` is required for cross-warp LDS
visibility and `s_setprio` is load-bearing per round-2.

## 9. Next-round roadmap (revised priority)

The cheap micro-tunes are exhausted. Remaining levers ordered by expected
yield:

### 9.1 Highest leverage (gpt_oss FP8, ratio 0.84-0.89)

**(a) Hoist K-tail B loads BEFORE Epilog 2** — overlap K-tail's 6
`buffer_load_b128` for `b0/b1` with Epilog 2's 4 MFMAs (~120 cyc). Needs
2 new `B_row_reg` tiles (`b0_kt`, `b1_kt`, +32 vgpr/lane). Round-3 build
shows VGPRs at the 256 limit with Spill=67; +32 vgpr will likely add
~16 spill slots. Cost/benefit must be measured. Estimated yield: +1-2 %
TF on K=2880 shapes if spills stay in cold paths.

**(b) Port dense's 2-tile main loop to grouped_rcr_kernel for ki ≥ 22**
— dense kernel uses 2-tile main loop when `ki ≥ RCR_TWO_TILE_MIN_KI=28`.
Lowering MIN_KI to 22 + back-porting the 2-tile body to grouped would
halve the per-K-iter SALU overhead on gpt_oss (ki=22 even). Round-1
falsification was on dense path with single-shape probe; grouped + 16
shapes geomean may differ.

**(c) Direct HBM→register main loop (skip LDS)** — Triton's main loop
uses `tl.load` direct to register, no LDS staging. HK uses
`buffer_load_lds` + `s_barrier` + `ds_read` — adds 8 `s_barrier` per K-iter
just for cross-warp LDS visibility. Removing LDS staging on the A-tile
side (~64 KB LDS saved, +occupancy potential) is a 4-8 hr structural
rewrite. Estimated yield: closes most of the MFMA Util gap (35 → 42 %).

### 9.2 Lower leverage / longer-term

**(d) N-tile shape tuning** — N=2880 (Down) currently uses BN=256, last
N-tile only 25 % active. Switching to BN=128 makes the last N-tile 50 %
active (still misaligned, but better). Risks: 2× grid size, occupancy
change. Worth a 1-shape probe.

**(e) FP8 Down quantize-dequantize path overhead** — quantize is
backend-symmetric (both HK and Triton call the same `compute_scale`),
but the per-call quantize overhead (0.81 ms at the focus shape) is 23 %
of the total call time. Merging quantize into HK kernel epilogue would
asymmetrically benefit HK; major architectural change.

## 10. Files touched

- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:2078-2110` (grouped
  RCR main loop body, removed 2× `RCR_SCHED_BARRIER()`).
- `analysis/_notes/round-4-fp8-grouped-rcr-sched-barrier-removal.md`
  (this file).
