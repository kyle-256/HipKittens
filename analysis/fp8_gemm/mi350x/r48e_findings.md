# R48 Dev E — MXFP8 RRR Full-Unroll Lever (K=4096) — REFUTED

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 80bf861e (R48B wrap)
**GPU:** MI355X (gfx950), HIP_VISIBLE_DEVICES=3
**Lever:** R48 Dev D §2.2 / P3 — force MXFP8 RRR K-loop full unroll at K=4096 to
match FP8 RRR's compiler-elided loop (1024 straight-line MFMAs vs MXFP8's 16
K-pair iterations with ~3 loop-control instr per pair).

**TL;DR — VERDICT: REFUTED.** Forcing `#pragma unroll N` (any N >= 2) on the
MXFP8 RRR K-pair loop in `rrr_mxfp8_exact_8wave_fastpath.inc` causes
**catastrophic VGPR spill** (62-67 spill bytes, 208 bytes/lane scratch) and
**~70% performance regression across ALL 7 RRR shapes** (K=4096, 8192, 14336,
28672). The looped baseline is the local optimum given the current `do_k_iter`
lambda (always_inline) structure. Macro `MXFP8_RRR_MAIN_UNROLL` left in tree
**default OFF (=0)** for one-flag repro and as future hook if the lambda is
refactored.

This is one of the two outcomes Dev D's prediction P3 explicitly bracketed:
> "Forcing full unroll (`RRR_MAIN_UNROLL=32`) will produce +1-2pp on 8B Q/O RRR
> ... **OR** will produce a register-spill regression that nets ≤0pp. Either
> outcome confirms the loop-control overhead identification."

The result confirms the loop-control overhead identification, but rules out
this lever as a path to closing the gap.

---

## 1. Method

### 1.1 Code change

`analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc`:

```cpp
// Macro gate (default 0 = no #pragma unroll, baseline behavior).
#ifndef MXFP8_RRR_MAIN_UNROLL
#define MXFP8_RRR_MAIN_UNROLL 0
#endif
#if MXFP8_RRR_MAIN_UNROLL > 0
#define MXFP8_RRR_PRAGMA_UNROLL_MAIN \
    _Pragma(TK_STRINGIFY(unroll MXFP8_RRR_MAIN_UNROLL))
#else
#define MXFP8_RRR_PRAGMA_UNROLL_MAIN
#endif

// Applied to the K-pair loop:
{
    constexpr int k_iters_main = k_iters - 2;
    constexpr int k_pairs = k_iters_main / 2;
    MXFP8_RRR_PRAGMA_UNROLL_MAIN
    for (int kp = 0; kp < k_pairs; kp++) {
        load_scale_packs(kp);
        do_k_iter(kp * 2,     phase_t<0>{}); tic ^= 1; toc ^= 1;
        do_k_iter(kp * 2 + 1, phase_t<1>{}); tic ^= 1; toc ^= 1;
    }
}
```

Also promoted `k_iters_main` and `k_pairs` to `constexpr` so the loop trip
count is a compile-time constant for the unroll directive.

### 1.2 Build / dump / bench

Same env + Makefile invocation as r48d_findings.md §1.1. Per-K-pair body
extraction unchanged.

```bash
cd analysis/fp8_gemm/mi350x
# baseline
$HIPCC kernel_mxfp8_layouts.cpp ... -DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096 \
  --cuda-device-only -S -o r48e_isa_dumps/mxfp8_4096_baseline_device.s
# unroll=32
$HIPCC kernel_mxfp8_layouts.cpp ... -DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096 \
  -DMXFP8_RRR_MAIN_UNROLL=32 \
  --cuda-device-only -S -o r48e_isa_dumps/mxfp8_4096_unroll32_device.s
```

Bench: `r48e_bench.sh` (full 7-shape × 2-tag matrix) and `r48e_sweep.sh`
(unroll factor sweep on 2 K=4096 cells).

---

## 2. ISA Verification

K=4096, MXFP8 RRR V2 kernel
(`_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EE...`):

| Metric                                | Baseline (no unroll) | Unroll=32             |
|---------------------------------------|---------------------:|----------------------:|
| Kernel size (lines of `.s`)           | 1711                 | 8355 (4.9× larger)    |
| `v_mfma_*` count                      | 128 (loop body × 1 unrolled K-pair + epilogue) | **1024** (fully straight-line) |
| `; =>This Inner Loop Header` markers  | 1 (LBB3_7)           | **0** (loop eliminated) |
| `scratch_load`/`scratch_store` count  | **0**                | **392** (heavy spill) |
| VGPRs                                 | 254                  | 256                   |
| **VGPRs spill (bytes)**               | **0**                | **67**                |
| ScratchSize [bytes/lane]              | 0                    | 208                   |
| Occupancy [waves/SIMD]                | 2                    | 2 (unchanged)         |

**Loop-control elimination CONFIRMED.** The unrolled MXFP8 RRR matches the FP8
RRR fully-unrolled structure exactly (1024 MFMA, no inner loop header).

**But:** 392 scratch ops = 392 register spills/reloads per kernel invocation.
The do_k_iter lambda (`__attribute__((always_inline))`) inlines 8 MFMAs +
16 ds_reads + scale-pack registers + 4 buffer_loads + setprio + 6 barriers
per call; with 30 inlined copies (15 K-pairs × 2 phases), the live-range
pressure exceeds the 256-VGPR window. The compiler resolves with scratch
spill, which destroys per-cycle MFMA throughput.

The single-pass sweep on smaller unroll factors (2, 4, 8) showed
identical 208 bytes/lane scratch and identical regression. The compiler
unrolls at any value of N>=2 and immediately spills. This is a structural
limitation of the current `do_k_iter` always-inline + fused scale-pack
design, not an unroll-factor tuning issue.

---

## 3. Performance Table

3-run mean, 50 warmup + 100 iter, GPU 3 (HIP_VISIBLE_DEVICES=3),
20s cooldown between runs.

### 3.1 Full 7-shape matrix (baseline vs unroll=32)

| Cell | M×N×K | Baseline TFLOPS | Unroll=32 TFLOPS | Δ | Verdict |
|---|---|---:|---:|---:|---|
| **8B Q/O (K=4096 target)** | 4096³ | 2368.4 | 679.8 | **−71.30%** | **REGRESS** |
| **8B Gate/Up (K=4096 target)** | 4096×14336×4096 | 2491.5 | 749.8 | **−69.90%** | **REGRESS** |
| 8B Down | 4096×4096×14336 | 2954.4 | 595.0 | −79.86% | REGRESS |
| 70B Q/O | 4096×8192×8192 | 2878.6 | 706.9 | −75.44% | REGRESS |
| 70B Gate/Up | 4096×28672×8192 | 2788.3 | 739.5 | −73.48% | REGRESS |
| 70B Down | 4096×8192×28672 | 2976.8 | 572.3 | −80.78% | REGRESS |
| 8192³ | 8192×8192×8192 | 2893.0 | 765.2 | −73.55% | REGRESS |

### 3.2 Unroll factor sweep on K=4096 cells

| Cell | Baseline | u=2 | u=4 | u=8 | u=32 |
|---|---:|---:|---:|---:|---:|
| 8B Q/O 4096³ | 2368.4 | 804.8 | 810.2 | 803.2 | 679.8 |
| 8B Gate/Up | 2491.5 | 777.3 | 778.7 | 778.2 | 749.8 |

Every unroll factor produces ~70% regression. The compiler's `unroll N` heuristic
(for N>=2) commits to spilling once given the directive. There is **no sweet
spot** with the current kernel structure.

### 3.3 SHIP gate evaluation

- Perf gain ≥ +1.5% on K=4096 RRR cells: **NO** (−71% regression).
- Net positive across 7 RRR shapes: **NO** (all 7 regress).
- Worst regression ≤ 2%: **NO** (worst regression −80.78%).

→ **REFUTED.** No SHIP.

---

## 4. Why this happens (root-cause analysis)

The MXFP8 RRR fastpath body (`do_k_iter` lambda) per K-pair contains:
- 8 `rrr_mma_scaled_phase` invocations (each 8 MFMAs)
- 16 `ds_read*` (LDS reads for A, B tiles)
- 2 `buffer_load` (VMEM→LDS tile fills for the next K-pair)
- 2 `buffer_load` (scale loads via `load_scale_packs`)
- 6 `s_barrier` + `s_waitcnt`
- 4 `setprio` priority changes

The lambda is `__attribute__((always_inline))`, so each loop iteration inlines
this entire block. With 15 K-pair iterations × 2 phases = 30 inlined copies,
the resulting straight-line code's live-range graph exceeds the 256-VGPR
allocation window.

The per-K-pair scale state (`a0_scale_packs`, `a1_scale_packs`,
`b0_scale_packs`, `b1_scale_packs` — 4 packs × 4-byte each per K-pair) is the
key contributor: between adjacent K-pairs the compiler cannot prove the scale
registers are dead until the MFMA chain consumes them, so the live ranges
overlap across the inlined copies.

**FP8 RRR avoids this** because (a) it has no scale registers — the per-K-pair
body is structurally simpler (~234 instr vs ~268 in MXFP8 per Dev D §2.1),
and (b) the FP8 kernel's 4-`rrr_mma` structure (vs MXFP8's
4-`rrr_mma_scaled_phase` with embedded scale-pack consumption) leaves a
narrower live-range graph. The compiler heuristic for FP8 evaluates the unroll
to be profitable; for MXFP8 it correctly evaluates it as a regression and
stays looped.

### 4.1 What WOULD work (future R49+ exploration)

To recover the +1-2pp loop-control overhead estimated by Dev D §2.2, the
kernel would need restructuring:
1. **Move `do_k_iter` from `__attribute__((always_inline))` to noinline +
   manual specialization on phase_t**: forces the compiler to share register
   allocation across phases.
2. **Unroll K-pair loop manually by 2 (or 4)** with explicit live-range hints
   (e.g., `__builtin_amdgcn_kill` to mark scale packs dead before the next
   iteration's load).
3. **Use the V2-RRR scale-SRD pre-loaded indirectly** instead of explicit
   scale_packs registers, reducing live VGPR count.

These are structural rewrites — out of scope for a single-lever R48 cycle.

---

## 5. Outcome

- **Macro `MXFP8_RRR_MAIN_UNROLL` left in tree default OFF (=0)** — preserves
  baseline behavior. Set to any positive value to force `#pragma unroll N` on
  the K-pair loop (intended for future repro / experimentation).
- No kernel behavior change in production builds.
- ISA dumps + bench logs preserved as `r48e_isa_dumps/`, `r48e_*.log`.
- Confirms Dev D's P3 prediction in the "spill regression" branch.

## 6. Files

| Path | Purpose |
|---|---|
| `rrr_mxfp8_exact_8wave_fastpath.inc` | Modified — adds macro gate (default OFF) |
| `r48e_isa_dumps/mxfp8_rrr_4096_baseline_kernel.s` | Extracted V2 RRR kernel (1711 lines) |
| `r48e_isa_dumps/mxfp8_rrr_4096_unroll32_kernel.s` | Extracted V2 RRR kernel unrolled (8355 lines) |
| Full device `.s` (1.1 MB / 2.1 MB) | NOT committed — regenerate with §1.2 build commands |
| `r48e_bench.sh` | Full 7-shape × 2-tag bench driver |
| `r48e_sweep.sh` | Unroll factor sweep (2/4/8) on K=4096 cells |
| `r48e_baseline_*.log` (21 files) | Baseline TFLOPS, 7 shapes × 3 runs |
| `r48e_unroll_*.log` (21 files) | Unroll=32 TFLOPS, 7 shapes × 3 runs |
| `r48e_unroll{2,4,8}_*.log` (12 files) | Sweep TFLOPS on 2 K=4096 cells × 3 runs |
| `r48e_bench.out`, `r48e_sweep.out` | Raw bench script stdout |

## 7. Cross-references

- `r48d_findings.md` §2.2, §5.1 item 2, §6 P3 — origin of this lever.
- `rrr_exact_8wave_fastpath.inc` line 156 — FP8 RRR's `TK_PRAGMA_UNROLL(RRR_MAIN_UNROLL)`
  with `RRR_MAIN_UNROLL=4` baseline (works because FP8 body is smaller — see §4).
- `r48a_findings.md`, `r48b_findings.md` — sibling R48 cycle work.

## 8. One-Line Summary

**Forcing `#pragma unroll N` on MXFP8 RRR K-pair loop achieves the ISA goal
(1024 straight-line MFMAs, no loop control) but causes catastrophic VGPR
spill (392 scratch ops) and ~70% perf regression across ALL 7 RRR shapes.
Lever REFUTED. The +1-2pp loop-control overhead identified by Dev D is real
but unreachable via compiler-pragma alone — requires kernel restructure
(noinline phase split + manual register-pressure relief) for R49+.**
