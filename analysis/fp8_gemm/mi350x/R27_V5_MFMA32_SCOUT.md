# R27 Scout — Vector V5: MFMA_32X32X64 Tiling Rewrite

Date: 2026-04-18  Branch: mxfp4  Scout: claude-opus-4-7 (R27-Scout, no-GPU/no-build)

Background: After R26 we have 33/42 WIN, 9 LOSE. R26_PLAN §3 V5 flagged the
`v_mfma_scale_f32_32x32x64_f8f6f4` rewrite as the only structural axis still
untouched. R5/Optimizer-D (Round 5) wrote it off as "≥1 week, 0-5pp" but never
actually built a prototype. This doc evaluates whether to fund it.

---

## 1. Current MFMA usage (16x16x128 path)

### Where it lives
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp` is **fully self-contained**:
  the 16x16x128 MFMA is emitted **inline** as raw asm strings, **not** through
  the HipKittens `mma.cuh` wrapper. There are **512 occurrences** of the literal
  `v_mfma_scale_f32_16x16x128_f8f6f4` in the kernel (`grep -c`).
- The functions that emit MFMAs (all `__forceinline__`):
  - `kpair_64mfma_step12` — `kernel_mxfp4_gluon_cpp.cpp:1255` (steady-state K-loop body, A0×Bl + A0×Br interleaved with `ds_read_b128`)
  - `kpair_64mfma_step34` — `kernel_mxfp4_gluon_cpp.cpp:1392` (rare `FUSED_STEP34=1` path; default 0)
  - `kpair_64mfma_step12_swapped_sel` — `kernel_mxfp4_gluon_cpp.cpp:1937` (op_sel-swapped variant for SWAP_STEP12)
  - several other `step12` / `step34` site variants (lines 969, 1042, 1104, 1160, 1235, 1550, 1645, 1732, 1793, 1841, 2067) — all use the same MFMA shape, varying only in barrier/waitcnt layout.
- HipKittens `include/ops/warp/register/tile/mma.cuh` provides:
  - `mfma1616128(D[2], A[8], B[8], C[2])` at line 112 — unscaled FP4.
  - `mfma1616128_scaled<opsel_a, opsel_b>(D[2], A[8], B[8], C[2], scale_a, scale_b)` at line 128 — scaled FP4.
  - `mfma323264(D[8], A[8], B[8], C[8])` at line 97 — **unscaled FP8e4m3 only**, **no FP4 + scale variant exists**.
  - `mma_AB_base` dispatches by `D_shape` (line 162 ff.), routing only to `mfma1616128` for the FP4 path (line 188-193).
  - `mma_AB_base_scaled` (line 199) **only** dispatches to `mfma1616128_scaled` and `static_assert`s on any other shape (line 218-225).

### Per-warp tile / accumulator footprint (16x16x128)
- Block shape: `BLK=256, BK=128, WARPS_M=2, WARPS_N=2 → 4 warps/block`
  (`kernel_mxfp4_gluon_cpp.cpp:529-536`, with `RBM=RBN=64`).
- Per-warp output region = 64×64 floats (`kernel_mxfp4_gluon_cpp.cpp:544`):
  `using RT_C = rt_fl<RBM, RBN, col_l, rt_16x16_s>` → 4×4 grid of 16×16 base tiles.
- The kernel splits the per-warp 64×64 across **4 separate accumulator arrays**
  (`kernel_mxfp4_gluon_cpp.cpp:2304`):
  `acc_A0Bl[16], acc_A0Br[16], acc_A1Bl[16], acc_A1Br[16]` — each `floatx4_t[16]`.
  Each `floatx4_t` is the 4-fp32 output footprint of one 16x16x128 MFMA per lane.
- Total per-warp AGPR fp32 footprint = 4 × 16 × 4 = **256 fp32 AGPRs**.
- The four sub-blocks tile a 128×128 region per warp; combined with `WARPS_M*WARPS_N=4`
  warps this covers the 256×256 block. Each `acc_AxBy` is a 64×64 sub-block.

### Issued ASM — confirmed in K-loop body
`kernel_mxfp4_gluon_cpp-hip-amdgcn-amd-amdhsa-gfx950.s` lines 900-1100 confirm
back-to-back `v_mfma_scale_f32_16x16x128_f8f6f4 a[X:X+3], v[..], v[..], a[X:X+3], v_scale_a, v_scale_b op_sel*:[..] cbsz:4 blgp:4` with interleaved `ds_read_b128` / `buffer_load_dwordx4 ... lds`:
- L900-910: 11 consecutive 16x16x128 MFMAs writing `a[100:103]…a[124:127]` (one of the four step12 sub-blocks).
- L923-955: full step12 body — 24 MFMAs interleaved with 16 `ds_read_b128` and 4 `buffer_load_dwordx4 ... offen lds`.

### Kernel footprint @ gfx950 (from compiled `.s`)
`kernel_mxfp4_gluon_cpp-hip-amdgcn-amd-amdhsa-gfx950.s:5778, 5812-5815`:
- `.amdhsa_accum_offset 256`
- `TotalNumSgprs: 91`
- `NumVgprs: 253`
- `NumAgprs: 256`  ← **at the 256 ceiling for archVGPRs+accVGPRs split**
- `TotalNumVgprs: 512` ← gfx950 hard ceiling

So we are **fully saturated on the AGPR side** (256/256). VGPR side has 3 slots
free (253/256), giving the kernel 1 wave / SIMD = **occupancy 1**.

---

## 2. MFMA_32X32X64 candidate (CDNA4, MI355X / gfx950)

### Intrinsic exists in HipKittens
`include/ops/warp/register/tile/mma.cuh:97-110` (function `mfma323264`):
```
*(floatx16_t*)D = {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    *(intx8_t*)A, *(intx8_t*)B, *(floatx16_t*)C,
    0, 0, 0, 0, 0, 0)};
```
The **scaled** form takes the same 6-trailing-args pattern as `mfma_scale_f32_16x16x128_f8f6f4` (last 4 are `opsel_a, scale_a, opsel_b, scale_b`); `mfma1616128_scaled` (L137-147) shows the pattern. Adapting to 32x32x64 requires only changing the per-thread output vector size from `floatx4_t` (4 fp32 / lane / inst) to `floatx16_t` (16 fp32 / lane / inst) and the C/D reg span from `a[X:X+3]` to `a[X:X+15]`.

### Per-instruction shape comparison

| Metric | 16x16x128 | 32x32x64 |
|---|---|---|
| Output per inst | 16×16 fp32 (256 fp32 / inst, 4 per lane × 64 lanes) | 32×32 fp32 (1024 fp32 / inst, 16 per lane × 64 lanes) |
| K reduced per inst | 128 | 64 |
| FLOPs per inst | 16·16·128·2 = 65 536 | 32·32·64·2 = 131 072 (2×) |
| Issue interval (CU/cycle, theoretical) | 16 cycles | 32 cycles |
| FLOP-rate per CU/cycle | 4 096 | 4 096 (identical peak) |

**Insight**: 32x32 doubles FLOPs/inst at 2× the issue interval. **Peak compute throughput is identical** — there is no headline TFLOPS gain from raw arithmetic.

### Per-warp coverage @ same 64×64 output / same K step

To cover the kernel's per-warp 64×64 output and a single 128-K chunk:
- 16x16x128: **(4×4) tiles × 1 K-chunk = 16 instructions** per 64×64×128.
- 32x32x64: **(2×2) tiles × 2 K-chunks = 8 instructions** per 64×64×128.

So **32x32 cuts MFMA-instruction issue count in half** for the same compute. **This is the only structural win.** It directly addresses the thing R24B/C identified
(`TCP_DATA_STALL >290%`) only if MFMA-issue contention contributes to that stall window.

### AGPR footprint @ kernel's current per-warp 64×64×4 (acc_A0Bl + acc_A0Br + acc_A1Bl + acc_A1Br)
- 16x16x128: 4 sub-blocks × 16 tiles × 4 fp32 = **256 fp32 AGPRs** (current).
- 32x32x64: 4 sub-blocks × 4 tiles × 16 fp32 = **256 fp32 AGPRs** (identical).

**The AGPR footprint is mathematically the same** because both shapes write the same total 4×64×64 = 16 384 fp32 per warp. The difference is **how many MFMA instructions write into that AGPR pool** (32x32 issues each AGPR a wider write but fewer times per K-chunk).

The R5 / Optimizer-D claim that 32×32 "still needs 256 AGPRs" was correct, but the
implication that "therefore no gain" missed the **MFMA-issue-bandwidth** axis.

---

## 3. Rewrite scope estimate

### Files that need changes

| File | Change | LOC churn |
|---|---|---|
| `kernel_mxfp4_gluon_cpp.cpp` | Rewrite all 512 inline `v_mfma_scale_f32_16x16x128_f8f6f4` to `v_mfma_scale_f32_32x32x64_f8f6f4`. Each `kpair_64mfma_step{12,34}*` body shrinks from 32 MFMAs to 16 MFMAs but with `a[X:X+15]` operands instead of `a[X:X+3]`. The four `acc_AxBy` arrays change from `fp4_floatx4_t[16]` to `fp4_floatx16_t[4]`. The `op_sel*` bit pattern that currently picks 4 sub-tiles within a 32-wide register collapses (32x32 issues one full 32×32 per inst, no op_sel sub-selection of `[0,1,0]`/`[1,0,0]` to recover). The K-pair pattern (sa0/sa1 raw scales) stays — `mfma_scale_f32_32x32x64_f8f6f4` takes the same scale operand layout. **LARGE** | ~600-1000 LOC modified |
| `include/ops/warp/register/tile/mma.cuh` | Add `mfma323264_scaled<opsel_a,opsel_b>(...)` wrapper (mirror of L128-148). Add `mma_AB_base_scaled` dispatch arm for `D_shape == rt_32x32 && A_cols==64 && B_rows==64`. **SMALL** | ~30 LOC added |
| `kernel_mxfp4_gluon_cpp.cpp` LDS read pattern (lines 1277, 1279, … `ds_read_b128 %32, %70 offset:0/2048/4096/6144`) | The `ds_read_b128` offsets are computed from `RBM/RBN=64` and `BK=128` strides; the **value** of each lane's FP4 input changes: a 32x32x64 expects each lane to hold 8 ints = 32 FP4 values × 64 K-step = same 8 per-lane ints. Shape constraint on the LDS swizzle: `st_16x128_s` is built around 16-row sub-tiles; 32x32 needs the LDS A/B tiles to lay out 32-row sub-tiles. The HipKittens `st_32x*` swizzles exist (`include/ops/warp/register/tile/mma.cuh:97` works on `st_32x*` for the FP8 path), but `fp4_load_st_to_rt` (`kernel_mxfp4_gluon_cpp.cpp:610`) is hard-coded to `RT::base_tile_rows/cols`. **MEDIUM** | ~150 LOC modified |
| `kernel_mxfp4_gluon_cpp.cpp` store path (`store_block` L3135, `store_block_inner` L3161) | Output fragment shape changes from 4×4 sub-tiles of 16×16 to 2×2 sub-tiles of 32×32. The lane→element mapping for the 32x32 MFMA output is **different** from 16x16 (32x32 stores 16 fp32/lane in **4 disjoint 4-row groups** — see CDNA4 ISA Table for `v_mfma_scale_f32_32x32x64`). Both `store_bf16_val` and `store_bf16x2_packed` paths need rewriting. **MEDIUM** | ~120 LOC modified |
| `include/ops/warp/memory/util/util.cuh` and friends | LDS swizzle used by `st_fp8e4m3<HB, BK, st_16x128_s>`: needs verification that `st_32x*` (or similar) handles the FP4 packed-byte representation correctly. The 16x16 lane mapping is NOT compatible with 32x32 for SHARED→REGISTER loads. **MEDIUM-LARGE** | unknown; ~200 LOC if a new swizzle is needed |
| 17 build-time macro flags `BARRIER_TO_WAITCNT_*`, `STEP3_BARRIER_VMCNT`, `MXFP4_TAIL_*`, `R25C_TAIL_PF_OFF_ITERS`, `R25E_K_LOOP_PEEL`, `MXFP4_R19B_VMCNT_STR`, etc. | All `vmcnt(N)` and `lgkmcnt(N)` thresholds were tuned for **16x16 instruction counts**. Halving the MFMA count per K-iter shifts the optimal `vmcnt(N)` downward by ~2× — every R17-R26 wire needs re-sweeping. **MEDIUM** (sweeps, not LOC) | re-tune all of R17-R26 |

**Total estimated rewrite**: 1 100-1 500 LOC core + a parallel re-sweep of all
R17-R26 barrier/waitcnt knobs. Calendar time **1.5-2 weeks** of a focused
optimizer agent (longer than R5 estimated).

---

## 4. Risk assessment

### AGPR / VGPR pressure
- AGPR footprint is identical (256 → 256), so occupancy projection is unchanged: still 1 wave/SIMD.
- The risk is **transient register pressure**: each 32x32 MFMA writes a full
  16-fp32 vector at once, vs the 16x16 path that writes 4 fp32. The compiler
  needs to schedule the AGPR allocator around longer-lived regions. This **may
  spill** if the M-K-N interleave pattern is wrong, dropping into spill traffic
  that erases all gain. **Risk: MEDIUM-HIGH.**
- VGPR side: 32x32 input operand layout requires 8 ints/lane (same as 16x16x128 inputs), so VGPR pressure stays at ~253/256. **Risk: LOW.**

### LDS pressure
- Per-block LDS = `2 × HB × BK × sizeof(fp8e4m3) = 2 × 128 × 128 × 1 = 32 KB`
  (A double-buffered) + same for Bl/Br. Total `~96 KB` already exceeds 64 KB; the
  kernel already uses 160 KB LDS via XCD-resident swizzle. 32x32 needs **identical**
  LDS bytes (compute is the same), only the sub-tile **swizzle** differs.
  **Risk: LOW** for capacity. **Risk: MEDIUM** for new bank-conflict patterns.

### Correctness risk
- HipKittens has `mfma323264` (unscaled FP8) and the unscaled `mfma_scale_f32_32x32x64_f8f6f4` builtin works in the toolchain (just not wired for FP4-scaled). No known reference GPU kernel in this tree uses 32x32x64 + scale + FP4. AITER's ASM kernel (`competitor_tflops` source) is what we benchmark against, but its source is not in-tree. ROCm `composable_kernel` may have one — would require external lookup. **Risk: MEDIUM** — first 32x32x64-scaled-FP4 kernel in this codebase. SNR validation (vs the existing torch reference path used in `bench_all_42.py`) is mandatory at every checkpoint.

### Schedule risk
- All 17+ R17-R26 macro tunings (`STEP3_BARRIER_VMCNT`, `R25C_TAIL_PF_OFF_ITERS`, `MXFP4_R19B_VMCNT_STR`, etc.) were sweep-fitted to the **current MFMA instruction count**. After a 32x32 rewrite, every `vmcnt(N)` must be re-tuned. Conservative estimate: 2-3 days of sweep-and-bench just to recover R26-equivalent on the **WIN** shapes (regression risk). **Risk: HIGH for net regression** before any new gain materializes.

---

## 5. Predicted gain on the 9 LOSE shapes (from `bench_all42_results_R25_FINAL.json`)

| Shape (M×N×K) | gap | best variant | dominant bottleneck (per R21-recon / R24B-C) | 32x32x64 likely help? |
|---|---|---|---|---|
| 4096 × 32 768 × 128 256 (DLA1) | -7.2pp | `ts_gm8_v12_btw_step3` | K-loop length (501 iters), VMEM-issue saturation | **YES** — fewer MFMAs per iter frees VMEM-issue ports if MFMA is co-issued; +2-4pp realistic |
| 4096 × 32 768 × 28 672 | -5.3pp | `ts_gm8_v12_btw_step3` | mega-N + deep-K, B-tile reload pressure | **WEAK** — B-tile is the pain, MFMA cnt is not; +0-2pp |
| 14 336 × 4 096 × 32 768 | -5.3pp | `ts_pf4_memc_btw_step3` | deep-K, mega-M | **YES** — MFMA-issue saturation symptom; +2-3pp |
| 16 384 × 4 096 × 28 672 | -5.0pp | `v20_memc_btw_step3` | deep-K, mega-M | **YES** — same class; +2-3pp |
| 4096 × 28 672 × 32 768 | -3.5pp | `ts_pf4_memc_btw_step3` | deep-K | WEAK — see DLA1 cousin; +0-2pp |
| 4096 × 14 336 × 16 384 | -2.8pp | `ts_lgk2_memc_btw_all` | mid-K mega-N | **NO** — already at 97.2%, gap likely scale-load related; +0-1pp |
| 16 384 × 4 096 × 14 336 | -2.1pp | `ts_u16` | mega-M, mid-K | WEAK — gap is small; +0-1pp |
| 4096 × 32 768 × 14 336 | -1.1pp | `ts_lgk2_memc_btw_all` | mega-N, mid-K | NO — already 98.9%; not worth rewrite for 1pp |
| 4096 × 32 768 × 6 144 | -0.3pp | `ts_gm2_v12_memc_btw_all` | small-K | NO — already 99.7% |

**Aggregate prediction**: 4 shapes are plausibly +2-4pp candidates (DLA1, 14336×4096×32768, 16384×4096×28672, 4096×32768×128256). Best case: **all 4 flip to WIN, +3pp average → 37/42 WIN**. Worst case: rewrite regresses 2-3 currently-WIN shapes that were tightly tuned to 16x16 vmcnt thresholds, leaving us at 33/42 net. Honest p50 estimate: **34-35/42 WIN** after 2 weeks.

The R5/Optimizer-D estimate of "0-5pp on K-bound deep-LOSE" is essentially correct — but **the AGPR 4× claim was wrong** (footprint is identical). The real cost is **engineering time** + the **macro re-sweep**.

---

## 6. Recommendation

### Verdict: **BACKBURNER**

Reasoning:
1. **The structural mechanism is sound** (halving MFMA-issue rate is a real axis; AGPR footprint is unchanged, contradicting R5's pessimism). It is genuinely the only structural axis left.
2. **The expected gain is small** (1-2 net flips, 33→35 of 42, or +1-2pp avg on 4 shapes).
3. **The cost is high** (1.5-2 weeks of focused work + full macro re-tune) and the **regression risk on already-WIN shapes is real** (R17-R26 barrier wires were fitted to 16x16 cycle counts).
4. **There are no zero-build / cheap experiments left to confirm value before paying the rewrite cost** — unlike R25-G/H, you can't sweep a macro to validate.

This means V5 belongs in a **quiet-week sprint** when no other levers are available, not as the next round.

### What should happen first (cheaper R27 candidates)
Before paying the V5 cost, exhaust:
- **Shape-conditional pfoff** for the 9 LOSE shapes (extend R25-G/H K-EXACT gates to the actual K values listed above — 4 of the 9 have K NOT in the current K_EXACT set: 128 256, 28 672 already covered, 14 336 already covered, 16 384 already covered, 32 768 already covered → actually only K=6 144 is in the gate list among LOSEs). **Verify `best_variant` JSON to confirm each LOSE actually picked an R25-G/H wire.**
- **R26_PLAN V4 re-run**: TAIL_BARRIER_VMCNT × shape sweep on top of the R25-H stack (only partially run in R26).
- **R26_PLAN V3**: STEP3_PF_N=4/2 × R25-G stack (partially run, look at R26 results).

### If V5 is funded later — first 3 concrete code changes
1. **Add `mfma323264_scaled<opsel_a, opsel_b>` to `include/ops/warp/register/tile/mma.cuh`** mirroring the existing `mfma1616128_scaled` (L128-148). ~25 LOC, no behavior change for existing callers. Compile-check + write a 1-block correctness microtest using the unscaled `mfma323264` reference + manual scale.
2. **Add `mma_AB_base_scaled` 32x32 dispatch arm** (`include/ops/warp/register/tile/mma.cuh:218-225`). ~20 LOC.
3. **Build a `MXFP4_USE_32X32` macro-gated alternate path of `kpair_64mfma_step12`** that uses 8 MFMAs/half instead of 32, with `acc_A0Bl[4]` (`fp4_floatx16_t`) instead of `acc_A0Bl[16]` (`fp4_floatx4_t`). Run **single-shape SNR + perf** on DLA1 (4096×32 768×128 256) at 5-rep smoke, NOT the full 42-shape bench. Decision gate: if smoke shows ≥+1pp on DLA1 with no SNR regression, proceed to full rewrite. If <+1pp or any SNR drop, **DEAD**.

### Files referenced
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp:529-546` — block/tile constants
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp:610-630` — `fp4_load_st_to_rt` (LDS→reg, layout-coupled)
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp:1255-1391` — `kpair_64mfma_step12` (steady-state K-loop body)
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp:2304` — accumulator allocation
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp:3134-3207` — store path
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp-hip-amdgcn-amd-amdhsa-gfx950.s:5778, 5812-5815` — register footprint
- `include/ops/warp/register/tile/mma.cuh:97-110` — existing `mfma323264` (unscaled, FP8e4m3)
- `include/ops/warp/register/tile/mma.cuh:128-148` — existing `mfma1616128_scaled` (FP4 scaled, current)
- `analysis/fp8_gemm/mi350x/bench_all42_results_R25_FINAL.json` — 9 LOSE shapes
- `analysis/fp8_gemm/mi350x/R26_PLAN.md:74-79` — V5 original brief
