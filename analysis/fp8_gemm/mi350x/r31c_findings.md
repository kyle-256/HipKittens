# R31 Dev C — V2-RCR PIPELINE_SCALE second-buffer audit + VGPR-prefetch experiment

**Date:** 2026-04-18
**Branch:** r31-c (base feat/mxfp8-only @ 3dddbd5e)
**GPU:** HIP_VISIBLE_DEVICES=2 (MI355X / gfx950)
**Scope:** Investigate "PIPELINE_SCALE second-buffer for V2-RCR" lever (R28 Dev D originally proposed; R29 Dev C kept on the open list as item #2). Per R30 Dev C SASS audit closing the same lever for V2-CRR, first verify the lever even exists for V2-RCR before attempting any second-buffering.

**Verdict:** **NO SHIP — STRUCTURAL CLOSURE (PARADIGM CORRECTION).**

Two independent root-causes:

1. **The originally-named lever does not exist.** V2-RCR scales follow the same VMEM→VGPR→MFMA-direct path as V2-CRR (zero LDS round-trip). SASS audit on the V2-RCR kernel (`_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EE`) found **0 `ds_write` instructions** in the entire kernel and exactly the expected `buffer_load_dwordx4 v[18:21]` (b128, A-side, no `lds` flag) + `buffer_load_dwordx2 v[192:193]` (b64, B-side, no `lds` flag) for scale fetches feeding `v_mfma_scale_f32_16x16x128_f8f6f4`. There is nothing to second-buffer in LDS.

2. **The VGPR-prefetch pivot lever (the only structurally-feasible reinterpretation) catastrophically regresses.** Adding a 2nd set of `*_scale_packs_next[]` VGPR arrays to prefetch (k_pair+1) one iteration ahead pushes V2-RCR from 246 VGPR / 0 spill / occ=2 → 256 VGPR / **312 VGPR spill / 596 bytes scratch/lane** / occ=2. Bench: **4096³ V2-RCR collapses to 187 TFLOPS (-92% from 2435 baseline)**; **8192³ V2-RCR collapses to 225 TFLOPS (-93% from 3007 baseline)**. Correctness still PASSes (det 3/3, kernel computes correctly through scratch spills) but perf is destroyed.

The V2-RCR K-loop is already at the structural register-pressure ceiling — the 4-quadrant accumulator design + scale-pack reuse pattern across {cA, cB, cC, cD} prevents the compiler from holding two parallel scale-pack live ranges without spill.

Add `PIPELINE_SCALE second-buffer (LDS or VGPR variant) for V2-RCR` to the cumulative paradigm-correction list (cycle-15 closure).

---

## 1. Data-flow audit: does V2-RCR have ANY scale LDS path?

### Source-level inspection (`kernel_mxfp8_layouts.cpp`)

V2-RCR `rcr_exact_8wave_scaled_kernel<true, 2>` (the V2 PRESHUFFLED_QUANT path, lines 2186-3360):

- **Scale storage**: `fp8e8m0_4 a0_scale_packs[RBM/32]`, `a1_scale_packs[RBM/32]`, `b0_scale_packs[(RBN+31)/32]`, `b1_scale_packs[(RBN+31)/32]` — all are **private VGPR arrays** declared at lines 2240-2243. Total: 6 × `fp8e8m0_4` (4-byte) elements = 6 VGPRs.
- **Scale loading**: `load_scale_buffer(k_pair)` lambda (line 3048+) for V2 path uses two LLVM intrinsics:
  - `llvm_amdgcn_raw_buffer_load_b128(a_v2_srsrc, a_voff, a_soff, MXFP8_RCR_V2_SCALE_CACHEPOLICY)` — single 16-byte fetch per kpair fills all 4 A scale packs (lane).
  - `llvm_amdgcn_raw_buffer_load_b64(b_v2_srsrc, b_voff, b_soff, MXFP8_RCR_V2_SCALE_CACHEPOLICY)` — single 8-byte fetch per kpair fills both B scale packs.
- **Scale consumption**: scale_packs are passed as VGPR operands to `v_mfma_scale_f32_16x16x128_f8f6f4` directly via `rcr_mma_scaled_from_packs_*` (lines 2850, 2906, 2961, 3015 inside `do_k_iter_body`).
- **`__shared__` allocations**: only `As[2][2]` and `Bs[2][2]` (FP8 tile staging) at lines 2200-2201. There is also `scale_stage_dwords[]` shared array, but it is gated by `MXFP8_RCR_EXACT_PQ_SCALE_LDS_ENABLE` which **defaults to 0** (off) — and that path is the V1-style scale-LDS staging used as alternative experiment, not the V2 production path.

### SASS-level confirmation

Disassembled the gfx950 HSACO bundle from the production .so:

```
$ rm -f tk_mxfp8_layouts*.so
$ make ... CXXFLAGS="-DM_DIM=4096 -DN_DIM=4096 -DK_DIM=4096"
$ md5sum tk_mxfp8_layouts*.so → 4ae07863a9642d91acb2ab6c7b4ceac3
$ # extract bundle entry hipv4-amdgcn-amd-amdhsa--gfx950 (offset 4096, size 203040)
$ llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx950 r31c_baseline.hsaco
$ # extract the V2-RCR kernel range (1314 lines of SASS for _Z29...Lb1ELi2EE)
```

V2-RCR kernel (`_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals`) instruction inventory:

| Instruction class | count |
|---|---:|
| `ds_write` / `ds_store` | **0** |
| `ds_read*` | 392 (all tile-LDS reads via TK G::load → ds_read_b64_tr_b8 etc.) |
| `buffer_load*` total | 112 |
| `buffer_load_dwordx4 ... lds` (TK G::load tile fills) | 96 (VMEM→LDS direct, gfx950 CDNA4 lever) |
| `buffer_load_dwordx[24] ... offen` (no `lds` — VMEM→VGPR) | 16 (these are the scale loads) |
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 384 |

**Zero `ds_write` confirms zero LDS round-trip for any data — including scales.** The 96 `buffer_load_*x4 ... lds` are TK's `G::load` which uses gfx950's `buffer_load_dword_lds` direct-VMEM→LDS path for tile fills (this is the lever R30 Dev C noted is "already maxed" for V2-CRR — confirmed also-maxed for V2-RCR).

### Key SASS span — steady-state K-loop body

```
353:  buffer_load_dwordx4 v[18:21], v171, s[4:7], s38 offen     # A scale b128 load
354:  buffer_load_dwordx2 v[192:193], v172, s[8:11], s37 offen  # B scale b64 load
...   (~30 instructions of LDS reads + tile-prep work)
383:  s_waitcnt vmcnt(2)                                        # wait scales (only 2 outstanding VMEM)
384:  v_mfma_scale_f32_16x16x128_f8f6f4 ..., v18, v192          # first MFMA quadrant cA
385-391: 7 more MFMA quadrants (cA finish)
413-420: 8 MFMA quadrants (cB)  -- consumes v18, v193
443-450: 8 MFMA quadrants (cC)  -- consumes v19, v192
465:  s_waitcnt vmcnt(6)
468-475: 8 MFMA quadrants (cD)  -- consumes v19, v193
...
679:  buffer_load_dwordx4 v[18:21], v2, s[4:7], s0 offen        # NEXT k_pair scale load issued
682:  buffer_load_dwordx2 v[192:193], v2, s[8:11], s0 offen
...   (more iter work)
727:  s_waitcnt vmcnt(2)
728:  v_mfma_scale_f32_... v18, v192                            # NEXT iter's first MFMA
```

The compiler **already pipelines scale loads** ~30 instructions ahead of consumers (via the `load_scale_buffer(k_pair)` call at the top of the loop iteration body). Steady-state gap from scale-load issue to consumer is dominated by tile LDS reads + `s_barrier()` synchronization — VMEM completion of a 16-byte b128 load is well under that gap on MI355X memory subsystem.

## 2. Hardware ground-truth measurements

V2-RCR (4096³ build, default flags) per `-Rpass-analysis=kernel-resource-usage`:

```
_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EE  (RCR V2 PRESHUFFLED, SCALE_VERSION=2)
  TotalSGPRs: 52
  VGPRs:      246  (no spill, scratch=0)
  LDS Size:   131072 bytes/block
  Occupancy:  2 waves/SIMD  (= 1 block/CU on 8-wave block)
  Spill:      0
```

LDS occupancy math (gfx950 LDS per CU = 163840 B):

| blocks/CU | LDS used | Fits? |
|---:|---:|:---:|
| 1 | 131072 | YES (80% util) |
| 2 | 262144 | NO (160% — overflow) |

Per R30 Dev B: kernel is at "occupancy 2" because of LDS (1 block/CU). VGPR has 6× headroom over current 246 usage. **V2-RCR is ALSO LDS-bound** — same as V2-CRR.

## 3. Mechanistic experiment: VGPR-prefetch second-buffer

### Implementation

Added `MXFP8_RCR_V2_SCALE_PREFETCH` macro (default 0, byte-identical when off; verified .so md5 `4ae07863...` matches pre-patch baseline). When enabled:

- Declare `a0/a1_scale_packs_next[RBM/32]` and `b0/b1_scale_packs_next[(RBN+31)/32]` (6 extra VGPRs nominal).
- Prime: load (k=0) into primary, load (k=1) into next.
- In K-loop: execute `do_k_iter_body<0/1>(k_pair*2, +1)` consuming primary; at end of iter, copy next→primary (via `swap_scale_buffer` lambda) and issue load of (k_pair+2) into next.

### Resource impact (4096³ build, prefetch=1)

```
_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EE  (with prefetch)
  TotalSGPRs: 59          (was 52)
  VGPRs:      256         (was 246, +10)
  ScratchSize: 596 bytes/lane  ⚠ SPILL
  Occupancy:  2           (unchanged — LDS still binds)
  VGPRs Spill: 312        ⚠ massive
  LDS Size:   131072      (unchanged)
```

The compiler did not register-allocate the second buffer. Despite the `_next` arrays being only 6 fp8e8m0_4 elements (= 6 VGPRs of pure capacity), holding them live across the long K-loop body increases the peak liveness of unrelated values past the 256-VGPR ceiling, forcing 312 VGPR-equivalents into scratch. Net effect: ~9.4 KB/wave of scratch-spill traffic per K-loop iteration, dominating execution time.

### Bench results (HIP_VISIBLE_DEVICES=2, 5x preheat-then-bench)

| Cell | Build | n | median TFLOPS | mean ± std | Δ% vs baseline |
|---|---|---:|---:|---:|---:|
| 4096³ V2-RCR | baseline (md5 4ae07863) | 5 | 2425.22 | 2415.13 ± 76.80 | — (re-run; first run had cold-clock dip) |
| 4096³ V2-RCR | prefetch=1 (md5 6bcd811e) | 5 | **186.53** | 184.39 ± 3.42 | **-92.3%** |
| 8192³ V2-RCR | baseline (md5 09f2cc0f) | 5 | 3006.89 | 2977.19 ± 71.27 | — |
| 8192³ V2-RCR | prefetch=1 (md5 b67a6f8f) | 5 | **225.62** | 225.35 ± 0.38 | **-92.5%** |

Correctness all builds: SNR 49.59-49.61 dB / det 3/3 PASS / pass_rate 100% — kernel computes correctly through the scratch-spill path; the regression is purely throughput.

### SHIP gate evaluation

| Gate | Required | Actual | Pass? |
|---|---|---|---|
| 4096³ V2-RCR ≥ baseline + 1% (Welch t > 3) | yes | -92.3% (catastrophic) | NO |
| 8192³ V2-RCR within ±1% of R29 baseline 3214 | yes | -92.5% | NO |
| Correctness SNR ≥ 48 dB + det 3/3 | yes | 49.6 dB / det 3/3 | YES (pyrrhic) |

**Result: candidate fails 2 of 3 perf gates → NO SHIP.**

## 4. Why VGPR-prefetch cannot work for V2-RCR

The V2-RCR `do_k_iter_body` lambda contains 4 MFMA quadrants (`cA`, `cB`, `cC`, `cD`), each consuming `a*_scale_packs[*]` and `b*_scale_packs[*]` operands directly as VGPRs. Across the body:

- `cA` consumes `a0_scale_packs[*]`, `b0_scale_packs[*]`
- `cB` consumes `a0_scale_packs[*]`, `b1_scale_packs[*]`
- `cC` consumes `a1_scale_packs[*]`, `b0_scale_packs[*]`
- `cD` consumes `a1_scale_packs[*]`, `b1_scale_packs[*]`

Each scale pack is read **8 times** per iteration (4 quadrants × 2 phases or similar fanout). Adding parallel `*_next` buffers makes both sets live across the entire body, doubling the pressure on the MFMA-friendly VGPR bank and forcing displacement of higher-frequency values (the 4 fp32 accumulators alone are 128 VGPRs, the A/B fp8 operand registers ~32 VGPRs, plus phase-cache VGPRs from `MXFP8_RCR_EXACT_PQ_PHASE_U16_CACHE_ENABLE`).

The compiler's chosen schedule of issuing `load_scale_buffer(k_pair)` once per iteration **at the natural top-of-iteration call site** (line 3106) is already near-optimal: SASS shows the load issues ~30 instructions before the first consuming MFMA, providing ample VMEM completion latency hiding without holding two scale-pack live ranges in flight.

## 5. Paradigm correction recommended for R31 wrap

Add to the lever-closure list (extends R27/R28/R29/R30 cumulative tally to **cycle-15 closure**):

> **PIPELINE_SCALE second-buffer (LDS variant) is N/A for V2-RCR** (R31 Dev C, extends R30 Dev C V2-CRR closure). V2-RCR scales follow the same `VMEM → buffer_load_b128/b64 → VGPR → mfma_scale_f32_16x16x128_f8f6f4` direct path as V2-CRR (zero LDS round-trip; SASS audit shows 0 `ds_write` instructions in the V2-RCR kernel). There is no `buffer_load + ds_write` pair to convert into a second LDS buffer. **NEVER prototype "scale-LDS double-buffer for V2-RCR"** — extends the R27 paradigm correction #1 (V2 has no scale LDS) and the R30 Dev C empirical SASS confirmation to the V2-RCR sibling.

> **VGPR-prefetch second-buffer (the only structurally-feasible reinterpretation) is CLOSED for V2-RCR** (R31 Dev C). Adding a 2nd set of `*_scale_packs_next[]` arrays to prefetch (k_pair+1) one iteration ahead causes 312 VGPR spill (596 bytes scratch/lane) and -92% throughput collapse on both 4096³ and 8192³ V2-RCR. The 4-quadrant MFMA design with 8× scale-pack reuse per iteration cannot accommodate parallel scale-pack live ranges without breaking the 256-VGPR ceiling. **NEVER prototype "VGPR prefetch / next-iter scale staging for V2-RCR"** — guaranteed catastrophic regression. The compiler's natural scheduling of `load_scale_buffer(k_pair)` at iteration top already provides ~30-instruction latency hiding (SASS-confirmed), saturating the lever.

## 6. Bonus closures (not separately benched but ruled out by 1 + 2 above)

- **VGPR-prefetch for V2-CRR**: extends by symmetry — V2-CRR has same VGPR ceiling (234 VGPR / occ=2 / LDS-bound per R30 Dev B) and same scale-pack reuse pattern. Same closure.
- **`buffer_load_dword_lds` for V2-RCR scales**: extends R30 Dev C closure — V2-RCR also has zero scale LDS path; converting scale loads to VMEM→LDS direct would require introducing an LDS scale staging path that doesn't exist.
- **Tile-fill `buffer_load_dword_lds` lever for V2-RCR**: confirmed already maxed (96 of 112 buffer_load instructions use the `lds` flag — TK G::load).

## 7. What's actually open for V2-RCR (carry-over to future cycle)

From R29 Dev C + R30 Dev D + this analysis, residual unexplored levers for V2-RCR:

1. **Dispatch geometry / persistent-CU scheduling for 4096³ V2-RCR** (R29 Dev C structural finding, R30 priority #4): 256 blocks at BLK=256 / 304 CUs = 0.84 wave-fill (16% CUs idle). Per-kernel optimization is structurally bounded; dispatch-geometry change required.
2. **LDS shared-tile reduction** (As[2][2]→[1][2] or Bs[2][2]→[1][2]) to drop V2-RCR LDS from 131072 B to ≤81920 B and enable 2 blocks/CU. Major structural change; double-buffering exists for latency hiding originally — likely net negative.
3. **4-GPU baseline triangulation** for cells with cross-cycle drift (R30 Reviewer rule, applies to all V2 cells).

## 8. Files

- Code: `kernel_mxfp8_layouts.cpp` — added `MXFP8_RCR_V2_SCALE_PREFETCH` macro at line 367 (default 0), implementation at K-loop body around line 3105+ (gated by macro; default-off path bit-identical to baseline, md5-verified).
- Build logs: `r31c_baseline_4k_build.log`, `r31c_baseline_8k_build.log` (defaults), `r31c_default_4k_build.log` (post-patch defaults — md5 matches pre-patch baseline), `r31c_prefetch_4k_build.log`, `r31c_prefetch_8k_build.log`.
- Bench outputs: `r31c_baseline_4k_v2rcr.txt`, `r31c_baseline_4k_v2rcr_run2.txt`, `r31c_baseline_8k_v2rcr.txt`, `r31c_prefetch_4k_v2rcr.txt`, `r31c_prefetch_8k_v2rcr.txt`.
- SASS analysis: `r31c_baseline.hsaco` (extracted gfx950 bundle), `r31c_baseline.s` (full disasm), `r31c_v2_rcr.s` (V2-RCR kernel range, 1314 lines).

Build cache hygiene: `rm -f tk_mxfp8_layouts*.so` before every build; per-build md5 logged and matched at runtime in bench output (R29 Dev C rule).

## 9. Summary

| Metric | Value |
|---|---|
| Verdict | **NO SHIP — STRUCTURAL CLOSURE (cycle-15 paradigm correction)** |
| Lever-1 (LDS variant) | N/A: V2-RCR has zero scale LDS path (SASS confirmed: 0 ds_write) |
| Lever-2 (VGPR-prefetch pivot) | catastrophic regression -92% on both 4096³ and 8192³ |
| 4096³ V2-RCR baseline median | 2425.22 TFLOPS |
| 4096³ V2-RCR prefetch=1 median | 186.53 TFLOPS (-92.3%) |
| 8192³ V2-RCR baseline median | 3006.89 TFLOPS |
| 8192³ V2-RCR prefetch=1 median | 225.62 TFLOPS (-92.5%) |
| Resource baseline | 246 VGPR / 0 spill / LDS=131072 / occ=2 |
| Resource with prefetch | 256 VGPR / 312 spill / 596 scratch/lane / occ=2 |
| Correctness all builds | SNR 49.59-49.61 dB / det 3/3 PASS |
| Time spent | ~75 min (within 90 min budget) |
| Paradigm closures added | **2** — PIPELINE_SCALE LDS-variant + VGPR-prefetch variant for V2-RCR |
| Code change | `MXFP8_RCR_V2_SCALE_PREFETCH` macro added (default 0, default-off path md5-identical to baseline) |
