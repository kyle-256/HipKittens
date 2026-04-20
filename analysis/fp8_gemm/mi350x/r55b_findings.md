# R55 Dev B: 70B Down RCR P2 SALU hoist — REFUTED-EMPIRICAL-COMPILER-ALREADY-HOISTED

**Verdict line:** R55 Dev B: 70B Down RCR P2 SALU hoist — REFUTED-EMPIRICAL-COMPILER-ALREADY-HOISTED — compiler LICM has already hoisted the V2-RCR scale base addresses to the prologue and strength-reduced the per-iter `k_pair << N` arithmetic to two-instruction `s_addk_i32` strided accumulators; macro1 K-loop is byte-identical in SALU/MFMA counts to baseline (15 s_add + 13 s_addc + 3 s_addk_i32 + 128 v_mfma); only secondary effect is a 4-VGPR coalescing win (254 → 250) that does not change occupancy (waves/SIMD = 2 in both cases, both VGPR-class capped at 256/SIMD=>2-wave occupancy).

---

## Cell

| Item | Value |
|---|---|
| Shape | 70B Down RCR — M=4096 N=8192 K=28672 |
| Layout | RCR (V2 PRESHUFFLED-QUANT scale layout, SCALE_VERSION=2) |
| Kernel symbol | `_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals` |
| Source | `kernel_mxfp8_layouts.cpp:2415` |
| Hypothesis (R54 Dev B brief) | "scale-tensor base-address arithmetic (SALU) is recomputed inside the K-loop instead of hoisted to the prologue" |
| Lever | `MXFP8_RCR_SCALE_SALU_HOIST` (default 0; macro1 = manual prologue hoist) |

---

## Phase 0: ISA inspection of HEAD baseline

Script: `r55b_phase0_isa.sh` -> `r55b_results/isa/baseline_*.s`

### V2-RCR scale-pointer SALU pattern in K-loop body

Extracted K-loop body (763 lines) shows the following SALU profile per iteration:

| Op | Count |
|---|---|
| `s_add_*` (s_add_u32 / s_add_i32 / s_add_co_u32) | 15 |
| `s_addc_u32` | 13 |
| `s_addk_i32` | 3 |
| `v_mfma_*` | 128 |
| `s_lshl_*` | 0 |
| `s_mul_*` | 0 |
| `s_load_dword*` | (per-iter scale fetches; loop-invariant SRDs s[0:3], s[20:23] reused) |

Critical observations:

1. **Scale SRDs are loop-invariant.** The V2 per-wave-tile slab descriptors `s[0:3]` and `s[20:23]` are constructed in the prologue and not rebuilt inside the K-loop body. The compiler has correctly identified them as loop invariants (LICM).

2. **Strength reduction is already applied.** Where the source writes per-iteration offsets like `(k_pair << N)` against the slab base, the compiler emits a strided accumulator pattern using a 2-byte instruction:
   ```
       s_addk_i32 s39, 0x100   ; A scale stride
       s_addk_i32 s38, 0x400   ; B scale stride
       s_addk_i32 s27, 0x200   ; matrix offset stride
   ```
   This replaces a 5-cycle `s_lshl + s_add` pair with a single 1-instruction-issue `s_addk_i32` per scale-pointer, per iteration. There is no `s_lshl` or `s_mul` in the K-loop body at all — the compiler has already eliminated them.

3. **Pointer-add carry chain.** The 13 `s_addc_u32` instructions paired with 13 of the 15 `s_add` instructions form ~13 64-bit pointer additions (low + carry-propagating high half), all of which sit on the *minimum-cost* representation already.

4. **No SGPR spills.** Resource remarks: `TotalSGPRs: 72`, `SGPRs Spill: 0`, indicating the SALU register file is not under pressure.

### V2-RCR kernel resource summary (HEAD)

```
VGPRs:           254
TotalSGPRs:       72  (96 next_free)
SGPRs Spill:       0
Scratch (lane):    0
LDS (block):  131072  (= 128 KiB; CRR 139264 reserved separately)
Occupancy:    2 waves/SIMD
```

VGPR-class capped (>256 would round to 0-wave; 254 ≤ 256 still gives 2 waves/SIMD).

---

## Phase 1: macro implementation

Added `MXFP8_RCR_SCALE_SALU_HOIST` (default 0; byte-identical to HEAD). The =1 path manually hoists the scale-slab pointer arithmetic into the prologue and uses pre-built local pointer variables in place of any per-iter `<<` re-computation. (See `kernel_mxfp8_layouts.cpp:470-471` for the macro definition; the body uses standard `#if MXFP8_RCR_SCALE_SALU_HOIST` guards.)

---

## Phase 2: A/B ISA diff (=0 vs =1) on the V2-RCR kernel

Script: `r55b_phase2_isa.sh` -> `r55b_results/isa/macro1_*.s`

### K-loop body counts (V2-RCR, SCALE_VERSION=2, PRESHUFFLED_QUANT=true)

| Op | baseline (=0) | macro1 (=1) | Δ |
|---|---|---|---|
| `s_add_*` | 15 | 15 | **0** |
| `s_addc_u32` | 13 | 13 | **0** |
| `s_addk_i32` | 3 | 3 | **0** |
| `v_mfma_*` | 128 | 128 | **0** |
| Stride values present | `0x100`, `0x400`, `0x200` | `0x100`, `0x400`, `0x200` | identical |
| K-loop body length | 763 lines | 762 lines | -1 line (cosmetic, no insn) |

### Resource diff

| Metric | baseline | macro1 | Δ |
|---|---|---|---|
| VGPRs (`amdhsa_next_free_vgpr`, `num_vgpr`) | 254 | 250 | **-4** |
| TotalSGPRs (next_free) | 96 | 96 | 0 |
| LDS / block | 131072 | 131072 | 0 |
| Scratch / lane | 0 | 0 | 0 |
| Occupancy waves/SIMD | 2 | 2 | **0 (no change)** |

The 4-VGPR savings is real but does not move occupancy: 254 and 250 both round to 2 waves/SIMD on gfx950 (next-occupancy bucket is at ≤128 VGPR, which would require a structural restructure shown to fail in R49A/R53A/R53B per V2 RRR VGPR ceiling memory).

---

## Diagnosis

**The compiler has already done what the brief asked us to do.** The V2-RCR scale-base address arithmetic is hoisted out of the K-loop body by LICM in the prologue, and the only per-iter scale-pointer math is a 3-instruction strided-accumulator update (`s_addk_i32 sX, imm`) — which is the *minimum-cost* representation possible on gfx950's SALU. Manual source-level hoist via `MXFP8_RCR_SCALE_SALU_HOIST=1` produces byte-identical SALU+MFMA counts in the K-loop body; the compiler simply re-derives the same hoisted form regardless of whether the source presents the loop-invariant pointer math inside or outside the K-loop.

The R54 Dev B PMC observation that gave rise to the hypothesis (DIAGNOSTIC-SCALE-FETCH-WAIT bottleneck class on 70B Down RCR) is therefore **not caused by SALU recomputation**. The PMC signature must have a different physical root cause — most likely scale-VMEM fetch latency / L2 residency / `s_load_dword` issue spacing rather than scale base-address arithmetic. Other R55 dev branches (Dev A: P1 scale-fetch interleave; Dev D: P4 scale L2 residency) are positioned to test exactly those alternative root causes.

---

## Bench note (NOT REQUIRED FOR VERDICT)

The brief authorizes verdict from ISA evidence alone when LICM diff is zero:

> "If the compiler already hoists this (LICM), the diff will be zero and you can declare REFUTED-EMPIRICAL-COMPILER-ALREADY-HOISTED."

This condition is met. Bench was attempted but blocked by a separate, cwd-dependent V1-LEGACY-FALLBACK dispatcher issue in `r55b_workspace/` (the byte-identical `tk_mxfp8_layouts.so` produced 2317 TFLOPS from `/tmp/r55b_cleanroom` but 0.89 TFLOPS from `r55b_workspace` with a `V1-LEGACY-FALLBACK (no V2 predicate matched)` trace; debug fprintf on the predicate did not fire from `r55b_workspace` despite the dispatcher logging the fallback message from the same file). Root cause unknown; investigated extensively (stale .inc, stale builds/, HIP cache, comgr cache, NFS — none were the cause). Since SALU/MFMA counts are identical between =0 and =1 in K-loop ISA, even a successful bench could not show a positive delta from this lever; the ISA refutation is dispositive and bench would only confirm the null hypothesis at higher cost.

---

## Source change to ship

**None** for the K-loop. `MXFP8_RCR_SCALE_SALU_HOIST` is defined with default=0 in `kernel_mxfp8_layouts.cpp:470-471`, which keeps the kernel byte-identical to HEAD. The =1 implementation is preserved (gated, default-off) for reproducibility and for any future investigation that wants to re-verify the LICM equivalence under different compiler versions.

The 4-VGPR coalescing win at =1 is real but does not move occupancy and does not justify shipping a `=1` default that would have to be re-validated across all 4 RCR cells.

---

## Followups for the parent cycle

- **The scale-fetch bottleneck is not SALU.** R54 Dev B's DIAGNOSTIC-SCALE-FETCH-WAIT class needs a different physical mechanism. Strongest candidates:
  - VMEM scale-fetch latency / L2 residency (R55 Dev D in flight)
  - `s_load_dword` issue-window stall / scale-fetch interleave with MFMA (R55 Dev A in flight)
  - VALU dependency-break for the scale broadcast (R55 Dev C in flight)
- **VGPR coalescing under macro1 (-4 VGPR) is not actionable here**, but the diff is documented in `r55b_results/isa/baseline_device.s` vs `macro1_device.s` for any future structural redesign that wants to claim those 4 registers.

---

## Artifacts

- `r55b_phase0_isa.sh` — Phase 0 baseline ISA generator
- `r55b_phase2_isa.sh` — Phase 2 macro=0 vs macro=1 ISA differ
- `r55b_results/isa/baseline_device.s` — full baseline device-side ISA
- `r55b_results/isa/baseline_rcr_v2_kernel.s` — V2-RCR kernel symbol extract (HEAD)
- `r55b_results/isa/baseline_rcr_v2_kloop.s` — V2-RCR K-loop body extract (HEAD)
- `r55b_results/isa/macro1_device.s` — full macro=1 device-side ISA
- `r55b_results/isa/macro1_rcr_v2_kernel.s` — V2-RCR kernel symbol extract (macro=1)
- `r55b_results/isa/macro1_rcr_v2_kloop.s` — V2-RCR K-loop body extract (macro=1)
- `r55b_results/isa/baseline_device_remarks.log` — `-Rpass-analysis=kernel-resource-usage` remarks
- `r55b_results/isa/macro1_device_remarks.log` — same, macro=1 build
