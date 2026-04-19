# R51 Dev E — RRR B-side LDS bank conflict at N=14336 — REFUTED (analytical + ISA)

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ b35a5fce
**GPU:** MI355X (gfx950); diagnostics built with HIP_VISIBLE_DEVICES=1 (no GPU runtime needed — compile-only investigation)
**Scope:** R50 Dev B's §5 follow-up: investigate whether the production V2 RRR B-tile LDS swizzle (`st_16x128_v2`) produces an N=14336-specific bank-conflict pessimism that explains the 8B Gate/Up RRR HEADROOM cell (90.4% measured vs 93.5% predicted, +3.1pp gap — R48 Dev D §7.2). Last unattacked structural lever after R49B (noinline phase split — REFUTED), R50A (CRR scale lead — REFUTED), R50B (U-sweep — REFUTED), R50C (LDS boost — REFUTED).
**Verdict:** **REFUTED at Phase 1 (diagnostic-only).** The B-tile LDS swizzle is structurally invariant under N. Both analytical inspection of the swizzle formula AND ISA dump comparison across N=8192 / N=14336 / N=28672 prove zero N-dependence in the LDS access pattern. No bench was run; no kernel was modified. Lever closed.

---

## TL;DR

1. **Analytical:** the swizzle for `st_16x128_v2` (the production B-tile shared layout for RRR) is `swizzle = ((offset >> 7) & 7) << 4`, where `offset = sizeof(T) * (r*cols + c)` and `cols = 128` is a tile constant. **N (the global tensor dimension) does not appear in the swizzle.** Likewise `load_col_from_v2_st_half` computes the LDS read address from `tile_base`, `k_row` (laneid-derived), `col_off` (laneid-derived), `j` (per-iter), and `sw_k = (k_row & 7) << 4`. **N does not appear.**
2. **ISA evidence:** built MXFP8 device.s for three shapes — 8B Gate/Up (M=4096,N=14336,K=4096), 70B Gate/Up (M=4096,N=28672,K=8192), 70B Q/O (M=4096,N=8192,K=8192). The V2 RRR steady inner K-loop body is **441 lines in all three**, with byte-identical ds_read_b64_tr_b8 (×64), ds_read_b128 (×64), and buffer_load_dwordx4 lds (×16) sequences. The only difference inside the inner loop is **a single SGPR loop-counter compare** (`s_cmpk_eq_i32 s35, 0xf00` vs `0x1f00`) — pure loop control, not LDS arithmetic.
3. **R29 Dev D corroboration:** R29D's audit of the *V2-CRR* path already proved zero bank conflicts for the identical swizzle algebra (load helper `load_col_from_v2_st_half` + store-side `((offset>>7)&7)<<4`). Since RRR uses the same V2 / V2a load helpers and the same swizzle struct, the R29D conclusion transfers directly to RRR.
4. **No prototype.** Hypothesis explicitly required Phase 1 to gate Phase 2 on bank-conflict evidence. None found → REFUTE per the prompt's bail condition.

---

## 1. Method

### 1.1 Build commands (reproducible)

```bash
source /shared_nfs/kyle/test/Hipkittens2/env.src
cd /shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x

HIPCC=/opt/rocm/bin/hipcc
HIPFLAGS="-DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -I/opt/rocm/include/rocrand"
CPPF="-I${THUNDERKITTENS_ROOT}/include -I/opt/rocm/include/hip"
PYINC=$(python3 -m pybind11 --includes)

# 8B Gate/Up — N=14336, the suspect cell
$HIPCC kernel_mxfp8_layouts.cpp $HIPFLAGS -std=c++20 -w $CPPF \
  -DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096 $PYINC -shared -fPIC \
  --cuda-device-only -S -o r51e_isa_dumps/mxfp8_4096_14336_4096_device.s

# 70B Gate/Up — N=28672, known-good (R47A swizzle ON, at predicted ceiling)
$HIPCC kernel_mxfp8_layouts.cpp $HIPFLAGS -std=c++20 -w $CPPF \
  -DM_DIM=4096 -DN_DIM=28672 -DK_DIM=8192 $PYINC -shared -fPIC \
  --cuda-device-only -S -o r51e_isa_dumps/mxfp8_4096_28672_8192_device.s

# 70B Q/O — N=8192, known-good
$HIPCC kernel_mxfp8_layouts.cpp $HIPFLAGS -std=c++20 -w $CPPF \
  -DM_DIM=4096 -DN_DIM=8192 -DK_DIM=8192 $PYINC -shared -fPIC \
  --cuda-device-only -S -o r51e_isa_dumps/mxfp8_4096_8192_8192_device.s
```

All three compiled cleanly. Each .s ≈ 30700 lines.

### 1.2 V2 RRR kernel extraction

Symbol `_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals` (the production V2 RRR kernel — `PRESHUFFLED_QUANT=true, SCALE_VERSION=2`, the only path the runtime dispatches via `MXFP8_PRESHUFFLE_QUANT=1`).

| Cell | Kernel start–end | Body lines | num_vgpr | scratch |
|---|---|---:|---:|---:|
| 8B Gate/Up (N=14336) | 4692–6360 | 1669 | **254** | 0 |
| 70B Gate/Up (N=28672) | 4693–6361 | 1669 | **254** | 0 |
| 70B Q/O (N=8192) | 4691–6358 | 1668 | **254** | 0 |

Identical VGPR count + zero scratch across all three N values — already a strong signal that the kernel body structure is N-invariant.

### 1.3 Inner K-pair loop body extraction

Located via `; =>This Inner Loop Header: Depth=1` annotation on `.LBB3_7`, end at the first `s_cbranch …LBB3_7` backedge.

| Cell | LBB3_7 line | Backedge | Body lines |
|---|---:|---:|---:|
| N=14336 | 439 | 879 | **441** |
| N=28672 | 439 | 879 | **441** |
| N=8192  | 438 | 878 | **441** |

All three K-loop bodies are exactly 441 lines.

---

## 2. Analytical refutation: the V2 swizzle is N-independent

### 2.1 Store-side swizzle (in `include/types/shared/st_shape.cuh:267`)

```cpp
struct st_16x128_v2 {
    static constexpr int rows = 16;
    static constexpr int cols = 128;
    template<typename _T>
    __device__ __forceinline__ static const uint32_t swizzle (int2 coord) {
        const int r = coord.x, c = coord.y;
        const uint32_t offset = sizeof(T)*(r*cols + c);  // cols = 128 — TILE CONSTANT
        if constexpr (sizeof(T) == 1) {
            const int swizzle = ((offset >> 7) & 7) << 4;   // offset-of-3-bits-at-bit-7
            const int swizzled_offset = offset ^ swizzle;
            return swizzled_offset;
        }
    }
};
```

`st_16x128_v2a` (used when `RRR_USE_V2A_SWIZZLE=1`) has the **identical** swizzle (line 296). Both depend purely on the in-tile `(r, c)` coordinate; **N does not enter**.

### 2.2 Load-side address derivation (in `kernel_mxfp8_layouts.cpp:506-535`)

```cpp
const int laneid = kittens::laneid();
const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
const int col_off = (laneid % 2) * 8;
const uint32_t tile_base = reinterpret_cast<uintptr_t>(&tile.data[0]);
constexpr int idx = K_HALF * 4;
const int k_row = row_off + K_HALF * 64;
const uint32_t stidx  = k_row >> 4;
const uint32_t base_k = tile_base + (stidx << 11) + (stidx << 7) + ((k_row & 15) << 7);
const uint32_t sw_k   = (k_row & 7) << 4;
for (int j = 0; j < RT::width; j++) {
    const uint32_t nc   = col_start + j*16 + col_off;
    const uint32_t addr = base_k + (nc ^ sw_k);
    ds_read_b64_tr_b8 dst, addr offset:0
    ds_read_b64_tr_b8 dst, addr offset:1024
}
```

Free variables: `laneid`, `tile_base` (LDS allocation address — compile-time constant per CTA), `K_HALF` (template param 0/1), `col_start = wn * RBN = wn * 32` (RBN=32 in 8-wave), `j ∈ [0, RT::width=2)`. **N does not appear.**

### 2.3 Tile-fill (VMEM→LDS direct) address derivation

`include/ops/warp/memory/tile/global_to_shared.cuh:121-181` (`prefill_swizzled_offsets` + `load`). The LDS-side address is:

```cpp
uintptr_t lds_addr = lds_tile_base + warp_linear_offset + lds_subtile_id * subtile_padding;
```

Where `warp_linear_offset = warpid*bytes_per_warp + i*num_warps*bytes_per_warp` and `subtile_padding` is a tile constant. The N-stride enters only into the **global-side** voffset (the `swizzled_offsets[i]` value, which is the VMEM SRD voffset). **N does not affect LDS address arithmetic.**

### 2.4 R29 Dev D direct corroboration

`r29d_lds_bank_audit.md` (this directory) computed the per-cycle bank pattern for the identical `load_col_from_v2_st_half` helper as used in V2-CRR and proved **zero bank conflicts** across all 16-lane dispatch slots, both K_HALF values, all `j` iterations. RRR uses the same helper (`kernel_mxfp8_layouts.cpp:4693-4707`); the R29D conclusion transfers directly. Lines 7 and 169 of r29d:

> "No bank conflicts in the V2 column-load (B-tile) or V2a column-load (A-tile) paths. The existing swizzle (nc ^ sw_k) already distributes lane addresses across all 32 banks evenly within every 16-lane dispatch slot."
> "**Closure:** LDS bank conflicts are not contributing to V2-CRR under-performance. Lever closed."

Since the swizzle algebra and the load-helper are byte-identical, the same closure applies to V2-RRR for **every N value**.

---

## 3. ISA evidence: K-loop bodies byte-identical across N

### 3.1 Total LDS instructions per V2 RRR kernel

```
$ for N in 14336 28672 8192; do
    grep -c "ds_read_b64_tr_b8" rrr_v2_4096_${N}_*.s
    grep -c "ds_read_b128"      rrr_v2_4096_${N}_*.s
    grep -cE "buffer_load.*lds" rrr_v2_4096_${N}_*.s
  done
```

| N | ds_read_b64_tr_b8 (B-tile) | ds_read_b128 (A-tile) | buffer_load_dwordx4 lds (tile fills) |
|---|---:|---:|---:|
| 14336 | 64 | 64 | 16 (×… see kernel) |
| 28672 | 64 | 64 | 16 |
| 8192  | 64 | 64 | 16 |

Identical instruction counts.

### 3.2 Diff of B-tile ds_read sequences

```
$ diff <(grep ds_read_b64_tr_b8 rrr_v2_4096_14336_4096.s) \
       <(grep ds_read_b64_tr_b8 rrr_v2_4096_28672_8192.s)
(empty — zero diff)

$ diff <(grep ds_read_b64_tr_b8 rrr_v2_4096_14336_4096.s) \
       <(grep ds_read_b64_tr_b8 rrr_v2_4096_8192_8192.s)
(empty — zero diff)
```

Likewise for `ds_read_b128` (A-tile reads): zero diff across all three N values.

### 3.3 Diff of full inner K-loop body

```
$ diff kloop_4096_14336_4096.s kloop_4096_28672_8192.s
438c438
< 	s_cmpk_eq_i32 s35, 0xf00
---
> 	s_cmpk_eq_i32 s35, 0x1f00

$ diff kloop_4096_14336_4096.s kloop_4096_8192_8192.s
438c438
< 	s_cmpk_eq_i32 s35, 0xf00
---
> 	s_cmpk_eq_i32 s35, 0x1f00
```

**The only diff inside 441 lines of inner K-loop body is the loop-counter compare constant (`0xf00` = 3840 vs `0x1f00` = 7936)**, which encodes total K-iterations × loop-state encoding for K=4096 vs K=8192. This is loop-control, not LDS-related.

The kernel-prologue diffs (block-swizzle setup, `0x37f`/`0x6ff`/`0x1ff` = blocks-per-row constants etc.) are outside the K-loop body and concern only global-coord arithmetic.

---

## 4. Why the original hypothesis was geometrically infeasible

The R48D §7.2 conjecture was that "B-tile is row-major (K x N) and the swizzle modulo arithmetic may produce a degenerate stride at N=14336". This conflates two address spaces:

- **Global B tensor (K, N) in DRAM:** the row-major K-stride here is N elements. N=14336 is unaligned to 128 in interesting ways (14336 / 128 = 112; 14336 mod 256 = 0 actually). But this stride only affects **VMEM** access — bank-conflicts here would be L2/HBM, not LDS.
- **LDS B-tile (HB=128, BK=128) — the staging tile:** this is a fixed 16384-byte block per double-buffer slot, with a 128-byte row stride. The LDS-row-stride is **128 B**, not "N". The `((offset>>7)&7)<<4` swizzle distributes the 8 row-blocks of this 128×128 tile across 8 different `<<4` (16-byte) shifts — by R29D's per-cycle audit, this is conflict-free.

The N-of-the-global-tensor never reaches the LDS address calculation. The only N-dependent stride is the global voffset (`row_stride`), which feeds into VMEM SRD/buffer_load_dwordx4 instructions, not into LDS arithmetic.

---

## 5. Files

ISA dumps under `analysis/fp8_gemm/mi350x/r51e_isa_dumps/`:

| File | Contents |
|---|---|
| `mxfp8_4096_14336_4096_device.s` | Full MXFP8 device .s @ N=14336 (8B Gate/Up; 30714 lines) |
| `mxfp8_4096_28672_8192_device.s` | Full MXFP8 device .s @ N=28672 (70B Gate/Up; 30717 lines) |
| `mxfp8_4096_8192_8192_device.s`  | Full MXFP8 device .s @ N=8192 (70B Q/O; 30705 lines) |
| `rrr_v2_4096_14336_4096.s`       | Extracted V2 RRR kernel @ N=14336 (1669 lines) |
| `rrr_v2_4096_28672_8192.s`       | Extracted V2 RRR kernel @ N=28672 (1669 lines) |
| `rrr_v2_4096_8192_8192.s`        | Extracted V2 RRR kernel @ N=8192 (1668 lines) |
| `kloop_4096_14336_4096.s`        | Inner K-pair body @ N=14336 (441 lines) |
| `kloop_4096_28672_8192.s`        | Inner K-pair body @ N=28672 (441 lines) |
| `kloop_4096_8192_8192.s`         | Inner K-pair body @ N=8192 (441 lines) |
| `ds_reads_*.txt`                 | Just the ds_read_b64_tr_b8 lines, for diffing |
| `ds_b128_*.txt`                  | Just the ds_read_b128 lines, for diffing |

No kernel files modified. No bench scripts written. No GPU runs (this is a Phase-1-only diagnostic refutation per the prompt's explicit-bail condition).

---

## 6. Closure of the lever and downstream implications

The R48D §7.2 conjecture about "B-side LDS bank conflict for N=14336" is geometrically impossible in the current swizzle scheme: N never enters the LDS address. Together with the prior REFUTATIONS:

- R49 Dev B noinline phase split — REFUTED
- R50 Dev A CRR scale lead — REFUTED
- R50 Dev B U-sweep — REFUTED
- R50 Dev C LDS-boost — REFUTED
- **R51 Dev E B-side LDS swizzle — REFUTED (this work)**

…the 8B Gate/Up RRR HEADROOM cell is now exhausted of structural levers within the current kernel architecture. The remaining +3.1pp gap (90.4% measured vs 93.5% predicted) most likely reflects either:

(a) The predicted ceiling itself is too generous for K=4096 RRR (the K-tail / wave-tail effect at this CTA count interacts with the looped-vs-unrolled FP8 baseline asymmetry — R48D §3.3 already noted the K-tail amortization assumption is borderline).
(b) An overhead source not yet modeled in the static-instr-count framework (e.g., L2 line-replacement pressure from the fact that 8B Gate/Up is **the only RRR shape with M=4096, N=14336, K=4096** — small-K + large-N + small-CTAs may produce a unique L2-eviction pattern not present in K=8192 shapes). This would be a *system-level* bottleneck investigation, not a kernel-level lever.
(c) Measurement noise at the ±1pp level. The R48D ceiling has ±1pp uncertainty band; 90.4 vs 93.5 with both bands is 89.4–91.4 vs 92.5–94.5 — gap is 1.1–5.1pp range, with the lower bound below "interesting".

**Recommendation for R52+:** classify 8B Gate/Up RRR as **at-ceiling-within-uncertainty** alongside the cells already on the R48D "STOP list". Stop allocating dev-cycles to it. If anyone wants to chase the residual, the next investigation is a *system-level* L2 / SQ-WAVES-LIVE rocprof study, not a code-level swizzle / unroll / sched-barrier sweep.

---

## 7. One-line summary

**R48D §7.2 N=14336 B-side LDS bank-conflict hypothesis is geometrically impossible in the V2 swizzle (N never enters LDS address arithmetic) and is corroborated by zero diff on the inner K-loop body across N=8192 / N=14336 / N=28672 ISA dumps. Lever closed without prototype per the prompt's Phase-1 explicit-bail. 8B Gate/Up RRR HEADROOM cell is now exhausted of structural code-level levers; remaining +3.1pp gap likely reflects a too-generous ceiling-model or a system-level L2 effect, not a kernel-level conflict.**
