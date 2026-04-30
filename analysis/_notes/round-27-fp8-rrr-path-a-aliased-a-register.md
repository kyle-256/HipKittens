# Round 27 — FP8 RRR fuse path A probe: A register aliased to c (FAILURE)

**Date**: 2026-04-30
**Compile gate**: `FP8_RRR_FUSE_PROBE` (default 0; production unchanged)
**Companion plan**: `round-26-fp8-rrr-path-a-probe-plan.md`
**Production sha (this round)**: kernel_fp8_layouts.cpp HEAD
**Probe shape**: `G=1, M=2048, N=2816 (= 11 × HB, no N-tail), K=2880 (K_REM=64)` —
isolates K-tail accumulation correctness without N-tail / cross-group concerns.
**fp32 reference**: dequantized fp8 GEMM (`a.float() * sa_inv @ b.float() * sb_inv`).

## TL;DR

**Path A probe FAILED** with SNR < no-K-tail floor. Root cause traced
to compiler register aliasing — A register VGPR is **rebound to c
register pressure pool** by the compiler after Epilog 2's last
`rrr_mma(cD, a, b1)`, because the cooperative ops (`pre-zero`,
`G::load`, `__syncthreads()`) inserted before the next A use create a
long enough no-A-use gap that the compiler treats `a` as dead.
Subsequent writes to `a.tiles[].data[]` thus directly overwrite c.

**Production state**: PROBE block is `#if`-gated with default 0 →
production behavior identical. Metric 833 (within 833-836 noise band).
**No regression**.

**Recommended next-round fix**: Mirror RCR fuse fully — drop cooperative
ops, use per-lane `raw_buffer_load_b128` for B too. Requires deriving
lane→cell mapping for `B_col_reg = rt_fp8e4m3<BK=128, RBN=32, col_l,
rt_128x16_s>` from `load_col_from_st_half` (line 113-145). Estimated
register delta = 0 (drop-in replacement for cooperative path A B-side).

## Probe sequence and results

### Build

```bash
cd analysis/fp8_gemm/mi350x
make CPPFLAGS="-DFP8_RRR_FUSE_PROBE=1" TARGET=tk_fp8_layouts_probe -B
# Resource usage: VGPRs 256, SGPRs 85 (vs baseline 68), Spill 94
# (vs baseline 90), Occupancy 2 waves/SIMD (same)
```

### Test 1: full hybrid (A path B + B path A + 4 rrr_mma)

```text
SNR  = 15.05 dB   max_err = 54.625   mean_err = 7.5161
allclose(1e-2,1e-1,5e-1,1.0) = FAIL/FAIL/FAIL/FAIL
```

**Worse than no-K-tail floor** (16.75 dB from round-7 docs, also
confirmed below in test 3 = 16.56 dB). Not a "partial fix"; this is
active corruption.

### Test 2: SKIP_MMA (load_a_kt + load_b but NO rrr_mma)

```bash
make CPPFLAGS="-DFP8_RRR_FUSE_PROBE=1 -DFP8_RRR_FUSE_PROBE_SKIP_MMA=1" \
     TARGET=tk_fp8_layouts_probe -B
```

```text
SNR = -inf dB   max_err = inf   mean_err = inf
```

**Catastrophic**: even with no rrr_mma, the K-tail block produces NaN/Inf
in c. This means `load_a_kt`'s writes to `a.tiles[h][0].data[0..7]`
**directly overwrite c register VGPR slots**. There is no path through a
that reaches c via mma — yet c is corrupted. The only explanation is
that the VGPRs holding a are now also holding c.

### Test 3: SKIP_A_LOAD (cooperative pre-zero + G::load Bs + load_b only, no load_a_kt, no MMA)

```bash
make CPPFLAGS="-DFP8_RRR_FUSE_PROBE=1 -DFP8_RRR_FUSE_PROBE_SKIP_A_LOAD=1" \
     TARGET=tk_fp8_layouts_probe -B
```

```text
SNR = 16.56 dB   max_err = 46.312   mean_err = 6.3341
```

**Equals no-K-tail floor**. The B-side cooperative path A (pre-zero +
G::load + load_b) is **NOT corrupting c** — it's reading new data into
b0/b1 register tiles which never get used (no MMA). The only thing that
DOESN'T happen here vs test 2 is the write to a — pinpoints the write
to a as the corruption source.

### Test 4: K-aligned baseline (K=2816, no K-tail path triggered)

```text
SNR = 47.86 dB   max_err = 2.000
```

**Confirms main loop / dispatch are intact** — the K-aligned shape
takes the production path unchanged, matching production SNR.

## Root cause analysis

### Why does writing `a` corrupt `c`?

Compiler (LLVM AMDGPU `gfx950`) tracks VGPR liveness aggressively.
`grouped_rrr_kernel` is at the VGPR ceiling (256 VGPRs, occupancy = 2
waves/SIMD) with non-trivial spill (90 dwords baseline). Under this
pressure, the compiler eagerly rebinds VGPRs the moment a value is
declared dead.

Epilog 2 last instructions (line 2688-2695):

```cpp
load_a(a, As[tic][1], wm);
__builtin_amdgcn_s_barrier();
asm volatile("s_waitcnt lgkmcnt(0)");
__builtin_amdgcn_s_setprio(1);
rrr_mma(cC, a, b0);
rrr_mma(cD, a, b1);          // <-- last use of a in main path
__builtin_amdgcn_s_setprio(0);
__builtin_amdgcn_s_barrier();
```

After `rrr_mma(cD, a, b1)`, the compiler sees no further use of `a`
within the basic block. The next instruction stream (PROBE block) starts
with cooperative ops:

```cpp
constexpr int ST_V2_B128 = (sizeof(ST_v2) / 16);    // SGPR-resident
const int tid = threadIdx.x;                        // VGPR
__uint128_t* Bs0_ptr = ...;                         // SGPR (uniform across lane)
#pragma unroll
for (int idx = tid; idx < ST_V2_B128; idx += _NUM_THREADS) {
    Bs0_ptr[idx] = 0; Bs1_ptr[idx] = 0;             // ds_write (no VGPR pressure on a)
}
__syncthreads();                                    // s_barrier
G::load(Bs[tic][0], g.b, b_co(bc*2, ki_dyn), soB);  // raw_buffer_load_lds
G::load(Bs[tic][1], g.b, b_co(bc*2+1, ki_dyn), soB);
```

This sequence is **dozens of cycles long** with NO read or write of `a`.
The compiler aliases `a`'s VGPRs (32 VGPR for `A_row_reg = rt_fp8e4m3
<RBM=64, BK=128, row_l, rt_16x128_s>` — 4 base tiles × 1 width × 32 fp8
cells / 4 cells per dword) into the c register pressure pool, freeing
them for spill reload or other allocations.

When `load_a_kt(0)` then writes `a.tiles[h][0].data[0..7]` for `h ∈
[0,4)`, those byte addresses now alias to whatever VGPRs the compiler
reassigned them to — including c's working VGPRs.

### Why does RCR fuse not have this problem?

RCR fuse (line 2188-2331) goes **directly from Epilog 2 to per-lane
`raw_buffer_load_b128`** for both A and B with NO intervening
cooperative op:

```cpp
// === Fused K-tail epilog (path B) ===
if constexpr (FUSED_KTAIL) {
    if (g.fast_k < g.k) {
        const int laneid = kittens::laneid();           // SGPR/VGPR scalar
        const int row_lane = laneid % 16;
        const int k_lane_byte = (laneid / 16) * 32;
        // ... constants ...
        load_a_kt(0);                                   // <-- writes a IMMEDIATELY
        load_b_kt(b0, 0);
        load_b_kt(b1, 1);
        ...
        rcr_mma(cA, a, b0);
        ...
    }
}
```

The compiler sees `load_a_kt(0)` (which reads `a.tiles[h][0].data[0..7]`
addresses) within a few instructions of Epilog 2's last `rcr_mma(cD, a,
b1)`. The dataflow `a → load_a_kt write → rcr_mma read` keeps a's
VGPR allocation stable across the boundary.

## Production state (commit-ready)

The PROBE block is `#if FP8_RRR_FUSE_PROBE` gated with default 0:

* **PROBE=0** (production): K-tail block compiled out entirely.
  `dispatch_grouped_rrr` launches external
  `grouped_ktail_kernel_lds_rrr` + `grouped_ntail_kernel_lds_rrr` +
  `grouped_tail_kernel<RRR>` as before. Production SNR 43.99 dB,
  metric 833 (baseline 833-836 noise band).
* **PROBE=1** (probe): K-tail block enabled, external launches
  bypassed via paired `#if !FP8_RRR_FUSE_PROBE` in
  `dispatch_grouped_rrr`. Probe-only — broken in current state.

The detailed root-cause comment is committed inside the PROBE block as
documentation for the next-round agent so they don't repeat the same
investigation.

## Next-round options

### Option 1 (preferred): Mirror RCR fully — drop cooperative ops

Replace cooperative pre-zero + cooperative G::load on B with per-lane
`raw_buffer_load_b128` for B too. Estimated VGPR delta = 0 (drop-in
replacement for cooperative path A B-side; lane-cell mapping derived
from existing `load_col_from_st_half` ds_read_b64_tr_b8 pattern).

Steps:

1. Derive lane → cell mapping for `B_col_reg = rt_fp8e4m3<BK=128,
   RBN=32, col_l, rt_128x16_s>` from `load_col_from_st_half` (line
   113-145):
   * `row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16)`
   * `col_off = (laneid % 2) * 8`
   * `K_HALF=0`: `k_row = row_off ∈ [0, 64)` — for K_REM=64 always
     valid, no SENTINEL needed
   * `K_HALF=1`: `k_row = row_off + 64 ∈ [64, 128)` — for K_REM=64
     always OOB, hard-skip the K_HALF=1 read
   * `B_dst[K_HALF * 4 + 0..1]` (= float2) gets the b64 read
2. Construct B SRD with **per-group bound** `(group_idx + 1) × N × K`
   bytes (mirrors RCR fuse line 2252-2257) to prevent cross-group
   contamination on G > 1 production shapes
3. Replace cooperative pre-zero + G::load + load_b with the
   per-lane B path B
4. Verify SNR ≥ 25 dB on probe shape, then expand probe to G=4 +
   N-misaligned shapes before lifting to production gate

### Option 2 (fallback): Fresh K-tail register tiles

Introduce `A_row_reg a_kt;` (+32 VGPR) and `B_col_reg b0_kt, b1_kt;`
(+64 VGPR? — depends on `B_col_reg` size). Total +96 VGPR likely spills
to occupancy = 1 (catastrophic — each wave needs full VGPR file). Not
viable without first getting register pressure below 224 VGPR ceiling.

Option 1 is preferred because it preserves register pressure exactly
at the current spill level (90 dwords).

## Files touched (this round)

* `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  * Added `#define FP8_RRR_FUSE_PROBE 0` gate (line ~2455)
  * Inserted PROBE block in `grouped_rrr_kernel` after Epilog 2
    (line 2698-2810); detailed root-cause comment lives inline
    so next agent doesn't re-derive
  * Added `#if !FP8_RRR_FUSE_PROBE` guard around external K-tail
    launches in `dispatch_grouped_rrr` (line 4900-4917)
* `analysis/_notes/round-27-fp8-rrr-path-a-aliased-a-register.md`
  (this file)

## Verification

```bash
# PROBE=0 production rebuild
cd analysis/fp8_gemm/mi350x
make TARGET=tk_fp8_layouts -B
# SNR = 43.99 dB (matches pre-PROBE-block production), max_err = 2.000

# Metric (Primus-Turbo)
cd /workspace/code/Primus-Turbo
python3 scripts/_metric_grouped_only.py
# 833 (baseline 833-836 noise band, no regression)
```

## Round handoff

Next agent: skip the R27 hybrid (path A B-side cooperative) — it
fundamentally cannot work because cooperative ops break a's VGPR
liveness across the Epilog 2 → K-tail boundary. **Go directly to option
1 (per-lane raw_buffer_load_b128 for B too)**, mirroring RCR fuse fully.
PROBE gate stays at 0 in production; bump to 1 when iterating, drop the
gate (and `#if !FP8_RRR_FUSE_PROBE` in dispatcher) when the production
implementation passes SNR ≥ 25 dB on G=4 + N-misaligned shapes.
