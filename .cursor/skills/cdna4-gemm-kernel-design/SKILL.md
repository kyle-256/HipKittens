---
name: cdna4-gemm-kernel-design
description: Design and optimize GEMM kernels on AMD CDNA4 (gfx950/MI355X). Covers FP8/BF16 MFMA, LDS swizzling, double-buffering, 8-wave ping-pong scheduling, occupancy tuning, and global-to-LDS buffer loads. Use when writing or optimizing GEMM kernels for MI350X/MI355X, debugging LDS bank conflicts, choosing tile sizes, or understanding CDNA4 MFMA instruction scheduling.
---
# CDNA4 GEMM Kernel Design Guide

Based on AMD's official blog: "FP8 GEMM Optimization on AMD CDNA4 Architecture" (March 2026).

## CDNA4 vs CDNA3 Key Differences

| Feature | CDNA4 | CDNA3 |
|---|---|---|
| LDS capacity | **160 KB** per CU | 64 KB |
| LDS banks | **64** | 32 |
| LDS read BW | **256 B/clk** | 128 B/clk |
| GLOBAL_LOAD_LDS | **128 bits/lane** | 32 bits/lane |
| BF16 MFMA | 16x16x32, 32x32x16 (new) | 16x16x16, 32x32x8 |
| FP8 MFMA | 16x16x128, 32x32x64 | 16x16x128, 32x32x64 |
| FP4/FP6 | Supported | Not supported |

## Optimal Tile Configuration (256x256, 8-wave)

The best-performing kernel from AMD's analysis:

```
Output tile:     256x256
K-step:          128 (FP8) or 64 (BF16)
Threads/block:   512 (8 waves)
Waves/block:     8 (2x4 or 4x2 layout)
LDS:             2 buffers x (256x128) per operand
Occupancy:       2 waves/SIMD (target VGPRs ≤ 256)
```

### Performance benchmarks (FP8, MI355X)

| Kernel variant | M=N=K=4096 TFLOPS |
|---|---|
| Naive (1 thread = 1 element) | 1.15 |
| LDS tiling | 4.80 |
| + MFMA matrix-core | 30.05 |
| + Vectorized loads | 336.88 |
| + Direct global-to-LDS (buffer_load_lds) | 506.70 |
| + LDS swizzle + double-buffer | 1166.41 |
| + 8-wave ping-pong scheduling | **2288.16** |
| hipBLASLt reference | ~2750 |

### Tile comparison

| Config | Tile | Threads | Waves | TFLOPS (4096) |
|---|---|---|---|---|
| 128x128_t512 | 128x128 | 512 | 8 | 1828.74 |
| **256x256_t512** | **256x256** | **512** | **8** | **2288.16** |
| 256x256_t1024 | 256x256 | 1024 | 16 | 2228.01 |

256x256 with 512 threads (8 waves) is the sweet spot.

## Global-to-LDS Load (buffer_load_lds)

CDNA4's `GLOBAL_LOAD_LDS` transfers 128 bits/lane directly to LDS, bypassing VGPRs:

```cpp
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;

extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc, as3_uint32_ptr lds_ptr, int size,
    int voffset, int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");

// SRD (Scalar Resource Descriptor) setup
struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;  // 0x110000
};

__device__ inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes) {
    buffer_resource rsrc = {reinterpret_cast<uint64_t>(ptr), range_bytes, 0x110000};
    return *reinterpret_cast<i32x4*>(&rsrc);
}
```

Key: `vmcnt` tracks buffer_load_lds completion. `lgkmcnt` tracks `ds_read` (LDS→register).

## LDS Swizzle Pattern

For a 16x128 FP8 tile, the swizzle eliminates bank conflicts on `ds_read_b128`:

```cpp
int swizzle_col(int row, int col) {
    const int pair = (row >> 1) & 7;
    const int perm = pair ^ (((pair >> 1) ^ (pair >> 2)) & 1);
    const int mask = perm << 4;
    return col ^ mask;
}
```

XOR is self-inverse: same function for swizzle and un-swizzle. The swizzle redistributes lane accesses across LDS banks so each `ds_read_b128` phase is conflict-free.

## Double-Buffering (Ping-Pong LDS)

```
LDS: A_lds[2][2][128x128], B_lds[2][2][128x128]
     ↑ buffer  ↑ half-tile

Prologue: load tile 0 into buf[0], sync
Loop:
  load tile t+1 into buf[nxt] (async)
  compute tile t from buf[cur] (MFMA)
  wait + sync + swap buf
Epilogue: drain last 2 tiles
```

Overlapping load(t+1) with compute(t) hides global memory latency. With swizzle: +10% over non-swizzled double-buffer. With double-buffer: +2.3x over single-buffer.

## 8-Wave Ping-Pong Scheduling

Core technique from HipKittens. Two waves share one SIMD; they alternate between memory and compute:

```
SIMD 0:  Wave0(mem) | Wave4(mma) → Wave0(mma) | Wave4(mem) → ...
SIMD 1:  Wave1(mem) | Wave5(mma) → Wave1(mma) | Wave5(mem) → ...
SIMD 2:  Wave2(mem) | Wave6(mma) → Wave2(mma) | Wave6(mem) → ...
SIMD 3:  Wave3(mem) | Wave7(mma) → Wave3(mma) | Wave7(mem) → ...
```

### Key LLVM intrinsics

| Intrinsic | Purpose |
|---|---|
| `__builtin_amdgcn_s_barrier()` | Workgroup-level synchronization. Also creates execution distance between waves on same SIMD. |
| `__builtin_amdgcn_s_setprio(x)` | Set wave scheduling priority (0-3). Higher priority wave gets hardware resources first. |
| `__builtin_amdgcn_sched_barrier(0)` | Prevent compiler from reordering instructions across this point. |

### Wave stagger technique

```cpp
int wave_m = threadIdx.x / 64 / 4;  // 0 or 1

// All waves execute prologue...
if (wave_m == 1) {
    __builtin_amdgcn_s_barrier();  // waves 4-7 stall here
}
// waves 0-3 continue (memory ops)...
__builtin_amdgcn_s_barrier();      // waves 0-3 stall, release 4-7
// Now waves 4-7 do memory, waves 0-3 do compute
```

This creates the alternating pattern: one wave does `buffer_load_lds` + `ds_read` while its SIMD partner does MFMA.

### Main loop pseudo-code

```cpp
#pragma unroll 2  // controls register pressure
for (int k = 0; k < num_k_tiles - 2; k++) {
    // LDS reads (ds_read_b128)
    a_frag = lds_read(A_lds[cur]);
    b_frag = lds_read(B_lds[cur]);

    // Prefetch next tile (buffer_load_lds, async)
    buffer_load_lds(A_lds[nxt], A_global, k+2);
    buffer_load_lds(B_lds[nxt], B_global, k+2);

    // Sync: ensure LDS reads complete
    s_waitcnt lgkmcnt(0);
    s_barrier();

    // MFMA compute
    s_setprio(1);
    acc = mfma(acc, a_frag, b_frag);
    s_setprio(0);
    sched_barrier(0);

    // Sync: ensure MFMA consumed data before overwrite
    s_barrier();

    // Wait for prefetch + swap buffers
    s_waitcnt vmcnt(N);
    s_barrier();
    swap(cur, nxt);
}
// Epilogue: manually unroll last 2 iterations
```

## Critical Synchronization Rules

1. **`s_barrier` after MMA is mandatory**: protects LDS from being overwritten by `buffer_load_lds` while slow warps still have pending `ds_read` from previous barrier.
2. **`lgkmcnt` is per-warp**: only guarantees THIS warp's LDS reads complete. Need `s_barrier` to synchronize ALL warps.
3. **`vmcnt` tracks buffer_load_lds**: use `vmcnt(0)` to ensure all global→LDS transfers complete before reading LDS.
4. **Post-MMA barrier cannot be removed**: a fast warp reaching `buffer_load_lds` while a slow warp's `ds_read` is pending causes a data race on the same LDS address.

## VGPR Budget Guidelines

| VGPRs | Occupancy (waves/SIMD) | Recommendation |
|---|---|---|
| ≤ 170 | 3 | Ideal but often requires spilling |
| ≤ 256 | 2 | **Sweet spot for 8-wave kernel** |
| > 256 | 1 | Insufficient latency hiding for HLL |

`__launch_bounds__(512, 2)` = 512 threads, min 2 blocks/CU. Compiler targets ≤256 VGPRs.

Increasing to `__launch_bounds__(512, 3)` forces ≤170 VGPRs but causes ~280 VGPR spills → worse performance.

## hipBLASLt Reference (from rocprof)

hipBLASLt's BF16 kernel for 4096x28672x4096 uses:
```
MT256x256x64   = 256x256 tile, K_STEP=64
MI16x16x1      = 16x16 MFMA
WG32_8_1       = 256 threads (4 waves)
SK3            = Split-K = 3
```

Key difference from ThunderKittens: **4 waves** (not 8) with hand-tuned ISA scheduling at 1 wave/SIMD. HLL frameworks can't replicate this because 256x256 output / 4 waves = 256 VGPRs for accumulators alone (no room for anything else without spilling).
