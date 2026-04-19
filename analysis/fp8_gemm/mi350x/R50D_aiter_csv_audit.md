# R50 Opt D — aiter `f4gemm_bf16_per1x32Fp4` audit

## Source files inspected
- `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv`
- `/shared_nfs/kyle/test/aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu`
- `/shared_nfs/kyle/test/aiter/csrc/include/aiter_hip_common.h`
- `/shared_nfs/kyle/test/aiter/aiter/ops/gemm_op_a4w4.py`
- `/shared_nfs/kyle/test/aiter/op_tests/test_gemm_a4w4.py`

## Kernel selection (CSV row for our target)
```
tile_M, tile_N, splitK, bpreshuffle, knl_name, co_name
256,    256,    1,      1,           _ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E,
                                     f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co
```

- `tile_M=256, tile_N=256` → matches HipKittens 256×256 tile geometry.
- `splitK=1` → kernel exposes the splitK knob (gdz can be >1 if we set
  `log2_k_split`). For R50D we leave `log2_k_split=0` (no split, gdz=1) —
  same as the HipKittens production path.
- `bpreshuffle=1` → kernel REQUIRES B preshuffle.
- Symbol verified: `R50D_aiter_symbols.txt` shows
  `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E` at offset 0x2c00,
  type FUNC GLOBAL PROTECTED.

## Kernel ABI (KernelArgs struct, from `asm_gemm_a4w4.cu`)
```c
struct __attribute__((packed)) KernelArgs {
    void* ptr_D;          p2 _p0;       // 16B    (output bf16 [M_pad32, N])
    void* ptr_C;          p2 _p1;       // 16B    (bias; nullptr OK)
    void* ptr_A;          p2 _p2;       // 16B    (A: [M, K/2] fp4x2)
    void* ptr_B;          p2 _p3;       // 16B    (B: [N, K/2] fp4x2 PRESHUFFLED)
    float alpha;          p3 _p4;       // 16B
    float beta;           p3 _p5;       // 16B
    uint  stride_D0;      p3 _p6;       // 16B    each stride is row-major leading dim
    uint  stride_D1;      p3 _p7;       // 16B
    uint  stride_C0;      p3 _p8;       // 16B
    uint  stride_C1;      p3 _p9;       // 16B
    uint  stride_A0;      p3 _p10;      // 16B    A: stride * 2 (fp4x2)
    uint  stride_A1;      p3 _p11;
    uint  stride_B0;      p3 _p12;      // 16B    B: stride * 2 (fp4x2)
    uint  stride_B1;      p3 _p13;
    uint  M;              p3 _p14;      // 16B
    uint  N;              p3 _p15;      // 16B
    uint  K;              p3 _p16;      // 16B
    void* ptr_ScaleA;     p2 _p17;      // 16B    (e8m0 scales [M_pad, K/32])
    void* ptr_ScaleB;     p2 _p18;      // 16B    (e8m0 scales [N_pad, K/32])
    uint  stride_ScaleA0; p3 _p19;
    uint  stride_ScaleA1; p3 _p20;
    uint  stride_ScaleB0; p3 _p21;
    uint  stride_ScaleB1; p3 _p22;
    int   log2_k_split;                 // 4B    (0 = no splitK)
};
// total 372 bytes, aiter passes via HIP_LAUNCH_PARAM_BUFFER_POINTER
```

R50D shim mirrors this exactly (`static_assert(sizeof(KernelArgs)==372)` in `R50D_aiter_dlopen.cpp`).

## Launch parameters
From `asm_gemm_a4w4.cu` lines 282-293:
- `gdx = ceil(N / SUBN)`         where SUBN = `cfg.tile_N` = 256
- `gdy = ceil(M / SUBM)`         where SUBM = `cfg.tile_M` = 256
- `gdz = 1`                       (since `log2_k_split=0`)
- `bdx = 256` (4 wave64 threads)
- `bdy = 1, bdz = 1`
- `sharedMemBytes = 0` (asm kernel doesn't use external dynamic LDS allocation)

For our target `(M=4096, N=32768, K=28672)` with the 256×256 kernel:
- `gdx = 32768/256 = 128`
- `gdy = 4096/256  = 16`
- `gdz = 1`
- total threadgroups = 2048 (= 128*16)

## Layout requirements (load-bearing)
Mirroring `op_tests/test_gemm_a4w4.py` lines 95-104:

### Input A (left, activations) — `[M, K/2]` uint8
- Logical shape `(M, K)` of fp4 values.
- Physical packing: 2 fp4 per byte → `[M, K/2]` uint8.
- **NOT preshuffled** (per `aiter.gemm_a4w4` test code passes `x` directly).

### Input B (right, weights) — `[N, K/2]` uint8 PRESHUFFLED
- Logical shape `(N, K)` of fp4 values.
- Physical packing: `[N, K/2]` uint8.
- **PRESHUFFLED via** `aiter.ops.shuffle.shuffle_weight(w_packed, layout=(16, 16))`.
  - This is a tile-mode reshuffle that reorders weights into the layout
    aiter's f4gemm asm kernel expects.
  - Without this, output is garbage (HipKittens stores B directly without aiter's
    preshuffle).

### A_scale — `[M_pad, K/32]` uint8 (e8m0)
- Quantization granularity: per 1×32 (one scale per 32 contiguous K elements per row).
- M_pad = M padded up to multiples of `SCALE_GROUP_SIZE` (32 in our case the M is already 4096, divisible).
- **PRESHUFFLED via** `aiter.get_triton_quant(per_1x32)(x, shuffle=True)` which
  returns scales rearranged into aiter's per-row preshuffle.
- HipKittens uses a DIFFERENT scale preshuffle (the local `preshuffle()`
  function in `bench_all_42_R44_INTEG.py`) — INCOMPATIBLE.

### B_scale — `[N_pad, K/32]` uint8 (e8m0)
- Same quant rules as A_scale but on B (per-N scales).
- Padded to multiples of 32 along N.
- Also preshuffled via aiter's quant utility with `shuffle=True`.

### Output C — `[M_pad32, N]` bf16
- M_pad32 = `((M + 31) // 32) * 32`.
- Returned slice `[:M, :N]` is the logical output.

## Strides recorded in KernelArgs
Aiter only sets `*_0` (leading-dim) strides; `*_1` (column-stride) is left zero-init
(packed struct `memset` to zero, then only `_0` written). All inputs are row-major
contiguous so `stride_0 = ncols`.

For fp4x2 inputs aiter records `stride * 2` (per cu line 190-191) — meaning
`stride_A0 = K` (not `K/2`). The kernel internally treats this as the count
of fp4 values per row; the doubling accounts for the 2×fp4-per-byte packing.

## Verification chain for R50D shim
1. shim loads `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via `hipModuleLoad`.
2. Resolves `_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E` via `hipModuleGetFunction`.
3. Builds 372B `KernelArgs` packed struct identical to aiter cu.
4. Launches `(gdx=128, gdy=16, gdz=1)` × `(bdx=256, bdy=1, bdz=1)` with `sharedMemBytes=0`.
5. Python harness preps inputs via aiter's own quant + shuffle utilities to
   guarantee layout compatibility.

## Structural blockers (none found)
- `.co` symbol is GLOBAL PROTECTED → public, dlopen-able.
- KernelArgs struct is fully documented in aiter source.
- Launch parameters are deterministic from M, N, K, tile_M, tile_N.
- No external library link required (shim is self-contained).
