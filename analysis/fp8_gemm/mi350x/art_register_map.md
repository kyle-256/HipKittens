# ART MXFP4 GEMM Register Map

## Architecture: 256×256 tile, 4 warps (2×2), 64×64 per warp

## Key design: Swap MFMA operands for row-oriented output
- MFMA "A" operand ← B-matrix data (weights, from LDS)
- MFMA "B" operand ← A-matrix data (activations, direct from global)
- Result: D[i] at lane t = C[t%16, 4*(t/16)+i] → 4 consecutive COLUMNS per thread
- This enables v_permlane16_swap + buffer_store_dwordx4 for vectorized store

## MFMA instruction: v_mfma_scale_f32_16x16x128_f8f6f4
- A operand: 4 dwords (FP4, cbsz:4)
- B operand: 4 dwords (FP4, blgp:4)
- D/C operand: 4 dwords (f32, accumulator)
- Scale A, Scale B: 1 dword each (e8m0)

## Tile decomposition (per warp)
- Output: 64 rows × 64 cols → 4×4 = 16 MFMA tiles of 16×16
- B tile (MFMA A operand): 64 rows × BK cols (from LDS, one B-half)
- A tile (MFMA B operand): 64 rows × BK cols (direct from global, one A-half)
- With A0/A1 and Bl/Br split: 4 quadrants × 32 MFMAs = 128 MFMAs per K-step

## AGPR allocation (a[0:255] = registers 256-511)
```
a[0:63]    = Accumulator block 0 (A0×Bl): 16 tiles × 4 regs = 64 AGPRs
a[64:127]  = Accumulator block 1 (A0×Br): 16 tiles × 4 regs = 64 AGPRs
a[128:191] = Accumulator block 2 (A1×Bl): 16 tiles × 4 regs = 64 AGPRs
a[192:255] = Accumulator block 3 (A1×Br): 16 tiles × 4 regs = 64 AGPRs
Total: 256 AGPRs ✓
```

## VGPR allocation (v[0:255])
```
v[0:28]    = Compiler reserved (addresses, loop counters, SRDs, scalars)  [29 VGPRs]
v[29]      = Output scale factor                                          [1 VGPR]
v[30:61]   = B tile current (MFMA A operand, from LDS, Bl or Br)          [32 VGPRs]
             = 4 subtiles × 8 ints = 32 VGPRs (fp4_intx8_t[4])
v[62:93]   = A tile current (MFMA B operand, direct from global, A0 or A1) [32 VGPRs]
             = 4 subtiles × 8 ints = 32 VGPRs (fp4_intx8_t[4])
v[94:125]  = Next B tile (Br or Bl[nxt], loaded from LDS)                 [32 VGPRs]
             Aliased: during Steps 1-2 → Br data; Steps 3-4 → Bl[nxt] data
v[126:157] = Next A tile (A1 or A0[nxt], loaded from global)              [32 VGPRs]
             Aliased: during Steps 1-2 → A1 data; Steps 3-4 → A0[nxt] data
v[158:165] = B scales (4 scale values × 2 = 8 VGPRs)                     [8 VGPRs]
v[166:173] = A scales (4 scale values × 2 = 8 VGPRs)                     [8 VGPRs]
v[174:177] = B LDS base addresses (p0/p1 × 2 double-buffer slots)         [4 VGPRs]
v[178]     = A voffset base (per-thread, invariant)                       [1 VGPR]
v[179:186] = Store: permlane swap temporaries + output addresses          [8 VGPRs]
v[187:254] = Store: v_cvt_pk_bf16_f32 + buffer_store_dwordx4 output      [68 VGPRs]
             (reuses A/B tile VGPRs after compute loop ends)
v[255]     = Spare                                                        [1 VGPR]
Total: 256 VGPRs ✓
```

## Loop structure
```
for bt = 0..k_byte_iters-1:
    Step 1: Bl×A0 (32 MFMAs) + ds_read Br + buffer_load A1
    Step 2: Br×A0 (32 MFMAs) (pure compute, using data loaded in Step 1)
    Step 3: Bl×A1 (32 MFMAs) + ds_read Bl[nxt] + B LDS prefetches
    Step 4: Br×A1 (32 MFMAs) + buffer_load A0[nxt]
    barrier (for B LDS double-buffer)
    swap: B_cur ← B_nxt, A_cur ← A_nxt
```

## Store epilogue (using swapped MFMA output = row-oriented)
For each accumulator block:
1. v_accvgpr_read_b32: read 8 f32 from two MFMA tiles (same output row)
2. v_mul_f32: apply output scale
3. v_cvt_pk_bf16_f32: pack f32 pairs → bf16x2 (4 dwords)
4. v_permlane16_swap_b32: swap between lanes 0-15 and 16-31 (2 ops)
5. buffer_store_dwordx4: write 8 bf16 (16 bytes) contiguous in row-major
