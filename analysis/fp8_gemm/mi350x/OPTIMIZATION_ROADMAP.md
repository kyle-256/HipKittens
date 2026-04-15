# MXFP4 GEMM Optimization Roadmap

## Target
- **Shape**: 4096×32768×128256 (and all 42 LLaMA shapes)
- **aiter baseline**: 5653 TFLOPS on MI355X (ASM kernel)
- **Target**: 97% of aiter on ALL shapes
- **Current**: 22/42 shapes ≥97%, average 98.5% across all shapes
- **Peak**: 4926T on hardest shape (87.1% of aiter 5653T)
- **Best**: 5198T (159.8% of aiter) on 4096×128256×32768

## What's been tried and results

### C++ incremental optimizations (all ~4926T, structural limit)
- [x] GROUP_SIZE_M=8: 4926T (+1.2%)
- [x] XOR double-buffer toggle: 4917T (no improvement)
- [x] Vectorized store (LDS transpose): 4509T (WORSE — LDS overhead)
- [x] Compiler flags (-mllvm, early-inline): no improvement
- [x] UNROLL_K sweep (1,2,4,8,16): no improvement on large K
- [x] Direct-A C++ (23 VGPR spills): 2547T (spills kill perf)

### ART kernel (correct but slow)
- [x] ART V1: bit-exact, 0 spills, 2457T (separate asm blocks)
- [x] ART V4: monolithic Step1+2 and Step3+4 asm, 2855T
- [x] Key bugs found: compiler clobbers AGPRs in store, clobber ordering
- [x] MFMA FP4 macro added to HipKittens3

## Why 4926T is the C++ limit
1. **48 ds_reads per iteration** (aiter: 20) — A tiles go through LDS round-trip
2. **8 v_cndmask per iteration** (aiter: 0 VALU) — double-buffer select overhead
3. **~38 non-MFMA instructions per iteration** (aiter: ~33) — C++ overhead
4. **col_l MFMA output** → scalar bf16 stores (aiter: row-oriented → vectorized)

## Path to 5484T: Full HipKittens3 ART Rewrite

### Phase A: Swap MFMA operands for row-oriented output
- Feed B-matrix as MFMA "A" operand, A-matrix as MFMA "B" operand
- Each thread gets 4 consecutive COLUMNS of one ROW
- Enables v_permlane16_swap + buffer_store_dwordx4 (32× wide store vs 256× scalar)
- Saves ~5% on store-heavy shapes

### Phase B: Full hand-scheduled inner loop
Use HipKittens3 ART framework (`art<>` tiles, `mma_ABt<N,M,K>()`, `load<N,M>()`)
- Register map: see art_register_map.md
- All 128 MFMAs + 32 loads + scale loads + barrier in ONE hand-scheduled sequence
- Zero VALU in loop (no v_cndmask, no v_mov for scales)
- Direct-A loading via buffer_load_dwordx4 (no LDS for A)
- Reference: /shared_nfs/kyle/HipKittens3/kernels/attn/gqa_causal_backwards/attn_bkwd_causal.cpp

### Phase C: Instruction scheduling optimization  
- Profile with rocprofv2 (gfx950 supported)
- Tune MFMA-to-load interleave ratio
- Optimize scale loading (embed in MFMA asm blocks)
- Minimize barriers (target 1 per K-iteration)

### Phase D: 42-shape validation
- Run bench_all_42.py with the new kernel
- Auto-tune GROUP_SIZE_M per shape
- Verify all shapes ≥ 97% of aiter

## Key Files
| File | Status | Description |
|------|--------|-------------|
| kernel_mxfp4_gluon_cpp.cpp | Production (4926T) | Original C++ kernel |
| kernel_mxfp4_art.cpp | Working (2855T) | ART prototype with direct-A |
| kernel_mxfp4_direct_a.cpp | Working (2547T) | C++ direct-A (spills) |
| kernel_mxfp4_xor_toggle.cpp | Working (4917T) | XOR toggle (no improvement) |
| kernel_mxfp4_vecstore.cpp | Working (4509T) | LDS transpose store |
| art_register_map.md | Reference | VGPR/AGPR register allocation |
| /shared_nfs/kyle/HipKittens3/include/common/macros.cuh | Modified | Added MFMA FP4 macro |

## Critical Bugs to Remember
1. **Compiler clobbers AGPRs during store** → read all AGPRs in single asm block
2. **Clobber ordering** → clobbers must precede all writes to pinned VGPRs
3. **ds_read in Step3 can't overwrite Bl data** → put Bl[nxt] reads in Step4
4. **UNROLL_K>1 crashes with K=128256** → use UNROLL_K=1 for large K shapes
5. **amdgpu_num_vgpr(29) NOT a hard limit** → compiler may exceed in store epilogue

## Build Commands
```bash
cd /shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x
export THUNDERKITTENS_ROOT=/shared_nfs/kyle/test/HipKittens

# Original (production)
make -j TARGET=tk_mxfp4_gluon_cpp SRC=kernel_mxfp4_gluon_cpp.cpp \
  CPPFLAGS="-DK_DIM=128256 -DN_DIM=32768 -DGROUP_SIZE_M=8"

# ART kernel
make -j TARGET=tk_mxfp4_art SRC=kernel_mxfp4_art.cpp \
  CPPFLAGS="-DK_DIM=128256 -DN_DIM=32768 -DUNROLL_K=1"

# Benchmark
HIP_VISIBLE_DEVICES=N python3 bench_all_42.py
```

## Benchmark Rules (MANDATORY)
- warmup=200, iters=500, trimmed mean 10%
- HIP_VISIBLE_DEVICES=N on idle GPU (rocm-smi first)
- MI355X machine, gfx950
