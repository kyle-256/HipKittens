# MLA D192/V128 Causal Attention Optimization Progress Report

## Target
- Hardware: AMD MI300X (gfx950 / CDNA4)
- Kernel: MLA attention with D_QK=192, D_V=128, causal mask
- Goal: Forward + Backward both at 1200-1300 TFLOPS

## Forward Kernel Status: ~993 TFLOPS ✓ (was ~780T)

### Current: `kernel_d192v128.cpp`
- **Performance**: 993 TFLOPS (AITER reference: ~1096T)
- **Correctness**: O cos≈0.9999, LSE cos=1.0
- **Config**: KV_BLOCK=32, NUM_WARPS=8, occupancy=2, VGPR=214

### Key optimizations applied:
1. Switched from 4 warps to **8 warps** (biggest gain: ~780T → ~993T)
2. Lazy rescaling from D=128 kernel (`rv_all_below`/`wave_all_ok`)
3. `readfirstlane` for LDS base addresses
4. K prefetch overlap with QK compute
5. Scheduling barrier tuning for 8-warp config

### Remaining gap to 1200T+:
- AITER uses hand-written ASM with instruction-level scheduling
- Need `rocprofv3` profiling to identify MFMA utilization gaps
- Potential: V tile splitting with `subtile_inplace<16>`, stagger pattern

## Backward Kernel Status: dK CORRECT, dV/dQ WIP

### Current: `attn_bkwd_causal_d192v128.cpp`
- **dK**: ✅ Correct (finite values, matches reference pattern)
- **dV**: ❌ NaN in transpose epilogue
- **dQ**: Not yet implemented in current version

### Architecture (modeled after `attn_bkwd_mla_simple.cpp`):
- `rt_32x32_s` base tiles (32×32 MFMA) — matches working reference
- `amdgpu_num_vgpr(29)` forces AGPRs=0, avoids gfx950 AGPR aliasing bug
- Hardcoded `v_accvgpr_read` for dK epilogue
- Standard `transpose()` for dV epilogue (works in reference, NaN here)
- Heavy VGPR spill (499 regs to scratch) — correct but slow

### Bugs found and fixed:

| Bug | Impact | Fix |
|-----|--------|-----|
| Causal mask q/k lane indices swapped | Wrong softmax weights | Fixed indexing for col_l rt_16x16_s layout |
| dS = P*(dP-delta) in bf16 inline ASM | Precision loss, potential ASM bugs | Changed to float32 mul |
| atomic_add_bf16_tile wrong byte offsets | OOB memory access for D_QK=192 | Rewrote with correct (row,col) addressing |
| amdgpu_num_vgpr(128) + spill | Scratchpad crash for large N | Switched to vgpr(29) style from reference |
| No vgpr limit → AGPRs=168 | Accumulator register aliasing → NaN | vgpr(29) forces AGPRs=0 |
| dQ tensor BNHD vs kernel expects BHND | Wrong atomic add positions | Fixed test to use BHND layout |
| kittens store<1> buffer_resource OOB | Memory violation at tensor boundary | Replaced with scalar stores |

### Root cause of the AGPR problem:
On gfx950, without ART (Asymmetric Register Tiles), the compiler freely assigns
MFMA accumulator registers to AGPRs. When multiple accumulator tiles (dV_j_T, dK_j_T,
P_ij, dP_ij) share AGPR space, the compiler may alias them — writing to one overwrites
another. The D=128 backward kernel avoids this with 3000+ lines of hand-scheduled ART
code with explicit register range assignments.

### Next steps for dV:
1. Add hardcoded AGPR read for dV epilogue (like dK's `store_dK_from_agpr`)
   - dV_acc is 128×32 = 64 AGPRs, need to find which AGPR range
   - Or add a `store_dV_from_agpr` with indices a[96:159] (after dK's a[0:95])
2. Once dV correct, validate dV cosine similarity > 0.99 vs reference
3. Add dQ path with atomic bf16 adds (fixed addressing)
4. Performance optimization (reduce spill, add scheduling barriers)

### Alternative approach: full ART port
The most reliable solution is to port the D=128 ART layout to D192/V128:
- All register tiles get explicit AGPR/VGPR range assignments
- `transpose_2d` views for zero-cost transpose in epilogue
- `swap_layout_inplace` safe with non-overlapping ranges
- Estimated effort: ~1 week of careful register planning

## File Index

### Forward
- `kernels/attn/gqa_causal/kernel_d192v128.cpp` — canonical forward (993T)
- `kernels/attn/gqa_causal/kernel.cpp` — D=128 reference forward (1100T)
- `kernels/attn/gqa_causal/test_python_d192v128.py` — forward benchmark

### Backward
- `kernels/attn/gqa_causal_backwards/attn_bkwd_causal_d192v128.cpp` — current backward (dK ✓)
- `kernels/attn/gqa_causal_backwards/attn_bkwd_causal.cpp` — D=128 ART reference (correct)
- `training/llama/csrc/attn_bkwd_mla_simple.cpp` — MLA non-causal reference (correct, no ART)
- `kernels/attn/gqa_causal_backwards/test_python_d192v128.py` — backward test
