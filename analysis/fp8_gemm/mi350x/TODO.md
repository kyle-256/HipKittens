# MXFP4 GEMM Optimization TODO

## Current State (2026-04-15, updated)
- **4096×32768×128256**: Original C++ GM=8 = **4926 TFLOPS** (85.4% of aiter gross, **97.9% of net aiter**)
- **8192³**: Original C++ = 4572 TFLOPS; ART kernel = 3185 TFLOPS
- **42-shape gross comparison**: 13/41 WIN (vs aiter GEMM-only)
- **42-shape training-fair**: **28/41 WIN, 33/41 ≥ 97%** (accounting for B preshuffle overhead)
- **8 shapes remain < 97%** of net aiter
- **0 crashes** (128256×32768×4096 excluded)
- **Branch**: `mxfp4`

## KEY FINDING: Training-Fair Comparison
aiter's 5653T requires pre-shuffled B weights (offline preprocessing). In training,
B changes every iteration → preshuffle must run per-GEMM call, adding:
- B preshuffle: N×K bytes memory traffic (read+write)
- Scale preshuffle: ~10% of B cost
- CK without preshuffle: **165 TFLOPS** (30× worse than our 4926T!)

When accounting for preshuffle overhead in aiter's numbers:
- **4096×32768×128256**: aiter net = 5043T (preshuffle=872us), ours = 4935T → **97.9%**
- **28/41 shapes**: ours WINS outright (>100% of net aiter)
- **33/41 shapes**: ours ≥ 97% of net aiter

## Commits Made
```
20650a6f streamline store epilogue — free 4 VGPRs (252 VGPRs)
e8b2ddc5 fix int32 overflow crash on large M*N (128256x32768)
03148718 vmcnt+barrier into Step3 via EMIT_BARRIER
80675e9c merged scale preshuffle + buffer_load_dwordx2 (核心优化)
57c9d86f README rewrite + XCD dispatch fix + benchmark infra
```

## Environment Setup (换机器必读)
```bash
# 项目
cd /shared_nfs/kyle/test/HipKittens
git checkout mxfp4

# Gluon竞品 (需要 triton_for_gluon)
export PYTHONPATH=/shared_nfs/kyle/test/triton_for_gluon/python:$PYTHONPATH
export TRITON_ENABLE_LLIR_SCHED=1
export TRITON_ENABLE_AMDGCN_AS=1

# aiter竞品
# cd /shared_nfs/kyle/test/aiter && pip install -e .

# Benchmark
cd analysis/fp8_gemm/mi350x
HIP_VISIBLE_DEVICES=0 python3 bench_all_42.py         # 42-shape (ours)
HIP_VISIBLE_DEVICES=1 python3 bench_gluon_a4w4_42.py  # Gluon竞品
HIP_VISIBLE_DEVICES=2 python3 bench_aiter_a4w4_42.py  # aiter竞品
```

## Benchmark Rules
- **warmup=200, iters=500**, trimmed mean 10%
- 用空闲GPU (`rocm-smi` 确认0%)
- `HIP_VISIBLE_DEVICES=N`
- MI355X上 competitor_tflops 是正确baseline

## 确认的性能天花板 (C++ inline ASM, ThunderKittens框架)
- **93% of ASM** — 由三重硬件约束锁定
- col_l accumulator: store不可vectorize (~5%开销, memory bandwidth限制)
- 512 register (256V+256A): direct-A VGPR loading不可行
- compiler asm块边界: s_nop不可消除但是free的(dual-issue)

## 未完成的可能方向

### 1. 省VGPR + direct-A loading (最高优先级, 接近可行!)
**发现：LDS addressing可以省大量VGPR。** Agent实现了252→182 VGPRs(省70个)，但correctness有bug:
- `addr_p1 = addr_p0 + 64` 假设错误(XOR swizzle在256-byte边界不保持)
- double-buffer XOR toggle (v_xor_b32 16384) 思路正确但需要验证
- 只需省16个VGPR就够做direct-A (需要64 VGPR, 可从A LDS基础设施释放72)

aiter的VGPR布局 (反编译确认):
- v0-v7: thread ID + scratch (8)
- v8-v135: B_left + B_right tile data (128, 全部同时live)
- v136-v199: A tile data (64, direct VMEM load)
- v200-v211: scales (12)
- v212-v250: offsets + addresses (39)
- Total: 251 VGPRs

我们的loop body只用v0-v203 (204)。v204-v255在loop内空闲(epilogue only)。
差距: 我们204 + 64(direct-A) = 268, 超出12个。
但如果用aiter的LDS addressing(2 VGPRs代替10, 省8)+ 消除cndmask(省4-8)就够了。

**关键: phase1 offset不是+64！必须保留两个phase的地址。只优化double-buffer选择。**

### 2. v_permlane16_swap_b32 vectorized store
aiter的store epilogue用:
- v_permlane16_swap_b32 (64次, VALU pipe不走LDS) 做lane间数据交换
- v_cvt_pk_bf16_f32 (128次) 打包bf16对
- buffer_store_dwordx4 (32次 × 16字节) 代替我们的256次 × 2字节
之前所有store优化agent都没试过v_permlane16_swap (它们只试了ds_bpermute/LDS)

### 3. GROUP_SIZE_M=8 for large-N shapes
实测 4096x32768x128256: GM=8比GM=4快1.1% (4922 vs 4867 TFLOPS)

### 4. CK (Composable Kernel) 后端 — DEAD END
- CK without preshuffle (BLayout=Col): **165 TFLOPS** = 30× worse than ours
- CK with preshuffle (BLayout=MFMA): ~5200T gross, but preshuffle adds 5-10%
- Net CK with preshuffle: ~4700-5100T ≈ our 4926T — no meaningful improvement
- **Conclusion**: CK not viable for training scenario

## ART Kernel Progress (2026-04-15, latest)
- **kernel_mxfp4_art.cpp**: ART-style MXFP4 GEMM with direct-A loading
- **Status**: Bit-exact correct, 256V+256A, 0 spills, 0 scratch
- **Performance**: **3185T at 8192³** (70% of original 4572T)
- **Architecture**: Single monolithic asm per K-iteration (128 MFMAs + 32 loads + barrier)
- **Progress**: 2457T → 2855T → 3057T → 3185T (each iteration improved scheduling)

### Exhaustive C++ optimization analysis (all ~4926T = structural limit)
- GROUP_SIZE_M=8: 4926T (+1.2%, best C++ result)
- XOR double-buffer toggle: 4917T (no improvement — compiler already efficient)
- Vectorized store (LDS transpose): 4509T (WORSE — LDS overhead)
- Half-direct-A (A0 direct, A1 LDS): 3703T (WORSE — global load latency > LDS)
- Direct-A C++ (23 spills): 2547T (spills kill performance)
- Compiler flags, unroll sweep: no improvement
- sched_group_barrier + iglp_opt: 5023T vs 5025T baseline (NO improvement)
- ds_bpermute wide store: 3321T vs 3953T reference (16% WORSE — LDS latency)
- GROUP_SIZE_M=16: no improvement on any shape
- CK without preshuffle: 165T (30× WORSE)

### Assembly analysis findings
- Compiler generates only 4 instructions gap between MFMA blocks (2 waitcnts + barrier + 1 salu)
- extract_tile is optimized away (zero v_mov instructions between ds_read and MFMA)
- 384 MFMAs + 96 ds_reads + 48 buffer_load_lds per 3-unrolled loop body
- Structural overhead: 28 extra memory ops per iteration vs aiter (B through LDS)
- Per-MFMA overhead: ~3.6 cycles slower than aiter = ~14% on compute-bound shapes

### Key discoveries
1. **C++ structural limit = 4926T** — cannot be broken without ASM rewrite
2. **Direct-A without latency hiding = SLOWER** — buffer_load 200+cy vs ds_read 20-40cy
3. **Compiler clobbers AGPRs during store** → must read all 64 in single asm block
4. **`"=&v"` early-clobber mandatory** on all outputs when mixing with `"v"` scale input
5. **UNROLL_K>1 crashes with K=128256** on ART kernel (code size limit)
6. **aiter confirmed row-major output** (not transposed)

### Remaining 8 shapes below 97% of net aiter (structural gap)
| Shape | Best | net aiter | net% | Bottleneck |
|-------|------|-----------|------|------------|
| 14336×4096×32768 | 4621 | 5054 | 91.4% | Compute (direct-B advantage) |
| 16384×4096×28672 | 4899 | 5339 | 91.8% | Compute (direct-B advantage) |
| 28672×32768×4096 | 4000 | 4396 | 91.0% | Store + dispatch |
| 28672×4096×16384 | 4897 | 5249 | 93.3% | Compute |
| 16384×28672×2048 | 3216 | 3407 | 94.4% | Store + dispatch |
| 32768×4096×14336 | 4810 | 5138 | 93.6% | Compute |
| 14336×32768×4096 | 4086 | 4323 | 94.5% | Store + dispatch |
| 16384×28672×4096 | 4075 | 4292 | 94.9% | Store + dispatch |

### Next steps (require fundamental architecture change)
1. Full hand-written ASM kernel (multi-month effort) — eliminates B LDS overhead via direct-B with runtime shuffle
2. Or: accept 91-95% on 8 shapes as C++ structural limit

## 关键文件索引
| 文件 | 用途 |
|------|------|
| `kernel_mxfp4_gluon_cpp.cpp` | 主C++内核 (4926T w/ GM=8) |
| `kernel_mxfp4_art.cpp` | ART内核 (3185T, direct-A, 0 spills) |
| `kernel_mxfp4_half_direct.cpp` | Half-direct-A (3703T, 实验性) |
| `kernel_mxfp4_direct_a.cpp` | Full direct-A C++ (2547T, 19 spills) |
| `kernel_mxfp4_xor_toggle.cpp` | XOR toggle (4917T, 无提升) |
| `kernel_mxfp4_vecstore.cpp` | LDS transpose store (4509T, 无提升) |
| `art_register_map.md` | ART VGPR/AGPR寄存器分配 |
| `OPTIMIZATION_ROADMAP.md` | 完整优化路线图和分析 |
| `bench_all_42.py` | 42-shape benchmark (vs competitor_tflops) |
| `bench_gluon_a4w4_42.py` | Gluon竞品benchmark |
| `bench_aiter_a4w4_42.py` | aiter竞品benchmark |
| `test_mxfp4_gluon_cpp.py` | 快速正确性+性能测试 |
| `rewrite_mxfp4_gluon.py` | .s后处理器 (当前kernel已吸收其收益) |
| `README.md` | 完整文档 |

## 竞品参考数据 (MI355X实测)
| Kernel | 8192³ | 4096×32768×128256 |
|--------|-------|-------------------|
| Our C++ GM=8 | 4572T | **4926T** |
| ART (direct-A) | 3185T | ~3100T |
| aiter ASM | ~5090T | **5653T** |
| 97% target | — | **5484T** |
