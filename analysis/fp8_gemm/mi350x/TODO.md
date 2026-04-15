# MXFP4 GEMM Optimization TODO

## Current State (2026-04-15)
- **8192³**: 4740 TFLOPS (93.2% of ASM inline 5084T)
- **42-shape vs competitors**: 19/42 WIN, 22 LOSE, 1 shape SRD limit
- **0 crashes** (all fixed)
- **Branch**: `mxfp4`, 5 commits ahead of baseline

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

### 4. CK (Composable Kernel) 后端
aiter的ASM kernel来自CK codegen。CK有自动tuning和手写ASM template。
关键文件: `/opt/rocm/include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`

### 2. CK (Composable Kernel) 后端
aiter的ASM kernel来自CK codegen。CK有自动tuning和手写ASM template。
可以研究CK的FP4 GEMM实现,理解它如何做到direct-A + vectorized store。
关键文件: `/opt/rocm/include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`

## 关键文件索引
| 文件 | 用途 |
|------|------|
| `kernel_mxfp4_gluon_cpp.cpp` | 主C++内核 (4740T) |
| `bench_all_42.py` | 42-shape benchmark (vs competitor_tflops) |
| `bench_gluon_a4w4_42.py` | Gluon竞品benchmark |
| `bench_aiter_a4w4_42.py` | aiter竞品benchmark |
| `test_mxfp4_gluon_cpp.py` | 快速正确性+性能测试 |
| `rewrite_mxfp4_gluon.py` | .s后处理器 (当前kernel已吸收其收益) |
| `README.md` | 完整文档 |

## 竞品参考数据 (MI355X实测)
| Kernel | 8192³ | 42-shape avg |
|--------|-------|-------------|
| Our C++ | 4740T | 4286T |
| Gluon LLIR | 4983T | 4274T |
| aiter ASM | 4082T | 3850T |
