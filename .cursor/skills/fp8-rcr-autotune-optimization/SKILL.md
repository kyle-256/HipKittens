---
name: fp8-rcr-autotune-optimization
description: Continue optimizing HipKittens FP8 per-tensor GEMM RCR layout to surpass hipBLASLt. Covers autotune framework, block swizzle, BATCHED_READS experiment, and benchmark methodology.
---
# FP8 RCR Autotune & Optimization — 接力文档

## 目标

**RCR 全面超越 hipBLASLt**，RRR/CRR 相对 RCR 的比例保持不变。

## 当前分支

```
save/fp8-progress-20260319-native-layouts
```

最新 commit: `c41c261a` — Add experimental RCR_BATCHED_READS optimization

## 关键文件

| 文件 | 说明 |
|---|---|
| `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` | **核心 C++ 内核**。RCR/RRR/CRR 三个 layout 的 FP8 per-tensor GEMM |
| `analysis/fp8_gemm/mi350x/Makefile` | 编译配置，控制所有 `#define` 开关 |
| `analysis/fp8_gemm/mi350x/autotune.py` | **Python autotune 模块**。按 (M,N,K,layout) 搜索最优 `group_m`，结果缓存到 `.autotune_cache.json` |
| `analysis/fp8_gemm/mi350x/bench_vs_hipblaslt.py` | **对比 benchmark 脚本**。HipKittens vs `hipBLASLt` (Primus-Turbo) |
| `analysis/fp8_gemm/mi350x/test_python.py` | 单 shape 正确性 + 性能测试 |
| `analysis/fp8_gemm/mi350x/*_fastpath.inc` | 特定维度的编译期优化 fastpath kernel（仅 M_DIM=N_DIM=K_DIM=8192 且 scale=1.0 时命中） |

## 已完成的优化

### 1. XCD-aware Block Swizzle（已启用，已验证）

**Makefile 配置:**
```
-DGEMM_BLOCK_SWIZZLE=1 -DGEMM_BLOCK_SWIZZLE_NUM_XCDS=8
```

**效果:** RCR 大 N shapes 从 0.66x→0.93x，geo-mean 从 0.910x→0.958x

### 2. Runtime `group_m` + Python Autotune（已实现，已验证）

`group_m` 从编译期常量改为运行时参数。不同 shape 偏好不同 `group_m`:

| Shape 特征 | 最优 group_m |
|---|---|
| 小 shape (4096²) | 1 |
| 中 shape / 大 K | 4 |
| 中 N (28672) | 4-8 |
| 大 N (57344+) | 4 |

Python 端通过 `AutotunedGEMM` 类自动搜索 `group_m ∈ {1,2,4,8,16}`，缓存到 `.autotune_cache.json`。

**调用方式:**
```python
import tk_fp8_layouts
# 直接调用（指定 group_m）
tk_fp8_layouts.gemm_rcr(A, B, C, scale_a, scale_b, group_m=4)

# 或用 autotune
from autotune import AutotunedGEMM
gemm = AutotunedGEMM(verbose=True)
gemm.rcr(A, B, C, 1.0, 1.0)  # 自动选最优 group_m
```

### 3. Pybind 暴露的常量

```python
tk_fp8_layouts.DEFAULT_GROUP_M  # 4
tk_fp8_layouts.BLOCK_SIZE       # 256
tk_fp8_layouts.K_BLOCK          # 128
```

## 未完成 / 需要 GPU 测试

### A. Clean Benchmark（最高优先）

上次 benchmark 时所有 8 张 GPU 被其他进程占满 (~287 GiB/卡)，导致：
- hipBLASLt TFLOPS 暴跌 40%（缺少 workspace 内存）
- TK 也下降 ~10%
- 对比结果不可信（TK 55/56 胜但绝对值偏低）

**需要在干净 GPU 上重新跑:**
```bash
cd /shared_nfs/kyle/HipKittens2/analysis/fp8_gemm/mi350x
rm -f .autotune_cache.json
HIP_VISIBLE_DEVICES=<clean_gpu> python3 bench_vs_hipblaslt.py \
    --mode full --warmup 30 --iters 50 --mbs 1,2 \
    -o bench_vs_hipblaslt_clean.json
```

**上次（GPU 被占前）的干净基线数据:**
- `bench_vs_hipblaslt_swizzle.json` — block swizzle 启用后 group_m=4 固定值
- RCR geo-mean: **0.958x**（距离超越 hipBLASLt 还差 ~4-5%）

### B. RCR_BATCHED_READS 实验（需要调试）

**思路:** 将主循环中的 LDS 读取从交错式（每次1-2个，中间穿插 barrier）改为批量式（一次读4个 tile），减少 barrier 从 8→2 个/K迭代。

**状态:** 编译通过（242 VGPRs, 0 spills, occupancy=2），但正确性 FAIL。
- 所有非 fastpath shapes 的 SNR 降至 -2~4 dB
- 8192³ "通过"是因为命中了 fastpath kernel（完全不走新代码）

**启用方式（仅供调试）:**
```makefile
# 在 Makefile HIPFLAGS 中加:
-DRCR_BATCHED_READS=1
```

**疑似问题:** G::load 向 tic 缓冲区发起的异步 LDS store 与后续迭代的 LDS read 之间存在竞态。当前代码在 lgkm(0) 后加了一个额外 barrier 但仍然不够。需要更仔细分析 CDNA4 的 LGKMCNT 语义（ds_write 依赖 VMEM 数据时是否计入 LGKMCNT）。

**调试建议:**
1. 在 `lgkm(0)` 前加 `vmcnt(0)` 强制等所有 VMEM 完成
2. 用 `(256,256,256)` 这种小到不需要 tail kernel 的 shape 单步调试
3. 比较 BATCHED_READS=0 和 =1 的输出 tensor diff，定位出错的 block/warp

### C. 其他可尝试的优化方向

| 方向 | 预估收益 | 风险 |
|---|---|---|
| VMCNT 参数调优（编译多 variant） | 2-3% | 编译时间 ×N |
| RCR_BATCHED_EPILOGUE_MMA=1 | 1-2% | 上次单独测试未验证 |
| 增大 PREFETCH_LGKM（4→6） | 1-2% | 可能增加 stall |
| 减少 barrier（逐个移除验证） | 3-5% | 需要非常仔细的正确性验证 |
| Triple buffering | 5-10% | 大改，LDS 可能不够 |

## 编译 & 测试流程

```bash
cd /shared_nfs/kyle/HipKittens2/analysis/fp8_gemm/mi350x

# 编译
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 ROCM_PATH=/opt/rocm make clean && \
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 ROCM_PATH=/opt/rocm make -j4

# 单 shape 正确性测试
HIP_VISIBLE_DEVICES=7 FP8_LAYOUTS=rcr FP8_CHECK=1 FP8_DETERMINISM_RUNS=3 \
    python3 test_python.py 4096 28672 4096

# 多 shape 快速验证
HIP_VISIBLE_DEVICES=7 python3 -c "
import torch, math, tk_fp8_layouts
for M,N,K in [(4096,4096,4096),(4096,28672,4096),(4096,57344,8192)]:
    A = (torch.randn(M,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
    B = (torch.randn(N,K,device='cuda')*0.1).to(torch.float8_e4m3fn)
    C = torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
    ref = A.float()@B.float().T
    tk_fp8_layouts.gemm_rcr(A,B,C,1.0,1.0,4)
    snr = 10*math.log10((ref**2).sum().item()/((C.float()-ref)**2).sum().item())
    print(f'({M},{N},{K}) SNR={snr:.2f}dB {\"PASS\" if snr>48 else \"FAIL\"}')
    del A,B,C,ref; torch.cuda.empty_cache()
"

# autotune + 完整对比
HIP_VISIBLE_DEVICES=7 python3 bench_vs_hipblaslt.py --mode full --warmup 30 --iters 50 --mbs 1,2
```

## hipBLASLt 依赖

```python
# 需要 primus_turbo 安装在环境中
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import GEMMFP8HipBLASLtBackend
hipblaslt_fn = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp8
# 调用: hipblaslt_fn(A, scale_a, B, scale_b, torch.bfloat16, trans_a, trans_b, False, "TENSORWISE")
```

## 内核架构要点

- **Block size:** 256×256 output, 128 K-step
- **Warps:** 2×4 = 8 warps/block, 512 threads
- **Double buffering:** tic/toc 交替，k+1 的 A[1] 和 k+2 的 B[0],A[0],B[1] 分两步预取
- **Fastpath:** M=N=K=8192 且 scale=1.0 时走编译期优化内核（不受 dynamic kernel 改动影响）
- **VGPRs:** RCR=212, RRR/CRR=230-242, 所有 ≤256 (occupancy=2 waves/SIMD)
- **SRD 4GB 限制:** 输出矩阵 C 超过 4GB 时 GPU fault，MBS=4 的最大 shape 会触发

## 正确性标准

- **SNR > 48 dB**（相对于 FP32 参考）
- **Determinism:** 同输入多次运行结果 bit-exact
- 所有 group_m 值 (1,2,4,8,16) 都必须通过
