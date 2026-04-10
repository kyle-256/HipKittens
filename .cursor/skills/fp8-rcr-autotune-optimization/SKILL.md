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

## Benchmark 结果

### GPU7 权威结果 (2026-04-09, JIT 4-wave)

| Layout | Geo-mean | Wins | 状态 |
|---|---|---|---|
| **RCR** | **1.019x** | **29/48** | **已超越 hipBLASLt** |
| **RRR** | **~1.51x** | **48/48** | **远超** |
| **CRR** | **~1.95x** | **48/48** | **远超** |

**RCR 弱项 shapes (K=3584/4096 + 大 M/N):**
- (16384,37888,3584): 0.929x
- (16384,28672,4096): 0.933x
- (8192,37888,3584): 0.949x

**RCR 强项 shapes:**
- (4096,6144,4096): 1.306x
- (8192,4608,3584): 1.240x
- (16384,3584,3584): 1.177x

### 历史 baseline (2026-04-08, GPU4, dynamic 8-wave)

| Layout | Geo-mean | Wins |
|---|---|---|
| RCR | 0.949x | 8/56 |
| RRR | 1.508x | 56/56 |
| CRR | 1.951x | 56/56 |

## 已尝试的优化（2026-04-08）

### 1. RCR_BATCHED_PAIR_MMA=1（正确性 FAIL）

减少 barrier 从 8→4 per k-iter。编译 220 VGPRs, occupancy=2。
**正确性 FAIL**: 非 fastpath shapes SNR 降至 30-38dB。
可能原因: init 阶段 `vmcnt(4)` + `vmcnt(6)` 不够保证 As[0][0]、Bs[0][1] 的 G::load 完成。

### 2. VMCNT/PREFETCH_LGKM 参数扫描

| Config | 4096,28672,4096 | 8192,16384,16384 | 8192,57344,8192 |
|---|---|---|---|
| baseline (vm4,lgkm4) | 2577 | 3172 | — |
| vm8 | **2628** | **3184** | — |
| vm6_lgkm6 | 2626 | 3180 | — |
| vm2 | 2450 | 3008 | — |

**结论**: vm8 最佳但仅 ~1-2% 改善，远不够闭合差距。

### 3. Barrier 移除实验（RCR_REDUCED_BARRIERS）

移除 BARRIER3/BARRIER5（保护不同 LDS tile，理论上可移除）。
**结果**: 正确性 PASS 但性能反降 5-8%。在 CDNA4 上，barrier 起调度栅栏作用，移除导致指令排序变差。

### 4. 4-wave Fastpath 对比测试（关键发现）

编译多个 shape-specific 4-wave 和 8-wave fastpath 内核，在 GPU4 上测试。

| Shape | 8-wave dynamic | 8-wave exact | **4-wave exact** | hipBLASLt |
|---|---|---|---|---|
| 8192×16384×16384 | 3143 | 3079 | **3361** | 3312 |
| 16384×16384×16384 | 3152 | 3064 | **3367** | 3309 |
| 8192×57344×8192 | 3023 | 2330 | **3263** | 3290 |
| 4096×57344×8192 | 2931 | 2265 | **3205** | 3149 |
| 16384×106496×16384 | 3052 | 2323 | **3168** | 3287 |
| 8192×16384×53248 | 3105 | 3205 | **3369** | 3324 |

**4-wave exact 已在多个 shape 上超越 hipBLASLt！** 但仅限编译期固定维度。
8-wave exact 在大 N shape 上比 dynamic 更差，因为缺少 XCD swizzle。

### 5. 4-wave Dynamic 版本

创建了 `rcr_4wave_dynamic.inc` — 运行时维度的 4-wave 内核。
**结果**: 正确性 PASS，但性能比 8-wave dynamic 更差（~5-10%）。
**原因**: 运行时开销（地址计算、无循环展开）抵消了 4-warp 架构的优势。编译期优化是 4-wave 性能的关键。

## 新文件

| 文件 | 说明 |
|---|---|
| `rcr_4wave_dynamic.inc` | 4-wave 动态版本（`-DRCR_USE_4WAVE_DYNAMIC=1`），正确但性能不及 8-wave |
| `bench_vs_hipblaslt_clean_gpu4.json` | 干净 GPU 上的完整 benchmark |
| `sweep_vmcnt.py` | VMCNT 参数扫描脚本 |
| `bench_fastpath.py` | 4-wave/8-wave/dynamic 对比脚本 |

## 下一步方向（优先级排序）

### A. K-specialized 4-wave（最有前景，预估 5-10%）

4-wave exact 的优势来自编译期 K 优化。策略：
- 为常见 K 值（4096, 8192, 16384, 28672, 53248）编译特化内核
- M/N 使用运行时参数（网格和存储），K 使用编译期常量（内循环）
- 运行时根据 K 值 dispatch 到对应内核

```cpp
// 伪代码
switch (g.k) {
    case 4096:  dispatch_4wave<4096>(g); break;
    case 8192:  dispatch_4wave<8192>(g); break;
    case 16384: dispatch_4wave<16384>(g); break;
    default:    dispatch_8wave(g); break;
}
```

### B. 修复 8-wave exact fastpath 的 XCD swizzle

当前 8-wave exact fastpath 使用简单的 `br = bid / bpc` 导致大 N shape 上 L2 局部性差。
添加 XCD swizzle + group_m 可能恢复性能到 dynamic 水平或更好。

### C. RCR_BATCHED_PAIR_MMA 正确性修复

根因: init 阶段 VMCNT 不够保证所有 4 个 tic 缓冲区就绪。
修复: 将 `RCR_INIT0_VMCNT=0`（等所有 VMEM 完成）。需验证这是否修复正确性且不影响性能。

### D. JIT 编译（已实现，效果显著）

`jit_gemm.py` + `bench_jit.py` 实现了 shape-specific 4-wave 内核的 JIT 编译。

**JIT 4-wave 结果（vs hipBLASLt, GPU4 clean）:**

| Shape | JIT-4w | hipBLASLt | Ratio |
|---|---|---|---|
| (8192,16384,16384) | **3379** | 3312 | **1.020x** |
| (16384,16384,16384) | **3362** | 3309 | **1.016x** |
| (4096,57344,8192) | **3219** | 3149 | **1.022x** |
| (8192,16384,53248) | **3361** | 3324 | **1.011x** |
| (4096,4096,4096) | **2391** | 2318 | **1.032x** |
| (8192,57344,8192) | 3261 | 3290 | 0.991x |
| (8192,28672,4096) | 2957 | 3067 | 0.964x |
| **Geo-mean** | | | **0.997x** |

**使用方式:**
```bash
cd /shared_nfs/kyle/HipKittens2/analysis/fp8_gemm/mi350x
HIP_VISIBLE_DEVICES=4 python3 bench_jit.py
```

**注意:** 由于动态链接器限制，JIT 编译的 .so 不能和默认 tk_fp8_layouts 在同一进程中共存。`bench_jit.py` 用子进程绕过此限制。

**编译时间:** 每 shape 7-32s（首次），缓存后直接使用。

**限制:** M%256==0, N%256==0, K%128==0（不满足的 shape 回退到 8-wave dynamic）。

### E. 剩余差距分析

JIT 4-wave 在 K=4096 的 shape 上仍落后（0.91-0.96x）。可能原因：
- K=4096 时 ki=32，循环次数少，overhead 占比高
- 这些 shape 的 arithmetic intensity 较低
- hipBLASLt 可能对 K=4096 有特殊优化

可尝试方向：为 K=4096 shapes 调优 4-wave group_m、prefetch 参数。

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
