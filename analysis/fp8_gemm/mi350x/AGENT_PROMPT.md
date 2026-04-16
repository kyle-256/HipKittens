# MXFP4 GEMM 优化 — Agent提示词

你在继续推进 `Hipkittens2` 里的 MXFP4 GEMM 优化工作。

## 项目位置
- **Repo**: `/shared_nfs/kyle/test/Hipkittens2`
- **Branch**: `agent/mxfp4-art-rewrite`
- **工作目录**: `analysis/fp8_gemm/mi350x`

## 主目标
- **主测 shape**: `4096x32768x128256`
- **项目目标**: `5484 TFLOPS`（原始 `5653T` aiter baseline 的 `97%`）
- **当前默认 full-swap 可复现水平**: 约 `4902T` 到 `4911T`
- **当前 `bench_all42_results.json` 记录**: `4974.6T / 5781.1T = 86.0%`

## 当前已验证状态
- `kernel_mxfp4_gluon_cpp.cpp` 仍然是主生产内核和主线优化对象。
- guarded `permlane` / packed wide-store 路径已经接好并验证过。
- `4bffa75d` 证明了 `permlane -> v_cvt_pk_bf16_f32 -> global_store_dwordx4` 的 lowering 是真的。
- `399cd546` 与 `PACKED_WIDESTORE_PMC.md` 证明了：**store epilogue 不是 target shape 的主瓶颈**。
- 当前 42-shape 工件是：`18 WIN / 23 LOSE / 1 CRASH`，其中 `25/42 >= 97%`。
- 当前 sweep 里记录的 crash shape 是 `128256x32768x4096`。

## 不要再做的事
- 不要假设 ART 是唯一可行路径。用户已经明确允许继续优化原始 C++ kernel。
- 不要在没有 counter 证据时继续磨 store-only 优化。
- 不要重复老的死路：
  - LDS-transpose vecstore
  - 盲目的 barrier/prefetch 小修
  - 把 `GROUP_SIZE_M` 的噪声级波动包装成突破
- 不要混淆 `bench_all_42.py` 里的 `competitor_tflops` 和 live `aiter` 跑分；报告时必须说清口径。

## 当前主线判断
当前与 `aiter` 的主要差距更像在 **读路径和整体流水线重叠**，不是 store：

- `Frac_Active_VMEM` 更高
- `Frac_Wait_Any` 更高
- `TCP_TOTAL_READ_sum` 更高

所以下一阶段更值得做的是：

1. 减少 read-side traffic
2. 减少 LDS/global round-trips
3. 改善 inner-loop 的 load/compute overlap

## 优先实验方向
优先做 **最小、可 guard、可 A/B、可 PMC 验证** 的实验：

1. `kernel_mxfp4_gluon_cpp.cpp` 中 Step3/4 的 prefetch 交错方式
2. Step12 fused schedule 的局部重排
3. 最小化的 A0-direct / half-direct 风格 read-side POC

如果一个实验不能很快回答“对 target shape 有无实质收益”，就不要把它扩成大改。

## 验证要求
每个像样的实验都必须按这个顺序闭环：

1. 编译
2. 正确性
3. target-shape benchmark
4. 如果看起来有希望，再做 PMC
5. 决定 commit 还是回退

一个改动只有在满足下面至少一条时才值得 commit：

- 有可复现的性能正收益
- 以硬证据封死了一条很诱人的死路
- 留下了可复用的 scaffold，并且 lowering / geometry / PMC 结论已经坐实

## 关键发现
1. **Direct-A without latency hiding = slower**
2. **Store path 已经证明不是当前 target shape 的主瓶颈**
3. **`packed wide-store` 是有效 scaffold，不是主线性能解**
4. **以后做结论必须显式写 benchmark 口径**

## 常用文件
- `kernel_mxfp4_gluon_cpp.cpp`
- `PACKED_WIDESTORE_PMC.md`
- `TODO.md`
- `bench_all_42.py`
- `bench_all42_results.json`
- `docs/profiling/profile_pmc_counters.sh`
- `docs/profiling/analyze_pmc_counter_output.py`

## 关键数据目录
- `pmc_aiter_4096x32768x128256/`
- `pmc_baseline_4096x32768x128256/`
- `pmc_baseline_gm8_4096x32768x128256/`
- `pmc_rewrite_4096x32768x128256/`
- `pmc_fullswap_curr_4096x32768x128256/`
- `pmc_packedwide_curr_4096x32768x128256/`
