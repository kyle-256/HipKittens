# MXFP4 GEMM 优化 — Agent提示词

你在继续推进 `Hipkittens2` 里的 MXFP4 GEMM 优化工作。

## 项目位置
- **Repo**: `/shared_nfs/kyle/test/Hipkittens2`
- **Branch**: `agent/mxfp4-art-rewrite`
- **工作目录**: `analysis/fp8_gemm/mi350x`
- **GPU 使用约束**: 后续 benchmark / correctness / smoke 优先使用 `GPU 5/6/7`，不要再占用 `GPU 3`

## 主目标
- **主测 shape**: `4096x32768x128256`
- **项目目标**: `5484 TFLOPS`（原始 `5653T` aiter baseline 的 `97%`）
- **42-shape 硬门槛**: 所有 shape 都必须达到 `>=95% competitor_tflops`，并且保持 `0 CRASH`
- **当前 target-shape 主 benchmark 最优参考**: `5019.9T`（GPU7，`warmup=200 / iters=500`，`-DGROUP_SIZE_M=8`，`NONVOLATILE_SCALE_X2_POC=1`）
- **当前完整 `bench_all42_results.json` 工件**: 仍是 **pre-nvscale** sweep，target 记录为 `4909.9T / 5781.1T = 84.9%`

## 当前已验证状态
- `kernel_mxfp4_gluon_cpp.cpp` 仍然是主生产内核和主线优化对象。
- guarded `permlane` / packed wide-store 路径已经接好并验证过。
- `4bffa75d` 证明了 `permlane -> v_cvt_pk_bf16_f32 -> global_store_dwordx4` 的 lowering 是真的。
- `399cd546` 与 `PACKED_WIDESTORE_PMC.md` 证明了：**store epilogue 不是 target shape 的主瓶颈**。
- 当前第一个**可复现且跨 variant 家族为正**的 read-side win 是：`load_pq_scale_x2_async()` 改成可调度的 non-volatile `buffer_load_dwordx2`。
- 这个改动已经在多卡、多口径上被钉实：
  - target `4096x32768x128256`，`_gm8`，GPU7 主 benchmark harness: `4937.3T -> 5019.9T`
  - target `4096x32768x128256`，`_gm8`，GPU3 采样 A/B: `4885.2T -> 4988.9T`
  - `4096x32768x6144`，`_u8`: `3865.3T -> 3921.9T`
  - `default / _u32 / _gm2 / _u16` 的代表样本也全部小幅为正，因此该开关现在默认开启。
- GPU6 target-shape PMC 说明这更像**调度/等待收益**而不是**读量减少**：
  - `TCP_TOTAL_READ_sum`: `5.27224e9 -> 5.27224e9`（基本不变）
  - `TCC_REQ_sum`: `5.68768e8 -> 5.68768e8`（基本不变）
  - `SQ_WAIT_INST_ANY`: `1.05367e9 -> 1.03737e9`（约 `-1.55%`）
- 当前完整 42-shape artifact 还没有吸收这次 `NONVOLATILE_SCALE_X2_POC` 默认开启后的收益，所以 **variant 排序需要重新跑 42-shape 才能更新正式结论**。
- 新的 targeted probe 已确认：当前 `bench_all_42.py` 还漏了一条对 **small-K + large-N** 家族有实际价值的分支：
  - `16384x28672x2048`: 旧 harness 最优 `3204.4T` (`92.0%`) -> `_ts_gm2` 可到 `3279.5T` (`94.2%`)
  - `32768x28672x2048`: 旧 harness 最优 `3237.1T` (`96.5%`) -> `_ts_gm2u8` 可到 `3306.7T` (`98.6%`)
  - `4096x32768x6144`: `_spread` 基线附近只被 `_spread_gm2u8` 小幅抬高到 `4158.8T` (`91.4%`)，说明这类 `N=32768` 家族仍主要是 kernel 问题，不是单纯漏了一个 variant
  - `STEP3_BARRIER_VMCNT=12` 也在 `N=32768` 的 `TAIL_SPLIT` 家族上给出一致小胜：
    - `4096x32768x14336`: `_ts 4963.5T` -> `_ts_v12 4973.9T`
    - `4096x32768x28672`: `_ts 5115.4T` -> `_ts_v12 5132.7T`
    - `4096x32768x128256`: `_ts_gm8 5104.5T` -> `_ts_gm8_v12 5125.3T`
  - 同一个 `vmcnt=12` 旋钮在 default/`SPREAD_LDS` 家族也不是纯噪声：GPU6 同口径下 `4096x32768x6144` 有 `_spread_gm2u8 4104.9T` -> `_spread_gm2u8_v12 4164.6T`
  - 与之相对，`STEP3_BARRIER_VMCNT=0` 和 `STEP3_EMBED_BARRIER=0` 在已测大 `N` 家族上都是明显回退
  - `STEP3_PF_N=6` 只在 `4096x32768x6144/14336` 给出小胜，但在 `4096x32768x28672/128256` 回退，因此目前只保留为 probe 旋钮，不升格为通用 variant
  - 因此 `bench_all_42.py` 已补入 `_ts_gm2`、`_ts_gm2u8`、`_spread_gm2u8`、`_spread_gm2u8_v12`、`_ts_v12` 与 `_ts_gm8_v12`
- default path 与 `/shared_nfs/kyle/test/HipKittens` 的默认热路径基本一致，没有漏掉一个显而易见的现成 patch。
- 已有独立 kernel 证据显示：`half_direct`（≈ `3457.83T`）和 `direct_a`（≈ `2527.51T`）远低于当前主线，不应盲目 graft 回主核。
- `PF_N` 细调、低风险 barrier/prefetch flag 组合，以及 isolated Step12-swapped POC 目前都没有给出可信 target-shape 正收益。
- 当前 **pre-nvscale** 42-shape 工件是：`16 WIN / 26 LOSE / 0 CRASH`，平均 ratio 约 `98.8%`。
- 最新完整 sweep 中，`128256x32768x4096` 已不再 crash，但仍只有 `4063.3T / 4536.4T = 89.6%`。

## 不要再做的事
- 不要假设 ART 是唯一可行路径。用户已经明确允许继续优化原始 C++ kernel。
- 不要在没有 counter 证据时继续磨 store-only 优化。
- 不要重复老的死路：
  - LDS-transpose vecstore
  - 盲目的 barrier/prefetch 小修
  - 把 `GROUP_SIZE_M` 的噪声级波动包装成突破
  - 在没有新机制前提下，把 `half_direct` / `direct_a` 直接往主核里搬
- 不要混淆 `bench_all_42.py` 里的 `competitor_tflops` 和 live `aiter` 跑分；报告时必须说清口径。

## 当前主线判断
当前与 `aiter` 的主要差距更像在 **读路径和整体流水线重叠**，不是 store；最近的 `NONVOLATILE_SCALE_X2_POC` 正收益进一步支持这一点：

- `Frac_Active_VMEM` 更高
- `Frac_Wait_Any` 更高
- `TCP_TOTAL_READ_sum` 更高

这次 `nvscale` 的 PMC 也说明：收益主要来自**同等读量下等待减少**，而不是简单的流量下降。

所以下一阶段更值得做的是：

1. 减少 read-side traffic
2. 减少 LDS/global round-trips
3. 改善 inner-loop 的 load/compute overlap

## 优先实验方向
优先做 **最小、可 guard、可 A/B、可 PMC 验证** 的实验：

1. 先用 PMC 与新的 42-shape sweep验证 `NONVOLATILE_SCALE_X2_POC` 默认开启后的收益面
2. 先完成一轮带 `NONVOLATILE_SCALE_X2_POC=1` 的 42-shape 汇总，并按 `>=95% / 0 CRASH` 硬门槛报告剩余差距
3. `kernel_mxfp4_gluon_cpp.cpp` 中真正有新机制支撑的 read-side / overlap 改动
4. 再继续做 Step12 fused schedule 的最小结构改动
5. 仅在有明确机制与验证方案时再回到 A0-direct / half-direct / direct-B 风格 POC

如果一个实验不能很快回答“对 target shape 有无实质收益”，就不要把它扩成大改。

## 验证要求
每个像样的实验都必须按这个顺序闭环：

1. 编译
2. 正确性
3. target-shape benchmark（**固定 `warmup=200, iters=500`，禁止用更短 benchmark 下结论**）
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
5. **以后凡是 benchmark 结论，统一使用 `warmup=200, iters=500`**
6. **旧的手工 `C.zero_()` target-shape A/B 不能再当主 benchmark 结论**
7. **`bench_all_42.py` 的历史输出/JSON 里若写着 `50/100`，那是旧元数据，实际 runner 现应以 `200/500` 为准**
8. **`buffer_load_dwordx2` 的 scale 直读调度自由度是真问题，`asm volatile -> asm` 已给出首个广义正收益**
9. **判断这类 read-side 改动时，要优先看“同等读量下 wait 是否下降”，而不只是看 bytes/counter 总量**
10. **42-shape 汇报时必须显式给出 `<95%` 的 shape 数量，因为当前用户硬门槛就是全 shape 不得落后超过 `5%`**
11. **`TAIL_SPLIT` 不是单独万能，但 `TAIL_SPLIT + GM2` 已经证明对 `N=28672, K=2048` 家族有真收益**
12. **对当前 non-SWAP 主核，`STEP3_BARRIER_VMCNT=12` 比 `8` 更值得重点盯 `N=32768` 的 `TAIL_SPLIT` 家族**

## 常用文件
- `kernel_mxfp4_gluon_cpp.cpp`
- `NVSCALE_PMC.md`
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
- `pmc_seq_baseline_gm8_4096x32768x128256/`
- `pmc_seq_nvscale_gm8_4096x32768x128256/`
