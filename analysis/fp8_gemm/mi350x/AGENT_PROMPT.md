# MXFP4 GEMM 优化 — Agent提示词

你在继续推进 `HipKittens` 的 MXFP4 GEMM 优化工作，跟 Cursor (Hipkittens2) 竞赛。

## 项目位置
- **我们的 Repo**: `/shared_nfs/kyle/test/HipKittens`
- **Branch**: `mxfp4`
- **工作目录**: `analysis/fp8_gemm/mi350x`
- **Cursor Repo**: `/shared_nfs/kyle/test/Hipkittens2` (只读参考)

## 当前成绩
- **我们**: **18/42 WIN** (warmup=200, iters=500)
- **Cursor**: 17/42 WIN (同参数)
- **我们领先 1 WIN, 26/42 shapes 绝对 TFLOPS 更高**
- **28/42 在 5% 以内, 14/42 超过 5% gap**
- **Target shape** 4096×32768×128256: 我们 4967T (85.9%), Cursor 5089T (88.0%)

## 已做的优化
1. Store block reorder (A0Bl,A0Br,A1Bl,A1Br) — +0.8% commit `8f10b09e`
2. MFMA operand SWAP — 正确但慢, auto-tune 选项
3. TAIL_SPLIT=1 — 小K帮助, 大K退化, auto-tune 选项
4. SPREAD_LDS=1 — 退化1-2%, auto-tune 选项
5. 20-variant auto-tune (GM1/2/4/8/16, U8/16/32, SWAP, TS, 组合)

## 不要再做的事
- **Direct-B (不preshuffle)**: 正确但慢28% (buffer_load延迟)
- **BK=256**: LDS装不下 (256KB > 160KB max)
- **Preshuffle-B 1-pass**: VGPR spill → NaN
- **Preshuffle-B 2-pass**: 正确但慢56% (K-loop跑两遍)
- **Non-volatile scale loads**: 在我们代码上产出NaN (Cursor能用但我们不行)
- **Rowspread ds_reads**: 退化1-2%
- **sched_group_barrier / iglp_opt**: 无改善
- **ds_bpermute wide stores**: 退化16%
- 把 preshuffle 时间不算进比较 — 用户明确拒绝过

## 结构性限制
- B走LDS是根本瓶颈: +18% read traffic, 2x wait time vs aiter
- 256 AGPR (4 acc blocks) + B tiles 无法同时放进 256 VGPRs
- 不 preshuffle B 就不能跳过 LDS
- LDS swizzle 是 MFMA 需要的数据排列 (非 bank conflict avoidance)
- buffer_load 200cy vs ds_read 20cy → direct loading 总是更慢

## Cursor 在做的 (可学习)
- `NONVOLATILE_SCALE_X2_POC=1`: 去掉 scale load 的 volatile (在他们代码上有效)
- `STEP3_BARRIER_VMCNT=12`: large-N shapes 上有收益
- `SPREAD_LDS + VMCNT=12 + TAIL_SPLIT 组合`: 针对 large-N shapes
- `STEP3_PF_N/STEP4_PF_N` 调参: 不同 prefetch 深度

## 优先方向
1. **VMCNT=12 auto-tune**: 加入 bench_all_42.py variants
2. **Debug non-volatile scale**: 找到为什么在我们这里 NaN
3. **Per-shape 精准调参**: 针对 Cursor 赢的 10 个 shapes
4. **benchmark 跑完后分析**: GPU1 上的 bench_all_42 正在跑

## Benchmark 规则
- **warmup=200, iters=500**, trimmed mean 10%
- GPU 1-4 可用 (`HIP_VISIBLE_DEVICES=1,2,3,4`)
- `rocm-smi --showuse` 确认 GPU 空闲
- 所有 benchmark 结论必须标注 warmup/iters

## 关键文件
- `kernel_mxfp4_gluon_cpp.cpp` — 主内核
- `bench_all_42.py` — 42-shape benchmark (20 auto-tune variants)
- `spot_test.py` — 单shape多variant测试
- `bench_all42_results.json` — 最新结果
- `kernel_mxfp4_direct_b.cpp` — Direct-B 实验 (参考, 不用于生产)
