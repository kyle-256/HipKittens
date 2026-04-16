# MXFP4 GEMM 优化 — Agent提示词

你在继续推进 `HipKittens` 的 MXFP4 GEMM 优化工作，跟 Cursor (Hipkittens2) 竞赛。

## 项目位置
- **我们的 Repo**: `/shared_nfs/kyle/test/HipKittens`
- **Branch**: `mxfp4`
- **工作目录**: `analysis/fp8_gemm/mi350x`
- **Cursor Repo**: `/shared_nfs/kyle/test/Hipkittens2` (只读参考)

## 当前成绩
- **我们**: **19/42 WIN** (warmup=200, iters=500, 62 variants)
- **Cursor**: 16/42 WIN (同参数, 25 variants)
- **我们领先 3 WIN**
- **Avg ratio**: 100.8%
- **Auto-tune variants**: 62 (已饱和，详见下方)

## 已做的优化 (12项)
1. Store block reorder (A0Bl,A0Br,A1Bl,A1Br) — +0.8%
2. MFMA operand SWAP — 正确但慢, auto-tune 选项
3. TAIL_SPLIT=1 — 小K帮助, 大K退化, auto-tune 选项
4. SPREAD_LDS=1 — 退化1-2%, auto-tune 选项
5. NONVOLATILE_SCALE_X2_POC=1 (default ON) — +1-2%
6. STEP3_BARRIER_VMCNT (4,8,12,16) — 不同 shapes 最优不同
7. PF_N=4 (STEP3_PF_N/STEP4_PF_N) — 减少 prefetch 深度
8. STEP4_EXTERNAL_BR_PREFETCH — Br PF 分离
9. STEP12_BR_LGKMCNT (0,2,4) — Step1→Step2 lgkmcnt 放松
10. STEP3_EMBED_BARRIER (0,1) — barrier 独立/嵌入选择
11. 62-variant auto-tune (全组合)
12. gl.cuh size_t overflow fix

## Auto-tune 空间已饱和 (本轮验证)
以下全部测试过，无法翻转任何 LOSE shape:
- 62-variant 全量 benchmark (19/42 WIN, 稳定)
- Cross-product flag stacking (24 个新交叉组合 → 全部无效)
- UNROLL_K=1,2,4 (比编译器默认差)
- Fine-grained VMCNT=6,10,14,16,18,20,24 (不如 ts_lgk2)
- Fine-grained LGKMCNT=1,3,6 (LGKMCNT=2 最优)
- Cursor 无新思路 (14 个新 commit, 我们是超集)

## 不要再做的事
- **Direct-B (不preshuffle)**: 正确但慢28%
- **BK=256**: LDS装不下 (256KB > 160KB max)
- **Preshuffle-B 1-pass/2-pass**: spill/慢56%
- **sched_group_barrier / iglp_opt**: 无改善
- **ds_bpermute wide stores**: 退化16%
- **GROUP_SIZE_M=32/64**: 大N退化
- **UNROLL_K=1,2,4**: 比默认差
- **Cross-product flag stacking**: 无效
- **Fine-grained VMCNT/LGKMCNT**: 已穷举
- 把 preshuffle 时间不算进比较

## 近阈值 shapes (最接近翻WIN)
| Shape | Ratio | Best Variant | 差距 |
|-------|-------|-------------|------|
| 6144×32768×4096 | 99.5% | ts_lgk2 | 0.5% |
| 32768×28672×2048 | 99.1% | ts_gm2_v12 | 0.9% |
| 4096×14336×8192 | 98.9% | lgk2 | 1.1% |
| 6144×4096×16384 | 98.8% | ts_v4 | 1.2% |
| 4096×32768×4096 | 98.6% | ts_gm2_v12 | 1.4% |
| 16384×4096×14336 | 98.4% | ts_pf4 | 1.6% |
| 28672×4096×8192 | 97.5% | gm8_v12 | 2.5% |

## 结构性限制
- B走LDS是根本瓶颈: +18% read traffic, 2x wait time vs aiter
- 256 AGPR + B tiles 无法同时放进 256 VGPRs
- 不 preshuffle B 就不能跳过 LDS
- N=32768 shapes: B tile 大 → LDS traffic 成为瓶颈
- Store epilogue: 非SWAP路径无法 pack bf16 stores

## 可能的未来方向 (高风险/高工作量)
1. **Fused Step34** — 合并 Step3+Step4 为单个 asm block, 消除调度gap (~200行asm). 预期 1-2%
2. **Profile-guided** — rocprof 精确定位 stall
3. **Pre-shuffle B** — 唯一根本解决 B-LDS 的方案

## Benchmark 规则
- **warmup=200, iters=500**, trimmed mean 10%
- GPU 1-7 可用 (`HIP_VISIBLE_DEVICES=N`)
- `rocm-smi --showuse` 确认 GPU 空闲
- 所有 benchmark 结论必须标注 warmup/iters

## 关键文件
- `kernel_mxfp4_gluon_cpp.cpp` — 主内核 (62 auto-tune flags)
- `bench_all_42.py` — 42-shape benchmark (62 variants, sequential)
- `bench_all42_parallel.py` — 42-shape benchmark (parallel, needs pre-built .so)
- `build_all42_parallel.py` — 并行编译器 (62 variants × 26 N,K pairs)
- `spot_test.py` — 单shape多variant测试 (63 variants)
- `bench_all42_results.json` — 最新结果 (19/42 WIN)
