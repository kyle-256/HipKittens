# MXFP4 GEMM 优化 — Agent提示词

你在继续推进 `HipKittens` 的 MXFP4 GEMM 优化工作，跟 Cursor (Hipkittens2) 竞赛。

## 项目位置
- **我们的 Repo**: `/shared_nfs/kyle/test/HipKittens`
- **Branch**: `mxfp4`
- **工作目录**: `analysis/fp8_gemm/mi350x`
- **Cursor Repo**: `/shared_nfs/kyle/test/Hipkittens2` (只读参考)

## 当前成绩
- **我们**: **19/42 WIN** (warmup=200, iters=500, 66 variants)
- **Cursor**: 16/42 WIN (同参数, 25 variants)
- **我们领先 3 WIN**
- **Avg ratio**: 100.8%
- **Auto-tune variants**: 95 (已饱和，所有交叉积均已穷举)

## 已做的优化 (14项)
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
11. TAIL_BARRIER_VMCNT (0,4,16) — 尾部K迭代单独barrier VMCNT (ts_tv16在2shapes最优)
12. GM×LGK cross-products — gm8_lgk2, ts_gm2_lgk2, ts_gm2_lgk2_v12 (边际0.1-0.3pp)
13. 95-variant auto-tune (全组合穷举)
14. gl.cuh size_t overflow fix

## Auto-tune 空间已饱和 (多轮验证)
以下全部测试过，无法翻转任何 LOSE shape:
- 62-variant 全量 benchmark (19/42 WIN, 稳定)
- Cross-product flag stacking (24 个新交叉组合 → 全部无效)
- UNROLL_K=1,2,4 (比编译器默认差)
- Fine-grained VMCNT=6,10,14,16,18,20,24 (不如 ts_lgk2)
- Fine-grained LGKMCNT=1,3,6 (LGKMCNT=2 最优)
- LGKMCNT×VMCNT cross-products: 15 个新组合 (lgk{2,4}×v{4,12,16}×ts×no_embed) → 全部无效
- FUSED_STEP34: 14 variants → 全面退化 ~7%
- ASM rewriter: s_nop removal → 破坏正确性(m0 hazard), 且无性能增益 (+0.15% = noise)
- PF_N=1/2: 减少 prefetch 深度 → 全面退化 2-6%
- Asymmetric PF (STEP3_PF_N ≠ STEP4_PF_N): 2/8, 8/2 全部不如对称 PF
- GM×LGK cross-products (30 new variants on 7 near-threshold shapes): 边际0.1-0.3pp, 不翻 WIN
- TAIL_BARRIER_VMCNT (0,4,16): ts_tv16 部分 shapes 边际最优 (<0.5pp), 不翻 WIN
- Cursor 无新思路 (packed wide-store, permlane store POC, 均无增益)

## 不要再做的事
- **Direct-B (不preshuffle)**: 正确但慢28%
- **BK=256**: LDS装不下 (256KB > 160KB max)
- **Preshuffle-B 1-pass/2-pass**: spill/慢56%
- **sched_group_barrier / iglp_opt**: 无改善
- **ds_bpermute wide stores**: 退化16%
- **GROUP_SIZE_M=32/64**: 大N退化
- **UNROLL_K=1,2,4**: 比默认差
- **Cross-product flag stacking**: 全部穷举, 无效
- **Fine-grained VMCNT/LGKMCNT**: 已穷举
- **LGKMCNT×VMCNT cross-products**: 全15种组合 → 无效或退化
- **FUSED_STEP34**: Step3+Step4 合并 → 退化 ~7%
- **ASM rewriter (s_nop/PF redistribution)**: 移除 s_nop 破坏正确性, inter-block code 被 MFMA pipeline 完全隐藏, 无增益. Cursor rewriter 结构不适用
- **PF_N=1/2**: 减少 PF 深度 → 退化 2-6%
- **Asymmetric PF**: 不对称 Step3/Step4 PF 深度 → 无效
- **GM×LGK/TAIL_VMCNT cross-products**: 边际改进不翻WIN
- 把 preshuffle 时间不算进比较

## 近阈值 shapes (最接近翻WIN)
| Shape | Ratio | Best Variant | 差距 |
|-------|-------|-------------|------|
| 6144×32768×4096 | 99.5% | ts_lgk2 | 0.5% |
| 32768×28672×2048 | 99.1-99.2% | ts_gm2_lgk2_v12 | 0.8% |
| 4096×14336×8192 | 98.6-98.9% | lgk2 | 1.1% |
| 6144×4096×16384 | 97.6-98.8% | default/ts_v4 | 1.2% |
| 4096×32768×4096 | 98.1-98.6% | ts_gm2_lgk2 | 1.4% |
| 16384×4096×14336 | 98.4-98.7% | ts_tv16 | 1.3% |
| 28672×4096×8192 | 97.5-97.7% | gm8_lgk2 | 2.3% |

## 结构性限制
- B走LDS是根本瓶颈: +18% read traffic, 2x wait time vs aiter
- 256 AGPR + B tiles 无法同时放进 256 VGPRs
- 不 preshuffle B 就不能跳过 LDS
- N=32768 shapes: B tile 大 → LDS traffic 成为瓶颈
- Store epilogue: 非SWAP路径无法 pack bf16 stores

## 可能的未来方向 (高风险/高工作量)
1. ~~**Fused Step34**~~ — DEAD END (退化7%)
2. ~~**ASM rewriter**~~ — DEAD END (无增益, s_nop 硬件强制)
3. ~~**Profile-guided**~~ — DONE: s_nop=1.4%, waits=3.4%, epilogue=10.5%, 瓶颈在 B-LDS traffic
4. ~~**GM×LGK / TAIL_VMCNT cross-products**~~ — DEAD END (边际0.1-0.3pp, 不翻WIN)
5. **Pre-shuffle B** — 唯一根本解决 B-LDS 的方案, 需要 Python 层 preshuffle pass

## Benchmark 规则
- **warmup=200, iters=500**, trimmed mean 10%
- GPU 1-7 可用 (`HIP_VISIBLE_DEVICES=N`)
- `rocm-smi --showuse` 确认 GPU 空闲
- 所有 benchmark 结论必须标注 warmup/iters

## 关键文件
- `kernel_mxfp4_gluon_cpp.cpp` — 主内核 (auto-tune flags)
- `bench_all_42.py` — 42-shape benchmark (95 variants, sequential)
- `bench_all42_parallel.py` — 42-shape benchmark (66 variants, parallel, needs pre-built .so)
- `build_all42_parallel.py` — 并行编译器 (93 variants × 26 N,K pairs)
- `spot_test.py` — 单shape多variant测试 (95 variants)
- `spot_new_variants.py` — 新variant快速spot测试 (30 new + 9 reference)
- `bench_all42_results.json` — 最新结果 (19/42 WIN)
