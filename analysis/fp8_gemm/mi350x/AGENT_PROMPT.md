# MXFP4 GEMM 优化 — Agent提示词

你在继续推进 `HipKittens` 的 MXFP4 GEMM 优化工作，跟 Cursor (Hipkittens2) 竞赛。

## 项目位置
- **我们的 Repo**: `/shared_nfs/kyle/test/HipKittens`
- **Branch**: `mxfp4`
- **工作目录**: `analysis/fp8_gemm/mi350x`
- **Cursor Repo**: `/shared_nfs/kyle/test/Hipkittens2` (只读参考)

## 当前成绩
- **我们**: **19/42 WIN** (warmup=200, iters=500)
- **Cursor**: 17/42 WIN (同参数)
- **我们领先 2 WIN**
- **Avg ratio**: 100.4%
- **Target shape** 4096×32768×128256: 我们 5113T (88.4%), Cursor ~5089T (88.0%)

## 已做的优化
1. Store block reorder (A0Bl,A0Br,A1Bl,A1Br) — +0.8%
2. MFMA operand SWAP — 正确但慢, auto-tune 选项
3. TAIL_SPLIT=1 — 小K帮助, 大K退化, auto-tune 选项
4. SPREAD_LDS=1 — 退化1-2%, auto-tune 选项
5. NONVOLATILE_SCALE_X2_POC=1 (default ON) — +1-2% (scale load 非易失性)
6. STEP3_BARRIER_VMCNT=12 — large-N shapes 有收益
7. PF_N=4 (STEP3_PF_N/STEP4_PF_N) — 减少 prefetch 深度
8. STEP4_EXTERNAL_BR_PREFETCH — Br PF 分离, 翻 WIN 4096×4096×32768
9. 41-variant auto-tune (全组合)
10. gl.cuh size_t overflow fix — 大shape int溢出修复

## 不要再做的事
- **Direct-B (不preshuffle)**: 正确但慢28% (buffer_load延迟)
- **BK=256**: LDS装不下 (256KB > 160KB max)
- **Preshuffle-B 1-pass**: VGPR spill → NaN
- **Preshuffle-B 2-pass**: 正确但慢56% (K-loop跑两遍)
- **sched_group_barrier / iglp_opt**: 无改善
- **ds_bpermute wide stores**: 退化16%
- **GROUP_SIZE_M=32/64**: 大N shapes 退化10-16pp (已验证)
- 把 preshuffle 时间不算进比较 — 用户明确拒绝过

## 结构性限制
- B走LDS是根本瓶颈: +18% read traffic, 2x wait time vs aiter
- 256 AGPR (4 acc blocks) + B tiles 无法同时放进 256 VGPRs
- 不 preshuffle B 就不能跳过 LDS
- LDS swizzle 是 MFMA 需要的数据排列 (非 bank conflict avoidance)
- buffer_load 200cy vs ds_read 20cy → direct loading 总是更慢
- N=32768 shapes: B tile 大 → LDS traffic 成为瓶颈

## 近阈值 shapes (可能翻WIN)
| Shape | Ratio | Best Variant | 差距 |
|-------|-------|-------------|------|
| 32768×28672×2048 | 99.2% | ts_gm2 | 0.8% |
| 6144×32768×4096 | 98.9% | ts_v12 | 1.1% |
| 4096×32768×4096 | 98.4% | ts_gm2 | 1.6% |
| 6144×4096×16384 | 98.3% | v12 | 1.7% |
| 16384×4096×14336 | 98.2% | ts_v12 | 1.8% |
| 4096×14336×8192 | 98.1% | default | 1.9% |

## 优先方向
1. **Per-shape 精准调参** — 针对 98-99% 的 shapes 尝试更多组合
2. **lgkmcnt relaxation** — Step1→Step2 的 lgkmcnt 可能有微小空间
3. **结构性 gap** — 12个 >5% gap shapes 是 B-LDS 瓶颈，不做 preshuffle 无法突破

## Benchmark 规则
- **warmup=200, iters=500**, trimmed mean 10%
- GPU 1-4 可用 (`HIP_VISIBLE_DEVICES=1,2,3,4`)
- `rocm-smi --showuse` 确认 GPU 空闲
- 所有 benchmark 结论必须标注 warmup/iters

## 关键文件
- `kernel_mxfp4_gluon_cpp.cpp` — 主内核 (41 auto-tune flags)
- `bench_all_42.py` — 42-shape benchmark (41 variants, sequential)
- `bench_all42_parallel.py` — 42-shape benchmark (parallel, needs pre-built .so)
- `spot_test.py` — 单shape多variant测试 (42 variants)
- `bench_all42_results.json` — 最新结果 (19/42 WIN)
