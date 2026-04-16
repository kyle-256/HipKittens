# MXFP4 GEMM Optimization TODO

## Current State (2026-04-16)
- **Repo**: `/shared_nfs/kyle/test/HipKittens`
- **Branch**: `mxfp4`
- **42-shape result**: **19/42 WIN** (warmup=200, iters=500, 4-GPU parallel)
- **上一轮**: 18/42 WIN → 19/42 WIN (+1: 4096×4096×32768 via ts_ext_br)
- **Cursor (Hipkittens2)**: 17/42 WIN (同参数)
- **Target shape**: 4096×32768×128256 = 5113T (88.4% of comp 5781T)
- **Auto-tune variants**: 41 (33 base + 8 new: ext_br, gm32/64, ts_gm16/32)

## Recent Commits
```
4820d632 STEP4_EXTERNAL_BR_PREFETCH + expanded auto-tune → 19/42 WIN
887bbad9 NVS=1 default + expanded auto-tune (32 variants) + fixes
80867f26 update TODO + AGENT_PROMPT for machine handoff
c2873f92 direct-B kernel POC — correct but 28% slower (buffer_load latency)
```

## 已做的优化
1. **Store block reorder** (A0Bl,A0Br,A1Bl,A1Br) — +0.8% 全局
2. **SWAP operand port** from HK2 — 正确但略慢，作为 auto-tune 选项
3. **TAIL_SPLIT=1** — 小K(≤4096)帮助+1-2%, 大K(≥7168)退化
4. **SPREAD_LDS=1** (rowspread ds_reads) — 退化1-2%，作为选项
5. **NONVOLATILE_SCALE_X2_POC=1** (default ON) — 非易失性 scale loads, +1-2%
6. **STEP3_BARRIER_VMCNT=12** — large-N shapes 有收益，auto-tune 选项
7. **PF_N=4** (STEP3_PF_N/STEP4_PF_N) — 减少 prefetch 深度，部分 shapes 有益
8. **STEP4_EXTERNAL_BR_PREFETCH** — Br prefetch 从 Step4 MFMA 中分离，翻 WIN 1个 shape
9. **32→41-variant auto-tune** (GM, U, SWAP, TS, V12, SPREAD, PF4, NVS, EXT_BR, GM32/64)
10. **gl.cuh size_t overflow fix** — 大shape (128256×32768) int溢出修复
11. **SWAP+TAIL_SPLIT barrier cleanup** — 简化尾部 barrier emit

## 19 WIN shapes
| Shape | TFLOPS | Ratio | Best Variant |
|-------|--------|-------|-------------|
| 16384×4096×2048 | 3128 | 104.4% | ts_v12 |
| 16384×4096×3072 | 3693 | 105.7% | ts_v12 |
| 16384×6144×2048 | 3304 | 108.4% | ts_v12 |
| 32768×4096×2048 | 3290 | 105.1% | v12 |
| 32768×4096×3072 | 3815 | 105.1% | ts_v12 |
| 32768×6144×2048 | 3385 | 104.5% | ts_gm8_v12 |
| 16384×14336×2048 | 3403 | 103.1% | ts_v12 |
| 32768×14336×2048 | 3398 | 101.4% | ts_gm2 |
| 4096×4096×16384 | 4843 | 104.3% | ts_v12 |
| 4096×4096×8192 | 4349 | 109.8% | u16 |
| 4096×4096×32768 | 5226 | 101.4% | **ts_ext_br** (NEW) |
| 4096×6144×32768 | 4558 | 120.4% | u16 |
| 4096×128256×32768 | 5122 | 160.3% | v12 |
| 6144×4096×8192 | 3887 | 101.7% | v12 |
| 16384×4096×4096 | 4059 | 102.7% | ts_v12 |
| 16384×4096×6144 | 4499 | 105.6% | ts_v12 |
| 16384×4096×7168 | 4527 | 101.9% | v12 |
| 16384×6144×4096 | 4213 | 104.2% | ts_v12 |
| 16384×14336×4096 | 4264 | 100.2% | ts_v12 |

## 23 LOSE shapes 分析
### 接近 WIN (95-99.9%)
| Shape | TFLOPS | Ratio | Best | 差距 |
|-------|--------|-------|------|------|
| 4096×4096×32768 | 5146→5226 | 99.9→101.4% | ts_ext_br | **已翻WIN** |
| 32768×28672×2048 | 3326 | 99.2% | ts_gm2 | 0.8% |
| 6144×32768×4096 | 4245 | 98.9% | ts_v12 | 1.1% |
| 4096×32768×4096 | 4099 | 98.4% | ts_gm2 | 1.6% |
| 6144×4096×16384 | 4354 | 98.3% | v12 | 1.7% |
| 16384×4096×14336 | 5051 | 98.2% | ts_v12 | 1.8% |
| 4096×14336×8192 | 4264 | 98.1% | default | 1.9% |
| 28672×4096×8192 | 4684 | 97.4% | gm8_v12 | 2.6% |
| 32768×4096×7168 | 4504 | 96.5% | ts_gm8_v12 | 3.5% |
| 4096×14336×16384 | 4800 | 95.7% | ts_v12 | 4.3% |
| 16384×28672×2048 | 3297 | 94.7% | ts_gm2 | 5.3% |

### 结构性 gap (>5%)
| Shape | Best | Ratio | 限制因素 |
|-------|------|-------|---------|
| 4096×32768×128256 | 5113T | 88.4% | 超大K, B-LDS瓶颈 |
| 14336×4096×32768 | 4627T | 88.2% | 大K, B-LDS瓶颈 |
| 16384×4096×28672 | 4954T | 89.7% | 大K, B-LDS瓶颈 |
| 128256×32768×4096 | 4139T | 91.2% | 超大M, XCD dispatch |
| 28672×32768×4096 | 4077T | 91.3% | N=32768, B-LDS |
| 4096×32768×6144 | 4162T | 91.5% | N=32768 |
| 4096×32768×28672 | 5166T | 92.8% | N=32768, 大K |
| 14336×32768×4096 | 4137T | 92.7% | N=32768 |
| 28672×4096×16384 | 4972T | 92.9% | 大M大K |
| 16384×28672×4096 | 4110T | 93.2% | N=28672 |
| 4096×32768×14336 | 4936T | 93.2% | N=32768 |
| 32768×4096×14336 | 4884T | 93.5% | M=32768 |

## 结构性限制 (不preshuffle B 无法突破)
- **B走LDS**: 比aiter多18% TCP read traffic, 2x Frac_Wait_Any
- **256 AGPR**: 4 acc blocks 占满, B tile 数据必须在 256 VGPR 内
- **LDS swizzle**: MFMA 需要的数据排列 (已证明 identity)
- **buffer_load vs ds_read**: 10x延迟差距, LDS prefetch pipeline 完全隐藏
- **N=32768 shapes**: B tile 大 → LDS traffic 成为瓶颈

## 不要再做的事
- **Direct-B (不preshuffle)**: 正确但慢28% (buffer_load延迟)
- **BK=256**: LDS装不下 (256KB > 160KB max)
- **Preshuffle-B 1-pass**: VGPR spill → NaN
- **Preshuffle-B 2-pass**: 正确但慢56% (K-loop跑两遍)
- **sched_group_barrier / iglp_opt**: 无改善
- **ds_bpermute wide stores**: 退化16%
- **GROUP_SIZE_M=32/64**: 大N shapes 退化10-16pp
- 把 preshuffle 时间不算进比较 — 用户明确拒绝过

## 优先方向 (如果继续优化)
1. **近阈值 shapes 精准调参** — 10个 shapes 在 95-99.2%，可能通过更多组合翻WIN
2. **lgkmcnt relaxation** — Step1→Step2 的 lgkmcnt(0) 可能有 0.5% 空间（需验证）
3. **两个 kernel 已基本相同** — 经逐行对比确认, 剩余差距是结构性的

## Benchmark Rules
- **warmup=200, iters=500**, trimmed mean 10%
- 用空闲GPU (`rocm-smi` 确认0%)
- `HIP_VISIBLE_DEVICES=N`
- MI355X 上 competitor_tflops 是正确 baseline

## 关键文件
| 文件 | 用途 |
|------|------|
| `kernel_mxfp4_gluon_cpp.cpp` | 主生产内核 (NVS=1, 41 variants) |
| `bench_all_42.py` | 42-shape benchmark (41 auto-tune variants, sequential) |
| `bench_all42_parallel.py` | 42-shape benchmark (parallel across GPUs) |
| `spot_test.py` | 单shape多variant测试 (42 variants) |
| `bench_all42_results.json` | 最新42-shape结果 (19/42 WIN) |

## 环境设置 (换机器必读)
```bash
cd /shared_nfs/kyle/test/HipKittens
git checkout mxfp4
cd analysis/fp8_gemm/mi350x

# 用GPU 1-4跑benchmark (parallel)
python3 bench_all42_parallel.py 1,2,3,4

# 用GPU 1跑benchmark (sequential, handles compilation)
HIP_VISIBLE_DEVICES=1 python3 bench_all_42.py

# 单shape测试
HIP_VISIBLE_DEVICES=2 python3 spot_test.py 4096 32768 128256 5781.1

# Cursor的仓库 (只读参考)
# /shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x/
```
