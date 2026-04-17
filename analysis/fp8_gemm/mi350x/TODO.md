# MXFP4 GEMM Optimization TODO

## Current State (2026-04-17)
- **Repo**: `/shared_nfs/kyle/test/HipKittens`
- **Branch**: `mxfp4`
- **42-shape result**: **24/42 WIN** (warmup=200, iters=500, 4-GPU parallel, 115 variants)
- **上一轮**: 19/42 WIN → 24/42 WIN (+5 LOSE→WIN flip in Round 1, 0 regressions)
- **Cursor (Hipkittens2)**: 16/42 WIN (同参数, 2026-04-16T06:26)
- **我们领先**: **8 WIN**
- **Auto-tune variants**: 115 (expanded from 95 with memclause family + Optimizer A discoveries + ceiling sweep)
- **Saturation**: Round 2 (+28 ceiling variants) + deep-LOSE分析员 (+44 targeted variants on 10 stuck shapes) BOTH yielded **+0 LOSE→WIN flip**

## Recent Commits
```
f886d940 MXFP4: Round 1 final 24/42 WIN (+5 vs baseline 19/42)
06a416a8 TAIL_BARRIER_VMCNT + GM×LGK cross-products → 95 variants
03d3ba0a 62-variant auto-tune space saturated (docs)
78593c0c STEP12_BR_LGKMCNT tunable + expanded auto-tune (62 variants)
80b736b5 update docs + results for 19/42 WIN
4820d632 STEP4_EXTERNAL_BR_PREFETCH + expanded auto-tune → 19/42 WIN
```

## Round 1 LOSE→WIN flips (+5)
| Shape | Best Variant | Before → After |
|-------|-------------|----------------|
| 32768×28672×2048 | ts_gm2_v12_memc_dc | 99.1% → 100.3% |
| 4096×14336×8192 | u8 | 98.9% → 101.0% |
| 4096×32768×4096 | ts_no_embed_tv16_memc | 98.6% → 103.0% |
| 6144×32768×4096 | ts_v4_memc | 99.5% → 101.1% |
| 16384×4096×14336 | ts_v12_tv0_memc | 98.4% → 101.7% |

memclause family (`-mllvm -amdgpu-sched-strategy=max-memory-clause`) is the dominant new winner — appears in 14/24 WIN best variants.

## 已做的优化
1. **Store block reorder** (A0Bl,A0Br,A1Bl,A1Br) — +0.8% 全局
2. **SWAP operand port** from HK2 — 正确但略慢，作为 auto-tune 选项
3. **TAIL_SPLIT=1** — 小K(≤4096)帮助+1-2%, 大K(≥7168)退化
4. **SPREAD_LDS=1** (rowspread ds_reads) — 退化1-2%，作为选项
5. **NONVOLATILE_SCALE_X2_POC=1** (default ON) — 非易失性 scale loads, +1-2%
6. **STEP3_BARRIER_VMCNT** (4,8,12,16) — 不同 shapes 最优不同
7. **PF_N=4** (STEP3_PF_N/STEP4_PF_N) — 减少 prefetch 深度
8. **STEP4_EXTERNAL_BR_PREFETCH** — Br prefetch 从 Step4 MFMA 中分离
9. **STEP12_BR_LGKMCNT** (0,2,4) — Step1→Step2 lgkmcnt 放松
10. **STEP3_EMBED_BARRIER** (0,1) — barrier 独立/嵌入 Step3
11. **62-variant auto-tune** (GM, U, SWAP, TS, V4/8/12/16, SPREAD, PF4, NVS, EXT_BR, LGKMCNT, NO_EMBED, 多轴交叉)
12. **gl.cuh size_t overflow fix** — 大shape (128256×32768) int溢出修复
13. **TAIL_BARRIER_VMCNT** — 尾部K迭代单独barrier VMCNT调优 (ts_tv16 在2个shapes上最优)
14. **GM×LGK cross-products** — gm8_lgk2, ts_gm2_lgk2, ts_gm2_lgk2_v12 (提供 0.1-0.3pp 边际提升)

## 24 WIN shapes (115-variant benchmark, Round 2 final)
| Shape | TFLOPS | Ratio | Best Variant |
|-------|--------|-------|-------------|
| 16384×4096×2048 | 3264 | 109.0% | ts_v12_tv0_memc |
| 16384×4096×3072 | 3802 | 108.9% | ts_pf6_6_lgk2_memc |
| 16384×6144×2048 | 3494 | 114.6% | ts_v4_tv0 |
| 32768×4096×2048 | 3465 | 110.6% | ts_v12_tv0_memc |
| 32768×4096×3072 | 3927 | 108.2% | ts_lgk2_memc |
| 32768×6144×2048 | 3428 | 105.8% | ts_gm8_v12 |
| 16384×14336×2048 | 3553 | 107.6% | ts_lgk2_v20 |
| 32768×14336×2048 | 3478 | 103.8% | ts_gm2_v12_memc |
| **32768×28672×2048** | 3365 | **100.3%** | **ts_gm2_v12_memc_dc** ← R1 flip |
| 16384×4096×4096 | — | 106.3% | ts_lgk2_memc_dc |
| 16384×6144×4096 | — | 106.4% | ts_v12_tv0_memc |
| 16384×14336×4096 | — | 104.3% | ts_lgk2_v20_memc |
| 4096×4096×8192 | — | 110.9% | default |
| **4096×14336×8192** | — | **101.0%** | **u8** ← R1 flip |
| 6144×4096×8192 | — | 102.7% | v32 |
| **4096×32768×4096** | — | **103.0%** | **ts_no_embed_tv16_memc** ← R1 flip |
| **6144×32768×4096** | — | **101.1%** | **ts_v4_memc** ← R1 flip |
| 16384×4096×6144 | — | 107.7% | ts_lgk2_v20_memc |
| 16384×4096×7168 | — | 104.7% | ts_lgk2_v20_memc |
| 4096×4096×16384 | — | 104.4% | ts_v24 |
| **16384×4096×14336** | — | **101.7%** | **ts_v12_tv0_memc** ← R1 flip |
| 4096×4096×32768 | — | 100.9% | ts |
| 4096×6144×32768 | — | 124.3% | v16 |
| 4096×128256×32768 | — | 161.2% | memc |

## 18 LOSE shapes (Round 2 final)
### Near-threshold (≥98%, 3 shapes)
| Shape | Ratio | Best | Δ to WIN |
|-------|-------|------|----------|
| 32768×4096×7168 | 99.3% | ts_gm8_v12 | 0.7pp |
| 4096×14336×16384 | 98.6% | ts_lgk2 | 1.4pp |
| 6144×4096×16384 | 98.6% | ts_lgk2 | 1.4pp |

### Mid-LOSE (95-97%, 5 shapes)
| Shape | Ratio | Best |
|-------|-------|------|
| 16384×28672×2048 | 96.4% | ts_gm2_v12_memc_dc |
| 4096×32768×6144 | 96.4% | ts_pf4_memc |
| 16384×28672×4096 | 96.5% | ts_gm2_v12_memc |
| 28672×4096×8192 | 96.6% | ts_lgk2_memc_dc |
| 14336×32768×4096 | 96.9% | ts_v12_tv0_memc |

### Deep-LOSE (<95%, 10 shapes — confirmed STRUCTURAL by deep-LOSE分析员)
| Shape | Ratio | Best | 类别 |
|-------|-------|------|------|
| 4096×32768×128256 | 88.3% | ts_gm8 | mega-K + 大N |
| 14336×4096×32768 | 89.6% | lgk2_dc | 大K + 大M |
| 16384×4096×28672 | 90.1% | u32 | 大K + 大M |
| 128256×32768×4096 | 92.9% | ts_gm2_v12_memc_dc | mega-M+N |
| 4096×32768×28672 | 93.7% | v20_memc | 大K + 大N |
| 28672×4096×16384 | 93.7% | ts_gm8 | 大K + 大M |
| 4096×28672×32768 | 94.1% | u16 | 大K + 大N |
| 32768×4096×14336 | 94.1% | ts_gm8_v12 | 大K + 大M |
| 4096×32768×14336 | 94.5% | ts_lgk2_memc | 大K + 大N |
| 28672×32768×4096 | 94.5% | ts_lgk2_v12_memc | 大M+N |

## DEPRECATED — 19 WIN shapes (62-variant benchmark, pre-Round 1)
| Shape | TFLOPS | Ratio | Best Variant |
|-------|--------|-------|-------------|
| 16384×4096×2048 | 3147 | 105.1% | ts_no_embed_v12 |
| 16384×4096×3072 | 3698 | 105.9% | ts_lgk4 |
| 16384×6144×2048 | 3301 | 108.3% | ts_v16 |
| 32768×4096×2048 | 3330 | 106.3% | ts_no_embed_v12 |
| 32768×4096×3072 | 3823 | 105.3% | ts_no_embed |
| 32768×6144×2048 | 3379 | 104.3% | ts_gm8 |
| 16384×14336×2048 | 3408 | 103.2% | ts_v12 |
| 32768×14336×2048 | 3407 | 101.7% | ts_gm2_v12 |
| 4096×4096×16384 | 4858 | 104.6% | ts_lgk2_v12 |
| 4096×4096×8192 | 4353 | 109.9% | u8 |
| 4096×4096×32768 | 5158 | 100.1% | ts_no_embed_v12 |
| 4096×6144×32768 | 4645 | 122.8% | u16 |
| 4096×128256×32768 | 5146 | 161.1% | default |
| 6144×4096×8192 | 3906 | 102.2% | u8 |
| 16384×4096×4096 | 4061 | 102.8% | ts_no_embed |
| 16384×4096×6144 | 4510 | 105.9% | ts_v16 |
| 16384×4096×7168 | 4575 | 103.0% | lgk2 |
| 16384×6144×4096 | 4221 | 104.4% | ts_u16 |
| 16384×14336×4096 | 4273 | 100.4% | ts_lgk2_v12 |

## 23 LOSE shapes 分析
### 接近 WIN (97-99.5%)
| Shape | TFLOPS | Ratio | Best | 差距 |
|-------|--------|-------|------|------|
| 6144×32768×4096 | 4268 | 99.5% | ts_lgk2 | 0.5% |
| 32768×28672×2048 | 3325 | 99.1% | ts_gm2_v12 | 0.9% |
| 4096×14336×8192 | 4296 | 98.9% | lgk2 | 1.1% |
| 6144×4096×16384 | 4377 | 98.8% | ts_v4 | 1.2% |
| 4096×32768×4096 | 4110 | 98.6% | ts_gm2_v12 | 1.4% |
| 16384×4096×14336 | 5062 | 98.4% | ts_pf4 | 1.6% |
| 28672×4096×8192 | 4688 | 97.5% | gm8_v12 | 2.5% |

### 中等差距 (95-97%)
| Shape | TFLOPS | Ratio | Best |
|-------|--------|-------|------|
| 32768×4096×7168 | 4503 | 96.5% | gm8_ext_br |
| 4096×14336×16384 | 4830 | 96.3% | ts_v4 |

### 结构性 gap (>5%)
| Shape | Best | Ratio | 限制因素 |
|-------|------|-------|---------|
| 16384×28672×2048 | 3275 | 94.1% | 大N, B-LDS瓶颈 |
| 4096×32768×128256 | 5115 | 88.5% | 超大K, B-LDS瓶颈 |
| 14336×4096×32768 | 4671 | 89.1% | 大K, B-LDS瓶颈 |
| 16384×4096×28672 | 4967 | 89.9% | 大K, B-LDS瓶颈 |
| 128256×32768×4096 | 4129 | 91.0% | 超大M, XCD dispatch |
| 28672×32768×4096 | 4089 | 91.5% | N=32768, B-LDS |
| 4096×32768×6144 | 4193 | 92.2% | N=32768 |
| 4096×32768×28672 | 5194 | 93.3% | N=32768, 大K |
| 16384×28672×4096 | 4116 | 93.3% | N=28672 |
| 4096×32768×14336 | 4946 | 93.4% | N=32768 |
| 28672×4096×16384 | 5000 | 93.4% | 大M大K |
| 14336×32768×4096 | 4183 | 93.7% | N=32768 |
| 4096×28672×32768 | 5297 | 93.8% | N=28672, 大K |
| 32768×4096×14336 | 4905 | 93.9% | M=32768 |

## Auto-tune 空间已饱和 — 多轮验证
1. **62-variant 全量benchmark**: 41→62 variants, 结果稳定 19/42 WIN
2. **Cross-product spot tests**: 在 98-99% shapes 上测试 24 个新交叉组合 → 全部无效, flag stacking counterproductive
3. **UNROLL_K=1,2,4**: 全部比编译器默认差
4. **Fine-grained VMCNT (6,10,14,16,18,20,24)**: 全部不如 ts_lgk2 (99.5%)
5. **Fine-grained LGKMCNT (1,3,6)**: LGKMCNT=2 是最优, 其他值更差
6. **Cursor对比**: Cursor 14 个新 commit 无新思路, 我们变体超集
7. **LGKMCNT×VMCNT cross-products (15 new)**: lgk2_v4/v12/v16, lgk4_v4/v12/v16, ts_lgk2_v4/v16, ts_lgk4_v4/v12/v16, lgk2_no_embed, ts_lgk2_no_embed, ts_lgk2_no_embed_v12 → 全部无效或退化. 6144×4096×16384 best=99.3% (ts_lgk2_v4), 16384×4096×14336 best=98.2% (ts_lgk2)
8. **FUSED_STEP34 (14 variants)**: 合并 Step3+Step4 为 64-MFMA 单块. 全面退化 ~7% (f34 系列). ds_read/PF 冲突, 大块内编译器 MFMA 调度过激
9. **TAIL_BARRIER_VMCNT spot tests (30 new variants on 7 shapes)**: PF_N=2/1, PF4×LGK2, EXT_BR×LGK2, GM×LGK, TAIL_VMCNT, asymmetric PF — 全7个 near-threshold shapes 测试, 0 个 WIN flip. ts_tv16 在 2 shapes 上边际最优 (+0.3pp), gm8_lgk2/ts_gm2_lgk2/ts_gm2_lgk2_v12 各在1个shape上边际最优 (+0.1-0.2pp). PF_N=1/2 全面退化. 不对称PF无效

## 结构性限制 (不preshuffle B 无法突破)
- **B走LDS**: 比aiter多18% TCP read traffic, 2x Frac_Wait_Any
- **256 AGPR**: 4 acc blocks 占满, B tile 数据必须在 256 VGPR 内
- **LDS swizzle**: MFMA 需要的数据排列 (已证明 identity)
- **buffer_load vs ds_read**: 10x延迟差距, LDS prefetch pipeline 完全隐藏
- **N=32768 shapes**: B tile 大 → LDS traffic 成为瓶颈
- **Store epilogue**: 非SWAP路径 s[0..3] 是行连续, 无法pack成 dword stores
- **VMCNT/LGKMCNT gradient**: 99.5% 已是当前架构天花板

## 不要再做的事
- **Direct-B (不preshuffle)**: 正确但慢28% (buffer_load延迟)
- **BK=256**: LDS装不下 (256KB > 160KB max)
- **Preshuffle-B 1-pass**: VGPR spill → NaN
- **Preshuffle-B 2-pass**: 正确但慢56% (K-loop跑两遍)
- **sched_group_barrier / iglp_opt**: 无改善
- **ds_bpermute wide stores**: 退化16%
- **GROUP_SIZE_M=32/64**: 大N shapes 退化10-16pp
- **UNROLL_K=1,2,4**: 比编译器默认差
- **Cross-product flag stacking**: 98-99% shapes 上全部无效
- **Fine-grained VMCNT/LGKMCNT 微调**: 已穷举, 无增益
- **LGKMCNT×VMCNT cross-products**: lgk{2,4}×v{4,12,16}×ts×no_embed 全15种 → 无效或退化
- **FUSED_STEP34**: Step3+Step4 合并为 64-MFMA 块 → 全面退化 ~7% (14 variants tested)
- **ASM rewriter (s_nop removal)**: 后编译 ASM 重写移除 162 个 s_nop → 破坏正确性(26%元素错误, m0 hazard是硬件强制的), 且性能无变化(+0.15% = noise). Inter-block code 被 MFMA pipeline depth (16 cycles) 完全隐藏
- **ASM rewriter (PF redistribution)**: Cursor 的 rewriter 不适用(结构不同: 我们是 1×64-MFMA + 8×8-MFMA, Cursor 是 4×32-MFMA; 我们 K≤4096 全展开无 loop)
- **PF_N=1/2**: 减少 prefetch 深度 → 全面退化 2-6%. PF_N=4 是最优
- **Asymmetric PF (STEP3_PF_N ≠ STEP4_PF_N)**: 2/8, 8/2 全部不如对称 PF
- **GM×LGK cross-products**: 边际改进 0.1-0.3pp, 不足以 flip 任何 shape
- **TAIL_BARRIER_VMCNT tuning (0,4,16)**: ts_tv16 在部分 shapes 边际最优, 但 < 0.5pp 改进
- **Half-Direct Bl (DIRECT_BL)**: preshuffle B后, Bl 从 global buffer_load 直取 (跳过LDS). 0 spill (253 VGPRs, 87 SGPRs, 98KB LDS). 但 buffer_load 延迟 ~400 cycles, Step4 只有 ~128 cycles MFMA 隐藏 → 全 7 shapes 退化 8-15%. 大N shapes (32768) 退化 13-15%, 最差 -15.3% (6144×4096×16384). LDS 减少33% (131→98KB) 无法弥补 VMEM latency penalty
- **NT_STORE (non-temporal stores)**: 全部 global_store_short 加 `nt` modifier 绕过 L2. 结果: 全 7 近阈值 shapes 退化 15-25% (最差 -25%). CDNA L2 对 store coalescing 至关重要, 绕过 L2 = 直写 HBM = 慢. NT+SWAP 组合退化 34-72%
- **PACKED_STORE (bf16x2 dword stores)**: SWAP路径 store_block_inner 中用 pack_bf16x2 将 4×2B stores 合并为 2×4B stores. 需要 SWAP_STEP34_MAIN=1. SWAP 路径本身退化 13-34%, packed 不改善
- **Compiler flag tuning**: 测试 -O2, -mllvm -amdgpu-max-memory-clause=1/4, -mllvm -amdgpu-early-inline-all=true. 结果: 前两个shape ±0.4% noise, 6144×4096×16384 O2比O3好5%但仍远低于目标. 编译器调优无法翻WIN
- **18个未测variant组合**: ts_lgk2_ext_br, ts_lgk2_gm8, ts_lgk2_tv16/tv0, ts_gm2_lgk2_v4/v16/ext_br/no_embed, lgk2_ext_br/gm2/gm1/no_embed_v12 等 → 全部 0 WIN, 全部不如已有best variant
- 把 preshuffle 时间不算进比较 — 用户明确拒绝过

## 可能的未来方向 (高风险/高工作量)
1. ~~**Fused Step34**~~ — DEAD END, 退化 ~7%
2. ~~**ASM rewriter**~~ — DEAD END, s_nop 硬件强制, 无增益
3. ~~**Pre-shuffle B (Half-Direct Bl)**~~ — DEAD END, 8-15% 慢 (buffer_load latency)
4. ~~**rocprof 分析**~~ — DONE: 瓶颈在 B-LDS traffic, 不在 scheduling
5. ~~**NT_STORE (non-temporal stores)**~~ — DEAD END, 15-25% 慢 (L2 对 store coalescing 必要)
6. ~~**PACKED_STORE (bf16x2 dword stores)**~~ — DEAD END, SWAP路径必需, SWAP本身退化
7. ~~**Compiler flag tuning**~~ — DEAD END, ±0.4% noise
8. ~~**Extended variant combos (18 new)**~~ — DEAD END, 0 WINs

**结论 (2026-04-17 更新)**: 24/42 WIN 是当前内核架构 (B-through-LDS, 4-step K-loop, 256 AGPR) 的性能天花板. memclause + Round 2 ceiling sweep + deep-LOSE 分析员 (44 targeted variants) 全部确认: 18 个 LOSE shape 的剩余 gap 是结构性的, flag-axis 关不上.

**Round 2 (115 variants × 42 shapes) 与 deep-LOSE 分析员 (44 variants × 10 stuck shapes) 双重验证后**:
- 0 个 LOSE→WIN flip
- deep-LOSE 最大边际增益 1.2pp (16384×4096×28672: ts_v12 89.2% → u8 90.4%)
- 突破需要根本性的内核重构 (aiter 架构: 只有 A 走 LDS, B 通过深度软件流水直接从 global 加载, 需要 >256 VGPRs 的寄存器预算)

## Round 2 + Deep-LOSE 新增 dead-end (2026-04-17)
- **memclause × 28 ceiling variants** (Round 2): 0 LOSE→WIN flip vs Round 1, 边际 ±25 TFLOPS movement only. WIN shapes 上 best variant 漂移 (e.g., `ts_lgk2_v12_memc` → `ts_lgk2_v20_memc` on 16384×14336×4096) 但 ratio 几乎不变
- **VMCNT ceiling sweep (v20/v24/v32)**: v20 在已 WIN shape 上偶尔最优, 但不翻 LOSE
- **TAIL_BARRIER_VMCNT × VMCNT cross**: tv0/tv16 × v4/v12 — `_ts_v12_tv0_memc` 在 4 个 WIN shape 上是 best variant, 但不翻任何 LOSE
- **PF asymmetric (pf4_6, pf6_4, pf3_8)**: pf6_6 对称在 4096×32768×28672 等 2 shape 边际最优, 不对称无效
- **waves_per_eu(2,2) attribute**: deep-LOSE 上 +0.4pp 仅 1 shape, 其余 sub-best — REFUTED
- **AGPR hint (amdgpu_num_agpr=192)**: 全部 sub-best — FAILED
- **UNROLL_K=8/16 × LGK × VMCNT × memc cross**: 在 deep-LOSE 上 +0.3-1.2pp 边际 (4 个 shape), 不翻 LOSE (推翻 AGENT_PROMPT 之前 "UNROLL=1,2,4 worse" 的过早结论 — 8/16 配合 lgk/v12/memc 才有效, 但仍不够)
- **128256×32768×4096 (mega-M)**: 完全 IMPENETRABLE — Round 2 + deep-LOSE 都无法逼近 R1 best (`ts_gm2_v12_memc_dc` 92.9%)

## Rocprof 分析结论 (2026-04-16)
对生产 .s (N=32768, K=4096, TS=1, LGK2) 做了 PC sampling 和 assembly 分析:
- 2048 MFMAs, 512 ds_reads, 165 s_nop, 313 asm block pairs
- **s_nop 全部在 PF 组里** (162/165 在 buffer_load...lds 前), 是 m0→buffer_load 硬件 hazard delay
- **Inter-block code 被 MFMA pipeline 完全隐藏**: MFMA pipeline depth ≥16 cycles, inter-block gap ≤6 cycles
- **63 s_nop after ASMEND**: m0 在 asm 块前设定, 块内不 clobber m0, 理论可移除但实际增益 0
- **99 s_nop in PF pairs**: m0→buffer_load 硬件 hazard, 移除导致 26% 计算结果错误
- **编译→重写→重组装 pipeline 验证可行** (rewrite_asm.py + build_rewrite.sh), 但无可行的重写优化

## Benchmark Rules
- **warmup=200, iters=500**, trimmed mean 10%
- 用空闲GPU (`rocm-smi` 确认0%)
- `HIP_VISIBLE_DEVICES=N`
- MI355X 上 competitor_tflops 是正确 baseline

## 关键文件
| 文件 | 用途 |
|------|------|
| `kernel_mxfp4_gluon_cpp.cpp` | 主生产内核 (62 auto-tune flags) |
| `bench_all_42.py` | 42-shape benchmark (62 auto-tune variants, sequential) |
| `bench_all42_parallel.py` | 42-shape benchmark (parallel across GPUs, 62 variants) |
| `build_all42_parallel.py` | 并行编译器 (62 variants × 26 N,K pairs) |
| `spot_test.py` | 单shape多variant测试 (95 variants) |
| `spot_new_variants.py` | 新variant快速spot测试 (30 new + 9 reference) |
| `bench_all42_results.json` | 最新42-shape结果 (24/42 WIN, 115 variants, Round 2) |
| `bench_all42_results_round1.json` | Round 1 snapshot (24/42 WIN, 87 variants) |
| `bench_all42_results_round2.json` | Round 2 snapshot (24/42 WIN, 115 variants) |
| `build_new_variants.py` | 并行编译 (memclause + Optimizer A 21 variants) |
| `build_round2_variants.py` | 并行编译 (Round 2 ceiling 28 variants) |
| `build_deep_lose_variants.py` | 并行编译 (deep-LOSE 44 targeted variants) |
| `bench_deep_lose.py` / `bench_deep_lose_results.json` | 10-shape spot bench (deep-LOSE 分析员) |

## 环境设置 (换机器必读)
```bash
cd /shared_nfs/kyle/test/HipKittens
git checkout mxfp4
cd analysis/fp8_gemm/mi350x

# 并行编译所有.so (16线程)
python3 build_all42_parallel.py 16

# 用GPU 1-4跑benchmark (parallel, needs pre-built .so)
python3 bench_all42_parallel.py 1,2,3,4

# 用GPU 1跑benchmark (sequential, handles compilation)
HIP_VISIBLE_DEVICES=1 python3 bench_all_42.py

# 单shape测试
HIP_VISIBLE_DEVICES=2 python3 spot_test.py 6144 32768 4096 4291.0

# Cursor的仓库 (只读参考)
# /shared_nfs/kyle/test/Hipkittens2/analysis/fp8_gemm/mi350x/
```
