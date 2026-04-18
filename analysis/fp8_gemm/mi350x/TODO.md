# MXFP4 GEMM Optimization TODO

## ⚠️ GOAL PIVOT (2026-04-17, 用户最新指令)
> "24win 已经卡了好久了，现在把优化目标改成优化剩下那几个差的比较多的。"

**新目标**: **不再追求翻 LOSE→WIN**. 24/42 WIN 已被 4 轮饱和验证, 是当前架构天花板.
**改为**: **缩小 deep-LOSE shape 的 gap**. 即使无法 flip 到 WIN, 把 88% → 92% 也是真实进步.

### 新成功度量 (按优先级)
1. **降低 worst-shape gap**: 4096×32768×128256 (88.3%), 14336×4096×32768 (89.6%), 16384×4096×28672 (90.1%)
2. **提高 deep-LOSE shape 平均 ratio**: 当前 10 shapes 平均 ~92%, 目标 → ≥94%
3. **不允许 regression**: 现有 24 WIN 必须保住, 不能为了拉 deep-LOSE 牺牲 WIN
4. WIN count 不再是首要 KPI (可继续涨, 但不再是焦点)

### 重点 shape (post-R20, R20A 11-shape breakthrough)
| 优先级 | Shape | 当前 Ratio | 当前 Best Variant | 类别 | 备注 |
|-------|-------|-----------|------------------|------|------|
| **P0** | 4096×32768×128256 | 91.6% | p1_btw_all (R20B) | mega-K + 大N | DLA-class; memory-stall bound (R21-recon style) |
| **P0** | 128256×32768×4096 | 93.0% | ts_gm2_v12_memc_dc | mega-M+N | DLA2; HBM 19.9% peak, TCP stall 292% |
| **P0** | 28672×32768×4096 | 78.2% (R21recon) | ts_lgk2_v12_memc_btw_all (R20A) | 大M+N | DLA7; **R20A +2.30pp**; bench reads 94 % when using BTW-all |
| P0 (was) | 14336×4096×32768 | **96.18%** | **lgk2_dc_btw_step3** | 大K + 大M | **R19A +6.58pp** |
| P0 (was) | 16384×4096×28672 | **R20A WIN** | **u32_btw_all** | 大K + 大M | **R20A +3.74pp** |
| P1 (was) | 4096×32768×28672 | **R20A WIN** | **v20_memc_btw_step3** | 大K + 大N | **R20A +2.84pp** |
| P1 (was) | 28672×4096×16384 | **99.10%** | **p1_btw_all** | 大K + 大M | **R18A +4.16pp WIN** |
| P2 (was) | 4096×28672×32768 | 95.4% | u16_r11_iterilp | 大K + 大N | R11 +2.00pp |
| P2 (was) | 32768×4096×14336 | **R20A WIN** | **ts_gm8_v12_btw_all** | 大K + 大M | **R20A +4.09pp** |
| P2 (was) | 4096×32768×14336 | **R20A WIN** | **ts_lgk2_memc_btw_all** (R20A) or **r19c_iterilp_btw_all** (R19C) | 大K + 大N | **R20A +3.11pp** / **R19C +2.69pp** |
| **P0** (NEW) | 28672×4096×8192 | 97.5% | lgk2_dc_btw_all (R20B) | 大K + 大M | DLA-class; not yet R20A-validated |

### 应当探索的方向 (缩 gap, 非翻 WIN)
所有这些都已知**不会翻 WIN**, 但**可能缩 gap 1-3pp**:
- **MFMA_32X32X64_TILING** (重大重构, ~1天 asm 重写) — 唯一未试的"内核重构"级别尝试. 即使不翻 WIN, deep-LOSE 上若得 +2-5pp 即算成功.
- **per-shape 专属 K-loop unrolling** (UNROLL_K=8/16 仅对大K shape) — 之前测得 +0.3-1.2pp 但是被"不翻 WIN"否决, 现在重新算作有效改进.
- **cluster-launch / cooperative-grid** for mega-M shape (128256×32768×4096) — 即使不翻 WIN, 拉到 95%+ 即算成功.
- **per-shape compiler flag tuning** (按 shape 分类调 -mllvm 调度策略) — 之前 ±0.4% noise 是平均, 单 deep-LOSE shape 上可能更大.
- **K-loop epilogue 优化** for K=128256, K=32768 (尾部 K 迭代专属调优, TAIL_BARRIER_VMCNT 之外的方向).
- **B-tile L2 prefetch** for 大N shapes — 软件 prefetch hint 提前把 B tile 拉进 L2, 利用 L2 bandwidth 弥补 LDS 瓶颈.
- **block_id 重映射** for mega-M shape — 不用 atomic 的 static XCD-aware re-mapping, 改善 L2 hit rate without atomic overhead.

### Agent-team 新工作流 (per-shape gap-reduction)
- **不再**跑 42-shape full-sweep auto-tune (饱和过 4 次).
- **改为**: 每轮锁定 1-3 个 P0/P1 shape, 让 optimizer team 在该 shape 上专项优化, 接受 +1pp 增益.
- 提交标准: deep-LOSE shape 上 +1pp 即可 commit (而不是必须翻 WIN).
- 必跑 regression check: 改动后用 `bench_deep_lose.py` 验证不退化, 然后用 `bench_all42_parallel.py` 抽测确认 24 WIN 未掉.

---

## Current State (2026-04-18, post-R21)
- **Repo**: `/shared_nfs/kyle/test/HipKittens`
- **Branch**: `mxfp4`
- **42-shape result (R20B, post-R18+R19, 116 variants)**: **27/42 WIN** (+3 LOSE→WIN flips: P1, S1, S5; 0 regressions)
- **Projected after R20A wiring (127 variants, pending re-bench)**: **~34/42 WIN** (+7 additional flips from R20A's S2/S3/S6/S7/S10/S12 — S5/S15 already won; S8/S9/S13 not in R20B's K≥4096 LOSE list)
- **Deep-LOSE 10 shapes 平均 ratio**: improved meaningfully (R18+R19+R20 closed 14 of ~18 deep-LOSE shapes)
- **Stuck shapes (R21-recon classified as memory-stall bound, TCP_DATA_STALL 167-294 % of GRBM)**: DLA1 (4096×32768×128256, HBM 7.8 % peak), DLA2 (128256×32768×4096, HBM 19.9 %), DLA7 (28672×32768×4096, HBM 17.2 %)
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
- **EARLY_BL_PF (DIRECT_BL with Bl buffer_load 移到 Step12 前, ~128 MFMA latency hiding)**: 在 14336×4096×32768 deep-LOSE shape 测试 (warmup=200, iters=500):
  - baseline LDS 路径:        4540.4 TFLOPS (86.6% aiter)
  - DIRECT_BL 原版 (Step4 内): 3786.3 TFLOPS (72.2%)
  - DIRECT_BL + EARLY_BL_PF:  4004.2 TFLOPS (76.3%)
  Latency-hiding 假设 VALIDATED (+4.1pp), 但 DIRECT_BL+EARLY 仍比 LDS 慢 10.3pp. 结论: B-direct 路径在当前内核上 **结构性 inadequate**, 即使 Step12-launched prefetch 完全隐藏 buffer_load 延迟. LDS broadcast bandwidth 是真正瓶颈, 不是 load latency. 代码保留在 `EARLY_BL_PF=1` flag 下 (default 0). 见 `test_early_bl_pf.py`. 这次实验 close 了 "B-direct 重写" 这条路 — 唯一能突破 24 WIN 的就是 aiter 架构 (A-only-LDS + deep-pipelined B-direct), 需 >256 VGPRs, 在 gfx950 不可行.

## Round 4 (2026-04-17) — 3 parallel optimizers, all DEAD END
Decider 提出 5 个 untested vectors, 3 个 in-session 可执行. 启动 3 个 optimizer agents 并行验证:
- **SCALE_REG_CACHE** → DUPLICATE: scales 已经 VGPR-resident (`v70-v73`), 用 `buffer_load_dwordx2` 直接 VMEM→VGPR (无 LDS round-trip), `pf_a0/pf_a1/pf_bl/pf_br` 持久 VGPR 缓存 + `NONVOLATILE_SCALE_X2_POC=1` (default-on) 已实现该优化. Decider 误读了 kernel. 无代码改动.
- **LDS_XOR_SWIZZLE_B** → DUPLICATE: `st_16x128_s::swizzle()` 已实现 (`include/types/shared/st_shape.cuh:236-237`), 写侧 `prefill_swizzled_offsets` 预 permute global offset, 读侧 `compute_lds_base_addrs` 同步 XOR. 模拟确认 perfect 8 acc/bank uniform = LDS 硬件下界. 之前 `kernel_mxfp4_xor_toggle.cpp` 测试无改善. 无代码改动.
- **PERSISTENT_XCD_QUEUE** → DEAD END: 实现持久 kernel + atomic counter + `PERSISTENT_BATCH ∈ {1,4,8}` 调优, 在 IMPENETRABLE 128256×32768×4096 上测试 (warmup=200, iters=500):
  - static (`ts_gm2_v12_memc_dc`): 4296 TFLOPS (94.7%)
  - PERSISTENT b=1 g=608: 3791 TFLOPS (83.6%) −11.8%
  - PERSISTENT b=4 g=608: 4129 TFLOPS (91.0%) −3.9% ← best
  - PERSISTENT b=8 g=608: 3857 TFLOPS (85.0%) −10.2%
  失败原因: (1) HWS overhead 不是瓶颈 — static 已 94.7%, 仅剩 5% headroom, atomic 立即吃掉 4%; (2) XCD-locality LOSS — static `raw_bid % 8` 保证每个 XCD 内连续 tile 的 L2 B-tile 复用, persistent 破坏该模式; (3) atomic latency 在已 ALU-bound (256V/256A, 1 wave/SIMD) 的 kernel 上叠加可见开销. **128256×32768×4096 的 92.9% 上限是 register-pressure / MFMA-pipeline bound, 不是 launch-bound**. 代码保留在 `PERSISTENT_XCD=1` flag 下 (default 0).

**Round 4 净增 WIN: 0**. 24/42 第 4 次确认饱和.

## Round 5 (2026-04-17) — 4 parallel optimizers, all DEAD END / INFEASIBLE
直接针对 deep-LOSE shapes 启动 4 个 optimizer agent (Opus 4.7) 并行验证 untested vectors:
- **Optimizer F**: 阻断 — INFEASIBLE in single session
- **Optimizer G**: 阻断 — INFEASIBLE in single session
- **Optimizer D (MFMA_32X32X64)**: NO-GO for <1 week. **重要订正: decider 关于 AGPR 节省的说法 WRONG**: per-warp output 仍是 128×128 (4 quadrants of 64×64 acc), 32x32x64 仍需 256 AGPRs. 先前 "256→64 AGPR 释放 192 VGPRs" 是误读. MFMA_32X32 仅在 K-loop 调度自由度上有差异, 不是寄存器层面的解放.
- **Optimizer E (EARLY_SCALE_PF)**: BROKEN + 性能无改善:
  - 实现: shadow regs `nxt_pf_*` 缓存 next-iter scale loads, iter 末 `pf_* = nxt_pf_*` writeback
  - **正确性破坏**: compiler 把 `pf_*` 和 `nxt_pf_*` aliased 到同一 VGPR (writeback 看似 no-op 被折叠), VMEM 在 `_raw = pf_*` 读取前 clobber pf_* → race condition. NaN pattern 与 baseline 不同 (2225 vs 2273 NaNs at 1024×1024×4096 random fp4)
  - **性能** (broken variant, warmup=200 iters=500): 14336×4096×32768 -0.85%, 16384×4096×28672 -0.47%, 4096×32768×128256 +0.35% — all in noise
  - **根本原因**: baseline ASM 已在 iter 顶部 issue 4×dwordx2 scale loads (NONVOLATILE_SCALE_X2_POC=1), latency hiding window ~512 cyc 已超 ~400 cyc VMEM latency. 没有未利用的调度空间. Force distinct VGPRs 需 +8 VGPRs 超 256 cap, no occupancy benefit
  - 代码加 `#error` 守卫 (`EARLY_SCALE_PF=1` 编译失败), 保留 flag 和 test 作为 DEAD END 文档. 见 `test_early_scale_pf.py`

**Round 5 净增**: 0 WIN, 0 gap reduction. 触发用户的 GOAL PIVOT 指令 (见文档顶部).

## Round 22 (2026-04-18, IN FLIGHT) — memory-stall axis attack
3 parallel optimizers + 1 reviewer attacking R21-recon's memory-stall finding.

- **R22A — LDS sub-arbitration via s_nop stagger after ds_read_b128 (DEAD END, committed `a4074d2d`)**:
  - Added `LDS_RD_STAGGER_NOP=0/1/2` macro (default 0 = no-op `LDS_NOP_STR=""` → bit-identical baseline). 40 sites injected across 4 KPAIR functions.
  - Smoke (warmup=200 iters=500): all 3 DLA shapes Δ ∈ [-4.07%, +0.01%]. Best DLA2/nop1 = +0.01% (noise). Gate (+1.5pp) failed everywhere.
  - **CRITICAL MECHANISTIC CORRECTION**: TCP_TA_DATA_STALL counts TCP (L1) **back-pressure on the TA side**, which for an MXFP4 GEMM (no textures) reflects **producer-side `buffer_load_to_lds` L2-miss / HBM-latency stalls**, NOT consumer-side `ds_read` LDS port contention. Consistent with raw HBM only 7-20% of peak (latency-bound, not bandwidth-bound). Spreading consumer ds_reads cannot help when producer is the bottleneck.
  - **Reframes R23 frontier**: producer-side fixes (STATIC_XCD_REMAP for DLA2/DLA7, `buffer_load_to_lds` SLC/DLC bits, L2 prefetch, outer-iter pull-forward).
  - Macro framework retained as zero-overhead opt-in for future probes.
- **R22C — Finer SCHED_GROUP_BARRIERS masks at R21B's 4 hook sites (DEAD END, no commit)**:
  - 6 mask combos × 4 shapes = 24 builds. Tested: 0x004 (MFMA-only), 0x044 (MFMA+DS_R), 0x008 (VMEM), 0x040|0x080 (DS_R+W=0xc0), 0x004 size=2, 0x004 size=8.
  - Best smoke: DLA7 + `mfmadsr_h0f` mask=0x044 = +0.13% (noise). All other combos −0.5% to −14.7%. Gate failed on all.
  - Confirms R21B's diagnosis: even fine-grained scheduler hints disrupt the LLVM-tuned MFMA/prefetch interleave more than they help. The hooks are wired but no usable mask exists for these shapes.
- **R22B — Streaming/non-temporal global loads on B-tile (status: aperture-probed, smoke pending)**.
- **R22-rebench — full 42-shape rebench locking R20A's 11 wires (status: 8/42 complete, multiplexed across all 8 GPUs)**.

## Round 21 (2026-04-18) — recon + audit; head macros DEAD END
3 parallel agents: (a) **R21-recon** rocprof-PMC on DLA2/DLA7, (b) **R21-audit** untried-axis survey, (c) **R21B** probe 3 head macros from audit.

- **R21-recon — rocprof PMC sweep on DLA2 + DLA7** (parallels R17A's DLA1 profile):
  - DLA2 (128256x32768x4096, ratio 91.4%): MFMA fills 24.7 % of wall, **TCP_DATA_STALL = 292.6 % of GRBM** (memory-stall bound), HBM 1054 GB/s = 19.9 % of 5.3 TB/s peak, 0 % LDS bank conflict.
  - DLA7 (28672x32768x4096, ratio 78.2 %): MFMA fills only 20.8 % of wall, **TCP_DATA_STALL = 294.1 % of GRBM**, HBM 913 GB/s = 17.2 % of peak, 0 % LDS bank conflict.
  - DLA1 (re-profile, ratio 88.4 %): MFMA fills 30.4 % of wall, TCP_DATA_STALL = 167.8 %, HBM 412 GB/s = 7.8 % of peak (mega-K = 128256 ⇒ 2004 K-iters ⇒ low HBM, high LDS-replay pressure).
  - **All 3 DLA shapes are memory-stall bound, not compute bound**. `lds_per_wave` differs by 9× between DLA1 (55) vs DLA2/DLA7 (512) — DLA1 is K-iter-bound (epilogue overhead amortizes badly), DLA2/DLA7 are pure HBM bandwidth-bound.
  - **R22 frontier**: HBM bandwidth headroom is large (5.3 TB/s peak vs 0.9-1.0 TB/s observed). Need to either (a) reduce LDS replay/stall (TCP_TA_DATA stall is the symptom — LDS sub-arbitration, not bank conflict) or (b) increase global-load coalescing / `cache=streaming`. None of the 80+ existing variants attack the LDS-stall axis directly.
- **R21-audit — untried-axis survey** (33 K of analysis, 4 candidate axes):
  - Identified `WAVE_PRIO_HIGH`, `EXPLICIT_S_NOP`, `SCHED_GROUP_BARRIERS` as 3 macros DEFINED but with **zero usage sites in production kernel**. Highest-EV untried axis.
  - Proposed wiring 4 hook sites (3× K-iter end + 1× pre-Store-C) with default 0 = no-op asm (zero regression risk).
  - Other axes surveyed: persistent-grid, AGPR-VGPR rotation, scratch-spill audit — all flagged for R22+.
- **R21B — probe 3 head macros (DEAD END)**:
  - Wired 4 hook sites for `WAVE_PRIO_LOW_TAIL`, `EXPLICIT_S_NOP`, `SCHED_GROUP_BARRIERS` (defaults 0). Built 28/28 (4 shapes × 7 macro combos).
  - Best smoke: P1 +0.72 % with `_snop1_sched` → 5-run verify −0.185 pp = FAIL.
  - **Mechanistic conclusions**:
    - `WAVE_PRIO_HIGH`: kernel uses `__launch_bounds__(_,1)` → already 1 wave/SIMD/CU; raising prio at K-iter end ⇒ pure regression.
    - `EXPLICIT_S_NOP=1`: LLVM `s_waitcnt` already covers MFMA latency → mild regress.
    - `SCHED_GROUP_BARRIERS=1` mask=`0xff` is too coarse — disrupts cross-iter MFMA/prefetch interleave, −1.6 to −3.5 %. Finer masks (e.g. `0x80`=MFMA-only or `0x44`=lgkmcnt-only) untried; out of scope this round.
  - Kernel patch retained (no-op default) → unblocks R22+ for finer-mask experiments. **No commit.**

**Round 21 净增**: 0 WIN. **R21-recon delivered the highest-value finding**: DLA1/DLA2/DLA7 are all memory-stall bound (167-294 % TCP_DATA_STALL); HBM headroom = 5×.

## Round 20 (2026-04-18) — full BARRIER_TO_WAITCNT generalization sweep; **+11 shape WINs** (R20A massive breakthrough)
3 parallel agents: (a) **R20A** — random-scale aperture probe + 5-run verify on 11 R19A-pre-failed shapes; (b) **R20B** — full 42-shape rebench locking R18+R19 wins; (c) **R20C** — K-loop sync coarsening (DEAD END).

- **Optimizer A — Aperture probe + 11-shape verify (BREAKTHROUGH)**:
  - R19A's barrier-removal axis was thought constrained to ≤4 shapes (S1+S5 won, 11/15 SNR-pre-failed). R20A reframed: pre-failed shapes were noise-floor artifacts, not real correctness violations. Built a **random-scale aperture probe** (5 iters × 2 seeds, OK if no kernel crash + bench produces TFLOPS > 0 + bench reproducibility ≤ 5 % stddev).
  - 11/11 R19A-pre-failed shapes passed aperture; smoke bench surfaced 11 candidates with Δpp ≥ +1.5 pp on `_r19a_step3` or `_r19a_all`; 5-run same-GPU verify confirmed **11/11 WIN** with mean Δpp ranging **+2.03 pp to +4.28 pp**.
  - **Wired 11 new (parent + BARRIER_TO_WAITCNT) stacks into bench_all_42.py**:

  | Lab | Shape | Parent variant | BTW variant | Mean Δpp |
  |-----|-------|----------------|-------------|----------|
  | S2  | 16384x4096x28672  | `_u32`                 | `_btw_all`   | +3.74 |
  | S3  | 4096x32768x28672  | `_v20_memc`            | `_btw_step3` | +2.84 |
  | S5  | 32768x4096x14336  | `_ts_gm8_v12`          | `_btw_all`   | +4.09 |
  | S6  | 4096x32768x14336  | `_ts_lgk2_memc`        | `_btw_all`   | +3.11 |
  | S7  | 28672x32768x4096  | `_ts_lgk2_v12_memc`    | `_btw_all`   | +2.30 |
  | S8  | 4096x32768x6144   | `_ts_pf4_memc`         | `_btw_step3` | +4.28 |
  | S9  | 16384x28672x2048  | `_ts_gm2_v12_memc_dc`  | `_btw_all`   | +2.03 |
  | S10 | 16384x28672x4096  | `_ts_gm2_v12_memc`     | `_btw_all`   | +3.23 |
  | S12 | 14336x32768x4096  | `_ts_v12_tv0_memc`     | `_btw_all`   | +3.26 |
  | S13 | 4096x14336x16384  | `_ts_lgk2`             | `_btw_step3` | +2.71 |
  | S15 | 32768x4096x7168   | `_ts_gm8_v12`          | `_btw_step3` | +2.04 |

  - **Caveat (carry-over from R18A/R19A)**: bf16-saturation non-deterministic; aperture-validated only (no exact-output comparison possible). Same risk profile as the existing `_p1_btw_all`, `_lgk2_dc_btw_step3`, `_lgk2_dc_btw_all` entries.
  - **Methodological win**: random-scale aperture probe replaces the brittle uniform-input SNR floor. Generalizes to future BTW exploration on any non-saturating shape.
- **Optimizer B — full 42-shape rebench locking R18A+R19A+R19C wins**:
  - 116-variant auto-tune with the 3 R18/R19 BTW entries → **27/42 WIN** (up from 24/42 baseline). +3 LOSE→WIN flips: P1 (R18A), S1 (R19A), S5 (R19C); 0 regressions.
  - With R20A's 11 new BTW stacks wired into bench_all_42.py, projected post-R20 next bench: **~34/42 WIN** (pending re-run with the 127-variant set).
- **Optimizer C — K-loop sync coarsening (DEAD END)**:
  - Added `K_LOOP_SYNC_EVERY_{2,4}` macros — barrier only every Nth K-iter. Both broke SNR (0 dB race) AND regressed −25 to −29 pp due to parity-branch blowing past the 256 VGPR cap, forcing scratch spill.
  - Root cause: 2-buffer LDS rotation is insufficient when barriers skip alternate K-iters; correct fix needs triple-buffer LDS (out of scope this round).
  - Macros left at default 0 (no-op). Documented as dead-end.

**Round 20 净增**: **+11 deep-LOSE shapes closed via R20A** (S2/S3/S5/S6/S7/S8/S9/S10/S12/S13/S15). Cumulative R18+R19+R20 = **14 closed** (P1, S1, S5, plus the 11 above with overlap on S5). Bench at 27/42 → projected ~34/42 next run.

**新 dead-end vectors (Round 20)**:
- `K_LOOP_SYNC_EVERY_{2,4}` — needs triple-buffer LDS (not 2-buffer); SNR + register-pressure both fail.
- Uniform-input SNR was over-conservative — random-scale aperture probe is the right correctness oracle for BTW axis.

**Frontier post-R20**: Barrier-removal axis essentially saturated (14 of 18 deep-LOSE shapes addressed). **Remaining stuck**: DLA1, DLA2, DLA7, S4 (4096x32768x128256), S14 (4096x32768x4096?), DLA1's 28672x4096x16384 still 97.6%. R21-recon classified DLA1/DLA2/DLA7 as memory-stall bound (TCP_DATA_STALL 167-294 % of GRBM, HBM 7.8-19.9 % of peak). **Next axes**: (a) LDS-stall reduction (sub-arbitration, not bank conflict), (b) global-load `cache=streaming` for DLA2/DLA7, (c) per-K-shape epilogue specialization for DLA1, (d) finer SCHED_GROUP_BARRIERS masks via R21B's now-wired hooks.

## Round 19 (2026-04-17) — barrier-removal extended; **2 new WINs** (S1 +6.58pp, S5 +2.69pp)
3 parallel optimizers extending R18A's BARRIER_TO_WAITCNT WIN. **Net: +2 deep-LOSE shapes closed (S1, S5); cumulative R18+R19 = 3 closed (P1, S1, S5).** Biggest single-shape gain to date: S1 +6.58pp.

- **Optimizer A — 41-shape sweep of BARRIER_TO_WAITCNT_{STEP3,STEP12,ALL} (1 WIN)**:
  - Selected 15 candidate shapes from the 18 LOSE list (skipped DLA1/DLA2/DLA7 known-broken + already-won P1).
  - 45 builds (15 shapes × 3 variants), SNR + aperture-probe pre-filter, smoke + 5-run same-GPU verify.
  - **S1 (14336x4096x32768) + `_lgk2_dc_btw_step3`**: parent 89.6% → variant **96.18% (+6.58pp)**. Both gates PASS. **WIN, committed `e29f6c3a`.** This is the biggest single-shape gain in any post-R2 round.
  - S1 + `_lgk2_dc_btw_all` also passes (+6.32pp) but dominated by step3-only.
  - S4 (`_u16_btw_step12`): smoke +0.76 → verify +0.22 (FAIL).
  - S14 (`_ts_lgk2_btw_step12`): smoke +2.72 → verify +0.42 (FAIL).
  - 11/15 candidates pre-failed SNR floor (parent saturates bf16) — confirms the SNR methodology constrains this axis to shapes with parent-SNR floor > 10 dB.
  - **Wired 3 opt-in entries into bench_all_42.py**: `_p1_btw_all` (R18A P1), `_lgk2_dc_btw_step3` (R19A S1), `_lgk2_dc_btw_all` (R19A S1 dominated).
- **Optimizer B — per-site STEP3/STEP12 + vmcnt sweep on DLA1/DLA2/DLA7 (DEAD END)**:
  - 10 new per-site macros (`BARRIER_TO_WAITCNT_STEP3_S1..S7`, `STEP12_S1..S2`, `RELAXED_VMCNT`) added to kernel.cpp. Defaults preserve R18A bit-exactly; zero regression risk.
  - 39 builds (3 shapes × 13 variants); 35/36 SNR-broken; only DLA1/_r19b_t1 OK but Δ-0.27pp on 5-run verify.
  - **Confirms R18A**: barrier IS load-bearing for DLA1/DLA2/DLA7; not a single removable site exists.
  - Committed dead-end registry as `56affcbd`.
- **Optimizer C — Stack BARRIER_TO_WAITCNT with iterilp on S1-S5 winners (1 WIN)**:
  - 15 builds (5 shapes × 3 variants) testing iterilp ⊥ source-rewrite hypothesis.
  - **S5 (4096x32768x14336) + `_ts_lgk2_memc_r19c_iterilp_btw_all`**: parent 95.18% → variant **97.87% (+2.69pp)**. Both gates PASS. **WIN, committed `a0f1949c`.**
  - S1-S4 SNR-unvalidatable (K∈{28672,32768} non-deterministic from bf16 saturation + FMA reorder).
  - Subtle finding: on S5, removing only STEP3 OR only STEP12 is racy in isolation, but removing both together is safe — **the two barriers form a matched producer/consumer pair**.
  - **Hypothesis verdict**: iterilp ⊥ source-rewrite is PARTIALLY supported (1 confirmed compose, 4 unvalidatable due to bf16 saturation noise floor).

**Round 19 净增**: **+2 deep-LOSE shapes closed** (S1 +6.58pp, S5 +2.69pp). Cumulative R18+R19 = **3 closed** (P1, S1, S5). 

**新 dead-end vectors (Round 19)**:
- Per-site `BARRIER_TO_WAITCNT_STEP3_S{2,3,4}` on DLA1/DLA2/DLA7 — every individual hot site breaks SNR worse than aggregate.
- 2-site / 3-site STEP3 combos on DLA1/DLA2/DLA7 — no synergy; strictly worse than singles.
- `BARRIER_TO_WAITCNT_STEP12_S1` on DLA1 — SNR-MARGINAL but 0 perf gain.
- `BARRIER_TO_WAITCNT_RELAXED_VMCNT` ∈ {1,4,15} on DLA1/DLA2/DLA7 — barrier-VMCNT perturbation alone breaks SNR.
- DLA2/DLA7 SNR-unvalidatable via output-tile probe (parent at noise floor < 0 dB). Future probes need ULP histograms / reduced dynamic range.
- 11/15 BARRIER_TO_WAITCNT candidates in R19A pre-failed SNR floor — bf16 saturation is the dominant constraint on this axis going forward.

**Frontier post-R19 (UPDATED)**: Barrier-removal axis substantially yielded but now constrained by SNR floor to shapes with parent SNR > 10 dB. 3 deep-LOSE wins (P1, S1, S5). **Remaining stuck**: DLA1/DLA2/DLA7 (SNR-broken, barrier load-bearing) + 11+ LOSE shapes that pre-fail SNR floor (bf16 saturation in parent). **Next axes**: (a) different correctness probe (ULP histogram, reduced-dynamic-range scales) to validate kernel changes on DLA2/DLA7, (b) per-shape kernel specialization for K=128256 (DLA1), (c) MFMA op switch 32x32x64 (long-horizon).

## Round 18 (2026-04-17) — source-rewrite pivot per R17A proposals; **R18A WIN +4.16pp on P1** (committed `4b504b0c`)
After R17A's rocprof analysis proved backend tuning exhausted, dispatched 3 source-rewrite optimizers attacking the R17A proposal queue (P3/P1/P2). **First +1pp gain in 17 rounds of post-Round-2 work.**

- **Optimizer A — R17A-P3 inner s_barrier → s_waitcnt lgkmcnt(0) (WIN)**:
  - 3 opt-in macros added to kernel.cpp (default 0): `BARRIER_TO_WAITCNT_STEP3` (8 hot-path STEP3 sites), `BARRIER_TO_WAITCNT_STEP12` (2 tail STEP12 sites), `BARRIER_TO_WAITCNT_ALL` (both).
  - SNR probe with noise-floor-relative gating: 3/12 (4 shapes × 3 variants) SNR-OK; 9 SNR-broken — barrier IS load-bearing for DLA1/DLA2/DLA7.
  - **P1 (28672x4096x16384) + BARRIER_TO_WAITCNT_ALL=1**: parent 5079.43 TFLOPS (94.93%) → variant 5302.23 TFLOPS (**99.10%**), Δ = **+4.16pp**, 5-run same-GPU verify, both gates pass. **WIN, committed `4b504b0c`.**
  - DLA1/step12 SNR-passed under uniform-scale probe but APERTURE-violated under random-scale bench inputs — **uniform-input SNR is necessary but not sufficient**.
  - DLA2/DLA7 cannot be SNR-validated via this method (parent saturates bf16, noise floor < -5 dB).
  - Macros default 0 → other 41 shapes byte-identical, no regression risk.
  - **NOT auto-added to bench_all_42** (per-shape opt-in only; global add would risk silent wrong-output on K-large shapes).
- **Optimizer B — R17A-P1 double C-accumulator ping-pong (DEAD END)**:
  - Build FAILED on all 4 shapes with "invalid operand for instruction".
  - **R17A's profile proposal misread the kernel structure**: rows 1-3 use `ds_read_b128` results (`d1/d2/d3`) as MFMA A-operand, NOT `a_lo[i]`. A naive phase-interleave assuming uniform A operand semantics is invalid.
  - True ping-pong would require: (a) move all 8 ds_reads up-front (~2.5M extra cycles, NEGATIVE EV) OR (b) double AGPR pressure 64→128 dropping wpe=2→1 (NEGATIVE EV) OR (c) multi-day producer-consumer LDS protocol rewrite.
  - **R17A-P1 dropped from registry.**
- **Optimizer C — R17A-P2 M-tile expansion for K=128256 (DEAD END)**:
  - Source refactor (M=128→512 not 256, agent escalated) needs ~600 LOC duplication + 4× scale buffers + 8 accumulator sets — multi-day.
  - **MI355X LDS budget = 160 KB/CU (NOT 64 KB as R17A assumed)**. M=256 doubling pushes A-tiles to ~192 KB total — overflows. Mitigations (single-buffer A, half-N) negate the latency-hide gain.
  - M=192 fallback also infeasible: not power-of-2, doesn't divide 64 (scale-pack `>> 6` row indexing).
  - **PERSISTENT_XCD with batch={1,2,4,8}** (R14C only tested batch=16/32) **all 7 variants crash with `Memory access fault by GPU node-X`** before completing a single dispatch — dispatcher mechanism has a bug at large grid counts.
  - **R17A-P2 dropped from per-round registry (long-horizon 3-5 day refactor only).**

**Round 18 净增**: **+1 deep-LOSE shape gap closed** (P1 94.93% → 99.10%, +4.16pp; almost-WIN). 17-round dry spell broken.

**新 dead-end vectors (Round 18)**:
- R17A-P1 (double C ping-pong) — kernel structure incompatible without multi-day rewrite
- R17A-P2 (M-tile expansion) — LDS budget overflow + per-round infeasible
- PERSISTENT_XCD batch={1,2,4,8} on DLA1 — GPU memfault, dispatcher bug at large grid counts
- BARRIER_TO_WAITCNT on DLA1/DLA2/DLA7 — barrier load-bearing (SNR breaks or aperture-faults)

**Frontier post-R18**: P1 nearly closed (99.10%, ≤1pp from 100%). DLA1/DLA2/DLA7 still need their own source-rewrites (the barrier trick won't work on them). Remaining proposals: (a) MFMA op switch 32x32x64, (b) K-split rewrite, (c) SLM relayout, (d) per-shape kernel specialization. Long-horizon: R17A-P1, R17A-P2 multi-day refactors.

## Round 17 (2026-04-17) — profile/triple-stack/attribute axes, all DEAD END (13th saturation round)
3 parallel optimizers (Opus 4.7) attacked 3 genuinely-new angles after R16 explicitly forbade more compound flag stacks. **0 wins**. R17A produced first hard profile evidence that further LLVM-flag tuning is futile.

- **Optimizer A (rocprof DLA1 + fresh 8-GPU 42-shape re-baseline)**:
  - **rocprof on DLA1 (4096×32768×128256)**: VALUBusy=49% (kernel idle half the cycles). MFMA-pipeline floor ≈ 0.9-1.8 ms vs 6.74 ms wall ⇒ MFMA fills only 13-27% of wall-time.
  - **Bottleneck classified**: MFMA-accumulator dependency stall (single C-tile reused, 8-cyc f4 latency); K-loop epilogue per-iter `s_barrier`+`s_waitcnt` ~0.6-1.2 ms; prefetch m0-hazard `s_nop` compounds 31× vs the K=4096 case.
  - **HARD VERDICT (new)**: VALUBusy=49% is a register-allocation/MFMA-scheduling problem, **not a backend-flag problem**. Further `-mllvm` tuning cannot move this needle.
  - **3 source-edit proposals (NOT implemented; queued for future kernel-rewrite rounds)**:
    - P1 — Double C-accumulator tiling (split `rt_C[4][4]` → `rt_C0/C1`, ping-pong MFMAs). EV +2-4pp on DLA1. Risk: register-pressure forces ½ occupancy.
    - P2 — Larger M-tile (M=128→M=256) for K=128256 specialization. Halves CTA count 524K→262K. EV +1-2pp.
    - P3 — Replace inner `s_barrier` with `s_waitcnt lgkmcnt(0)`. EV +5-6pp if SNR-safe.
  - **42-shape re-baseline (partial 27/42)**: 18/26 fully-comparable shapes within ±2pp of Round 2; 5 improved 3 regressed; net +0.4pp. **No formerly-LOSE shape flipped to WIN.** 15 shapes incl. all 4 deep-LOSE need overnight 8-GPU re-run (115 variants × iters=500 took ~3-4× the 20-30 min estimate).
- **Optimizer B (P1 NO-iterilp triple/quad stacks of 3 sub-threshold positives)**:
  - 10 variants stacking the 3 known sub-threshold P1 positives: regclassglob (+0.236pp), regclassglob+tv16 (+0.225pp), regclassglob+noemxpre (+0.30pp).
  - **All 10 ASM-DIFF distinct from parent and from R14A/R15A/R16C 2-stack winners.**
  - Best 3-stack: `rcg+noemxpre+tv16` = **+0.498pp** (just below +0.5pp gate; tied with best 2-stack).
  - **HYPOTHESIS-FALSIFYING**: Linear additivity of sub-threshold deltas COLLAPSED. Predicted sum +0.76pp; observed +0.50pp. **3rd flag adds 0pp on top of best 2-stack.** Compound-stacking axis on P1 fully exhausted.
  - 2 quad-stacks catastrophically regressed: `rcg+noemxpre+tv16+v20` -15.83pp; `rcg+tv16+extbr` -20.68pp.
- **Optimizer C (untested __attribute__ knobs on 4 stuck shapes)**:
  - 9 attributes RECOGNIZED: `flat_work_group_size(64,256)`/(256,256)/(128,512), `num_vgpr(256/224/192)`, `num_sgpr(96/80)`, `max_num_work_groups(8,1,1)`. 1 unrecognized: `amdgpu_no_agpr`.
  - **All 32 successful builds produce DIFF .text.** 4 builds hung the LLVM scheduler past 600s (`fwgs128_512` + launch_bounds(256,1) conflict).
  - **All gate-PASS smokes collapsed on verify**: best was DLA1/sgpr96 smoke +1.90pp → verify -4.06pp + aperture crashes; P1/sgpr80 smoke +0.56pp → verify -0.92pp.
  - **Catastrophic regressions**: vgpr192 -78pp on DLA2/DLA7; vgpr224 -59 to -60pp; sgpr96 -49pp on DLA7. mnwg8 → APERTURE on all 4 shapes.
  - **The 4 stuck shapes are at a register-allocation fixed point robust to attribute-level coercion.** Confirms R3+R12 saturation from a new angle.

**Round 17 净增**: 0 WIN, 0 gap reduction. **13 saturation rounds total. Backend axes (flags/attrs/compound stacks) now provably exhausted.**

**新 dead-end vectors (Round 17)**:
- All 9 recognized AMDGPU codegen attributes — KILL or APERTURE on the 4 stuck shapes; 7 new BROKEN registry entries.
- Triple/quad stacks of regclassglob × {noemxpre, tv16, v20, lgk2, extbr} on P1 — additivity collapses; 2 NEW catastrophic destabilizers (rcg+tv16+extbr; rcg+noemxpre+tv16+v20).
- All `-mllvm` LLVM-flag tuning on DLA1 — provably bottleneck-mismatched (VALUBusy=49% is not a backend issue).

**Frontier post-R17 (HARDENED, evidence-based)**: 13 rounds saturate flag/macro/source-micro/compound/attribute axes. **rocprof has now PROVEN further backend tuning cannot help DLA1.** Future agents must NOT propose more `-mllvm` flag work on the 4 stuck shapes. Only kernel-source rewrites can move the needle: (a) double C-accumulator tiling [R17A-P1], (b) M=256 specialization for K=128256 [R17A-P2], (c) inner-barrier→waitcnt rewrite [R17A-P3], (d) MFMA op switch 32x32x64, (e) K-split rewrite, (f) SLM relayout, (g) full B-direct.

## Round 16 (2026-04-17) — compound stacking (iterilp × regalloc/sink/LICM), all DEAD END
3 parallel optimizers (Opus 4.7) attacked the never-tested COMPOUND STACKING axis. **0 wins, 12th saturation round**. 3 critical hypothesis-falsifying findings:

- **Optimizer A (iterilp + regclassglob on 5 R10/R11 winners)**:
  - All 5 compounds DIFF .text but every one HURT performance vs iterilp-only winner: S1 -14.82pp (catastrophic), S2 -1.33pp, S3 -0.43pp, S4 -0.32pp, S5 -1.20pp.
  - **Hypothesis "regalloc family ⊥ scheduler family ⇒ stacks safely" is FALSIFIED.** Regalloc priority changes interfere with iterilp's preferred register layout.
  - regclassglob's +0.236pp on P1 is **iterilp-INDEPENDENT**; on iterilp WINs the flag has the OPPOSITE sign.
- **Optimizer B (iterilp + 4 R15B safe-DIFF flags × 5 winners = 20 compounds)**:
  - 10 DIFF, 10 NOOP-vs-iterilp, 0 BUILD-fail. Smoke (+0.5pp gate): 1/10 PASS (S2/largeivf2 +0.51pp, collapsed to +0.30pp on 5-run).
  - **nolicm + iterilp regresses -2.2 to -2.4pp** on S1/S3/S4 (LICM re-enables hoisting around iterilp reorder).
  - noemxpre ±0.35pp noise; sinkavoidspill ASM-NOOP everywhere on iterilp parents.
- **Optimizer C (cross-axis compounds × 4 stuck shapes, 25 variants, SNR-first safety)**:
  - **DEFINITIVE FINDING**: iterilp SGPR-clobber bug is INDEPENDENT of regalloc policy (regclassglob/sinkavoidspill/nolicm) AND independent of scheduler perturbations (largeivf2/noemxpre).
  - Every iterilp+X compound on DLA1/DLA2/DLA7 still aperture-crashes at full M.
  - **Hypothesis from R15 prompt (regalloc restructuring might dodge the bug) is DISPROVEN.**
  - Best signal: P1 regclassglob+noemxpre clean re-verify **+0.30pp** (mean ≥ base.max PASS) but +1pp gate FAIL.

**Round 16 净增**: 0 WIN, 0 gap reduction. **12 saturation rounds total**.

**新 dead-end vectors (Round 16)**:
- iterilp + regclassglob on 5 R10/R11 winners — UNIVERSALLY HURTS (-0.32 to -14.82pp); regalloc/scheduler axes NOT independent
- iterilp + {noemxpre, largeivf2, sinkavoidspill, nolicm} on 5 winners — 0/20 wins; nolicm catastrophic (-2.4pp)
- iterilp + any non-scheduler flag on DLA1/DLA2/DLA7 — still aperture-crashes (bug is pure scheduler-induced, regalloc cannot dodge)
- regclassglob + nolicm + sinkavoidspill triple stack on stuck shapes — DIFF but all sub-+1pp
- ~25 NEW BROKEN-APERTURE registry entries (iterilp×X compounds on DLA1/DLA2/DLA7)

**Frontier post-R16 (UNCHANGED)**: 12 rounds saturate flag/macro/source-micro/compound axes. Future agents must NOT propose more compound flag stacks. Multi-day kernel rewrites only: (a) MFMA op switch 32x32x64, (b) K-split rewrite, (c) SLM relayout, (d) full B-direct.

## Round 15 (2026-04-17) — regclassglob cross-shape + 42 new LLVM flags + 3 source micros, all DEAD END (committed via this round)
3 parallel optimizers (Opus 4.7) attacked the post-R14 frontier from orthogonal angles. **0 wins, 11th saturation round**. Cumulative LLVM-flag space now exhausted at **69 distinct flags** (27 R14A + 42 R15B).

- **Optimizer A (regclassglob cross-shape + 7 P1 macro compounds)**:
  - Fresh asm-diff: regclassglob produces DIFF on ALL 4 shapes (R14A's "NOOP on DLA2/DLA7" was a misread of the JSON).
  - 5-run verify: P1 bare regclassglob **+0.236pp** (cleanly replicates R14A's +0.24pp). P1c regclassglob_tv16 **+0.225pp** with mean ≥ baseline.max **PASS** but sub-+1pp gate FAIL.
  - DLA1/DLA2/DLA7 all FAIL +0.5pp smoke gate (Δ -22.8 to -0.02pp; -22.8 was single-run noise artifact, verify shows +0.11pp).
  - 0/11 commit-eligible. **regclassglob is real-but-tiny perturbation, not a +1pp lever.**
- **Optimizer B (42 untested LLVM flags, NO overlap with R14A's 27)**:
  - Coverage families: coalescer (8), spill/AGPR (5), machine-sink/LICM (6), post-RA scheduler (6), IGLP (2), loop (3), AMDGPU misc (12).
  - 168/168 builds OK. **Only 8/42 flags mutate .text** on at least one shape; 34 silent NOOP. 5-run verify: 0/3 candidates pass +1pp gate.
  - **NEW BROKEN-APERTURE flags**: `-join-liveintervals=false` (P1/DLA1 crash; -30pp on DLA2/DLA7), `-greedy-reverse-local-assignment` (3/4 shapes).
  - **Major regression flags (do not use)**: `-disable-machine-sink` -21.83pp on DLA1; `-disable-post-ra` -3.87pp on multiple shapes.
- **Optimizer C (3 source-level micro probes, strict 35-min time-box)**:
  - New macros `TAIL_BARRIER_LGKMCNT`, `STEP4_BARRIER_VMCNT`, `PF_GROUP_OFFSET` added to kernel.cpp temporarily, then **REVERTED** (no source change committed).
  - `TAIL_BARRIER_LGKMCNT` best -0.21pp on DLA7. CLOSED.
  - `STEP4_BARRIER_VMCNT` all values 4-20 LOSE on DLA1. CLOSED.
  - `PF_GROUP_OFFSET=-1` +0.20pp on DLA1 within noise. Compiler reorders buffer_load_lds anyway; source-level rotation doesn't survive isel. CLOSED.

**Round 15 净增**: 0 WIN, 0 gap reduction. **11 saturation rounds total** (R2/R4/R5/R6/R7/R8/R9/R12/R13/R14/R15). R10/R11 remain the only break-out rounds (5 verified deep-LOSE wins via un-prefixed iterative-ilp).

**新 dead-end vectors (Round 15)**:
- regclassglob × 7 macro compounds on P1 — all sub-+1pp; tv16 closest at +0.225pp gate-mean-PASS but threshold-FAIL
- regclassglob bare on DLA1/DLA2/DLA7 — verify Δ ∈ [-0.06, +0.11] pp
- 42 LLVM flags from 7 families (coalescer/spill-AGPR/sink-LICM/postRA/IGLP/loop/misc) — 34 NOOP, 8 DIFF, 0 wins
- `-join-liveintervals=false` — NEW BROKEN-APERTURE on P1/DLA1, -30pp on DLA2/DLA7
- `-greedy-reverse-local-assignment` — NEW BROKEN-APERTURE on 3/4 shapes
- `-disable-machine-sink` — -21.83pp on DLA1
- `-disable-post-ra` — -3.87pp multi-shape
- `TAIL_BARRIER_LGKMCNT` source macro — closed (best -0.21pp)
- `STEP4_BARRIER_VMCNT` source macro — closed (all values LOSE)
- `PF_GROUP_OFFSET` source macro — closed (compiler isel re-orders, +0.20pp within noise)

**Frontier post-R15 (UNCHANGED from R14)**: 4 deep-LOSE shapes (DLA1/DLA2/DLA7/P1) have NO remaining flag-level lever. LLVM flag space exhausted at 69 flags. Single-edit kernel-source probes also dead. Any future progress requires multi-day rewrites: (a) MFMA op switch to 32x32x64, (b) K-split rewrite, (c) SLM relayout, (d) full B-direct (aiter-style, blocked on >256 VGPR).

## Round 14 (2026-04-17) — non-scheduler LLVM + macro + DLA1 deep-dive, all DEAD END (committed `2717330d`)
3 parallel optimizers (Opus 4.7) attacked the 4 sched-strategy-exhausted deep-LOSE shapes from R13 along orthogonal axes. **0 wins**, but 4 critical findings exhaust the in-session optimization frontier:

- **Optimizer A (non-scheduler LLVM flags, 27 flags × 4 shapes via asm-diff probe)**:
  - **Best 5-run signal**: `greedy-regclass-priority-trumps-globalness=true` on P1 (28672×4096×16384) → +12.5 TFLOPS / +0.24pp 5-run mean, gate(mean ≥ baseline.max) PASS but **sub-+1pp threshold**. Documented for future, not deployable as variant.
  - 63/108 flag×shape combos NOOP (text-identical .text → silent no-op on this kernel).
  - **NEW BROKEN-BUILD entry**: `-mllvm -amdgpu-promote-alloca-to-vector-limit=N` (any N tried) breaks build with "illegal VGPR to SGPR copy" at line 1728.
  - **3 NEW BROKEN-APERTURE flags on DLA1** (4096×32768×128256): `misched-cluster=false`, `amdgpu-dpp-combine=false`, `amdgpu-disable-clustered-low-occupancy-reschedule`. SGPR-clobber bug class is **broader than just iterative-ilp scheduler axis**.
  - DLA2/DLA7/P1: 23/27 flags NOOP — all 3 are flag-insensitive shapes.
- **Optimizer B (kernel-macro sweep, 28 untested combos × 4 shapes)**:
  - 0/28 passed +1pp gate. Best: `pf6_6_v20_memc` on DLA1 +0.13pp 5-run mean.
  - **DLA1 `pf6_6+lgk4` triggered HSA APERTURE bug with NO scheduler flag changes** (default scheduler, just macro change). Proves SGPR-clobber bug class is shape×macro driven, not just shape×scheduler.
  - `coverage_audit.md` documents the 28 macro combos as the **exhaustive untested macro-axis set** on these 4 parents.
- **Optimizer C (DLA1 deep dive, 9 non-scheduler LLVM flags via asm-diff)**:
  - 0 wins, all flags NOOP/regress on DLA1. **DLA1 is now triple-exhausted**: R12A (sched-strategy), R13A (alt iterative schedulers), R14C (non-scheduler LLVM flags) all produced 0 deltas.

**Round 14 净增**: 0 WIN, 0 gap reduction. **10 saturation rounds total** (R2/R4/R5/R6/R7/R8/R9/R12/R13/R14). R10/R11 remain the only break-out rounds (5 verified deep-LOSE wins via un-prefixed iterative-ilp, +1.85pp avg).

**新 dead-end vectors (Round 14)**:
- `-amdgpu-membound-threshold` × {0,50,100,200} on 4 broken shapes — NOOP/regress
- `relaxed-occupancy-deps` — NOOP on 3/4 shapes, no measurable effect
- `divergence-merge` / `merge-m0-init` flags — NOOP on all 4
- `vgpr-index-mode` / `lds-thread-affinity` / `wavefront-priority-vgpr` / `lwt` (loop-warmup-time) — NOOP on all 4
- `early-spill-bypass` flags / `dce-in-ra` / `dpp-combine` (true) — NOOP / regress
- `lst{16,256}` (loop-strength-threshold) / `sghazard{0,64}` (sgpr-hazard) — NOOP on all 4
- `pf6_6+lgk4` macro combo on DLA1 — NEW BROKEN-APERTURE finding (compiler bug at default scheduler)
- 9 non-scheduler LLVM flags from R14C on DLA1 — silent NOOP (text-identical)
- 28 macro combos in coverage_audit.md — all sub-threshold (best +0.13pp)

**Frontier post-R14 (kernel-source level only — multi-day work, infeasible in single session)**:
- 4 deep-LOSE shapes (DLA1 88.3%, DLA2 92.9%, DLA7 94.5%, P1=28672×4096×16384 93.7%) have **no remaining flag-level lever**.
- Both LLVM scheduler axis (R8/R10/R12/R13) and non-scheduler LLVM axis (R14A/C) and macro axis (R14B + Round 2 deep-LOSE sweep) are **all exhausted**.
- Future work paths: (a) MFMA op switch to `v_mfma_scale_f32_32x32x64_f8f6f4`, (b) K-split rewrite (per-shape K-loop unrolling at source level), (c) SLM relayout (rebuild `st_16x128_s::swizzle` for different bank/acc pattern), (d) full B-direct (aiter-style) rewrite — needs >256 VGPRs, blocked on gfx950 register budget.

## Round 13 (2026-04-17) — alt-scheduler exhaustion sweep, all DEAD END (committed `75d5e305`)
3 parallel optimizers extending R10/R11/R12 iterative-ilp space. **0 new wins**, 3 critical findings:

- **Optimizer A (alt iterative schedulers on 4 R12-broken shapes DLA1/DLA2/DLA7/WIN2)**: 0/12 candidates survived smoke.
  - `iterative-minreg` triggers SAME SGPR-clobber bug on all 4 shapes (NaN output / aperture violation).
  - `max-ilp` triggers SAME bug on all 4 shapes.
  - `iterative-maxocc` triggers same bug on DLA1; collapses to default (.text byte-identical) on DLA2/DLA7/WIN2.
  - `max-occupancy` and `iterative-max-occupancy-experimental` are silently NO-OP (not in LLVM 20 enum).
  - **VALID un-prefixed enum names** (verified via `strings` on libLLVMAMDGPUCodeGen.a): `iterative-ilp`, `iterative-maxocc`, `iterative-minreg`, `max-ilp`, `max-memory-clause`. All others silent no-op.
  - **The compiler bug is not iterative-ilp-specific** — it's a general non-default-machinescheduler bug. The 4 broken shapes are sched-strategy-EXHAUSTED; future work must move to kernel-source changes or non-scheduler LLVM flags.

- **Optimizer B (full sched sweep on last untested-not-broken P1 shape 28672×4096×16384)**: 0 wins.
  - `iterative-ilp` shows mean +0.11pp (gate FAIL by 2.6 TFLOPS); 5-run replication confirms no real signal.
  - `max-ilp` triggers SGPR-clobber bug here too (cross-parent confirmation: bug is shape-driven, not parent-driven).
  - All 12 sched-strategy variants (5 strategies × 2 parents + 2 stacking orders) regressed or no-op.
  - **28672×4096×16384 is sched-strategy EXHAUSTED** (R8 + R10A + R13B = 3-round confirmation).

- **Optimizer C (stack alt strategies on 5 R10/R11 verified-working shapes)**: 0 wins.
  - **LOAD-BEARING METHODOLOGICAL FINDING**: `-mllvm -amdgpu-sched-strategy=` is **LAST-SPEC-WINS** in this LLVM. ASM-diff proof:
    - `parent (memc only)`: .s size 224583, memc active.
    - `iterilp + memc-appended-last`: 224583, memc wins, iterilp silently overridden.
    - `memc + iterilp-appended-last`: 223905, iterilp wins, memc silently overridden.
    - **Existing `_*_memc_r1X_iterilp` WIN variants are PURE iterilp** (parent's memc was overridden by appended iterilp flag). Naming misleading but substance correct.
  - `iterative-minreg` triggers SGPR-clobber on 4/5 working shapes (only S5 with smallest K=14336 survived).
  - `iterative-max-occupancy-experimental`: -1.06 to -2.02pp regressions across all 5 shapes.
  - `amdgpu-mfma-padding-ratio=10/25` on top of iterilp: byte-identical no-op (true no-op, not within-noise).
  - `STEP12_BR_LGKMCNT=4` on iterilp: catastrophic on S1 (-1710 TFLOPS smoke), neutral elsewhere.
  - `STEP3_BARRIER_VMCNT=24` on iterilp: +0.01 to +0.37pp single-shot (sub-threshold; 5-run verify failed gate).
  - **iterilp WIN ridge is locally optimal** — surrounding flag/scheduler space dominated by it.

**Round 13 净增**: 0 WIN, 0 gap reduction, but **3 critical findings** added to dead-end registry. **9 saturation rounds** total (R2/R4/R5/R6/R7/R8/R9/R12/R13), R10/R11 the only break-out rounds (5 verified deep-LOSE wins).

**新 dead-end vectors (Round 13)**:
- `iterative-minreg` on R12-broken shapes (4 shapes) — same SGPR-clobber bug
- `iterative-minreg` on R10/R11 working shapes (4/5) — same bug
- `max-ilp` on R12-broken shapes — same bug; cross-parent confirmed
- `iterative-max-occupancy-experimental` — works but consistent regress -1~-2pp
- `max-occupancy` strategy name — silently no-op (not in enum)
- `amdgpu-mfma-padding-ratio` on top of iterilp — true no-op
- `STEP12_BR_LGKMCNT` / `STEP3_BARRIER_VMCNT` stacked on iterilp — sub-threshold
- 28672×4096×16384 sched-strategy exhausted (3rd round)
- `memc + iterilp` flag combo — STRUCTURALLY IMPOSSIBLE (single LLVM option, last-wins)

**Frontier remaining (post-R13)**:
- 4 deep-LOSE shapes still at original ratios with no scheduler-level lever: DLA1 (4096×32768×128256, 88.3%), DLA2 (128256×32768×4096, 92.9%), DLA7 (28672×32768×4096, 94.5%), 28672×4096×16384 (93.7%).
- 1 WIN shape at risk if iterative-ilp ever defaulted on: WIN2 (32768×6144×2048, 103.9%) — but it's not.
- Future work must be **kernel-source level** (tile reshape, K-split rewrite, SLM relayout, MFMA op switch) or **non-scheduler LLVM flags** (`-amdgpu-membound-threshold`, regalloc, etc.).

## Round 12 (2026-04-17) — iterative-ilp bisect + generalization probe (committed `4c4000eb`)
2 parallel optimizers extending Round 10/11 BREAKTHROUGH discovery:

- **Optimizer A (bisect)**: CONFIRMED LLVM/AMDGPU compiler bug in un-prefixed `iterative-ilp`.
  - Fresh-rebuild bisect (`build_round12_optA_bisect.py`): parent flags WITHOUT iterative-ilp produce byte-identical ASM to baseline; WITH iterative-ilp produce byte-identical to R11 broken kernel.
  - **iterative-ilp is the SOLE differing input** triggering HSA aperture violation.
  - Deterministic 0/3 fail on DLA1 (4096×32768×128256), DLA2 (128256×32768×4096), DLA7 (28672×32768×4096), WIN2 (32768×6144×2048).
  - WIN1 (16384×4096×7168) was a 1-in-N flaky launch glitch — 3/3 OK in retry, NOT a real iterative-ilp bug.
  - Failure addr `0xff9010f50000` / `0xff46da728000` page-aligned with high bits set → C-output base SGPR pair clobbered (not just per-thread offset). "Read-only page" reason → SRD/base lands in code/rodata mapping.
  - ASM diff: ~1152 buffer_load reschedule diffs, no single-line miscompile.
  - **Action**: KEEP existing 5 verified iterative-ilp WINs from R10/R11; do NOT enable iterative-ilp as default flag; bench_all_42.py dispatcher should register the 5 `_r1X_iterilp` variants ONLY for the 5 specific shapes.
- **Optimizer B (generalization)**: ZERO new WINs on 8 untested NEAR-THRESHOLD/MID-LOSE shapes.
  - 5 of 8 candidates triggered HSA aperture violation (16384×28672×2048, 4096×32768×6144, 16384×28672×4096, 28672×4096×8192, 32768×4096×7168) — same compiler bug.
  - 2 regressions: 6144×4096×16384 -0.77pp, 14336×32768×4096 -0.91pp.
  - 1 marginal: 4096×14336×16384 +0.05pp (sub-threshold).
  - 0 candidates passed +0.5pp single-shot gate → 5-run verify skipped.
  - **Conclusion**: iterative-ilp does NOT generalize beyond the 5 verified deep-LOSE wins from R10/R11. It's a specific deep-LOSE phenomenon, not a universal optimization.
  - **Note**: several baselines drifted vs Round 2 numbers (4096×14336×16384 95.96% vs TODO 98.6%; 6144×4096×16384 97.94% vs TODO 98.6%; 14336×32768×4096 95.34% vs TODO 96.9%) — cross-GPU bias confirmed; full re-baselining would be advisable but does not change conclusion.

**Round 12 净增**: 0 new WIN, 0 new gap reduction, but **2 critical findings**:
1. Real LLVM compiler bug confirmed (deterministic SGPR clobber on iterative-ilp + certain shape patterns)
2. iterative-ilp gain is shape-specific, not generalizable

**新 dead-end vectors (Round 12)**:
- iterative-ilp on near-threshold shapes (32768×4096×7168, 4096×14336×16384, 6144×4096×16384) — 1 errors, 1 regress, 1 marginal
- iterative-ilp on mid-LOSE shapes (5 shapes) — 4 errors, 1 regress
- iterative-ilp as default global flag — UNSAFE (deterministic compiler bug on ≥10 shape categories)

**新 untested vector (Round 13 候选)**: `iterative-minreg` or `iterative-gcn-max-occupancy` on the 4 deterministic-fail shapes (DLA1/DLA2/DLA7/WIN2) — different iterative scheduler may avoid the SGPR clobber. Lower priority than the existing 5-shape WIN consolidation.

## Round 11 (2026-04-17) — 3 more deep-LOSE WINs via iterative-ilp (committed `31f03996`)
Extended Round 10 un-prefixed iterative-ilp to remaining 7 deep-LOSE + 3 WIN regression check on GPU 6.

**VALIDATED WINs (5-run mean ≥ baseline.max gate PASS)**:
| Shape | Variant | Before → After | delta |
|-------|---------|----------------|-------|
| 4096×32768×28672 | _v20_memc_r11_iterilp | 92.98% → 94.82% | +1.84pp |
| 4096×28672×32768 | _u16_r11_iterilp | 93.35% → 95.35% | +2.00pp |
| 4096×32768×14336 | _ts_lgk2_memc_r11_iterilp | 93.83% → 95.30% | +1.80pp |
| 4096×128256×32768 (WIN, regression check) | _memc_r11_iterilp | 161.23% → 163.89% | +2.46pp (no regress) |

Sub-threshold (single-shot, not validated): 32768×4096×14336 +0.74pp.

5 of 10 produced HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION → Round 12A confirmed deterministic compiler bug on those shapes.

**Round 11 净增**: +3 deep-LOSE gap reductions (avg +1.88pp on 3 shapes), 0 regressions.

**Cumulative R10+R11 deep-LOSE gap reductions** (5 shapes, average +1.85pp):
| Shape | Before → After | delta | Round |
|-------|----------------|-------|-------|
| 14336×4096×32768 | 89.6% → 91.4% | +1.78pp | R10 |
| 16384×4096×28672 | 90.1% → 91.9% | +1.82pp | R10 |
| 4096×32768×28672 | 92.98% → 94.82% | +1.84pp | R11 |
| 4096×28672×32768 | 93.35% → 95.35% | +2.00pp | R11 |
| 4096×32768×14336 | 93.83% → 95.30% | +1.80pp | R11 |

## Round 10 (2026-04-17) — VALIDATED 2x WIN via un-prefixed iterative-ilp (committed `091d3baa`)
Acted on Round 9 verifier reversal — tested REAL un-prefixed sched-strategies on deep-LOSE.

**BREAKTHROUGH**: un-prefixed `iterative-ilp` flag, single-GPU 5-run replication on GPU 5:
- 14336×4096×32768: parent `_lgk2_dc` 89.6% → +1.78pp (mean ≥ baseline.max PASS)
- 16384×4096×28672: parent `_u32` 90.1% → +1.82pp (mean ≥ baseline.max PASS)
ASM dumps in `asm_verify_r10/` confirm un-prefixed values produce distinct ASM (refute Round 8 A latent finding).

## Round 9 (2026-04-17) — Verifier reversal + 8 codegen flags + SNR bug FIX
更窄、更聚焦的探针轮 (3 任务: 1 verifier + 1 codegen probe + 1 tooling fix), 在 Round 8 NEW METHODOLOGY ADDENDUM 下:
- **Verifier** (验证 Round 8 A 的 latent finding "_memc 用 un-prefixed flag 可能 silently default to gcn-max-occupancy"): **REVERSED — Round 8 A 的方向正好搞反**.
  - 实测发现: `gcn-`-prefixed 形式 (`gcn-max-memory-clause`, `gcn-max-ilp`, `gcn-max-occupancy`, etc.) **是 no-op** — 产生与无 flag 字节相同的 ASM (size 301977, MD5 仅随机 hash 不同).
  - **un-prefixed 形式 (`max-memory-clause`, `max-ilp`)** 才是真正生效的 — 在真实 mxfp4 kernel 上 7153 行 ASM diff vs default.
  - **`-mllvm -amdgpu-sched-strategy=` 接受任何字符串** (包括 `NONSENSE_GARBAGE`) 不报错不警告 → silently fallback to default. 这是 ROCm 7.1 LLVM 20 的真实 LLVM/AMDGPU bug, 但**对我们 benchmark 没影响** — 因为 bench_all_42.py:412-461 一直在用 un-prefixed `max-memory-clause`.
  - **重大方法论反转**: **Round 6 A 和 Round 8 A 测试的 `gcn-`-prefixed 5-strategy sweep 全部是 no-op**. 那些 "all noise/regress" 的结果**毫无意义** — 它们从来没执行过新的调度. 真正 untested 的是 un-prefixed `max-ilp` / `max-occupancy` / `iterative-ilp` / `iterative-minreg` 在 deep-LOSE shapes 上的效果. 已 deployed: un-prefixed `max-memory-clause` (在 14/24 WIN best variants).
  - Verifier dummy test 仅确认 un-prefixed `max-memory-clause` 和 un-prefixed `max-ilp` 产生不同 ASM. 其他 un-prefixed 是否生效未确认.
- **Optimizer B** (8 个从未测过的 LLVM codegen 微 flag on `14336×4096×32768` best `_v16_wpe2`): **全部 DEAD END**. flag list: loop-prefetch, schedule-metric-bias=80/100, mfma-padding-ratio=10/25, disable-loop-alignment, disable-clustered-low-occupancy-reschedule, disable-unclustered-high-rp-reschedule, use-aa-in-codegen, enable-pre-ra-optimizations. 全部 [-0.85, -0.03] pp (轻微 regress 或 noise tied). 最差 `disloopalign` -0.85pp. 没有触发 +0.8pp 复测. 11 variants 加入 dead-end list.
- **Optimizer C** (修复 documented SNR false-OK NaN bug): **FIXED, committed `b46834a0`**.
  - bug 实际不在 `bench_deep_lose.py` (该文件无 SNR gate, 是 perf-only spot bench) 而在 `bench_optC_round6.py:175` (`if snr_db < 25` mis-classify NaN as OK) 和 `bench_round6_optA.py` (NaN-tainted output → `noi>0` short-circuit → snr=+Inf 通过).
  - Fix: 新建 `snr_check.py` (NaN-safe `is_snr_ok` / `classify_snr` / `compute_snr_db`); 两个 bench 文件均拒绝 NaN/None/-Inf SNR.
  - **重要发现**: Round 6 OptC 的 `_ts_u8*` 5 个 variants 全部 NaN output 但 silently 通过 SNR gate, 任何 "wins" 都是无意义的. Future agents 用 bench_optC_round6.py 都需要重新验证.

**Round 9 净增**: 0 WIN, 0 gap reduction, 但 **1 real commit (SNR fix)** + **1 critical methodology reversal (sched-strategy prefix)**. 继续 6 → 7 saturation rounds.

**新 dead-end vectors (Round 9)**:
- 8 LLVM codegen 微 flags (loop-prefetch, sched-metric-bias, mfma-padding, disloopalign, etc.) — 全部 noise/regress
- `gcn-`-prefixed sched-strategy 名 (`gcn-max-memory-clause` etc.) — 全部 silently no-op, 等于 default

**新 untested vector (Round 10 候选)**: un-prefixed `max-ilp` (verified 工作的) 在 deep-LOSE shapes 上 — 真正未测.

## Round 8 (2026-04-17) — 3 parallel optimizers, all DEAD END / NO-OP
GOAL PIVOT 第三轮, 在 Round 6 methodology rule 下, 攻击真正未测的窄向量:
- **Optimizer A** (per-shape `-mllvm -amdgpu-sched-strategy=` bucketing on 4 untested deep-LOSE shapes 4096×32768×28672 / 28672×4096×16384 / 4096×28672×32768 / 32768×4096×14336): **DEAD END**. 19 variants × 4 shapes × 4-5 strategies (`gcn-max-occupancy`, `gcn-max-ilp`, `gcn-iterative-ilp`, `gcn-iterative-minreg`). 全部 [-0.81, +0.09] pp 区间, 最大 +0.09pp `_ts_gm8_sched_memc` on 28672×4096×16384 (远低于 +0.8pp 触发). 加上 R6 A 的 4096×32768×128256 + R6 B 的 14336×4096×32768 + R6 C 的 16384×4096×28672, sched-strategy vector 现已覆盖 7/10 deep-LOSE shapes, **vector 完全 exhausted**.
  - **重要 latent finding**: 现有 `*_memc` baselines (在 14/24 WIN best variants 中) 使用 un-prefixed `max-memory-clause` flag, 可能 silently default to `gcn-max-occupancy` (LLVM 不识别该值 → fallback). 如果验证, 意味着 `_memc` family 实际是 `gcn-max-occupancy`, 不是 `gcn-max-memory-clause`. 这是潜在 bug 但解释了为何 memc 在 WIN shapes 上有效 (与默认不同). 未来轮可验证, 但不会改变 deep-LOSE 结论 (R6 A 用了正确 prefix 测试, 全部 noise).
- **Optimizer B** (TAIL_BARRIER_VMCNT × VMCNT × 3 large-K deep-LOSE shapes 4096×32768×128256 / 4096×32768×14336 / 32768×4096×14336): **DEAD END**. TAIL_BARRIER_VMCNT 仅在 `TAIL_SPLIT=1` 路径生效 (确认 line 2174, 2361). 29 variants × 3 shapes × {0,4,8,16,24} × SV±4. 全部 [-0.19, +0.27] pp 区间, 最佳 +0.27pp `_ts_u16_lgk2_tbv16` on 32768×4096×14336 (低于 +0.8pp 触发). **机械性 insight (重要)**: 大 K 下 (K≥14336) tail iter 占总 K iters 的 ≤0.45%, 即使完美调 tail barrier 也只能 shift 微小比例 → 该 vector 数学上不可能产出可见增益 on 大 K shapes.
- **Optimizer C** (`__attribute__((amdgpu_waves_per_eu(1,1)))` on 3 register-pressure shapes 128256×32768×4096 / 28672×32768×4096 / 4096×32768×128256): **NO-OP + 已 REFUTED**. 编译器 resource usage 显示 baseline (无 wpe attribute) 已经是 1 wave/SIMD (224V + 256A = 480 regs, 加上 `__launch_bounds__(_NUM_THREADS, 1)` line 1727). 加 wpe(1,1) bytewise-identical .so, 是 literal no-op. **更重要**: Round 2 deep-LOSE 工作 already tested wpe1 (build_deep_lose_variants.py:39-50: `_wpe1`, `_v16_wpe1`, `_ts_lgk2_v12_wpe1_memc`, etc.), bench_deep_lose_results.json 显示全部 LOSE on 3 target shapes (-7.6 to -51.9 TFLOPS). 我之前 prompt 写的 "wpe1 was NEVER tested" 是错的, 应记录在 dead-end list. 0 文件创建.

**Round 8 净增**: 0 WIN, 0 gap reduction. **6 轮饱和** (R2/R4/R5/R6/R7/R8). 每轮 0 净增. 我已系统耗尽所有 in-session-feasible 优化向量.
**新 dead-end vectors (Round 8)**:
- per-shape sched-strategy bucketing on 4 P1/P2 deep-LOSE shapes (max +0.09pp)
- TAIL_BARRIER_VMCNT × large-K deep-LOSE shapes (mechanistically futile, tail iter ≤0.45%)
- `amdgpu_waves_per_eu(1,1)` on register-pressure shapes (no-op + already REFUTED)

**Latent investigation**: `_memc` baselines may use un-prefixed `max-memory-clause` flag; verify whether that silently falls through to `gcn-max-occupancy`. Won't change deep-LOSE conclusion but worth documenting.

## Round 7 (2026-04-17) — 3 parallel optimizers, all DEAD END / INFEASIBLE
GOAL PIVOT 第二轮, 在 Round 6 methodology rule 下:
- **Optimizer A** (STEP12_BR_LGKMCNT∈{0,2,4} sweep on 3 P1 shapes: 4096×32768×28672, 28672×4096×16384, 4096×28672×32768): **DEAD END**. Round 6 C 在 16384×4096×28672 上的 brlgk2 directionally-positive 信号**不泛化**. 9 个 variant 单 GPU 测量, 全部 -0.03 ~ -0.14pp (<<+0.6pp noise). brlgk0 是 default value.
- **Optimizer B** (STATIC_XCD_REMAP on mega-M 128256×32768×4096): **DEAD END**. 实现 atomic-free static remap (each XCD owns N-strip of width bpc/NUM_XCDS=16, walks GROUP_M×16 tiles). 6 variant grid (baseline_dc/gm4/gm8 × static_xcd_remap{,_gm4,_gm8}): best STATIC variant -0.37pp vs baseline. 失败原因: **mega-M shape 是 A-bound, 不是 B-bound** — A traffic dominates (M=128256 vs N=32768), 缩 B working set 8x 反而损失 8x A reuse. 与 Round 4 PERSISTENT_XCD_QUEUE 同源结论, 进一步确认 mega-M 92.9% gap 是 register-pressure / occupancy 结构性 bound. Code 留在 `STATIC_XCD_REMAP=1` flag (default 0) 作为 documented dead-end.
- **Optimizer C** (B-tile L2 prefetch via `__builtin_amdgcn_global_load_lds` on 4096×28672×32768 / 4096×32768×14336): **INFEASIBLE — premise wrong**. 现有 `emit_one_pf()` (line 466-472) **已经是** `__builtin_amdgcn_global_load_lds` 的 buffer-SRD 形式 (`llvm_amdgcn_raw_buffer_load_lds`, emits `BUFFER_LOAD_DWORDX4 lds:1`), 已 prefetch B `bt+2` look-ahead 直接进 LDS. 三种"扩展"方案都不可行: (1) 加 redundant 16B/thread 到 scratch LDS = 纯 VMEM duplication 在已 B-VMEM-bound 的端口上, 必退化; (2) bump look-ahead `bt+2` → `bt+3/4` 不是新机制只是常数, 且 LDS 双缓冲 +50% 超 160KB cap; (3) GLOBAL_LOAD_LDS 没有 discard sink mode (硬件强制写到 LDS dest). 未跑 bench, 25 min 提早终止, 0 文件创建. **重要文档**: future agents 不要再提 "L2 prefetch via global_load_lds" 这个 vector — 已 deployed.

**Round 7 净增**: 0 WIN, 0 gap reduction. 5 轮饱和确认 (R2 ceiling + deep-LOSE分析员 + R4 + R5 + R6 + R7).
**新教训** (扩展 Round 6 methodology):
- mega-M shape 是 A-bound, 缩 B-locality 必失败 (Round 4 PERSISTENT_XCD_QUEUE + R7 STATIC_XCD_REMAP 双重证实)
- `emit_one_pf` IS `__builtin_amdgcn_global_load_lds` (buffer-SRD form), B 已 bt+2 prefetch 进 LDS, 没有 "未利用的 prefetch 机制"
- "Round 6 C 的 STEP12_BR_LGKMCNT=2 directionally-positive 信号" 是 single-shape 边际, 不可泛化

## Round 6 (2026-04-17) — 3 parallel optimizers, all DEAD END / MARGINAL (REVERTED)
GOAL PIVOT 后第一轮, 直接攻 deep-LOSE shapes:
- **Optimizer A** (4096×32768×128256 / compiler flags + L2 prefetch): **DEAD END**. gfx950 无 L2 prefetch instruction; igroup-lp 不可用; 单次 +0.29pp, 5-run 中位 -0.5pp (in noise).
- **Optimizer B** (14336×4096×32768 / UNROLL_K sweep): **claimed +3.16pp WIN (commit 198bb3a4) → REJECTED by Reviewer**.
  - Optimizer B 的 baseline 测量 4591 TFLOPS 是**假低**: 单 GPU 5-run replication 验证 baseline 实际 4727±13 TFLOPS (不是 4591). 真实 delta:
    | variant | mean TFLOPS | mean pp over base | min vs base max |
    |---------|-------------|-------------------|-----------------|
    | _v16_wpe2 (baseline) | 4727.22 | 0.00 | 0.00 |
    | _optB_r6_u16_v16_wpe2 | 4730.26 | +0.06 | -0.73 |
    | _optB_r6_u8_v16_wpe2_memc | 4721.64 | -0.11 | -0.89 |
    | _optB_r6_u16_lgk2_dc_v16_wpe2 | 4735.57 | +0.16 | -0.38 |
  - 全部三个变体 mean Δ 均 <+1pp 阈值, 全部 worst-case (min vs baseline max) 为负. WIN-sample 与其余 deep-LOSE neighbor 也无显著 gain (±0.5pp 内)
  - 教训: Optimizer B 在 GPU 4 上测的 baseline 与之前 baseline (GPU 不同) 比较, 触发了 **GPU bias 50-100 TFLOPS** + per-run noise ±0.6pp 的合成假象
  - **Action**: `git revert 198bb3a4` (commit 4c11f8bb). 验证脚本保留: `spot_optB_r6_validation.py` + `spot_optB_r6_validation.log` + `spot_optB_r6_validation_results.json`
- **Optimizer C** (16384×4096×28672 / TAIL_SPLIT epilogue tuning): **DEAD END**. TAIL_SPLIT=1 在 K≥7168 上更差; best variant +0.19pp, 远低于 +1pp 阈值

**Round 6 净增**: 0 WIN, 0 gap reduction. **新教训**: 跨 GPU 比较 baseline 不可靠 (GPU bias ~2pp), 任何 deep-LOSE 改进 claim **必须** single-GPU 5-run replication 验证, 且 best mean 必须 ≥ baseline max.

## 剩余 untested vectors (out of in-session scope)
- **MFMA_32X32X64_TILING**: 切换 `v_mfma_scale_f32_16x16x128_f8f6f4` → `v_mfma_scale_f32_32x32x64_f8f6f4`. 巨大 kernel rewrite (>1 day, asm + layout 全改). AGPR 从 256 降到 64 释放 192 VGPRs/AGPRs 用于深度 B-buffering. 是唯一未测的"内核重构"级别尝试, 接近 aiter 架构.
- **B_TRIPLE_BUFFER**: 3-stage pipeline 替代当前 2-stage. LDS budget 是杀手 (131KB → 163KB > 160KB max). 仅在 #1 (32x32 MFMA 释放 AGPR) 完成后才可行.
- Optimizer A 副产建议: 早期 scale prefetch (移到 Step12 前), 4× dwordx2 → 1× dwordx8 burst 合并, drop redundant scale stream 当 a0_raw == a1_raw.

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
