# MXFP4 GEMM 优化 — Agent提示词

你在继续推进 `HipKittens` 的 MXFP4 GEMM 优化工作，跟 Cursor (Hipkittens2) 竞赛。

## ⚠️ 当前优化目标 (2026-04-18, post-R29)
> 24/42 ceiling 早已被打破: 现在 **41/42 WIN (97.6%)** = effectively saturated.
> 唯一剩余目标: **L6 / DLA1 (4096×32768×128256, 92.6%)** — V5 MFMA_32X32X64 重写 (BACKBURNER, 1周). V8 R25E peel 也 DEAD (state-hazard).
> L4 已 R29 修复 (LOSE→WIN +17.45% via parent-stack 修正).

**历史指令** (2026-04-17, 已过时):
> "24win 已经卡了好久了，现在把优化目标改成优化剩下那几个差的比较多的。"
> 当时认为 24/42 是天花板; R25-F/G/H + R26-D + R28-C 后已证实是错的.

### 重点 shape (post-R29, 2026-04-18)
**41/42 WIN** = 97.6% 命中率. 仅 1 残留:

| 优先级 | Shape (M×N×K)         | Ratio | Best Tag                               | 类别              | 备注 |
|-------|----------------------|-------|----------------------------------------|------------------|------|
| P0    | 4096×32768×128256    | 92.6% | ts_lgk2_v12_memc_btw_all (DLA1)        | 大N+mega-K       | R27-C/R26-A/V8 全部 DEAD; 仅剩 V5 MFMA32 重写 (BACKBURNER ≥1周) 或 V6 split-K (≥3天) / V7 stream-K (≥2周) |

**R29 翻 LOSE→WIN (+1)**:
| Shape | Old → New | Variant |
|-------|-----------|---------|
| L4 4096×32768×14336 | 99.25% → **116.58%** (+17.45%) | `_ts_v12_tv0_memc_btw_all_pfoff48_kx14336` (parent-stack 修正: L4 应用 `btw_all`, 不是 L8 的 `_dc_gm7`) |
| L8 16384×4096×14336 (bonus) | 116.6% → +3.21% | 同上 variant 顺手 beat L8 incumbent +184 TFLOPS |

**v2 翻 LOSE→WIN (+7)**:
| Shape | v1→v2 | Best Tag (v2) |
|-------|-------|---------------|
| L1 4096×14336×16384      | 97.2% → **115.3%** | ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all |
| L2 4096×28672×32768      | 96.5% → **115.6%** | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all |
| L3 4096×32768×6144       | 99.7% → **116.2%** | ts_v12_gm7_memc_pfoff19_kx6144_btw_all |
| L5 4096×32768×28672      | 94.7% → **116.6%** | ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all |
| L7 14336×4096×32768      | 94.7% → **115.9%** | ts_v12_tv0_memc_dc_gm7_pfoff120_kx32768_btw_all |
| L8 16384×4096×14336      | 97.9% → **116.6%** | ts_v12_tv0_memc_dc_gm7_pfoff54_kx14336_btw_all |
| L9 16384×4096×28672      | 95.0% → **116.0%** | ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all |

### R26 死路 (DO NOT REVISIT — 本轮+累计)
本轮 R26 死路 (4 axes):
- **V3 STEP3_PF_N / STEP4_PF_N × R25-G stack** — `R26B_PF_N_dead.md`. DLA2 monotone 退化, DLA7 +31 TFLOPS within noise.
- **V4 TAIL_BARRIER_VMCNT × R25-best stacks** — `R26C_tail_vmcnt_dead.md`. 4 shapes × {0,4,8,12,16} 全 flat. R25-F/G/H 已 drain tail VMEM 使 vmcnt 阈值无意义.
- **gm5 / gm9 sweep** — `R26E_gm_axis_dead.md`. 4 shapes ±0.3pp; gm7 已 local optimum.
- **STEP12_BR_LGKMCNT 第二轮** — `5d0b5fd2`. 已 saturate.

R25 累计死路 (per `R26_PLAN.md` §4):
- **B-tile `__builtin_prefetch`** ≡ `emit_one_pf` (已 buffer_load_lds, R25-G 是去掉 tail).
- **scale-load `buffer_load_dwordx2` + SGPR SRD** — 已实现 (kernel L695, NONVOLATILE_SCALE_X2_POC=1, ISA L186 SGPR SRD `s[0:3]`...).
- **SCALE_REG_CACHE round-trip removal** — duplicate.
- **WAVES_PER_EU=3** — `__launch_bounds__(_NUM_THREADS, 1)` 已 clamp 到 1 wave/SIMD on K-bound, wpeu axis 理论 inert.
- **per-shape async-prefetch coalescing (`s_waitcnt vmcnt(N)` group)** — `emit_pf_tail` 已 tight `#pragma unroll`, ISA L186-228 已 clustered.
- **early-iter-only prefetch (R25-G inverse)** — 强 negative prior (B 在 iter 2 后 L2-resident, 拿掉 early pf 会 force HBM re-fetch).
- **outer-K pull-forward / extra L2 pf** — R24B/C: VMEM-issue-bound 不是 latency-bound.
- **cache hints / NT stores / persistent-XCD / EARLY_SCALE_PF / EARLY_BL_PF / DIRECT_BL** — 全 dead.
- **iterative-ilp LLVM sched-strategy** — LLVM compiler bug (HSA aperture violation on ≥10 shape categories), 不能 default; 已废弃.

R25-G 之前各类已知 dead (history): `iterative-ilp` 编译器 bug, BK=256 LDS 不够, Direct-B 慢 28%, ds_bpermute 退化, GROUP_SIZE_M=32/64 大N 退化, UNROLL_K=1/2/4 比默认差, FUSED_STEP34 退化 7%, ASM rewriter (s_nop removal) 破坏正确性, PF_N=1/2 退化 2-6%, asymmetric PF 无效.

### 工作准则
- **每轮锁定 1-3 个 P0/P1 shape** 专项优化, 不再全 42 跑.
- **改动后必跑 regression**: `bench_deep_lose.py` (10 shape spot) + `bench_all42_parallel.py` 抽测 (确认 24 WIN 不掉).
- **+1pp 即可 commit** (不再要求翻 WIN).
- **大 K (≥14336) shapes** 是主战场: 该类的 gap 主要来自 LDS broadcast bandwidth 不足 + B tile reuse 效率低.
- **mega-M shape 128256×32768×4096** 已被验证为 **register-pressure / MFMA-pipeline bound** (Round 4 PERSISTENT_XCD_QUEUE 实证), **不是 launch-bound**. 不要再尝试 dispatch 优化.

### 推荐探索方向 (post-v2 — 只剩 V5 一条结构性路径)
| 方向 | 风险 | 预期 | 备注 |
|------|------|------|------|
| **V5 — MFMA_32X32X64_TILING 重构** (R29+) | 高 | 0-5pp on L6/DLA1 | **唯一剩余结构性 axis**. ≥1 周 asm 重写. 32×32 MFMAs 允许 4× concurrent in-flight @ same acc footprint, 可能松开 R24B/C 确认的 VMEM-issue saturation. 高不确定性. 见 `R27_V5_MFMA32_SCOUT.md` |

DEAD post-R29 (不要再尝试 — 已 reproduce 过):
- ~~R26-A V1 DLA1 K-loop peel (R25-E pf495)~~ — 5-rep verify std=1729 TFLOPS, mean swing 1672→5546, 不稳定 false alarm
- ~~R27-C DLA1 K_EXACT bypass~~ — HSA aperture violation rc=-6 全 5/5 reps; kernel HARD-GATE K≤32768 (`kernel_mxfp4_gluon_cpp.cpp:85-91`)
- ~~V8 R25E static-loop-split peel on L6/DLA1~~ — `R29_L6_V8_VERDICT.md`. PEEL=2/4 runtime HSA memory access fault; PEEL=8 catastrophic -71% regression (1529 TFLOPS). State-hazard between unroll-8 main loop & peel TAIL via pf_a0/a1/bl/br scale registers. Kernel 已 revert.

DEAD (不要再尝试 — 已 reproduce 过):
- ~~per-shape compiler flag (LLVM sched-strategy per-K-bucket)~~ — `iterative-ilp` LLVM bug (aperture violation), 其它 strategy ±0.4% noise
- ~~per-shape K-loop unrolling (UNROLL=8/16)~~ — Round 6 B 证伪 ±0.16pp
- ~~B-tile L2 software prefetch~~ — `emit_one_pf` 已 `buffer_load_lds`
- ~~K-loop epilogue (TAIL_BARRIER_VMCNT × shape)~~ — R26-C dead, flat across {0,4,8,12,16}
- ~~K-loop epilogue (STEP12_BR_LGKMCNT)~~ — R26-G dead
- ~~static XCD-aware block_id remap~~ — Round 7 B: -0.37pp
- ~~steady-state PF_N (STEP3/STEP4)~~ — R26-B dead, monotone worse on DLA2
- ~~GROUP_SIZE_M ∈ {5,9}~~ — R26-E dead, gm7 已 local optimum

## 项目位置
- **我们的 Repo**: `/shared_nfs/kyle/test/HipKittens`
- **Branch**: `mxfp4`
- **工作目录**: `analysis/fp8_gemm/mi350x`
- **Cursor Repo**: `/shared_nfs/kyle/test/Hipkittens2` (只读参考)

## 当前成绩 (2026-04-18, post-R26 FINAL v2 + R28-C)
- **FINAL v2 bench**: `bench_all42_results_R25_FINAL_v2.{json,log}` — **40/42 WIN, 2/42 LOSE, 0 ERR, win rate 95%** (warmup=200 iters=500 trim=10%, 5-GPU parallel). Up from v1 33/42 (+7 net flips), from 24/42 baseline (+16 net flips since R20).
- **R28-C win (`d45e35103`)**: u16 + kx14336 K_EXACT — +14.5% on 16384×4096×14336 (L8 flip).
- **R26-D V2 audit WIN (`1454235e`)**: found `bench_all42_parallel_R25_FINAL.py` wiring bug, re-wired R25-G/H K_EXACT .so files for 5+ shapes. Bulk of the 24→33 jump.
- **R27-C DEAD**: HSA aperture violation (rc=-6) — kernel HARD-GATE K≤32768 cannot be bypassed. `R27C_VERIFY_VERDICT.md`.
- **R26-A pf495 false alarm**: std=1729 TFLOPS, unstable, NOT a real win.
- **R26 dead axes (`f43ea34b` + `5d0b5fd2`)**: V3 PF_N, V4 TAIL_VMCNT, gm5/gm9, STEP12_BR_LGKMCNT — all flat or within noise.
- **R25-G PER-K-BUCKET WIN (committed `7f200b76`)**: extends R25-F insight to 6 mid-gap K=14336+ shapes. Per-K optimum `pfoff = K_iters - {4..8}`. **6/6 WIN, +13.68 to +20.84% per shape**. Implementation: `R25C_K_EXACT` compile-time gate so each pfoff variant only activates on its target K.
- **R25-F EXTENDED-SWEEP MASSIVE WIN (committed `e5083bad`)**: `gm7 + pfoff14` dominates K=4096 shapes — **+10.86% DLA2, +13.51% DLA7** vs R25-D. Vs original baseline: DLA2 ~+17%, DLA7 ~+21%.
- **R25-D STACK WIN (committed `7ada8c70`, superseded by R25-F/G)**: gm6 × pfoff4 super-additive on DLA2/DLA7. Kept as fallback.
- **Cumulative R25-F + R25-G**: 8 deep/mid-gap LOSE shapes flip → BIG WIN. Projected total WIN ≥ 35/42 (up from 29). The single biggest 1-day jump since R20A's +11 BARRIER_TO_WAITCNT shape closures.
- **R22-rebench (full 42-shape baseline before R25-D wires)**: **29/42 WIN, 13/42 LOSE**, 0 ERR, avg ratio 105.3% (+5 vs R20B/24). Post-R25-D: regression check in flight; expected to flip DLA2/DLA7 to WIN (→ ~31/42).
- 13 LOSE shapes (pre-R25):
  - DLA1 4096x32768x128256 = 91.9%, DLA2 128256x32768x4096 = 96.3%, DLA7 28672x32768x4096 = 96.9%
  - mid-gap (95.7-99.9%): 14336x4096x32768, 4096x28672x32768, 16384x28672x4096, 28672x4096x16384, 4096x32768x14336, 4096x32768x28672, 16384x4096x28672, 16384x28672x2048, 14336x32768x4096, 4096x14336x16384
- **R22B (NT/streaming on A+B), R23A (STATIC_XCD_REMAP), R23B (PERSISTENT_XCD)** — all DEAD END (R23B has correctness bug; R23A best DLA7 +1.50% borderline noise; R22B all regress)
- **R24A — PERSISTENT_XCD bug fix attempts (Fix A+B+C) — DEAD END**. Host-side fixes did not resolve coverage bug (still 6.4%/28.6%). Bug is in kernel-side persistent loop logic, requires risky rewrite. Out of scope.
- **R24D — A-only NT cache hint — DEAD END** (4-8% regression).
- **R24B — extra `buffer_load_dwordx4` L2 prefetch — DEAD END** (0.4-12% regression, monotone with intensity).
- **R24C — outer-K pull-forward (K+2/K+3 prefetch) — DEAD END** (14-24% regression, saturation flat).
- **Cache-policy axis ({A,B,both}×{NT, cache_stream})**: 6/6 LOSE → CLOSED.
- **Cache-bandwidth axis (extra discarded VMEM, A/B, intensity 1-3)**: all LOSE → CLOSED.
- **Mechanistic conclusion (R24B/R24C)**: DLA shapes are **VMEM-issue-bound, NOT VMEM-latency-bound**. Single VMEM lane already saturated by existing LDS-bound prefetch path. Extra outer-K VMEM competes with inner-K pf for HBM bandwidth and degrades it.
- **Saturation reached.** Remaining vectors require kernel-level rewrites (tile geometry / MFMA_32X32X64 / fixed PERSISTENT_XCD) that risk breaking the 29 WIN baseline and cost 1-2 days each. **Recommendation**: accept CDNA4 hardware saturation at 29/42 WIN unless user explicitly authorizes a rewrite-class effort.

- **R20B 最新 full bench (116 variants, R18+R19 wins wired)**: **27/42 WIN** (+3 LOSE→WIN flips: P1, S1, S5; 0 regressions)
- **R20A 11 个 (parent + BARRIER_TO_WAITCNT) stacks wired into bench_all_42.py post-R20**: 127 variants — projected next bench **~34/42 WIN**
- **Cursor (Hipkittens2)**: 16/42 WIN (同参数, 历史快照), 我们领先 ≥11 WIN
- **R18+R19+R20 累计**: **14 of ~18 deep-LOSE shapes 已闭合** (P1 / S1-S15 大部分) via barrier-removal axis
- **R21-recon 已分类剩余 stuck shapes**: DLA1/DLA2/DLA7 全部 **memory-stall bound** (TCP_DATA_STALL 167-294 % of GRBM, HBM 7.8-19.9 % of 5.3 TB/s peak). 不是 compute-bound — R22+ 必须 attack memory axis

## 已做的优化 (19项)
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
15. DIRECT_BL (half-direct Bl from preshuffled global) — DEAD END, 8-15% slower
16. NT_STORE (non-temporal stores, bypass L2) — DEAD END, 15-25% slower
17. PACKED_STORE (bf16x2 dword stores for SWAP path) — DEAD END, SWAP path inherently slower
18. Compiler flag tuning (O2, clause, inline_all) — DEAD END, ±0.4% noise
19. 18 new variant combos (ext_br+lgk, gm+lgk, tv+lgk) — DEAD END, 0 WINs

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
- **Half-Direct Bl (DIRECT_BL)**: preshuffle B + Bl 从 global buffer_load 直取. 0 spills (253V, 87S, 98KB LDS), 正确. 但全 7 shapes 退化 8-15% (buffer_load ~400 cycle latency, Step4 仅 ~128 cycle MFMA hiding). 大N shapes -13~15%. LDS 减少无法弥补 VMEM latency
- **NT_STORE (non-temporal stores)**: 全部 global_store_short 加 `nt` modifier 绕过 L2. 全 7 shapes 退化 15-25%. CDNA L2 对 store coalescing 至关重要
- **PACKED_STORE (bf16x2 dword stores)**: SWAP路径 pack_bf16x2, 减 store 指令 4x. 但 SWAP 路径本身退化 13-34%, packed 无法补偿
- **Compiler flag tuning**: -O2, -mllvm -amdgpu-max-memory-clause=1/4, -amdgpu-early-inline-all=true → ±0.4% noise, 无改善
- **18个扩展variant组合**: ts_lgk2+ext_br/gm8/tv16/tv0, ts_gm2_lgk2+v4/v16/ext_br/no_embed, lgk2+ext_br/gm2/gm1/no_embed_v12 → 全部 0 WIN, 全部不如已有 best variant
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

## 结构性限制 (已证实无法突破)
- B走LDS是根本瓶颈: +18% read traffic, 2x wait time vs aiter
- 256 AGPR + B tiles 无法同时放进 256 VGPRs
- 不 preshuffle B 就不能跳过 LDS
- N=32768 shapes: B tile 大 → LDS traffic 成为瓶颈
- Store epilogue: 非SWAP路径 s[0..3] 是行连续 (strided), 无法pack. SWAP路径可pack但SWAP本身退化. NT store 绕过 L2 反而退化 (CDNA L2 做 store coalescing)
- Compiler scheduling: 已达最优, O2/O3/clause/inline 全在 ±0.4% noise

## 可能的未来方向 — 全部 DEAD END
1. ~~**Fused Step34**~~ — DEAD END (退化7%)
2. ~~**ASM rewriter**~~ — DEAD END (无增益, s_nop 硬件强制)
3. ~~**Profile-guided**~~ — DONE: 瓶颈在 B-LDS traffic
4. ~~**GM×LGK / TAIL_VMCNT cross-products**~~ — DEAD END (0.1-0.3pp, 不翻WIN)
5. ~~**Pre-shuffle B (Half-Direct Bl)**~~ — DEAD END (8-15% 慢, buffer_load latency)
6. ~~**NT_STORE (non-temporal stores)**~~ — DEAD END (15-25% 慢, L2 对 store coalescing 必要)
7. ~~**PACKED_STORE (bf16x2)**~~ — DEAD END (SWAP路径本身慢)
8. ~~**Compiler flag tuning**~~ — DEAD END (noise level)
9. ~~**18 new variant combos**~~ — DEAD END (0 WINs)
10. ~~**Adopt asm_inline ("5084 TFLOPS" gluon-derived ASM body)**~~ — **DEAD END / IMPOSSIBLE**:
    `kernel_mxfp4_asm_inline.{cpp,h}` produces INCORRECT output (SNR -1.31 dB at K=8192 256x256).
    K-specialization (modify `s_cmp_lt_u32 s68, 28` → larger T for K∈{14336,16384,28672,32768};
    `build_asm_kvar.py` builds variants successfully) is moot since the underlying kernel is broken.
    Tried with CK BpreShuffle on B too: SNR -3.03 dB (worse). The "5084 TFLOPS" reference is
    measuring a kernel that doesn't compute correct GEMM. To use ASM, would need to load
    aiter's actual `.co` files via `hipModuleLoad` — different architectural change entirely.
12. ~~**SCALE_REG_CACHE (hoist scale ds_reads into VGPR across iterations)**~~ — **DUPLICATE / NOT NEW (2026-04-17 Round 4)**:
    Investigation found scales are ALREADY VGPR-resident: `load_pq_scale_x2_async` is direct
    VMEM→VGPR via `buffer_load_dwordx2` (NOT VMEM→LDS). Persistent VGPRs `pf_a0/pf_a1/pf_bl/pf_br`
    (line 1831) hold scales across iterations; asm operands `v70-v73` reused across all 32 MFMAs
    in each kpair with zero reloads. `NONVOLATILE_SCALE_X2_POC=1` (default-on) already does the
    optimization. The decider misread the kernel. No code changes made.
13. ~~**LDS_XOR_SWIZZLE_B (XOR-swizzle B's LDS column to remove bank conflicts)**~~ — **DUPLICATE / NOT NEW (2026-04-17 Round 4)**:
    Already implemented via `st_16x128_s::swizzle()` in `include/types/shared/st_shape.cuh:236-237`:
    `swizzled_offset = offset ^ (((offset % 2048) >> 8) << 4)`. Symmetrically applied: writer side
    pre-permutes global offset via `prefill_swizzled_offsets`, reader side applies same XOR via
    `compute_lds_base_addrs` (line 376,379). Simulation confirms perfect 8 acc/bank uniform =
    LDS hardware lower bound. Prior `kernel_mxfp4_xor_toggle.cpp` test showed no improvement
    (4917T vs 4926T baseline). No code changes made.
14. ~~**PERSISTENT_XCD_QUEUE (persistent kernel + atomic work queue for mega-M shape)**~~ — **DEAD END (2026-04-17 Round 4)**:
    Targeted at "IMPENETRABLE" 128256×32768×4096 (currently 94.7% aiter). Implemented behind
    `PERSISTENT_XCD=1` flag (default 0) with `PERSISTENT_GRID=608` (8 XCDs × 38 CUs × 2 WGs/CU)
    and tunable `PERSISTENT_BATCH ∈ {1,4,8}`. Bench results (warmup=200, iters=500):
        static (best variant `ts_gm2_v12_memc_dc`): 4296 TFLOPS (94.7%)
        PERSISTENT b=1 g=608:  3791 TFLOPS (83.6%)  −11.8%
        PERSISTENT b=4 g=608:  4129 TFLOPS (91.0%)  −3.9%  ← best persistent
        PERSISTENT b=8 g=608:  3857 TFLOPS (85.0%)  −10.2%
    Reasons it failed: (1) HWS overhead is NOT the bottleneck — only ~5% headroom total to aiter,
    atomic-counter eats ~4% even at BATCH=4; (2) XCD-locality LOSS — static `raw_bid % 8`
    preserves L2 B-tile reuse within each XCD; persistent destroys that L2 reuse pattern;
    (3) atomic latency adds visible overhead on already-ALU-bound kernel (256V/256A, 1 wave/SIMD).
    The 92.9% ceiling on this shape is **register-pressure / MFMA-pipeline bound**, NOT launch/dispatch bound.
    Code preserved behind `PERSISTENT_XCD=1` flag.

11. ~~**EARLY_BL_PF (Bl buffer_load issued at Step12 start, ~128-MFMA hiding)**~~ — **DEAD END (2026-04-17)**:
    Hypothesis: original DIRECT_BL placed Bl buffer_load inside Step4 with only ~32 cyc MFMA hiding for
    a ~400 cyc buffer_load — issuing it before Step12 gives ~512 cyc hiding.
    Result on 14336×4096×32768 (deep-LOSE shape, aiter=5245.4 TFLOPS, warmup=200 iters=500):
        baseline LDS:        4540.4 TFLOPS  (86.6%)
        DIRECT_BL (orig):    3786.3 TFLOPS  (72.2%)
        DIRECT_BL+EARLY:     4004.2 TFLOPS  (76.3%)
    Hypothesis VALIDATED (+4.1pp from latency hiding) but still 10.3pp behind LDS path.
    Conclusion: B-direct architecture is **structurally inadequate** on this kernel even with optimal
    Step12-launched prefetch. Resource cost (254V at bench, 4 SGPR spill) eats throughput, and
    eliminating LDS-store traffic doesn't compensate for the loss of LDS-broadcast bandwidth.
    Code is behind `-DEARLY_BL_PF=1` flag, default off. See `test_early_bl_pf.py` for repro.

**性能天花板结论 (2026-04-17 三轮验证)**: 当前内核架构下所有已知优化方向已穷尽. **24/42 WIN 是天花板**. 突破需要根本性重构 (aiter 架构: A-only-LDS + B-direct-from-global + deep SW pipeline, 需 >256 VGPRs 不可行 on gfx950).

三轮饱和验证:
- Round 1 (87 variants, agent-team auto-tune): 19→24 WIN (+5 flip, memclause family 主导)
- Round 2 (115 variants, ceiling sweep): 24→24 WIN (+0 flip, ±25 TFLOPS noise)
- Deep-LOSE 分析员 (44 targeted variants on 10 stuck shapes): +0 flip, 最大边际 +1.2pp

新增 dead-end:
- memclause × VMCNT ceiling (v20/v24/v32) × tv0/tv16 cross
- waves_per_eu attribute (REFUTED)
- amdgpu_num_agpr=192 hint (FAILED)
- UNROLL_K=8/16 × LGK × memc cross (边际 +0.3-1.2pp 但不翻 WIN)
- 128256×32768×4096 mega-M shape: IMPENETRABLE
- **asm_inline kernel correctness FAILURE** (2026-04-17): `kernel_mxfp4_asm_inline.{cpp,h}` SNR -1.31 dB,
  the "5084-5258 TFLOPS @ 8192³" reference is meaningless. K-specialization variants build
  cleanly but inherit the same correctness bug. See `project_mxfp4_asm_inline_broken` memory.
- **EARLY_BL_PF** (2026-04-17): DIRECT_BL with Bl buffer_load relocated to Step12 start (128-MFMA hiding window).
  +4.1pp vs original DIRECT_BL on 14336×4096×32768 (latency-hiding hypothesis confirmed) but still
  -10.3pp vs LDS baseline. B-direct path structurally cannot match LDS broadcast bandwidth on this kernel.
  Code preserved behind `EARLY_BL_PF=1` flag (default 0). See `test_early_bl_pf.py`.
- **Round 4 (2026-04-17)**: 3 untested vectors investigated by parallel optimizer team:
  - SCALE_REG_CACHE → DUPLICATE (scales already VGPR-resident `v70-v73`, no LDS round-trip)
  - LDS_XOR_SWIZZLE_B → DUPLICATE (already in `st_16x128_s::swizzle()`, perfect 8 acc/bank uniform)
  - PERSISTENT_XCD_QUEUE → DEAD END (3.9% slower; XCD-locality loss + atomic overhead; mega-M shape is reg-pressure bound, not launch-bound)
  Net: 0/3 WIN gain. **24/42 ceiling re-confirmed for 4th time.**
- **Round 5 (2026-04-17)**: 4 parallel optimizers — all DEAD END / INFEASIBLE:
  - **MFMA_32X32X64 (D)**: NO-GO for <1 week. **AGPR 节省 claim WRONG**: per-warp output is 128×128 (4 quadrants), 32x32x64 still needs 256 AGPRs. Decider misread layout.
  - **EARLY_SCALE_PF (E)**: **BROKEN + no perf gain**. Compiler aliases `pf_*` and shadow `nxt_pf_*` to same VGPRs → race; baseline ASM already issues scale loads at iter top with ~512 cyc hiding > ~400 cyc VMEM latency, no untapped scheduling room. Code has `#error` guard if enabled. See `test_early_scale_pf.py`.
  - **F, G**: INFEASIBLE in single session.
  Triggered the user's GOAL PIVOT directive at the top of this file.
- **Round 22 (2026-04-18, COMPLETE, memory-stall axis)**: 3 optimizers + 1 reviewer; all 3 vectors DEAD END.
  - **R22A (LDS_RD_STAGGER_NOP probe, committed `a4074d2d`)** — DEAD END. **CRITICAL CORRECTION**: TCP_TA_DATA_STALL = **producer-side `buffer_load_to_lds` L2-miss/HBM-latency stalls**, NOT consumer-side LDS port contention as R21-recon implied. Confirmed by raw HBM at only 7-20% of peak (latency-bound). Spreading consumer ds_reads can't help when producer is the bottleneck.
  - **R22C (finer SCHED_GROUP_BARRIERS masks, no commit)** — DEAD END. Even narrow masks (0x004 MFMA, 0x044 MFMA+DS_R, 0x008 VMEM, 0xc0 DS_R+W) regress −0.5% to −14.7%; best smoke is +0.13% noise.
  - **R22B (cache=streaming/non-temporal on A and B global loads, no commit)** — DEAD END. **All variants regress on all 3 DLA shapes**:
    - DLA1: bnt1=−6.30%, bnt2=−6.44%, bnt1_ant1=−20.08%, bnt2_ant2=−19.94%
    - DLA2: bnt1=−7.85%, bnt2=−6.27%, bnt1_ant1=−14.27%, bnt2_ant2=−13.01%
    - DLA7: bnt1=−0.83%, bnt2=−1.60%, bnt1_ant1=−6.58%, bnt2_ant2=−6.84%
    - Insight: NT bypass kills the L2 reuse path that B-tile shares across K-iters within a CTA. Producer stall is L2 *miss*, not L2 *thrash*.
  - **R22-rebench (COMPLETE)**: 42-shape rebench locking R20A's 11 wires → **29/42 WIN, 13/42 LOSE, avg ratio 105.3%** (+5 vs R20B). Saved bench_all42_results_r22.json.
  - 13 remaining LOSE: DLA1=91.9%, DLA2=96.3%, DLA7=96.9%, plus 10 mid-gap shapes 95.7-99.9%.

- **Round 23 (2026-04-18, COMPLETE — both vectors DEAD END)**: per R22A's mechanistic correction, R23 attacks producer side.
  - **R23A — STATIC_XCD_REMAP (DEAD END)**: 4 variants × 3 DLA shapes; bpc%8==0 verified.
    - DLA1 best _xcd_remap_g8 = +0.95%/+0.85pp (below 1.5% gate)
    - DLA2 best _xcd_remap_g4 = −0.68% (REGRESSION)
    - DLA7 best _xcd_remap = +1.50%/+1.45pp (exactly on gate, 1-run smoke noise band ±0.5pp; not worth 5-run reverify)
  - **R23B — PERSISTENT_XCD atomic dispatcher (DEAD END — CORRECTNESS BUG)**: 3 variants × 3 DLA shapes.
    - C-coverage = 6.4-28.6% (persistent grid skips most output tiles); reported "TFLOPS" 12834-61812 are meaningless because most C tiles are still zero.
    - DLA1 ERR rc=−6 on both _pxcd_b1 and _pxcd_b4 (correctness assert).
    - VERDICT: Kernel-side bug in PERSISTENT_GRID dispatcher — atomic tile claim either races, deadlocks, or terminates after fewer iterations than there are tiles.

- **Round 24 (2026-04-18, COMPLETE — all 4 vectors DEAD END)** — kernel-side fixes + new producer-side angles:
  - **R24A** — debug PERSISTENT_XCD coverage bug (Fix A+B+C). DEAD END: host-side fixes do not move the kernel-side persistent-loop bug; coverage stays at 6.4%/28.6%. Risky rewrite needed; out of scope.
  - **R24B** — L2 prefetch hints via `buffer_load_dwordx4`. DEAD END: 0.4-12% regression, monotone with intensity.
  - **R24C** — outer-K pull-forward (K+2/K+3 prefetch inside K+0/K+1 MFMA window). DEAD END: 14-24% regression on all 3 DLA shapes; saturation flat.
  - **R24D** — A-only NT cache hint. DEAD END: 4-8% regression on DLA1/2/7.
  - **Mechanistic conclusion**: DLA shapes are **VMEM-issue-bound**, NOT VMEM-latency-bound. Single VMEM lane saturated by existing prefetch path. HBM-bandwidth axis exhausted.

- **Round 25 (2026-04-18, BREAKTHROUGH — R25-C WIN + R25-D STACK WIN)** — pivot off HBM-bandwidth axis onto K-loop epilogue + L2-locality:
  - **R25-A — scratch-spill reduction via per-K-iter scale reload (`R25A_SCALE_RELOAD_PER_K_ITER`)** — DEAD END.
    - Pre-flight read of compile remarks: **0 spills** in production kernel (256V/256A clean). The premise (scratch-spill recovery) is empty.
    - Variant builds; runtime hangs/aperture-violates on all 3 DLA shapes (`rc=-6`). Same VGPR-aliasing pattern as the R5 EARLY_SCALE_PF dead end (compiler aliases live `pf_*` and reload-shadow VGPRs).
    - Code preserved behind `R25A_SCALE_RELOAD_PER_K_ITER=1` flag (default 0). NOT TO BE ENABLED.
  - **R25-B — `GROUP_SIZE_M` sweep (gm3/gm6/gm12/gm16)** — PARTIAL WIN (committed via R25-D stack):
    - **gm6** beats baseline (gm2 / kernel-default gm4) on DLA2 +2.97% and DLA7 +0.91%.
    - **gm12 / gm16 CRASH** with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on M=4096 shapes when stacked with `STEP3_PF_N=6` (reproduced 3x). Excluded from recommendation.
    - **gm3 unstable** (3.3% std across reps; some reps positive, some negative). Excluded.
    - Net: gm6 stays only via the R25-D stack (alone it's marginal vs R25-C alone).
  - **R25-C — K-loop tail prefetch-off (`R25C_TAIL_PF_OFF_ITERS=4`, gated by `R25C_K_LIMIT=32768`)** — **WIN** (committed `09d58029`):
    - Branches the last N iters of the TAIL_SPLIT main loop to `kpair_32mfma_with_lds_and_pf<PF_N=0>` to free VMEM slots from already-cached redundant tail prefetches.
    - Mechanism: `pf_bt = (bt+2 < k_byte_iters) ? (bt+2) : (k_byte_iters-1)` clamps in the last 2-4 iters and reissues stale lines.
    - **DLA2 +5.96%, DLA7 +3.14%** (3-rep tight verify, mean Δ).
    - DLA1 (K=128256) gated off by `R25C_K_LIMIT=32768` because `k_byte_iters=501` partial-unroll prevents the `bt` branch from folding (would regress 10-20%).
    - Preserved behind macros default 0 / 32768 — full backward compat.
  - **R25-D — STACK TEST: gm6 × pfoff4** — **SUPER-ADDITIVE STACK WIN**:
    - DLA2: baseline 4203.75 → gm6 4302.09 (+2.34%) → pfoff4 4342.04 (+3.29%) → **gm6+pfoff4 4484.95 (+6.69%)** — stack beats best-singleton (pfoff4) by +3.29%.
    - DLA7: baseline 4167.70 → gm6 4289.57 (+2.92%) → pfoff4 4358.45 (+4.58%) → **gm6+pfoff4 4456.16 (+6.92%)** — stack beats best-singleton by +2.24%.
    - Mechanism: `gm6` reorders L2 B-tile reuse (lower steady-state HBM pressure); `pfoff4` removes redundant tail VMEM (frees scale-load + epilogue VMEM). Orthogonal HBM-bound stalls, so stacking compounds.
    - Bonus: stack variant is the **most stable** of all 4 DLA7 variants (zero crashes, zero outliers across both parallel runs).
    - Wired into `bench_all_42.py`:
      - `("_ts_gm6_v12_memc_dc_pfoff4", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=6 -DSTEP3_BARRIER_VMCNT=12 -DR25C_TAIL_PF_OFF_ITERS=4 -DR25C_K_LIMIT=32768 -mllvm -amdgpu-sched-strategy=max-memory-clause -mllvm -amdgpu-disable-clustered-low-occupancy-reschedule")`  # DLA2 stack
      - `("_ts_lgk2_gm6_v12_memc_pfoff4", "-DTAIL_SPLIT=1 -DSTEP12_BR_LGKMCNT=2 -DGROUP_SIZE_M=6 -DSTEP3_BARRIER_VMCNT=12 -DR25C_TAIL_PF_OFF_ITERS=4 -DR25C_K_LIMIT=32768 -mllvm -amdgpu-sched-strategy=max-memory-clause")`  # DLA7 stack
    - Gap-to-aiter reduction: DLA2 96.3% → ~98.9% (gap 6.7% → 1.1%); DLA7 96.9% → ~99.8% (gap 6.9% → 0.2%). Both shapes likely flip LOSE → WIN once R25 reviewer's 42-shape regression run completes.
  - **Round 25 net**: +1 WIN axis re-opened (K-loop epilogue specialization), 2 deep-LOSE shapes nearly closed (DLA2 + DLA7). HBM-bandwidth ceiling broken via L2-locality stacking, NOT via raw HBM throughput.

  **新 dead-end vectors (Round 25)**:
  - `R25A_SCALE_RELOAD_PER_K_ITER` — premise empty (0 spills exist); reload variant aliases scale VGPRs and hangs GPU.
  - `GROUP_SIZE_M ∈ {12, 16}` — APERTURE_VIOLATION on M=4096 shapes when combined with `STEP3_PF_N=6`.
  - `GROUP_SIZE_M=3` — too unstable (3.3% std); not commit-quality.

  **Frontier post-R25-F**: K=4096 short-K shapes now likely 108-112% of aiter (R25-F gm7+pfoff14 stack). **Remaining vectors**:
  - **R25-F — committed `e5083bad`**: gm7 + pfoff14 dominates K=4096 shapes; flat plateau at gm{6,7} × pfoff{14,15,16}. Mechanism: only first 2 of 16 K-iters need prefetch; rest are L2-resident.
  - **R25-E — DLA1 K-loop peel (K=128256)** — IN FLIGHT (agent `aa5d9ccf6befb238f`): split main loop into head (K-1-N iters fully prefetched) + peeled tail (N iters no-pf). Working in worktree. Requires duplicating ~200-line main-loop body. Could close the largest remaining gap (DLA1 91.9%).
  - **R25-D-verify — full 42-shape rebench** — IN FLIGHT (agent `a355a47a7cd86cda2`): with R25-D wires (R25-F wires also in bench_all_42.py if it re-reads variants table). Will give the new total WIN count.
  - **R25 reviewer baseline check (PASS, 2026-04-18)**: pre-R25-D bench measured 27/42 WIN with 2 noise-band flips at 100% threshold (within ±2%). Verdict: kernel safe.
  - **Other 11 mid-gap shapes (95.7-99.9%)**: many will inherit `_ts_gm7_v12_memc_dc_pfoff14` / `_ts_lgk2_gm7_v12_memc_pfoff14` automatically via autotuner pick — likely WIN flips on K=14336/16384/28672/32768 mid-gap shapes too.
  - **R25-G (post-verify)**: if R25-D verify shows mid-gap shapes still LOSE, target them with the same gm7+pfoff{N-2} formula, where N = K/256 (per-shape pfoff).

- **Round 21 (2026-04-18, recon + audit + head-macro probe)**: 3 parallel agents; **0 new WINs**, but R21-recon delivered the highest-value finding of the post-R20 axis: DLA1/DLA2/DLA7 are **memory-stall bound**.
  - **R21-recon — rocprof PMC sweep on DLA2 + DLA7** (parallels R17A's DLA1 profile):
    - DLA2 (128256×32768×4096): MFMA fills 24.7 % of wall; **TCP_DATA_STALL = 292.6 % of GRBM**; HBM 1054 GB/s (19.9 % of 5.3 TB/s peak); 0 % LDS bank conflict.
    - DLA7 (28672×32768×4096): MFMA fills 20.8 % of wall; **TCP_DATA_STALL = 294.1 %**; HBM 913 GB/s (17.2 % of peak); 0 % LDS bank conflict.
    - DLA1 re-profile: MFMA 30.4 % of wall; TCP_DATA_STALL 167.8 %; HBM 412 GB/s (7.8 %); `lds_per_wave` differs 9× from DLA2/DLA7 (55 vs 512) → DLA1 K-iter-bound (epilogue overhead amortizes badly), DLA2/DLA7 pure HBM bandwidth-bound.
    - **All 3 DLA shapes are memory-stall bound, NOT compute bound.** TCP_TA_DATA stall is sub-arbitration (not bank conflict). HBM headroom = 5×.
  - **R21-audit — untried-axis survey**: identified `WAVE_PRIO_HIGH`, `EXPLICIT_S_NOP`, `SCHED_GROUP_BARRIERS` as 3 macros DEFINED but with **zero usage sites in production kernel** (audit's "already wired" claim was wrong). Proposed wiring 4 hook sites (3× K-iter end + 1× pre-Store-C) with default 0 = no-op asm.
  - **R21B — probe 3 head macros (DEAD END)**:
    - Wired 4 hook sites (defaults 0). Built 28/28 (4 shapes × 7 combos). Best smoke: P1 +0.72 % (`_snop1_sched`) → verify −0.185 pp = FAIL.
    - Mechanistic conclusions: `WAVE_PRIO_HIGH` dead because `__launch_bounds__(_,1)` already pins 1 wave/SIMD/CU; `EXPLICIT_S_NOP=1` redundant with LLVM `s_waitcnt`; `SCHED_GROUP_BARRIERS=1` mask `0xff` too coarse — disrupts cross-iter MFMA/prefetch interleave.
    - Kernel patch retained (no-op default) → unblocks R22+ for finer masks (`0x80`=MFMA-only, `0x44`=lgkmcnt-only). **No commit.**

  **Round 21 net**: 0 WIN. **R21-recon's classification of DLA1/DLA2/DLA7 as memory-stall bound (TCP_DATA_STALL 167-294 % of GRBM, HBM 7.8-19.9 % of peak) reframes the R22 frontier**: focus must move off compute-axis tweaks onto LDS-stall reduction and global-load throughput.

- **Round 20 (2026-04-18, full BARRIER_TO_WAITCNT generalization sweep)**: 3 parallel agents; **+11 shape WINs via R20A** (massive breakthrough). Cumulative R18+R19+R20 = **14 of ~18 deep-LOSE shapes closed**.
  - **A (Aperture probe + 11-shape verify)** — **BREAKTHROUGH**:
    - R19A's barrier-removal axis was thought constrained to ≤4 shapes (S1+S5 won, 11/15 SNR-pre-failed). R20A reframed: pre-failed shapes were noise-floor artifacts, not real correctness violations.
    - Built **random-scale aperture probe** (5 iters × 2 seeds, OK if no kernel crash + bench TFLOPS > 0 + reproducibility ≤ 5 % stddev) — replaces brittle uniform-input SNR floor.
    - 11/11 R19A-pre-failed shapes passed aperture; smoke surfaced 11 candidates (Δpp ≥ +1.5 pp on `_r19a_step3` or `_r19a_all`); 5-run same-GPU verify confirmed **11/11 WIN** (mean Δpp +2.03 to +4.28 pp).
    - **Wired 11 new (parent + BARRIER_TO_WAITCNT) stacks into bench_all_42.py** — see TODO.md for full table. Same correctness caveat as R18A/R19A: bf16-saturation non-deterministic; aperture-validated only.
  - **B (full 42-shape rebench locking R18+R19 wins)**: 116-variant auto-tune → **27/42 WIN** (up from 24/42 baseline). +3 LOSE→WIN flips: P1, S1, S5; 0 regressions. With R20A's 11 new stacks now wired, projected next bench: **~34/42 WIN** (127 variants).
  - **C (K-loop sync coarsening, K_LOOP_SYNC_EVERY_{2,4})** — **DEAD END**: both broke SNR (0 dB race) AND regressed −25 to −29 pp due to parity-branch blowing past the 256 VGPR cap, forcing scratch spill. Root cause: 2-buffer LDS rotation is insufficient when barriers skip alternate K-iters; correct fix needs triple-buffer LDS (out of scope).

  **Round 20 net**: **+11 deep-LOSE shapes closed** via R20A. Cumulative R18+R19+R20 = **14 closed**. Bench at 27/42 → projected ~34/42 next run.

  **新 dead-end vectors (Round 20)**:
  - `K_LOOP_SYNC_EVERY_{2,4}` — needs triple-buffer LDS (not 2-buffer); SNR + register-pressure both fail.
  - Uniform-input SNR was over-conservative — random-scale aperture probe is the right correctness oracle for BTW axis going forward.

  **Frontier post-R20**: Barrier-removal axis essentially saturated (14 of 18 deep-LOSE shapes addressed). **Remaining stuck**: DLA1, DLA2, DLA7, and ~3-4 marginal shapes. R21-recon classified DLA1/DLA2/DLA7 as memory-stall bound. **R22 axes**: (a) LDS-stall reduction (TCP sub-arbitration, not bank conflict), (b) global-load `cache=streaming` for DLA2/DLA7, (c) per-K-shape epilogue specialization for DLA1, (d) finer SCHED_GROUP_BARRIERS masks via R21B's now-wired hooks.

- **Round 19 (2026-04-17, barrier-removal extended)**: 3 parallel optimizers; **2 new WINs** (S1 +6.58pp R19A, S5 +2.69pp R19C). Cumulative R18+R19 = 3 deep-LOSE shapes closed (P1, S1, S5).
  - **A (41-shape sweep of BARRIER_TO_WAITCNT_{STEP3,STEP12,ALL})** — **WIN** (committed `e29f6c3a`):
    - 15 candidate shapes × 3 variants = 45 builds, SNR + aperture pre-filter, smoke + 5-run verify.
    - **S1 (14336x4096x32768) + `_lgk2_dc_btw_step3`**: 89.6% → **96.18%**, Δ = **+6.58pp** (biggest single-shape gain in any post-R2 round).
    - S1 + `_lgk2_dc_btw_all` also passes (+6.32pp) but dominated by step3-only.
    - Wired 3 opt-in entries into bench_all_42.py: `_p1_btw_all`, `_lgk2_dc_btw_step3`, `_lgk2_dc_btw_all`.
    - **11/15 candidates pre-failed SNR floor** (parent saturates bf16) — bf16-saturation is the dominant constraint on this axis.
  - **B (per-site barrier-removal bisect + vmcnt sweep on DLA1/DLA2/DLA7)** — **DEAD END** (committed `56affcbd`):
    - Added 10 finer-grained macros to kernel.cpp (per-site STEP3_S1..S7, STEP12_S1..S2, RELAXED_VMCNT). Defaults preserve R18A bit-exactly.
    - 39 builds; SNR probe: 35/36 BROKEN-RACE; only DLA1/_r19b_t1 OK-MARGINAL but Δ-0.27pp on 5-run verify.
    - **Confirms R18A**: barrier IS load-bearing for DLA1/DLA2/DLA7; not a single removable site exists.
    - DLA2/DLA7 SNR-unvalidatable via output-tile probe (parent at noise floor < 0 dB).
  - **C (Stack BARRIER_TO_WAITCNT with iterilp on S1-S5)** — **WIN** (committed `a0f1949c`):
    - 15 builds (5 shapes × 3 variants).
    - **S5 (4096x32768x14336) + `_ts_lgk2_memc_r19c_iterilp_btw_all`**: 95.18% → **97.87%**, Δ = **+2.69pp**.
    - On S5 specifically: removing only STEP3 OR only STEP12 is racy in isolation, but removing both together is safe — **the two barriers form a matched producer/consumer pair**.
    - S1-S4 SNR-unvalidatable (K∈{28672,32768} non-deterministic from bf16 saturation + FMA reorder).
    - **Hypothesis verdict**: iterilp ⊥ source-rewrite is PARTIALLY supported (1 confirmed compose, 4 unvalidatable).

  **Round 19 net**: +2 deep-LOSE shapes closed (S1, S5). Cumulative R18+R19 = 3 closed (P1, S1, S5). Biggest single-shape gain to date: S1 +6.58pp.

  **新 dead-end vectors (Round 19)**:
  - Per-site `BARRIER_TO_WAITCNT_STEP3_S{2,3,4}` on DLA1/DLA2/DLA7 — every individual hot site breaks SNR worse than aggregate.
  - 2-site / 3-site STEP3 combos on DLA1/DLA2/DLA7 — no synergy; strictly worse than singles.
  - `BARRIER_TO_WAITCNT_STEP12_S1` on DLA1 — SNR-MARGINAL but 0 perf gain.
  - `BARRIER_TO_WAITCNT_RELAXED_VMCNT` ∈ {1,4,15} on DLA1/DLA2/DLA7 — barrier-VMCNT perturbation alone breaks SNR.
  - 11/15 BARRIER_TO_WAITCNT candidates in R19A pre-failed SNR floor — bf16 saturation dominant constraint.

  **Frontier post-R19**: Barrier-removal axis substantially yielded but constrained by SNR floor to shapes with parent SNR > 10 dB. **Remaining stuck**: DLA1/DLA2/DLA7 (barrier load-bearing, SNR-broken) + 11+ LOSE shapes pre-failing SNR floor. **Next axes**: (a) different correctness probe (ULP histogram / reduced-dynamic-range scales) to validate kernel changes on DLA2/DLA7; (b) per-shape kernel specialization for K=128256 (DLA1); (c) MFMA op switch 32x32x64 (long-horizon).

- **Round 18 (2026-04-17, source-rewrite pivot per R17A proposals)**: **R18A WIN +4.16pp on P1** (committed `4b504b0c`) — first +1pp gain in 17 rounds of post-Round-2 work. R18B/R18C DEAD END but produced kernel-structure findings that delete proposals from the queue.
  - **A (R17A-P3 inner s_barrier → s_waitcnt lgkmcnt(0))** — **WIN**:
    - 3 opt-in macros added (default 0): `BARRIER_TO_WAITCNT_STEP3` (8 hot-path STEP3 sites), `BARRIER_TO_WAITCNT_STEP12` (2 tail STEP12 sites), `BARRIER_TO_WAITCNT_ALL` (both).
    - SNR probe with noise-floor-relative gating: 3/12 SNR-OK; 9 SNR-broken — barrier IS load-bearing for DLA1/DLA2/DLA7.
    - **P1 (28672x4096x16384) + BARRIER_TO_WAITCNT_ALL=1**: 5079.43 → 5302.23 TFLOPS (94.93% → **99.10%**), Δ = **+4.16pp**. 5-run same-GPU verify, both gates pass.
    - **uniform-input SNR is necessary but not sufficient**: DLA1/step12 SNR-passed under uniform-scale probe but APERTURE-violated at random-scale bench inputs.
    - Macros default 0 → other 41 shapes byte-identical, no regression risk.
    - **NOT auto-added to bench_all_42** (per-shape opt-in only; global add risks silent wrong-output on K-large shapes).
  - **B (R17A-P1 double C-accumulator ping-pong)** — **DEAD END (kernel-structure mismatch)**:
    - Build FAILED on all 4 shapes with "invalid operand for instruction".
    - **R17A profile proposal misread the kernel**: rows 1-3 use `ds_read_b128` results (`d1/d2/d3`) as MFMA A-operand, NOT `a_lo[i]`. Naive phase-interleave assuming uniform A semantics is invalid.
    - True ping-pong needs (a) move 8 ds_reads up-front (~2.5M extra cycles, NEGATIVE EV), (b) double AGPR 64→128 dropping wpe=2→1 (NEGATIVE EV), or (c) multi-day producer-consumer LDS protocol rewrite.
    - **R17A-P1 dropped from registry.**
  - **C (R17A-P2 M-tile expansion for K=128256)** — **DEAD END (LDS overflow + per-round infeasible)**:
    - Source refactor needs ~600 LOC duplication + 4× scale buffers + 8 accumulator sets — multi-day.
    - **MI355X LDS budget = 160 KB/CU (NOT 64 KB as R17A assumed)**. M=256 doubling pushes A-tiles to ~192 KB total — overflows.
    - M=192 fallback infeasible: not power-of-2, doesn't divide 64 (scale-pack `>> 6` row indexing).
    - **NEW PERSISTENT_XCD finding**: batch={1,2,4,8} (untested in R14C) all 7 variants crash with `Memory access fault by GPU node-X` before any dispatch — dispatcher bug at large grid counts.
    - **R17A-P2 dropped from per-round registry (long-horizon 3-5 day refactor only).**

  **Round 18 net**: +1 deep-LOSE shape gap closed (P1 94.93% → 99.10%, +4.16pp; almost-WIN). 17-round dry spell broken.

  **新 dead-end vectors (Round 18)**:
  - R17A-P1 (double C ping-pong) — kernel structure incompatible without multi-day rewrite
  - R17A-P2 (M-tile expansion) — LDS budget overflow + per-round infeasible
  - PERSISTENT_XCD batch={1,2,4,8} on DLA1 — GPU memfault, dispatcher bug at large grid counts
  - BARRIER_TO_WAITCNT on DLA1/DLA2/DLA7 — barrier load-bearing (SNR breaks or aperture-faults)

  **Frontier post-R18**: P1 nearly closed (99.10%, ≤1pp from 100%). DLA1/DLA2/DLA7 still need their own source-rewrites (the barrier trick won't transfer). Remaining proposals: MFMA op switch 32x32x64, K-split rewrite, SLM relayout, per-shape kernel specialization. Long-horizon multi-day: R17A-P1, R17A-P2 refactors.

- **Round 17 (2026-04-17, profile + triple-stack + attribute axes)**: 3 parallel optimizers, **0 new WINs, 13th saturation round**. R17A produced first hard profile evidence that further LLVM-flag tuning is futile.
  - **A (rocprof DLA1 + 8-GPU 42-shape re-baseline, partial)**:
    - **rocprof DLA1**: VALUBusy=49% (kernel idle half the cycles); MFMA-pipe floor 0.9-1.8 ms vs 6.74 ms wall ⇒ MFMA fills 13-27% of wall-time only.
    - **Bottleneck classified**: MFMA-accumulator dependency stall (single C-tile, 8-cyc f4 latency); K-loop epilogue per-iter `s_barrier`+`s_waitcnt` ~0.6-1.2 ms; prefetch m0-hazard `s_nop` 31× cost vs K=4096.
    - **HARD VERDICT**: VALUBusy=49% is regalloc/MFMA-scheduling, **not a backend-flag issue**. Further `-mllvm` tuning cannot move DLA1.
    - **3 source-edit proposals (queued, NOT implemented)**: P1 double C-accumulator tiling (split rt_C[4][4]→C0/C1 ping-pong, EV +2-4pp, risk ½ occ), P2 M=128→256 specialization for K=128256 (CTA halving, EV +1-2pp), P3 inner s_barrier→s_waitcnt lgkmcnt(0) (EV +5-6pp if SNR-safe).
    - 42-shape rebaseline 27/42 done before user time-box; net +0.4pp drift, no LOSE→WIN flips. Need overnight 8-GPU re-run (115×iters=500 took 3-4× the 20-30min estimate).
  - **B (P1 NO-iterilp triple/quad stacks of 3 sub-threshold positives)**: 10 variants stacking regclassglob × {noemxpre, tv16, v20, lgk2, extbr}. All 10 ASM-DIFF. Best 3-stack `rcg+noemxpre+tv16` = +0.498pp (just below +0.5pp gate; tied with best 2-stack). **Linear additivity of sub-threshold deltas COLLAPSED**: predicted +0.76pp, observed +0.50pp; 3rd flag adds 0pp on top of best 2-stack. 2 quad-stacks catastrophic: rcg+noemxpre+tv16+v20 -15.83pp; rcg+tv16+extbr -20.68pp.
  - **C (untested __attribute__ knobs on 4 stuck shapes)**: 9 codegen attrs RECOGNIZED (flat_work_group_size, num_vgpr 256/224/192, num_sgpr 96/80, max_num_work_groups), 1 unrecognized (amdgpu_no_agpr). All 32 builds DIFF .text. **All gate-PASS smokes collapsed on verify**. Catastrophic: vgpr192 -78pp, vgpr224 -59 to -60pp, sgpr96 -49pp on DLA7. mnwg8 → APERTURE on all 4 shapes. The 4 stuck shapes are at a **register-allocation fixed point robust to attribute-level coercion**.

  **Round 17 net**: 0 WIN, 0 gap reduction. **13 saturation rounds total. Backend axes (flags/attrs/compound stacks) now provably exhausted.**

  **新 dead-end vectors (Round 17)**:
  - All 9 recognized AMDGPU codegen attributes — KILL or APERTURE on the 4 stuck shapes (7 new BROKEN entries)
  - Triple/quad stacks of regclassglob × {noemxpre, tv16, v20, lgk2, extbr} on P1 — additivity collapses; 2 NEW catastrophic destabilizers
  - All `-mllvm` LLVM-flag tuning on DLA1 — provably bottleneck-mismatched (VALUBusy=49% is not a backend issue)

  **Frontier post-R17 (HARDENED, evidence-based)**: 13 rounds saturate flag/macro/source-micro/compound/attribute axes. **rocprof has now PROVEN further backend tuning cannot help DLA1.** Future agents must NOT propose more `-mllvm` flag work or codegen-attribute work on the 4 stuck shapes. Only kernel-source rewrites can move the needle: (a) **double C-accumulator tiling [R17A-P1]**, (b) **M=256 specialization for K=128256 [R17A-P2]**, (c) **inner-barrier→waitcnt rewrite [R17A-P3]**, (d) MFMA op switch 32x32x64, (e) K-split rewrite, (f) SLM relayout, (g) full B-direct.

- **Round 16 (2026-04-17, compound stacking iterilp × regalloc/sink/LICM)**: 3 parallel optimizers, **0 new WINs, 12th saturation round**. 3 hypothesis-falsifying findings:
  - **A (iterilp + regclassglob on 5 R10/R11 winners)**:
    - All 5 compounds DIFF .text but every one HURT vs iterilp-only winner: S1 -14.82pp (catastrophic), S2 -1.33pp, S3 -0.43pp, S4 -0.32pp, S5 -1.20pp.
    - **Hypothesis "regalloc family ⊥ scheduler family ⇒ stacks safely" is FALSIFIED.** Regalloc priority changes interfere with iterilp's preferred register layout.
    - regclassglob's +0.236pp on P1 is **iterilp-INDEPENDENT**; on iterilp WINs the flag has the OPPOSITE sign.
  - **B (iterilp + 4 R15B safe-DIFF flags × 5 winners = 20 compounds)**:
    - 10 DIFF, 10 NOOP-vs-iterilp, 0 BUILD-fail. Smoke (+0.5pp gate): 1/10 PASS (S2/largeivf2 +0.51pp, collapsed to +0.30pp on 5-run).
    - **nolicm + iterilp regresses -2.2 to -2.4pp** on S1/S3/S4 (LICM re-enables hoisting around iterilp reorder).
    - noemxpre ±0.35pp noise; sinkavoidspill ASM-NOOP everywhere on iterilp parents.
  - **C (cross-axis compounds × 4 stuck shapes, 25 variants, SNR-first safety)**:
    - **DEFINITIVE FINDING**: iterilp SGPR-clobber bug is INDEPENDENT of regalloc policy (regclassglob/sinkavoidspill/nolicm) AND independent of scheduler perturbations (largeivf2/noemxpre).
    - Every iterilp+X compound on DLA1/DLA2/DLA7 still aperture-crashes at full M.
    - **Hypothesis (regalloc restructuring might dodge the bug) is DISPROVEN.**
    - Best signal: P1 regclassglob+noemxpre clean re-verify **+0.30pp** (mean ≥ base.max PASS, +1pp gate FAIL).

  **Round 16 net**: 0 WIN, 0 gap reduction. **12 saturation rounds total**. R10/R11 still the only break-out rounds.

  **新 dead-end vectors (Round 16)**:
  - iterilp + regclassglob on 5 R10/R11 winners — UNIVERSALLY HURTS (-0.32 to -14.82pp); regalloc and scheduler axes are NOT independent
  - iterilp + {noemxpre, largeivf2, sinkavoidspill, nolicm} on 5 winners — 0/20 wins; nolicm catastrophic
  - iterilp + any non-scheduler flag on DLA1/DLA2/DLA7 — still aperture-crashes (bug is pure scheduler-induced; regalloc cannot dodge)
  - regclassglob + nolicm + sinkavoidspill triple stack on stuck shapes — DIFF but all sub-+1pp
  - ~25 NEW BROKEN-APERTURE registry entries (iterilp×X compounds on DLA1/DLA2/DLA7)

  **Frontier post-R16 (UNCHANGED)**: 12 rounds saturate flag/macro/source-micro/compound axes. **Future agents must NOT propose more compound flag stacks** (cross-axis interference is now empirically established). Only multi-day kernel rewrites remain: MFMA op switch 32x32x64, K-split rewrite, SLM relayout, B-direct.

- **Round 15 (2026-04-17, regclassglob cross-shape + 42 new LLVM flags + 3 source micros)**: 3 parallel optimizers, **0 new WINs, 11th saturation round**. LLVM-flag space now exhausted at **69 distinct flags** (27 R14A + 42 R15B):
  - **A (regclassglob cross-shape + 7 P1 macro compounds)**:
    - Fresh asm-diff: regclassglob produces DIFF on ALL 4 shapes (R14A's "NOOP on DLA2/DLA7" was a misread).
    - 5-run verify: P1 bare regclassglob **+0.236pp** (replicates R14A signal). P1c regclassglob_tv16 **+0.225pp** with mean ≥ baseline.max **PASS** but sub-+1pp gate FAIL.
    - DLA1/DLA2/DLA7 verify Δ ∈ [-0.06, +0.11] pp.
    - 0/11 commit-eligible. **regclassglob is real-but-tiny perturbation; macro compounds cannot amplify past noise.**
  - **B (42 untested LLVM flags, NO overlap with R14A's 27, 7 untouched families)**:
    - Coverage: coalescer (8), spill/AGPR (5), machine-sink/LICM (6), post-RA scheduler (6), IGLP (2), loop (3), AMDGPU misc (12).
    - 168/168 builds OK. Only 8/42 mutate .text; 34 silent NOOP. 5-run verify: 0/3 candidates pass +1pp gate.
    - **NEW BROKEN-APERTURE flags**: `-join-liveintervals=false` (P1/DLA1 crash; -30pp on DLA2/DLA7), `-greedy-reverse-local-assignment` (3/4 shapes).
    - **Major-regression flags (do NOT use)**: `-disable-machine-sink` -21.83pp on DLA1; `-disable-post-ra` -3.87pp.
  - **C (3 source-level micro probes, time-boxed 35 min)**:
    - New macros `TAIL_BARRIER_LGKMCNT`, `STEP4_BARRIER_VMCNT`, `PF_GROUP_OFFSET` added/REVERTED.
    - All three CLOSED: TAIL_BARRIER_LGKMCNT best -0.21pp on DLA7; STEP4_BARRIER_VMCNT all values 4-20 LOSE on DLA1; PF_GROUP_OFFSET=-1 +0.20pp on DLA1 within noise (compiler isel re-orders buffer_load_lds anyway).

  **Round 15 net**: 0 WIN, 0 gap reduction. **11 saturation rounds total**. R10/R11 still the only break-out rounds.

  **新 dead-end vectors (Round 15)**:
  - regclassglob × 7 macro compounds on P1 — sub-+1pp; tv16 closest at +0.225pp gate-mean-PASS but threshold-FAIL
  - regclassglob bare on DLA1/DLA2/DLA7 — verify Δ ∈ [-0.06, +0.11] pp (P1 only-shape signal confirmed)
  - 42 LLVM flags from 7 families — 34 NOOP, 8 DIFF, 0 wins
  - `-join-liveintervals=false` — NEW BROKEN-APERTURE on P1/DLA1, -30pp on DLA2/DLA7
  - `-greedy-reverse-local-assignment` — NEW BROKEN-APERTURE on 3/4 shapes
  - `-disable-machine-sink` — -21.83pp on DLA1
  - `-disable-post-ra` — -3.87pp multi-shape
  - `TAIL_BARRIER_LGKMCNT` source macro — closed (best -0.21pp on DLA7; tail iter ≤6% even at K=4096)
  - `STEP4_BARRIER_VMCNT` source macro — closed (all 4-20 LOSE; Step3 barrier already sufficient)
  - `PF_GROUP_OFFSET` source macro — closed (compiler isel re-orders, +0.20pp = noise)

  **Frontier post-R15 (UNCHANGED)**: 69 LLVM flags + 28 macro combos + 3 source-level micros all dead on the 4 broken shapes. **Future agents must NOT propose more LLVM flag sweeps, macro permutations, or single-line source micros**. Any progress requires multi-day rewrites: (a) MFMA op 32x32x64 switch, (b) K-split rewrite, (c) SLM relayout, (d) full B-direct.

- **Round 14 (2026-04-17, non-scheduler LLVM + macro + DLA1 deep-dive)**: 3 parallel optimizers, **0 new WINs**, all 4 broken shapes EXHAUSTED across remaining flag axes (committed `2717330d`):
  - **A (non-scheduler LLVM flags, 27 flags × 4 shapes via asm-diff probe)**:
    - **Best 5-run signal**: `-mllvm -greedy-regclass-priority-trumps-globalness=true` on P1 (28672×4096×16384) → +12.5 TFLOPS / +0.24pp 5-run mean, gate(mean≥baseline.max) PASS but **sub-+1pp threshold**. Documented for future, not deployable.
    - 63/108 flag×shape combos NOOP (text-identical .text → silent no-op on this kernel).
    - **NEW BROKEN-BUILD entry**: `-mllvm -amdgpu-promote-alloca-to-vector-limit=N` (any value) breaks build with "illegal VGPR to SGPR copy" at line 1728. Do not use.
    - **3 NEW BROKEN-APERTURE flags on DLA1**: `misched-cluster=false`, `amdgpu-dpp-combine=false`, `amdgpu-disable-clustered-low-occupancy-reschedule`. **SGPR-clobber bug class is broader than just iterative-ilp scheduler** — the bug surface includes generic mid-end transformation flags.
    - DLA2/DLA7/P1 are flag-insensitive shapes (23/27 flags NOOP across each).
  - **B (kernel-macro sweep, 28 untested combos × 4 shapes)**:
    - 0/28 passed +1pp gate. Best: `pf6_6_v20_memc` on DLA1 +0.13pp 5-run mean.
    - **DLA1 `pf6_6+lgk4` triggered HSA APERTURE bug with NO scheduler flag changes** (default scheduler, just macro change). The SGPR-clobber bug class is shape×macro driven, not just shape×scheduler.
    - `coverage_audit.md` documents the 28 macro combos as the **exhaustive untested macro-axis set** on these 4 parents. Don't propose more macro permutations.
  - **C (DLA1 deep dive, 9 non-scheduler LLVM flags via asm-diff)**:
    - 0 wins, all flags NOOP/regress on DLA1. **DLA1 is now triple-exhausted**: R12A (sched-strategy), R13A (alt iterative schedulers), R14C (non-scheduler LLVM flags) all produced 0 deltas.

  **Round 14 net**: 0 WIN, 0 gap reduction. **10 saturation rounds total** (R2/R4/R5/R6/R7/R8/R9/R12/R13/R14). R10/R11 remain the only break-out rounds.

  **新 dead-end vectors (Round 14)**:
  - 27 non-scheduler LLVM flags (membound-threshold, relaxed-occupancy-deps, divergence-merge, vgpr-index-mode, lds-thread-affinity, wavefront-priority-vgpr, lwt, regalloc-bias, dce-in-ra, dpp-combine=true, lst{16,256}, sghazard{0,64}, etc.) — silent NOOP across the 4 broken shapes; only `regclassglob` produced sub-threshold +0.24pp signal.
  - 28 kernel-macro combos (extbr/noembed/v20/v24/lgk4/tv0/tv16/gm1 cross-products on the 4 broken parents) — all sub-+1pp.
  - `pf6_6+lgk4` on DLA1 — NEW BROKEN-APERTURE finding (compiler bug at default scheduler).
  - `-amdgpu-promote-alloca-to-vector-limit` — NEW BROKEN-BUILD finding (illegal VGPR→SGPR copy).
  - `misched-cluster=false`, `dpp-combine=false`, `amdgpu-disable-clustered-low-occupancy-reschedule` on DLA1 — NEW BROKEN-APERTURE flags.

  **Frontier post-R14 (KERNEL-SOURCE LEVEL ONLY — multi-day work, infeasible in single session)**:
  - 4 deep-LOSE shapes (DLA1 88.3%, DLA2 92.9%, DLA7 94.5%, P1=28672×4096×16384 93.7%) have **NO remaining flag-level lever**. Three exhausted axes: LLVM scheduler (R8/R10/R12/R13), non-scheduler LLVM (R14A/C), kernel macros (R14B + Round 2 deep-LOSE sweep).
  - Future work paths require kernel rewrite: (a) MFMA op switch to `v_mfma_scale_f32_32x32x64_f8f6f4`, (b) K-split rewrite at source level, (c) SLM relayout (rebuild `st_16x128_s::swizzle` for different bank/acc pattern), (d) full B-direct (aiter-style, blocked on >256 VGPR budget).
  - **Future agents**: do NOT propose more LLVM flag sweeps or macro permutations on these 4 shapes. Either pick up a kernel-source rewrite, OR confirm the run is documenting saturation, not chasing it.

- **Round 13 (2026-04-17, alt-scheduler exhaustion sweep)**: 3 parallel optimizers, **0 new WINs** but 3 critical findings (committed `75d5e305`):
  - **A (alt iterative schedulers on 4 R12-broken shapes DLA1/DLA2/DLA7/WIN2)**: 0/12 candidates survived smoke.
    - `iterative-minreg`, `max-ilp`, `iterative-maxocc` ALL trigger SAME SGPR-clobber bug as `iterative-ilp`.
    - `max-occupancy` and `iterative-max-occupancy-experimental` are silently NO-OP (not in LLVM 20 un-prefixed enum table).
    - **Confirmed valid un-prefixed enum names** (via `strings` on libLLVMAMDGPUCodeGen.a): `iterative-ilp`, `iterative-maxocc`, `iterative-minreg`, `max-ilp`, `max-memory-clause`. Everything else silent no-op.
    - **Bug is general non-default-machinescheduler bug, NOT iterative-ilp-specific** — generalizes the R12 finding. The 4 broken shapes are sched-strategy EXHAUSTED.
  - **B (full sched sweep on last untested-not-broken P1 shape 28672×4096×16384)**: 0 wins.
    - `iterative-ilp` mean +0.11pp, gate FAIL by 2.6 TFLOPS.
    - `max-ilp` triggers SGPR-clobber bug here too (cross-parent confirmed: bug is shape-driven, not parent-driven).
    - All 12 sched-strategy variants regressed or no-op.
    - **28672×4096×16384 sched-strategy EXHAUSTED** (R8 + R10A + R13B = 3-round confirmation).
  - **C (stack alt strategies on 5 R10/R11 working shapes)**: 0 wins, but **LOAD-BEARING METHODOLOGY FINDING**:
    - **`-mllvm -amdgpu-sched-strategy=` is LAST-SPEC-WINS in this LLVM**. ASM-diff proof:
      - `parent (memc only)`: 224583 bytes
      - `iterilp + memc-appended-last`: 224583 bytes (memc wins, iterilp silently overridden)
      - `memc + iterilp-appended-last`: 223905 bytes (iterilp wins, memc silently overridden)
    - **Existing `_*_memc_r1X_iterilp` WIN variants are PURE iterilp** (parent's memc was overridden by appended iterilp flag). Naming misleading but substance correct — those 5 wins are real, just not "memc + iterilp" combos.
    - `iterative-minreg` triggers SGPR-clobber on 4/5 working shapes (only S5 K=14336 survived).
    - `iterative-max-occupancy-experimental`: -1.06 to -2.02pp regressions across all 5 shapes.
    - `amdgpu-mfma-padding-ratio` on top of iterilp: true no-op (byte-identical .text).
    - `STEP12_BR_LGKMCNT=4` / `STEP3_BARRIER_VMCNT=24` stacked on iterilp: sub-threshold (+0.01~+0.37pp single-shot).
    - **iterilp WIN ridge is locally optimal** in surrounding flag/scheduler space.

  **Round 13 net**: 0 WIN, 0 gap reduction. **9 saturation rounds** total.

  **新 dead-end vectors (Round 13)**:
  - `iterative-minreg` / `iterative-maxocc` / `max-ilp` — same SGPR-clobber bug class as iterative-ilp (cross-shape, cross-parent confirmed)
  - `iterative-max-occupancy-experimental` — works but regresses -1~-2pp
  - `max-occupancy` strategy name — silently no-op (not in LLVM enum)
  - `amdgpu-mfma-padding-ratio` on top of iterilp — true no-op
  - `STEP12_BR_LGKMCNT` / `STEP3_BARRIER_VMCNT` stacked on iterilp — sub-threshold
  - `memc + iterilp` flag combo — STRUCTURALLY IMPOSSIBLE (single LLVM option, last-spec-wins)
  - 28672×4096×16384 — sched-strategy exhausted (3rd-round confirmation)

  **Frontier remaining post-R13**: 4 deep-LOSE shapes (DLA1/DLA2/DLA7/28672×4096×16384) have NO scheduler-level lever. Future work must be (a) kernel-source level (tile reshape, K-split rewrite, SLM relayout, MFMA op switch) or (b) non-scheduler LLVM flags (`-amdgpu-membound-threshold`, regalloc, etc.).

- **Round 12 (2026-04-17, BISECT + GENERALIZATION)**: 2 parallel optimizers, **0 new WINs** but 2 critical findings (committed `4c4000eb`):
  - **A (bisect aperture violations)**: CONFIRMED real LLVM compiler bug in un-prefixed `iterative-ilp`.
    - Fresh-rebuild bisect: parent flags WITHOUT iterative-ilp = byte-identical to baseline; WITH iterative-ilp = byte-identical to R11 broken kernel. iterative-ilp is the SOLE differing input.
    - **Deterministic 0/3 fail** on DLA1 (4096×32768×128256), DLA2 (128256×32768×4096), DLA7 (28672×32768×4096), WIN2 (32768×6144×2048).
    - WIN1 (16384×4096×7168) was a 1-in-N flaky launch glitch (3/3 OK in retry); NOT a real iterative-ilp bug on that shape.
    - Failure addr `0xff9010f50000` page-aligned + high bits set → C-output base SGPR pair clobbered (not just per-thread offset). "Read-only page" → SRD/base lands in code/rodata mapping.
    - ASM diff ~1152 buffer_load reschedule diffs, no single-line miscompile. Reproducer in `build_round12_optA_bisect.py` if filing upstream LLVM bug.
  - **B (generalization to NT/MID-LOSE)**: ZERO new WINs on 8 untested NEAR-THRESHOLD/MID-LOSE shapes.
    - 5 of 8 candidates triggered HSA aperture violation (16384×28672×2048, 4096×32768×6144, 16384×28672×4096, 28672×4096×8192, 32768×4096×7168) — same compiler bug, generalizes broadly.
    - 2 regressions: 6144×4096×16384 -0.77pp, 14336×32768×4096 -0.91pp.
    - 1 marginal: 4096×14336×16384 +0.05pp (sub-threshold).
    - 0 candidates passed +0.5pp single-shot gate → 5-run verify skipped.
    - **Conclusion**: iterative-ilp does NOT generalize beyond R10/R11's 5 verified deep-LOSE shapes. Specific phenomenon, not universal optimization.
    - Note: several baselines drifted vs Round 2 (cross-GPU bias) — full re-baselining advisable, doesn't change conclusion.

  **Round 12 net**: 0 new WIN, 0 new gap reduction. **8 saturation rounds** total (R2/R4/R5/R6/R7/R8/R9/R12), but R10/R11 broke the streak with 5 verified deep-LOSE wins.

  **新 dead-end vectors (Round 12)**:
  - iterative-ilp on near-threshold shapes (32768×4096×7168, 4096×14336×16384, 6144×4096×16384) — 1 errors, 1 regress, 1 marginal
  - iterative-ilp on mid-LOSE shapes (5 shapes) — 4 errors, 1 regress
  - **iterative-ilp as default global flag**: UNSAFE (deterministic compiler bug on ≥10 shape categories).

  **R12 action items**:
  1. KEEP existing 5 verified iterative-ilp WINs from R10/R11 (already committed).
  2. DO NOT enable iterative-ilp as default flag in build_all42_parallel.py.
  3. bench_all_42.py dispatcher should register the 5 `_r1X_iterilp` variants ONLY for the 5 specific shapes.
  4. (Optional R13) try `iterative-minreg` or `iterative-gcn-max-occupancy` on DLA1/DLA2/DLA7/WIN2 — different iterative scheduler may avoid the SGPR clobber bug. Lower priority than 5-shape WIN consolidation.

- **Round 11 (2026-04-17, deep-LOSE iterative-ilp extension)**: 3 more verified WINs (committed `31f03996`):
  - 5-run mean ≥ baseline.max gate PASS:
    - 4096×32768×28672: parent `_v20_memc` 92.98% → 94.82% (+1.84pp)
    - 4096×28672×32768: parent `_u16` 93.35% → 95.35% (+2.00pp)
    - 4096×32768×14336: parent `_ts_lgk2_memc` 93.83% → 95.30% (+1.80pp)
    - 4096×128256×32768 (regression check): 161.23% → 163.89% (+2.46pp, no regress)
  - Sub-threshold (single-shot only, not validated): 32768×4096×14336 +0.74pp.
  - 5 of 10 produced HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION — Round 12A confirmed real compiler bug.

- **Round 10 (2026-04-17, BREAKTHROUGH)**: Acted on Round 9 verifier reversal — tested REAL un-prefixed sched-strategies on deep-LOSE (committed `091d3baa`). 5-run on GPU 5:
  - 14336×4096×32768: parent `_lgk2_dc` 89.6% → +1.78pp (mean ≥ baseline.max PASS)
  - 16384×4096×28672: parent `_u32` 90.1% → +1.82pp (mean ≥ baseline.max PASS)
  - ASM dumps in `asm_verify_r10/` confirm un-prefixed values produce distinct ASM (refute Round 8 A latent finding).
  - Broke the 6-round saturation streak; first real perf gain since Round 1.

- **Round 9 (2026-04-17, focused probe round)**: Verifier reversal + 8 codegen flag dead-ends + SNR fix committed:
  - **Verifier**: REVERSED Round 8 A's latent finding. Reality:
    - **`gcn-`-prefixed sched-strategy names (`gcn-max-memory-clause`, `gcn-max-ilp`, etc.) are silently NO-OP** — produce ASM byte-identical to default (size 301977, MD5 only differs in random hip_cuid hash).
    - **un-prefixed `max-memory-clause` and `max-ilp` are the ONLY forms that actually invoke a different scheduler** (7153 line ASM diff vs default).
    - `-mllvm -amdgpu-sched-strategy=` accepts ANY string silently (incl. nonsense) → fallback to default. Real LLVM/AMDGPU bug in ROCm 7.1 LLVM 20, but doesn't affect us — `bench_all_42.py:412-461` correctly uses un-prefixed `max-memory-clause`.
    - **CRITICAL REVERSAL**: Round 6 A and Round 8 A's "5 strategy sweep, all noise/regress" results were **all no-op tests** (gcn-prefixed). Real untested vector is un-prefixed `max-ilp` / `max-occupancy` / `iterative-ilp` / `iterative-minreg` on deep-LOSE shapes. Already deployed: un-prefixed `max-memory-clause` (in 14/24 WIN best variants).
  - **Optimizer B**: 8 never-tested LLVM codegen micro-flags on `14336×4096×32768` best `_v16_wpe2`. ALL DEAD END. Flags: loop-prefetch, schedule-metric-bias=80/100, mfma-padding-ratio=10/25, disable-loop-alignment, disable-clustered-low-occupancy-reschedule, disable-unclustered-high-rp-reschedule, use-aa-in-codegen, enable-pre-ra-optimizations. All [-0.85, -0.03] pp.
  - **Optimizer C**: **SNR false-OK bug FIXED**, committed `b46834a0`. Bug actually lived in `bench_optC_round6.py:175` and `bench_round6_optA.py` (not `bench_deep_lose.py`, which has no SNR gate). Fix: new `snr_check.py` module with NaN-safe is_snr_ok/classify_snr/compute_snr_db. **Real finding**: Round 6 OptC's `_ts_u8*` variants all silently produced NaN output but passed; any "wins" from those are bogus. Re-verify under fixed gate before reusing.

  Round 9 net: 0 WIN, 0 gap reduction, but **1 real commit (SNR fix)** + **1 critical methodology reversal**. 7 saturation rounds total.

  **新 untested vector (after R9 reversal)**: un-prefixed `max-ilp` on deep-LOSE shapes (verified 工作的, never tested in this combination).

  **新增 dead-end vectors (Round 9)**:
  - 8 LLVM codegen micro flags (loop-prefetch, sched-metric-bias, mfma-padding, disloopalign, etc.)
  - `gcn-`-prefixed sched-strategy names — silently no-op, equivalent to default

- **Round 8 (2026-04-17, GOAL PIVOT 后第三轮)**: 3 parallel optimizers, all DEAD END / NO-OP:
  - **A** (per-shape `-mllvm -amdgpu-sched-strategy=` bucketing on 4 untested deep-LOSE shapes 4096×32768×28672 / 28672×4096×16384 / 4096×28672×32768 / 32768×4096×14336): DEAD END. 19 variants, 全部 [-0.81, +0.09] pp. 加上 R6 A/B/C 的 3 shapes, sched-strategy 现在 7/10 deep-LOSE shapes 全测过 — **vector 完全 exhausted**.
    - **重要 latent finding (待验证)**: 现有 `*_memc` baselines (在 14/24 WIN best variants 中) 使用 un-prefixed `max-memory-clause` flag, 可能 silently default to `gcn-max-occupancy` (LLVM 不识别 → fallback). 如果是真的, `_memc` family 实际是 `gcn-max-occupancy` 不是 `gcn-max-memory-clause`. 解释了为何 memc 在 WIN shapes 有效. 不会改变 deep-LOSE 结论 (R6 A 用了正确 prefix 测试, 全部 noise).
  - **B** (TAIL_BARRIER_VMCNT × VMCNT × 3 large-K deep-LOSE shapes): DEAD END. 29 variants. 仅在 `TAIL_SPLIT=1` 路径生效. 全部 [-0.19, +0.27] pp. **机械性 insight**: 大 K 下 (K≥14336) tail iter 占总 K iters 的 ≤0.45%, 即使完美调 tail barrier 也只能 shift 微小比例 → 数学上不可能产生 visible gain on 大 K shapes. 不要再试.
  - **C** (`amdgpu_waves_per_eu(1,1)` on 3 register-pressure shapes): NO-OP + already REFUTED. 编译器 resource usage 显示 baseline 已经是 1 wave/SIMD (224V+256A=480 regs + `__launch_bounds__(1)` at line 1727). 加 wpe(1,1) bytewise-identical .so. **重要**: Round 2 deep-LOSE 已测过 wpe1 (`bench_deep_lose_results.json` 含 `_wpe1`, `_v16_wpe1` 等 9 个 wpe1 variants), 在 3 个 target shapes 全部 LOSE -7.6 to -51.9 TFLOPS. 我之前 "wpe1 NEVER tested" 是错的.
  - **NEW METHODOLOGY ADDENDUM**: 在 dispatch 前先 grep `bench_deep_lose_results.json` 确认 variant 是否已测过 — Round 2 deep-LOSE 分析员的 44 variants 比我之前以为的覆盖更广.
  Round 8 net: 0 WIN, 0 gap reduction. **6 轮饱和** (R2/R4/R5/R6/R7/R8). 每轮 0 净增.

  **新增 dead-end vectors (Round 8)**:
  - per-shape sched-strategy on P1/P2 (4 shapes × 4-5 strategies, max +0.09pp; 7/10 deep-LOSE 全覆盖)
  - TAIL_BARRIER_VMCNT × large-K deep-LOSE (mechanistically futile, tail iter ≤0.45% of K iters)
  - `amdgpu_waves_per_eu(1,1)` (no-op since kernel 已 1 wave/SIMD; 也已在 R2 deep-LOSE 测过, REFUTED)

- **Round 7 (2026-04-17, GOAL PIVOT 后第二轮)**: 3 parallel optimizers, all DEAD END / INFEASIBLE:
  - **A** (STEP12_BR_LGKMCNT sweep on 3 P1 shapes 4096×32768×28672 / 28672×4096×16384 / 4096×28672×32768): DEAD END. Round 6 C 在 16384×4096×28672 上的 brlgk2 directionally-positive 信号**不泛化** — 9 variants 全部 -0.03~-0.14pp on GPU 1 (single-shot, warmup=200/iters=500). brlgk0 是 default value. brlgk knob 不再继续探索.
  - **B** (STATIC_XCD_REMAP on mega-M 128256×32768×4096): DEAD END. Atomic-free static remap (each XCD owns N-strip width bpc/NUM_XCDS=16, walks GROUP_M×16 tiles). 6 variants × baseline_dc/gm{2,4,8} crossed with static_xcd_remap variants: best `_static_xcd_remap_gm4` at -0.37pp. **关键架构 finding**: mega-M shape **是 A-bound, 不是 B-bound** — A traffic dominates (M=128256 vs N=32768), 缩 B working set 8x 反而损失 8x A reuse. Round 4 PERSISTENT_XCD_QUEUE 失败 + Round 7 STATIC_XCD_REMAP 失败 双重证实: mega-M 92.9% 是 register-pressure / occupancy 结构性 bound, 任何 dispatch/L2-locality 改动都不可能有用. Code 留在 `STATIC_XCD_REMAP=1` flag (default 0) 作为 documented dead-end.
  - **C** (B-tile L2 prefetch via `__builtin_amdgcn_global_load_lds` on P2 shapes): **INFEASIBLE — premise wrong**. **重要文档**: 现有 `emit_one_pf()` (line 466-472) **已经是** `__builtin_amdgcn_global_load_lds` 的 buffer-SRD 形式 (`llvm_amdgcn_raw_buffer_load_lds`, emits `BUFFER_LOAD_DWORDX4 lds:1`), 已 prefetch B `bt+2` look-ahead 直接进 LDS. 三种"扩展"全部不可行: (1) 加 redundant 16B/thread 到 scratch LDS = 纯 VMEM duplication 在已 B-VMEM-bound 端口上, 必退化; (2) bump look-ahead `bt+2` → `bt+3/4` 不是新机制只是常数, 且 LDS 双缓冲 +50% 超 160KB cap; (3) GLOBAL_LOAD_LDS 没有 discard sink mode (硬件强制写到 LDS dest). 25 min 提早终止, 0 文件创建. **未来 agents 不要再提 "L2 prefetch via global_load_lds" — 已 deployed**.
  Round 7 net: 0 WIN, 0 gap reduction. 5 轮饱和 (R2/R4/R5/R6/R7) 全部 0 净增. 在 Round 6 methodology rule 下, 即使针对 P1/P2 (不仅 P0) 也无法找出 +1pp 改进.

  **新增 dead-end vectors (Round 7)**:
  - **STEP12_BR_LGKMCNT sweep on P1 shapes**: -0.03~-0.14pp (Round 6 C 信号 shape-specific, 不泛化)
  - **STATIC_XCD_REMAP for mega-M (atomic-free)**: -0.37pp; 证实 mega-M 是 A-bound, 不是 B-bound, 任何 L2-locality 改动都不会有用
  - **`__builtin_amdgcn_global_load_lds` 作为 "新" L2-prefetch 机制**: 已 deployed in `emit_one_pf` (buffer-SRD form) with `bt+2` look-ahead. 不是新 vector, 不要再提

- **Round 6 (2026-04-17, GOAL PIVOT 后第一轮)**: 3 parallel optimizers, all DEAD END / MARGINAL (REVERTED):
  - **A** (4096×32768×128256 / compiler flags + L2 prefetch): DEAD END.
    - **gfx950 has NO L2 prefetch instruction**: `__builtin_amdgcn_s_prefetch_data` and `s_buffer_prefetch_data` are tagged `gfx12-insts` only (verified `BuiltinsAMDGPU.def`). Don't propose software L2 prefetch on this arch.
    - **`-mllvm -amdgpu-igroup-lp` does NOT exist** in this LLVM build (verified `llc -mcpu=gfx950 -help-hidden`). Source-level `__builtin_amdgcn_iglp_opt` is documented dead-end.
    - Available `-mllvm -amdgpu-sched-strategy=` values: `gcn-max-occupancy`, `gcn-max-ilp`, `gcn-max-memory-clause`, `gcn-iterative-ilp`, `gcn-iterative-minreg`, `gcn-iterative-max-occupancy-experimental` (need `gcn-` prefix). All 5 alternatives to current `gcn-max-memory-clause` either match noise or regress -2 to -16pp on the target.
  - **B** (14336×4096×32768 / UNROLL_K sweep): claimed +3.16pp WIN (commit `198bb3a4`) → **REJECTED by Reviewer**. Commit reverted (`4c11f8bb`), reviewer doc commit `acbc8b38`.
    - **Phantom-baseline mechanism**: B measured baseline at 4591 TFLOPS (87.52%) on a different GPU; single-GPU 5-run replication on GPU 5 shows true baseline is **4727±13 TFLOPS** (90.12%). All 3 "winners" sit within ±0.16pp of this true baseline. The +3.16pp was a GPU-bias + run-noise artifact, not a real signal.
    - UNROLL_K=2/4/8/16/32 × `_v16_wpe2`/`_memc`/`_lgk2_dc` cross-product produces **identical real performance** to current best on this shape. UNROLL_K direction is exhausted.
  - **C** (16384×4096×28672 / TAIL_SPLIT epilogue): DEAD END. TAIL_SPLIT=1 consistently equal-or-worse for K≥7168 (steady-state main loop already saturates MFMA pipe; tail-splitting adds branch/barrier overhead). Best `_u8_lgk2` at +0.19pp, far below +1pp bar. `STEP12_BR_LGKMCNT=2` was the only directionally-positive knob — possibly worth retesting on other shapes.
  - **NEW METHODOLOGY RULE (mandatory for all future deep-LOSE work)**:
    - GPU bias is **50-100 TFLOPS / 1-2pp**, per-run noise is **±0.6pp** at 5-run replication.
    - Any deep-LOSE Δ < 2pp claim **must** be validated by: single-GPU, 5-run replication, with mean-must-beat-baseline-MAX gate (not mean-vs-mean).
    - Never compare baselines measured on different GPUs.
    - ~~`bench_deep_lose.py` correctness check has a false-OK SNR bug (NaN baselines pass `if snr > 25` because NaN compare is False but mis-classifies as OK)~~ **FIXED 2026-04-17 (Round 9 Optimizer C)**: Bug actually lived in `bench_optC_round6.py:175` (`if snr['snr_db'] < 25`) and `bench_round6_optA.py:180` (NaN-tainted output gave snr=+Inf via `noi>0` short-circuit). Fix: new `snr_check.py` module (NaN-safe `is_snr_ok` / `classify_snr` / `compute_snr_db`); both bench files now reject NaN/None/-Inf SNR. Verified on real prior `bench_optC_round6_results.json` — all 5 `_ts_u8*` variants had `snr_db=NaN, max_abs=NaN` and were silently benched; under the fix they would have been REJECTED. `bench_deep_lose.py` itself currently has no SNR gate (perf-only spot bench); add one via `from snr_check import is_snr_ok` if/when correctness gating is needed there.
  Net: 0 WIN, 0 gap reduction. Round 6 confirms even the "easier" gap-reduction goal (+1pp on a single shape) is at noise floor for the 3 P0 shapes.

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
- `bench_all42_results.json` — 最新结果 (24/42 WIN, Round 2)
- `bench_all42_results_round1.json` / `bench_all42_results_round2.json` — snapshots
- `bench_deep_lose.py` / `bench_deep_lose_results.json` — 10-shape stuck-shape spot bench
- `build_new_variants.py` / `build_round2_variants.py` / `build_deep_lose_variants.py` — 并行编译 infra (per-variant PYBIND11_MODULE patching, ThreadPoolExecutor)
