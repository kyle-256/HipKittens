# MXFP4 GEMM 优化 — Agent提示词

你在继续推进 `HipKittens` 的 MXFP4 GEMM 优化工作，跟 Cursor (Hipkittens2) 竞赛。

## ⚠️ 当前优化目标 (2026-04-17 用户最新指令 — 必读)
> "24win 已经卡了好久了，现在把优化目标改成优化剩下那几个差的比较多的。"

**翻 LOSE→WIN 的工作 STOP**. 24/42 WIN 已被 4 轮饱和, 是当前架构天花板. 不要再跑 full-sweep auto-tune 了.

**改为**: 缩小 deep-LOSE shape 的 gap. 即使 88% → 92% 也算真实进步, **+1pp 即可 commit**.

### 重点 shape (R10/R11 部分突破, R12 generalization 失败)
**关键发现 (R10-R12)**: un-prefixed `iterative-ilp` LLVM sched-strategy 在 5 个 deep-LOSE shapes 上验证有效 (+1.78~+2.00pp, 5-run mean ≥ baseline.max gate), 但**有 LLVM 编译器 bug**: deterministic SGPR clobber 在 ≥10 个 shape categories 上触发 HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION. **不可作为 default flag**. 仅 whitelist 已验证 5 个 shape parents.

| 优先级 | Shape | 当前 Ratio | 当前 Best | 类别 | iterative-ilp |
|-------|-------|-----------|-----------|------|----------------|
| **P0** | 4096×32768×128256 | 88.3% | _ts_pf6_6_v12_memc | mega-K + 大N | **BROKEN (compiler bug)** |
| **P0 → P2** | 14336×4096×32768 | **91.4%** | **_lgk2_dc_r10_iterilp** ✓ | 大K + 大M | R10 +1.78pp WIN |
| **P0 → P2** | 16384×4096×28672 | **91.9%** | **_u32_r10_iterilp** ✓ | 大K + 大M | R10 +1.82pp WIN |
| P1 | 128256×32768×4096 | 92.9% | ts_gm2_v12_memc_dc | mega-M+N | **BROKEN (compiler bug)** |
| P1 → P2 | 4096×32768×28672 | **94.8%** | **_v20_memc_r11_iterilp** ✓ | 大K + 大N | R11 +1.84pp WIN |
| P1 | 28672×4096×16384 | 93.7% | ts_gm8 | 大K + 大M | UNTESTED (R13 候选) |
| P2 → P2+ | 4096×28672×32768 | **95.4%** | **_u16_r11_iterilp** ✓ | 大K + 大N | R11 +2.00pp WIN |
| P2 | 32768×4096×14336 | 94.6% | _ts_gm8_v12_r11_iterilp (UNVERIFIED) | 大K + 大M | R11 +0.74pp 未到 gate |
| P2 → P2+ | 4096×32768×14336 | **95.3%** | **_ts_lgk2_memc_r11_iterilp** ✓ | 大K + 大N | R11 +1.80pp WIN |
| P2 | 28672×32768×4096 | 94.5% | ts_lgk2_v12_memc | 大M+N | **BROKEN (compiler bug)** |

**Cumulative R10+R11 deep-LOSE gap reduction**: 5/10 shapes, 平均 +1.85pp (toward GOAL PIVOT goal). 10-shape deep-LOSE 平均从 ~92% → ~93.4%.

**仍未试**: 28672×4096×16384 — 唯一 untested-not-broken shape. R13 候选.

### 工作准则
- **每轮锁定 1-3 个 P0/P1 shape** 专项优化, 不再全 42 跑.
- **改动后必跑 regression**: `bench_deep_lose.py` (10 shape spot) + `bench_all42_parallel.py` 抽测 (确认 24 WIN 不掉).
- **+1pp 即可 commit** (不再要求翻 WIN).
- **大 K (≥14336) shapes** 是主战场: 该类的 gap 主要来自 LDS broadcast bandwidth 不足 + B tile reuse 效率低.
- **mega-M shape 128256×32768×4096** 已被验证为 **register-pressure / MFMA-pipeline bound** (Round 4 PERSISTENT_XCD_QUEUE 实证), **不是 launch-bound**. 不要再尝试 dispatch 优化.

### 推荐探索方向 (按可行性, Round 7 后更新)
| 方向 | 风险 | 预期 | 备注 |
|------|------|------|------|
| **per-shape compiler flag** (LLVM 调度策略 per-K-bucket) | 低 | +0.5-2pp | 之前 ±0.4% 是 average, 单 deep-LOSE shape 可能更大 |
| **per-shape K-loop unrolling** (UNROLL=8/16 仅 K≥14336) | 低 | +0.3-1.2pp | Round 6 B 在单 GPU 5-run 下证伪了 14336×4096×32768 上 UNROLL_K knob (within ±0.16pp). 仅可能在 K=128256 上还有 untested space |
| ~~**B-tile L2 software prefetch**~~ | — | DEAD END | Round 7 C: `emit_one_pf` 已经是 `__builtin_amdgcn_global_load_lds`, B 已 bt+2 prefetch 进 LDS. 不是新 vector |
| **K-loop epilogue 专项调优** (尾部 K 迭代 PF/barrier) | 中 | +0.5-1pp | 大 K shape 最后一组 K iter 的 barrier/VMCNT — 仅 STEP12_BR_LGKMCNT 测过, 还有 TAIL_BARRIER_VMCNT × shape 维度未覆盖 |
| ~~**static XCD-aware block_id remap**~~ | — | DEAD END | Round 7 B: -0.37pp. mega-M 是 A-bound 不是 B-bound. 任何 dispatch/L2-locality 改动都不会有用 |
| **MFMA_32X32X64_TILING** (大重构, ~1天 asm 重写) | 高 | +2-5pp on deep-LOSE | 唯一未试的内核级重构, AGPR 256→256 (per-warp 输出仍 128×128, 不省 AGPR), 但 K loop 调度自由度可能更高 |

## 项目位置
- **我们的 Repo**: `/shared_nfs/kyle/test/HipKittens`
- **Branch**: `mxfp4`
- **工作目录**: `analysis/fp8_gemm/mi350x`
- **Cursor Repo**: `/shared_nfs/kyle/test/Hipkittens2` (只读参考)

## 当前成绩 (2026-04-17, 历史背景, 不再追求扩大)
- **我们**: **24/42 WIN** (warmup=200, iters=500, 115 variants, Round 2 final) — 已饱和, 不再是 KPI
- **Cursor**: 16/42 WIN (同参数), 我们领先 8 WIN
- **Round 1 → Round 2 → deep-LOSE → Round 4**: +5 / +0 / +0 / +0 LOSE→WIN flip — 4 轮饱和
- **新 KPI**: 10 个 deep-LOSE shape 平均 ratio (当前 ~92%, 目标 ≥94%)

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
