# MXFP4 GEMM 优化 — Agent提示词

你在继续推进 `HipKittens` 的 MXFP4 GEMM 优化工作，跟 Cursor (Hipkittens2) 竞赛。

## 项目位置
- **我们的 Repo**: `/shared_nfs/kyle/test/HipKittens`
- **Branch**: `mxfp4`
- **工作目录**: `analysis/fp8_gemm/mi350x`
- **Cursor Repo**: `/shared_nfs/kyle/test/Hipkittens2` (只读参考)

## 当前成绩 (2026-04-17)
- **我们**: **24/42 WIN** (warmup=200, iters=500, 115 variants, Round 2 final)
- **Cursor**: 16/42 WIN (同参数)
- **我们领先 8 WIN**
- **Auto-tune variants**: 115 (115 variants × 42 shapes 全sweep + 44 deep-LOSE targeted variants 双重确认饱和)
- **Round 1 → Round 2 → deep-LOSE**: +5 / +0 / +0 LOSE→WIN flip — 已彻底饱和

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
