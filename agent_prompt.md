# Agent Team Runbook — MXFP8 Optimization

## 任务目标（两条线并行）

1. **长期**：MXFP8 RCR 追平 FP8 per-tensor（GPU0 今日 FP8 RCR 3253.80；MXFP8 RCR 3015.73；差 238 TFLOPS / 7.32%）
2. **静态 gate（已达标）**：MXFP8 RRR / CRR ≥ 历史 RCR 2926.61 × 95% = **2780.28 TFLOPS**
   - RRR: 2889.36 TFLOPS ✅ (+109.08)
   - CRR: 2830.67 TFLOPS ✅ (+50.39)
3. **动态 gate（参考）**：≥ 今日 RCR × 95% = 3015.73 × 0.95 = **2864.94 TFLOPS**
   - RRR: 2889.36 ✅ (+24.42)
   - CRR: 2830.67 ❌ (−34.27, 93.86% of today's RCR — 小幅 miss，与 R14 同向)

## 总则

- 分支：`feat/mxfp8-only`
- 工作目录：`analysis/fp8_gemm/mi350x`
- GPU：8× MI355X (gfx950)，**formal 验收固定 `HIP_VISIBLE_DEVICES=7`**
- Skill：
  - `.cursor/skills/mxfp8-layout-tuning/SKILL.md`
  - `.cursor/skills/fp8-per-tensor-layout-tuning/SKILL.md`
- 测试协议：`test_mxfp8_python.py` / `test_python.py` 内的 **per-iteration `torch.cuda.synchronize()` + `output.zero_()`**，`warmup=100, iters=200`

## 测试 shape 矩阵（R25 起新增）

**正方形 8192³ 仍为 primary formal**，但 SHIP candidate 还须在 LLaMA 实际 shape 上确认 correctness + 无 regression：

| 名称 | M | N | K | 用途 |
|---|---:|---:|---:|---|
| **8192³** (primary) | 8192 | 8192 | 8192 | 所有 formal A/B + SNR + det gate |
| **LLaMA-8B Gate** | 4096 | 14336 | 4096 | 非正方形 MLP shape |
| **LLaMA-8B Down** | 4096 | 4096 | 14336 | K > N 非正方形 |
| **LLaMA-70B Gate** | 4096 | 28672 | 8192 | 大 N shape |
| **LLaMA-70B Q** | 4096 | 8192 | 8192 | attention shape |
| **batch decode** | 128 | 8192 | 8192 | 小 M decode |

**编译**：dispatcher gates on compile-time `M_DIM`/`N_DIM`/`K_DIM`（`kernel_mxfp8_layouts.cpp:5-12`，默认 8192）。非正方形需 rebuild：
```bash
make CXXFLAGS_EXTRA="-DM_DIM=4096 -DN_DIM=14336 -DK_DIM=4096"
python3 test_mxfp8_python.py 4096 14336 4096
```
每 shape 需**单独 rebuild .so**。

**SHIP gate 扩展（R25 起）**：8192³ formal PASS **且** ≥2 个 LLaMA shape 全 gate PASS 才算 SHIP-ready：
1. **Correctness**: SNR ≥ 48 dB + det 3/3 PASS
2. **性能**: MXFP8 V2 TFLOPS ≥ 同 shape FP8 per-tensor × 95%（MXFP8 不能比 FP8 慢超 5%）
3. **无回归**: 同 shape MXFP8 V2 优化前后 Δ ≥ 0（不能因优化 8192³ 而在 LLaMA shape 上退化）

## 核心约束

1. 不破坏 FP8 per-tensor baseline（每次 formal 必须附 FP8 回归确认）
2. 只改 MXFP8 相关文件（`kernel_mxfp8_layouts.cpp`，`*_mxfp8_*.inc`，`kernel_mxfp8_4wave_rewrite.cpp`，`rewrite_mxfp8*.py`，`build_rewrite*.sh`）
3. 每次改动必须过门禁：smoke OK → formal 8192^3 OK → SNR > 48 dB → 3 次 determinism 一致 → **≥2 LLaMA shapes: correctness PASS + perf ≥ FP8×95% + 无回归**
4. 有提升才 commit，Commit 时**必须同步更新** `TODO.md` + `agent_prompt.md` +（如有 durable finding）SKILL
5. 禁止提交 `*.so`、`*.s`、`*_layout_results_*.json`、`.bak*`、`gpucore.*`、`__pycache__` 等（`.gitignore` 已覆盖）
6. 每个子 agent 使用不同 `HIP_VISIBLE_DEVICES` 以免 GPU 冲突：Dev A → 0，Dev B → 1，Dev C → 2，Reviewer/formal → 7

## R28 cycle 进行中 (2026-04-18) ★ 1 SHIP so far (cachepolicy auto-select)

### R28 SHIP #1 cherry-pick: `88d5a7d5` cachepolicy=2 auto-select gate (Dev A)

Extends R27's `MXFP8_CRR_V2_SCALE_CACHEPOLICY` macro at `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc:9-21` so it auto-defaults to 2 (SLC) when compile-time `N_DIM>=28672 && K_DIM>=8192`. Outside region stays 0 = binary identical to R27. **+2.7% on 70B Gate V2-CRR (4096×28672×8192) without manual flag.** No regression on 70B KV / 8B Gate / 8192³.

5x preheat-then-bench (GPU0, sclk-verified): B vs B0 Welch t=+12.83, Δ=+61.49 TFLOPS / +2.61%. SNR≥49.59 dB, det 3/3 PASS on all 6 cells. Cherry-pick excludes 254-line build logs (kept on r28-a side branch).

R28 dev B in flight: s_setprio sweep on V2-CRR cp=2 baseline.

### R28 Dev B = NO SHIP + paradigm correction (closed lever)

`s_setprio` sweep on cp=2 baseline: best (MFMA=2, VMEM=0) only +7.72 TFLOPS / +0.33% / Welch t=+2.30 (gate ≥+30/+1.2%/t>3.0).

**Paradigm correction (DO NOT re-litigate)**: The 8-wave V2-CRR kernel does NOT have a wave-id branch. All 8 waves run identical interleaved VMEM+MFMA code in lockstep. Pre-R28 code ALREADY uses `__builtin_amdgcn_s_setprio(1)` during MFMA segments and `s_setprio(0)` to restore. Removing it = -21 TFLOPS; keeping MFMA elevated without restore = -74 TFLOPS. Pushing MFMA prio above 1 hits a hard ceiling (+0.3%) because 8 waves hit setprio in lockstep — no relative reordering possible. **NEVER prototype "elevate MFMA wave priority" again — the lever is fully exploited.**

### R28 Dev C/D in flight
- Dev C (GPU0): cp=3 vs cp=2 A/B on auto-select gate (~+0.4% if cp=3 holds with t>3)
- Dev D (GPU1): rectangular BLK_M=256/N=128 scaffolding (R28 #2 critical, 90 min compile/correctness goal NOT perf SHIP this cycle)

## R27 cycle 完结 (2026-04-18) ★ 1 partial production ship (cachepolicy macro infra) + 3 paradigm correction

5 agent (Dev A GPU0, Dev B GPU1, Dev C GPU2, Dev D GPU3, Reviewer GPU4) 攻 R26 后剩下的 V2 levers。

### R27 SHIP cherry-pick: `12785d98` `MXFP8_*_V2_SCALE_CACHEPOLICY` macro infrastructure (Dev A)
- 4 处 `+ MXFP8_{CRR,RRR}_V2_SCALE_CACHEPOLICY` macro hook on V2 `buffer_load_b128/b64` scale loads in `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc:320,332` (+ rcr/rrr counterparts)
- Default 0 = binary identical to current behavior (zero risk to production baseline)
- Per-shape opt-in build flag: `make CXXFLAGS="-DMXFP8_CRR_V2_SCALE_CACHEPOLICY=2"` for **70B Gate V2-CRR** (M=4096 N=28672 K=8192) gives **+63 TFLOPS / +2.7% Welch t≈+13** (5x preheat, GPU0, SNR 49.60 dB, det 3/3 PASS)
- **Shape-dependent — DO NOT enable globally**: cp=2 catastrophic on 4096³ V2-RCR (-25×) and regression on 70B KV (-6.9%) / 8B Gate (-4.2%) / 8192³ V2-RCR (-2.3% breaks 3180 floor)
- Auto-select gate is R28 candidate, NOT yet shipped

### R27 paradigm corrections (write-down for future cycles)
1. **V2 has NO scale LDS** (Dev C verified): R23 Dev B's "SCALE_LDS REPLACE" kill misled subsequent rounds. V2 actually loads scales VMEM→VGPR direct (`crr_mxfp8_exact_8wave_fastpath.inc:304-333`, SCALE_VERSION==2 branch). LDS budget V2-CRR = **128 KB** (As/Bs double-buffer only), 32 KB headroom. R26 "131-139 KB" figure was a different kernel variant. **NEVER prototype "scale LDS double-buffer"** — it is a re-litigation of an already-killed lever.
2. **Split-K-along-K is universally DEAD-END** (Dev B verified): -12% on 70B KV. Each K-chunk sub-grid still launches the same 64 blocks → no CU exposure gain, but pay 2× launch + 2× epilogue + lose K=8192 single-pass B-tile cache reuse. Reduction overhead fine (7.7%, well under 30% gate). Real fix needs split-along-M/N (= R26 Dev A's static_assert path) or streamk (multi-day).
3. **BLK rewrite path: rectangular BLK_M=256/N=128, NOT square BLK=128** (Dev D audit): 22 BLK=256 dependency sites, complexity 1=8 / 2=4 / 3=4 / 4=5 / 5=1. Square BLK=128 forces RBM/RBN below MFMA sweet spot. **Rectangular BLK_M=256/BLK_N=128 keeps RBM=64 (no A-side helper rewrite)**, only needs new B-side helper variant + B-only preshuffle. Est 2-3 days. Hardest blocker = `load_col_from_v2_st_half` family at `kernel_mxfp8_layouts.cpp:414-491` (`ds_read_b64_tr_b8 offset:1024` hardcodes K-stride for HB=128). Dev D shipped 25-line `MXFP8_BLK128` macro proof-of-concept on r27-d branch (default unchanged, BLK128 V1-fallback PASS at 2.71 TFLOPS) — useful infrastructure for R28 rectangular path.

### R28+ priority list (rebuilt from R27 root-causes)
1. **【high / 1-2 day】Auto-select cachepolicy=2 by N_DIM/K_DIM compile-time gate**: extend Dev A macro infra with `#if N_DIM>=28672 && K_DIM>=8192 && IS_CRR` auto-select. Direct +2.7% on 70B Gate without manual flag. Need to find K-threshold precisely (8B Gate K=4096 lost; 70B Gate K=8192 won).
2. **【critical / 2-3 day】Rectangular BLK_M=256/BLK_N=128 V2 path**: per Dev D audit, smaller scope than original square BLK=128 plan. Add new B-side `load_col_from_v2_st_quarter` helper + B-only preshuffle. Targets 70B KV V2-CRR 0.84 (4 N-tile → 8 N-tile → ~2× CU utilization). Validate with `MXFP8_BLK128` infrastructure already shipped on r27-d branch (`c019bec8`).
3. **【medium】s_setprio on MFMA-side waves stacked on cp=3 for 70B Gate** (Dev A H2 not reached): could push +2.7% to +4-5%.
4. **【medium】Streamk scheduling for small-N**: alternative to BLK rewrite. One kernel, internal K-reduction across CUs.
5. **【low】8192³ V2-CRR -8.9%**: still no production-impacting fix path. Skip until others addressed.

### R27 worktree status
- `r27-a`: cherry-picked SHIP `12785d98` to main feat/mxfp8-only
- `r27-b`/`r27-c`/`r27-d`: side-branch only, dead-end docs (`59d32212` / `9b2e1384` / `c019bec8`); r27-d's `MXFP8_BLK128` macro is reference for R28 rectangular path

## R26 cycle 完结 (2026-04-18) ★ 3 关键 paradigm correction，0 production fix

5 agent (Reviewer GPU1, Dev A GPU3, Dev B GPU5, Dev C GPU6, Dev D GPU7) 攻 R25 LLaMA worst cells. 0 production commit, 2 infra cherry-pick to main:
- `954ba8b4`: preheat wrappers (`preheat_then_bench.py` + `preheat_fp8_bench.py`) — **新规范：multi-shape baseline 必须用 preheat 避免 cold-DPM artifact**
- `ee5f985a`: corrected reviewer JSON `r26_reverify.json`

### R26 paradigm corrections

1. **70B KV V2-CRR 0.69 是 cold-DPM throttle artifact**: Reviewer (10x clean GPU7) + Dev A (per-shape preheat, GPU3 throttle 1700 MHz) **independently** measure ~0.84 (+21pp recovery). R25 sweep 协议 cold-throttle 命中 V2 (heavier scale traffic) 比 FP8 重。**R25 baseline 70B KV CRR cell INVALIDATED**; 其余 4 cell Reviewer 10x reverify 全 CONFIRM 在 ≤0.7% drift. Real corrected ratio 0.84.
2. **Down-RRR 1.05 V2-WIN 是 FP8-RRR spill bug, not V2 mechanism**: FP8-RRR `RRR_MAIN_UNROLL=4` 在 K≥14336 spill **39 VGPR + 160 B/lane scratch** (K=4096 时 0/0)；V2-RRR 同 K spill 仅 1 VGPR. Spill 量 32→112→224 (K=4096/14336/28672) 线性匹配 FP8 RRR/RCR ratio collapse 1.00→0.87→0.84. V2 没有"赢"。Port 到 V2-CRR/RCR (后者已 0 spill) NOT APPLICABLE. Dev C 试 `RRR_UNROLL=2` 修 spill 但 -41% (latency hide 不足)。R27+: FP8-RRR 修复需深度重构。
3. **V1 vs V2 size-gating 永久 KILLED**: Dev D 7-size sweep (1024-8192) + Dev A small-N 独立验证。V1 vs V2 全 ±2.5% 内 (Dev A: V1 791.64 vs V2 796.09, Welch t=-5.87 V2 微胜)。**V2 default everywhere is correct**.

### R26 root-cause (no fix)

- **70B Gate/Up V2-CRR 0.84 (large N=28672)**: Dev B PMC 找到 dominant cap = TA backpressure + per-wave SQ_INST_LEVEL_VMEM 4.895x scaling vs work-ratio 4.0x = +22% per-wave VMEM-stall queueing. 4 knobs (CRR_STEADY_VMCNT 2/6, MID_BARRIER off, PREFETCH_LGKM=2) 全 NULL 或 correctness FAIL. R27+ 候选: L2 cache-tag pinning of scale arrays (8 MB scale fits 32 MB L2), persistent-scale LDS prefill (DEAD-END at occ=2 / 160 KB cap), per-buffer TCC counter split.
- **小 N (KV) 7-19% gap**: Dev A H1 (BLOCK_N=128) blocked by hardcoded `static_assert(BLK==256)` in `crr_mxfp8_exact_8wave_fastpath.inc:37` + `crr_mxfp8_4wave_fastpath.inc:86` + 多个 V2 preshuffle helper assumption. RBM/RBN/scale-pack/ST_v2 全 hardcoded. Multi-day rewrite.
- **4096³ Q/O V2 0.92**: Dev D amortization curve 揭示 gap is **structural to MXFP8 not V2-specific** (FP8 -21%, V2 -24% from 8192→4096). 修复 = macro-tile prologue 缩短 OR 新增 128×128 tile (与 small-N 同样 multi-day rewrite).

### R27+ 优先级 (基于 R26 root-cause)

1. **【critical / multi-day】Tile shape rewrite**: BLK=256 hardcoded 是 small-N + 4096³ 共同根本约束. 需重写 V2 preshuffle helper 支持 BLK=128, RBM/RBN 参数化, ST_v2 重 shape.
2. **【high / 1-2 day】L2 cache-tag pinning of scale arrays for large-N CRR** (Dev B R27 候选 #1): 消除 V2 在 N=28672 上的 TA/VMEM scaling
3. **【high / FP8-side】Fix FP8-RRR spill at K≥12000** (Dev C finding): production benefit 12-15% on Down-RRR but with MXFP8 work 正交
4. **【medium】Per-buffer TCC split for large-N CRR**: 确认 scale vs A vs B miss attribution
5. **【low】8192³ V2-CRR -8.9%**: production impact 最低 (LLaMA 不跑 8192³)

## R25 LLaMA baseline 结果 (2026-04-18, GPU5, sclk 2320 MHz, commit `6be73744`/`05bf4fef`) ★ paradigm 再修正：8192³ near-parity 不可推广 — **R26 修正：70B KV CRR 0.69 实为 0.84 throttle artifact**

R25 LLaMA shape baseline 跑完（8 build shape，10 logical shape，60 measurement run）。**Pass rate 2/24 cells**（gate = V2 ≥ FP8 × 0.95）。

| Shape (M×N×K) | RCR ratio | RRR ratio | CRR ratio |
|---|---:|---:|---:|
| 8B Q/O 4096³ | 0.92 ❌ | 0.94 ❌ | 0.92 ❌ |
| 8B KV 4096×1024×4096 | 0.84 ❌ | 0.81 ❌ | 0.83 ❌ |
| 8B Gate/Up 4096×14336×4096 | 0.94 ❌ | 0.93 ❌ | 0.94 ❌ |
| 8B Down 4096×4096×14336 | 0.93 ❌ | **1.05 ✅** | 0.94 ❌ |
| 70B Q/O 4096×8192×8192 | 0.94 ❌ | 0.93 ❌ | 0.93 ❌ |
| 70B KV 4096×1024×8192 | 0.84 ❌ | 0.82 ❌ | **0.69 ❌** |
| 70B Gate/Up 4096×28672×8192 | 0.90 ❌ | 0.90 ❌ | 0.84 ❌ |
| 70B Down 4096×8192×28672 | 0.87 ❌ | **1.05 ✅** | 0.84 ❌ |

**结论**：
1. 唯一 V2 winning regime 是 Down-RRR (1.05) — large-K + RRR 共同特征
2. 小 N (KV-attn N=1024) 是最差 regime: V2-CRR @ 70B 跌到 0.69
3. 8192³ "V2 ≈ FP8" 在 4096³ Q/O 退化到 0.92 → V2 vs FP8 gap 与问题规模强相关
4. R25 8192³ CRR optimization 不再是 #1 priority — LLaMA gaps 全部更大

**完整 baseline JSON**: `llama_baseline_r25.json` @ feat/mxfp8-only HEAD `6be73744`
**Driver script**: `analysis/fp8_gemm/mi350x/run_llama_baseline.sh`

## R25 mainline (8192³ CRR optim) 完结：跨会话 task lost，0 commit，priorities reshuffled

R25 8192³ CRR optimization 5 个 agent (Reviewer + Dev A/B/C/D) 因 session compaction task lost；5 个 worktree (/tmp/wt-r25-{a,b,c,d,rev}) head 仍在 c285cb70。R25 LLaMA findings 让 8192³ CRR -8.9% 不再是 priority — LLaMA shape gap 远大。R26 priority list 完全 rebuild around production shape:

1. **【critical】KV-attn N=1024**: 70B KV V2-CRR 0.69 (最差). 假说: 256×256 block tile 在 N=1024 仅 4 N-tile/grid，CU 利用率严重不足. 需考虑 dynamic dispatch (N<2048 fallback / 专用 small-N kernel) 或 tile shape 调整
2. **【high】大 N CRR (70B Gate/Up V2-CRR 0.84)**: MLP gate/up. N=14336 (V2 0.94) → N=28672 (V2 0.84) 退化. 与 V2 LDS budget / scale b128 浪费 关联
3. **【medium】Down-RRR 1.05 win 推广**: 唯一 V2 wins. mechanism: large K 让 V2 scale-load 摊销充分？ port 到 Down-RCR/CRR
4. **【medium】4096³ Q/O 0.92**: real "square" production. 与 8192³ 0.99 形成 reference, 找 amortization breakpoint

## Baseline (R24 2026-04-18 GPU0 fresh reverify ★ paradigm correction：V2 实际 ≈ FP8 parity；R23 cycle-level 数字 invalidated as PMC-mode + cold-throttle artifacts) **— 注意：仅适用 8192³，LLaMA shapes 见上**

| 版本 | TFLOPS | SNR | 备注 |
| --- | ---: | --- | --- |
| **FP8 RCR (长期 target)** | **3232** (R24) | 49.61 PASS | R24 5x median GPU0, std 9.48 |
| **★ MXFP8 RCR PRESHUFFLE V2 (default on, R21 SHIP)** | **3214** (R24 fresh) | 49.60 PASS | **gap -17 / -0.5% ★ near-parity** |
| MXFP8 RCR V1 (RUNTIME=0 fallback) | 3022.43 | 49.60 PASS | unchanged |
| **★ MXFP8 RRR PRESHUFFLE V2 (default on, R22-A SHIP, commit dabeffa0)** | **3156** (R24 fresh) | 49.59 PASS | **gap -76 / -2.4%** |
| MXFP8 RRR V1 (RUNTIME=0 fallback) | ~2878 | 49.59 PASS | V1 had 19 spills/80B scratch; V2 collapsed to 0 |
| **★ MXFP8 CRR PRESHUFFLE V2 (default on, R22-B SHIP, commit 9a0d0624)** | **2943** (R24 fresh) | 49.60 PASS | **gap -289 / -8.9% ← ONLY meaningful gap remaining** |
| MXFP8 CRR V1 (RUNTIME=0 fallback) | 2711.78 | 49.60 PASS | V1 already PIPELINE_SCALE single-shot 4×b32; CRR baseline more MFMA-bound |

**R22 综合**：V2 preshuffle paradigm 完整覆盖三 layout（RCR + RRR + CRR）全 SHIPPED default-on。V1→V2 收益排序 RRR (5.54%) > RCR (1.84%) > CRR (0.76%)，与 V1 baseline 的 spills + VMEM-issue rate 排序一致。所有 V1 路径保留为 RUNTIME=0 fallback。

**R24 综合 (paradigm correction)**：R23 cycle-level diagnostic 数字全部 invalidated as PMC-mode + GPU3 cold-throttle artifacts。R24 fresh GPU0 measurement (sclk 2353 MHz verified, warmup=100, per-iter sync) 显示 V2-RCR -0.5% / V2-RRR -2.4% / V2-CRR -8.9%，远好于 R23 报告的 -5.5/-6.7/-13.6%。R24 派 3 dev (SQC dcache cut / RRR TCC re-tile / SPI occupancy) 全部 DEAD-END，0 production commit, 2 side-branch commits (5cf85e58 + 76848ea9, NOT cherry-picked)。**Real status**: V2-RCR essentially at FP8 parity; V2-RRR -2.4% within close range; V2-CRR -8.9% 是唯一 structurally real remaining gap (col-major A-LDS layout, R10 census + R11/R23 SNR wall)。

**R24 INVALIDATED hypotheses** (don't re-invest):
- SQC_DCACHE_BUSY +441% (R23 #1 lever) → R24 fresh +38%, readfirstlane SRD-pin no-op (V2 SRDs already SGPR), Welch t -1.06 null
- TCC_MISS V2-RRR +166% (R23 #3 lever) → R24 fresh +3.1%, sched_barrier TA spread Welch t -0.17 null
- SPI VGPR_SIMD_FULL +18.7% (R23 #5 lever) → V2 actually uses fewer VGPRs than FP8 (V2-RCR 246 vs FP8 254); occ=3 architecturally impossible (V2 LDS 131-139 KB > 160000/3 cap)
- All R23 PMC numbers should be treated as cold-cache + dispatch-aggregation artifacts unless reproduced with warmup≥100 + GPU sclk verified ≥2GHz

**R23 综合 (HISTORICAL, INVALIDATED by R24)**：~~R22 V2 三 layout 经一个 session reverify 全 stable，drift < 0.5%。3 个 dev 全 exhausted（V3 preshuffle REJECTED -2.99%；CRR A-LDS Route X FAIL SNR -2.71 dB 第二次 hit R11 wall；Dev C cycle diagnostic 找到 SQC_DCACHE_BUSY_CYCLES +441% 作为 NEW #1 lever）。~~ R24 reverify 推翻 R23 cycle-level numbers; R23 ranked NEW levers (SQC/TCC/SPI/MFMA-VALU coexec) 全部 invalidated. **关键 paradigm 仍正确**: SQ_INSTS_VMEM 是 transaction count 不是 byte count → V3 preshuffle b64→b128 width promotion 给 0% VMEM cut, R23 Dev A confirmed -2.99% regression。

**R21 VERIFIED PASS metrics (Reviewer GPU0 indep reproduction, RCR)**: V2 mean 3080.18 / median 3078.09 / std 4.63；V1 mean 3021.53 / median 3022.43 / std 3.19；Δ +55.66 TFLOPS / +1.84%；Welch t=23.33 (p<<0.001)；rocprofv3 SQ_INSTS_VMEM byte-exact match V1=6,815,744 → V2=5,767,168 (-15.38%)；correctness 256³/1024³/8192³ 全 SNR ≥49.5 + det 3/3 PASS。

**R22 cherry-picked commits (production)**:
- R21: `4abd4f62` (V2 layout foundation) + `efc389ff` (V2-RCR fastpath wiring) + `f7ae35f6` (R21 docs)
- R22-A: `dabeffa0` (V2-RRR fastpath wiring + `gemm_rrr_pq_v2` pybind + `MXFP8_RRR_PRESHUFFLE_V2_RUNTIME` default 1)
- R22-B: `9a0d0624` (V2-CRR fastpath wiring + `gemm_crr_pq_v2` pybind + `MXFP8_CRR_PRESHUFFLE_V2_RUNTIME` default 1)

**V2 implementation 关键 fact** (供未来调试):
- 新 kernel template parameter `SCALE_VERSION` (1=V1, 2=V2)；hot symbol `*_exact_8wave_scaled_kernel<true, 2>` 资源:
  - RCR: VGPR 246 (V1: 254, **-8 VGPR**), occ 2, LDS 131 KB
  - RRR: VGPR 256 (V1: 256), spills 0 (V1: **19 spills + 80B scratch**), occ 2, LDS 135 KB
  - CRR: VGPR 234 (V1: 232, +2), spills 0, occ 2, LDS 139 KB
- Python `preshuffle_scale_matrix_mfma16_v2_rcr_a/_b` **同一函数复用三 layout**：A/B scale shapes 在 RCR/RRR/CRR 之间相同 (M-major / N-major × k_blocks)
- **dual-patch rule** (R21 silent FAIL learning): 必须同时 patch `load_scale_buffer` (main loop) AND `load_scale_packs_for_pair` (warmup/pre-tail/tail) — RRR/CRR 只有单一 lambda 所以简化为 single patch；RCR 有独立 helper 是关键陷阱
- `--offload-device-only -S` 必须 emit `buffer_load_dwordx4` (A) + `buffer_load_dwordx2` (B), 不是 b32 chains
- **V2 已 hide 所有 vmcnt waits**: R22 Dev C diagnostic 证明 V2-RCR 的 vmcnt-wait 比 FP8 LESS (-8.8%)；sched hints 路径永久关闭

**GPU sclk verify (R22 起强制规范)**: 任何 perf 测量前必须 `rocm-smi` 查 sclk ≥ 2GHz；GPU0 throttled at sclk 260MHz / mclk 2000MHz 是 R22 CRR 第一次 verify FAIL 的根因。Reviewer 必须 dispatch fallback GPU (e.g., GPU1/GPU2)。

**Dispatcher per-size rebuild**: dispatcher gates on compile-time `M_DIM`，每 size 必须 rebuild .so

**R21 VERIFIED PASS metrics (Reviewer GPU0 indep reproduction)**: V2 mean 3080.18 / median 3078.09 / std 4.63；V1 mean 3021.53 / median 3022.43 / std 3.19；Δ +55.66 TFLOPS / +1.84%；Welch t=23.33 (p<<0.001)；rocprofv3 SQ_INSTS_VMEM byte-exact match V1=6,815,744 → V2=5,767,168 (-15.38%)；correctness 256³/1024³/8192³ 全 SNR ≥49.5 + det 3/3 PASS, 67108864/67108864 element pass-rate at 8192³。

**R21 cherry-picked commits (production)**: `4abd4f62` (V2 layout foundation + Python preshuffle + verify drivers) + `efc389ff` (V2 fastpath wiring with `llvm_amdgcn_raw_buffer_load_b128`/`b64` + `dispatch_pq_v2<RCR>` + `gemm_rcr_pq_v2` pybind + `MXFP8_RCR_PRESHUFFLE_V2_RUNTIME` runtime gate).

**V2 implementation 关键 fact** (供未来调试):
- 新 kernel template parameter `SCALE_VERSION` (1=V1, 2=V2)；hot symbol `rcr_exact_8wave_scaled_kernel<true, 2>` VGPR **246** (V1: 254, **-8 VGPR**), occupancy 2 unchanged, LDS 131072 unchanged
- Python `preshuffle_scale_matrix_mfma16_v2_rcr_a/_b` 在 V2 packing 前 reorder source row_groups → wave-tile 顺序匹配 b128 dword 顺序 (option a from R20)
- 必须同时 patch `load_scale_buffer` (main loop) AND `load_scale_packs_for_pair` (warmup/pre-tail/tail)；R21 Dev A 初始 256³ FAIL 真因是后者漏掉
- `--offload-device-only -S` 必须 emit `buffer_load_dwordx4` (A) + `buffer_load_dwordx2` (B), 不是 b32 chains

**R18 真瓶颈定性纠正（推翻 R17 假说）**：rocprofv3 cycle-counter 证明 MXFP8 vmcnt + lgkmcnt 等待 cycles 反而比 FP8 少 -157K + -85K = -242K，但 GRBM_GUI_ACTIVE 多 +845K cyc。差异 100% 由 +30% VMEM-issue 速率（6 extra scale buffer_loads/K-block）造成的 dispatch back-pressure (`SQ_WAIT_ANY +1.14M cyc`) 引起。MFMA cycles 完全相同。SQ_LDS_BANK_CONFLICT = 0。**R17 vmcnt-MFMA critical path 假说 FALSIFIED**——compiler scheduler 已经在最优位。

**历史 baseline (GPU7, 2026-04-17 早期会话)** — 仅作 95% gate 锚点，不再用作回归对照：MXFP8 RCR 2926.61 / RRR 2794.26 / CRR 2740.55

**静态 gate** = 2926.61 × 0.95 = **2780.28 TFLOPS** → R15 RRR / CRR 全 PASS  
**动态 gate** = 3015.73 × 0.95 = **2864.94 TFLOPS** → RRR PASS, CRR 差 34.27 TFLOPS (1.14%)

> **R14 → R15 paradigm shift**：R14 GPU0 测得 RCR=2806 / RRR=2873 / CRR=2841（"RRR > RCR 倒置"），R15 fresh GPU0 测得 RCR=3015 / RRR=2889 / CRR=2830（历史顺序 RCR > RRR > CRR 恢复）。R14 倒置是 cold GPU + missing production flag 双重 artifact

> **VGPR 读数陷阱**：`-Rpass-analysis=kernel-resource-usage` 会为每个符号各报一次；MXFP8 RCR PQ 路径的真实 hot kernel 是 `rcr_exact_8wave_scaled_kernel<Lb1>`（VGPR **254** / LDS **131 KB**）。外壳 `gemm_kernel<Layout0,*>` 只是 dispatcher，显示 VGPR 212 / LDS 139 KB，**不是**可用于 headroom 推断的数字。任何基于「40 VGPR headroom」的优化提案都是错的，请以 scaled kernel 符号的 remark 为准。CRR 同理，看 `crr_exact_8wave_scaled_kernel<Lb1>` 符号的 remark。

## 角色定义

### Decision Maker（主 agent）
- 汇编级差距分析、派活、审议结果、决定 commit
- 汇总 reviewer 结论，记录新的死路到 SKILL
- **每次对话结束前必须**：更新 `TODO.md`（进度 + baseline）和 `agent_prompt.md`（若有规则变化）

### Dev A — 4-Wave 完整升级路径
- **目标**：把 8-wave 的 KPAIR_LOOP + SGPR SRD + scale pipeline 都迁到 4-wave fastpath
- **核心依据**：4-wave 有 14–44 VGPR headroom，唯一有空间做 scale cross-iter pipeline 的路径
- **关键文件**：`rcr_mxfp8_4wave_fastpath.inc`，`kernel_mxfp8_4wave_rewrite.cpp`
- **陷阱**：若用 inline ASM，所有 accumulator AGPRs 必须每个 split block 都 `"+a"`
- **里程碑**：formal ≥ 2897 (超 8-wave) 就 commit

### Dev B — 8-Wave v_lshr 消除
- **目标**：隐藏或消除每 iter 6 个 `v_lshrrev_b32` scale remap
- **核心依据**：这 6 条在 RAW 关键路径；但可以尝试 opsel 字节选择（HW 直接读 32-bit pack 里的第 N 字节，无需 shift）
- ~~**已有开关**~~（**R16 永久关闭**：以下 3 个 flag 与 HOIST_HI 在 `#elif` chain 架构互斥；启用任何一个会自动禁用 HOIST_HI 触发 SNR FAIL 或 spill）：
  - ~~`MXFP8_RCR_EXACT_PQ_PHASE_U16_CACHE_ENABLE=1`~~ (HOIST_HI 关闭后 K_PHASE templated lambda + tail path 不兼容，SNR -1.18 dB)
  - ~~`MXFP8_RCR_EXACT_PQ_REMAP_ONCE_ENABLE=1`~~ (VGPR 254→256 + 1 spill + 8B scratch)
  - ~~`MXFP8_RCR_EXACT_PQ_SCALAR_PHASE_PACKS_ENABLE=1`~~ (同上 256+1 spill+8B)
- **已完成**：HOIST_HI opsel 方案上线（reviewer GPU7 验收 2925.64，A/B +17.80；GPU1 head-to-head +25.67；main-loop v_lshr 0；spills 0；VGPR 256→254）。构建加 `-DMXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE=1` 即可。
- **经验**：K_PHASE 必须是 **编译期常量**（C++20 templated lambda）；tail 区域走 `rcr_mma_scaled_from_packs_exact`（runtime k_phase）——route tail 到 opsel 会复制 MFMA code path 并制造 31 spills。
- **禁地**：skill 明确 `op_sel_hi` phase1 **inline-ASM** 变体「语义能对，长跑回归」——当前 HOIST_HI 用的是 `__builtin_amdgcn_mfma_scale_*` + `op_sel`/`op_sel_hi`，不是 inline ASM，工作正常。

### Dev C — ASM Rewriter 路径
- **目标**：改写 `rewrite_mxfp8.py`（当前只处理 AGPR，对 8-wave PQ=1 是 no-op）使其对 8-wave 生效
- **思路**：把 6 个 `v_lshr` 前移到**上一 phase** 的 MFMA shadow（64-cycle 延迟里隐藏）
- ~~**stash 已有 Dev C 写到一半的 `rewrite_mxfp8_8wave.py` + `build_rewrite_8wave.sh`**~~ **已关闭**：第二轮实测 +0.18%，HOIST_HI 已从根源消除 `v_lshr`，ASM rewriter 无有效优化空间。该方向已废弃，rewriter 脚本未入库
- **pipeline 陷阱**（来自 MXFP4 经验）：
  - `clang -x assembler` 必须带 `-c`，否则 ld.lld 视为预建 DYN ELF 会吞 kernel
  - cuid 必须从 `grep -oP '__hip_cuid_\K[0-9a-f]+' device.s` 动态获取

### Reviewer / Tester
门禁三步（任何一项失败 → 拒收）：
1. **Smoke**：`HIP_VISIBLE_DEVICES=<dev> MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr MXFP8_WARMUP=5 MXFP8_ITERS=10 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 python3 test_mxfp8_python.py 256 256 256`
2. **Formal**：同上但 `HIP_VISIBLE_DEVICES=7 MXFP8_WARMUP=100 MXFP8_ITERS=200` + `8192 8192 8192`
3. **FP8 回归**：`HIP_VISIBLE_DEVICES=7 FP8_WARMUP=100 FP8_ITERS=200 FP8_CHECK=1 FP8_DETERMINISM_RUNS=3 python3 test_python.py 8192 8192 8192` 必须仍 ≥ 3050

通过 → commit 信息：
```
MXFP8 RCR <change>: <TFLOPS> TFLOPS (<+delta%>)

SNR: XX.XX dB, determinism: PASS (3 runs)
VGPR: X, AGPR: Y, spills: Z, LDS: W KB
```

## 第二轮评审结果 (2026-04-17)

stash@{0} 已 pop 并逐条评审。三条路径：

| 路径 | 结果 | 采纳？ |
|---|---|---|
| Dev B — HOIST_HI opsel 消除 v_lshr | reviewer GPU7 formal 2925.64 (SNR+det PASS)；A/B +17.80；Dev B GPU1 A/B +25.67；VGPR 256→254；spills 0；main-loop `v_lshr` 6→0 | **采纳，已 commit** |
| Dev A — 4-wave KPAIR+SCALE_PIPE | GPU0 同 GPU A/B：4-wave bare 2878 → 4-wave KPAIR+PIPE 2900 (+22)；但仍显著低于同 GPU 上的 8-wave KPAIR+SRD+PIPE (2993)；且 4-wave 走 inline ASM 占 256 AGPR + 256 VGPR → occupancy=1，结构劣势根深 | **拒绝**（不能超越 8-wave） |
| Dev C — ASM rewriter `rewrite_mxfp8_8wave.py` + `build_rewrite_8wave.sh` | GPU2 A/B 仅 +5.94 TFLOPS (+0.18%)；HOIST_HI 已从源头消除 `v_lshr`，rewriter 的重排空间被覆盖 | **拒绝**（边际收益 + 维护成本不划算，脚本未入库） |

任何后续 agent 想重启 Dev C 的 ASM rewriter 方向之前，必须先证明 HOIST_HI 框架下还有可被 rewriter 独占抢到的非 `v_lshr` 结构性收益，否则视为重蹈覆辙。

## 第三轮评审结果 (2026-04-17)

主 agent 基于错误假设（40 VGPR headroom）派了 3 个 dev，结果真实 baseline 是 254 VGPR / ~2 headroom。三条路径全 reject：

| 路径 | flag | 结果 | 采纳？ |
|---|---|---|---|
| Dev A — scale cross-iter prefetch (n+1 ring) | `MXFP8_RCR_EXACT_PQ_SCALE_PREFETCH_N1_ENABLE` | scope=0 full ring 4 spills A/B −2.26%；scope=1 B-only spill-free A/B −1.58%；VGPR 不够 | **拒绝** |
| Dev B — tail compile-time K_PHASE dispatch | `MXFP8_RCR_EXACT_PQ_TAIL_DISPATCH_ENABLE` | 正确+资源 clean（少 12 tail v_lshr）但 reviewer GPU7 A/B 10+20 rounds Δ = −0.04% ~ +0.18%，低于 +0.25% 门槛 | **拒绝**（技术正确但噪声级） |
| Dev C — KPAIR 2× unroll (4× k-pairs / iter) | `MXFP8_RCR_EXACT_PQ_KPAIR_UNROLL2_ENABLE` | body 翻倍→live-range 爆 256 VGPR，51 spills 208 B scratch，A/B −54.87% | **拒绝** |

**本轮无代码 commit，仅文档更正**。SKILL / TODO / agent_prompt 三处都加了「真实 VGPR=254，读 compile remarks 必须取 `rcr_exact_8wave_scaled_kernel` 符号，不要取 outer `gemm_kernel` 包装」的纠偏说明。

下一轮必须先解决 **baseline VGPR 压力** 才能再动 cross-iter pipeline / deeper unroll；否则任何需要额外寄存器的方案都会直接 spill。

## 第四轮评审结果 (2026-04-17)

两条路径，都是**不加 VGPR 不加 LDS** 的尝试，仍全 reject：

| 路径 | flag | 结果 | 采纳？ |
|---|---|---|---|
| Dev D — SCALE_LDS 叠加 PIPELINE_SCALE | `MXFP8_RCR_EXACT_PQ_SCALE_LDS_ENABLE` | VGPR 243 / LDS 132 KB / occ 2 / smoke PASS；但 8192 formal A/B −0.76%（5 runs 全输）**且 determinism FAIL**（max abs 1.43）。这个 flag 以前只是「未验证」，现在有硬数据 | **拒绝（regression + det fail）** |
| Dev E — sched V2 `TK_WAIT_VMCNT(6→8)` | `MXFP8_RCR_EXACT_PQ_SCHED_V2_ENABLE` | 正确+资源不变；GPU1 A/B 10 runs Δ +0.11%，Welch-t 0.58，纯噪声 | **拒绝（marginal）** |

关键新 dead-end 已写入 SKILL：SCALE_LDS 叠加现有 PIPELINE_SCALE 是不可走通的，只能**替换**（且需解决 LDS 同步 determinism），非小改动。

**本轮再次无代码 commit，仅 SKILL / TODO / agent_prompt 更新**。剩余差距仍是 145 TFLOPS / 4.73%。下一轮建议的高收益方向只剩**降低 baseline VGPR 压力**（accumulator reshape 或主动下调到 occupancy=1），因为任何 VGPR-neutral 的调度优化都已被压到噪声之下。

## 第五轮评审结果 (2026-04-17)

三条路径，全 reject：

| 路径 | flag | 结果 | 采纳？ |
|---|---|---|---|
| Dev F — AGPR accumulator 重新绑定（per-MFMA `"+a"` inline asm） | `MXFP8_RCR_EXACT_PQ_AGPR_ACC_ENABLE` | VGPR 254→128, AGPR 0→128；但 Spills 0→13, Scratch 56 B/lane；141+144 V↔A shuffle；GPU3 A/B 10 runs Δ −2.19% | **拒绝（regression）** |
| Dev G — scale L2 cache-policy hint (sc0) | `MXFP8_RCR_EXACT_PQ_SCALE_L2_HINT_ENABLE` | 12 scale buffer_load 用 sc0，正确+资源不变；GPU4 A/B 15/20 runs Δ +0.066% Welch-t 0.20 | **拒绝（marginal）** |
| Dev G2 — scale buffer_load_b32 × 2 → b64 合并 | `MXFP8_RCR_EXACT_PQ_SCALE_LOAD_B64_ENABLE` | **结构不可行**：6 scale dwords 来自 6 独立 SRD，最小间距 8192 B；b64 要 X 与 X+4 同 SRD → 无法合并 | **拒绝（broken）** |

**5 轮 7 个 dev 全 reject**。MXFP8 在当前结构下的微调空间已饱和。剩余 145 TFLOPS 差距只能靠高风险结构重构：
- AGPR accumulator **fused-asm block**（8-32 MFMA 打包一个 asm，重写 `_row`/`_impl`，破坏 HOIST_HI 约定）
- SCALE_LDS **完全替代** PIPELINE_SCALE（不是叠加）+ 解 determinism 问题
- `preshuffle_scale_matrix_mfma16` **layout 重设计**让 scale 可 b64 合并（影响所有 MXFP8 kernel + Python 参考）
- 主动下调 **occupancy=1** 换更深 cross-iter pipeline

下一轮建议**不再并行派 3 dev**，选 1 条结构路径深入做 1-2 周。本轮所有 dead-ends 已写入 SKILL。

## 第六轮评审结果 (2026-04-17)

Dev H 单条路径：**强制 occupancy=1 架构性不可能**：

- **Dev H** — FORCE_OCC1 flag：编译器忽略 request（物理事实：512-thread block 在 4-SIMD CU 上最少 2 waves/SIMD），A/B +0.023% 噪声；叠加 pipeline 扩展 → 63 spills −63.36%。**Occupancy-knob 轴已关闭**。
- **结论更新**：occupancy=1 不是"难"也不是"高风险"，是**算术不可能**。要 occ=1 只能改 block 大小（不同 kernel 结构，基本是整个项目重写）。下一轮可行的结构方向收窄到 3 条：SCALE_LDS 完全替代 / AGPR fused-asm block / preshuffle layout 重设计。

**6 轮 / 8 dev agents 全 reject**。SKILL dead-ends 列表再 +1（FORCE_OCC1 的架构不可行证明）。

## 第七轮起：RRR / CRR 95% gate 任务 (2026-04-17)

RCR 方向已达微调上限，决策者转向 RRR / CRR 对齐目标：**≥ 2780.28 TFLOPS**（RCR × 0.95）。

### GPU7 实测 baseline

| Layout | TFLOPS | vs RCR | 状态 |
|---|---:|---:|---|
| RCR | 2926.61 | 100.0% | 参照 |
| RRR | 2794.26 | 95.48% | ✅ 已达标 |
| CRR | 2737.94 | 93.55% | ❌ 差 42.34 TFLOPS |

### CRR 差距诊断

`crr_mxfp8_exact_8wave_fastpath.inc` 主循环 (L407-461) 在每奇数 k 做：
```cpp
for (int g = 0; g < crr_a_pack_count; g++) {
    a0_scale_packs[g] = std::bit_cast<fp8e8m0_4>(
        std::bit_cast<uint32_t>(a0_scale_packs[g]) >> 16);   // C++ 层 shift
    a1_scale_packs[g] = ...;  // 再一条
}
// 再 4 条 for b0, b1
```
这 6 条 C++ shift 编出 6 × `v_lshrrev_b32` per kpair，**跟 RCR pre-HOIST_HI 完全同构**。

关键资产（已经写好，只是主循环没用）：
- `crr_mma_scaled_base<opsel_a, opsel_b>` 支持 2-bit opsel（bit 1 = byte-select lo/hi）
- `crr_exact_cA_with_b1_interleave_raw_phase<K_PHASE, INSERT_AFTER, ...>` compile-time K_PHASE
- `crr_mma_scaled_phase<K_PHASE>(...)` compile-time
- tail 路径（L463-536）已经用 compile-time `_phase<0>` / `_phase<1>`

### CRR R7-R9 评审结果 (2026-04-17)

总计 9 个 dev attempt，1 个采纳 (Dev D)，8 个 reject。

| 路径 | flag / 做法 | 结果 | 采纳？ |
|---|---|---|---|
| R7 Dev A — HOIST_HI K_PHASE templated lambda v1 | 主循环 K_PHASE 模板化 | VGPR 256 + 53 spills, A/B −54% | **拒绝** |
| R7 Dev B — HOIST_HI v2 (削减 helper inline) | 同上 + try减少模板内 inline | VGPR 256 + 91 spills, A/B −67% | **拒绝** |
| R7 Dev C — HOIST_HI v3 (再削) | 同上 | VGPR 256 + 173 spills, A/B −54% | **拒绝** |
| **R7 Dev D — PIPELINE_SCALE only** | `MXFP8_CRR_EXACT_PQ_PIPELINE_SCALE_ENABLE=1` | ✅ GPU7 reviewer 2740.55 (+0.243%)，VGPR 247→232 (−15)，spills 0，det 3/3 | **采纳，commit `8934e95c`** |
| R7 Dev E — sched_barrier 实验 (no body change) | sched_barrier 加 hint | A/B Δ ≈ 0% (噪声)。**结论：v_lshr 不在 critical path** | **拒绝（marginal）** |
| R7 Dev F — `__noinline__` outlined helper | 把 main loop body 拆出 noinline 函数 | catastrophic：correctness 46% / scratch 800–888 B/lane / A/B −97%。**AMDGPU calling conv 无法跨 noinline 边界保持 4 个 accumulator live** | **拒绝（broken）** |
| R8 Dev G — runtime branch HOIST_HI | runtime if/else 替代 templated lambda（避免 codegen 翻倍） | 3 变体全 FAIL gate；V1 256 VGPR + 13 spills，V2 manual unroll 256 + 347 spills；A/B −94%，correctness FAIL | **拒绝（broken）** |
| R8 Dev H — PIPELINE+HOIST 组合 | 同时叠加 PIPELINE_SCALE 与 HOIST_HI templated lambda | 256 VGPR + 29 spills；PIPELINE 的 SRSRC（24×32-bit）与 HOIST_HI 双 phase packs live 互相挤兑 | **拒绝（spills）** |
| R9 Dev I — HOIST + PIPELINE + 关 INTERLEAVE_B1_LDS | 删 `crr_exact_cA_with_b1_interleave`（重 8-MFMA helper），按 RRR 简单 4-MMA 结构走 + HOIST_HI K_PHASE templated lambda + PIPELINE_SCALE | 256 VGPR + 145 spills + 332 B/lane scratch | **拒绝（spills）** |

### 关键架构发现 (R7-R9)

**HOIST_HI K_PHASE templating 与 CRR 4-accumulator main loop 根本不兼容**：

- CRR baseline 247 VGPR / 254 cap → 仅 7 VGPR headroom
- K_PHASE templated lambda 把 4 MMA × 2 phase = 8 inlined MMA blocks 同时展开
- 任何 templated body doubling 都要 9–25 VGPR
- → 必然爆 256 VGPR + spills，**与是否叠加 PIPELINE_SCALE / 是否关 INTERLEAVE_B1_LDS 无关**
- 在 5 个独立尝试（Dev A/B/C/H/I）观察到同一 256-VGPR ceiling，证伪假设

**RRR 为何成功 HOIST_HI 移植**：RRR 用简单 2-MMA-call body（无 `crr_exact_cA_with_b1_interleave` 那种 8 单 MFMA + LDS interleave），K_PHASE 模板化后 inlined codegen 体积可控，在 254 VGPR cap 下能容纳。CRR 4-accumulator + interleave helper 是结构差异，删 interleave 也救不回（Dev I 已证）。

**v_lshr 不在 critical path** (Dev E)：之前以为 6 × `v_lshrrev_b32` 是 main loop bottleneck，sched_barrier 实验证伪。即使理论上消除这 6 条 shift，对总时间影响也在噪声里。HOIST_HI 在 RCR 上的成功更多来自 codegen 重排（更优的 issue order）而不是单纯减 shift。

### 已关闭的死路

- ~~HOIST_HI 移植 CRR (任何形式：templated lambda / runtime branch / outlined noinline)~~：架构性 VGPR 不足
- ~~CRR HOIST_HI + 任意辅助 flag 组合（PIPELINE_SCALE / 关 INTERLEAVE）~~：仍然 spills
- ~~`__noinline__` outlined helper~~：AMDGPU calling conv 不能跨 noinline 保持 4 accumulator live
- ~~sched_barrier-only 调度优化~~：v_lshr 不在 critical path，调度变化在噪声内

### 未来可探索方向（高风险结构性，本会话不做）

- **CRR 4-accumulator → 2-accumulator 重构**：合并 cA/cB/cC/cD → 减一半 VGPR live，留出 templated lambda 空间
- **KPAIR_LOOP 移植 CRR**：参考 RCR 的 KPAIR loop 改造，可能改变 register lifetime 形状
- **重写 `crr_exact_cA_with_b1_interleave_*`**：拆掉 8 个独立 MFMA + LDS interleave，换更紧凑的实现
- **CRR scale loader 重设计**：bypass 现有 scale pack 路径，类似 RCR 的 SGPR-SRD path

### 后续计划

1. CRR PIPELINE_SCALE 已 commit (`8934e95c`)，达到当前微调上限
2. **CRR 不再值得短期投入小改动**：剩 39.73 TFLOPS 必须靠结构性重构，下一会话若要继续应**单条深入**而非并行 dev fan-out
3. RRR 监控；若后续回退到 95% 以下再补
4. 回到 RCR 的剩余 4.70% 差距（三条高风险结构方向）

## 第十轮评审结果 (2026-04-17)

主 agent 转向**先诊断后开方**：先用 rocprofv3 + ASM census 找真实瓶颈，再派 dev。这一轮的核心收获**不是优化**，而是**用三角验证锁定根因**（非-MFMA issue 争用，源头在 LDS 管道）。

### 诊断阶段（2 个并行 analysis agent）
- **rocprofv3 GPU6 8192³ counters (CRR vs RRR)**：MFMA 数量相同（16.7M），MFMA busy cycles 完全相同；但 SQ_BUSY_CU_CYCLES +3.75% / SQ_INSTS_VALU **+61%** / SQ_INSTS_LDS **+50%** / SQ_WAIT_INST_LDS +20%。Per-MFMA 执行：CRR 2.70 VALU + 1.50 LDS vs RRR 1.68 + 1.00。**MFMA 管道完全饱和**，差距 100% 来自非-MFMA issue 争用
- **ASM census per body**：CRR 144 `ds_read_b64_tr_b8` (8 B/读) vs RRR 64 `ds_read_b128` (16 B/读) + 64 `ds_read_b64_tr_b8`。CRR 多 **80 个** LDS 读指令
- **结构根因**：CRR 的 A 侧用 col-major LDS 布局（`A_col_reg` = `rt_fp8e4m3<BK=128,RBM=64,col_l,rt_128x16_s>` = 128 dwords） vs RRR 的 row-major（`A_row_reg` = 16 dwords，**8× 小**）。Col-major A 必须用窄的转置读，这是 layout 内禀属性
- **重要纠偏**：之前以为 RRR 只有 2 个 accumulator——错的，RRR 也是 4 个 cA/cB/cC/cD（同 CRR）。R7-R9 提出的 "CRR 4-accumulator vs RRR 2-accumulator" 假说**完全错误**。真正区别只在 A_col_reg vs A_row_reg

### 开方阶段（3 个并行 dev agent，全 reject）

| 路径 | flag / 做法 | 结果 | 采纳？ |
|---|---|---|---|
| **R10 Dev J** — 2× kpair unroll **WITHOUT** K_PHASE templating | `MXFP8_CRR_EXACT_PQ_KPAIR_UNROLL2_ENABLE` | FAIL，VGPR 232→256 + 26 spills + 104 B/lane scratch。即使无 templating，body doubling 仍触发 live-range 翻倍（phase-0 的 a/b prefetch 撑到 phase-1）。**与 R7-R9 templated 失败同根** | **拒绝（spills）** |
| **R10 Dev K** — `>>16` shift coalesce + `wn*RBN` precompute | `MXFP8_CRR_EXACT_PQ_VALU_TRIM_ENABLE` (split: `_LSHR_PK_ENABLE` + `_INTERLEAVE_PRECOMPUTE_ENABLE`) | MARGINAL，Δ ≈ 0% (VGPR 232 不变 / 0 spills)。**编译器已自动 hoist `wn*RBN`**——profiler "32 v_add per body" 是 pre-hoist 静态分析，不是最终 ISA。`v_alignbit_b32` 与 `v_lshrrev_b32` 占同一 issue pipe | **拒绝（marginal，编译器已优化）** |
| **R10 Dev L** — load reordering (b1 pre-issue + scale hoist) | `MXFP8_CRR_EXACT_PQ_LOAD_REORDER_ENABLE` (sub-A 'b1 pre-issue', sub-B 'scale reorder') | FAIL，sub-A −0.49% / sub-B −1.78% / combined −3.14% (correctness/det 全 OK)。**`lgkmcnt` 等待同时覆盖 LDS + scalar/VMEM scope**——重排不能让 scale buffer_load 与 A/B LDS 真正并行；反而把 scale dest VGPR live range 撑过 A/B 读寄存器期 (232→254) | **拒绝（regression + 揭示 lgkmcnt 同步语义）** |

### R10 关键产出（不是优化，是死路确认 + 真实瓶颈定性）

**真实瓶颈**：CRR 受限于 LDS 管道争用 + lgkmcnt 同步范围，**不是** MFMA、**不是** v_lshr、**不是** opsel 计算、**不是** cache miss、**不是** VMEM。要破 gate 必须改 A 的 LDS 布局（global→LDS 阶段做 transpose 让 A 侧用 `ds_read_b128` 宽读），这是大型重写不在本 sprint scope。

**已穷尽的微调维度（10 轮 / 12 个 dev attempt 全 reject 或 marginal）**：
- HOIST_HI templating (R7-R9 Dev A/B/C/H/I) — VGPR ceiling
- noinline outline (R7 Dev F) — calling conv 不能保 4 acc live
- runtime branch (R8 Dev G) — 同 templating ceiling
- 2× unroll (R10 Dev J) — 同 live-range 翻倍 ceiling
- VALU coalescing (R10 Dev K) — 编译器已优化
- Load reorder (R10 Dev L) — lgkmcnt 同步范围阻塞
- sched_barrier (R7 Dev E) — v_lshr 不在 critical path
- PIPELINE_SCALE (R7 Dev D) — **唯一 win**，+0.243%, 已 commit

### 后续可探索方向（高风险结构性，本会话不做）

1. **A LDS 布局 transpose（最高潜力）**：在 global→LDS 阶段做 transpose 让 A 侧能用 `ds_read_b128`，把 A 侧 LDS 读指令砍半。代价：global load 阶段的 swizzle 复杂化，可能影响 occupancy 和 bank conflicts。需要重写 `load_col_from_v2_st` 配合
2. **MMA 重排：cA + cC fusion**（同 a_pack 复用）：cA 和 cC 都用 b0，cB 和 cD 都用 b1。当前是 cA→cB→cC→cD（每次切 b），改成 cA→cC→cB→cD 可能减少 b 切换，但需重审 a 的 live range
3. **FORCE 4-wave with fused asm block**：换 4-wave 结构（256-thread block），单 SIMD per CU，更深 ILP——但 4-wave 历史已证劣于 8-wave (Dev A round 1)

下一会话不要再用并行 dev fan-out 微调，**单条深入做 LDS 布局 transpose**。

## 第十二轮评审结果 (2026-04-17) — Diagnostic-S 推翻 R10 LDS-pipe 假设，新瓶颈：SPI launch allocator stall

### Diagnostic-S 用 rocprofv3 测了 33 个 cycle-level counter（5 PMC chunk，GPU6 vs GPU6）

**新发现（与 R10 矛盾的核心）**：
- `SQ_LDS_BANK_CONFLICT`、`SQ_LDS_ADDR_CONFLICT`、`SQ_LDS_UNALIGNED_STALL` 在 CRR 和 RRR **都是 0**。R10 的"LDS pipe contention"假设错了，根本没有 bank 冲突
- `SQ_LDS_IDX_ACTIVE` CRR 与 RRR **完全相同**（5.03e7 cycles）。LDS unit 实际"忙碌"程度一样。CRR 的 +50% 指令数没让 LDS 单元更忙——`ds_read_b64_tr_b8` 在 LDS 内就是更轻的 op
- CRR 的 `TCP_PENDING_STALL_CYCLES` -34%，`TA_ADDR_STALLED_BY_TC` -92%，`TCP_TCP_TA_DATA_STALL` -50% —— 访存 backend 比 RRR 更轻
- CRR 的 `SQ_VALU_MFMA_COEXEC_CYCLES` +61% —— ILP 反而好

**真正瓶颈**：
- **`SPI_RA_LDS_CU_FULL_CSN +388%`** （CRR 9.80e11 vs RRR 2.01e11）—— wave 启动器在 CU 上被 LDS 占用槽位卡住，下一个 workgroup 等 5× 长才能 launch
- **`SPI_RA_RES_STALL_CSN +388%`**（同向）
- **`SQC_DCACHE_BUSY_CYCLES +129%`** —— 标量 cache 偏热
- 估算：`(9.8e11 − 2.0e11) / (224 CU × launch overhead) ≈ 3-5%` 端到端代价 —— 与 1.43% gate gap 同量级

**机理**：CRR 用 136 KB LDS/block，RCR 用 131 KB（5 KB 差）；2 blocks/CU × 136 KB = 272 KB，把 CU 的 LDS 池吃满，下一 block 等。RRR 因为 LDS 用量更小，新 block 上得快

### R12 dev fan-out 全部 timeout

派了 4 个 dev：
- **Dev O**（CRR_ROW_SHARED_TRANSPOSE 深度调试 + LDS dump 验证）：worktree 在 42f5407b base，16:51 后静默 1.5h+，**无 commit**。可能在 Step 1 LDS dump 调试卡住
- **Dev P**（CRR_USE_V3_SWIZZLE）：worktree 在 **stale main base** (b027c06b 无源)，**无 commit**
- **Dev R**（`-mllvm` 编译 flag sweep）：worktree 在 stale base，从 main checkout 拷源 build .so，**无 commit**
- **Dev T**（LDS 分配缩减 136→131 KB，**Diagnostic-S 角度**）：worktree 在 42f5407b base，活跃到 18:26（最后 .so build），**无 commit**

R12 唯一产出 = Diagnostic-S 的瓶颈定性更正。R12 commit 只能是文档。

### R12 关键产出

**正确的优化方向（按 SPI launch 假说排序）**：
1. **缩减 CRR LDS/block**（最直接命中瓶颈）：
   - 单缓冲 A 或 B（如果 PIPELINE_SCALE 允许）
   - 把 A 和 B staging 通过 union/手动 offset 重叠（生命周期不重叠时可行）
   - Scale staging area 复用
   - 目标：从 136 KB 降到 ≤131 KB（RCR 等量），让 SPI 占用槽位降到 RRR 水平
2. **`__launch_bounds__(512, 3)`**：尝试提示 SPI 预留 3 blocks/CU。需先确认 LDS 是否能容（≤ 64 KB hard 还是 160 KB？需查 gfx950 spec 或 occupancy 实测）
3. **减少 SQC_DCACHE 压力**：per-CTA 常量改 `s_load_b256` 单次加载，而不是每 iter `s_load`

**已死的方向（R12 证伪，下轮不要再投资）**：
- LDS bank conflict / address conflict 修复 —— 已证 0 个 conflict
- LDS pipe issue rate 优化 —— LDS unit 利用率 CRR=RRR
- TCP/TA 缓存 prefetch —— CRR 已经更轻
- MFMA-VALU 调度重排 —— CRR 已经 +61% coexec

### 新会话建议

1. **不要重派 R7-R11 类型的微调**（HOIST_HI、KPAIR、scale prefetch 等）—— 都已 saturated
2. **优先派 1 个深 dev 做 LDS pack/shrink** —— Diagnostic-S 的发现需要被实测验证
3. 保留 1 个 reviewer agent 做 A/B Welch-t formal
4. 全部用同一个 base（42f5407b），不要让 worktree 落在 stale main 上

## 第十四轮评审结果 (2026-04-17) — paradigm shift：gate 已在 GPU0 上达标，R7-R14 追的"1.43% gap"是测量幻觉

### R14 行动

派 2 个并行 dev：
- **Dev A** — sweep 8 个未试过的 CRR fastpath knob：CRR_INIT0_VMCNT, CRR_INIT1_VMCNT, CRR_STEADY_VMCNT, CRR_EPILOGUE_VMCNT, CRR_PREFETCH_LGKM, CRR_EXACT_B1_LDS_INSERT_AFTER (0-8), CRR_ENABLE_SCHED_BARRIER, CRR_ENABLE_STEADY_MID_BARRIER。无 LDS 改动，纯调度
- **Dev B** — single-buffer B (`Bs[2][2]→Bs[1][2]`), 在 SPI launch-allocator 假说下应是直接命中（−34 KB / −24% LDS shrink）

### R14 实测结果

**Dev A — DEAD-END**：
- 30+ 单 knob/组合 build + bench
- 最佳：b1_8+i0_3+i1_7 在 GPU7 formal **2745.59** TFLOPS（baseline 2750.92，−0.12%）
- 所有信号在 ±25 TFLOPS quick / ±5 TFLOPS formal 噪声带内
- STEADY_MID_BARRIER=0 correctness FAIL；STEADY_VMCNT≤3 −15% 大退化；其余可正确编译/运行但无 win

**Dev B — DEAD-END but PARADIGM-INVALIDATING**：
- 实现 `MXFP8_CRR_SINGLE_BUF_B` 宏（默认 0），correctness 100% PASS, SNR 49.60
- LDS 139264→104448 byte（**−34816 = −34 KB，验证 4× 于 R13 V3 swap 的 8 KB 缩减**）
- VGPR 232→256 (+24，因为新加 vmcnt 等 force scheduler 保更多 state live)
- GPU7 formal: CRR **2747.59** vs baseline 2750.92 = **−0.12% (NOISE)**
- **关键**：如果 R12 Diagnostic-S 的 SPI 假说是真的 dominant bottleneck，那么 24% LDS relief 本应大幅释放 SPI 槽位；实测仅 −0.12% → **SPI_RA_LDS_CU_FULL_CSN +388% 是 symptom 不是 cause**
- 真实 cancel mechanism: single-buffer B 强制 vmcnt(0) wait 在每个 iter，B prefetch 失去与 compute overlap，正好抵消 SPI 收益
- Diff 已 revert

### R14 决策者重测 baseline（critical paradigm-shift）

Dev B 的 GPU7 baseline 显示 RCR = 2726 TFLOPS，**远低于历史 2925.64**。立即 cross-check：

| GPU | RCR | RRR | CRR | CRR/RCR |
|-----|----:|----:|----:|--------:|
| **GPU0** | **2806.17** | **2873.02** | **2840.98** | **101.24%** ✅ |
| GPU0 (run 2) | 2813.02 | — | 2841.60 | 101.02% ✅ |
| GPU0 (run 3) | 2806.74 | — | 2838.67 | 101.14% ✅ |
| **GPU0 formal (200i × 3 det)** | 2808.44 | — | **2835.16** | 100.95% ✅ |
| GPU6 | 2708.25 | 2780.65 | 2715.83 | 100.28% |
| GPU7 (今日) | 2726.20 | 2775.90 | 2736.45 | 100.37% |

**Gate 2780.28 在 GPU0 上达标 (+54.88 TFLOPS, formal SNR 49.60 PASS, det 3/3 PASS, correctness 100%)**

### R14 综合结论 — 8 轮 sprint paradigm shift

1. **Gate 已达标**：CRR 在 GPU0 上 formal 测量 2835.16 TFLOPS > 2780.28 gate
2. **CRR 在所有 GPU 上其实都 ≥ RCR** (CRR/RCR = 100.28-101.24%)，从未存在真实的"CRR 弱于 RCR"问题
3. **R7-R14 追的"1.43% gap"是测量幻觉**：
   - 历史 baseline RCR = 2925.64 (在某个 GPU/状态下测得)
   - 当时 CRR = 2740.55 (在不同 GPU/状态下测得)
   - 两个数字组合产生"-1.43%"幻觉
   - 任何同会话同 GPU 的 RCR vs CRR 实测都显示 CRR ≥ RCR
4. **GPU7 已退化**：RCR 从 2925→2726 (−7%)，整个板子整体性能下降，不是 CRR-specific 问题
5. **R12 SPI hypothesis 已伪证**：34 KB LDS shrink → −0.12% perf 不可能是 SPI bound
6. **PIPELINE_SCALE only (commit 8934e95c) 是真正的最终 production fix**

### 已死的方向（R14 证伪，下轮不要再投资）

- CRR fastpath VMCNT/LGKMCNT/INSERT_AFTER 单 knob 调优 —— 全噪声带
- CRR LDS 缩减作为 SPI 假说命中路径 —— 34 KB shrink = -0.12%, SPI 不是 dominant
- 跨会话/跨 GPU 比较 RCR 与 CRR baseline —— 必然产生伪 gap
- "继续追 CRR +1.43%" 作为目标 —— gate 已 met

### 新规范（R14 起）

1. **Gate 验收必须同会话同 GPU 同时测 RCR + CRR**（per-iteration sync, warmup=100, iters=200, det 3, correctness ON）
2. **GPU0 是当前唯一持续达到历史性能水平的板子**；GPU7 已退化，GPU6 整体偏慢
3. **派 reviewer 时显式 `HIP_VISIBLE_DEVICES=0`** 做 final gate 验收
4. **所有 baseline 数字必须标注测量 GPU + 日期**，未来 sprint 不要再重蹈"跨会话 phantom gap"

### 新会话建议

如果 user 强制继续 CRR 优化（达标后还要继续）：
1. **先在 GPU0 上重测 RCR baseline** —— 可能 RCR 本身有 untapped 收益（GPU0 CRR > GPU0 RCR 说明 CRR 可能已经更优）
2. **或者重新定义 gate 为"今天 RCR × 0.95"** (dynamic gate) —— CRR 也已在所有 GPU 上达标
3. **不要继续微调 CRR 的 sched/wait knobs** —— R14 Dev A 已穷尽且全 noise
4. **不要继续追 LDS 缩减作为 SPI 命中** —— Dev B 已伪证
5. 如果一定要找新方向，看 RCR / RRR 是否有未优化的 inner-loop hint（GPU0 RCR 2806 vs RRR 2873 说明 RRR > RCR by 67 TFLOPS，RCR 可能有 2-3% headroom）

## 第十三轮评审结果 (2026-04-17) — V3 swizzle swap 缩 LDS 8 KB 但 fastpath 正确性破坏

### R13 行动

按 R12 Diagnostic-S 假说路径 (1)（缩 CRR LDS）选了最直接路径：把 CRR fastpath 的 A/B tile 从 `ST_v2a`/`ST_v2`（含 128 B subtile padding）改成 `ST_v3`（0 padding，预期 8 KB shrink）。代码变更：
- `crr_mxfp8_exact_8wave_fastpath.inc:41` 把 `static_assert(!CRR_USE_V3_SWIZZLE, ...)` wrap 进 `#if !CRR_USE_V3_SWIZZLE`
- 同样 wrap 4 个 `CRR_*_REG_ROW_LOAD_*` asserts
- `crr_exact_cA_with_b1_interleave_fixed_phase` 和 `_raw_phase` 函数 templated on STB（之前 hardcoded `const ST_v2&`），加 lambda：if constexpr `std::is_same_v<STB, ST_v3>` → `load_col_from_v3_st(b1, b1_tile, wn*RBN)`，else → `load_col_from_v2_st(...)`

### R13 实测结果

**构建 (V3=1 + fastpath=1)**：✅ VGPR 232→242 (+10), spills 0, occ 2 不变，**LDS Size 139264→131072 byte 完全匹配预期 8 KB shrink**

**正确性**：fastpath FAIL — 8192³ pass-rate **66966620/67108864 = 99.79%** = **142244 个 NaN 输出**，TFLOPS 表面 2457（NaN 下游传播触发 division-by-NaN slowdown），SNR=NaN

**关键 diagnostic（隔离 bug）**：
1. **fastpath OFF + V3=1**（强制走 generic gemm_kernel<Layout::CRR> 用同样 ST_v3 + load_col_from_v3_st）：✅ **PASS** SNR 49.60 dB，pass-rate 100%（但只有 2.71 TFLOPS，generic kernel 慢 1000×）→ **证明 ST_v3 + V3 col-load helpers 本身正确**
2. **fastpath ON + V3=1 + b1 interleave OFF**（`CRR_EXACT_INTERLEAVE_B1_LDS=0`）：仍 FAIL **完全相同 142244 NaN** → bug **不在 b1 interleave**，在 fastpath 更深层的 LDS write/read pipeline ordering

**结论**：V3 swap **架构上对 CRR fastpath 不兼容**，即使 LDS shrink 完美匹配 SPI 假说预期。non-fastpath barrier 重所以不暴露；fastpath pipelined ds_write 与 ds_read_b64_tr_b8 在 double-buffer 循环里有 ordering 问题（具体是否 prefill_swizzled_offsets 与 v3 swizzle 在 double-buffer Bs[2][2] 上有 stride 假设差异，未深查）

**已 revert** `crr_mxfp8_exact_8wave_fastpath.inc`，工作树恢复干净

### R13 关键产出（dead-end，不是优化）

- **V3 + non-fastpath PASSES** —— 可作为 ground truth 验证 V3 byte 布局是 OK 的
- **V3 + fastpath fails identically with/without b1 interleave** —— bug 在更基础层（global→LDS write 与 col-load read 的 pipeline ordering），不是 R13 加的 lambda
- **LDS 8 KB shrink 是真实可达的**（编译器报告确认），但触发的 fastpath 正确性破坏需要重写 fastpath 同步层才能消除

### 已死的方向（R13 证伪，下轮不要再投资）

- ST_v3 swizzle swap 的任何变体 —— fastpath 不兼容（已实测）
- 加 sched_barrier/extra `s_waitcnt lgkmcnt(0)` 救 V3 fastpath —— 任何加粗 barrier 都会消掉 SPI 收益（瓶颈是 launch allocator，不是 LDS bank）

### 新会话建议（按 SPI 假说剩余路径优先级）

1. **`__launch_bounds__(512, 3)`** —— 纯 compile-time 试验，无 LDS swizzle 改动，只在 `crr_exact_8wave_scaled_kernel` 的 `__launch_bounds__` 加 `, 3`。如果 register usage 不超 cap、不强制 spill，会让 SPI 预留 3 blocks/CU 而不是 2，直接命中 SPI_RA_LDS_CU_FULL 瓶颈。**最低风险，最高收益候选**
2. **单缓冲 B（`Bs[1][2]` 而不是 `Bs[2][2]`）** —— 直接砍 32 KB LDS（136→104 KB），让 SPI 占用槽位降到 RRR 以下。代价：B 的 global→LDS prefetch 与 register load 重叠机会减半，需要重新分析 vmcnt schedule。中等风险，需 ~50-100 LoC
3. **per-CTA 常量改 `s_load_b256` 单次加载** —— 减 `SQC_DCACHE_BUSY_CYCLES +129%`。低风险但收益不确定（可能 < 1%）
4. **静默 dev fan-out 模式不工作** —— R12/R13 都证明了，下轮直接 inline 干活更快

## 第十一轮评审结果 (2026-04-17) — A LDS 布局 transpose 两条路径全 BROKEN，B 也是窄读

主 agent 派 2 个 scout 调研可行性 → 选定两条并行 dev：

### Scout 阶段产出
- **Scout-1（A LDS transpose 可行性）**：`ST_crr_a` 现为 `ST_v2a = st_fp8e4m3<HB,BK,st_16x128_v2a_s>`（kernel_mxfp8_layouts.cpp:399-401, 4227-4234），byte-XOR swizzle `((offset>>7)&7)<<4`。RRR 用 `ST_row = st_16x128_s`。`load_transpose` 已存在（kernel_mxfp8_layouts.cpp:1726-1825），处理 col-global→row-LDS。已存在的 `CRR_A_LDS_REENCODE` macro（line 4235-4276）就是这条 transpose 的 partial impl，但通过额外 LDS write/read 而不是直接写。**Verdict**：ATTEMPTABLE，~80-150 LoC
- **Scout-2（替代 MFMA shape）**：仅有 `mfma_scale_f32_16x16x128_f8f6f4` 和 `mfma_scale_f32_32x32x64_f8f6f4` 两个 scaled FP8 intrinsic（mma.cuh:104,119,137）。RRR 和 CRR **都用 16x16x128**——no signal。32x32x64 的 A 操作数 dword 数量相同（8 dwords），但 D=floatx16 vs floatx4 → 4× 输出/issue → 可减少总 issue 量。但需要新 scaled wrapper + RBM/RBN 半化 + scale-pack 索引重设。Effort 150-300 LoC + 1-2 天

### 开方阶段（2 个并行 dev agent，全 reject）

| 路径 | 做法 | 结果 | 采纳？ |
|---|---|---|---|
| **R11 Dev M** — 直接 ST_row 替换 + reg transpose | `CRR_A_LDS_ROW_MAJOR` flag, `ST_crr_a` 改 `st_16x128_s`，主循环用 `load(A_row_reg, sub) → transpose(A_col_reg, A_row_reg)` | **BROKEN**：编译过 (VGPR 248 / 0 spills / occ 2)，但 SNR=−2.71 dB / 1340 TFLOPS。`load_transpose` 写出的 LDS 布局与通用 `load(A_row_reg, ...)` 期望的 row-major 消费不匹配。`CRR_ROW_SHARED_TRANSPOSE` 参考路径自身被 fastpath `static_assert` 关掉，无 known-good baseline | **拒绝（correctness）** |
| **R11 Dev N** — 启用现成 `CRR_A_LDS_REENCODE=1` (stepping-stone) | 强制走 reencode 路径，`load_col_from_v2a_st → transpose → store(Aenc) → load(b128)` | **BROKEN**：1017 TFLOPS / SNR=1 dB。被迫关 8-wave fastpath（`crr_mxfp8_exact_8wave_fastpath.inc:32` 硬 `#error`）；REENCODE 分支调用未-scaled `mma_AB(...)` 而不是宏 `CRR_DO_MMA(...)`——MXFP8 scale 通路未接入。**ISA 验证 wide read 原理正确**：基线 0× ds_read_b128 + 144× ds_read_b64_tr_b8 → REENCODE 48× ds_read_b128 + 144× ds_read_b64_tr_b8 | **拒绝（correctness + 非 fastpath）** |

### R11 关键新发现：B 操作数也是窄读

ASM census 144 个 ds_read_b64_tr_b8 中 ~96 来自 **B**，仅 ~48 来自 A。CRR 的 col-major B layout 同样阻塞 b128 宽读。即使 A 完美修复，**只解决 ~25% 的 LDS pressure**——A-only 修复**结构上无法**到达 +1.43% gate。

### R11 综合结论

CRR gate (2780.28 TFLOPS) 在当前架构下需要 multi-day 结构重写：
1. `load_transpose` 与 `ST_row` 的 LDS 布局对齐调试（需要 instrumented LDS dump 或先把 `CRR_ROW_SHARED_TRANSPOSE` 在 gemm_kernel 非-fastpath 跑通做对照）
2. 加 `A_row_reg` 重载到 `crr_mma_scaled_from_packs` 把 MXFP8 scale 通路接进新 A 路径（约 50-100 LoC，需 mirror RRR 的 `mma_ABt` pattern）
3. **B 操作数 layout 重设计**（解锁剩余 75% LDS pressure）—— Scout 未调研，未知是否有 partial impl

**承认 gate 当前架构不可达**。已 commit 的 `8934e95c` PIPELINE_SCALE 默认开（+0.243%, 2740.55 TFLOPS）是 R7-R11 共 11 轮唯一 strict win。下一会话若续做 CRR：先开 `CRR_ROW_SHARED_TRANSPOSE` 在非 fastpath 跑通做 known-good baseline，再决定要不要投入 multi-day 重写；或转向 RCR 剩余 4.70% 差距。

## 第二十三轮评审结果 (2026-04-18) — ★ R22 V2 stable reverify；3 dev 全 exhausted (V3 REJECTED / CRR A-LDS R11 wall reproduced / Dev C diagnostic 找 SQC dcache +441% 作为 NEW #1 lever)；0 production commit；6 ranked NEW levers for R24+

### R23 派 1 Reviewer + 3 Dev (A/B/C) 并行（GPU0/1/2/3）

按 R22 R23+ priority list 执行：(A) V2 milestone-3 进一步 VMEM-cut, (B) CRR A-LDS row-major rewrite (long-pending R11 wall), (C) V2 cycle-level diagnostic via rocprofv3 PMC.

- **Reviewer (GPU0)** — Task 1: 5x 全 V2 layout baseline reverify
- **Dev A (GPU1)** — V3 preshuffle prototype: b128+b128 替换 V2 b128+b64 via wider scale packing
- **Dev B (GPU2)** — CRR A-LDS row-major Route X: ST_v2a → ST_row + load_transpose + load(A_row_reg) + memcpy reinterpret
- **Dev C (GPU3)** — V2 cycle-level diagnostic: 32 PMC counters × 6 kernels (V2/FP8 × RCR/RRR/CRR)

### Reviewer Task 1 — 5x V2 baseline reverify: stable

- V2-RCR median **3074** (std 6.20)
- V2-RRR median **3033** (std 4.14)
- V2-CRR median **2811** (std 6.83)
- FP8-RCR median **3252** (std 6.40)
- 全 SNR ≥49.5 + det 3/3 PASS
- 跨 R22→R23 drift < 0.5% (V2-RCR -4 / V2-RRR -5 / V2-CRR +79 cross-session noise); 系统 reproducibly stable
- Gaps (R23 reverify): V2-RCR -178 / -5.5%, V2-RRR -219 / -6.7%, V2-CRR -441 / -13.6%

### Dev A — V3 preshuffle prototype: ★ REJECTED, NO COMMIT ★

**Approach**: 推 V2 b128+b64 (24B/wave) 升到 b128+b128 (32B/wave) 通过 V3 preshuffle 让 B scale pack 也能 b128。

**Side commits** (NOT cherry-picked): `3fbfa416` (V3 prototype) + `aef1d03e` (bench harness) on branch `r23-a-preshuffle-v3`.

**关键 finding (negative paradigm-shift)**: **SQ_INSTS_VMEM 是 transaction count, 不是 byte count**；b64→b128 width promotion 给 0% VMEM-issue reduction (transactions 数量不变，仅 width 增大)。R18+R21 "减 VMEM-issue count" paradigm 在 V2 之后 SATURATED — 减不动了。

**5x A/B**: V3 mean **3018** vs V2 baseline mean **3091**, **Δ -92 TFLOPS / -2.99%, Welch t=-7.46** (统计显著 regression)。

**结论**: 路径完全废弃；R24+ 必须找其他 bottleneck class (SQC dcache / SPI launch / TCC working-set)，不再追 VMEM count 削减。

### Dev B — CRR A-LDS row-major Route X: ★ FAIL CORRECTNESS, NO COMMIT ★ (R11 wall 第二次 reproduced)

**Approach**: ST_v2a (col-major) → ST_row (row-major)；`load_transpose<NT>` global→LDS + `load(A_row_reg, sub)` ds_read_b128 (16B) 替换 `load_col_from_v2a_st` ds_read_b64_tr_b8 (8B)；A_col_reg 经 `__builtin_memcpy` reinterpret 复用而非 explicit `transpose(dst, tmp)` register call。

**Build**: clean compile (VGPR 249, +15 vs baseline; LDS 135168, **-4096B**; 0 spills, occ 2 — 资源 healthy)。

**Worktree**: `/tmp/wt-r23-b` on branch `r23-b-crr-a-lds`，修改 uncommitted。

**Correctness FAIL at 8192³**: SNR **-2.71 dB** (threshold 48), 97.62% partial pass-rate, det 3/3 PASS。

**完全复刻 R11 Dev M 的失败模式** (R11 also hit -2.71 dB SNR wall on same approach)。

**Root cause analysis (R23 Dev B 比 R11 进一步)**:
- `load_transpose` + `prefill_transpose_swizzled_offsets` 写入 LDS 的物理 layout 与 `ds_read_b128` 直接读取期望的 row-MMA layout 在 lane-element correspondence 上不匹配
- 97.62% partial pass 表明 systematic permutation within K-blocks 而非随机 (suggests opsel-aware vs straight-read lane-element mapping divergence)
- 三种 next-attempt: (a) element-dump kernel 验证 actual lane-element mapping, (b) keep explicit `transpose(dst, tmp)` register call (CRR_ROW_SHARED_TRANSPOSE generic path) 而非 memcpy, 或 (c) custom `load_transpose` variant matching `ds_read_b128` semantics

**Verdict**: dead-end-with-caveat；R11 + R23 两次 hit 同一墙；CRR A-LDS 需要 multi-day microbenchmark unblock，且 Dev C diagnostic 显示上限 +50-100 TFLOPS（5-10% only）；不是 R24+ top priority。

### Dev C — V2 cycle-level diagnostic via rocprofv3 PMC: ★ COMPLETE, 6 NEW LEVERS RANKED ★

32 PMC counters × 6 kernels (V2-RCR/RRR/CRR + FP8-RCR/RRR/CRR) × 4 chunks。

**mangled name verify**: `_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv` 等等 (V2 = `<true, 2>`)；rocprofv3 substring filter `<layout>_exact_8wave_scaled_kernel` (V2) / `<layout>_exact_8wave_kernel` (FP8) confirmed 32 dispatches/chunk。

**GPU3 sclk caveat**: GPU3 idle low-power state but ramps to 1872-1995 MHz under load；`--setperflevel high` 反向 pin 到 low DPM, `auto` 是正确设置；rocprofv3 PMC mode (serialized) wall-clock degrades 但 **per-dispatch event counters 是 clock-invariant** (analysis robust)。

**6 NEW lever findings (ranked by est. upside)**:

1. **SQC_DCACHE_BUSY_CYCLES (NEW #1, R10/R12/R18 历史从未 instrumented)**:
   - V2 vs FP8 = **+441% RCR** (705k vs 130k cycles) / **+91.6% RRR** (991k vs 517k) / **+254% CRR** (571k vs 161k)
   - 根因: wave-tile preshuffle 让 scale pointer arithmetic 变 per-wave scalar code 击中 constant cache 频率剧增
   - **Lever**: precompute slab base ptrs into LDS-scalar/SGPR before K-loop OR single `s_load_dword_x4` for `(a0,a1,b0,b1)_ptr` 替换 K-loop 内 repeated SMEM loads
   - **Est upside +80-150 TFLOPS aggregate**, easy 1-day audit

2. **CRR A-LDS row-major (Dev B path)**:
   - +50% SQ_INSTS_LDS in CRR vs RRR (25.17M vs 16.78M) = 8.4M extra LDS insts/dispatch structural
   - **Bounded ~+50-100 TFLOPS** (5-10% only)
   - **R11+R23 已两次 hit dead-end SNR -2.71 dB wall** → 必须先做 layout-matching microbenchmark
   - 不是 R24+ top priority

3. **RRR TCC working-set re-tile (NEW)**:
   - V2-RRR shows **TCC_MISS +166%** (19.5M vs 7.34M) / **SQ_VMEM_TA_ADDR_FIFO_FULL +597%** vs FP8-RRR
   - 根因: B-operand fp8 fetch in RRR row-major 击中 TCC poorly
   - **Lever**: B-side block re-tile (e.g. 32×32 tiles 替换 16×128) 适配 TCC working set
   - **Est upside +50-80 TFLOPS RRR-only**

4. **V3 scale-preshuffle (Dev A path) — SATURATED on RCR**:
   - SQ_INST_LEVEL_VMEM V2 已 -41% 比 FP8；TA_ADDR_FIFO_FULL 仅 10% utilization
   - Est +0-30 TFLOPS RCR / +30-80 RRR / 0-20 CRR — Dev A REJECTED 已实测确认
   - 不是 the lever；SQC dcache 才是

5. **SPI launch occupancy fix (NEW)**:
   - 全 V2 layout SPI_RA_LDS_CU_FULL_CSN/RES_STALL_CSN/VGPR_SIMD_FULL_CSN uniformly **+18.7-18.95% over FP8**
   - VGPR pressure signal (V2 kernel 用 more VGPRs to hold wave-tile slab pack)
   - **Lever**: audit `Rpass-analysis=kernel-resource-usage` for V2 vs FP8 VGPR delta；recover 8-16 VGPRs via lifetime mgmt 解锁 occupancy
   - **Est +30-80 TFLOPS aggregate**

6. **MFMA-VALU coexec on V2-CRR (NEW)**:
   - SQ_VALU_MFMA_COEXEC_CYCLES V2-CRR vs FP8-CRR **-28.6%** (56.5M vs 79.1M)
   - 根因: V2 preshuffle scales 挤掉了 MFMA shadow 中的 helper VALU
   - **Lever**: schedule helper math (s_load → v_mov / cvt) into MFMA bubbles
   - **Est +20-40 TFLOPS CRR-only**

**Bottleneck attribution per gap pair**:
- V2-RCR (-178 TFLOPS): ~55% VMEM-issue + ~30% other (SPI/SQC) + ~10% LDS + ~5% MFMA
- V2-RRR (-219 TFLOPS): ~65% VMEM (TCC working-set) + ~15% LDS + ~15% other
- V2-CRR (-441 TFLOPS): ~45% VMEM + ~25% LDS + ~15% MFMA-coexec + ~15% other

### R23 综合产出

1. **R22 V2 三 layout 全 stable reproducible** (drift < 0.5% after 1 session)
2. **V3 preshuffle 路径 REJECTED -2.99%**: 关键 paradigm finding 是 SQ_INSTS_VMEM 是 transaction count，b64→b128 width promotion 给 0% VMEM 减少
3. **CRR A-LDS row-major 第二次 hit R11 SNR -2.71 dB wall** (Dev B confirmed naive memcpy approach 不 work)
4. **SQC_DCACHE_BUSY_CYCLES 历史从未 instrumented**: V2 paradigm 引入 +441% scalar-cache pressure
5. **R23 = 0 production commits + 1 docs commit**

### R23 confirms

- V2 paradigm 三 layout 完全 stable, no regressions across session boundaries
- VMEM-issue count cuts has saturated as a lever (R18→R21→R22 paradigm complete)
- **新瓶颈类别**: SQC dcache + SPI launch + TCC working-set 是 R24+ 焦点
- **R11 SNR wall reproducible**: CRR A-LDS layout transpose naive approach (load_transpose + memcpy) 在 R11 + R23 两次 hit -2.71 dB

### R24+ 路径（按优先级，基于 Dev C diagnostic）

1. **SQC dcache pressure cut** (NEW #1，估 +80-150 TFLOPS aggregate)：precompute slab base pointers per CTA into LDS-scalar/SGPR before K-loop；OR single `s_load_dword_x4` for `(a0,a1,b0,b1)_ptr` 替换 K-loop 内 repeated SMEM loads
2. **RRR TCC working-set re-tile** (估 +50-80 TFLOPS RRR-only)：B-side block re-tile 32×32 替换 16×128 适配 TCC working set
3. **SPI launch occupancy fix** (估 +30-80 TFLOPS)：audit V2 VGPR delta vs FP8，recover 8-16 VGPR 解锁 occupancy
4. **MFMA-VALU coexec on V2-CRR** (估 +20-40 TFLOPS CRR-only)：schedule helper math into MFMA bubbles
5. **CRR A-LDS layout-matching microbenchmark** (R11+R23 wall unblock prerequisite)：element-dump kernel mapping `load_transpose+ds_read_b128` vs `load_col_from_v2a_st` 实际 lane-element layout
6. **不要再** revisit V3 scale-preshuffle / V1 b64 / sched hints / SCALE_LDS / naive CRR A-LDS memcpy approach — 全 dead-end

### 新经验 (R23 起)

- **SQ_INSTS_VMEM 是 transaction count, 不是 byte count**：b32→b64→b128 width promotion 不减 VMEM-issue count，所以 R18+R21 paradigm "减 VMEM" 在 V2 后 saturated；下一步必须找其他 bottleneck (SQC/SPI/TCC)
- **SQC_DCACHE_BUSY_CYCLES 历史 R10/R12/R18 从未 instrumented**：V2 paradigm 引入了大幅 scalar cache pressure 是隐藏多轮的关键 overhead
- **rocprofv3 PMC mode caveat**：serialized dispatch 让 wall-clock degrade，但 per-dispatch event counters 是 clock-invariant 所以分析仍 robust（不要用 PMC mode 的 elapsed-time 做 perf 比较）
- **R11 SNR -2.71 dB wall (CRR A-LDS naive transpose)**: 不要再用 `load_transpose` + memcpy reinterpret approach；要么用 explicit `transpose(dst, tmp)` register call (CRR_ROW_SHARED_TRANSPOSE generic path), 要么写 custom `load_transpose` variant matching `ds_read_b128` semantics

## 第二十二轮评审结果 (2026-04-18) — ★ V2 推广至 RRR + CRR 双 SHIPPED ★ Dev A V2-RRR PASS BIG (+159.61 / +5.54%) cherry-picked dabeffa0; Dev B V2-CRR PASS (+20.55 / +0.76%) cherry-picked 9a0d0624; Dev C sched-hints 永久 KILLED；2 production commit；R21 V2 paradigm 完整覆盖三 layout

### R22 派 1 Reviewer + 3 Dev (A/B/C) 并行（GPU0/1/2/3）

按 R21 R22+ path priority list 第一项执行（V2 推广到 RRR/CRR）。同时 Dev C 重新 evaluate sched hints 在 V2 之上（R18 旧 baseline 已变化）。

- **Reviewer (GPU0)** — Task 1: 5x 全 layout baseline；Task 2: standby for V2 verify on each layout
- **Dev A (GPU1)** — V2-RRR milestone-1+2: mirror R21 RCR wiring，复用 Python `preshuffle_scale_matrix_mfma16_v2_rcr_a/_b`
- **Dev B (GPU2)** — V2-CRR milestone-1+2: mirror R21 RCR wiring 到 CRR fastpath
- **Dev C (GPU3)** — sched-hints diagnostic 在 V2-RCR baseline 重新 measure

### Reviewer Task 1 — baseline stable (cross-session drift < 0.5%)

- MXFP8 RCR V2 median 3071, V1 3022 (R21 reproduced)
- MXFP8 RRR V1 median ~2878
- MXFP8 CRR V1 median 2706
- FP8 RCR median 3242
- 全 SNR + det 3/3 PASS

### Dev A — V2-RRR milestone-1+2: ★ PASS BIG, COMMITTED (dabeffa0) ★

**关键 finding 解释为何 RRR 收益最大**：V1 RRR 此前 256 VGPR / **19 spills / 80B scratch** (compiler 在 V1 SRD chains + opsel templating 边界刚好溢出)；V2 collapse 到 256/0/0/0 — 这是 RRR 比 RCR (1.84%) 收益更大的根因 (5.54%)。

**dual-patch rule 简化**: RRR 单一 `load_scale_packs` lambda（无独立 warmup helper），所有 callers (main loop + pre-tail) 统一消费 V2 packs。比 RCR 简化（RCR 必须双 patch main loop + `load_scale_packs_for_pair`）。

**复用 V2-RCR Python preshuffle**: RRR A/B scale base address formulas (`a_base = br*BLK + half*HB + wm*RBM`, `b_base = bc*BLK + half*HB + wn*RBN`) byte-identical to RCR；只 A/B matrix layouts 不同，scales 完全一样。

**Production wiring**:
- `SCALE_VERSION=2` specialization of `rrr_exact_8wave_scaled_kernel`
- `dispatch_rrr_exact_8wave_scaled_v2` + `dispatch_pq_v2<RRR>` + `gemm_rrr_pq_v2` pybind
- `MXFP8_RRR_PRESHUFFLE_V2_RUNTIME` env var (default ON)

**Resource (V2)**: VGPR 256, **0 spills (V1: 19), 0B scratch (V1: 80B)**, occ 2, LDS 135 KB.

**Correctness gates** (HIP_VISIBLE_DEVICES=1, det 3/3): 256/1024/8192³ 全 SNR ≥49.5 PASS, 100% pass-rate.

**8192³ 5x A/B (Dev A GPU1)**: V1 mean ~2878, V2 mean ~3046, **Δ +168.64 TFLOPS / +5.94%, Welch t=4.76**.

### Reviewer Task 2-RRR — independent verify on GPU0: VERIFIED PASS

**5x A/B (Reviewer GPU0)**: **Δ +159.61 TFLOPS / +5.54%**, byte-exact rocprofv3 counter match.

GPU0 vs GPU1 cross-drift < 9 TFLOPS, Δ same-session 一致.

**Gap closure**: V2-RRR vs FP8 RCR ~-212 / -6.3% (vs R21 RRR estimate -371) → **43% gap close in 1 round**.

### Dev B — V2-CRR milestone-1+2: ★ PASS, COMMITTED (9a0d0624) ★

**复用 V2-RCR Python preshuffle**: scale shapes 在 RCR/CRR 之间相同 (M-major/N-major × k_blocks)。

**dual-patch rule 简化**: CRR 单一 `load_raw_scales` 加载器从 main loop / pre-tail / tail 全调用，dual-patch 简化为 single patch covering all paths.

**Production wiring**:
- `SCALE_VERSION=2` specialization of `crr_exact_8wave_scaled_kernel`
- `dispatch_crr_exact_8wave_scaled_v2` + `dispatch_pq_v2<CRR>` + `gemm_crr_pq_v2` pybind
- `MXFP8_CRR_PRESHUFFLE_V2_RUNTIME` env var (default ON)

**rocprofv3 SQ_INSTS_VMEM**: V1 212,992 → V2 180,224 = **-15.38%** (byte-exact match R21 V2-RCR).

**Resource (V2)**: VGPR 234 (V1: 232, +2), 0 spills, occ 2, LDS 139 KB.

**Correctness gates** (HIP_VISIBLE_DEVICES=2, det 3/3): 256/1024/**8192³ 49.60 dB**, 67108864/67108864 PASS.

**8192³ 5x A/B (Dev B GPU2)**: V1 median 2711.78 (std 10.61), V2 median 2732.33 (std 2.99), **Δ +20.55 / +0.76%, Welch t=4.92**.

**CRR 收益小于 RRR/RCR 的原因**: CRR baseline 已 MFMA-bound (not VMEM-issue-bound) + 已用 PIPELINE_SCALE 单 shot 4×b32；marginal cost 4 b32 vs 1 b128 + 1 b64 在 MFMA 吞吐相对小。

### Reviewer Task 2-CRR — environmental noise (GPU0 throttled)，sanity re-verify on GPU1 PASS

第一次 verify FAIL on GPU0：sclk 卡在 260MHz / mclk 2000MHz (severely throttled)；Reviewer 同时 dispatcher M_DIM rebuild 错误；Reviewer 不可能地 claim R21 RCR V2（早 ship 工作）也 FAIL → 系统性 environmental issue。

**Sanity Reviewer dispatch on GPU1** (sclk 2090MHz healthy)：byte-for-byte match Dev B's numbers，**VERIFIED PASS**.

**新规范 (R22 起)**: 任何 perf 测量前必须 `rocm-smi` 查 sclk ≥ 2GHz；GPU sclk throttle 给 misleading FAIL；dispatcher per-size rebuild required (gates on compile-time `M_DIM`)。

### Dev C — sched-hints diagnostic on V2-RCR: ABORT-DIAGNOSTIC, 路径永久 KILLED

在 V2-RCR baseline 重新 measure VMCNT-wait cycles：**V2 vmcnt wait 已 LOWER 比 FP8 (-8.8%)**.

sched hints (sched_barrier / sched_group_barrier) 没有任何可发挥空间；剩余 gap 100% 来自 structural VMEM-issue rate 而非 idle stall.

**永久关闭 sched-hints / sched_barrier / sched_group_barrier 调优方向**（R18 sched-hint REJECT + R22 V2 baseline reaffirm）.

要继续 close gap 必须做 structural VMEM-issue cuts (preshuffle V3 进一步合并？或 cache-line layout 重设计).

### R22 综合产出

1. **★ V2-RRR SHIPPED ★** +159.61 TFLOPS / +5.54% (Reviewer indep verify), gap close 43% in 1 round, commit `dabeffa0`
2. **★ V2-CRR SHIPPED ★** +20.55 TFLOPS / +0.76% (GPU1 sanity re-verify byte-exact), commit `9a0d0624`
3. **sched-hints 路径永久 KILLED** (V2 已 hide 所有 vmcnt waits, 不可能再 hide more)
4. **三 layout V2 paradigm 完整覆盖**：RCR/RRR/CRR 全 wired through `dispatch_pq_v2<L>` + `gemm_*_pq_v2` pybind + `MXFP8_*_PRESHUFFLE_V2_RUNTIME=1` default

### R22 confirms

- V2 layout paradigm 在 3 个 layout 均成功 (R21 RCR + R22 RRR + R22 CRR)
- V1→V2 收益排序 RRR (5.54%) > RCR (1.84%) > CRR (0.76%)，与 V1 baseline spills + VMEM-issue rate 排序一致
- dual-patch rule (R21 silent FAIL learning): 必须 patch BOTH main loop AND `load_scale_packs_for_pair` helper；R22-A RRR / R22-B CRR 都验证正确
- GPU benchmark sanity: 测试前必须 verify GPU sclk (`rocm-smi`) — GPU0 throttled 是 R22 CRR 第一次 verify FAIL 的根因，per-size rebuild 也是必要

### R23+ 路径（按优先级）

1. **V2 milestone-3 进一步 VMEM-cut**：V2 已 -15.4%；可能还能用 64B-aligned coalesce 或 V3 preshuffle 让 b128+b128 (32B) 替换 b128+b64 (24B)
2. **CRR 长期 gap (-15.9%)** 是新焦点（RCR/RRR 已 < 6.3%）：根因仍是 LDS-pipe / col-major A 布局 (R10 census)；A LDS row-major transpose 仍 unattempted as multi-day rewrite
3. **不要再** revisit sched hints / SCALE_LDS REPLACE / V1 layout 下 b64 — 全 dead-end

### 新经验 (R22 起)

- **复用 R21 V2 paradigm 跨 layout 极快**: RRR/CRR scale shapes 与 RCR 相同（M-major/N-major × k_blocks），所以 Python `preshuffle_scale_matrix_mfma16_v2_rcr_a/b` 直接复用，仅需 layout-specific kernel template specialization
- **GPU sclk verify 必须前置**: rocm-smi 检查 sclk ≥ 2GHz；throttled GPU 给出 misleading FAIL
- **dispatcher per-size rebuild**: dispatcher gates on compile-time `M_DIM`，必须按 size rebuild .so

## 第二十一轮评审结果 (2026-04-17) — ★ V2 SHIPPED ★ Dev A milestone-2 PASS (+55.66 TFLOPS / +1.84% / Welch t=23.33), cherry-picked 到 main (4abd4f62 + efc389ff), default-on; Dev B SCALE_LDS REPLACE 永久 KILLED；首次 21 轮 sub-200 gap 关闭（gap -228→-166）

### R21 派 1 Reviewer + 2 Dev (A/B) 并行（GPU0/1/2）

跳过 Diagnostic（R18+R19+R20 paradigm 已 triple-confirmed）。

- **Reviewer (GPU0)** — Task 1: 5x baseline；Task 2: standby for milestone PASS verify
- **Dev A (GPU1)** — preshuffle V2 milestone-2: fastpath wiring + 8192³ A/B benchmark from `r20-a-preshuffle-v2 @ f54e6dfc`
- **Dev B (GPU2)** — SCALE_LDS REPLACE milestone-1.5: defeat compiler LDS aliasing from `r20-b-scale-lds @ 5ac3229d`

### Reviewer Task 1 — baseline stable

- MXFP8 RCR median **3018.44** (std 3.00, very tight, 5 runs in [3015.46, 3022.61])
- FP8 RCR median **3243.67** (std 6.86, 5 runs in [3235.08, 3250.74])
- Gap **-225.23 / -6.94%**, drift vs R18 < 0.2% — STABLE
- 全 SNR + det 3/3 PASS

### Dev A — preshuffle V2 milestone-2: ★ PASS, COMMITTED ★

Branch `r20-a-preshuffle-v2` @ commit `1a29c562`, cherry-picked to `feat/mxfp8-only` as `efc389ff` (with foundation `4abd4f62` from R20-A milestone-1).

**Wave-tile order fix (option a from R20)**: Python `preshuffle_scale_matrix_mfma16_v2_rcr_a/_b` 在 V2 packing 前 reorder source row_groups → wave-tile gather 顺序变成 `{a0p0, a1p0, a0p1, a1p1}` (A pc=4) / `{b0p0, b1p0}` (B pc=2)，与 b128 dword 顺序匹配。

**Production wiring**:
- 新增 kernel template parameter `SCALE_VERSION` (1=V1, 2=V2)
- 新 V2 path 用 explicit `llvm_amdgcn_raw_buffer_load_b128` (A) + `llvm_amdgcn_raw_buffer_load_b64` (B) 替换 4+2 b32 chains
- 一个 wave-tile slab SRD per A and per B
- 新 dispatch `dispatch_pq_v2<RCR>` + pybind `gemm_rcr_pq_v2`
- Runtime gate `MXFP8_RCR_PRESHUFFLE_V2_RUNTIME` (default 1 = V2 on)

**关键 bug fix (worth remembering for R22+)**: 初始 256³ FAIL (SNR -0.40 dB) 真因是 `load_scale_packs_for_pair` 缺 SCALE_VERSION==2 branch；warmup/pre-tail/tail 用 V2 memory through V1 row-base pointers → garbage。**Mirror V2 b128/b64 logic 进 helper** 修复。任何 V2 推广到 RRR/CRR 都必须双 patch (main loop + helper)。

**b128/b64 emission verified** via `--offload-device-only -S`：
```
buffer_load_dwordx4 v[18:21], v175, s[28:31], s26 offen
buffer_load_dwordx2 v[192:193], v176, s[36:39], s25 offen
```

**Resource usage (V2)**: VGPR **246** (V1: 254, **-8 VGPR**), SGPR 52, 0 spills, occupancy 2 unchanged, LDS 131072 unchanged.

**Correctness gates** (HIP_VISIBLE_DEVICES=1, det 3/3 PASS, 100% pass-rate):
| Size | SNR |
|---|---|
| 256³ | 49.56 dB |
| 1024³ | 49.62 dB |
| 8192³ | **49.60 dB** |

**rocprofv3 SQ_INSTS_VMEM @ 8192³**: V1=6,815,744 → V2=5,767,168 = **-15.38%** (matches R18 model prediction exactly).

**8192³ 5x A/B (Dev A GPU1)**: V1 mean 2986.44 (std 5.77), V2 mean 3046.35 (std 4.08), Δ +59.90 / +2.0%, Welch t=18.95.

### Reviewer Task 2 — independent verify on GPU0: VERIFIED PASS

干净 worktree `/tmp/wt-r21-rev` @ HEAD `1a29c5628cbca6824f96993905c053c4ca129d4c` 重建。

**Correctness gates** (GPU0): 256³/1024³/**8192³ 49.60 dB**, 67108864/67108864 (100.00%) at 8192³, det 3/3 全 PASS。

**rocprofv3 byte-exact 匹配 Dev A**: V1 6,815,744 / V2 5,767,168 / -15.38%。

**8192³ 5x A/B (Reviewer GPU0)**:
| Path | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Median | Std |
|---|---|---|---|---|---|---|---|
| V1 | 3023.62 | 3022.43 | 3016.12 | 3021.44 | 3024.03 | **3022.43** | 3.19 |
| V2 | 3087.46 | 3078.09 | 3081.97 | 3076.06 | 3077.33 | **3078.09** | 4.63 |

**Δ median = +55.66 TFLOPS / +1.84%, Welch t = 23.33** (p<<0.001), df ≈ 7.10。

GPU0 数字略低于 Dev A GPU1 (V1 -36 / V2 -32) 是 cross-GPU drift；session-internal Δ 一致 (Dev A: +59.90 / Reviewer: +55.66)。

**Gap closure**: V2 vs FP8 RCR 3243.67 = **-165.58 / -5.10%**（vs R18 -228.42 / -7.03%；vs Reviewer Task 1 V1 -225.23 / -6.94%）→ **第一次 21 轮把 gap 关到 sub-200**。

Decision rule (Δ ≥ +30 TFLOPS AND Welch t > 3.0 AND correctness PASS): **MET on all three**, recommend MERGE.

### R21 综合产出

1. **★ V2 preshuffle SHIPPED ★** — 真实 production-impacting MXFP8 RCR 优化，+55.66 TFLOPS / +1.84% / Welch t=23.33, gap close to -165.58, 默认开启
2. **VGPR 254→246 (-8)** with same occupancy → V2 留出 future optimization headroom
3. **SCALE_LDS REPLACE 永久 KILLED** (Dev B 见下)

### R21 confirms

- V2 layout 是 21 轮唯一真实 production-impacting MXFP8 优化
- VMEM-issue rate 还是核心 bottleneck（R18 paradigm 第三次 reaffirm），但减少手段必须 native VMEM-cut（如 V2 b128/b64），不能借 LDS 中转
- V2 留 1.84% / +55.66 TFLOPS gap residual：剩余 -165.58 vs FP8 RCR

### R22+ 路径（按优先级）

1. **V2 推广到 RRR/CRR layouts**（R21 仅 wired RCR；RRR/CRR fastpath 同结构应该也能 +1-2%；effort 估 1-2 day each）
2. **V2 milestone-3：进一步压 VMEM**（V2 已 -15.4%；可能还能用 sched hints 让 b128/b64 更早 issue 来 hide 更多 latency）
3. **不要再** revisit SCALE_LDS / reuse hunt / vmcnt 假说 / 现 V1 layout 下的 b64/b128 — 全 dead-end

### R21 关键经验沉淀

- **Side-branch + Reviewer-confirm-then-cherry-pick pattern**: Dev A side branch commit → Reviewer GPU0 独立 reproduce → Decision Maker cherry-pick to main。R21 第一次成功完整跑通 production-grade workflow
- **Counter delta + correctness PASS 不足以判 GO**: SCALE_LDS counter PASS (-15.4%) 仍然 -270 TFLOPS regression。necessary but not sufficient
- **R19 linear counter-to-TFLOPS model 在 LDS path 下偏差 ~3-5x**：减 1 VMEM-issue 不等价加 1 LDS-issue + barrier cycle。仅适用 native VMEM-cut path
- **V2 推广必须 dual-patch**: main loop + warmup/pre-tail/tail helper 同时改，否则小尺寸 silent corruption (Dev A 初始 256³ FAIL 真因)

---

### R21 Dev B SCALE_LDS REPLACE milestone-1.5：**STRUCTURAL NO-GO**（Fix A + Fix B 均无效，且 SCALE_LDS 即使绕过 correctness 也是 -270 TFLOPS regression vs baseline）

### R21 Dev B (GPU2) — SCALE_LDS REPLACE milestone-1.5：**REJECT — STRUCTURAL NO-GO**

Branch `r20-b-scale-lds` @ commit `5ac3229d` 沿用，worktree `/tmp/wt-r21-b` 已 revert（纯探索，0 commit）。

**任务**：defeat R20-B 报告的 "compiler folds 6 LDS slot addresses to 3 VGPRs" miscompile，让 2048³/8192³ correctness PASS。

**Fix A (opaque pointer wrap + memory barrier)** — 试 1：
- `auto opaque_u32 = [](u32 v) { asm("v_mov_b32 %0, %1" : "=v"(out) : "v"(v)); return out; };`
- 在每个 LDS 指针 reinterpret_cast 后过一遍 opaque
- 加 `asm volatile("" ::: "memory")` barrier
- **结果**：8192³ SNR=6.62 dB，DEBUG printf 直接观察到 `cached=(00020000,00020200) direct=(817e7d81,7e7e7e7f)` ⇒ ds_read dst register **持有 LDS 地址值** 而非 loaded data。Fix A 改写 root cause 假设。

**Fix A2 (trailing s_waitcnt lgkmcnt(0) at end of load_scale_packs_from_stage)** — 试 2：
- 强制 sync 在 6 个 ds_read 之后
- **结果**：相同 cached=(00020000,...) pattern。waitcnt 来得太晚（compiler 已经 emit waitcnt 在 mfma 之前，但 dst register 在 unrolled 路径中已被读为 address）。

**Fix B (`=&v` early-clobber + per-read waitcnt)** — 试 3：
- 把 `macros::ds_read_b32` 局部展开为 inline asm，使用 `=&v` early-clobber 强制 dst VGPR ≠ smem_ptr VGPR
- 在每个 ds_read 后立即 emit `s_waitcnt lgkmcnt(0)`
- 8192³ SNR=6.63 dB **仍 FAIL**, det FAIL（max abs 0.94），TFLOPS 2340

**核心实测**（R20-B 假说 verified FALSE）：

读 `/tmp/scale_lds_fixB.s` (Fix B 编译产物) line 22639-22692 的 inner loop：
```
v_mov_b32_e32 v2, v151        ; v151 = 0x20000 + lane (slot 0 a0_pack0) ✓
v_mov_b32_e32 v3, v188        ; v188 = 0x20200 + lane (slot 2 a1_pack0) ✓
ds_read_b32 v159, v2          ; → a0_pack0 ✓
s_waitcnt lgkmcnt(0)          ; ✓
ds_read_b32 v175, v3          ; → a1_pack0 ✓
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v2, v189        ; v189 = 0x20100 + lane (slot 1 a0_pack1) ✓
v_mov_b32_e32 v3, v190        ; v190 = 0x20300 + lane (slot 3 a1_pack1) ✓
ds_read_b32 v178, v2          ; → a0_pack1 ✓
s_waitcnt lgkmcnt(0)
ds_read_b32 v176, v3          ; → a1_pack1 ✓
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v2, v191        ; v191 = 0x20800 (b0p0) ✓
v_mov_b32_e32 v3, v192        ; v192 = 0x20900 (b1p0) ✓
ds_read_b32 v171, v2          ; → b0_pack0 ✓
s_waitcnt lgkmcnt(0)
ds_read_b32 v177, v3          ; → b1_pack0 ✓
s_waitcnt lgkmcnt(0)
```
v151, v188, v189, v190, v191, v192 = **6 distinct address VGPRs** holding `0x20000 + lane`, `0x20200 + lane`, `0x20100 + lane`, `0x20300 + lane`, `0x20800 + lane`, `0x20900 + lane` (verified at lines 22317-22324)。下游 mfma_scale 在 line 22757 用 `v159, v171` (a0_p0, b0_p0)，line 22761 用 `v178, v171` (a0_p1, b0_p0) — **dst register assignments to mfma operands 全部 correct**。

**结论：R20-B 的 "compiler folds to 3 VGPRs" 假说 falsified**。assembly 没有 aliasing 问题。

**Bug source 仍未 isolate**。可能性：
1. Inter-wave LDS write/read 顺序 race（虽然 `s_barrier` 在两边都有，但 CTA-wide barrier 可能与 ds_write 的实际 sequencer commit 不同步）
2. lane permutation 不匹配 — writer wave 的 lane L 写 LDS offset L*4，reader wave 的 lane L 期望 LDS offset L*4，但 preshuffled scale 在 register 中的 lane allocation 可能与 LDS read-back 不一致
3. `volatile uint32_t scale_stage_dwords` 与 `ds_write_b32` macro 的 cache coherence — write 进 LDS bank 但 read pulls stale due to某种 SP/DC interaction

**Performance NO-GO（独立于 correctness）**：
即使绕过 correctness，**SCALE_LDS 8192³ TFLOPS = 2340 vs baseline (PIPELINE_SCALE) = 2610**，**Δ = -270 TFLOPS / -10.3% regression**。R19 Diagnostic 模型预测 +50-150 TFLOPS。模型与实测**矛盾**。可能原因：
- LDS-issue 增量（CTA-wide ds_write 16 + ds_read 6 × 8 waves = 64 ds-issues/kpair）超过 R19 估算的 +0.34M / disp
- s_barrier × 2 per kpair latency 远超 50-100 cyc 假设
- VGPR 256 (was 254) + scratch 80 bytes/lane 触发 occupancy/regalloc 退化

**最终判断**：
1. R20-B + R21-B 的 milestone-1 + milestone-1.5 共投入 ~2 day，没有定位 correctness root cause
2. 即便 fix correctness，performance 实测 **regression**，违反 R19 GO 路径前提
3. **SCALE_LDS REPLACE 路径 KILL**

### R21-B 综合产出 = 0 commit + 1 dead-end 永久封死 + R20-B 假说 falsified

1. **R20-B "compiler folds to 3 VGPRs" 假说 falsified**：实测 assembly 6 distinct VGPRs，addresses correct，dst→mfma operand mapping correct
2. **SCALE_LDS REPLACE 路径 STRUCTURAL NO-GO**：correctness root cause 未 isolate **且** performance regression -270 TFLOPS
3. **新 dead-end**：cooperative LDS scale staging + per-wave 6 ds_read pattern under current geometry (BLK=256, BK=128, RBM=64, RBN=32, WARPS_M=2, WARPS_N=4) 永久关闭

### R22+ 路径

1. **集中精力 Dev A preshuffle V2 milestone-2**（`r20-a-preshuffle-v2` branch）— 唯一剩余结构 GO 路径
2. **不要再尝试 SCALE_LDS REPLACE 任何变体**（cooperative staging + ds_write/ds_read 路径 R20-B + R21-B 已穷举主要 workaround）
3. **R20-B 报告的 LDS-aliasing miscompile 不再视为有效假说**（R21-B 实测 assembly falsified）

### R21-B 关键经验沉淀

- **Assembly 必须实测**：R20-B 凭 DEBUG printf 推论"3 VGPRs folding"是误诊。R21-B 通过 hipcc -S 直接读 device assembly，定量驳斥。下次任何"compiler miscompile"假说都要附 .s 文件 line-level 证据
- **early-clobber `=&v` 不是 silver bullet**：当下游 dst register 必须 alias 上游 src register 时（256 VGPR limit 下 register pressure 极高），early-clobber 只能让 compiler 多 emit 一次 v_mov，根本 bug 在别处
- **Counter improvement ≠ TFLOPS improvement**：R20-B SQ_INSTS_VMEM -15.4% 看上去是 R18 paradigm 实证，但 R21-B 实测 TFLOPS -10.3% regression。R19 linear model（counter 改进 → TFLOPS 改进）需要 LDS-issue cost 修正

## 第二十轮评审结果 (2026-04-17) — R19 双 GO 路径 milestone-1 实测：preshuffle V2 PASS + SCALE_LDS REPLACE counter-PASS 但 correctness FAIL；R18 paradigm 实测 reaffirm（-15.4% VMEM cut）；2 commit on side branches，0 production fastpath touch

### R20 派 2 Dev (A/B) 并行（GPU1/2 隔离）

跳过 Reviewer baseline（沿用 R18 5x median 3021.29 / FP8 RCR 3249.71 / gap -228）。

- **Dev A (GPU1)** — preshuffle V2 milestone-1：Python preshuffle + 最小 kernel reference consumer + 256³ correctness gate（fastpath 不触动）
- **Dev B (GPU2)** — SCALE_LDS REPLACE milestone-1：barrier topology 修复 + 重新启用现有 flag + rocprofv3 SQ_INSTS_VMEM kill switch

### Dev A — preshuffle V2 milestone-1：**PASS**

Branch `r20-a-preshuffle-v2` @ commit `f54e6dfc` (worktree `/tmp/wt-r20-a` 保留供 R21)。

**Files** (+549 LOC across 4 files)：
- `test_mxfp8_python.py` (+64) — `preshuffle_scale_matrix_mfma16_v2(scale_exp, pack_count)` Python 实现
- `kernel_mxfp8_layouts.cpp` (+182 at 1721-1858 + 5000-5040 pybind) — V2 lane-offset 模板 + b128/b64 loaders + verify kernel + pybind 入口，全 gated `MXFP8_RCR_PRESHUFFLE_V2_ENABLE` (default 0)
- `verify_preshuffle_v2.py` (+185 new) — CPU byte-equivalence harness
- `test_preshuffle_v2_consumer.py` (+118 new) — GPU 256³ correctness driver

**Bytewise equivalence (CPU)**：
| Size | pack_count | Match |
|---|---|---|
| 256³ | 4 / 2 | 512/512 ✅ |
| 1024³ | 4 / 2 | 8192/8192 ✅ |
| 8192³ | 4 / 2 | 524288/524288 ✅ |

**Kernel correctness gate (HIP_VISIBLE_DEVICES=1)**：
- 256³ V2 b128 (A pc=4) + b64 (B pc=2) vs V1 ref loader：**0 mismatches / 3072 compares**
- 1024³：0 mismatches / 196608 compares
- Production fastpath 256³ sanity (V2 enabled in build but fastpath untouched)：SNR 49.67 dB，100% pass-rate，证明 V2 代码完全孤立
- Resource counters：production `rcr_exact_8wave_scaled_kernel<true>` VGPR 254 / Spills 0 / LDS 131072 = identical to baseline

**R19 spec 修正**：literal `lane_byte_offset_v2 = lane_kblk*256 + lane_nonk*16` 仅适用 PC=4。PC=2 是 `lane_kblk*128 + lane_nonk*8`，k_pair stride `PC*256` not `PC*128`。已抽象为模板 `preshuffle_v2_lane_byte_offset<PC>` / `preshuffle_v2_kpair_byte_offset<PC>` 解决。

**Milestone-2 open knob (R21 必须解决)**：
- V2 slab packing 当前是 consecutive row_groups
- 但 fastpath wave-tile gather 顺序是 `{a0p0, a0p1, a1p0, a1p1}`（M offsets `{wm*RBM, +32, +HB, +HB+32}`）
- Milestone-2 要么 (a) 在 V2 packing 前 reorder source row_groups 让 wave-tile 顺序变成 `{a0p0, a1p0, a0p1, a1p1}` 匹配 b128 dword 顺序（cleaner，保持 b128 byte-stride 不变），要么 (b) 在 consumer emit dword permutation
- 现 verifier kernel 用 `__uint128_t` 直接 deref（compiler 降为普通 global load）；R21 production wiring 必须用显式 `llvm_amdgcn_raw_buffer_load_b128` intrinsic on buffer SRD（mirror `MXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE` SRD pattern at `kernel_mxfp8_layouts.cpp:2159-2186`）
- 1000-1200 LOC budget 估计仍然 realistic

### Dev B — SCALE_LDS REPLACE milestone-1：**PARTIAL（counter PASS, 2048³ correctness FAIL）**

Branch `r20-b-scale-lds` @ commit `5ac3229d` 保留；worktree `/tmp/wt-r20-b` 已删。

**Files** (+18, -2)：`kernel_mxfp8_layouts.cpp` 仅。

**R15 Dev C 真因 1 (lines ~363-373)**：SCALE_LDS 与 PIPELINE_SCALE 同时开启时两条路径都写 `*_scale_packs[]`，造成 double-write + wave-divergent VMEM arrival times at SCALE_LDS CTA-wide barrier → R15 det FAIL 根因。**Fix**：在 SCALE_LDS define block 显式 `#undef MXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE`。

**R15 Dev C 真因 2 (lines ~2225-2236)**：`sync_scale_stage_for_pair` 缺少 leading `s_waitcnt lgkmcnt(0)` + `s_barrier` 让前一轮 ds_read 在慢 wave 上排空，否则快 wave 已经覆写了 staging slabs。**Fix**：在 lambda 顶部加 leading wait + barrier。

**Gates**：
| Gate | Result | Evidence |
|---|---|---|
| 256³ MXFP8 RCR correctness | **PASS** | max_abs_err 0.0071 (= flag-OFF baseline 0.0071), SNR 49.56 dB, det PASS |
| Kill-switch SQ_INSTS_VMEM @ 8192³ | **PASS** | **5,767,168 vs baseline 6,820,000 = -15.4%** (target <6.3M) |
| 2048³ MXFP8 RCR correctness | **FAIL** | SNR 6.62 dB, deterministic 但语义错 |
| 8192³ formal benchmark | NOT RUN | per task spec correctness FAIL 跳过 |

**2048³ FAIL 根因（out-of-scope finding）**：`hipcc --offload-device-only -S` 显示 compiler 把 6 个 distinct LDS slot addresses (`block_scale_b_row_base_index(wn,0,0)` vs `(wn,1,0)` etc.) 折叠成只 3 个 address VGPRs (v2/v3/v4)，导致 6 个 ds_read 出 cached 索引值 (`00020000`, `00020100`, `00020800`, `00020900`) 而非 scale data。bug 在 K > BK*2 时浮现因为更早 slot writes 仍 alias 后续 reads。Fix 需 (a) volatile pointers / (b) opaque address casts / (c) 不同 LDS staging layout — 显式属于 milestone-1.5 而非 milestone-1。

**Resource at 8192³ build**：VGPR 256 (was 254), SGPR Spill 20, VGPR Spill 0, LDS 135168 (+4 KB scale_stage_dwords), Occupancy 2 unchanged。

**R18 paradigm 实测 reaffirm**：-15.4% VMEM-issue cut **直接证明 R19 Diagnostic 的定量模型（"减少 scale buffer_loads 数量真的会降 VMEM-issue rate"）是对的**。该路径 still GO 但需 milestone-1.5 解锁 +50-150 TFLOPS upside。

### R20 综合产出 = 2 commit on side branches + 1 production commit (docs only) + 0 fastpath touch + R18 paradigm 实测 reaffirm

1. **Preshuffle V2 layout milestone-1 PASS**：byte-equiv 全 sizes + kernel ref consumer 0 error，foundation for R21 milestone-2 (full fastpath wiring + 8192³ benchmark)
2. **SCALE_LDS REPLACE empirically validates R18 paradigm**：-15.4% VMEM-issue cut 是 R18 paradigm shift 第一次实测验证。该路径 still GO 但需 milestone-1.5 (defeat compiler LDS aliasing) 才能解锁 +50-150 TFLOPS upside
3. **Compiler LDS-aliasing miscompile** 是新发现的 hipcc/LLVM 现象

### R20 confirms

- R18 + R19 paradigm 全部经 R20 实测验证（counter 实测 -15.4%）
- Preshuffle V2 是当前唯一可执行无副作用的高 upside 路径，R21 应集中精力 milestone-2
- SCALE_LDS REPLACE 不是 dead-end，但需要 compiler workaround

### R21+ 路径（按优先级）

1. **Dev A 接力 milestone-2**：preshuffle V2 fastpath wiring + 8192³ A/B benchmark（基于 `r20-a-preshuffle-v2` branch）。必须解决 wave-tile gather 顺序 vs slab packing 顺序的失配（option a 推荐）
2. **Dev B 接力 milestone-1.5**：SCALE_LDS LDS-aliasing fix（基于 `r20-b-scale-lds` branch），优先尝试 volatile/opaque address，若仍 alias 改 LDS staging layout
3. **不要再追** reuse hunt / vmcnt 假说 / 现 V1 layout 下的 b64/b128 — R19 三类 dead-end 仍然成立

### 关键经验沉淀（R20 起）

- **Milestone gating works**：R20 前 16 轮全是 "怀疑 GO" → benchmark → fail；R20 第一次用 "milestone-1 (counter+correctness only) 才 commit production" pattern，产出 0 production breakage 但拿到了硬数据
- **Side-branch policy**：milestone-1 PASS 但未到 production-ready 时，commit 到 side branch 保留；production branch 仅 commit docs。R20 worktree `/tmp/wt-r20-a` (Dev A PASS) 保留，`/tmp/wt-r20-b` (Dev B PARTIAL) 已删但 branch 保留
- **Compiler LDS aliasing 是新已知坑**：未来任何 LDS-staging 重构都需用 volatile/opaque address 防御

## 第十九轮评审结果 (2026-04-17) — R18 paradigm shift 验证 + 两条结构性 GO 路径找到（preshuffle V2 / SCALE_LDS REPLACE）；R5/R15 dead-end 在 R18 模型下重新评估为 GO；0 commit（R20+ 实施）

### R19 派 1 Diagnostic + 2 Dev (A/B) 并行（GPU1/2/3）

跳过 Reviewer baseline（R18 刚做完 5x：MXFP8 RCR median 3021.29 / FP8 RCR median 3249.71 / gap -228 TFLOPS / -7.03%）

- **Diagnostic (GPU1)** — SCALE_LDS REPLACE 在 R18 model 下重新评估
- **Dev A (GPU2)** — preshuffle V2 layout redesign concrete prototype + integration plan
- **Dev B (GPU3)** — scale data reuse hunt (cross-iter / cross-wave / cross-half)

### Diagnostic — SCALE_LDS REPLACE: CONDITIONAL GO（R15 dead-end overturned）

R15 Dev C 的 -0.3% ~ +0.2% 估算基于 R17 vmcnt-MFMA 假说（已 falsified by R18）。R19 用 R18 cycle-counter 重新计算：

| 项目 | 当前 PIPELINE_SCALE | SCALE_LDS REPLACE | Δ |
|---|---:|---:|---|
| Per-wave per-kpair scale loads | 6 buffer_load_b32 | 2 buffer_load_b32 | -4 |
| Per-CTA per-kpair VMEM | 48 | 16 | -32 (cuts 2/3) |
| Per dispatch VMEM-issue | (full count) | (full - ~1.05M) | -1.05M |
| 关闭 +30% issue-rate gap | 100% | 33% | **67%** |
| LDS-issue 增量 | 0 | +0.34M / disp + barrier ~50-100 cyc | (small) |
| **TFLOPS recovery** | 0 | **+50-150 best case** | (vs R15 -0.3%~+0.2%) |

**Det fix path**：R15 Dev C 的"加 1 个 barrier"NOT enough。Correct fix:
- (a) 把 row_bases load 提到 CTA prologue 一次（消除 per-k_pair barrier）
- (b) re-sequence 外层 A/B barriers 让 scale barrier 嵌套同 phase
- 估 2-3 day

**GO with milestone guardrails**:
1. Milestone 1 (1 day)：实现 + rocprofv3 验证 SQ_INSTS_VMEM 6.82M → ~5.7-5.8M。如不验证 → ABORT
2. Milestone 2 (1-2 day)：修 det
3. Milestone 3 (0.5 day)：A/B 5x formal，需 ≥+30 TFLOPS 才 commit

### Dev A — Preshuffle V2 Layout Redesign：GO (highest upside)

**当前 V1 layout 限制**（R5 Dev G2 + R16 Dev C 二次 confirm）：6 scale dwords 来自 6 distinct row_groups，最小 stride 8192 B → b64/b128 不可行

**新 V2 layout**：在 wave-tile slab 内交错 row_groups，dword 级粒度
- byte 顺序：`[pack_count] x [half=2] x [k_phase_lo=2] x [lane_nonk=16] x [lane_kblk=4] x [k_pair=padded_kb/8]`
- A 侧 (RBM=64, pack_count=4)：4 dwords 连续 4-byte stride → **单 buffer_load_b128 (16B)**
- B 侧 (RBN=32, pack_count=2)：2 dwords 连续 → **单 buffer_load_b64 (8B)**
- **6 scale loads → 2，83% drop**

**TFLOPS recovery**：
- R18 +1.57M VMEM-issue / +30% rate ≈ 6 loads × 262K instr/load
- 2 loads → +0.52M / +10% issue rate
- Apply linear model：228 × (10/30) ≈ 76 TFLOPS residual gap → **recovery 150-200 TFLOPS**
- Net MXFP8 RCR 期望 **3070-3120 TFLOPS** = 95-97% of FP8 RCR
- floor 150 含 b128 wider load 2× per-issue cycle 调整，ceiling 200 best case

**Python prototype byte-verified**（worktree 已删，但 design preserved in R19 Dev A report）:
```python
# rewrite_mxfp8_v2.py: preshuffle_scale_matrix_mfma16_v2(scale_exp, pack_count)
# Smoke: rows=128, k_blocks=16, pack_count=4 → 4 packed g-bytes match
# encode_scale_matrix_raw byte-for-byte at expected (row, k_block) coords
```

**Kernel-side consumer sketch**:
```cpp
const uint32_t kpair_off = k_pair*1024 + lane_byte_offset_v2;
fp8e8m0_4_x4 packed = bit_cast<...>(buffer_load_b128(a_wave_srd, kpair_off));
a0_scale_packs[0] = packed.x; a1_scale_packs[0] = packed.y;
a0_scale_packs[1] = packed.z; a1_scale_packs[1] = packed.w;
// B: single buffer_load_b64 yielding {b0_pack[0], b1_pack[0]}.
```
`lane_byte_offset_v2 = lane_kblk*256 + lane_nonk*16` (was `lane_kblk*64 + lane_nonk*4`)

**Effort**：~3 day, 1000-1200 LOC across 5+ files
- `test_mxfp8_python.py` preshuffle (+50 LOC, parameterize pack_count)
- `kernel_mxfp8_layouts.cpp` consumer (+150/-100 LOC)
- 4 fastpath `.inc` (~200 LOC each via `build_rewrite.sh` regen)
- `rewrite_mxfp8.py` (~50 LOC)
- 3 test_mxfp8_python.py callsites

**Risk**：(1) det 低（同 byte set，只改 addr mapping）；(2) 正确性中（pack_count parameterization 紧耦合 scale tensor / kernel template）；(3) wider loads per-issue latency 高 → rocprofv3 验证

**GO recommend R20 多日实施**：sequence Dev A (Python preshuffle + ref consumer) → Dev B (fastpath asm regen + kernel rewire) → Dev C (test/benchmark/det)，~1 day each

### Dev B — Scale Data Reuse Hunt: 3 angles 全 NO

| Angle | Verdict | 原因 |
|---|---|---|
| 1 cross-iter A reuse | NO | byte_offset 推进 256 B = 64 dwords，相邻 k_pair 加载完全 disjoint dwords，0 overlap to hoist。intra-k_pair 的 k_phase=0/1 复用已被 HOIST_HI 通过 op_sel 充分利用 |
| 2 cross-wave broadcast via permlane/DPP | NO | AMD CDNA3/4 没有 inter-wave register-to-register primitive (`ds_bpermute`/`permlane16` 都是 intra-wave)。Inter-wave broadcast 必须走 LDS = SCALE_LDS path |
| 3 B scale share between A halves | NO | half=0/1 by HB=128 = distinct row_bases / distinct VMEM transactions。a0/a1 cover M-rows top/bottom 128，independent slabs not same data viewed differently |

**3 angles 全 NO confirms 唯一未证伪结构方向 = preshuffle layout 重设计**（与 Dev A V2 一致）

### R19 综合产出 = 0 commit + 2 GO 路径 + 1 paradigm shift 验证

1. **SCALE_LDS REPLACE GO** (R15 dead-end overturned under R18 model)：3-5 day, +50-150 TFLOPS, det 修复需 structural barrier re-sequence
2. **Preshuffle V2 layout GO** (R5/R16 b64 broken assumption overturned by redesigning layout)：~3 day, **+150-200 TFLOPS**, Python prototype 已 byte-verified
3. **Reuse hunt confirms** preshuffle 是唯一结构方向

### 已死的方向（R19 进一步证伪）

- 任何 cross-iter / cross-wave / cross-half scale reuse 想节省 buffer_loads（R19 Dev B 三角度 NO）
- 任何"调度 / cache 策略 / inline / prefetch / sched_barrier"角度（R3-R18 saturated；R18 Diagnostic 直接 cycle counter 证伪）
- 任何 b64/b128 coalesce with current V1 layout（R5 Dev G2 + R16 Dev C 字节 math 二次 confirm impossible）

### 新会话规范（R19 起）

1. **R20+ 推荐路径**：preshuffle V2 layout（Dev A 已 byte-verified prototype）
   - 3 day 实施，1000-1200 LOC
   - 上限 +150-200 TFLOPS（87.7% of 228 gap），Net MXFP8 RCR 期望 3070-3120 = 95-97% FP8
   - sequence Dev A (Python preshuffle + ref consumer) → Dev B (fastpath regen + kernel rewire) → Dev C (test/benchmark/det)
2. **R20+ backup 路径**：SCALE_LDS REPLACE
   - 3-5 day, +50-150 TFLOPS
   - milestone-1 (1 day) kill switch via rocprofv3 SQ_INSTS_VMEM verification
3. **不要再做 reuse hunt sprint**——R19 Dev B 已证 cross-iter/cross-wave/cross-half 全 NO
4. **不要再追 vmcnt 假说**——R18 直接 cycle counter 证伪
5. **不要再尝试 b64/b128 with current V1 layout**——byte math 二次 confirm impossible
6. 仍坚持 R15 规范：每会话必须 GPU0 baseline 重测；commit author 用 "MXFP8 Decision Maker"；worktree 必须清理

## 第十八轮评审结果 (2026-04-17) — R18 Diagnostic 推翻 R17 vmcnt-MFMA 假说；真瓶颈是 VMEM-issue 速率；1 sched_hint dead-end + AGPR feasibility downgrade；0 commit

### R18 派 1 Reviewer + 1 Diagnostic + 2 dev (A/B) 并行（GPU0/1/2/3 隔离）

- **Reviewer (GPU0)** — 5x RCR + 5x FP8 RCR baseline 重测
- **Diagnostic (GPU1)** — rocprofv3 deep cycle quantification: vmcnt vs lgkmcnt vs VMEM-issue 三方拆分
- **Dev A (GPU2)** — `MXFP8_RCR_EXACT_PQ_SCALE_SCHED_HINT_ENABLE` (sched_group_barrier 强制 scale 早 issue)
- **Dev B (GPU3)** — AGPR fused-asm Path B feasibility scout (R5 Dev F 失败真因 + 修正 path)

### Reviewer (GPU0) — baseline 重测稳定
- MXFP8 RCR 5x median **3021.29** (std 10.45, min 2998.36, max 3023.81)
- FP8 RCR 5x median **3249.71** (std 51.44, 含 2 cold-start 低尾)
- gap **-228.42 TFLOPS / -7.03%**（vs R17 -219/-6.79%）
- 所有 5+5 = 10 次 SNR 49.60-49.61 + det 3/3 PASS

### Diagnostic (GPU1) — paradigm shift：R17 假说 FALSIFIED

| Counter | MXFP8 RCR | FP8 RCR | Δ (MX-FP) |
|---|---:|---:|---:|
| GRBM_GUI_ACTIVE (cyc) | 5,888,754 | 5,043,685 | **+845K (+16.8%)** |
| SQ_VALU_MFMA_BUSY_CYCLES | 536.87M | 536.87M | 0 (identical) |
| MFMA util | **66.8%** | **78.0%** | **−11.2pp** (匹配 R17) |
| SQ_WAIT_INST_ANY (cyc) | 4.09M | 4.33M | **−241K** |
| SQ_WAIT_INST_LDS (cyc) | 0.51M | 0.59M | **−85K** |
| Derived vmcnt wait | 3.58M | 3.74M | **−157K** |
| **SQ_INSTS_VMEM** | 6.82M | 5.24M | **+1.57M (+30%)** |
| SQ_INST_LEVEL_VMEM (in-flight·time) | 60.86M | 43.77M | **+39%** |
| SQ_LDS_BANK_CONFLICT | 0 | 0 | 0 |

- **MXFP8 vmcnt + lgkmcnt 等待都比 FP8 LESS** —— compiler scheduler 已经隐藏了等待
- 真正 +845K GUI gap 来源：6 extra scale buffer_loads / K-block → +30% VMEM-issue → dispatch back-pressure (`SQ_WAIT_ANY +1.14M`)
- **TFLOPS attribution**：完美隐藏 vmcnt 期望恢复 ≈ **0 TFLOPS**（delta 为负）
- 要破 -10pp gap **必须减少 VMEM-issue 速率本身**（即减少 scale buffer_loads/K-block 的数量）

### Dev A (GPU2) — `MXFP8_RCR_EXACT_PQ_SCALE_SCHED_HINT_ENABLE` REJECT（新 dead-end）

- 在 `do_k_iter_body` 前插 `__builtin_amdgcn_sched_group_barrier(0x20, 6, 0)` + `sched_barrier(0)` 强制 6 个 scale VMEM 在 ds_read 之前 issue
- 资源完美 clean：VGPR 254 / 0 spill / occ 2 / SNR 49.56 PASS
- A/B 15 rounds GPU2：BASE 2894.93 vs EXP 2894.76，**Δ -0.006% / Welch-t -0.20**，纯噪声
- **完美 confirms Diagnostic 结论**：compiler 已经在最优位，sched hint 无可发挥空间
- **永久关闭 sched_barrier hint 方向**

### Dev B (GPU3) — AGPR fused-asm feasibility scout

- MFMA helpers map：`kernel_mxfp8_layouts.cpp:805-850` (raw + opsel_phase wrappers); `1001-1043` (per-row 2-MFMA + per-acc 8-MFMA `_impl`); 4 call sites cA/cB/cC/cD per body × KPAIR_LOOP 2 phase = **64 MFMAs/kpair**
- **R5 Dev F 失败真因找到**：4-wave fastpath (`rcr_mxfp8_4wave_fastpath.inc:213-284`) 已用 per-MFMA `asm volatile` + **`ACC16` 宏在每个 MFMA 都列出全部 16 acc tiles 为 `+a`**——R5 Dev F 只列 d0/d1，所以 compiler 在每个 MFMA 边界都重排 V↔A
- 3 条可行 path：
  - Path A 每 row 2-MFMA fuse：~80 LOC, 0.5-1 day
  - Path B 每 acc 8-MFMA fuse：~250 LOC + ACC8 macro, 2-3 days（推荐）
  - Path C 每 kpair 64-MFMA fuse：**结构不可行**（`_impl` 已被 `s_barrier` + Bs subtile loads 切开 cA/cB/cC/cD）
- HOIST_HI + KPAIR_LOOP + PIPELINE_SCALE 模板兼容性：compatible
- **Realistic upside 估算**：原本 +50-100 TFLOPS (1.5-3%)
- **R18 配合 Diagnostic 后下调至 ≈0**：AGPR fusion 解决 VGPR live-range 不解决 VMEM-issue 速率，而 R18 证明 gap 100% 来自 VMEM-issue 而非寄存器压力

### R18 综合产出 = 0 commit + 1 paradigm shift + 1 sched_hint dead-end + AGPR feasibility downgrade

1. **Paradigm shift**：R17 vmcnt-MFMA on critical path 假说 FALSIFIED。真瓶颈是 +30% VMEM-issue 速率造成的 dispatch back-pressure
2. **新 dead-end**：`MXFP8_RCR_EXACT_PQ_SCALE_SCHED_HINT_ENABLE` 永久关闭
3. **AGPR fused-asm 期望收益从 +50-100 TFLOPS 进一步下调至 ≈0**——因为不动 VMEM-issue 速率
4. **唯一仍未证伪的结构方向**：preshuffle scale layout 重设计 — 让 6 个 K-block scale loads 跨 K iteration 摊销（影响 4 fastpath + reference + 3 test caller，2-4 天工作量，上限估 ~0.5-3% TFLOPS）

### 已死的方向（R18 证伪/穷尽，下轮不要再投资）

- vmcnt-MFMA on critical path 假说（R18 Diagnostic 直接 cycle-counter 证伪：vmcnt 等待 MXFP8 反而 LESS 比 FP8）
- sched_group_barrier / sched_barrier hint（R18 Dev A：Δ -0.006% / t -0.20）
- AGPR fused-asm Path B 期望收益 ≈ 0（R18 Diagnostic：gap 是 VMEM-issue 速率，不是 VGPR live-range）
- 任何"调度 / cache 策略 / inline / prefetch"角度（R3-R18 共 16 轮，0 win）

### 新会话规范（R18 起）

1. **R17 假说 vmcnt-MFMA on critical path 已伪证**——文档已更正，下次不要再追这条
2. 真瓶颈是 VMEM-issue 速率，bottleneck 是 dispatch back-pressure 不是 idle stall
3. 唯一未证伪结构方向：preshuffle scale layout 重设计（多文件影响，上限低于 gap，但是唯一可能动 VMEM-issue 速率的杠杆）
4. **AGPR fused-asm 期望收益 ≈ 0**（除非配合 VMEM-issue 速率减少，但那需要 preshuffle layout 改动）
5. 仍坚持 R15 规范：每会话必须 GPU0 baseline 重测；commit author 用 "MXFP8 Decision Maker"

## 第十七轮评审结果 (2026-04-17) — rocprofv3 FP8-vs-MXFP8 RCR 第一次横向比较找到 vmcnt-MFMA critical-path 信号；2 条 scale-pipeline tweak 全 reject + 1 个 SMEM 神话破解；0 commit

### R17 派 1 Reviewer + 1 Diagnostic + 3 dev (A/B/C) 并行（GPU0/1/2/3 隔离）

- **Reviewer (GPU0)** — CRR 5x re-measurement 反驳 R16 的 2775.96 漂移
- **Diagnostic (GPU1)** — 第一次做 rocprofv3 FP8-RCR vs MXFP8-RCR 对比（之前 R10/R12 只比 CRR/RRR）
- **Dev A (GPU0)** — FP8 vs MXFP8 RCR 内层 ASM diff（与 Diagnostic 信号收敛）
- **Dev B (GPU2)** — KPAIR_INLINE_SCALE + SCALE_PREFETCH_N2 实验
- **Dev C (GPU3)** — "+300% SMEM" 源头 hunt via -save-temps + propose fix

### Reviewer (GPU0) — CRR 5x median 2822.30 / std 16.67
- 5/5 sample 全过 static gate 2780.28（min 2796.5, max 2841.7）
- 所有 5 次 SNR 49.60 + det 3/3 PASS
- **R16 的单次 2775.96 是 2.6σ 低端样本**，不是真实退化
- Static gate ✅ confirmed；dynamic gate ❌ -42.6 TFLOPS（与 R15 同向）

### Diagnostic (GPU1) — rocprofv3 FP8-vs-MXFP8 RCR 对比新信号
- MFMA busy% 78%→68%（**-10pp idle**）
- SQ_INSTS_SMEM 表面 "+300%"（24576 → 98304）— **后来 Dev C 证明是 cosmetic**
- SQC_DCACHE_BUSY +18%
- **关键新信号**：vmcnt(3) / vmcnt(4) waitcnt 在 MXFP8 中**紧贴 MFMA 簇之前**，FP8 中是**之后**——暗示 scale→MFMA 数据依赖在 critical path

### Dev A (GPU0) — ASM diff 收敛
确认 Diagnostic 假说。FP8 内层是 MFMA-pure；MXFP8 在每 K iter 主体之前都有一个 scale `buffer_load + vmcnt + MFMA` 的 dependency triple。**这是 -10pp MFMA util gap 的根因**（不是寄存器，不是 LDS bank conflict，不是 cache miss）

### Dev B (GPU2) — KPAIR_INLINE_SCALE + SCALE_PREFETCH_N2 全 REJECT（2 条新 dead-end）

| 实验 | flag | 结果 | 资源 | 采纳？ |
|---|---|---|---|---|
| EXP1 KPAIR_INLINE_SCALE | `MXFP8_RCR_EXACT_PQ_KPAIR_INLINE_SCALE_ENABLE=1` | formal A/B Welch-t -9.59 / **-1.95% 退化** | VGPR 254→256 + 8 spills + 36B scratch | **拒绝（永久关闭）** |
| EXP2-A SCALE_PREFETCH_N2 (B-only, BEFORE body) | `MXFP8_RCR_EXACT_PQ_SCALE_PREFETCH_N2_ENABLE=1` (variant A) | formal A/B Welch-t -8.51 / **-0.81% 退化** | clean +2 VGPR / 0 spill | **拒绝（永久关闭）** |
| EXP2-B SCALE_PREFETCH_N2 (B-only, AFTER body) | (variant B) | 主动 abort | 174 spills / 588B scratch | **拒绝（永久关闭）** |

- EXP1 根因：PIPELINE_SCALE 已经在 body 之前用 SRD/buffer_load_b32 把 scale 拿到，再 inline 一次纯属重复加载
- EXP2-A 根因：强制 per-iter A 重载（为给 prefetch slot 让位），新增的 vmcnt 又落到 critical path 上

### Dev C (GPU3) — "+300% SMEM" 神话破解（measurement artifact，不是 bottleneck）

- 通过 `-save-temps` + ISA 对比 + waves-per-block 反推
- **+73,728 extra SMEM ops 全部来自 prologue 的 `layout_globals` struct**（kernel_mxfp8_layouts.cpp:1906-1919）比 FP8 的 `rcr_exact_8wave_globals` (lean: 3 ptrs + stream，M/N/K constexpr) 多 9 个 s_load_bxxx 字段
- 8192 waves × 9 extra s_loads = **73,728 exactly**（精确匹配 perf counter）
- **量化估算**：73,728 × ~16 cycle / 1216 SIMDs / 1.7 GHz ≈ 570 ns / 10 ms kernel = **<0.006%**
- 真正的 -10pp MFMA util gap 来自 Dev A 的 per-iter scale dependency，**不是** prologue s_loads
- Refactor `layout_globals` → lean 需要碰所有 dispatch site 与 `gemm_kernel` 模板，回归风险高，benefit 低于噪声 → **不投入**
- **永久关闭 "prologue SMEM 是瓶颈" 调查方向**

### R17 综合产出 = 0 commit + 2 个新 dead-end + 1 个 myth-busting + 1 个有价值诊断

1. **新 dead-end**：`MXFP8_RCR_EXACT_PQ_KPAIR_INLINE_SCALE_ENABLE` 与 PIPELINE_SCALE 叠加 -1.95%
2. **新 dead-end**：`SCALE_PREFETCH_N2` (B-only) 变体 A -0.81% / 变体 B 174 spills
3. **Myth-busting**：FP8-vs-MXFP8 "+300% SMEM" 是 cosmetic struct artifact，runtime 占比 <0.006%
4. **有价值诊断**：vmcnt-MFMA dependency triple 是 -10pp MFMA util gap 的真因，但已知所有 prefetch 深一层尝试都触发 spill 或 vmcnt 重新落到 critical path —— 在当前 PIPELINE_SCALE 框架下 saturated
5. **R17 Reviewer 数据修正 R16**：CRR static gate 5/5 PASS，median 2822.30

### R17 confirms

MXFP8 RCR 在当前结构 + PIPELINE_SCALE pipeline 下，**所有非结构性 scale-pipeline tweak 都已饱和**。R3-R17 共 15 轮短-cycle dev fan-out 累计 0 win（R15 hygiene fix 不算优化是默认值修正）。

### 已死的方向（R17 证伪/穷尽，下轮不要再投资）

- `MXFP8_RCR_EXACT_PQ_KPAIR_INLINE_SCALE_ENABLE` 与现有 PIPELINE_SCALE 叠加（R17 Dev B EXP1）
- `SCALE_PREFETCH_N2` (B-only) 任何变体（R17 Dev B EXP2 A/B）
- 把 prologue SMEM count 当性能瓶颈调（R17 Dev C 量化证伪 <0.006% runtime）
- "再深一层 scale prefetch" / "inline 一份 scale 加载" 模式 —— PIPELINE_SCALE 已经覆盖最优，所有重叠方案都退化

### 新会话规范（R17 起）

1. **不要再做"试新 flag"或"调 prefetch / 缓存策略"sprint** —— R3-R17 反复证明短-cycle fan-out 0 win
2. 不要把 SMEM count 当 perf 信号——可能是 cosmetic struct 差异（量化估算 cycles 验证）
3. rocprofv3 FP8-vs-MXFP8 横向比较是新增的诊断手段，但 vmcnt-MFMA critical path 信号已经被 R17 EXP 证伪有可调空间
4. 如果 user 强制继续：必须**单条深度做 multi-day 结构重写**之一（AGPR fused-asm block 接续 R5 Dev F partial impl，或 preshuffle scale layout 重设计影响面 = 4 fastpath + reference + 3 test caller）
5. 仍坚持 R15 规范：每会话必须 GPU0 baseline 重测；commit author 用 "MXFP8 Decision Maker"

## 第十六轮评审结果 (2026-04-17) — 长期目标 RCR vs FP8 (-7.32%) 三条非破坏性路径全 dead-end，0 commit

### R16 派 1 Reviewer + 3 dev 并行

- **Reviewer (GPU0)** — fresh baseline，pure source defaults
- **Dev A (GPU1)** — 测试 agent_prompt.md 老 "Dev B" 段列出的 3 个未试过 flag (`PHASE_U16_CACHE / REMAP_ONCE / SCALAR_PHASE_PACKS`)
- **Dev B (GPU2)** — `-mllvm` AMDGPU compiler flag sweep（R12 Dev R timeout 路径的清算）
- **Dev C (GPU3)** — scale `buffer_load_b64` coalesce 字节级可行性研究（R5 Dev G2 的二次确认）

### R16 GPU0 baseline (Reviewer)

| Layout | TFLOPS | SNR | 状态 | vs R15 drift |
|---|---:|---|---|---:|
| **MXFP8 RCR** | 3010.57 | 49.60 PASS | ✅ | -0.17% |
| **MXFP8 RRR** | 2862.99 | 49.59 PASS | static ✅ / dynamic ✅ (+2.95) | -0.91% |
| **MXFP8 CRR** | 2775.96 | 49.60 PASS | static ❌ -4.32 / dynamic ❌ -84 | -1.93% (噪声边缘) |
| **FP8 RCR** | 3229.67 | 49.61 PASS | (gap 219 TFLOPS / 6.79%) | -0.74% |

注：CRR 这次落到 static gate 下 4.32 TFLOPS 是跨会话漂移，**非真实退化**（同代码同 commit）。FP8 RCR 第一次冷启动 1984 TFLOPS（DVFS 低功耗），后续恢复 3229。

### Dev A 结果 — 3 个老 flag 全 REJECT (架构互斥)

源码 `kernel_mxfp8_layouts.cpp:2587-2605, 2643-2661, 2698-2716, 2752-2770` 的 dispatch chain：
```
#if SCALAR_PHASE_PACKS
#elif PHASE_U16_CACHE
#elif REMAP_ONCE
#elif HOIST_HI       ← 当前 production
#elif OPSEL_PHASE
#else fallback
```

- `PHASE_U16_CACHE=1` —— 关 HOIST_HI → SNR -1.18 dB FAIL（K_PHASE templated lambda + tail path 不兼容）
- `REMAP_ONCE=1` —— VGPR 254→256 + 1 spill + 8B scratch
- `SCALAR_PHASE_PACKS=1` —— VGPR 254→256 + 1 spill + 8B scratch

**永久关闭这 3 个 flag**。HOIST_HI 是它们的 successor。已从下方 "Dev B 已有开关" 段删除。

### Dev B 结果 — `-mllvm` 30+ flag sweep 全 NO-WIN

测过：`promote-alloca-to-vector-limit`, `loop-prefetch`, `set-wave-priority`, `schedule-relaxed-occupancy`, `schedule-metric-bias={50,100}`, `kernarg-preload-count={2,4,8,16}`, `use-amdgpu-trackers`, `disable-clustered-low-occupancy-reschedule`, `disable-unclustered-high-rp-reschedule`, `enable-vopd`, `reassign-regs`, `misched-{cluster,fusion,cyclicpath}`, `enable-post-misched={true,false}`, `enable-pipeliner`, `sched-strategy={minreg,max-ilp,iterative-ilp,iterative-minreg}`, `enable-merge-m0`, `opt-vgpr-liverange`, `dce-in-ra`, `enable-amdgpu-aa`, `prealloc-sgpr-spill-vgprs`, `membound-threshold`, `disable-loop-alignment`

Top-2 quick-bench candidate formal A/B：
| Flag | Quick Δ% | Formal Welch-t | Verdict |
|---|---:|---:|---|
| `promote-alloca-to-vector-limit=2` | +0.51% | -0.01 | noise |
| `use-amdgpu-trackers` | +0.48% | -0.71 | noise |

资源 byte-identical baseline → 编译器对该 hot kernel 是 no-op。Makefile 已经默认 `-O3 -ffast-math --offload-arch=gfx950 -DKITTENS_CDNA4`。**永久关闭 `-mllvm` 调优方向**。

关键发现：
- `-enable-pipeliner` (LLVM SWP) 在 AMDGPU MFMA 循环上 silently inert
- `-enable-post-misched=false` 退化 25% → 默认开是必要的
- 所有 `sched-strategy` 替代项都退化 → 默认 GCN scheduler 是最优

### Dev C 结果 — scale b64 coalesce 字节级证伪

6 个 scale `buffer_load` 的精确映射（生产 ASM 内部循环 `.LBB16_4`）：

| Dest VGPR | SRD | 角色 |
|---|---|---|
| v183 | s[24:27] | A scale half=0 pack=0 |
| v188 | s[40:43] | A scale half=1 pack=0 |
| v190 | s[48:51] | A scale half=0 pack=1 |
| v187 | s[52:55] | A scale half=1 pack=1 |
| v184 | s[36:39] | B scale half=0 pack=0 |
| v189 | s[44:47] | B scale half=1 pack=0 |

`preshuffle_scale_matrix_mfma16` 输出 `(num_row_groups, padded_k_blocks*32)`，每个 row_group 是 8192-byte 连续 slab。6 个 scale dword 落 6 个不同 row_group，最小 dword 间距 a0p0→a0p1 = **+8192 B**（同 SRD 内）。

`buffer_load_b64` 要求 4-byte 间距 → **没有任何一对 dword 满足**。R5 Dev G2 字节 math 二次 confirm。真 b64 coalesce 需 Python preshuffle 重排（dword 级 interleave row_groups），影响 4 个 fastpath + reference + 3 个 test caller，**估 2-4 天，上限 ~0.5% TFLOPS**。**永久关闭**。

### R16 综合产出 = 4 个永久 dead-end + 0 commit

1. `PHASE_U16_CACHE / REMAP_ONCE / SCALAR_PHASE_PACKS` flag（HOIST_HI 互斥）
2. compiler `-mllvm` flag 调优（30+ flag saturated）
3. scale `buffer_load_b64` 合并（preshuffle layout 不允许，字节 math 证伪）
4. 跨 session GPU0 baseline 1-2% 漂移（R14/R15/R16 一致 confirm）

### R16 confirms

MXFP8 RCR 在当前结构下 **3010-3015 TFLOPS 是硬 ceiling**。FP8 RCR 6.79% 差距只能通过 multi-day 结构重写攻克：

**剩余仅两条结构性方向**：
- AGPR fused-asm block（R5 Dev F 有 partial impl 可接续，破坏 HOIST_HI）
- preshuffle scale layout 重设计（影响 4 fastpath + reference + 3 test caller）

A LDS row-major transpose 在 R10/R11 已半路证伪 fastpath 不兼容（pipeline ordering bug），第三条结构方向可视为 also dead unless 重写整个 LDS 同步层。

### 新会话规范（R16 起）

1. **不要再派"试新 flag"sprint** —— R3-R16 共 14 轮证明了短-cycle dev fan-out 在当前结构下 0 win
2. 如果 user 强制继续：必须**单条深度做 multi-day 结构重写**之一
3. 优先 AGPR fused-asm block（接续 R5 Dev F 的 256V→128V/128A 分配框架，但需要重写 fused asm 让 ALL MFMAs 在 1 个 asm block 内避免 V↔A shuffle 开销）
4. preshuffle scale layout 影响面太大，risk 高，benefit 上限 0.5% — 不优先
5. 仍坚持 R15 规范：每会话必须 GPU0 baseline 重测；commit author 用 "MXFP8 Decision Maker"

## 第十五轮评审结果 (2026-04-17) — defaults hygiene fix：源默认值终于匹配生产 build；R14 倒置反转

### R15 派 4 并行 agent
- **Reviewer**（GPU0 formal baseline RCR/RRR/CRR + FP8 RCR/RRR/CRR）
- **Dev A**（RRR vs RCR ASM diff，调查 R14 "RRR > RCR" 倒置）→ defaults hygiene 重大发现
- **Dev B**（`__launch_bounds__(512, 3)` 提示 SPI 预留更多 slots）
- **Dev C**（SCALE_LDS REPLACE PIPELINE_SCALE feasibility study）

### Reviewer GPU0 baseline 结果
- RCR 3000.19 / RRR 2886.93 / CRR 2838.08 / FP8 RCR 3242.25
- **历史顺序 RCR > RRR > CRR 恢复**——R14 的 "RRR > RCR 倒置" 是 cold GPU 状态 artifact

### Dev A — defaults hygiene 重大发现
- 源文件 `kernel_mxfp8_layouts.cpp` 的 `MXFP8_RCR_EXACT_PQ_KPAIR_LOOP_ENABLE` 与 `MXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE` 默认值是 `0`，但 README/SKILL 列为 production "current best" flag
- `Makefile` 与 `build_rewrite.sh` **不传任何 -D flag**，所以 fresh `make` 走 fallback 慢路径（gemm_tail_kernel 0.88 TFLOPS）
- Dev A 5+5 A/B：default-0 RCR=2810 → flip-to-1 RCR=2989 (+178 TFLOPS / +6.34%)
- Commit `98c80c20` cherry-pick 到主分支

### 决策者深度审计：扩大 hygiene fix
发现不只 KPAIR_LOOP/PIPELINE_SCALE，**所有** production flag 默认 0，包括：
- `MXFP8_RCR_EXACT_8WAVE_FAST_ENABLE 0` — 不 enable 这个，整个 RCR 8-wave 内核不会编译进二进制
- `MXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE 0`
- `MXFP8_RRR_EXACT_8WAVE_FAST_ENABLE 0`
- `MXFP8_CRR_EXACT_8WAVE_FAST_ENABLE 0`

**Commit `a8237d01`**：把 4 个 fastpath gate flag 全翻 0→1。Pure source defaults rebuild → RCR 3015.73 / RRR 2889.36 / CRR 2830.67，formal SNR 49.59-49.60 PASS, det 3/3 PASS, correctness 100%。FP8 unchanged.

### Dev B — DEAD-END（永久关闭）
`__launch_bounds__(512, 3)` 编译器**完全 ignore**——MI355X CU LDS = 160 KB，CRR 用 135-139 KB/block 已经把 occupancy cap 在 1 block/CU。VGPR/LDS/Spill/Occ 全部 byte-identical baseline。要 occ 提升必须先解决 LDS 预算（R14 已证 -34KB → -0.12% 净变化，结构上无路）

### Dev C — DEAD-END（永久关闭）
SCALE_LDS REPLACE PIPELINE_SCALE feasibility study 完成：
- R4 stack 失败的 det bug 根源：`sync_scale_stage_for_pair` 在 `do_k_iter_body` 内的 barrier topology mismatch（与 SGPR-SRD 路径并发）
- 即使完美修复 det，cost-benefit 净 −0.3% ~ +0.2%（节省的 ~6 个 SGPR-SRD scale load 已被 MFMA latency 隐藏；新增的 16 ds_write + 24 ds_read + 1 CTA-wide barrier 反而吃 50-100 cycle）
- 估计 2-4 天工作量，期望收益低于噪声 floor
- R5 三条结构性高风险路径（SCALE_LDS / AGPR fused-asm / preshuffle layout）减为两条

### R15 关键产出
1. **Defaults hygiene commit `a8237d01`** —— 源默认值终于匹配文档化生产 build；fresh `make` 不再产出慢 0.88 TFLOPS tail kernel
2. **R14 paradigm-shift 反转**：RCR > RRR > CRR 顺序恢复（R14 的"倒置"是 cold GPU + missing flag 双重测量 artifact）
3. **gate 双 PASS**：静态 gate 2780.28 RRR/CRR 全 over；动态 gate 2864.94 RRR over，CRR 差 1.14%（小幅 miss）
4. **MXFP8 RCR 真实数字 3015.73**（不是 R14 测的 2806），与 FP8 RCR 3253.80 仍差 238 TFLOPS / 7.32%（长期目标）

### 已死的方向（R15 证伪/穷尽，下轮不要再投资）
- `__launch_bounds__(_, N>1)` 调 occupancy（R15 Dev B）—— LDS 预算硬 cap 在 1 block/CU
- SCALE_LDS REPLACE PIPELINE_SCALE（R15 Dev C）—— 即使完美修复 det，期望净收益 < 噪声
- 重做 RCR vs RRR ASM diff 想找"RRR 比 RCR 快"的根因（R15 Dev A）—— 已证那是 missing flag artifact

### 新会话规范（R15 起）
1. **每次 session 必须重测 GPU0 baseline**——R14 的 GPU0 RCR=2806 vs R15 的 RCR=3000 差 194 TFLOPS，可能源于 GPU 热状态/firmware/clock，不能跨 session 直接对比
2. **不要再做"补 95% gate" sprint**——R14/R15 多次 confirm 多 GPU 多状态都达标
3. 若 user 强制继续：转向 RCR vs FP8 的 7.32% 差距（238 TFLOPS）。剩余结构性方向：AGPR fused-asm block / preshuffle scale layout 重设计 / A LDS row-major transpose（R10/R11 半路尝试但破 fastpath）
4. **commit/cherry-pick 时 author 用 "MXFP8 Decision Maker"**（无全局 git config，用 `-c user.email=... -c user.name=...`）
5. **dev worktree 一律 `/tmp/wt-<round>-<letter>`**，最后做 `git worktree remove --force` + `git branch -D <round>-dev-<letter>` 清理

## 工作流

1. Decision maker 每轮选 1–3 个最有把握的方向（不要 4 个同时开花）
2. 并行派 dev agents（Task 工具，不同 `HIP_VISIBLE_DEVICES`）
3. Dev 完成 → reviewer 门禁 → 胜出者 commit
4. 对话结束前：
   - `git status` 必须干净或只有预期的未跟踪文件
   - `TODO.md` 进度已刷新
   - `agent_prompt.md` baseline 已更新
   - SKILL 若有新 durable finding / dead-end 已补充

## 禁止清单

- Python 侧的正确性绕行（`.t().contiguous()`、host-side padding）
- commit `.so`、生成的 `.s`、`_layout_results_*.json`、`__pycache__`
- 未过 smoke 就跑 formal
- 基于短跑（iters < 50）宣布胜利
- 混用 batch timing 和 per-iter timing 的数字做对比
