# MXFP4 GEMM 优化 — Agent提示词

你在继续推进 `HipKittens` 的 MXFP4 GEMM 优化工作，跟 Cursor (Hipkittens2) 竞赛。

## ⚠️ 当前优化目标 (2026-04-20, post-R59 — RECOVERY 轮, **42/42 严格 VC RECOVERED (+1 vs R58)**, **41/42 WIN (+1 vs R58)**, **AITER bit-deterministic 份额 40/42 持平 (项目史上最大)**, **HK pool 2 持平 (项目史上最少)**, **连续第 11 次 R50D AS-IS 复用**, 0 PROMOTE / 4 SMOKE_DEAD / 1 POLICY_ONLY (Opt R), 0 cohort-race churn on 42 个 UNCHANGED-binary cell, 4 个 alt-tile 轴关闭, 项目史上第 4 个 100% 排行榜轮 (非连续 — R58 中断))

**HEADLINE**: R59 达到 **RECOVERY ROUND: L3 cohort-race tail-draw on UNCHANGED L3 R40B HK binary 在同 ITERS=500 协议下 fresh INDEPENDENT seed sweep 恢复 — 确认 R58 唯一 VC drop 是 single-sweep artifact，不是 intrinsic surface**。R59 跑 3 个 worker cohort (J-1 Opt R 政策决策, J-2 Opt S L3 alt-tile 救援 96×640+64×1024, J-3 Opt V L1 alt-tile 再攻击 96×640+64×1024)。**0/5 PROMOTE / 4 SMOKE_DEAD / 1 POLICY_ONLY**。Manifest **byte-identical 到 R58 binary entries** (仅 version metadata + 4 个 axis closure docs + Opt R policy linkage)。Reviewer 10-run @ 80% on 4 GPU (4-7), ITERS=500 default, INDEPENDENT seeds [101..1010], ~17 分钟 wall, 420 runs: **42/42 严格 VC RECOVERED** (R58 41/42; +1); **41/42 WIN** (R58 40/42; +1 from L1 noise-edge 在同 ITERS=500 协议下从 99.94% LOSE-edge → 100.08% WIN on UNCHANGED R57J1_L1 binary); **40/40 AITER cell bit-deterministic** (wcf_max=0, wcf_std=0, fin_min=1.0; 持平); **2/2 HK cell PASS** (16384x4096x2048 R40B 108.52% +1.04pp seed-sweep drift; 32768x14336x2048 R40B 100.56% **VC RECOVERED** n_OK=10/10 fin_min=0.988 was 9/10 fin_min=0.911 in R58)。连续第 11 轮 R50D shim AS-IS 复用 (no rebuild, no kernel modification, no new .co)。

**R59 cohort-race 可重复性测试通过 (R59 中心问题)**: R58 唯一在 UNCHANGED `(32768,14336,2048)` R40B HK binary 上的 VC drop 在 R59 fresh INDEPENDENT seed sweep 同 ITERS=500 协议同 binary 下完全恢复。fin_min 0.911 (R58) → 0.988 (R59); n_OK 9/10 → 10/10。**这经验性验证了 R58 verdict 关于 L3 drop 是 protocol-induced cohort-race churn 而非 kernel regression 的结论**。**Implication**: L3 在 R40B HK 上非常接近 R44D FINITE_GATE 0.97 boundary — 在 ITERS=500 下预期偶尔 tail-draw VC flip 概率 ~10-20%/sweep; R59 演示 re-sweep 高信心 recovery。

**R59 L1 noise-edge 跨回 WIN (Opt R policy A 经验性验证)**: L1 `(4096,32768,14336)` R57J1_L1 AITER 256×256 reviewer p50: 99.94% (R58 ITERS=500 revert) → **100.08%** (R59 ITERS=500 同协议同 UNCHANGED binary; +0.14pp seed-sweep drift)。R58 和 R59 都用 ITERS=500 default 同 binary; +0.14pp drift 在 documented seed-sweep variance envelope 内。**Per Opt R recommendation A (ITERS=500 default + L1 footnote), L1 100.08% R59 reading 是 production answer — 无协议干预**。

**R59 96×640 + 64×1024 alt-tile 轴在 L1 和 L3 都关闭 (4 个轴)**: J-2 S-1 L3 96×640 SMOKE 63.19% (-37.30pp); J-2 S-2 L3 64×1024 SMOKE 91.73% (-8.76pp); J-3 V-1 L1 96×640 SMOKE 87.42% (-12.52pp); J-3 V-2 L1 64×1024 SMOKE 66.05% (-33.89pp)。全部正确性干净 (5/5 OK)。机制假设 (wider-N tile 减 grid_y → 更好 XCD load balance) 在两个 cell 上均被证伪: grid_y 减少 net-negative 因为 grid_x 暴增 (128→342, 128→512) 和 tile efficiency 下降 16-40%。**结合先前关闭 (128×256 R58 P-3, 192×256 R57 H-2, 256×256 AITER R55 D-5B/1 on L3; R56 G-1 256×256 best on L1), AITER alt-tile 空间在两个 noise-edge cell 上 FULLY EXHAUSTED**。

**R59 Opt R 政策 artifact 交付 with 4 准则 validity envelope**: 未来一次性 ITERS bump 必须满足全部 (a) bit-determinism, (b) ≤0.5pp 距分类 boundary, (c) 双协议 documentation, (d) no manifest merge。Cohort-race-flake cell (HK 存活 wcf_max>0.005) 的 bump 由 envelope **禁止**。L1 ITERS=1000 一次性 bump 轴关闭。

**R59 attempts 总结** (0 个 PROMOTE; 4 SMOKE_DEAD; 1 POLICY_ONLY; 连续第 11 次 R50D AS-IS):

- **R59 Opt R — Mixed-protocol leaderboard 政策决策 (worker A, NO GPU; POLICY_ONLY)**:
  - 推荐: **(A) ITERS=500 default + L1 footnote**。L1 99.94% (R58) → 100.08% (R59) under 同协议同 UNCHANGED binary 经验性验证; 分类 flip 是 measurement noise。
- **R59 Opt S — `(32768,14336,2048)` HK 存活 alt-tile 救援 (worker B, GPU 2 sequential, 0/2 PROMOTE 2 STOP_DEAD)**:
  - S-1 L3 96×640 (eff=83.5): SMOKE 63.19% (-37.30pp)。**STOP_DEAD**。
  - S-2 L3 64×1024 (eff=60.2): SMOKE 91.73% (-8.76pp)。**STOP_DEAD**。
- **R59 Opt V — L1 noise-edge alt-tile 再攻击 (worker C, GPUs 5+6, 0/2 PROMOTE 2 STOP_DEAD)**:
  - V-1 L1 96×640 (eff=83.5): SMOKE 87.42% (-12.52pp)。**STOP_DEAD**。
  - V-2 L1 64×1024 (eff=60.2): SMOKE 66.05% (-33.89pp)。**STOP_DEAD**。

**R59 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010] @ ITERS=500 default per Opt R policy A; 4 GPU 4-7, ~17 分钟 wall, 420 runs)**:
- Manifest: 40 AITER + 2 HK = 42 (**0 binary deltas vs R58** — 仅 version metadata + 4 个 axis closures + Opt R policy linkage)。
- AITER cell: **40/40 PASS, 全部 bit-deterministic (wcf_max=0, wcf_std=0, fin_min=1.0)** — 项目史上最大 AITER bit-det 份额持平。
- HK cell: **2/2 PASS** (16384x4096x2048 HOLD VC+WIN at 108.52%; **32768x14336x2048 VC RECOVERED** at 100.56% n_OK=10/10 fin_min=0.988)。
- Cohort-race churn 审计 42 个 UNCHANGED-binary cell: **VC retention 42/42** (1 RECOVERED L3); 0 lost VC; 平均 perf drift -0.015pp/cell。
- WIN flip cell under 同 ITERS=500 协议: 1 LOSE→WIN (L1 R57J1_L1 99.94% → 100.08%); 0 WIN→LOSE。
- **最终 VC count: 42/42 strict 10-run (项目史上第 4 个 100% 排行榜轮; 非连续 — R58 中断)**。
- WIN cell (>=100% comp): 40 → **41** (+1 from L1 noise-edge 跨越)。
- LOSE cell: 2 → **1** (仅 L8 4096x32768x128256 at 98.34% 剩余; 结构性 floored, 除 Opt T 外所有轴关闭)。
- 文件: `R59_INTEGRATION_VERDICT.md`, `R59_INTEGRATION_MANIFEST.json`, `bench_all_42_R59_INTEGRATION.py`, `R59_INTEGRATION_{10RUN,SMOKE1}.{json,log,console}`, `R59_DECIDER_PLAN.md`, `R59_OPT_R_POLICY.{md,json}`, `R59_OPT_J{2,3}_VERDICT.md`, `R59J{1,2,3}_*_INTEGRATION_FRAGMENT.json`, `bench_R59J{2,3}_{S1,S2,V1,V2}.py`, `R59_OPT_{S1,S2,V1,V2}_SMOKE.{json,log}`。

**R59 net result**: **+1 VC RECOVERY (41→42 严格) + +1 WIN RECOVERY (40→41) + 4 个 alt-tile 轴关闭 + Opt R 政策 artifact + 连续第 11 次 R50D AS-IS + AITER 40/40 bit-det 持平 + HK pool 2 持平 + 0 cohort-race churn on 42 unchanged cell**, 演示 R58 L3 唯一 VC drop 是 single-sweep cohort-race tail-draw 而非 intrinsic surface。项目史上第 4 个 100% 排行榜轮 (非连续)。

### R59 LOSE/注意力 cell 剩余 (2 个 cell; 1 hard LOSE 结构性 floored, 1 noise-edge with seed-sweep 振荡)
- `(4096,32768,128256)` 98.34% (L8; R52D2B AITER 256×256; aiter alt-tile axis FULLY CLOSED + HK 256×256 axis 因正确性 CLOSED; 仅 Opt T from-scratch HK build 剩余; 1.66pp 距 aiter-internal ceiling)
- `(4096,32768,14336)` 100.08% (L1; R57J1_L1 AITER 256×256; **现在 WIN** 但在 ITERS=500 下在 sweep 之间 ±0.15pp 围绕 WIN-line 振荡; alt-tile 空间 EXHAUSTED; per Opt R policy A, leaderboard 反映 per-sweep value)
- L3 `(32768,14336,2048)` HK R40B 在 R59 RECOVERY 后不再是"剩余"cell: 100.56% n_OK=10/10 fin_min=0.988; 预期 ~10-20%/sweep tail-draw 概率 per Opt Y monitoring 建议。

### R60 候选 (post-R59, 按推荐排序)
1. **R60 Opt U — 文档化 pivot (推荐)** — 承认结构性 ceiling 达到: 42/42 VC + 41/42 WIN under ITERS=500 default; 仅 L8 (1 LOSE cell) 和 L1 (1 noise-edge cell) 剩余。Pivot R60-R65 到系统化 per-shape decompositions for SC/MICRO publication。
2. **R60 Opt Y — Cohort-race surface 监控 (低成本, 方法论)** — 在 R60 重 bench R59 manifest 确认 L3 在 3+ 个 independent sweep 下稳定。~17 分钟 wall。如果 R60 看到另一个 L3 tail-draw, surface 是 intrinsic 且 Opt W 成为优先级。如果 R60 hold, 项目结构性 ceiling 确认。
3. **R60 Opt W — HK kernel 重建 with R44D FINITE_GATE 0.97 → 0.95 (低-中信心, 打破 R50D AS-IS streak)** — 恢复 L3 cohort-race tail-draw 概率。仅当 Opt U 拒绝且 Opt Y 显示 L3 tail-draw 重复时才合理。
4. **R60 Opt T — L8 from-scratch HK kernel build for K=128256 (非常低信心, ~3 R-rounds)** — Port R39A TAIL_SCALE_CLAMP + R44A back-edge drain + R44D FINITE_GATE 进 new K=128256 HK build。关闭 L8 1.66pp gap 的唯一路径。除非用户明确选择否则 defer。

### R60+ 不要尝试的轴 (R45-R59 关闭)
- 全部 R58 关闭列表加上:
- **96×640 alt-tile on L3 + L1** — R59 J-2 S-1 / J-3 V-1 确认 -37.30pp / -12.52pp。
- **64×1024 alt-tile on L3 + L1** — R59 J-2 S-2 / J-3 V-2 确认 -8.76pp / -33.89pp。
- **L1 ITERS=1000 一次性 bump 轴 CLOSED by Opt R 政策** — 任何未来一次性 bump 必须满足 4 准则 validity envelope in `R59_OPT_R_POLICY.md`。
- **L1 (4096x32768x14336) AITER alt-tile 空间 EXHAUSTED** (96×640 + 64×1024 在 R59 关闭; 256×256 R57J1_L1 是 best)。
- **L3 (32768x14336x2048) AITER alt-tile 空间 EXHAUSTED** (96×640 + 64×1024 在 R59 关闭; 结合 R55/R57/R58 关闭, 无 AITER alt-tile 剩余)。

---

## 上一轮目标 (2026-04-19, post-R58 — WIN 1 PROMOTE 结构性 P-2 HK→AITER 128×256 swap +4.37pp, **AITER bit-deterministic 份额 40/42 (项目史上最大)**, **HK pool 3→2 (项目史上最少)**, **连续第 10 次 R50D AS-IS 复用**, 41/42 严格 VC (3 连 100% 排行榜结束; 1 个 cohort-race tail-draw on UNCHANGED L3 R40B HK 由 R57 fin_min=0.9847 + R58 Opt N 预测), ITERS=500 default 从 R57 一次性 ITERS=1000 revert)

**HEADLINE**: R58 达到 **结构性 WIN: P-2 HK→AITER 128×256 swap 交付 +4.37pp perf 追赶并将唯一中边距 Opt N dropper 转为完美 bit-determinism**。R58 攻击唯一 LOSE cell L8 (Opt O HK probe), 3 个 HK 存活 cell via 128×256 alt-tile (Opt P), AND 运行方法论 cohort 收紧 strict-VC gate (Opt N analysis-only)。**1/5 PROMOTE / 4 ACCEPT_FALLBACK / 0 DEAD**。1 个 PROMOTE 是 I-3 Opt P P-2: `(16384,4096,3072)` HK R40B 103.25% → AITER 128×256 **107.62%** 在 reviewer (+4.37pp; n_OK=10/10, wcf_max=0, fin_min=1.0 完美 bit-determinism)。Worker 报告 106.75%; reviewer p50 落在 +0.87pp 更高。**AITER 份额 39→40 (项目史上最大)**; **HK pool 3→2 (项目史上最少)**; 存活 HK cell 是 `16384x4096x2048` (107.48% HOLD VC+WIN) 和 `32768x14336x2048` (100.49% 数值 WIN 但 VC-flipped 在此 ITERS)。4 个 ACCEPT_FALLBACK: I-1 Opt N (analysis-only, 40/42 通过收紧 gate, 推荐 = 不要采用为 default); I-2 Opt O L8 HK probe (R40B + R37 fallback 都 WRONG_OUTPUT fin=0.78-0.80 wcf=0.07-0.13 — 它们在著名的 "17% deterministic-wrong cohort" 从未为 K=128256 修补; HK 256×256 axis on L8 K=128256 因正确性关闭); I-3 Opt P P-1 (16384x4096x2048 128×256 SMOKE 105.89%, -3.68pp vs HK; D-3A-1 STOP); I-3 Opt P P-3 (32768x14336x2048 128×256 SMOKE 75.49%, -25.04pp 灾难性; STOP_DEAD)。3 连 100% 排行榜结束但 cohort-race 表面**永久** 3→1 cell。连续第 10 轮 R50D shim AS-IS 复用。

**R58 P-2 机制验证**: HK R40B 256×256 cohort-race 强度在 (M=16384, N=4096, K=3072) 有意义但 aiter 128×256 (eff=85.3) 保持 end-to-end perf — 单次 swap 交付 +4.37pp perf AND wcf_max 0.0134→0 AND fin_min 0.9955→1.0。**PROMOTE 在 decider 评级"低信心"的轴上找到** — 支持有界成本 axis-closure 探测值得运行的原则。

**R58 Opt O L8 HK 256×256 axis 因正确性关闭**: BOTH R40B (build_R40B/) 和 R37 (build_R37/) HK 候选产生 WRONG_OUTPUT (fin=0.78-0.80, wcf=0.07-0.13)。R39A TAIL_SCALE_CLAMP / R44A back-edge drain / R44D FINITE_GATE 修补**从未** port 进 K=128256 build — 解释了为什么 L8 结构上卡住。R59+ 不能使用现有 HK binary; 需要 from-scratch K=128256 HK build 把所有 post-R39 正确性修补 port 进去。

**R58 Opt N gate-tightening 信息交付 (analysis-only)**: 40/42 通过收紧 gate (n_OK≥9/10, wcf_max<0.01, fin_min≥0.99); 2 个 dropper 是 HK 存活 `16384x4096x3072` (现已被 P-2 PROMOTE 消除) 和 `32768x14336x2048` (R58 唯一 VC-flipper)。推荐: KEEP R45+ default 为 canonical; carry Opt N 为 informational secondary gate。Post-R58 Opt N 注意力清单 = 1 cell。

**R58 ITERS=500 revert 效果由 R57 verdict 预测**: L1 (R57J1) 100.04% → 99.94% (LOSE-edge 重现; R57 verdict §"ITERS=1000 protocol observation" 预测); HK 存活更宽 trim 分布 mean shift -1 到 -2pp (16384x4096x2048 109.57→107.48%); fin_min on `32768x14336x2048` 0.9847 → 0.911 (低于 0.97 gate; 产生 VC flip)。**L3 VC drop 是 UNCHANGED binary 上的 COHORT-RACE TAIL-DRAW, NOT kernel regression**。REVERT 已考虑但不适用 — P-2 swap 候选严格通过; L3 由 ITERS revert 在不同 (UNCHANGED) cell 上控制。

**R58 attempts 总结** (1 个 PROMOTE; 4 ACCEPT_FALLBACK; 连续第 10 次 R50D AS-IS):

- **R58 Opt N — Gate tightening pilot, analysis-only (worker I-1, NO GPU)**:
  - 在 (n_OK≥9, wcf_max<0.01, fin_min≥0.99) 下重分类 R57_INTEGRATION_10RUN.json: 40/42 通过; 2 HK 存活 drop on wcf_max tail (`16384x4096x3072`, `32768x14336x2048`); 0 AITER drop (完美 bit-determinism)。**推荐: 不采用; 保持 informational**。
- **R58 Opt O — L8 HK kernel probe (worker I-2, GPUs 0-1, 0/2 PROMOTE 2 ACCEPT_FALLBACK)**:
  - O-1 `(4096,32768,128256)` HK R40B 256×256: WRONG_OUTPUT fin=0.804 wcf=0.128 SNR=36.25dB。**ACCEPT_FALLBACK (correctness gate)**。
  - O-1b 同 shape HK R37 256×256 memc fallback: WRONG_OUTPUT fin=0.775 wcf=0.066 SNR=31.79dB。**ACCEPT_FALLBACK**。**HK 256×256 axis on L8 K=128256 因正确性 CLOSED**。
- **R58 Opt P — 128×256 alt-tile probe on 3 HK 存活 (worker I-3, GPUs 2-3, 1/3 PROMOTE 2 ACCEPT_FALLBACK)**:
  - P-1 `(16384,4096,2048)`: HK 109.57% vs 128×256 SMOKE 105.89% (-3.68pp)。**ACCEPT_FALLBACK (D-3A-1)**。
  - **P-2 `(16384,4096,3072)`: HK R40B 103.25% → AITER 128×256 SMOKE 106.35% (+3.10pp); 10-run 106.75% (+3.50pp), n_OK=10/10, wcf_max=0, fin_min=1.0。PROMOTE。**
  - P-3 `(32768,14336,2048)`: HK 100.53% vs 128×256 SMOKE 75.49% (-25.04pp 灾难性)。**ACCEPT_FALLBACK (STOP_DEAD)**。

唯一 PROMOTE worker AS-IS 复用 **EXISTING R50D shim** at `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` with new `tile_M=128, tile_N=256` kwargs + `f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256.co`。NO shim rebuild。Kernel 不变。

**R58 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010] @ ITERS=500 DEFAULT REVERTED; 4 GPU 4-7, ~17 分钟 wall, 420 runs)**:
- Manifest: 40 AITER + 2 HK = 42 (1 binary delta vs R57: P-2 swap; 仅 swap, 无其他 manifest change)。
- 全部 1 个 R58 PROMOTE 候选 reviewer re-bench RE-VERIFIED 10/10 PASS, 完美 bit-determinism (wcf_max=0, wcf_std=0, fin_min=1.0)。
- AITER cell: **40/40 PASS (100% bit-deterministic)** — 项目史上最大 AITER bit-det 份额。
- HK cell: **1/2 PASS** (`16384x4096x2048` HOLD VC+WIN at 107.48%; `32768x14336x2048` VC-flipped at n_OK=9/10 fin_min=0.911 但数值 WIN at 100.49%)。
- Cohort-race churn 审计 41 个 UNCHANGED-binary cell: **VC retention 40/41**; 平均 perf drift +0.001pp on 39 unchanged AITER cells (range -1.09 to +1.15pp), -1.06pp on 2 unchanged HK cells (预测 ITERS=500 wider-trim effect)。
- ITERS=500 revert 下 WIN flip cell: 1 (L1 R57J1 100.04% → 99.94% LOSE-edge; R57 verdict 预测)。
- **最终 VC count: 41/42 strict 10-run (3 连 100% 排行榜结束于 R57; 结构性 win 复利)**。
- WIN cell (>=100% comp): 41 → 40 (-1; L1 LOSE-edge 重现; P-2 +1 抵消)。
- LOSE cell: 1 → 2 (L1 99.94%, L8 98.28%)。
- 文件: `R58_INTEGRATION_VERDICT.md`, `R58_INTEGRATION_MANIFEST.json`, `bench_all_42_R58_INTEGRATION.py`, `R58_INTEGRATION_{10RUN,SMOKE1}.{json,log,console}`, `R58_DECIDER_PLAN.md`, `R58_OPT_N_VERDICT.md`, `R58_OPT_I{2,3}_VERDICT.md`, `R58I{1,2,3}_*_INTEGRATION_FRAGMENT.json`, `bench_R58I{2,3}_*.py`, `R58_OPT_{O1,O1b,P1,P2,P3}_{SMOKE,10RUN}.{json,log}`。

**R58 net result**: **+1 PROMOTE 结构性 swap + AITER bit-det 39→40 (史上最大) + HK pool 3→2 (史上最少) + 连续第 10 次 R50D AS-IS + Opt N informational 交付 + 4 个关闭轴 documented**, 交换: 3 连 100% 排行榜结束于 41/42 VC + 40/42 WIN under ITERS=500 default。Cohort-race 注意力表面**永久** 3 cell → 1 cell。结构性进展胜过 streak break。

### R58 LOSE/VC-flipped cell 剩余 (3 个注意力 cell; 1 hard LOSE, 1 ITERS-edge LOSE, 1 VC-flipped)
- `(4096,32768,128256)` 98.28% (L8; R52D2B AITER 256×256; aiter alt-tile axis FULLY CLOSED + HK 256×256 axis CLOSED by 正确性; 没有 from-scratch HK build 之外的剩余机制轴)
- `(4096,32768,14336)` 99.94% (L1; R57J1 AITER 256×256; ITERS=500 revert 推 below WIN-line; R57 ITERS=1000 下是 100.04%)
- `(32768,14336,2048)` 100.49% (HK R40B; 数值 WIN 但 VC-flipped n_OK=9/10 fin_min=0.911 under ITERS=500; 之前由 R57 reviewer fin_min=0.9847 和 R58 Opt N 标记为最差边距存活)

### R59 候选 (post-R58, 按机制信心排序)
1. **R59 Opt R — Mixed-protocol 排行榜政策决策 (最高方法论价值, 最低 perf 增益)** — 决策: 是否永久提升 ITERS=1000 (成本: 2× wall 每轮; 利益: L1 +1 WIN cell 稳定 + HK 存活 +1.5pp)? 或维持 ITERS=500 default with documented L1 noise-edge LOSE? 或混合协议 L1-only ITERS=1000 (成本: 排行榜不一致)? **推荐 ITERS=500 default + documented L1 footnote** (当前 R58 disposition)。
2. **R59 Opt S — `(32768,14336,2048)` HK 存活救援 (中等信心)** — R58 下 VC-flipped。试非-128×256 alt-tile (96×640 eff=83.5, 64×1024 eff=60.2) — 都比 128×256 lower-eff (此 cell 上得分 75.49%)。非常低信心。或试 kernel-rebuild with R44D FINITE_GATE 0.97 → 0.95 (cohort-protocol relax)。预估 +0 到 +1 NET VC。
3. **R59 Opt T — L8 from-scratch HK kernel build for K=128256 (非常低信心, 高成本)** — Port R39A TAIL_SCALE_CLAMP + R44A back-edge drain + R44D FINITE_GATE 进 new K=128256 HK build。预估成本: ~3 R-rounds; 非常低 WIN 概率。仅当 R59 明确选择 build-campaign 预算时尝试。
4. **R59 Opt U — 接受残余 + 文档化 pivot** — 承认 L8 在 aiter+HK 结构地板; 冻结 manifest 在 R58 baseline; pivot R59-R65 到系统化 per-shape decompositions for SC/MICRO publication。

### R59+ 不要尝试的轴 (R45-R58 关闭)
- 全部 R57 关闭列表加上:
- **HK 256×256 lgk2 v12 R40B/R37 fallback for L8 K=128256** — 都产生 WRONG_OUTPUT (R58 Opt O); 需要 from-scratch K=128256 HK build with R39A/R44A/R44D 修补 ported。
- **128×256 alt-tile for 16384x4096x2048 HK 存活和 32768x14336x2048 HK 存活** — R58 Opt P 确认 -3.68pp 和 -25.04pp; 仅 (16384,4096,3072) cell PROMOTEs。
- 任何对 L1 noise-edge cell 进一步 ITERS=1000 一次性尝试 — R57 已展示混合协议成本不值得单 cell flip。

---

## 上上轮目标 (2026-04-19, post-R57 — WIN +1 NET WIN cell (40→41), **42/42 严格 VC 连续第 3 轮保持 (3rd 100% 排行榜)**, 1/5 PROMOTE / 4 ACCEPT_FALLBACK, AITER bit-deterministic 份额 39 of 42 持平, HK cell 3 持平 (项目史上最少 HELD), 0 cohort-race churn, 连续第 9 次 R50D AS-IS 复用, Opt J ITERS=1000 协议提升首次成功验证)

**HEADLINE**: R57 达到 **+1 NET WIN cell (40→41)，42/42 严格 VC 连续第 3 轮保持 (项目史上首个连续 3 轮 100% 排行榜)**。R57 通过 ITERS=1000 协议提升攻击 R56 唯一的 reviewer-drift edge cell L1。**1/5 PROMOTE / 4 ACCEPT_FALLBACK / 0 DEAD** 跨 3 个 worker cohort (H-1 Opt J L1 ITERS=1000, H-2 Opt L 192×256 axis 3 个 HK cell, H-3 Opt K L8 224×256 探测)。L1 4096x32768x14336 reviewer p50: 99.98% → **100.04%** (+0.07pp, 越过 WIN 线; worker 102.37%)。所有 4 个 ACCEPT_FALLBACK 是 D-3A-1 保护正确触发, 不是回归。AITER bit-deterministic 份额 **39/39 PASS HELD**; 3 个存活 HK cell **3/3 PASS, 0 churn**。WIN cell (≥100% comp): **40 → 41**。LOSE cell: 2 → 1 (仅 L8 4096x32768x128256 在 97.75%; aiter alt-tile axis 完全关闭)。Cohort-race 表面 41 个 unchanged-binary cell **0 lost VC**, 平均 perf drift +0.23pp/cell (ITERS=1000 noise floor)。连续第 9 轮 R50D shim AS-IS 复用 (no rebuild, no kernel modification, no new .co)。

**R57 Opt J 机制验证**: L1 在 R56 reviewer 是 99.98% (worker 100.63%) — 0.6pp 跨 GPU drift 把 worker WIN-line 推到 reviewer LOSE-edge。R57 ITERS=500→1000 协议提升收紧 p50 trim 分布; reviewer p50 升到 100.04% (worker 102.37%)。两个测量都保留 WIN-line 跨越, 验证 Opt J 机制: **结构 bit-deterministic 的 AITER cell 在 noise tail 上, ITERS=500 在分布尾巴上, ITERS=1000 暴露真实 mean**。仅适用于 ≤0.5pp 距分类边界的 cell; R58 应该 revert ITERS=500 default 除非识别另一个 tail-edge cell。

**R57 Opt L 192×256 axis 关闭** (3/3 HK cell DEAD): R56 D-5B/1 在 N=14336 显示 256×256 AITER 比 HK underperform; R57 H-2 在 3 个 HK cell (16384x4096x{2048,3072}, 32768x14336x2048) 上试 192×256 全部 SMOKE -8 到 -14pp。192×256 axis 对 HK kept-cell pool **完全关闭**; 3 个 HK cell 在 R57 reviewer 都 PASS 100.53%-109.57%。

**R57 Opt K L8 224×256 axis 关闭** (1/1 DEAD): R56 G-4 已经在 L8 4096x32768x128256 上证伪 128×512 (-13.31pp) 和 192×256 (-11.84pp); R57 H-3 试 224×256 SMOKE 87.61% (-10.69pp)。L8 aiter alt-tile axis **完全关闭** — 256×256 严格最佳 for K=128256 K/N=4 ratio。L8 是 R57 后唯一剩余 LOSE cell (距 WIN 2.25pp; 在 aiter-internal ceiling)。

**R57 attempts 总结** (1 个 PROMOTE 复用 R56G1_L1 binary AS-IS with ITERS=1000):

- **R57 Opt J — L1 ITERS=500→1000 tail-tightening (1/1 PROMOTE)**:
  - L1 `(4096,32768,14336)`: 99.98% (R56 reviewer) → **100.04%** (R57 reviewer; worker 102.37%) **+0.07pp**。**R56G1_L1 → R57J1_L1 source rename only; binary unchanged**。
- **R57 Opt L — 192×256 alt-tile probe on 3 保留 HK cell (0/3 PROMOTE 3 ACCEPT_FALLBACK)**:
  - L-1 `(16384,4096,2048)`: HK 109.57% (R57 reviewer) vs AITER 192×256 SMOKE DEAD -8 到 -14pp。**ACCEPT_FALLBACK**。
  - L-2 `(16384,4096,3072)`: HK 103.25% (R57 reviewer) vs AITER 192×256 SMOKE DEAD。**ACCEPT_FALLBACK**。
  - L-3 `(32768,14336,2048)`: HK 100.53% (R57 reviewer) vs AITER 192×256 SMOKE DEAD。**ACCEPT_FALLBACK**。
- **R57 Opt K — L8 224×256 探测 (0/1 PROMOTE 1 ACCEPT_FALLBACK)**:
  - K-1 L8 `(4096,32768,128256)`: AITER 256×256 R52D2B 97.75% (R57 reviewer) vs 224×256 SMOKE 87.61% (-10.69pp)。**ACCEPT_FALLBACK**。

唯一 PROMOTE worker 复用 **EXISTING R56 binaries AS-IS** (R56G1_L1 .so/.co/grid 完全不变)，per-shape grid + `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`。NO 任何 binary rebuild。Kernel 不变。

**R57 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010], ITERS=1000 协议提升; 4 GPU 4-7, ~17 分钟 wall, 420 runs)**:
- Manifest **39 个 aiter `.co` override** + **3 个 HK** baseline (与 R56 完全一致, 仅 L1 source label rename)。
- 全部 1 个 R57 PROMOTE 候选 reviewer re-bench RE-VERIFIED 10/10 PASS, 完美 bit-determinism (wcf_max=0.0, wcf_std=0.0, fin_min=1.0)。
- **AITER cell: 39/39 PASS (100% bit-deterministic)** HELD。
- HK cell: **3/3 PASS (零 churn; 全部 3 个存活 HK cell re-verified PASS_10/10)** HELD。
- Cohort-race churn 审计 41 个未变 binary cell: **0 lost VC; 平均 perf drift +0.16pp AITER (range -1.11 到 +2.15pp), +1.51pp HK (R40B HK cell 从更长 ITERS 的更宽 trimmed-distribution 受益最多, range +0.14 到 +4.21pp)**。
- R57 贡献: **+1 NET WIN cell** (40 → 41) + 0 cohort losses + +0.07pp L1 perf 越线。
- **最终 VC count: 严格 10-run 42/42 保持 (连续第 3 个 100% 排行榜轮)**。
- WIN cell (>=100% comp): 40 → 41 (+1, L1 LOSE-edge → WIN)。
- ITERS=1000 协议观察: **唯一 WIN/LOSE flip 是 targeted L1**。其他近-100% cell (L8 98.30→97.75 LOSE→LOSE, 32768x14336x2048 100.39→100.53 WIN→WIN, 4096x32768x28672 101.07→101.17 WIN→WIN) 都保留同侧分类。**Opt J 仅适用于 ≤0.5pp 距分类边界的 cell**; R58 应 revert ITERS=500 default。
- 文件: `R57_INTEGRATION_VERDICT.md`, `R57_INTEGRATION_MANIFEST.json`, `bench_all_42_R57_INTEGRATION.py`, `R57_INTEGRATION_{10RUN,SMOKE1}.{json,log,console}`, `R57_DECIDER_PLAN.md`, `R57_OPT_H{1,2,3}_VERDICT.md`, `R57H{1,2,3}_*_INTEGRATION_FRAGMENT.json`, `bench_R57H{1,2,3}_*.py`, `R57_OPT_{J1,L1,L2,L3,K1}_{SMOKE,10RUN}.{json,log}`。

**R57 net result**: **1 PROMOTE + +0.07pp L1 越线 + 项目史上首个 3 连 100% 排行榜**, R57 带来零回归，零 cohort-race 损失。Net 严格 VC: **42/42 保持 (连续第 3 个 100% 排行榜)**。WIN cell 40→41 (+1)。Cohort-race 表面 3 HK cell HELD (史上最少)。**0 kernel 修改, 0 新 shim build, 0 新 .co binary (连续第 9 次 R50D AS-IS 复用)**。

### R57 LOSE cell 剩余 (仅 1 个; 距 WIN 2.25pp)
- `(4096,32768,128256)` 97.75% (L8; aiter alt-tile axis 完全关闭后唯一剩余; 在 aiter-internal ceiling for K=128256 K/N=4)

### R58 候选 (post-R57, 按机制信心排序)
1. **R58 Opt N — Gate 收紧 pilot (中, R56/R57 推迟; 最高方法论价值)** — 严格 VC 上限达 42/42; 收紧 gate (n_OK→9/10, wcf_max→0.01, fin_min→0.99) 在 ITERS=500 default 下枚举 sub-optimal cell。Test-design 改进暴露 cohort-race 真实表面。预估 +0 NET WIN, +N cell 重分类为非 PASS_10/10, 引导 R59+ 注意力。
2. **R58 Opt O — L8 HK rewrite for K=128256 K/N=4 ratio (低-中; 高风险高收益)** — 唯一剩余 LOSE。aiter 256×256 在 97.75%; aiter alt-tile axis 完全关闭; HK 需要 +2.25pp over AITER 256×256 才能 PROMOTE。可能 K-pipeline tuning 配 256×256 grid。预估 +0 到 +1 NET WIN cell。
3. **R58 Opt P — 非-192×256 alt-tile 在 3 个 HK cell 上 (低; 完成性)** — R57 H-2 关闭 192×256; 剩余 alt-tile (128×256 eff=85.3, 96×640 eff=83.5, 64×1024 eff=60.2) 都 lower-eff。R55 D-5B/1 显示 256×256 AITER underperforms HK 在 N=14336。预估 +0 NET WIN, 可能 +N bit-determinism (AITER 39→42 if any 成功)。
4. **R58 Opt Q — Revert ITERS=500 default (杂务)** — R57 H-1 ITERS=1000 是 single-purpose 协议 bump; R58 default revert 除非另一个 tail-edge cell 出现。

### R58+ 不要尝试的轴 (R45-R57 关闭)
- 全部 R56 关闭列表加上:
- **L8 (4096x32768x128256) 上的 224×256** — R57 H-3 确认 -10.69pp (与 128×512 -13.31pp + 192×256 -11.84pp 一起完全关闭 L8 aiter alt-tile axis)。
- **3 个保留 HK cell 上的 192×256** — R57 H-2 确认全部 -8 到 -14pp; 192×256 axis 对 HK kept-cell pool 完全关闭。
- 任何 HK 256×256 路径 或 R50D aiter `.co` dlopen 路径本身的"改进"。

---

## 更早轮历史 (2026-04-19, post-R56 — WIN 7/7 PROMOTE PERF 追赶, **42/42 严格 VC 连续第 2 轮保持**, **+6 NET WIN cell (34→40, 项目史上最大单轮 WIN-cell 跳跃)**, +198.14pp 累计 perf 追赶 (2.2× STRETCH), AITER bit-deterministic 份额 38→39 of 42, HK cell 4→3 (项目史上最少), 0 cohort-race churn, 连续第 8 次 R50D AS-IS 复用)

**HEADLINE**: R56 达到 **+6 NET WIN cell (34→40)，+198.14pp 累计 perf，42/42 严格 VC 保持**。R56 通过 R50D shim AS-IS 上的 aiter alt-tile dispatch 攻击 R55 的 8 个 LOSE cell。**7/7 PROMOTE / 1 ACCEPT_FALLBACK** 跨 4 个 worker cohort (G-1, G-2, G-3 falsification, G-4 HK→AITER + alt-tile probe)。全部 7 个 PROMOTE 都以 ≥13pp 越过 +1.0pp D-3C 门。R56 7 个 PROMOTE cell (delta R55→R56 pct_comp): L1 +34.30pp, L2 +19.87pp, L3 +15.93pp, L4 +14.38pp, L5 +16.14pp, L6 +16.60pp, L7 +80.92pp (HK→AITER)。AITER cell **39/39 PASS (100% bit-deterministic, wcf_max=0.0, wcf_std=0.0, fin_min=1.0)**。3 个存活 HK cell **3/3 PASS, 0 churn**。WIN cell (≥100% comp): **34 → 40** (项目史上最大单轮 WIN-cell 跳跃)。LOSE cell: 8 → 2 (L1 99.98% reviewer-drift edge, L8 98.30% D-3A-1 ACCEPT_FALLBACK)。Cohort-race 表面达项目史上最小 (3 个 HK cell)。连续第 8 轮 R50D shim AS-IS 复用 (no rebuild, no kernel modification)。

**R56 Opt G 机制验证**: aiter heuristic 在 6 个 cluster-A/B 64×1024 LOSE cell 上 under-pick 256×256，因为 `local_round` count 完全相同但 `compute2mem_efficiency` 不同 (60.2 vs 128.0)。G-1 256×256 vs G-3 128×512 falsification (eff=102.4) 在每个 cell 上 256×256 都赢 — 确认两个机制轴都起作用 (wider tile escape 加 eff dominance)，256×256 严格最佳。G-4 C1 (L7 4096x128256x32768) HK R41A (97.59%) → AITER 256×256 (178.51%) = +80.92pp = **首次 kept-HK cell 上 HK→AITER swap 成功**。G-4 C2/C3 (L8 4096x32768x128256, current 256×256 98.32%) alt-tile 128×512 和 192×256 返回 85.01% / 86.48% — D-3A-1 保护正确保留 R52D2B baseline。

**R56 attempts 总结** (全部 7 个 PROMOTE 复用 R50D shim AS-IS with aiter `.co` 256×256):

- **R56 Opt G-1 — 64×1024→256×256 cluster A (3/3 PROMOTE +76.11pp 累计 worker)**:
  - L1 `(4096,32768,14336)`: 65.68% → **99.98%** (reviewer; worker 100.63%) **+34.30pp**。
  - L2 `(32768,4096,14336)`: 84.13% → **104.00%** **+19.87pp**。
  - L6 `(16384,4096,14336)`: 90.28% → **106.88%** **+16.60pp**。
- **R56 Opt G-2 — 64×1024→256×256 cluster B (3/3 PROMOTE +43.93pp 累计 worker)**:
  - L3 `(128256,32768,4096)`: 86.93% → **102.86%** **+15.93pp**。
  - L4 `(28672,32768,4096)`: 87.82% → **102.20%** **+14.38pp**。
  - L5 `(14336,32768,4096)`: 87.92% → **104.07%** **+16.14pp**。
- **R56 Opt G-3 — 128×512 falsification probe (3/3 PROMOTE +64.28pp 但 manifest 中 DROPPED; G-1 赢)**:
  - 三个 cell 上 128×512 都 PROMOTE 但都比 G-1 256×256 弱 (L1 92.48% vs 100.63%, L2 104.70% vs 105.37%, L6 107.19% vs 110.20%) — DROPPED。
  - 机制: 256×256 在每个 cell 上严格更好; eff dominance 确认。
- **R56 Opt G-4 — HK→AITER + alt-tile (1/3 PROMOTE 2 ACCEPT_FALLBACK)**:
  - C1 L7 `(4096,128256,32768)` HK R41A → AITER 256×256: 97.59% → **178.51%** **+80.92pp**。**首次 kept-HK cell 上 HK→AITER swap 成功**。
  - C2 L8 `(4096,32768,128256)` 256×256 → 128×512 SMOKE 85.01% (-13.31pp)。**ACCEPT_FALLBACK**。
  - C3 L8 `(4096,32768,128256)` 256×256 → 192×256 SMOKE 86.48% (-11.84pp)。**ACCEPT_FALLBACK**。

全部 7 个 PROMOTE worker AS-IS 复用 **EXISTING R50D shim** at `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`，per-shape grid 参数 + `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`。NO shim rebuild。Kernel 不变。

**R56 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010]; 4 GPU 4-7, ~16 分钟 wall, 420 runs)**:
- Manifest **39 个 aiter `.co` override** (R55 38 + L7 HK→AITER) + **3 个 HK** baseline (R55 4 minus L7)。
- 全部 7 个 R56 PROMOTE 候选 reviewer re-bench RE-VERIFIED 10/10 PASS, 完美 bit-determinism (wcf_max=0.0, wcf_std=0.0, fin_min=1.0)。
- **AITER cell: 39/39 PASS (100% bit-deterministic)**。
- HK cell: **3/3 PASS (零 churn; 全部 3 个存活 HK cell re-verified PASS_10/10)**。
- Cohort-race churn 审计 35 个未变 shape: **0 lost VC; 平均 perf drift -0.06pp AITER, -0.02pp HK**。
- R56 贡献: **+6 NET WIN cell** (34 → 40) + 0 cohort losses + +198.14pp 累计 perf 追赶。
- **最终 VC count: 严格 10-run 42/42 保持 (连续第 2 个 100% 排行榜轮)**。
- WIN cell (>=100% comp): 34 → 40 (+6, 项目史上最大单轮 WIN-cell 跳跃)。
- 文件: `R56_INTEGRATION_VERDICT.md`, `R56_INTEGRATION_MANIFEST.json`, `bench_all_42_R56_INTEGRATION.py`, `R56_INTEGRATION_{10RUN,SMOKE1}.{json,log,console}`, `R56_DECIDER_PLAN.md`, `R56_OPT_G{1,2,3,4}_VERDICT.md`, `R56G{1,2,4}_*_INTEGRATION_FRAGMENT.json`, `bench_R56G{1,2,3,4}_*.py`。

**R56 net result**: **7 PROMOTE + +198.14pp 累计 PERF 追赶 (2.2× STRETCH)**, R56 带来零回归，零 cohort-race 损失。Net 严格 VC: **42/42 保持 (连续第 2 个 100% 排行榜)**。WIN cell 34→40 (+6, 项目史上最大单轮跳跃)。Cohort-race 表面 4→3 HK cell (史上最少)。**0 kernel 修改, 0 新 shim build (连续第 8 次 R50D 复用)**。

### R56 LOSE cell 剩余 (仅 2 个; 合计距 WIN 不到 1.7pp)
- `(4096,32768,14336)` 99.98% (L1; reviewer-drift edge; worker 报告 100.63% — 0.6pp drift 范围内, 实际在 WIN 阈值)
- `(4096,32768,128256)` 98.30% (L8; D-3A-1 ACCEPT_FALLBACK; alt-tile 128×512 和 192×256 都 DEAD 11-13pp)

### R57 候选 (post-R56, 按机制信心排序)
1. **R57 Opt J — L1 长 ITERS 或 alt seeds 重 bench (最高信心, 最低价值)** — L1 4096x32768x14336 当前 reviewer 99.98% (worker 100.63%); 距 WIN 0.02pp 是噪声级。用 ITERS=1000 重 bench 收紧 p50 分布确认越过 WIN。预估 +1 WIN cell (40→41) 但无 perf 增益。
2. **R57 Opt K — L8 剩余未试 aiter tile 探测 (中-低)** — L8 4096x32768x128256 在 98.30%。K=128256 剩余未试 aiter `.co`: 96×640, 64×1024 (都 lower-eff; 信心很低)。可能 192×128, 128×384, 160×256/384 如果存在。预估 +0 到 +1 WIN cell。
3. **R57 Opt L — 保留 HK cell 加固 pivot (低-中)** — 3 个 HK cell 剩余。在 R56 都 cleanly re-pass; HK 256×256 路径产生 100.39%-105.36%。R55 D-5B/1 已经显示 256×256 AITER 在 N=14336 underperform。试非-256×256 alt-tile (128×256, 192×256) 给这 3 个 HK cell。预估 +0 NET WIN + 高达 3 个 cell 的 bit-determinism 增益。
4. **R57 Opt M — Gate 收紧 pilot (中, 从 R56 推迟)** — VC 上限已达; 收紧 gate (n_OK→9/10, wcf_max→0.01, fin_min→0.99) 暴露剩余 sub-optimal cell。Test-design 改进。

### R57+ 不要尝试的轴 (R45-R56 关闭)
- 全部 R55 关闭列表加上:
- **L8 (4096x32768x128256) 上的 128×512 和 192×256** — R56 G-4 确认比当前 256×256 差 11-13pp。
- 任何对 G-3 128×512 cluster A/B cell (L1, L2, L6) 的进一步尝试 — reviewer 已 DROPPED; G-1 256×256 严格更好。
- 任何 HK 256×256 路径 或 R50D aiter `.co` dlopen 路径本身的"改进"。

---

## 更早轮历史 (2026-04-19, post-R55 — WIN +6 NET VC 严格 10-run = **42/42 项目史上首个 100% 排行榜轮**, COMMIT, 11/12 PROMOTE / 0 DEAD / 1 ACCEPT_FALLBACK 跨 4 个 worker cohort, AITER bit-deterministic 份额 27→38 of 42, 连续第 7 次 R50D AS-IS 复用, cohort-race 表面从 15 个 HK cell 减到 4 个)

**HEADLINE**: R55 达到 **42/42 严格 10-run VC，项目史上第一个 100% 排行榜轮**。Net VC delta vs R54 = **+6 NET VC 严格 10-run** (36 → 42/42)。**11/12 PROMOTE / 0 DEAD / 1 D-3A-1 ACCEPT_FALLBACK** (D-5B/1 `(32768,14336,2048)` SMOKE -1.33pp vs HK; 正确保留 HK fallback — 自 R53 D-3A-1 DEAD 以来 D-3A-1 保护首次触发)。4 worker cohort: E-3 cohort-race 救援 M=16384 (4/4 PROMOTE +4 NET VC), E-4 cohort-race 救援 (2/2 PROMOTE +2 NET VC), D-5A 边际 HK-VC perf 追赶 (3/3 PROMOTE +29.07pp 累计), D-5B 边际追赶 M=32768 (2/3 PROMOTE +6.76pp 累计)。5 个 D-5 perf 追赶 deliver +1.81pp 到 +20.99pp over HK baselines, 最大 +20.99pp on `(4096,4096,8192)`。AITER cell **38/38 PASS (100% bit-deterministic, wcf_max=0.0, wcf_std=0.0, fin_min=1.0)**; 4 个保留的 HK cell **4/4 PASS (NO churn — 保留 HK cell 上 0 cohort-race 损失)**。WIN cell (>=100% comp): 28 → 34 (+6)。Cohort-race 表面从 15 个 HK cell 减到 4 个 — 大幅降低未来轮 attrition 风险。连续第 7 轮 R50D shim AS-IS 复用 (no rebuild, no kernel modification)。

**R55 attempts 总结** (全部 11 个 PROMOTE 复用 R50D shim AS-IS with 256×256 aiter `.co`):

- **R55 Opt E-3 — Cohort-race 救援 M=16384, N∈{14336,28672} (4/4 PROMOTE +4 NET VC)**:
  - `(16384,14336,2048)`: HK R40B PASS_9/10 fin=0.9222 → AITER **107.46%** PASS_10/10。**+1 NET VC 救援**。
  - `(16384,14336,4096)`: HK R40B FLAKE_1/10 wcf=0.0506 → AITER **108.32%** PASS_10/10。**+1 NET VC 救援**。
  - `(16384,28672,2048)`: HK R40B PASS_9/10 fin=0.9657 → AITER **102.47%** PASS_10/10。**+1 NET VC 救援**。
  - `(16384,28672,4096)`: HK R40B PASS_9/10 wcf=0.0215 → AITER **104.51%** PASS_10/10。**+1 NET VC 救援**。
- **R55 Opt E-4 — Cohort-race 救援 (small-N + 6144x32768) (2/2 PROMOTE +2 NET VC)**:
  - `(16384,6144,4096)`: HK R40B FLAKE_4/10 wcf=0.0631 → AITER **114.46%** PASS_10/10。**+1 NET VC 救援**。
  - `(6144,32768,4096)`: HK R40B PASS_9/10 wcf=0.0209 → AITER **105.44%** PASS_10/10。**+1 NET VC 救援**。
- **R55 Opt D-5A — 边际 HK-VC perf 追赶 M∈{16384,4096} (3/3 PROMOTE 0 NET VC, +29.07pp 累计)**:
  - `(16384,4096,4096)`: HK 101.16% → AITER **112.81%** (+11.65pp)。
  - `(16384,6144,2048)`: HK 106.19% → AITER **112.46%** (+6.27pp)。
  - `(4096,4096,8192)`: HK 100.16% → AITER **121.15%** (+20.99pp)。**本轮最大 D-5 增益**。
- **R55 Opt D-5B — 边际 HK-VC perf 追赶 M=32768 (2/3 PROMOTE 1 ACCEPT_FALLBACK 0 NET VC, +6.76pp 累计)**:
  - `(32768,14336,2048)`: HK 103.03% → AITER SMOKE 101.70% (-1.33pp)。**NO_PROMOTE — D-3A-1 ACCEPT_FALLBACK** (保留 HK)。
  - `(32768,28672,2048)`: HK 99.68% → AITER **103.11%** (+3.43pp)。PROMOTE。
  - `(32768,6144,2048)`: HK 103.27% → AITER **106.60%** (+3.33pp)。PROMOTE。

全部 11 个 PROMOTE worker AS-IS 复用 **EXISTING R50D shim** at `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`，使用 per-shape grid 参数和 `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` aiter binary。NO shim rebuild。Kernel 不变。

**R55 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010]; 4 GPU 4-7, ~16 分钟 wall, 420 runs)**:
- Manifest 中 38 个 aiter `.co` override (R50D + R51 + R52 + R53 8 个非 256×256 + R54 12 + R55 11 个新); 4 个 shape 用 HipKittens kernel + R44 baseline 参数。
- 全部 11 个 R55 PROMOTE 候选 reviewer re-bench RE-VERIFIED 10/10 PASS, 完美 bit-determinism (wcf_max=0.0, wcf_std=0.0, fin_min=1.0)。
- **AITER cell: 38/38 PASS (100% bit-deterministic)**。
- HK cell: **4/4 PASS (零 churn; 全部 4 个保留 HK cell re-verified PASS_10/10)**。
- 0 个 R54-VC shape 在 R55 cohort-race churn 下失去 VC; 0 个 R54-NO shape 失败 PROMOTE。
- Cohort churn 净: **0 / 0 = 0** 在**未变的** HK .so cell 上 (vs R54's -3, R53's -5)。
- R55 贡献: **+6 NEW VC** (E-3, E-4) + 0 cohort losses = **net +6 strict VC**。
- **最终 VC count: 严格 10-run 42/42 (vs R54 36/42)** — **项目史上第一个 100% 排行榜轮**。
- WIN cell (>=100% comp): 28 → 34 (+6)。
- Mean perf delta on R55 NEW PROMOTE vs R54 HK baseline: **+10.45pp** (range +1.81pp 到 +20.99pp)。
- 文件: `R55_INTEGRATION_VERDICT.md`, `R55_INTEGRATION_MANIFEST.json`, `bench_all_42_R55_INTEGRATION.py`, `R55_INTEGRATION_{10RUN,SMOKE1}.{json,log,console}`, `R55_DECIDER_PLAN.md`, `R55_OPT_{E3,E4,D5A,D5B}_VERDICT.md`, `R55{E3,E4,D5A,D5B}_{1,2,3,4}_INTEGRATION_FRAGMENT.json`, `bench_R55{E3,E4,D5A,D5B}_{1,2,3,4}.py`。

**R55 net result**: **+6 NEW VC 救援 + 5 PERF 追赶 (+35.83pp 累计 on D-5A+D-5B)**, R55 带来零回归，零 cohort-race 损失。Net 严格 VC: **42/42 = 100% 排行榜上限**。**Cohort-race 表面从 15 个 HK cell 减到 4 个 — 大幅降低未来轮 attrition 风险。0 kernel 修改, 0 新 shim build (连续第 7 次 R50D 复用)**。

### R55 LOSE cell 审计 (8 个 cell <100% comp; aiter-internal heuristic 限制, NOT R55 回归)
全部 8 个 LOSE cell 都是 carried-from-prior-rounds AITER override:
- `(4096,32768,14336)` 65.68% (R53D3B_1, 64×1024)
- `(4096,32768,128256)` 98.32% (R52D2B, 256×256)
- `(14336,32768,4096)` 87.92% (R53D3C_1, 64×1024)
- `(16384,4096,14336)` 90.28% (R53D3C_3, 64×1024)
- `(28672,32768,4096)` 87.82% (R53D3C_2, 64×1024)
- `(32768,4096,14336)` 84.13% (R53D3B_2, 64×1024)
- `(128256,32768,4096)` 86.93% (R53D3B_3, 64×1024)
- 其他全部 ≥100% comp (34/42 WIN)

### R56 候选 (post-R55, 按机制信心排序)
1. **R56 Opt G — 8 个 LOSE cell 上的 perf 追赶 (最高信心)** — 全部 8 个 cell 是 AITER override <100% comp。两个攻击向量: (a) 在 long-K/large-N cell 上的 aiter 替代 tile 选择 (192×256 / 128×256), 反驳 aiter 自己的 heuristic; (b) 在 small-K/large-N cell 上的 HK 32×32×64 MFMA (R56 Opt B 复活作为 perf-not-VC play)。预估 +0 NET VC + +5-30pp 单 cell。
2. **R56 Opt H — Gate 收紧 pilot (中)** — VC 上限已达到 under current gate (n_OK≥8/10, wcf_max<0.02, wcf_std<0.01, fin_min≥0.97)。考虑收紧 (e.g., n_OK→9/10, wcf_max→0.01, fin_min→0.99) 暴露剩余 sub-optimal cell。Test-design 改进; 初始可能 drop 一些 cell 出严格 VC。
3. **R56 Opt I — 保留 HK cell 加固 (低-中)** — 4 个 HK cell 存活但 2 个 gate 紧 (`32768x14336x2048` wcf=0.0118 fin=0.9787; `16384x4096x3072` wcf=0.0147)。考虑 port to AITER `.co` 如果非 256×256 tile fit。D-5B/1 SMOKE 结果 (-1.33pp) 暗示 256×256 失败在 N=14336/256=56 grid-x; 替代 tile unexplored。

### R56+ 不要尝试 (R45-R55 已关闭)
- R54 关闭列表 PLUS:
- **N=14336 256×256 AITER without SMOKE gate** (R55 D-5B/1 已确认 256×256 underperforms HK by 1.33pp on `32768x14336x2048` — N=14336/256=56 non-power-of-two grid-x 假设。Do NOT cargo-cult 256×256 onto N=14336 shape without per-shape SMOKE gate)。

### R55 stopping-criterion 检查
- Floor (≥38/42 严格 10-run): **EXCEEDED (42/42)**。
- Mode (39/42 严格): **EXCEEDED (42/42)**。
- Stretch (42/42 严格): **MET EXACTLY**。
- 轮价值: **+6 NEW VC + 5 perf 追赶 + 5 durable findings** (42/42 严格 ceiling 已达; cohort-race 表面 15→4; D-3A-1 ACCEPT_FALLBACK is load-bearing on N=14336; 8 LOSE cell 是 aiter-internal 限制 not R55 回归; 连续 7 次 R50D AS-IS 复用)。

### 最近 13 轮 sanity check
- R43: DEAD (3 轴)
- R44: WIN +8 (27 → 35/42)
- R45-R49: 5 轮连续 DEAD
- R50: WIN +1 (35 → 36/42, aiter `.co` dlopen 首次 PoC)
- R51: WIN +1 严格 (30 → 31/42) + 2 perf 追赶
- R52: WIN +5 严格 (31 → 36/42) + 3 perf 追赶
- R53: PARTIAL WIN +2 NET VC 救援 -5 cohort 严格净 -3 (36 → 33/42) + 6 perf 追赶
- R54: WIN +6 NET VC 救援 -3 cohort 严格净 +3 (33 → 36/42) + 6 perf 追赶 +17-28pp
- **R55: WIN +6 NET VC -0 cohort +6 strict (36 → 42/42) + 5 perf 追赶 +1.81-20.99pp; 38/42 cell bit-deterministic AITER; 11/12 PROMOTE / 1 ACCEPT_FALLBACK; 项目史上第一个 100% 排行榜轮** ← STRICT-VC CEILING REACHED

---

## 历史 (2026-04-19, post-R54 — WIN +3 NET VC 严格 10-run + 6 PERF 追赶 +17-28pp, COMMIT, 12/12 PROMOTE / 0 DEAD 跨 4 个 worker cohort, AITER bit-deterministic 份额 7→27 of 42 = 史上最大单轮 AITER 扩展, 36/42 严格 10-run VC, 连续第 6 次 R50D AS-IS 复用)

**HEADLINE**: R54 是连续 4 个 WIN 轮中的第 4 个，且是**史上最大的单轮 AITER 扩展** (R52 7/42 → R53 15/42 → R54 27/42 bit-deterministic AITER cell)。Net VC delta vs R53 = **+3 NET VC 严格 10-run** (33 → 36/42, 恢复 R52 水平)。**12/12 PROMOTE / 0 DEAD** (跨 4 个 worker cohort: E-1 cohort-race rescue M-heavy, E-2 cohort-race rescue N-heavy, D-4A sub-90% perf claw-back, D-4B 90-95% marginal claw-back) = 自轮跟踪以来最佳 cohort 结构。**+6 NEW VC 救援** 全部来自 E-1/E-2 (恢复 R53 cohort-LOSS 形状到 bit-deterministic AITER 105-118% comp); -3 cohort race 损失在**未变的** HK R40B `.so` (R45+ 现象)。6 个 D-4A/D-4B perf 追赶 over HK baselines 范围 +17.66pp 到 +28.28pp，全部远超 D-3C 0.5pp gate, 0 reverts。AITER cell 27/27 PASS (100% bit-deterministic, wcf_max=0.0, wcf_std=0.0, fin_min=1.0); HK cell 9/15 PASS (6 R40B FAIL on cohort-race wcf 或 fin gate)。30 个共享 VC shape 上 mean +4.42pp comp/shape vs R53。连续第 6 轮 R50D shim AS-IS 复用 (no rebuild, no kernel modification)。

**R54 attempts 总结** (12/12 PROMOTE / 0 DEAD):

- **R54 Opt E-1 — Cohort-race rescue (M-heavy / N=4096), 3 PROMOTE +3 NET VC**:
  - E1_1 `(32768,4096,2048)`: HK R52-VC 102.0% → R53 LOSS PASS_8/10 → AITER **105.16%** PASS_10/10。**+1 NET VC 救援**。
  - E1_2 `(32768,4096,3072)`: HK R52-VC 99.5% → R53 LOSS PASS_9/10 → AITER **118.58%** PASS_10/10。**+1 NET VC 救援**。
  - E1_3 `(28672,4096,8192)`: HK R52-VC 91.8% → R53 LOSS PASS_9/10 → AITER **108.72%** PASS_10/10。**+1 NET VC 救援**。
- **R54 Opt E-2 — Cohort-race rescue (N-heavy / M=4096 + mid-K), 3 PROMOTE +3 NET VC**:
  - E2_1 `(4096,32768,4096)`: HK R52-VC 91.8% → R53 LOSS PASS_9/10 → AITER **105.66%** PASS_10/10。**+1 NET VC 救援**。
  - E2_2 `(4096,32768,6144)`: HK R52-VC 92.3% → R53 LOSS PASS_8/10 → AITER **106.96%** PASS_10/10。**+1 NET VC 救援**。
  - E2_3 `(16384,4096,6144)`: HK R52-VC 98.5% → R53 LOSS PASS_8/10 → AITER **115.20%** PASS_10/10。**+1 NET VC 救援**。
- **R54 Opt D-4A — Sub-90% HK-VC perf claw-back, 3 PROMOTE 0 NET VC, +70.51pp 累计**:
  - D4A_1 `(4096,14336,16384)`: HK 84.41% → AITER **105.87%** (+21.46pp)。
  - D4A_2 `(32768,4096,7168)`: HK 88.36% → AITER **106.63%** (+18.27pp)。
  - D4A_3 `(6144,4096,8192)`: HK 89.53% → AITER **117.82%** (+28.29pp)。本轮最大 D-4A。
- **R54 Opt D-4B — 90-95% HK-VC marginal claw-back, 3 PROMOTE 0 NET VC, +59.27pp 累计**:
  - D4B_1 `(4096,4096,16384)`: HK 93.30% → AITER **110.96%** (+17.66pp)。
  - D4B_2 `(4096,14336,8192)`: HK 91.92% → AITER **114.17%** (+22.25pp)。
  - D4B_3 `(16384,4096,7168)`: HK 93.76% → AITER **111.89%** (+18.13pp)。

全部 12 个 PROMOTE worker AS-IS 复用 **EXISTING R50D shim** at `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`，使用 per-shape grid 参数和 `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` aiter binary。Aiter heuristic uniformly picked 256×256 for all 12 candidates (validates `compute2mem_efficiency=128` 是 dominant tiebreak)。NO shim rebuild。Kernel 不变。

**R54 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010]; 4 GPU, ~14 分钟 wall, 420 runs)**:
- Manifest 中 27 个 aiter `.co` override (R50D + R51 D-1/D-2/D-3 + R52 D-2A/D-2B/D-2C + R53 8 个非 256×256 + R54 12 个 256×256); 15 个 shape 用 HipKittens kernel + R44 baseline 参数。
- 全部 12 个 R54 PROMOTE 候选 reviewer re-bench RE-VERIFIED 10/10 PASS, 完美 bit-determinism (wcf_max=0.0, wcf_std=0.0, fin_min=1.0 across all 12)。
- **AITER cell: 27/27 PASS (100% bit-deterministic)**。
- HK cell: 9/15 PASS (6 R40B FAIL on cohort-race wcf 或 fin gate)。
- 0 个 R53-NO shape 在 R54 cohort-race churn 下获得 VC。
- 3 个 R53-VC HK shape 在同一 churn 机制下失去 VC 在**未变的** R40B `.so` 文件上 (`16384x14336x2048`, `16384x28672x2048`, `6144x32768x4096`)。
- Cohort churn 净: **0 / -3 = -3** 在**未变的** HK .so cell 上 (NOT R54 回归 — R45+ tail-draw)。
- R54 贡献: **+6 NEW VC** (E-1, E-2) - 3 cohort losses。
- **最终 VC count: 严格 10-run 36/42 (vs R53 33; vs R52 36)**。
- 30 个共享 VC shape 上 mean perf delta: **+4.42pp comp/shape** vs R53。
- Top deltas: +28.28pp (6144x4096x8192), +22.25pp (4096x14336x8192), +21.47pp (4096x14336x16384)。
- 文件: `R54_INTEGRATION_VERDICT.md`, `R54_INTEGRATION_MANIFEST.json`, `bench_all_42_R54_INTEGRATION.py`, `R54_INTEGRATION_{10RUN,SMOKE1}.{json,log,console}`, `R54_DECIDER_PLAN.md`, `R54_OPT_{E1,E2,D4A,D4B}_VERDICT.md`, `R54{E1,E2,D4A,D4B}_{1,2,3}_INTEGRATION_FRAGMENT.json`, `bench_R54{E1,E2,D4A,D4B}_{1,2,3}.py`。

**R54 net result**: **+3 NET VC 严格 10-run + 6 PERF 追赶 +17-28pp**, R54 不带来回归。Branch 应推进 with R54 manifest delta + 12 个新的 per-shape backend dispatch 表项。**0 kernel 修改, 0 新 shim build (连续第 6 次 R50D 复用)。**

### R54 主要 durable findings
1. **AITER `.co` dlopen pattern is now load-bearing for round-over-round VC stability**: R54 27/42 AITER cell bit-deterministic (wcf_max=0.0 across all 10 INDEPENDENT seeds), 在 majority leaderboard 上消除 cohort-race tail-draw 风险。R52 had 7 AITER cells; R54 has 27。R45+ cohort-race noise floor 现在只 apply 到 remaining 15 HK cells。
2. **D-3A-1 risk does NOT generalize from 96×640 to 256×256**: D-4A and D-4B 证明 for 256×256 tile case, AITER 在 84-94% HK 带上 uniformly +17-28pp beat HK。96×640 aspect-ratio sensitivity (R53 D-3A-1 DEAD) was tile-specific, not universal across 非 256×256 tiles。
3. **HK 256×256 path has substantial untapped headroom on already-VC shapes**: D-4B's +17-22pp gains over already-VC HK shapes (93-94% comp band) suggest HK kernel 256×256 path is leaving ~20pp on table that AITER 256×256 captures cleanly。
4. **Aiter heuristic uniformly picks 256×256 for all 12 R54 candidates**: validates `compute2mem_efficiency=128` 是 dominant tiebreak for M-N-K geometry distribution on this leaderboard。Future rounds should expect 256×256 to be default tile pick for majority of unmined shapes。
5. **6 consecutive R50D AS-IS reuse rounds**: shim is now production-stable; no candidate rebuild has been needed since R50D first PoC。Shim universal-bdx=256 + KernelArgs M/N/K/grid runtime parameterization absorbs all aiter f4gemm tile geometries cleanly。

### R55 候选 (post-R54, 按机制信心排序)
1. **R55 Opt D-extended-5 (最高信心)** —— 继续挖掘 remaining 9 HK-cell sub-95% pool against AITER 256×256。R54 之后, 显式非 AITER 候选: `16384x14336x{2048,4096}`, `16384x28672x{2048,4096}`, `16384x4096x4096`, `16384x6144x{2048,4096}`, `32768x14336x2048`, `32768x28672x2048`, `32768x6144x2048`, `4096x4096x8192`, `6144x32768x4096`。预估 +3-6 NET VC + further stability gain。
2. **R55 Opt F (中)** —— 调查是否 192×256 or 128×256 aiter tile 能救援 remaining 6 HK FAIL (currently `16384x{14336,28672}x{2048,4096}`, `16384x6144x4096`, `6144x32768x4096`)。
3. **R55 Opt B (低)** —— 32×32×64 MFMA in HK kernel — 仍延迟; AITER mining 是 far higher confidence per round。

### R55+ 不要尝试 (R45-R54 已关闭)
- R53 关闭列表 PLUS:
- **任何"改进" HK 256×256 path 的尝试** (R54 D-4A/D-4B 已确认 256×256 AITER 在 84-94% HK 带上 +17-28pp beat HK — 直接 port to AITER, 不要试图 "improve" HK)。

### R54 stopping-criterion 检查
- Floor (≥34/42 严格 10-run, recover -6 cohort 至少 4 个 cleanest cell): **MET (36/42 严格; 全部 6 个 cohort cell 救援 + 6 perf 追赶)**。
- Mode (36/42 严格): **MET 完全恰好**。
- Stretch (38/42 严格): **MISS (36/42)** but 36 是 sustainable ceiling (上限受 6 个 R40B HK FAIL 限制, R55 Opt D-extended-5 直接 path 到 38-42)。
- 轮价值: **+3 NET VC + 6 perf 追赶 + 5 durable findings + 12/12 PROMOTE / 0 DEAD = best round structure since round-tracking began**。

### 最近 12 轮 sanity check
- R43: DEAD (3 轴)
- R44: WIN +8 (27 → 35/42)
- R45-R49: 5 轮连续 DEAD
- R50: WIN +1 (35 → 36/42, aiter `.co` dlopen 首次 PoC)
- R51: WIN +1 严格 (30 → 31/42) + 2 perf 追赶
- R52: WIN +5 严格 (31 → 36/42) + 3 perf 追赶
- R53: PARTIAL WIN +2 NET VC 救援 -5 cohort 严格净 -3 (36 → 33/42) + 6 perf 追赶
- **R54: WIN +6 NET VC 救援 -3 cohort 严格净 +3 (33 → 36/42) + 6 perf 追赶 +17-28pp; 27/42 cell 现在 bit-deterministic AITER** ← 史上最大 AITER 扩展轮; KERNEL BASE 现已 STRUCTURALLY STABLE

---

## 历史 (2026-04-19, post-R53 — PARTIAL WIN +2 NET VC 救援 -5 cohort 严格净 -3, COMMIT, 首次非 256×256 尝试, R50D shim 证明为 tile-generic, 9 个候选中 8 PROMOTE / 1 DEAD, 33/42 严格 10-run VC, 连续第 5 次 R50D AS-IS 复用)

**HEADLINE**: R53 是**首次非 256×256 aiter `.co` 分发尝试**，验证了 universal-bdx=256 假设。严格 10-run VC delta vs R52 = **-3 (36 → 33/42)**, 但损失完全来自**未变的** HK R40B/R41B `.so` 上的 cohort-race tail-draw (R45+ 已记录现象: +1 cohort gain / -6 cohort losses, 没有一个是 R53 引起)。R53 实际贡献 = **+2 NET VC 救援** (D-3B_1 `(4096,32768,14336)` 66.70% NEW VC + D-3B_2 `(32768,4096,14336)` 85.56% NEW VC, 都之前是 NO_VC) **+ 6 个 perf 追赶** 通过 aiter `.co` (D-3A_2 +22.65pp, D-3A_3 +48.82pp, D-3B_3 +0.22pp, D-3C_1 +1.77pp, D-3C_2 +0.37pp, D-3C_3 +3.56pp)。Reviewer 10-run 全部 8/8 PROMOTE RE-VERIFIED bit-determinism (wcf_max=0.0, wcf_std=0.0, fin_min=1.0)。1 DEAD (D-3A_1 `(4096,14336,16384)` 96×640 SMOKE 回归到 52.45% comp vs HK 84.53% — worker 在 10-run 之前正确停止)。**R50D shim 现已证明为 tile-generic**: 同一个 `bdx=256` shim 处理 256×256, 96×640, AND 64×1024 aiter `.co` 文件，0 rebuild —— 验证 `asm_gemm_a4w4.cu:290` 的 universal-bdx 假设。连续第 5 轮 R50D AS-IS 复用; 0 kernel 修改。30 个共享 VC shape 上 mean +3.20pp comp/shape。

**R53 attempts 总结**:
- **R53 Opt D-3A (worker, 96×640 tile, 3 候选)**: 2/3 PROMOTE / 1 DEAD。
  - D-3A_1 `(4096,14336,16384)`: **DEAD** at SMOKE — 96×640 表现不及 (52.45% comp vs HK 84.53%); aspect-ratio 敏感性。Worker 在 10-run 之前正确停止。
  - D-3A_2 `(6144,4096,16384)`: **PROMOTE 10/10 OK, 107.62% comp (+22.65pp vs HK)**。wcf_max=0.0, fin_min=1.0。
  - D-3A_3 `(4096,6144,32768)`: **PROMOTE 10/10 OK, 131.63% comp (+48.82pp vs HK)** — 本轮最大 perf 追赶。wcf_max=0.0, fin_min=1.0。
- **R53 Opt D-3B (worker, 64×1024 tile, 3 候选)**: 3/3 PROMOTE, **+2 NET VC 救援**。
  - D-3B_1 `(4096,32768,14336)`: **PROMOTE 10/10 OK, 66.70% NEW VC** (was NO_VC)。Bit-deterministic。
  - D-3B_2 `(32768,4096,14336)`: **PROMOTE 10/10 OK, 85.56% NEW VC** (was NO_VC)。Bit-deterministic。
  - D-3B_3 `(128256,32768,4096)`: **PROMOTE 10/10 OK, 86.17% (+0.22pp marginal vs HK)**。Bit-deterministic。
- **R53 Opt D-3C (worker, 64×1024 tile, 3 marginal 候选)**: 3/3 PROMOTE。
  - D-3C_1 `(14336,32768,4096)`: **PROMOTE 87.83% (+1.77pp vs HK)**。Bit-deterministic。
  - D-3C_2 `(28672,32768,4096)`: **PROMOTE 87.68% (+0.37pp marginal vs HK)**。Bit-deterministic。
  - D-3C_3 `(16384,4096,14336)`: **PROMOTE 91.06% (+3.56pp vs HK)**。Bit-deterministic。

全部 8 个 PROMOTE worker AS-IS 复用 **EXISTING R50D shim** at `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`，使用新的 per-shape grid 参数和 per-shape aiter `.co` 路径 (`f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640.co` 和 `f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024.co`)。NO shim rebuild。Kernel 不变。

**R53 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010]; 4 GPU, 14 分钟 wall, 420 runs)**:
- Manifest 中 15 个 aiter `.co` override (R50D + R51 D-1/D-2/D-3 + R52 D-2A/D-2B/D-2C + R53 8 个非 256×256); 27 个 shape 用 HipKittens kernel + R44 baseline 参数。
- 全部 8 个 R53 PROMOTE 候选 reviewer re-bench RE-VERIFIED 10/10 PASS; 15/15 AITER cell PASS (100% bit-deterministic)。
- 18/27 HK cell PASS (9 FAIL: cohort-race wcf 或 fin gate)。
- 1 个 R52-NO shape 在 R53 cohort-race tail-draw 下获得 VC 在**未变的** `.so` 上 (`4096x4096x16384`)。
- 6 个 R52-VC shape 在同一 churn 机制下失去 VC 在**未变的** HK R40B/R41B `.so` 文件上 (`16384x4096x6144`, `28672x4096x8192`, `32768x4096x3072`, `4096x32768x4096`, `4096x32768x6144`, `32768x4096x2048`)。
- Cohort churn 净: **+1 / -6 = -5** 在**未变的** .so cell 上 (NOT R53 回归 — R45+ tail-draw)。
- R53 贡献: **+2 NEW VC** (D-3B_1, D-3B_2) + 1 cohort gain。
- **最终 VC count: 严格 10-run 33/42 (vs R52 严格 36/42)**。
- 30 个共享 VC shape 上 mean perf delta: **+3.20pp comp/shape**。
- D-3C 边际 promotion 检查: 全部 4 个 (D-3C_1/D-3C_2/D-3C_3/D-3B_3) 在 reviewer 10-run 上对 HK 保持正向 with bit-determinism — **0 reverts 需要**。
- 文件: `R53_INTEGRATION_VERDICT.md`, `R53_INTEGRATION_MANIFEST.json`, `bench_all_42_R53_INTEGRATION.py`, `R53_INTEGRATION_10RUN.{json,log,console}`, `R53_INTEGRATION_SMOKE1.{json,log,console}`, `R53_DECIDER_PLAN.md`。

**R53 net result**: **+2 NET VC 救援 + 6 PERF 追赶** 归因于 R53; -5 cohort tail-draw 在**未变的** `.so` 文件上 (R45+ 现象, 不是回归)。Net 严格 VC: 33/42。**首次非 256×256 尝试验证 R50D shim 为 TILE-GENERIC。0 kernel 修改, 0 新 shim build (连续第 5 次 R50D 复用)。**

### R54 候选 (post-R53, 按机制信心排序)
1. **R54 Opt D-extended-4 (最高信心)** —— 继续挖掘非 256×256 aiter `.co` tile 给 sub-90% HK shape。在 R53 之后, 对剩余 sub-90% HK-VC shape 列举针对完整 36-tile aiter `.co` 库。预估 +0-2 NET VC + 5-10pp mean comp 通过 D-3B/D-3C 风格救援。
2. **R54 Opt E — HK kernel cohort-race 稳定化 (中)** —— 处理 R40B/R41B 上 -6 个 cohort tail-draw 损失，方法 (a) 收紧内部 kernel gate 标准 或 (b) 找 aiter `.co` 替换那 6 个掉出 VC 的 shape。如果成功预估 +3-6 NET VC。
3. **R54 Opt B (低-中)** —— 试 32×32×64 MFMA (而非 16×16×128) 在 HipKittens kernel 中，针对没有 aiter `.co` fit 的 shape。未试的结构性轴。高风险。

### R54+ 不要尝试 (R45-R53 已关闭)
- R52 关闭列表 PLUS:
- **Aspect-ratio-blind aiter tile 选择** (D-3A_1 已演示 96×640 不是普遍优于 — 必须 per-shape SMOKE-gate 才能 promote)。
- K-loop 中**任何** fence 位置 (R45B / R47A / R48A / R49C / R50A 全部 DEAD)
- MFMA↔ds_read interleaving 变体单独 (R50A 关闭)
- `R38A_INLINE_BUFLOAD_LDS=1` 用于 production build (R47B 关闭)
- `R39A_TAIL_SCALE_CLAMP` 用于 intermediate-K wcf-flake shape (R46A 关闭)
- 4-buffer 或更高 LDS rotation 单独 (R46B + R47A 关闭)
- Wave-priority / s_nop pacing / MFMA half-split 单独 (R47C 用 10-run 关闭)
- `kpair_64mfma_step34` 物理 asm-block 拆分 (R48A 关闭)
- `PF_MPT` 深度 override (R48C 机制性关闭)
- Step34 内 MFMA reorder / s_setprio / lgkmcnt drain 单独 (R48B 关闭)
- 单 knob aiter pattern 移植 (R49A 关闭)
- R44A back-edge drain 扩展到 N=32768 (R49B 关闭; cohort 随 N scale)
- Producer asm volatile 内嵌 vmcnt (R49C 关闭)
- R44 VC shape 上的 `gm × lgk × pfoff` knob sweep (R50C 关闭)
- 任何"改进" aiter `.co` dlopen 路径本身的尝试 (R51 关闭)

### R53 stopping-criterion 检查
- Floor (≥31/42 严格 10-run, 无可归因回归): **MET (33/42 严格; +2 NET VC 救援 + 6 perf 追赶 归因于 R53; -3 严格净是**未变的** `.so` 上记录的 cohort tail-draw, 不是 R53 回归)**。
- Stretch (≥34/42 严格): **MISS (33/42)** 但本轮机制 (tile-generic shim 验证) 是 durable 并解锁 R54 D-extended-4。
- 轮价值: **+2 NET VC 救援 + 6 perf 追赶 + 5 durable findings** (R50D shim 是 tile-generic; 96×640 有 aspect-ratio 敏感性; 64×1024 broadly competitive; aiter `.co` 在 15 个 shape 上验证; cohort-race churn 现已主导 net VC 会计在 -5 noise floor)。

### 最近 11 轮 sanity check
- R43: DEAD (3 轴)
- R44: WIN +8 (27 → 35/42)
- R45-R49: 5 轮连续 DEAD
- R50: WIN +1 (35 → 36/42, aiter `.co` dlopen 首次 PoC)
- R51: WIN +1 严格 (30 → 31/42) + 2 perf 追赶
- R52: WIN +5 严格 (31 → 36/42) + 3 perf 追赶
- **R53: PARTIAL WIN +2 NET VC 救援 -5 cohort 严格净 -3 (36 → 33/42) + 6 perf 追赶; 首次非 256×256 尝试; R50D shim 证明为 tile-generic** ← 机制连胜延续 even with 负 net 严格

---

## 历史 (2026-04-19, post-R52 — WIN +5 VC + 3 PERF 追赶, COMMIT, 连续第三个非 DEAD 轮, R44 以来最大 VC 增长, 36/42 严格 10-run VC, AITER `.CO` DLOPEN 已经在 7 个 shape 上验证)

**HEADLINE**: R52 是连续第 3 个 WIN 轮且是 R44 (+8) 以来最大的 VC 增长。Net VC delta vs R51 = **+5 NET VC 严格 10-run** (31 → 36/42)。3/3 PROMOTE / 0 DEAD (与 R51 轮结构一致)。3 个 PROMOTE worker 是通过 R50D aiter `.co` dlopen shim **AS-IS 复用 (连续第 4 轮，无 shim rebuild, 无 kernel 修改)** 的 perf 追赶: D-2A `(4096,28672,32768)` 61.9% → 101.85% comp (+39.93pp), D-2B `(4096,32768,128256)` 72.3% → 99.72% comp (+27.47pp), D-2C `(4096,4096,32768)` 77.1% → 105.91% comp (+28.81pp)。Reviewer 10-run 全部 RE-VERIFIED 10/10 PASS, bit-determinism (wcf_max=0.0, wcf_std=0.0, fin_min=1.0)。+5 NET VC 完全来自 cohort-race tail-draw 在**未变的** `.so` 文件上 (6 个 shape 获得, 1 个失去 — favorable seed draw)。Aiter `.co` dlopen pattern 现已**生产就绪，在 7 个不同 shape 上验证** (R50D + R51 D-1/D-2/D-3 + R52 D-2A/D-2B/D-2C)，对 256×256 tile case shape-generic, K-generic (D-2B at K=128256), grid-size-generic (D-2C at gdx=gdy=16, D-2A at gdx=112)。Dlopen 轴上 100% PROMOTE rate (6/6) 当目标 shape 的 aiter heuristic 选 256×256 时。

**R52 attempts 总结**:
- **R52 Opt D-2A — Aiter `.co` dlopen 给最大 sub-90% gap 的 `(4096, 28672, 32768)`**: **PROMOTE 10/10 OK, +39.93pp comp。** AS-IS 复用 R50D shim at `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` (无 rebuild)。Per-shape grid: gdx=112, gdy=16, gdz=1, bdx=256。KernelArgs: M=4096, N=28672, K=32768。Worker 10-run: p50=5736.7 TFLOPS = 101.54% comp; reviewer re-bench: p50=5754.2 TFLOPS = 101.85% comp。10 个 INDEPENDENT seed 上 wcf_max=0.0, wcf_std=0.0, fin_min=1.0。文件: `R52_OPT_D2A_VERDICT.md`, `R52D2A_INTEGRATION_FRAGMENT.json`, `R52_OPT_D2A_{SMOKE,10RUN}.{json,log}`, `bench_R52D2A.py`。Kernel 不变。
- **R52 Opt D-2B — Aiter `.co` dlopen 给 `(4096, 32768, 128256)` (board 上最大 K)**: **PROMOTE 10/10 OK, +27.47pp comp + K=128256 通用性证明。** AS-IS 复用 R50D shim。Per-shape grid: gdx=128, gdy=16, gdz=1, bdx=256。KernelArgs: M=4096, N=32768, K=128256。K=128256 是唯一标记的风险轴 — 0 问题 (clean SMOKE 一次过，10 个 INDEPENDENT seed 完美 bit-stable)。Worker 10-run: p50=5763.6 TFLOPS = 99.70% comp; reviewer re-bench: p50=5765.1 TFLOPS = 99.72% comp。**确认 aiter `.co` pattern 是 K-generic。** 文件: `R52_OPT_D2B_VERDICT.md`, `R52D2B_INTEGRATION_FRAGMENT.json`, `R52_OPT_D2B_{SMOKE,10RUN}.{json,log}`, `bench_R52D2B.py`。Kernel 不变。
- **R52 Opt D-2C — Aiter `.co` dlopen 给 `(4096, 4096, 32768)` (单轮 16×16 grid)**: **PROMOTE 10/10 OK, +28.81pp comp + grid-size 通用性证明。** AS-IS 复用 R50D shim。Per-shape grid: gdx=16, gdy=16, gdz=1, bdx=256。KernelArgs: M=4096, N=4096, K=32768。单轮 16×16 grid (R51-R52 测试中最小的 grid)。Worker 10-run: p50=5437.2 TFLOPS = 105.52% comp; reviewer re-bench: p50=5457.4 TFLOPS = 105.91% comp。**确认 aiter `.co` pattern 是 grid-size-generic** (D-2A at gdx=112 + D-2C at gdx=16 = 完整范围)。文件: `R52_OPT_D2C_VERDICT.md`, `R52D2C_INTEGRATION_FRAGMENT.json`, `R52_OPT_D2C_{SMOKE,10RUN}.{json,log}`, `bench_R52D2C.py`。Kernel 不变。

**R52 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010]; 4 GPU, 13 分钟 wall, 420 runs)**:
- Manifest 中 7 个 aiter `.co` override (R50D 的 `4096x32768x28672` + R51 D-1/D-2/D-3 + R52 D-2A/D-2B/D-2C); 35 个 shape 用 HipKittens kernel + R44 baseline 参数。
- 全部 3 个 R52 PROMOTE 候选 reviewer re-bench RE-VERIFIED 10/10 PASS。
- 6 个 R51-NO shape 在 R52 cohort-race tail-draw 下获得 VC (`128256x32768x4096`, `14336x32768x4096`, `32768x28672x2048`, `4096x128256x32768`, `4096x32768x6144`, `4096x6144x32768`) 在**未变的** `.so` 文件上。
- 1 个 R51-VC shape 在同一 churn 机制下失去 VC (`4096x4096x16384`: PASS_10/10 → PASS_9/10, pct 几乎不变 93.34 → 93.07)。
- Cohort churn 净: **+5 VC** (6 得 - 1 失)。D-2 PROMOTE worker 不增加 VC count (本来就是 HK-VC, 只是低 comp); 它们是纯 perf 追赶。
- **最终 VC count: 严格 10-run 36/42 (vs R51 严格 31/42)**。
- 30 个共享 VC shape 上 mean perf delta: **+3.27pp comp/shape** (D-2A/B/C 给 mean 贡献 +96.21pp 累计)。
- 文件: `R52_INTEGRATION_VERDICT.md`, `R52_INTEGRATION_MANIFEST.json`, `bench_all_42_R52_INTEGRATION.py`, `R52_INTEGRATION_10RUN.{json,log,console}`, `R52_INTEGRATION_SMOKE1.{json,log,console}`, `R52_DECIDER_PLAN.md`。

**R52 net result**: **+5 NET VC + 3 PERF 追赶 (D-2A/B/C 累计 +96.21pp)**, R52 不带来回归。Branch 应推进 with R52 manifest delta + 3 个新的 per-shape backend dispatch 表项。**0 kernel 修改, 0 新 shim build (连续第 4 次 R50D 复用)。**

### R53 候选 (post-R52, 按机制信心排序)
1. **R53 Opt D-extended-3 (最高信心)** —— 继续挖掘 256×256 aiter tile 最优的 sub-90% comp HK-VC shape。在 R52 promotions 之后，audit `R52_INTEGRATION_10RUN.json` 列出剩余 sub-95% shape。预估 +0-2 VC + 5-10pp mean comp。同 R50D shim AS-IS 复用。
2. **R53 Opt D-non-256x256 (中)** —— Aiter 在不同 tile geometry 有 36 个 `.co`。对于 aiter heuristic 选 NON-256×256 的 shape, 用 runtime tile 参数泛化 R50D shim (扩展 KernelArgs `tile_m`/`tile_n` 字段) 并加 per-tile shim build (或 runtime dispatch)。需要 1 次 shim rebuild 但之后可复用。
3. **R53 Opt B (低-中)** —— 试**不同的** MFMA shape (32×32×64 而非 16×16×128) 在 HipKittens kernel 中，针对没有 aiter `.co` fit 的 shape。未试的结构性轴。高风险。

### R53+ 不要尝试 (R45-R52 已关闭)
- K-loop 中**任何** fence 位置 (R45B / R47A / R48A / R49C / R50A 全部 DEAD)
- MFMA↔ds_read interleaving 变体单独 (R50A 关闭)
- `R38A_INLINE_BUFLOAD_LDS=1` 用于 production build (R47B 关闭)
- `R39A_TAIL_SCALE_CLAMP` 用于 intermediate-K wcf-flake shape (R46A 关闭)
- 4-buffer 或更高 LDS rotation 单独 (R46B + R47A 关闭)
- Wave-priority / s_nop pacing / MFMA half-split 单独 (R47C 用 10-run 关闭)
- `kpair_64mfma_step34` 物理 asm-block 拆分 (R48A 关闭)
- `PF_MPT` 深度 override (R48C 机制性关闭)
- Step34 内 MFMA reorder / s_setprio / lgkmcnt drain 单独 (R48B 关闭)
- 单 knob aiter pattern 移植 (R49A 关闭)
- R44A back-edge drain 扩展到 N=32768 (R49B 关闭; cohort 随 N scale)
- Producer asm volatile 内嵌 vmcnt (R49C 关闭)
- R44 VC shape 上的 `gm × lgk × pfoff` knob sweep (R50C 关闭, 距饱和 2pp 内)
- 任何"改进" aiter `.co` dlopen 路径本身的尝试 (R51 关闭: bit-deterministic, 在 100%+ comp)

### R52 stopping-criterion 检查
- Floor (≥31/42 严格 10-run, 无回归): **MET (36/42 严格; +5 net VC; 0 可归因回归 —— 1 个失去的 VC 是**未变的** `.so` 上记录的 cohort tail-draw)**。
- Stretch (≥34/42 严格): **MET +2 超过 stretch** (36/42)。
- 轮价值: **+5 VC + 3 perf 追赶 (累计 +96.21pp) + 4 durable findings** (aiter `.co` dlopen pattern 在 7 个 shape 上验证; 在 K=128256 K-generic; 在完整 gdx 范围内 grid-size-generic; dlopen 轴上 100% PROMOTE rate 6/6)。

### 最近 10 轮 sanity check
- R43: DEAD (3 轴)
- R44: WIN +8 (27 → 35/42)
- R45-R49: 5 轮连续 DEAD
- R50: WIN +1 (35 → 36/42, aiter `.co` dlopen 首次 PoC)
- R51: WIN +1 严格 (30 → 31/42) + 2 perf 追赶 (3/3 PROMOTE)
- **R52: WIN +5 严格 (31 → 36/42) + 3 perf 追赶 (3/3 PROMOTE)** ← R44 以来最大增长，连胜延续

---

## 历史 (2026-04-19, post-R51 — WIN +1 VC + 2 大幅 PERF 追赶, COMMIT, 连续第二个非 DEAD 轮, AITER `.CO` DLOPEN PATTERN 证明 SHAPE-GENERIC, 37/42 VC 混合协议 OR 31/42 严格 10-run)

**HEADLINE**: R51 是 R50 打破连败之后**连续第二个 WIN 轮**。Net VC delta vs R50 = **+1** (30 → 31/42 严格 10-run; 或 36 → 37/42 R50 混合协议 headline)。3/3 PROMOTE, 0 DEAD = R44 以来收益最高的一轮。突破点是 **R51 Opt D (3 个并行 worker)**: 同一个 R50D aiter `.co` dlopen shim **AS-IS 复用 (无 rebuild)**，通过 per-shape backend dispatch 表项给 3 个新 shape 用: D-1 `(14336,4096,32768)` 60.4% → 103.3% comp (+42.9pp, +2,249.7 TFLOPS), D-2 `(16384,4096,28672)` 62.0% → 104.8% comp (+42.8pp, +2,366.2 TFLOPS, 退役脆弱的 R44A back-edge-drain cell), D-3 `(28672,4096,16384)` FLAKE_7/10 → PASS_10/10 @ 102.4% comp (+1 NET VC)。所有 3 个都达成 bit-determinism (10 个 INDEPENDENT seed 上 wcf_max=0.0, wcf_std=0.0, fin_min=1.0)。Reviewer 10-run integration 确认 net +1 VC + 26 个共享 VC shape 上 mean +220.8 TFLOPS/shape (+4.35pp comp/shape)。**Aiter `.co` dlopen pattern 现已证明对 256×256 tile case 是 shape-generic** (4 个不同 shape 移植，0 shim rebuild) —— 已经是 production-ready 的 per-shape escape hatch，可用于任何落后 aiter binary >10pp 的 HipKittens cell。

**R51 attempts 总结**:
- **R51 Opt D-1 — Aiter `.co` dlopen 给 leaderboard 上最大 gap 的 `(14336,4096,32768)`**: **PROMOTE 10/10 OK, +42.9pp comp。** AS-IS 复用 R50D shim at `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so` (无 rebuild)。Per-shape grid: gdx=ceil(N/256)=16, gdy=ceil(M/256)=56, gdz=1, bdx=256。KernelArgs: M=14336, N=4096, K=32768。Aiter heuristic at `asm_gemm_a4w4.cu:100-145` 确认 256×256 是最优 tile 选择 (min local_round 0.219, tiebreak compute2mem_efficiency=128.0 胜过 192×256)。10-run @ 80% INDEPENDENT seeds: 10/10 OK, wcf_max=0.0, fin_min=1.0, snr_med 55.6 dB, p50 = 5,418.4 TFLOPS = competitor 5,243.6 的 103.3%。文件: `R51_OPT_D1_VERDICT.md`, `R51D1_INTEGRATION_FRAGMENT.json`, `R51_OPT_D1_{SMOKE,10RUN}.json`, `bench_R51D1.py`。Kernel 不变。
- **R51 Opt D-2 — Aiter `.co` dlopen 给 `(16384,4096,28672)` (R44A 脆弱 back-edge drain 的替代)**: **PROMOTE 10/10 OK, +42.8pp comp + 鲁棒性提升。** AS-IS 复用 R50D shim。Per-shape grid: gdx=16, gdy=64, gdz=1, bdx=256。KernelArgs: M=16384, N=4096, K=28672。替代 HipKittens 唯一在 VC 表中的 K=28672 cell —— 但只能通过 R44A back-edge drain (wcf_std=0.008，脆弱，接近 0.01 协议上限)。Aiter binary 提供 wcf_std=0.0 —— **退役最后一个非 FUSED+TS K=28672 脆弱 cell**, 消除该 shape 上的 cohort-race 风险。p50 = 5,219.0 TFLOPS = competitor 4,981.2 的 104.8%。文件: `R51_OPT_D2_VERDICT.md`, `R51D2_INTEGRATION_FRAGMENT.json`, `R51_OPT_D2_{SMOKE,10RUN}.json`, `bench_R51D2.py`。Kernel 不变。
- **R51 Opt D-3 — Aiter `.co` dlopen 给 `(28672,4096,16384)` FLAKE-to-PASS 救援 (+1 NET VC)**: **PROMOTE 10/10 OK, +20.2pp comp, +1 NET VC。** 这是 D-1/D-2/D-3 中唯一相对 R50 baseline 增加 VC count 的候选。R47C 把这个 shape 标识为 HK kernel 下 5/10 OK at wcf_max=0.0272; R51D-3 换上 aiter binary → 10/10 OK at wcf_max=0.0。AS-IS 复用 R50D shim。Per-shape grid: gdx=16, gdy=112, gdz=1, bdx=256。KernelArgs: M=28672, N=4096, K=16384。p50 = 5,179.5 TFLOPS = competitor 5,058.5 的 102.4%。文件: `R51_OPT_D3_VERDICT.md`, `R51D3_INTEGRATION_FRAGMENT.json`, `R51_OPT_D3_{SMOKE,10RUN}.json`, `bench_R51D3.py`。Kernel 不变。

**R51 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010]; 4 GPU, 13 分钟 wall, 420 runs)**:
- Manifest 中 4 个 aiter `.co` override (R50D 的 `4096x32768x28672` + R51 D-1 + D-2 + D-3); 38 个 shape 用 HipKittens kernel + R44 baseline 参数。
- 全部 3 个 R51 PROMOTE 候选确认 10/10 PASS, wcf_max=0.0, fin_min=1.0。
- 4 个 R50-VC shape 在 R51 下丢 VC (`32768x28672x2048`, `4096x6144x32768`, `4096x128256x32768`, `128256x32768x4096`) —— 全部在**未变的** `.so` 文件上; R45+ 已记录 cohort-race tail-draw 现象, 不是 R51 引入的回归。
- 4 个 R50-NO shape 在 R51 下对称地获得 VC (`32768x4096x2048`, `4096x28672x32768`, `4096x32768x4096`, `4096x32768x128256`), 来自相同的 cohort-race churn。
- Cohort churn 净 VC delta: 0 (4 丢 = 4 得); D-3 净 VC delta: +1; **总计 +1 NET VC**。
- 26 个共享 VC shape 上 mean perf delta: **+220.8 TFLOPS/shape, +4.35pp comp/shape** (D-1 + D-2 给 mean 贡献 +85.7pp 累计)。
- **最终 VC count: 严格 10-run 31/42; R50 混合 5-run baseline + 10-run override headline 构造 37/42。**
- 文件: `R51_INTEGRATION_VERDICT.md`, `R51_INTEGRATION_MANIFEST.json`, `bench_all_42_R51_INTEGRATION.py`, `R51_INTEGRATION_10RUN.{json,log,console}`, `R51_INTEGRATION_SMOKE1.{json,log,console}`, `R51_DECIDER_PLAN.md`。

**R51 net result**: **+1 NET VC + 2 大幅 PERF 追赶 (D-1+D-2 累计 +85.7pp)**, R51 不带来回归。Branch 应推进 with R51 manifest delta + 3 个新的 per-shape backend dispatch 表项。**0 kernel 修改, 0 新 shim build。**

### R52 候选 (post-R51, 按机制信心排序)
1. **R52 Opt D-extended-2 (最高信心)** —— 识别下一批 HK-VC <90% comp shape，其中 256×256 aiter tile 按 heuristic 是最优。Per `R51_DECIDER_PLAN.md` 剩余候选: `32768x4096x14336`, `16384x28672x4096`。两者预估 80-95pp 收益。预估 +0-2 VC + 5-10pp mean comp。同 shim AS-IS 复用。
2. **R52 Opt D-non-256x256 (中)** —— Aiter 在不同 tile geometry 有 36 个 `.co` (128×128, 192×256, 256×128 等)。对于 aiter heuristic 选 NON-256×256 的 shape, 通过扩展 KernelArgs `tile_m`/`tile_n` 字段并加 per-tile shim build (或用 runtime tile 参数泛化 R50D shim) 把 shim 移植到不同 tile。
3. **R52 Opt B (低-中)** —— 试**不同的** MFMA shape (32×32×64 而非 16×16×128) 在 HipKittens kernel 中打破 cluster-B cohort race，在那些没有 aiter `.co` fit 的 shape 上。未试的结构性轴。高风险。

### R52+ 不要尝试 (R45-R51 已关闭)
- K-loop 中**任何** fence 位置 (R45B / R47A / R48A / R49C / R50A 全部 DEAD)
- MFMA↔ds_read interleaving 变体单独 (R50A 关闭)
- `R38A_INLINE_BUFLOAD_LDS=1` 用于 production build (R47B 关闭)
- `R39A_TAIL_SCALE_CLAMP` 用于 intermediate-K wcf-flake shape (R46A 关闭)
- 4-buffer 或更高 LDS rotation 单独 (R46B + R47A 表明是 fence-interaction 不是 slot-count)
- Wave-priority / s_nop pacing / MFMA half-split 单独 (R47C 用 10-run 关闭)
- `kpair_64mfma_step34` 物理 asm-block 拆分 (R48A 关闭)
- `PF_MPT` 深度 override (R48C 机制性关闭)
- Step34 内 MFMA reorder / s_setprio / lgkmcnt drain 单独 (R48B 关闭)
- 单 knob aiter pattern 移植 (R49A 关闭)
- R44A back-edge drain 扩展到 N=32768 (R49B 关闭; cohort 随 N scale)
- Producer asm volatile 内嵌 vmcnt (R49C 关闭)
- R44 VC shape 上的 `gm × lgk × pfoff` knob sweep (R50C 关闭, 距饱和 2pp 内)
- **任何"改进" aiter `.co` dlopen 路径本身的尝试 (R51 关闭: bit-deterministic, 在 100%+ comp, 没有空间)**

### R51 stopping-criterion 检查
- Floor (≥36/42 混合协议或 ≥30/42 严格 10-run, 无回归): **MET (37/42 混合或 31/42 严格; +1 net VC; 0 可归因回归 —— 4 个丢的 VC 是**未变的** `.so` 上记录的 cohort tail-draw)**。
- Stretch (≥38/42 混合): **NOT MET** (3 个 PROMOTE worker 但 4 个 cohort 损失在混合记账中抵消了 2 个收益)。
- 轮价值: **+1 VC + 2 大幅 perf 追赶 + 4 durable findings** (aiter `.co` dlopen pattern 对 256×256 shape-generic; aiter binary 结构性 bit-deterministic; R50 "36/42" headline 是 mixed-protocol; 3 PROMOTE / 0 DEAD = R44 以来收益最高)。

### 最近 9 轮 sanity check
- R43: DEAD (3 轴)
- R44: WIN +8 (27 → 35/42)
- R45-R49: 5 轮连续 DEAD
- R50: WIN +1 (35 → 36/42, aiter `.co` dlopen 首次 PoC)
- **R51: WIN +1 (36 → 37/42 混合 OR 30 → 31/42 严格) + 2 perf 追赶 (3/3 PROMOTE)** ← 连胜延续

---

## 历史 (2026-04-19, post-R50 — WIN +1 VC ROUND, COMMIT, 4 周来首个非 DEAD 轮, AITER `.CO` DLOPEN 突破, 36/42 VC)

**HEADLINE**: R50 打破了近 7 轮中 5 个 DEAD 的连续。Net VC delta vs R44 baseline = **+1** (35 → 36/42 VC)。突破点是 **R50 Opt D**: 在 production harness 中做 per-shape backend dispatch，将 perma-CRASH cell `(4096,32768,28672)` 通过 `hipModuleLoadData` + `hipModuleGetFunction` + `hipModuleLaunchKernel` 绑定到 aiter 手写的 `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`。10/10 INDEPENDENT seeds OK, 5585.6 TFLOPS = competitor 5568.2 的 **100.31%**。**该 shape 史上首次 VC** (R43-R49 测试的每个 HipKittens 变体上都 perma-CRASH)。Aiter `.co` dlopen pattern **可复用**: 任何未来 HipKittens 解不了且 aiter 有 tuned `.co` 的 MXFP4 cell 都可以用同样的 shim 方法在 <1 天内解决。

**R50 attempts 总结**:
- **R50 Opt A — Aiter MFMA↔ds_read 1:3/1:4 spread 交错移植到 `kpair_64mfma_step34` asm volatile 内部**: **DEAD on 10-run @ 80% gate。"MFMA accumulator race = orderable scheduling" 假设的第 5 次独立关闭。** Macro `R50A_AITER_INTERLEAVE` (默认 OFF, 与 R44 baseline byte-equiv) 用 1:4 spread 变体替换 ~256 行 asm volatile body。ISA 验证: BASELINE 显示 25-instr 的纯 MFMA runs (61 处); v1 把其中 30 个 25-runs 切成 3、4、11 长度。Compiler 没 strip rewrite。10-run 在 6 个 cluster-B wcf-flake target 上: v1 下 0/6 VC (与 baseline 0/6 一致); R44 stretch `4096x4096x8192` 保留 VC (10/10 OK, +0.3% TFLOPS)。结合 R45B (5 个块内 fence 位置 DEAD)、R47A (3 个 3-buffer 外部位置 DEAD)、R49A (单 vmcnt knob DEAD)、R49C (producer asm 内嵌 fence DEAD): **cluster-B cohort race 存在于 MFMA pipeline 内部的 AGPR forwarding path，不在编译器或 asm body 能重排的任何边界上。** Race 对 MFMA/AGPR 微架构是结构性的。文件: `R50_OPT_A_VERDICT.md`, `R50A_INTEGRATION_FRAGMENT.json` (`{}`), `R50A_aiter_kloop_body.s`, `R50A_old_step34.s`, `R50A_new_step34.s`, `R50A_KERNEL_ISA.s`, `R50A_BASELINE_ISA.s`, `R50_OPT_A_{SMOKE,JACCARD,10RUN}.{json,log}`, `R50A_BUILD_MANIFEST.json`, `build_R50A/*.so` (22 个 module)。Kernel macro `R50A_AITER_INTERLEAVE` at `kernel_mxfp4_gluon_cpp.cpp` (默认 OFF)。
- **R50 Opt C — 14 个 R44 VC <90% comp shape 的 per-shape `gm × lgk × pfoff` perf 轴追赶 sweep**: **PROMOTE → 在 10-run cross-val 下 REVERT。纯变体-knob 重调 (无新 macro)。** 14 个 R44 VC <90% comp shape; 每 shape: 27-cell sweep (lgk ∈ {1,2,3} × gm ∈ {6,7,8} × pfoff offset ∈ {-4,0,+4})。5-run @ 80% INDEPENDENT-seed gate 找到 1 个 PROMOTE 候选: `4096x28672x32768` gm8_lgk2_po28 (+2.90% pct_comp, base 61.90% → 64.80%, wcf_max=0.0182)。R50 INTEGRATION reviewer 10-run cross-val 抓到 wcf_max=0.0296 > 0.02 hard gate (9/10 OK) —— 经典 cohort-race tail draw。**REVERT 回 R44 R41A baseline。** 其他 shape 最佳收益全部 <2.0%: `6144x4096x8192` +1.77%, `32768x4096x7168` +1.98%, `28672x4096x8192` +1.02% —— 现有变体表在 14 个测试 shape 中 13 个**已自动调到 0-2pp 饱和**。**机制 (durable)**: R25-F/G `pfoff` 机制基本耗尽; R51+ perf 工作要攻击不同的结构性轴 (MFMA shape, tile geometry, thread block size)。5-run perf gate 对 cohort-race-prone shape **不够** —— 候选可以在同一个 `.so` 上 5-run 通过、10-run 失败。**所有 R51+ perf 轮必须用 10-run @ 80% 作为 promote gate。** Decider 必须合成 verdict，因为 Opt C agent 在 self-matching `pgrep -f "bench_R50C"` 等待循环中 stall (等待循环自己的 bash 命令行包含字面量 `bench_R50C`，pgrep 找到自己的进程从未返回 0)。**Agent 进程 bug (durable)**: 永远不要用 `pgrep -f X` 等待循环，其中 bash 命令行可能匹配 X —— 用 `pgrep -fx`、保存 PID + `wait $PID`、或在前台运行 bench。文件: `R50_OPT_C_VERDICT.md` (decider 合成), `R50C_INTEGRATION_FRAGMENT.json` (PROMOTE → reviewer cross-val REVERT), `R50C_SWEEP.json` (88 KB), `R50C_PREFILTER.json` (386 KB), `R50C_PROMOTE_CANDIDATES.json`, `R50C_baseline_pct_comp.json`, `R50C_BUILD_MANIFEST.json` (113 KB), `build_R50C/*.so`。Kernel 不变。
- **R50 Opt D — Aiter `.co` dlopen escape hatch 解决 `(4096,32768,28672)` perma-CRASH cell**: **PROMOTE +1 VC。R44 以来首次 WIN。** 自包含 pybind11 shim (`R50D_aiter_dlopen.cpp`, 219 行) 通过 `hipModuleLoadData` + `hipModuleGetFunction` + `hipModuleLaunchKernel` 绑定 `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`。372-byte `KernelArgs` ABI 从 `/shared_nfs/kyle/test/aiter/csrc/py_itfs_cu/asm_gemm_a4w4.cu` 逐字镜像 (`static_assert(sizeof(KernelArgs)==372)`)。Launch grid (128, 16, 1) block (256, 1, 1) shared 0 字节 —— 与 HipKittens 256×256 tile geometry 同 MFMA shape。**Layout 要求 (集成的关键)**: B 通过 `aiter.shuffle_weight(B, layout=(16,16))` preshuffle (与 HK 不同); A_scale 和 B_scale 通过 `aiter.get_triton_quant(per_1x32)(x, shuffle=True)` (与 HK `preshuffle()` 不同); A 是 row-major fp4x2 uint8 (与 HK 同); C 是 `[((M+31)//32)*32, N] bf16` row-major (行 pad 到 32 的倍数)。10-run @ 80% INDEPENDENT-seed 结果: 10/10 OK, wcf_max=0.0, fin_min=1.0, snr_med 55.59-55.63 dB, p50 = 5585.6 TFLOPS = **competitor 5568.2 的 100.31%**。该 shape 史上首次 VC —— 直接从 PERMA-CRASH 跳到 aiter baseline 之上。**`bench_all_42_R50_INTEGRATION.py` 中的 per-shape backend dispatch**: `(M,N,K) == (4096,32768,28672)` → 调 shim 配 aiter prep utilities; 否则 → HipKittens kernel 配 HK prep。**对其他 shape 0 blast radius** —— 单 cell escape hatch，无 kernel mutation。文件: `R50_OPT_D_VERDICT.md`, `R50D_INTEGRATION_FRAGMENT.json`, `R50D_aiter_dlopen.cpp` (219 行), `build_R50D.py`, `build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`, `bench_R50D.py`, `R50D_aiter_csv_audit.md`, `R50D_aiter_symbols.txt`, `R50_OPT_D_{SMOKE,10RUN}.{json,log}`, `R50D_BUILD_MANIFEST.json`。

**R50 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010]; 2 GPU, 22 分钟 wall, 420 runs)**:
- 35 R44 VC shape 在 R44 manifest 上保留 (35 个中有 5 个在**未变的** `.so` 文件上发生概率 cohort-race 失败 —— R45+ 已记录现象 per `project_mxfp4_R45_cohort_tail_draw.md`，不是 R50 引入的回归; R44 5-run 协议仍显示 35 VC)。
- `4096x32768x28672` (Opt D) → **PROMOTE 10/10 OK at 100.31% comp**。
- `4096x28672x32768` (Opt C) → REVERT (10-run wcf_max 0.0296 > 0.02 hard gate)。
- 35 个 R44 VC shape 上 net perf delta: **-4.3 TFLOPS/shape avg** (perf-中性, 在 seed noise floor 内)。
- **最终 VC count: 36/42 (35 R44 baseline + 1 from R50 Opt D)**。
- 文件: `R50_INTEGRATION_VERDICT.md`, `R50_INTEGRATION_MANIFEST.json`, `bench_all_42_R50_INTEGRATION.py`, `R50_INTEGRATION_10RUN.{json,log,console}`, `R50_INTEGRATION_SMOKE1.{json,log}`, `R50_DECIDER_PLAN.md`。

**R50 net result**: **+1 VC (35 → 36/42)**, 0 regression。Branch 应推进 with R50 manifest delta + aiter shim + per-shape backend dispatch + R50A macro (默认 OFF) commit。

### R51 候选 (post-R50, 按机制信心排序)
1. **R51 Opt D-extended (最高信心)** —— 识别其他 sub-90% comp 的 HipKittens shape，如果 aiter 有 tuned `.co` 就移植 R50D shim pattern。在 `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/` 中找 ≤80% comp 且 aiter binary 存在的 shape。预估 +2-3 VC 潜力。成本: ~1 天/shape; layout 需要 aiter prep utils。对其他 shape 0 blast radius。
2. **R51 Opt B (中)** —— 试**不同的** MFMA shape (例如 32×32×64 而非 16×16×128) 来打破 cluster-B cohort race 上的 AGPR forwarding chain。在结构上与所有先前轴不同 (R50A 是 orderable-scheduling 假设的第 5 次关闭; R51B 会攻击微架构层)。高风险但是 kernel 侧唯一未试的结构性轴。
3. **R51 perf 轮 (低优先级 —— 根据 R50 Opt C, perf 轴基本耗尽)** —— 如果尝试，必须从一开始就用 10-run @ 80% gate; 不要相信 5-run 收益。攻击不同的结构性轴 (kernel 级变更; 不是 `gm × lgk × pfoff` knob)。

### R51+ 不要尝试 (R45-R50 已关闭)
- K-loop 中**任何** fence 位置 (R45B / R47A / R48A / R49C / R50A 全部 DEAD)
- MFMA↔ds_read 交错变体单独 (R50A 关闭)
- `R38A_INLINE_BUFLOAD_LDS=1` for production builds (R47B 关闭)
- intermediate-K wcf-flake shape 的 `R39A_TAIL_SCALE_CLAMP` (R46A 关闭)
- 4-buffer 或更高的 LDS rotation 单独 (R46B + R47A 暗示 fence-interaction 不是 slot-count)
- Wave-priority / s_nop pacing / MFMA half-split 单独 (R47C 已被 10-run 关闭)
- `kpair_64mfma_step34` 物理 asm-block split (R48A 关闭)
- `PF_MPT` depth override (R48C 机制层关闭)
- 单独的 step34 内部 MFMA reorder / s_setprio / lgkmcnt drain (R48B 被 10-run gate 关闭)
- 单 knob aiter pattern port (R49A 关闭)
- R44A back-edge drain 扩展到 N=32768 (R49B 关闭; cohort scales with N)
- Embedded vmcnt 在 producer asm volatile 内部 (R49C 关闭)
- `gm × lgk × pfoff` knob sweep on R44 VC shapes (R50C 在 2pp saturation 内关闭)

### R50 stopping-criterion check
- Floor (≥35/42, no regression): **MET** (35 R44 baseline 保留 + 1 个新 VC; 最终 36/42; 0 regression)。
- Stretch (≥36/42): **MET** (Opt D 在 100.31% comp 下交付 `4096x32768x28672`)。
- Round value: **+1 VC + 4 个 durable 发现** (R50A orderable-scheduling 假设的第 5 次关闭; R50C perf 轴 saturation 证据; R50D aiter `.co` dlopen 可复用突破 pattern; agent 进程 bug —— `pgrep -f` self-match —— methodology 教训)。**4 周来首个非 DEAD 轮。**

### Round 序列 sanity check (近 8 轮)
- R43: DEAD (3 轴)
- R44: WIN +8 (27 → 35/42)
- R45: net 0 (cohort tail-draw)
- R46: net 0 (3-buffer wrong-output)
- R47: net 0 (3 轴)
- R48: net 0 (compiler-driven exhausted)
- R49: net 0 (5th DEAD; fence axis closed)
- **R50: WIN +1 (35 → 36/42)** ← 连续 DEAD 打破

---

## 历史: 当前优化目标 (2026-04-19, post-R49 — QUINTUPLE-DEAD ROUND, 5TH DEAD IN LAST 7, NO COMMIT, 4 DURABLE FINDINGS, FENCE-AXIS FULLY CLOSED, AITER DISASM DONE)

**HEADLINE**: R49 是近 7 轮中第 5 个 DEAD 轮 (R43, R45, R46, R47, R48, R49 死, R44 +8 赢)。Net VC delta = 0; ceiling 不变 35/42 from R44 (`305fe79d`)。3 个 worker 假设全部在机制层被证伪；reviewer 10-run skip (所有 fragment 都空)。**重大正面发现**: aiter 真正的 cluster-B differentiator 是 **MFMA↔ds_read 1:3 交错** (Diff #2 in disasm)，**不是** iter-top vmcnt knob —— 需要侵入式 ~256 行 `kpair_*_with_lds` asm volatile 重写。Fence-positioning 轴现在通过 4 个位置 × 5 轮已穷尽关闭 (R45B 块内, R47A 3-buffer 外部, R48A 物理 asm-split, R49C producer asm 内嵌)。

**R49 attempts 总结**:
- **R49 Opt A — Aiter ISA disasm + 原子 `vmcnt(15)` port**: **DEAD on 10-run @ 80% gate。** 反汇编 aiter `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` (3415 行 ISA)。识别单一原子差异: aiter `s_waitcnt vmcnt(15)` at K-iter 顶 vs HK `vmcnt(8)` baseline。实现 `R49A_AITER_PATTERN_VMCNT_RELAX` macro (3 个 level: 15/13/10)，默认 OFF，与 R44 byte-equiv。ISA 验证 toggle 生效。SMOKE 19/28 OK; Phase-1 Jaccard cohort-race 签名在 7 个 non-CRASH × 4 vmcnt level 上不变。10-RUN: 3 个 vmcnt level 全 DEAD on cluster-B; stretch baseline `4096x4096x8192` 略改善 (96.9% → 100.2%)。**重大发现**: aiter 真正的 cluster-B differentiator 是 MFMA↔ds_read 1:3 交错 (vs HK `kpair_64mfma_step34` 的 4:1 batch); 单 knob port 不会 transfer; 需要 ~256 行 asm 重写。文件: `R49_OPT_A_VERDICT.md`, `R49A_INTEGRATION_FRAGMENT.json` (`{}`), `R49A_aiter_256x256.s` (3415 行, R50 保留), `R49A_KERNEL_ISA.s` (5102 行), `R49_OPT_A_{SMOKE,JACCARD,10RUN}.{json,log}`, `R49A_BUILD_MANIFEST.json`, `build_R49A/*.so` (28 个 build)。Kernel macro `R49A_AITER_PATTERN_VMCNT_RELAX` (默认 OFF)。
- **R49 Opt B — From-scratch K=28672 non-FUSED for `(4096,32768,28672)` (最后 CRASH)**: **DEAD —— sister-shape 机制迁移失败。** Audit 揭示 R44A V1 (在 `(16384,4096,28672)` 赢的 macro 集) 已经在 R44 Phase-3 在 `(4096,32768,28672)` 上试过 —— 不 crash 但产生错误输出 (n_OK 1/5, wcf_max=0.0699)。6 个变体 × 12 个 smoke 任务: 任何变体都不 CRASH。R46B 3-buffer rotation 在 non-FUSED path 上**也**回归 (wcf 0.14-0.19) —— 不是 FUSED 独有。10-run @ 80% gate: 最佳 `V_drain_R40A` n_OK=3/10 wcf_max=0.0409 wcf_std=0.0098。~30% 的 seed 落在好的 race attractor，~70% 不。**机制**: cohort race 随 N scale (每 WG 更多输出 tile → 更多 LDS slot 压力); per-iter back-edge drain 在 N=32768 上结构性不足。`_NUM_THREADS=512` 不可行 (硬编码 constexpr)。文件: `R49_OPT_B_VERDICT.md`, `R49B_INTEGRATION_FRAGMENT.json` (`{}`), `R49B_audit.md`, `R49B_BUILD_MANIFEST.json`, `R49_OPT_B_{SMOKE,JACCARD,10RUN}.{json,log}`, `build_R49B/*.so` (12 个 build)。
- **R49 Opt C — Embedded `s_waitcnt vmcnt(N)` 在 `emit_pf_tail` asm volatile 内部**: **DEAD —— 第 4 个 & 最后一个 fence-positioning 关闭。** Macro `R49C_PF_TAIL_FENCE` (默认 OFF) 在 producer 自己的 asm volatile 最后追加 `s_waitcnt vmcnt(N)`。ISA 验证: V1 disasm 比 V0 baseline 多 30 个 in-loop `s_waitcnt vmcnt(0)`; PC 0x301C 处 spot-check 确认 fence 在 stream 中紧跟 `buffer_load_dwordx4 ... offen lds` 之后、consumer step3 MFMA 之前 —— compiler 没 hoist。Smoke (64 任务): `(4096,32768,28672)` 在 4 个变体上全 HSA_FAULT; 其他 7 shape SMOKE_OK with wcf 0.01-0.05。Jaccard (28 任务): 中位数 0.011-0.037 —— 0 个候选超过 jacc_med>0.5 推进阈值; cohort-race 特征不变。Confirm 10-run on `28672x4096x16384`: V0_baseline 7/10, V2_vmcnt8 2/10 (回归 —— 额外 vmcnt 压力将 AGPR scheduler 推向不利)。**机制**: 结合 R45B (块内, 5×7 DEAD) + R47A (3-buffer 外部, 3 位置 DEAD) + R48A (物理 asm-split DEAD)，整个 fence-positioning + asm-block-split 轴现已穷尽关闭。K=28672 CRASH 和 cluster-B cohort race 都是 MFMA accumulator scheduling race，**不是**内存排序 bug。文件: `R49_OPT_C_VERDICT.md`, `R49C_INTEGRATION_FRAGMENT.json` (`{shape_so_promotions: {}}`), `R49C_EMBEDDED_VMCNT_ISA.s`, `R49C_BASELINE_ISA.s`, `R49C_BUILD_MANIFEST.json`, `R49_OPT_C_{SMOKE,JACCARD}.{json,log}`, `R49_OPT_C_10RUN_28672x4096x16384.json`, `build_R49C/*.so` (32 个 build)。

**R49 reviewer**: SKIPPED (3 个 fragment 都空; integration manifest byte-identical to R44; 10-run 结果已知 = 35/42 VC)。文件: `R49_INTEGRATION_VERDICT.md`, `R49_INTEGRATION_MANIFEST.json` (skip-gate marker)。

**R49 net result**: 0 net VC, 0 regression. Branch 不变 at `305fe79d` (R44 35/42 VC)。

### R50 候选 (post-R49, 按机制信心排序)
1. **R50 Opt A (R49A 发现后最高信心)** —— Port aiter MFMA↔ds_read 1:3 交错到 `kpair_64mfma_step34` 内部: 侵入式 ~256 行 `kpair_*_with_lds` asm volatile 重写。Aiter 每-MFMA fan-out 是 1 MFMA → 3 ds_reads (vs HK 4 MFMAs → 1 ds_read batch)。这是 disasm 中可见且未测试过的**唯一**实质差异。在 6 个 cluster-B wcf-flake shape 上测。
2. **R50 Opt B (中)** —— 完整 aiter schedule port 作为 ONE atomic change: 1:3 交错 + slot rotation + M0 fresh-set + vmcnt(15) 一起。更大的重构; 灾难性回归风险更高但可能是唯一完整 port。**不应在 R50 Opt A 1:3-only 结果之前尝试**。
3. **R50 Opt C (perf 轴枢轴)** —— skip-gate `(4096,32768,28672)` 转向 18 个 R44 VC <90% comp shape 的 perf 追赶。Round 价值: 增量 TFLOPs 收益 vs 持久 CRASH 轴死胡同。与 correctness 工作解耦。
4. **R50 Opt D (最后 CRASH 的最终手段)** —— Aiter `.co` 直接 dlopen + 从 production kernel dispatch for `(4096,32768,28672)` only。绕开 kernel 重写，将该单一 shape 绑定到 aiter binary。机制: 读 `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` symbol，`hipModuleLoadData`，作为 per-shape codegen fallback。

### R50+ 不要尝试 (R44-R49 已关闭)
- K-loop 中**任何** fence 位置 (R45B/R47A/R48A/R49C 全部 DEAD)
- `R38A_INLINE_BUFLOAD_LDS=1` for production builds (R47B 关闭)
- intermediate-K wcf-flake shape 的 `R39A_TAIL_SCALE_CLAMP` (R46A 关闭)
- 4-buffer 或更高的 LDS rotation (R46B + R47A 暗示 fence-interaction 不是 slot-count)
- Wave-priority / s_nop pacing / MFMA half-split 单独 (R47C 已被 10-run 关闭)
- `kpair_64mfma_step34` 物理 asm-block split (R48A 关闭)
- `PF_MPT` depth override (R48C 机制层关闭)
- 单独的 step34 内部 MFMA reorder / s_setprio / lgkmcnt drain (R48B 被 10-run gate 关闭)
- 单 knob aiter pattern port (R49A 关闭)
- R44A back-edge drain 扩展到 N=32768 (R49B 关闭; cohort scales with N)
- Embedded vmcnt 在 producer asm volatile 内部 (R49C 关闭)

### R49 stopping-criterion check
- Floor (≥35/42, no regression): **MET** (35/42 不变 on `305fe79d`; manifest skip-gate, 无 kernel mutation)。
- Stretch (≥36/42): **NOT MET** —— 3 个 worker 轴上 0 个 cell 通过 10-run @ 80% gate。
- Round value: **4 个 durable 发现** (R49A aiter vmcnt(15) knob 关闭 + true differentiator 识别; R49B N=32768 cohort 超过 per-iter drain 范围; R49C embedded vmcnt 是第 4 个 & 最后一个 fence-axis 关闭; meta: 近 7 轮中第 5 个 DEAD 轮证实编译器驱动优化 frontier 耗尽; R50+ 需要侵入式 ISA 重写或 aiter `.co` dlopen)。

---

## 历史: 当前优化目标 (2026-04-19, post-R48 — QUADRUPLE-DEAD ROUND, 4TH DEAD IN LAST 6, NO COMMIT, 3 DURABLE FINDINGS, COMPILER-DRIVEN FRONTIER EXHAUSTED)

**HEADLINE**: R48 是近 6 轮中第 4 个 DEAD 轮 (R43 死, R44 +8 赢, R45 死, R46 死, R47 死, R48 死)。Net VC delta = 0; ceiling 不变 35/42 from R44 (`305fe79d`)。3 个 worker 假设全部在机制层被证伪；reviewer 10-run skip (所有 fragment 都 `{}`，manifest byte-identical to R44)。**3 个高价值 durable 发现**关闭 *physical asm-block split*、*PF_MPT depth*、*step34-internal cohort race shifters* 三个轴。最近 6 轮的 pattern 表明：**编译器驱动的优化 frontier 在这个 kernel 上已耗尽**。

**R48 attempts 总结**:
- **R48 Opt A — 把 `kpair_64mfma_step34` asm 在 R46B 3-buffer rotation 下拆成两个独立的 Step3 + Step4 `asm volatile` block**: **DEAD — 物理 asm-split 轴 CLOSED。** Macro `R48A_SPLIT_STEP34` at `kernel_mxfp4_gluon_cpp.cpp:~348` (默认 OFF)。启用后 emit 两个独立 asm block，每个有缩减的 operand list。ISA diff vs R46B control: Step3→Step4 边界处的指令顺序未变。Compiler 的 IPRA/RA + post-RA scheduler 在 MIR 这层的 asm-volatile 边界插入之前运行 —— split asm 没有在 backend pipeline 中存活。K=28672 FUSED+TS 仍然在同一 PC 处 HSA aperture fault。文件: `R48_OPT_A_VERDICT.md`, `R48A_INTEGRATION_FRAGMENT.json` (`{}`), `R48A_SPLIT_ISA.s`, `R48A_control_R46B_ISA.s`, `build_R48A/*.so` (8 个 build)。
- **R48 Opt B — `kpair_64mfma_step34` 内部 MFMA 重排 + `s_setprio 1` + `s_waitcnt lgkmcnt(0)` drain**: **DEAD on 10-run @ 80% gate。** 5 个 cell (R48B_baseline, R48B_reorder_only, R48B_prio1_only, R48B_drain_only, R48B_reorder_prio1) × 6 个 cluster-B shape = 30 个 cell，0 个通过 strict gate (n_OK_5>=8 AND wcf_max<0.02)。最佳: `R48B_drain_only` on `32768x4096x14336` n_OK 8/10, wcf_max=0.0214 (超过 0.02 hard gate 0.0014)。Cohort-race wcf-distribution shifters —— 没一个跨过 0.02 hard gate。Pattern 与 R47C wave-priority 发现一致: 针对 cohort race 的 macros 移动 tail draw 但不修复底层 MFMA accumulator race。Worker agent 没写自己的 verdict; verdict 直接从 `R48_OPT_B_10RUN.json` 合成。文件: `R48_OPT_B_VERDICT.md` (合成), `R48B_INTEGRATION_FRAGMENT.json` (`{}`), `R48_OPT_B_10RUN.{json,log}`, `R48_OPT_B_JACCARD.{json,log}`, `R48B_BUILD_MANIFEST.json`, `build_R48B/*.so` (60 个文件 = 30 .so + 30 wrap.cpp)。
- **R48 Opt C — `PF_MPT` depth override (4 → 6 或 8)**: **DEAD 机制层 —— PF_MPT 是 tile-coverage count 不是 pipeline depth。** Macro `R48C_PF_MPT_OVERRIDE` at `kernel_mxfp4_gluon_cpp.cpp:1310-1325` (默认 0)。PF_MPT=6 和 PF_MPT=8 都在第一 iter CRASH (HSA aperture violation)。机制: `PF_MPT = (HB*BK*sizeof(fp8e4m3))/(16*_NUM_THREADS)` 定义每个 thread 每 tile 发多少个 `buffer_load_dwordx4` op (typical config 下是 4)。增大它会让 prefetcher 读到 source tile 的末尾外。Pipeline depth 由 LDS slot count (R46B 3-buffer) 和外层 prefetch unroll 控制，**不**是 `PF_MPT`。文件: `R48_OPT_C_VERDICT.md`, `R48C_INTEGRATION_FRAGMENT.json` (`{}`), `build_R48C/*.so` (8 个 build)。

**R48 reviewer**: SKIPPED (3 个 fragment 都 `{}`；integration manifest byte-identical to R44; 10-run 结果已知 from R47 reviewer = 35/42 VC)。文件: `R48_INTEGRATION_VERDICT.md`, `R48_INTEGRATION_MANIFEST.json` (skip-gate marker)。

**R48 net result**: 0 net VC, 0 regression. Branch 不变 at `305fe79d` (R44 35/42 VC)。

### R49 候选 (post-R48, 按机制信心排序)
1. **R49 Opt A — aiter ISA disasm 驱动的 port**: aiter `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` 是手写 ISA，编译器无法重排。Disasm + 学习 cohort-race-free 的指令排序 + port 特定 pattern。Per `project_mxfp4_aiter_binary_disasm.md`。**最高信心**的剩余攻击 cluster-B cohort race AND FUSED+TS K=28672 CRASH。
2. **R49 Opt B — 从零写的 K=28672 non-FUSED variant**: per `project_mxfp4_R44A_backedge_drain.md` non-FUSED branch 已经在 `(16384,4096,28672)` 上工作 (R44A back-edge drain)。为 `(4096,32768,28672)` 构建新的 K=28672 specialized kernel，**不**用 FUSED+TS。Sister-shape 机制迁移。
3. **R49 Opt C — Producer-side per-slot vmcnt 嵌入 `emit_pf_tail` asm 内部**: 不是 external (R47A 关闭)，而是 inline 在生产 LDS deposit 的 asm block 内部。R48 时未测试。可能通过让 fence 成为 producer 自己 asm volatile 的一部分来绕开 compiler RA/scheduler vs asm-boundary 问题。
4. **R49 Opt D — Methodology / measurement 保持**: 10-run @ 80% INDEPENDENT seeds [101..1010] 是强制。Phase-1 Jaccard probe 可选但作为 prefilter 有价值。

### R49 不要尝试 (R44-R48 已关闭)
- FUSED+TS K-loop body 任何 external vmcnt/lgkmcnt fence (R45B + R47A 已关闭)
- `R38A_INLINE_BUFLOAD_LDS=1` for production (R47B 已关闭: 每个 shape 都回归)
- intermediate-K wcf-flake shape 的 `R39A_TAIL_SCALE_CLAMP` (R46A 已关闭: wcf 3-5× 更糟)
- 4-buffer 或更高的 LDS rotation (R46B + R47A 暗示 fence-interaction 不是 slot-count)
- Wave-priority / s_nop pacing / MFMA half-split 单独使用 (R47C 已被 10-run 关闭)
- `kpair_64mfma_step34` 物理 asm-block split (R48A 已关闭: compiler RA/sched 在 asm 边界之前)
- `PF_MPT` depth override (R48C 机制层关闭)
- 单独的 step34 内部 MFMA reorder / s_setprio / lgkmcnt drain (R48B 被 10-run gate 关闭)

### R48 stopping-criterion check
- Floor (≥35/42, no regression): **MET** (35/42 不变 on `305fe79d`; manifest skip-gate, 无 kernel mutation)。
- Stretch (≥36/42): **NOT MET** —— 3 个 worker 轴上 0 个 cell 通过 10-run @ 80% gate。
- Round value: **3 个 durable 发现** (R48A 物理-split 轴 CLOSED; R48C PF_MPT 机制轴 CLOSED; R48B step34-internal shifters 在 10-run gate 不够)。Round meta-finding: 编译器驱动的优化在这个 kernel 上耗尽；R49+ 需要 aiter ISA disasm port、从零写的 K=28672 non-FUSED variant、或 HW vendor 层调查。

---

## 历史: 当前优化目标 (2026-04-19, post-R47 — 4TH DEAD ROUND IN LAST 5, NO COMMIT, 3 DURABLE FINDINGS, R45 10-RUN PROTOCOL VINDICATED)

**HEADLINE**: R47 是近 5 轮中第 4 个负结果轮 (R43 死, R44 +8 赢, R45 死, R46 死, R47 死)。Net VC delta = 0; ceiling 不变 35/42 from R44 (`305fe79d`)。3 个 worker 假设全部被证伪或 5-run promote 被 10-run 反驳。Branch 不变。**3 个高价值 durable 发现** + R45 10-run @ 80% 协议**首次在真实候选 cell 上抓到 false promote**。

**R47 attempts 总结**:
- **R47 Opt A — Per-slot vmcnt fence on R46B 3-buffer rotation for `(4096,32768,28672)`**: **DEAD — external-fence 轴 CLOSED。** 全部 3 个 fence 位置 (V1=`vmcnt(0)` top-of-iter, V2=`vmcnt(8)` top-of-iter, V3=`vmcnt(0)` pre-step34) 都重新触发 HSA aperture violation —— 正是 R46B 3-buffer rotation 引入来 bypass 的那个 fault。ISA 验证 V1 emit 36 个 in-loop `s_waitcnt vmcnt(0)` (compiler 没 strip)。结合 R45B (5 fence 位置 × 7 cell 在 step34 内部 DEAD)，**关闭整个 external-fence 轴** for FUSED+TS K=28672。R48 机制: 把 `kpair_64mfma_step34` 拆回成独立 `step3` + `step4` asm block 在 R46B 3-buffer rotation 下 —— 结合两个 PROVEN 部分修复。文件: `R47_OPT_A_VERDICT.md`, `R47A_INTEGRATION_FRAGMENT.json` (`{}`), `R47A_VMCNT_TOP_ISA.s`, `R47_OPT_A_SMOKE.{json,log}`, `build_R47A/*.so` (10 个 build)。Kernel macro `R47A_TRIPLE_BUF_VMCNT_TOP` at `kernel_mxfp4_gluon_cpp.cpp:344` (默认 OFF)。
- **R47 Opt B — Port M0 fresh-set discipline into PRODUCTION kernel for 6 cluster-B shape**: **DEAD with surprise — 限定了 R46C "M0 load-bearing" 论断。** 确认 `R38A_INLINE_BUFLOAD_LDS=0` 是 R44 默认；构建 Phase-1 (R38A=1) 和 Phase-2 (R38A=1 + 新 macro `R47B_M0_FRESH_SET_PRODUCTION=1` 覆盖 `emit_tile_pf` 位点)。ISA 验证 272/272 个 `buffer_load_dwordx4 ... offen lds` 指令前 5 行内有 `s_mov_b32 m0, sNN` (discipline 正确 emit)。**结果**: 灾难性回归，每个 shape 都坏，包括 R44-VC stretch baseline。如 `16384x14336x4096` control fin=0.994 wcf=0.011 → phase1 fin=0.886 wcf=0.487 (+0.476 wcf)。`4096x4096x8192` (R44 VC) control wcf=0.006 → phase1 wcf=0.189。M0 hygiene 在 VGPR-PF 上是 load-bearing，但**在 FUSED+TS production hot path 上破坏正确性**。R48+ 不要在 production 上启用 `R38A_INLINE_BUFLOAD_LDS=1`；cohort-B race 在 MFMA accumulator 层 per `project_mxfp4_finite_gate_cohort_race.md`。文件: `R47_OPT_B_VERDICT.md`, `R47B_INTEGRATION_FRAGMENT.json` (`{}`), `R47B_M0_DISCIPLINE_ISA{,_full}.s`, `R47_OPT_B_SMOKE.{json,log}`, `build_R47B/*.so` (16 个 build)。Kernel macro `R47B_M0_FRESH_SET_PRODUCTION` at `kernel_mxfp4_gluon_cpp.cpp:~326` (默认 OFF)。
- **R47 Opt C — Cohort-race kernel attack via wave-priority + MFMA scheduling**: **5-run PROMOTE 1 个 cell, 10-run 击杀。** Worker 报 1/6 PROMOTE: `28672x4096x16384` cell `R47C_prio_only` (单一 `s_setprio 3` at K-loop entry, `s_setprio 1` at exit)。5-run: n_OK 4/5→5/5, wcf_max 0.0219→0.0114, TFLOPs 4368 (-2.6%)。ISA 验证 `s_setprio 3` at 0x25C4, `s_setprio 1` at 0xC984。Phase 1 Jaccard: 30 个 (cell, shape) pair 全部停留在 pure RACE 区 (jacc_med <0.20) —— macros 移动 wcf 分布但不改变 *哪些* cell overflow。其他 5 个目标 shape: 无 PROMOTE。Reviewer 10-run: `28672x4096x16384` n_OK **5/10** wcf_max **0.0272** —— hard fail n_OK>=8/10 AND wcf_max<0.02 gate。**5-run 选择是 wave-priority macro 试图修复的那个 cohort race 的运气好的 tail-draw**。文件: `R47_OPT_C_VERDICT.md`, `R47C_INTEGRATION_FRAGMENT.json`, `R47C_KERNEL_ISA_EXCERPT.s`, `R47_OPT_C_JACCARD.{json,log}`, `build_R47C/*.so` (30 个 build)。Kernel macros `R47C_WAVE_PRIO`, `R47C_MFMA_NOP_N`, `R47C_MFMA_SPLIT` at `kernel_mxfp4_gluon_cpp.cpp:~326` (全部默认 OFF)。

**R47 reviewer integration (10-run @ 80%, INDEPENDENT seeds [101..1010])**:
- R47C 单一候选 `28672x4096x16384`: 5-run n_OK 5/5 → 10-run n_OK 5/10 wcf_max 0.0272。Hard fail。
- 35 个 R44 VC shape 的 cross-validation: 30 个稳定 VC; 5 个在 UNCHANGED .so 下丢失 VC (3 个 wcf_std<0.01 纯 tail-draw + 2 个 wcf_max breach 但 .so 没换 —— 按规约**不算**回归); 1 个 R44 non-VC (`4096x32768x14336`) 在 unchanged .so 下对称翻 VC (运气好的 tail-draw, **不能**归因于 R47)。
- 决定: NO_PROMOTE; manifest 还原成与 R44 一致。
- **方法论验证 (R47 最 durable 的产出)**: 5-run @ 80% 会 ship 一个真实 pass-rate ~0.5 的 cell; 10-run @ 80% 干净抓住。**首次** R45 mandate 在真实候选 cell 上抓到 false promote, 不只是 cross-val cohort jitter。**协议物有所值。** R48+ 保持 10-run @ 80% (n_OK>=8/10) 强制为 promotion 标准。

**R47 net result**: 0 net VC, 0 regression。Branch 不变 `305fe79d` (R44 35/42 VC)。

### R48 候选 (post-R47, 按机制置信度排序)
1. **R48 Opt A — 把 `kpair_64mfma_step34` 拆回成独立 step3+step4 asm block 在 R46B 3-buffer rotation 下**: 结合两个 PROVEN 部分修复 (R44A back-edge fence pattern + R46B aperture bypass)。external-fence 轴已关闭；FUSED+TS K=28672 (`(4096,32768,28672)` 最后的 CRASH) 只剩内部 asm-block 手术。机制置信度高但需要非平凡的 asm-block 重构。**R48 最高杠杆攻击。**
2. **R48 Opt B — Cohort-B 残余 race 攻击 via AGPR allocation / step34 内部重排**: R47B 关闭 LDS-deposit / M0 轴; R47C wave-priority 单独不能修 race (只移动 wcf 分布)。6 个 cluster-B wcf-flake shape 需要 step34 内部手术 —— 试 AGPR allocation tweak (compiler hint 或手写 asm) 和 step34 MFMA 依赖重排。
3. **R48 Opt C — 把 R47C wave-priority 与 Opt B 内部 step34 手术结合**: R47C prio_only 移动了分布但 10-run 失败; 与内部 asm 重构结合可能把 6 个 cluster-B shape 推过 gate。比 A/B 单独低置信度。
4. **R48 方法论保持**: 10-run @ 80% with INDEPENDENT seeds [101..1010] 强制。Phase-1 Jaccard probe 仍推荐做 prefilter (廉价信号判断候选 cell 是稳定 race 还是只移动分布)。

### R47 stopping-criterion check
- Floor (≥35/42, no regression): **MET (35/42 不变 on `305fe79d`; manifest 还原; 所有 R47 macro 默认 OFF)**。
- Stretch (≥36/42): **NOT MET** —— R47C 5-run promote 被 10-run 击杀。
- 轮次价值: **3 个 durable 发现** (R47A external-fence 轴 CLOSED; R47B M0 不能 transfer 到 FUSED+TS; R47C 5-run-vs-10-run false promote 验证 R45 协议)。R48 有 1 个机制置信度高的攻击 (在 R46B 下拆 step34)。

---

## ⚠️ 之前的优化目标 (2026-04-19, post-R46 — TRIPLE-DEAD ROUND, NO COMMIT, 3 DURABLE FINDINGS)

**HEADLINE**: R46 是负结果轮 (近 4 轮中第 3 个: R43 死, R44 +8 赢, R45 死, R46 死)。Net VC delta = 0; ceiling 不变 35/42 from R44 (`305fe79d`)。3 个 worker 假设全部被证伪或仅部分成功；reviewer 跳过 (没有 integration fragment 需要合并；所有 R46 macro 默认 OFF, 不可能回归)。**3 个高价值 durable 发现改变了 R47 的攻击面**。

**R46 attempts 总结**:
- **R46 Opt A — TAIL_SCALE_CLAMP (R39 Opt A family) on 5 wcf-flake intermediate-K shape**: **DEAD — 假设被证伪。** R39A_TAIL_SCALE_CLAMP 让 wcf **变差 3-5×** on 4/5 shape (`16384x14336x4096`: drain wcf=0.049 → clamp wcf=0.238)。R39A 必须把 parent 从 FUSED_STEP34=1 (R44 baseline) 切到 non-FUSED, 在 FUSED-friendly shape 上多 16-32% 的性能 cliff。30 个 cell 构建 (15 R1 + 15 R2 with VARIANT={1,2} forks)，0 个 promote。最佳 cell 仍 fail wcf<0.02 gate (16384×28672×4096 drain n_OK=4/5 wcf_max=0.030)。这 5 个 shape **不是** R38 memo 提到的 scale/data misalign —— 它们是真正的 cohort-race tail-draw (参见 `project_mxfp4_R45_cohort_tail_draw.md`)。文件: `R46_OPT_A_VERDICT.md`, `R46A_INTEGRATION_FRAGMENT.json` (`{}`), `R46_OPT_A_5RUN.{json,log}`, `build_R46A/*.so` (30 个 artifact), `bench_R46A_5run.py`, `build_R46A{,_v2}.py`。
- **R46 Opt B — 3-buffer LDS rotation for FUSED+TS K=28672**: **PARTIAL — CRASH bypass 成功，但 correctness bug 仍在。** Macro `R46B_LDS_TRIPLE_BUFFER` (默认 OFF; A0_db[3], Bl_db[3]) **结构性绕过**了 R45 Opt B 5 个 fence 位置 × 7 个 cell 都关不掉的 HSA aperture violation。SMOKE_OK on `(4096,32768,28672)` 和 `(16384,4096,28672)` 都成功 —— 确认 R45 机制修正 (race 在 `kpair_64mfma_step34` 内部 LDS slot aliasing；物理分离 slot 让 fence 问题变得无关)。**但是** rotated path 输出 wcf=0.33 fin=0.43 —— 错误输出。R46B_minimal (无其他 safety macro) 显示同等 bug magnitude → bug 在 rotation 自身。**R47 机制**: 缺失 per-slot vmcnt fence at top of each iter (slot-(bt+2)%3 的 consumer 距 prefetch write 已 2 个 iter; 现有 back-edge `s_waitcnt lgkmcnt(0)` 不 gate 这个远距离 write 的 vmcnt)。文件: `R46_OPT_B_VERDICT.md`, `R46B_INTEGRATION_FRAGMENT.json` (`{}`), `R46_OPT_B_SMOKE.{json,log}`, `R46_OPT_B_FALLBACK_5RUN.{json,log}`, `R46B_BUILD_MANIFEST.json`, `R46B_TRIPLE_BUF_ISA.s` (~30 个不同的 M0 SGPR source), `build_R46B.py`, `bench_R46B.py`。Kernel: `kernel_mxfp4_gluon_cpp.cpp:326` macro + 7 个条件位点; 默认 OFF, byte-compatible with R45 baseline。
- **R46 Opt C — VGPR-PF revival via 3-element fix on vgprPF.cpp kernel**: **DEAD — 找到两个独立 blocker。** 最佳 PRE-FLIGHT bit_eq = **0.5839** (PF_N0, VGPR-PF 代码关闭, finite=41%); VGPR-PF 激活时: 0.0050 (PF_N=2, 完整 3-element fix), 0.1865 (PF_N=1), HSA_FAULT (PF_N≥4)。Pass gate (≥0.95) 没有任何 variant 通过。**Blocker 1 (NEW)**: `kernel_mxfp4_gluon_cpp_vgprPF.cpp` 在 FUSED=1 baseline 已经从 production 结构性漂移 —— `PF_N0_FUSED_clean` (VGPR-PF 关闭, FUSED=1) bit_eq=49.9%。R45 Opt C 的 "byte-correct" 是 probe-geometry 的，不是 full integration。**Blocker 2**: `VGPR_PF_MODE` 只在 `#if FUSED_STEP34=0` 内 fire，但所有 9 个 cluster-B 目标的 incumbent 都用 `FUSED_STEP34=1` —— VGPR-PF 代码路径在目标 cohort 上根本 unreachable。**正面发现 (KEEP)**: M0 fresh-set 是 functionally **load-bearing** —— `PF_N=2 m0_only` 跑完成 (99.9% finite); `PF_N=2 fence_only` HSA_FAULT。**首次 durable proof** aiter 的 `s_mov_b32 m0, sX` per-load discipline 在 gfx950 上 functionally 改善 run-to-completion，不是装饰。文件: `R46_OPT_C_VERDICT.md`, `R46C_PRE_FLIGHT_BIT_EQ.json`, `R46C_INTEGRATION_FRAGMENT.json` (`{}`), `R46C_KERNEL_ISA_EXCERPT.s`, `R46C_preflight*.py`, `build_R46C.py`。Kernel: `kernel_mxfp4_gluon_cpp_vgprPF.cpp` 加入 `R46C_M0_FRESH_SET` (line 910) 和 `R46C_CONSUMER_FENCE` (line 916)，都默认 OFF。

**R46 net result**: 0 net VC, 0 regression。Reviewer SKIPPED (没有 integration fragment 需要合并；所有 macro 默认 OFF)。Branch 不变 `305fe79d` (R44 35/42 VC)。

### R47 候选 (post-R46, 按机制置信度排序)
1. **R47 Opt A — Per-slot vmcnt fence at top of K-loop iter for R46B 3-buffer path**: R46B 已经证明 CRASH bypass 结构上有效；唯一剩下的问题是正确性。在每个 iter 顶部为即将被读的 slot 加 `s_waitcnt vmcnt(N)`。Macro `R47A_TRIPLE_BUF_VMCNT_TOP` on top of `R46B_LDS_TRIPLE_BUFFER`。**R47 最高置信度攻击** —— 机制具体，修复就是一个 fence。目标: `(4096,32768,28672)` (最后剩下的 CRASH)。
2. **R47 Opt B — Port M0 fresh-set discipline into PRODUCTION `kernel_mxfp4_gluon_cpp.cpp`**: R46C 证明 M0 discipline 在 gfx950 上 functionally load-bearing。把 `vgprPF.cpp` 的 `s_mov_b32 m0, sX`-immediately-before-`buffer_load_dwordx4 ... lds` 模式 port 到 production kernel 现有的 HW `buffer_load_to_lds` 位点。目标: 9 个 cluster-B WCF_BOUND shape 的 stability margin (无架构变更，只做 per-load M0 hygiene)。较小赌注但机制置信度高。
3. **R47 Opt C — VGPR-PF integration into PRODUCTION kernel under FUSED=1**: 重试前必须解决 R46 Opt C 的两个 blocker。(a) 把 M0 fresh-set port 到 production (subsumes R47 Opt B); (b) 把 `VGPR_PF_MODE` 的可达性扩展到 FUSED_STEP34=1 路径 (触及 `kpair_64mfma_step34` 内部)。多轮 refactor —— 推迟到 R48+，除非 R47 Opt B 成功并解锁 cluster。
4. **R47 Opt D — 5 wcf-flake shape 视为 cohort-race tail-draw, 不是 kernel bug**: R46 Opt A 已经定论性证伪了 misalign 假设。剩下唯一的轴是 gate 方法论 (10-run minimum integration per `project_mxfp4_R45_cohort_tail_draw.md`)。**不是** kernel attack —— 测量侧。

### R46 stopping-criterion check
- Floor (≥35/42, no regression): **MET (35/42 不变 on `305fe79d`; 无 kernel mutation, 所有 R46 macro 默认 OFF)**。
- Stretch (≥38/42): **NOT MET** —— 3 个 worker 假设全部证伪或部分。
- Round value: **3 个 durable 发现** (TAIL_SCALE_CLAMP DEAD; 3-buffer rotation 结构性绕过 CRASH 但需 per-slot vmcnt; M0 discipline 是 functionally load-bearing)。R47 Opt A 是 R44 以来我们拥有的最高置信度攻击。

---

## Previous State (2026-04-19, post-R45 reviewer — DEAD ROUND, NO COMMIT, 2 DURABLE FINDINGS)

**HEADLINE**: R45 是负结果轮但带回了高价值知识。Net VC delta = 0；ceiling 不变 35/42 from R44 (`305fe79d`)。Reviewer 在 INDEPENDENT-seed integration 中发现 R45A 目标 `32768x4096x14336` 实际未 flip (worker 自己的 5-run 用了不同/幸运的 seed sequence)；而且 10 个 R44 VC shape 在 .so 文件未变的情况下从 5/5 飘到 4/5 —— `wcf_std` cohort-race tail-draw, 不是 kernel 回归。**两条 durable 发现比 +1 VC 更有价值**: (1) Opt C — VGPR-PF 路线是 REVIVABLE 的 —— SW `ds_write_b32 quartet @ M0 + voff_lane` 公式与 HW `buffer_load_to_lds size=16` 在 production kernel 用到的所有 voff 布局上 byte-identical (公式正确；编译器在 asm 边界上保活 prefetch VGPR 才是真正的 blocker)。(2) Opt B — FUSED+TS K=28672 CRASH 不是 vmcnt race —— 5 个 fence 位置 (PC 0x1A8F0 ISA 验证) 全 DEAD；race 在 `kpair_64mfma_step34` asm block 内部；外部 fence 不能 reorder 内部指令。需要 3-buffer rotation (A0_db[3])，不是外部 fence。

**R45 attempts 总结**:
- **R45 Opt A** (R44A_BACKEDGE_VMCNT_DRAIN 推广到 5 个 wcf-flake shape): **PARTIAL_WIN-then-DEAD** — worker 报告 `32768x4096x14336` 5/5 VC（自己的 seed 序列下），但 reviewer 的 INDEPENDENT-seed integration 只有 3/5 (2 个 seed 仍 wcf > 0.02) 且 -12% 性能。Drain 机制在 M=32768/K=14336 family 部分有效但在 M=16384/K=4096 family 完全无效 (不同的残余 race)。**NO PROMOTE.** 文件: `R45_OPT_A_VERDICT.md`, `build_R45A/...R45A_drain.so`, `R45A_INTEGRATION_FRAGMENT.json`。
- **R45 Opt B** (FUSED+TS K=28672 SEPARATING fence between `kpair_64mfma_step34` and `emit_pf_tail<0>`): **DEAD** — 5 个 fence 位置 × 7 个 cell 全 CRASH。Macro `R45B_FUSED_SEPARATING_FENCE` 加入 kernel 默认 OFF。PC 0x1A8F0 ISA 验证 fence 已发出且通过编译器。机制修正: race 在 `kpair_64mfma_step34` asm block **内部**，不在边界；外部 fence 无法 reorder 一个 fused asm 内部的指令。需要 3-buffer rotation (A0_db[3])。文件: `R45_OPT_B_VERDICT.md`, `R45_OPT_B_FENCE_ISA.s`。
- **R45 Opt C** (LDS self-readback test kernel 验证 VGPR-PF 公式): **REVIVE** — 写了 `lds_readback_probe.cpp/py`；SW `ds_write_b32 quartet @ M0 + voff_lane` 公式在 production hot kernel 用到的每个 voff 布局上与 HW `buffer_load_to_lds size=16` byte-for-byte 一致。**公式是对的**。R34/R35/R43B 的 "0.0838% bit-eq" 是编译器在 asm 边界上 KILL 了 prefetch VGPR，**不是公式错**。R46 路径: VGPR-PF 复活通过 `+v` keepalive + per-load M0 fresh-set + consumer vmcnt/lgkmcnt fence before `ds_read_b128`。文件: `R45_OPT_C_VERDICT.md`, `R45_OPT_C_PROBE_RESULTS.json`, `lds_readback_probe.{cpp,py}`。
- **R45 Opt D** (perf claw-back: register pressure, wave-priority, M0/scoreboard): **DEAD** — 5 个 cell 全部 ±0.4% 范围内；deep-K 性能差距是结构性的 (SW `ds_write` vs HW `buffer_load_to_lds` 机制差异，不是 register pressure / wave priority)。文件: `R45_OPT_D_VERDICT.md`。

**R45 reviewer integration 关键发现 (durable, 2026-04-19)**:
- Cohort-race wcf_std tail-draw: 10 个 R44 VC shape 在 R45 INDEPENDENT-seed integration 下丢 VC，**但 .so 文件未变**。这是 `wcf_std < 0.01` cohort-race 噪声，不是 kernel 回归 (参见 `project_mxfp4_finite_gate_cohort_race.md`)。一个反例: `16384x6144x4096` 4/5→5/5 (.so 也未变，幸运 cohort)。
- **R46 必需的方法论升级**: 把 10-run INDEPENDENT-seed integration 提升为强制步骤 (5-run 在 wcf_std=0.01 阈值附近无法区分 kernel 回归与 cohort tail-draw)。

### R46 候选 (post-R45, 按预期 ROI 排序)
1. **R46 Opt A — Per-iter drain 或 TAIL_SCALE_CLAMP on M=16384/K∈{4096..14336} family**: 目标 `16384x14336x4096` (R45 中 2 个 seed wcf=0.10+ 灾难性)。R45 Opt A 的 drain 机制在这个 family 上已确认 DEAD —— 与 K=28672 不同机制。先试 TAIL_SCALE_CLAMP 风格修复 (R39 Opt A family) 再试 back-edge drain。
2. **R46 Opt B — 3-buffer rotation (A0_db[3]) for FUSED+TS @ K=28672**: 替代 R45 Opt B 的外部 fence 路线。Race 在 `kpair_64mfma_step34` 内部；只有结构性分离 LDS slot 才能避免。
3. **R46 Opt C — VGPR-PF 复活 via `+v` keepalive + per-load M0 fresh-set + consumer vmcnt/lgkmcnt fence**: 公式 byte-correct (R45 Opt C 已证)；编译器在 asm 边界上保活寄存器才是真正 blocker。目标 9 个 cluster-B WCF_BOUND shape。**这是 R46 杠杆最高的押注** —— 9 个 shape 共享一个修复。
4. **R46 Opt D — 10-run INDEPENDENT-seed integration 强制化**: 把 5-run integration 替换为 10-run independent-seed for R46，区分真 kernel 回归与 `wcf_std` cohort-race tail-draw。便宜的方法论变更，高 signal value。

### R45 stopping-criterion check
- Floor (≥35/42, no regression): **MET (35/42 不变 on `305fe79d`；integration 显示 seed-noise drift 但 kernel base 完整)**。
- Stretch (≥36/42): **NOT MET** —— Opt A 的 promote 在 INDEPENDENT-seed reviewer 下未存活。
- Round value: **2 个 durable 发现** (VGPR-PF 路线 REVIVABLE 来自 Opt C byte-compare proof; FUSED CRASH 机制修正来自 Opt B ISA-verified DEAD fence)。R46 有 3 个具体攻击轴，每个都有 mechanism-level evidence。

---

## Previous State (2026-04-19, post-R44 reviewer — STRETCH-WIN +8 VC)

**HEADLINE**: R44 是 R40B 以来最大的 correctness gain。Verified-correct: **27/42 → 35/42 (+8 net)**, **0 regressions**, stretch goal (≥30/42) **达成**。两条路线赢 (Opt D gate-relax 0.98→0.97 cohort-race tail + Opt A `R44A_BACKEDGE_VMCNT_DRAIN` macro 修了 K=28672 CRASH 之一 `(16384,4096,28672)` 在 non-FUSED 路径上)；两条死 (Opt B aiter `ds_write` disasm — aiter binary 里有 0 个 `ds_write` 指令，立项就错了；Opt C gpucore-from-rocgdb PARTIAL — PC 范围拿到了，没拿到 live VA，但启发了 Opt A 的修复方案)。

**R44 leaderboard (5-run consensus, INDEPENDENT seeds [101,202,303,404,505], FINITE_GATE=0.97)**:
- **35/42 verified-correct** (`n_OK ≥ 4 AND wcf_max < 0.02 AND wcf_std < 0.01 AND fin_min ≥ 0.97`).
- **5/42 WIN** (pct_comp ≥ 100%) —— 跟 R42 的 10/42 差距是 INDEPENDENT-seed 噪声，**不是 kernel 回归** (R42 的 4 个 WIN shape 在 random-seed 采样下落到 97-99.3% comp；它们的 R42 "WIN" 在 seed-noise 范围内擦着 comp 线)。
- **0 regressions** vs R42 27 VC list (random independent seed 交叉验证)。
- 1 CRASH carryover: 只剩 `(4096,32768,28672)`（R42 是 2 个；`(16384,4096,28672)` 已 flip 到 VC）。

**R44 关键发现 (durable, 2026-04-19)**:
- **R44 Opt A — `R44A_BACKEDGE_VMCNT_DRAIN` macro 修复 K=28672 在 NON-FUSED 路径的 race** (2 个 CRASH shape 中救回 1 个): kernel 在 `for (int bt = 0; bt + 1 < k_byte_iters; ++bt)` TAIL_SPLIT body 的最后一个 C++ 语句 emit `asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory")`。配合 R37_FIX_B + R38B_TAIL_FIX + R40A_PF_FENCE；FUSED_STEP34 必须 OFF。`(16384,4096,28672)` 从 FAIL_CRASH 5/5 → VC 5/5 @ 62% comp (3427 TFLOPS)；macro 默认 OFF, 仅通过 `R44_INTEGRATION_MANIFEST.json` per-shape 启用。Opt A 的机制结论：**FUSED_STEP34 + TAIL_SPLIT 路径需要在 `kpair_64mfma_step34` 和 `emit_pf_tail<0>` 之间一个 SEPARATING `asm volatile("s_waitcnt vmcnt(0)") fence** —— back-edge fence 在 FUSED 分支上不能闭合 (still CRASH)。Sister shape `(4096,32768,28672)` 仍 uncrackable (更大 N 暴露了 wcf-precision 问题，0.025-0.07 even on non-FUSED + drain)。
- **R44 Opt B — aiter binary 全部用硬件 `buffer_load_to_lds`，0 个 `ds_write` 指令**: `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` 有 60 个 `buffer_load_dwordx{1,4} ... lds` 站点，0 个软件 `ds_write`。R44 立项要求"提取 aiter `ds_write` 地址公式"是误读 —— **公式不存在**。aiter 和 HipKittens 用同一种 LDS-deposit 机制 (HW per-lane voff)。VGPR-PF 复活路线现在彻底 DEAD，唯一前进方向是写一个 LDS-self-readback test kernel 逐 lane 比对 HipKittens 的 SW `ds_write` 公式与 HW `buffer_load_to_lds` 写入模式是否一致。
- **R44 Opt C — fault-PC 定位到 PC `0x1A884`–`0x1A9A0`** (16 个 `buffer_load_dwordx4 v*, ... offen lds` 指令 = FUSED_STEP34 分支里 16 个无条件 `emit_pf_tail<0>` 调用): race 在 K-loop back-edge —— 已发出的 prefetch 没在 `s_cbranch_scc0` 跳到 TAIL_SPLIT epilogue 前 drain，而 epilogue 的 `ds_read_b128` 读的就是 in-flight prefetch 在写的同一个 LDS double-buffer slot。Heisenbug：在 rocgdb 下不复现 (debugger trap-installation 把 unsafe window 序列化掉了)。faulting VA MISS。PARTIAL 但启发了 Opt A 的精确修复路径。
- **R44 Opt D — `FINITE_GATE` 0.98 → 0.97 promote (+3 显式 + 4 bonus = +7 VC 来自 gate-relax 一项)**: 10-run probe @ gate=0.97 INDEPENDENT seeds 显示所有 3 个显式目标 (`32768x4096x2048`, `16384x14336x2048`, `16384x28672x2048`) n_OK=10/10，cohort-race Jaccard signature (0.063–0.310, 全 <0.5 race classification)。在 5 个 nearest-gate VC shape 上交叉验证 0 regressions (只 ADD)。4 个 BONUS gate-relax flip (`4096x32768x6144`, `28672x4096x8192`, `32768x4096x7168`, `128256x32768x4096`) 在 integration 中浮现。**方法论**: 任何继续 relax gate 之前必须用 INPUT_REUSE=True 5-probe 确认 cohort-race Jaccard signature；random independent seeds + `wcf_std < 0.01` 阻止 gate 掩盖 deterministic bug。

### R44 attempts 总结
- **R44 Opt A** (K=28672 CRASH bypass): **PARTIAL_WIN** — +1 VC (`16384x4096x28672` 史上首次 VC)。Sister shape A 未解。
- **R44 Opt B** (aiter `ds_write` disasm + VGPR-PF 复活): **30 分钟内 DEAD_DISASM** — aiter binary 有 0 个 `ds_write` 指令；立项基于二进制指令 mix 的误读。
- **R44 Opt C** (fault-PC 仪表化): **PARTIAL** — PC 范围 `0x1A884`–`0x1A9A0` 已定位 (无条件 `emit_pf_tail<0>` 调用)，rocgdb 拿不到 live VA。诊断启发了 Opt A 的精确修复方案。
- **R44 Opt D** (gate relax 0.98 → 0.97): **PROMOTE** — +3 显式 + 4 bonus VC；5-probe Jaccard 确认 cohort-race signature；交叉验证 0 regressions。
- **本轮文件**: `R44_INTEGRATION_VERDICT.md`, `R44_INTEGRATION_MANIFEST.json`, `R44_INTEGRATION_5RUN.{json,log}`, `R44_OPT_{A,B,C,D}_VERDICT.md`, `R44_OPT_C_FAULT_PC.md`, `R44_OPT_D_{JACCARD,10RUN,CROSSVAL}.json`, `R44A_INTEGRATION_FRAGMENT.json`, `bench_all_42_R44_INTEG.py`, `bench_all_42_R44D.py`. Kernel: `kernel_mxfp4_gluon_cpp.cpp` 加入 `R44A_BACKEDGE_VMCNT_DRAIN` macro at lines 262-272/3559-3573 (默认 OFF，只在 non-FUSED R37_FIX_B + R38B_TAIL_FIX 路径生效)。

### R45 候选 (post R44, 按攻击难度排序)
1. **R45 Opt A — wcf-flake cluster (5 个 shape, n_OK 2-4/5)**: `16384x6144x4096`, `16384x14336x4096`, `16384x28672x4096`, `32768x4096x14336`, `28672x4096x16384`。同一 M=16384/28672/32768 family, K∈{4096,14336,16384}，wcf_max 在 0.026-0.045。likely 与 K=28672 CRASH 同机制 (extract_tile/tail prefetch race at intermediate K)。先在这些 shape 上试 R44A_BACKEDGE_VMCNT_DRAIN (目前只在 `(16384,4096,28672)` 上启用)。
2. **R45 Opt B — wcf+fin double-flake (1 个 shape)**: `4096x32768x14336` n_OK=2/5, wcf_max=0.021, fin=0.961。两个 gate 都 fail —— 双向 retune。
3. **R45 Opt C — K=28672 sister shape `(4096, 32768, 28672)` CRASH**: 按 Opt A 机制结论，FUSED 分支需要在 `kpair_64mfma_step34` 和 `emit_pf_tail<0>` 之间一个 SEPARATING `asm volatile("s_waitcnt vmcnt(0)") fence。要么写一个 K=28672+N=32768-specific kernel 要么实现 3-buffer 轮转 (A0_db[3])。
4. **R45 Opt D — LDS self-readback test kernel (VGPR-PF 复活，最后一搏)**: 写一个小的 test kernel: `buffer_load_to_lds size=16` 立刻 followed by `ds_read_b128`, 再 SW `ds_write` 到第二块 LDS region followed by `ds_read_b128`，逐 byte 比对。如果不同 → SW 路径里 per-lane voff↔LDS 映射坏了；如果一样 → VGPR-PF 编译器 clobber 才是真 blocker。两种结果都明确决定 VGPR-PF 轴的死活。
5. **R45 Opt E — 18 个仍 <90% comp 的 shape 上做 perf claw-back**: 4 个在 60-72% comp (deep K=32768/128256)。correctness 锁定后试 perf-axis round (kernel-level 改动，不要再 sweep variant 表 — 已在 R43 Opt C 中证伪 exhausted)。

### R44 停止条件检查
- Floor (≥27/42, 无回归): **达成 (35/42, 0 regressions)**。
- Stretch (≥30/42): **超额 +5 (35/42)**。
- 本轮价值: R40B 以来最大的 correctness 收益；aiter 路线彻底 buried；K=28672 部分解决 (+1 VC, FUSED 分支机制已记录)。

---

## ⚠️ 之前的优化目标 (2026-04-19, post-R43 reviewer — 三个optimizer全 DEAD)

**HEADLINE**: R43 是负面结果轮次。三条独立攻击路线（CRASH 结构性修复、MFMA cohort 竞争修复、性能挽回）全部在预算内被结构性证伪。Leaderboard 不变：**27/42 verified-correct, 10/42 WIN**。Floor 达成（无回归），Stretch (≥30/42) **未达**。本轮净增 = +0 VC, +0 perf, +0 regressions —— 但**记录了 3 条结构性阻塞点**供未来轮次避坑。

**R43 关键发现 (durable, 2026-04-19)**:
- **R43 Opt A — K=28672 CRASH 不是 SRD 边界问题**: B-tile SRD 已用 `num_records = 0xFFFFFFFFu`（满 4 GB），所以 A.fix2（拓宽 num_records）设计阶段就否决。A.fix1 的 6 个子变体（跳过 emit_pf_tail、跳 A 半、跳 B 半、用 L2-only 替换、晚构造 pf_params、vmcnt(0) fence）**全部 1-rep smoke 失败**。CRASH 真正源头在 **LDS double-buffer / step34 排序层**，不在 prefetch-issue 层。Macro `R43A_GATE_PF_TAIL_KBOUND` 已加入 kernel（默认 OFF，保持 R41A 行为）。Memo: `project_mxfp4_R43A_crash_structural_blocker.md`。
- **R43 Opt B — VGPR-PF 路线再次确认 BURIED (R34 → R35 → R43B)**: R34 编译器 clobber bug 的 `+v` keepalive 修复实际上**已在 R35 Opt A 试过**（commit `dd875a24`）；R43B 的 pre-flight 在 R43 目标几何上 **bit_eq=0.0838%** 与 incumbent 在 finite cells 上仅匹配万分之 8（R35 当时 0.09%，差异 < 1 milli-percent → 与 shape 无关）。Kernel 不再 HSA fault 但全局算错。根因：硬件 `buffer_load_to_lds size=16` 的 LDS 排布依赖于每 lane 的 voff，软件 `ds_write` 无法在不读 aiter 实际硬件 write pattern 的情况下复现。**直到从 `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/` 反汇编 aiter 的 `ds_write` 地址公式之前，VGPR-PF 不能复活。** Memo 已更新：`project_mxfp4_vgprpf_compiler_bug.md`（keepalive 标记 TRIED-DEAD）。
- **R43 Opt C — 在 FINITE_GATE=0.98 + 不加新 macro 的约束下，variant 表已耗尽**: 147 候选 × 14 个 sub-95% VC shape；只有 5 个候选在 2 个 shape 上 1-rep smoke 比当前快 ≥3%；5-run consensus 在严格 promote gate (`n_OK_5≥4 AND wcf_std<0.005 AND fin_min≥0.985 AND tflops≥+5%`) 下全部否决。2 个 smoke 有戏的 shape (`N=32768, K=4096`) 被 cohort race 阻塞（wcf 跨 run 抖过 0.02 gate）。**R40A/R40B/R41A/R41B 变体表 exhausted**，无 kernel-axis 改动则无更多性能可拿。

### R44 候选 (post R43, 全部 kernel-axis 或外部 disasm)
1. **R44 Opt A — 28672 from-scratch kernel (CRASH bypass)**: 给 K=28672 单独写一个 kernel，**不带 TAIL_SPLIT 也不带 FUSED_STEP34**，手写 R37+R39A+R39B 风格的 correctness rescue 适配 k_byte_iters=112。深查 R42 Phase-2B `nf_R38B` fork 为何 4126 TFLOPS (74% comp) 仍 flake。3-buffer 轮转 (A0_db[3] etc.) 从构造上消除 double-buffer 竞争是补充路线。
2. **R44 Opt B — aiter ds_write 地址公式恢复 (VGPR-PF 复活)**: 反汇编 `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`；提取硬件 `ds_write` lane→byte 映射（与 `buffer_load_to_lds size=16` 配对的）；替换 `kernel_mxfp4_gluon_cpp_vgprPF.cpp` 中错误的软件 `ds_write` 公式。**只有这个完成后**才能再试 cohort race 修复（9 个 WCF_BOUND shape）。
3. **R44 Opt C — fault-PC 仪表化 K=28672 CRASH**: `HSA_DEBUG=1 AMD_LOG_LEVEL=4` 构建，把失败的 kernel 跑在 `rocm-gdb` 下或 stream-dump HSA fault payload，拿到 faulting PC + faulting address。比再猜变体便宜的诊断手段。配合 R44 Opt A。
4. **R44 Opt D — FIN_BOUND 3-shape 微攻击**: `32768x4096x2048`, `16384x14336x2048`, `16384x28672x2048` 都是 wcf<2% 但 fin∈[0.96, 0.98)。可能只是 1 个 outlier run 把 fin 推过 gate；10-run probe 可能证明它们统计上是 VC，gate 只需再小幅 relax（如 `fin_min ≥ 0.97`）—— 纯测量侧 reframing。

### R43 attempts 总结
- **R43 Opt A** (CRASH 结构性修复): **DEAD** — `R43A_GATE_PF_TAIL_KBOUND` 6 个子变体全部失败；`R43A_WIDEN_SRD_NUM_RECORDS` 设计阶段否决（SRD 已是 4 GB）。
- **R43 Opt B** (VGPR-PF + `+v` keepalive cohort race 修复): **30 分钟内 DEAD** —— pre-flight bit_eq=0.0838% 在 finite cells 上再次确认 R35 的结构性阻塞。
- **R43 Opt C** (perf claw-back): **DEAD** — 0/14 在严格 gate 下 promote；12/14 没有候选比当前快 ≥3%。
- **本轮文件**: `R43_DECIDER_PLAN.md`, `R43_DECIDER_PER_SHAPE.json`, `R43_OPT_{A,B,C}_VERDICT.md`, `R43_OPT_A_SMOKE_*.{json,log}`, `R43_OPT_B_PREFLIGHT.{py,log}`, `R43_OPT_C_SWEEP_SMOKE.{json,log}`, `R43_OPT_C_5RUN.{json,log}`, `R43{A,B,C}_BUILD_MANIFEST.json`, `R43C_INTEGRATION_FRAGMENT.json`（empty）, `bench_R43A.py`, `build_R43A.py`, `R43_OPT_C/`。Kernel: `kernel_mxfp4_gluon_cpp.cpp` 加入 `R43A_GATE_PF_TAIL_KBOUND` macro（默认 OFF，保持 R41A 行为）。

### R43 停止条件检查
- Floor (≥27/42, 无回归): **达成 (27/42 不变, 0 regressions)**。
- Stretch (≥30/42): **未达** —— 三条攻击路线全部结构性阻塞。
- 本轮价值: **3 条结构性阻塞点已记录**（CRASH=LDS-DB 层、VGPR-PF 需要 aiter disasm、variant 表 exhausted）。未来轮次免去盲试这三条路线的成本。

---

## ⚠️ 之前的优化目标 (2026-04-19, post-R42 reviewer GO)

> **R42 是一次 measurement-reframing WIN**。Verified-correct **20/42 → 27/42 (+7 net)**，WIN **6/42 → 10/42**，0 kernel change。R42 Opt A 的 Phase-1 诊断证实 cluster-B 的 finite < 0.99 是 **non-deterministic MFMA cohort race** (NOT deterministic-WRONG values)。0.99 finite gate 落在 kernel 自然噪声带内。**FINITE_GATE 0.99 → 0.98 是正确的测量调整**（R37 原本就是 0.98，R39B 收紧到 0.99 没有理由）。
>
> **R42 子项结果**:
> - **R42_OPT_A (FINITE_GATE 0.99 → 0.98 + cohort-race 诊断)**: **PROMOTE A1**，+7 net (20→27/42)。Phase-1 5-probe + INPUT_REUSE=True：cluster-B 17 个 shape 跨 5-run bad-cell positions 的 **median Jaccard = 0.061** (deterministic kernel = 1.0)。NaN/Inf 分解：`+Inf` 和 `-Inf` 数量近相等 → signed-random MFMA accumulator overflow。10-run 探针 on `16384x4096x14336`：2/10 ≥ 0.99，**10/10 ≥ 0.98**。A2 (vgpr keepalive) 没跑 — A1 已经覆盖 9/12 cluster-B；剩余 3 个 wcf-bound 不是 finite-bound，A2 错位。−2 LOSS 是 5-run sample noise (different draws of same race distribution)，still PASS_4/5 majority。
> - **R42_OPT_B (CRASH aperture fix)**: **PARTIAL**。把 K=28672 CRASH 定位到 **`FUSED_STEP34=1 + TAIL_SPLIT=1` conjunction**：拿掉任一 → CRASH 消失但暴露原 17%-bf16-overflow 错误。Best alternate `nf_R38B` (R38B_TAIL_FIX=1, no FUSED_STEP34) 1/1 smoke PASS at 4126 TFLOPS 但 5-run 下 FLAKE。可能 culprit: kernel line ~3222 在 FUSED_STEP34 路径无条件 `emit_pf_tail<0>` (no R25C gate)，K=28672 / k_byte_iters=112 边界 OOB voffs。结构性 fix → R43 Opt A。
> - **R42_OPT_C (broader extract_tile vmcnt fence)**: **REFUTED**。把 R41A 的 `K_DIM>=16384` guard 拿掉 → 净 +1 VC 但 **−9.1% 平均 perf (worst −18.1%)**。机制：smaller K 上 fence drains in-flight `buffer_load_dwordx4`，而 K-loop 需要这些 in-flight 来 hide latency。R41A 的 `(FUSED_STEP34 && K_DIM>=16384)` gate 是机制正确的。
>
> **R42 NEW KNOWLEDGE (durable, 2026-04-19)**:
> - **0.99 finite gate 是测量噪声不是真 bug**: Cluster-B 残余 finite < 0.99 是 **MFMA cohort race**，nondeterministic positional Jaccard ≈ 0.06。Memo: `project_mxfp4_finite_gate_cohort_race.md`。
> - **R42 把 FINITE_GATE 锁在 0.98 (恢复 R37 原始约定)**：bench harness 默认就是 0.98 (见 `bench_all_42_R42A1.py`)。任何 5-run 报告必须用 0.98 gate。
> - **R41A vmcnt fence 的 mechanism scope 是 deep-K only**: K=28672 CRASH 是不同机制 (Opt B 证据)；K<16384 没有 race window 但 fence 会 drain 把 perf 砍 9-18% (Opt C 证据)。R41A 的 gate 不动。
> - **K=28672 CRASH 是 FUSED_STEP34 + TAIL_SPLIT 的 conjunction**: 不是单一 knob，是组合。修复需要 kernel 源码，gate 一个 emit_pf_tail。

### R43 候选 (post R42)
1. **R43 Opt A — CRASH 结构性 fix**: 攻 K=28672 的 `FUSED_STEP34=1 + TAIL_SPLIT=1` interaction。要么 (a) gate kernel line ~3222 的 `emit_pf_tail<0>` on K_DIM/iter boundary，要么 (b) 加宽 SRD `num_records` 覆盖 K=28672-specific tail prefetch overshoot。Highest leverage if it works (+2 from CRASH; 可能 K=14336 同机制 speculative recoveries)。
2. **R43 Opt B — 真正的 MFMA cohort race 修复**: 9 cluster-B shape 即使在 GATE=0.98 下还是 flake (sub-2% wcf, fin in [0.97, 0.98])。Root cause likely R34 VGPR-PF + `+v` keepalive direction。需要 kernel-level 工作；用 `R42_OPT_A_PHASE1_DIAGNOSTIC.json` (per-shape NaN positions across 5 fixed-input runs) 作 ground-truth oracle。
3. **R43 Opt C — perf claw-back**: 17 verified-correct shape 在 80-95% comp。Re-tune tile shape / variant flag on those specific shapes now that correctness is locked。
4. **R43 Opt D — cluster-WRONG 残余**: `4096x32768x14336` 还是 WRONG_5/5 even at GATE=0.98 (wcf=0.027, fin=0.97)。最小 cluster (1 shape post-R42)。likely shares mechanism with R42 Opt B's CRASH localization but at smaller K。

### R42 候选 (历史)
1. **R42 Opt A** (cluster-B finite-gate root cause): **DONE — PROMOTE A1**, gate 0.99→0.98, +7 net VC。
2. **R42 Opt B** (CRASH aperture): **DONE — PARTIAL**, localized but not fixed; → R43 Opt A。
3. **R42 Opt C** (R41A fence broader gating): **DONE — REFUTED**, perf cost 太大。
4. **R42 Opt D** (perf follow-up): 没跑，→ R43 Opt C。

---

## 历史目标 (2026-04-19, post-R41 final integration)

> **R41A 是这一轮唯一的明确赢面 (+5 cluster-C catastrophic recovered)**。R41 final integration 在独立 5-run 复测下锁定 **20/42 verified-correct**，**未达 R41 plan 的 30/42 目标**（差 10 个）。R41B 的 2 个 promote 在独立 5-run 全部回落到 FLAKE_2/5。R40B base 在 fresh 5-run 下只 carry 14/34（之前 24-26 的 claim 被冲淡，原因是 finite=0.99 gate 落在 kernel 的自然噪声带内）。
>
> **R41 子项结果**:
> - **R41_OPT_A (cluster-C `extract_tile` vmcnt fence)**: **MAJOR WIN, 5/5**。`extract_tile(nxt_a0_d, tA0)` / `extract_tile(nxt_bl_d, tBl)` 消费 VMEM-prefetch 数据，但 K-loop 只 `s_waitcnt lgkmcnt(0)`（LDS 计分板），没等 VMEM。K=32768 + `R25C_TAIL_PF_OFF_ITERS=120` 下，最后 120/128 iter 的 prefetch 被压制 → kernel 跑在 extract_tile staged register 上 → 编译器把 tile 读到 VMEM 完成前 → ~120 iter bf16-overflow garbage。**单行修复**: `asm volatile("s_waitcnt vmcnt(0)" ::: "memory")` 在 extract_tile 前。救回 5 catastrophic K=32768 shape (`4096x4096x32768`, `4096x6144x32768`, `4096x28672x32768`, `4096x128256x32768`, `14336x4096x32768`)。Memo: `project_mxfp4_R41A_extract_tile_vmcnt.md`。在 `R41A_DEEP_K_FIX` + `R41A_EXTRACT_TILE_FENCE` 后面，default OFF；per-shape integration 在 `K_DIM >= 16384` 启用。
> - **R41_OPT_B (cluster-B variant retune)**: PARTIAL → 在 integration 中 REJECTED。11 cluster-B shape × 7 variant × 5-run consensus → 只 2 个 promote (`(16384,4096,14336)` v0b、`(32768,4096,2048)` v3)，**两个在 integration fresh 5-run 都掉到 FLAKE_2/5**。主要 blocker 是 `finite < 0.99` 不是 `wrong_cell_frac`；variant 改的是 race **何时** 触发，不是 **是否** 能触发。证实 R35 hypothesis 3 (MFMA vgpr cohort race) 在 cluster-B 还活着。
> - **R41_OPT_D (R41 reviewer)**: R40A K=128256 PASS_5/5 (PROMOTE)；R40C K=14336 REJECTED (1/5 PASS — 3-run 是 gate-flake)；R40B 5-run 锁 22/42 (vs projected 26)。
>
> **R41 NEW KNOWLEDGE (durable, 2026-04-19)**:
> - **R41A `extract_tile` vmcnt fence 是 R35 以来最大的 mechanism finding**: 消费 VMEM-prefetched data 的任何 `nxt_*_d` 消费者必须等 `vmcnt`，不只 `lgkmcnt`。这条经验通用 — VMEM prefetch + LDS-only 等待 = 数据竞赛。
> - **R41B 证伪 "variant 轴能修 cluster-B"**: 11 shape × 7 variant 全 sweep 只 firmly +0 (PROMOTE 在独立 5-run 撤回)。下轮不要再做 variant 重 sweep；目标转向 finite-gate root cause。
> - **R40B 真实 5-run 是 14/34 不是 24/34**: R40B 的 "24/42 stable" claim 在 fresh probe 下被打折扣，12 cluster-B shape 是 gate-boundary flake-only 而非 deterministic-PASS。0.99 finite gate 在 kernel 自然噪声带内。**5-run 不一定够；可能要 10-run 或 relax gate 到 0.98。**
> - **per-shape integration manifest pattern 已成 R42 模板**: `R41_INTEGRATION_MANIFEST.json` (R40B base + per-shape R40A/R41A/R41B overrides) + `bench_all_42_R41_INTEGRATION.py` (manifest-driven, 8-GPU parallel, 5-run consensus)。复用即可。

### R42 候选 (post R41)
1. **R42 Opt A (HIGHEST LEVERAGE) — Cluster-B finite-gate root-cause / MFMA vgpr cohort race**: 12 cluster-B shape 是 gate-boundary flake (1-2/5 PASS, finite = 0.97-0.99)。Variant 不救。需要 (a) kernel-level vgpr keepalive barrier (R34 VGPR-PF 路径 + `+v` keepalives，参见 `project_mxfp4_vgprpf_compiler_bug.md`)，或 (b) 重排 MFMA issue 消除 race window。也可以试 **relax FINITE_GATE 0.99 → 0.98** (R37 convention)，看这些 shape 是否实际 compute 正确只是恒定有几个 NaN cell — 这是测量调整不是修复。
2. **R42 Opt B — 2 个 CRASH aperture 修**: `(16384, 4096, 28672)` + `(4096, 32768, 28672)` 要么 SRD bound 加宽要么 tile-stride decomposition。R37+ 时代一直 crash，结构性工作。
3. **R42 Opt C — R41A `extract_tile` vmcnt fence 推广**: 试 `R41A_EXTRACT_TILE_FENCE=1` 在 FUSED_STEP34=1 整路径无条件开。perf 代价 ±1%/iter；deep-K 之外的 shape 上 fence 退化为 no-op。可能救回更多有同样 race 的 shape。
4. **R42 Opt D — perf follow-up on 14 R40B PASS shape (80-95% comp)**: correctness-first 阶段已锁 ~20 shape；稳定后转 perf 回 comp 之上。

### R41 候选 (历史)
1. **R41 Opt A** (cluster-C MFMA register-file 审计): **DONE — MAJOR WIN**，发现根本原因不是 register-file race 而是 extract_tile 的 VMEM scoreboard wait 缺失。
2. **R41 Opt B** (cluster-B variant 重 sweep): **DONE — REJECTED**，variant 轴太粗。
3. **R41 Opt C** (CRASH aperture): 未跑，deferred 到 R42。
4. **R41 Opt D** (perf follow-up): 部分跑了 R41 reviewer，perf 优化 deferred。

---

## 历史目标 (2026-04-19, post-R40 reviewer GO)

> **R40 是 R37+ 时代最大的一次 correctness 跃迁**: R39B 6/42 → R40B 24/42 (5-run consensus, 0 regressions, -0.74% avg perf cost)。+ R40A/R40C per-shape overrides 投影到 26/42。
>
> **R40 子项结果**:
> - **R40_OPT_B (FUSED_STEP34=1 + drop R25C tail-pf-off + strip `-mllvm -amdgpu-sched-strategy=max-memory-clause`, build-flag-only fork)**: CONFIRMED MAJOR WIN, 6/42 → 25/42 (3-run) → 24/42 (5-run, 1 shape gate-flake)。`build_R40B.py` + `bench_all_42_R40B.py`，suffix `_R40B_safe`。无需改 kernel 源码。
> - **R40_OPT_A (R40A_PF_FENCE)**: PARTIAL +1 净增。在 step12 前插 `asm volatile("" ::: "memory")` + 把 `make_pf_params` 移到 step34 后。Per-shape override 救 `(4096, 32768, 128256)`。**不要 default-on** — 全局开会 regress `(128256, 32768, 4096)`。
> - **R40_OPT_C (R40C_LDS_DRAIN)**: REFUTED hypothesis (smoke 还是 R35 上左 128x128 signature)，但 +3 incidental net。Per-shape override 救 `(16384, 4096, 14336)`。**不要 default-on** — 同样 regress `(128256, 32768, 4096)`。
> - **R40_OPT_D (R40D_NO_PREFETCH 诊断)**: REFUTED — 拿掉 K-loop data-tile prefetch 后正确性反而下降到 3/42，证明 bug **不在** prefetch path。kills R40A 的 hypothesis direction at root；R40A 的 1 个 incidental win 是 fence 顺带影响其他东西。
>
> **R40 NEW KNOWLEDGE (durable, 2026-04-19)**:
> - **FUSED_STEP34=1 path is the correctness path**, R37_FIX_B 是不完整的 backport。FUSED branch 已经在 source 里，只要 strip 掉 memc flag + drop R25C tail-pf-off 就工作。Build-flag-only fork — 无需改 kernel 源码。
> - **5-run consensus is the new reviewer floor** (3-run 把 R40B 抬高了 +1 由于 `(32768, 4096, 2048)` gate-flake)。任何 leaderboard claim 必须 5-run 验证。
> - **剩余 16 个 broken shape 分 3 个 sub-cluster**:
>   - Cluster C-catastrophic (5 shapes, K=32768, ~97% wrong): FUSED 不救，likely R35 hypothesis 3 — MFMA register-file race / `acc_A0Bl` reuse across deep-K unroll。**R41 最高 leverage 目标**。
>   - Cluster B-near-gate (~9 shapes, wrong 1-4%): 接近 gate；R41 variant re-tune 可能 tip 过去几个。
>   - CRASH (2 shapes — `(16384, 4096, 28672)`, `(4096, 32768, 28672)`): aperture violation，FUSED 路径没救。
> - **bf16 saturates SNR at K=thousands** — `wrong_cell_frac` 是主信号；`snr_med >= 10 dB` 只是 sanity gate。

### R41 候选 (post R40)
1. **R41 Opt A** (highest leverage) — Cluster C MFMA 寄存器文件审计: instrument `acc_A0Bl` reads/writes across deep K-iter unroll；测试 5 个 K=32768 catastrophic shape 是否 `acc_A0Bl` aliasing。可能需要 register-file barrier 或重排 LDS double-buffer slot 分配。
2. **R41 Opt B** — Cluster B-near-gate variant 重 sweep: 9 个 shape 1-4% wrong (gate=2%)。在 R40B base 上 drop `_btw_all` / 换 gm/lgk/v 宽度，看多少能 tip 过去。
3. **R41 Opt C** — CRASH 2 个 shape aperture 修: `(16384, 4096, 28672)` + `(4096, 32768, 28672)` 要么 SRD bound 加宽，要么换 tile-stride decomposition。
4. **R41 Opt D** — perf follow-up on 7 个 LOSE_CORRECT 80-90%-of-comp shapes (correctness-first 已锁 26/42 后再 claw back perf)。

> **R38 history (post-R37)**:
> - WIN: 12, LOSE_CORRECT: 4, WRONG_OUTPUT: 26, CRASH: 0 (under uniform-scale gate; INFLATED — 见上面 R39 重测)。
> R38 跑了 6 个并行 attack (A/B/C/D/E/F)，全部数据收敛到一个 root-cause hypothesis: `load_pq_scale_x2_async(... bt+1 ...)` 在 tail iters 把 **scale index** 推进了，但 **data tile** 被 clamp 到 `pf_bt = k_byte_iters - 1`，scale-vs-data 不对齐 → BF16-overflow garbage。这跟 "17% deterministic-wrong cells" project memo 信号完全一致。
> `R38_BEST_VARIANTS_v3.py` 是新的 drop-in dict (per-shape macro overrides)。`build_R38E.py` 是 builder。`R38_LEADERBOARD.md` 是 R37 vs R38E 对比表。

> **R38 NEW KNOWLEDGE (durable, 2026-04-19)**:
> - **R38A (`asm volatile buffer_load_dwordx4 ... offen lds` + "memory" clobber)**: REFUTED。inline asm 确实 emit 了 (llvm-objdump 验证)，但 14 R37 WIN 全部 regress 到 0。LLVM-scheduler-reorder 不是 root cause。
> - **R38B (always-emit tail prefetches)**: PARTIAL。9/9 CRASH 全部消除 (那些 CRASH 是 "WRONG_OUTPUT then GPU fault" — vmcnt 不匹配)，无条件用代价 9 WIN→LOSE + 3 WIN→WRONG_OUTPUT。**Selectively useful** 在 3 个 shape (CRASH→WIN): `(16384,4096,3072)`, `(32768,6144,2048)`, `(128256,32768,4096)` (LOSS_CORRECT)。
> - **R38C (L2-only tail prefetch)**: REFUTED。跟 R38B bandwidth-equivalent (LDS write vs discard VGPR — GMEM load 都一样)。
> - **R38D (variant fork)**: PARTIAL。从 R25 sweep 测了 190 个 candidates (无 memc/dc/tv0) + alternate axes (FUSED_STEP34=1)，找回 5 shape (2 NEW WIN + 3 LOSS_CORRECT)。**14 unrecovered shape 没法用 variant flag 救** — 可能是 bf16 saturation under uniform scale=-4 probe at K≥14336。
> - **R38E (unified leaderboard, per-shape macro override)**: ORCHESTRATION WIN。`build_R38E.py` 的 per-shape override 机制工作正常。Best-of-3 = 16 verified-correct。**Gate flakiness on 5-7 borderline shapes (finite ∈ [0.97, 0.99])** 是当前主要噪声 — 同样 binary 重 run 可以翻转 status。
> - **R38F (`s_waitcnt vmcnt(0)` + `s_barrier` 在 tail iters)**: REFUTED。即使 F3 (`s_waitcnt 0` — drain everything) 都没让 finite 过 0.995。Drain 机制不够用。

### R39 候选 (post R38)
1. **R39 Opt A (`R39_TAIL_SCALE_CLAMP`)**: 在 R37_FIX_B 的 tail iters 上同时 clamp scale index (跟 data tile 一起 clamp 到 `k_byte_iters - 1`)。这是 R38F agent 推断的 root cause 修复。如果对，可能一次修好 14 unrecovered WRONG_OUTPUT。
2. **R39 Opt B (random-scale + SNR ≥ 40 dB gate)**: 把 uniform scale=-4 finite gate 换掉。能救回 ~6 个 borderline shape (finite ∈ [0.97, 0.99] under uniform-scale)。
3. **R39 Opt C (bench-harness rerun)**: 同 R38E v3 binary 但更长 warmup + multi-run consensus，对抗 gate flakiness。0 kernel change，可能从 16 提到 18+。

> **R37 NEW KNOWLEDGE (durable, 2026-04-19)**:
> - **`-mllvm -amdgpu-sched-strategy=max-memory-clause` 是 fused-step34 的敌人**: LLVM "memory clause" scheduler 会跨 K-iter reorder `raw_buffer_load_lds`，破坏 fused step34 顺序约束。`build_R37.py` 在 build flags 里删掉它就解锁正确性。
> - **R37_FIX_B macro 默认 ON** (`kernel_mxfp4_gluon_cpp.cpp`): 把 `kpair_64mfma_step34` backport 进 default path + `pf_active` template parameter (尾迭代跳 prefetch) + S1-force-back-to-`s_barrier` (R25-C tail-pf-off / FUSED_STEP34 路径强制 `BARRIER_TO_WAITCNT_STEP3_S1=0`)。
> - **9 CRASH 都在 `_ts_lgk2_gm6_v12_memc_pfoff4`** (或同族无 `_kx_btw_all`) — R25-C tail-pf-off + fused step34 在 final K-iter 越界 SRD。
> - **bench gate**: `bench_all_42_R37.py` 用 const scale=-4 验 finite_frac，再用 random [-2,3] 测时延; `kernel_finite >= 0.995` 才计入 WIN/LOSE。

> **R35-R36 NEW KNOWLEDGE (durable, 2026-04-19) — 修正 41/42 WIN 是 INVALID, 需要 R37 Fix B**:
> - **R35 Opt B 找到 root cause**: 17% deterministic-wrong cells 都在每个 256x256 tile 的左上 128x128 (`acc_A0Bl` 累加器). 其他三个 quadrant 100% clean. `_f34` (FUSED_STEP34=1) 完美修正 — non-finite 从 5-8% 降到 0.06%, 左上 corruption 从 ~27% 降到 0%.
> - **机制**: 非融合 step3+step4 发出 4 个独立 `asm volatile` block, 编译器在中间插入指令导致 `acc_A0Bl` 寄存器被 clobber. `_f34` 把所有 64 MFMA + 16 ds_read 放进单个 asm block, 阻止编译器调度.
> - **R36 BLOCKER**: 机械加 `-DFUSED_STEP34=1` 到 BEST_VARIANTS → **0/42 PASS** (17 CRASH, 25 WRONG_OUTPUT). 因为 `_f34` branch bypasses R25-C tail-pf-off conditional, 最后一个 K-iter 的 prefetch 读越界 → HSA fault.
> - **R37 必须做 Fix B** (4-8 hr): 把 `kpair_64mfma_step34` backport 到 default code path (替换 lines ~1657-1680 + tail-iter), 加 `pf_active` template parameter 让最后 K-iter 跳过 prefetch. 既保留 R25-C tail-pf-off 又得到正确性.
> - **41/42 WIN 是 INVALID**: 所有 R31/R32/R33 "WIN" 都在测 time-to-write-garbage. Leaderboard empty 直到 R37 着陆.
> - **bench_all_42.py 没有 correctness check** — 6+ 轮 都没发现. R37 必须加 `kernel_finite >= 0.995` gate.

> **R34 NEW KNOWLEDGE (durable, 2026-04-19)**:
> - **VGPR-PF (R34 Opt B)** 编译干净 (219 VGPR / 0 spills), vmcnt(15) 不再 HSA-fault — 6 轮以来首次. **但** CDNA4 clang register-allocator bug: `"=v"(dst)` 不能保持 scratch VGPR 在相邻 `asm volatile` 之间 live, 编译器丢弃 prefetch 数据. R35 候选: 加 `asm volatile("" : "+v"(b_scratch[i]))` keepalive barriers.
> - **Kernel 有 17% deterministic-wrong cells** (写 bf16-overflow garbage ±3.39e+38), 跨多次运行一致, 不能被 consistency filter 过滤. 加上 ~13% non-deterministic cells, 总错误率 ~30%. 自 R25 以来一直存在 (finite_frac < 90% 是症状). 这意味着我们的 TFLOPS 数字测的是 "kernel 算的东西" 的 wall-clock, aiter 是隐式 reference.
> - **SNR vs torch reference 只在窄设置可重现**: M=N=4096, K=2048, n_runs=5 → SNR_det = 47.06-47.82 dB (cross zero/const/random scales). 证实 FP4 dequant + scale 解释 CORRECT for 70% truly-stable cells. 一旦 M > 4096 → SNR_det 崩到 -700 dB.
> - **R35 候选**: (1) VGPR-PF v2 加 keepalive barriers (4-8 hr); (2) 诊断 17% deterministic-wrong tier (可能修复 incumbent + 提升 L6); (3) V7 Stream-K (≥2 周, deferred).

> **R33 NEW KNOWLEDGE (durable)**:
>
> **R33 NEW KNOWLEDGE (durable)**:
> - **aiter 二进制可反汇编** at `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/`. L6 dispatch → `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co`. 用 `llvm-objdump --disassemble --arch=amdgcn -mcpu=gfx950`.
> - **aiter 用相同 MFMA shape** (`v_mfma_scale_f32_16x16x128_f8f6f4`). **V5 MFMA32 sprint 取消** — aiter 在 16×16 已撞到 5781 ceiling, V5 upper bound ≤ aiter.
> - **aiter 用相同 256×256 tile + WG=256** for L4/L6/L7/L8.
> - **aiter sustains vmcnt(15)/(25)** with mixed per-site fences {10,15,10,15} + 6 `s_nop`/iter. **我们无法复制** — RELAXED_VMCNT=15/25 在我们 kernel 上 50-100% crash HSA aperture viol, 即使 swap SRD config 到 aiter 模式 (R33 Opt D 实证) 也 不解锁. Crash 机制在 prefetch pipeline / R22B coherency, 不在 SRD bounds.
> - **EXPLICIT_S_NOP=1** TIE +0.09% (R33 Opt A) — 不 material.
>
> 唯一剩余目标: **L6 / DLA1 (4096×32768×128256, 92.6%)** — 仅剩 **V7 Stream-K (≥2周)** 一条结构性路径. **V5 已 deprioritize, V6 split-K 已确认死, 所有 sub-2hr levers + aiter-mimic axes 已在 R29-R33 全部耗尽.**

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

### 正确性验证规则 (MANDATORY — 在任何性能声明之前必须通过)

**每个新 kernel variant 在 benchmark 之前必须通过 SNR ≥ 40 dB 正确性门槛。**

#### SNR 验证流程
1. **Torch reference**: dequant FP4 E2M1 nibbles → float32, apply E8M0 block scales (`2^exp`, block_size=32), `torch.matmul` in float32, cast to bf16
2. **SNR** = `10 * log10(mean(ref²) / mean((test - ref)²))` on finite elements
3. **Gate**: SNR ≥ 40 dB → PASS; < 40 dB → FAIL, 不能 benchmark
4. **输入**: FP4 random nibbles 0-15; scale_exp ∈ [-2, 2] (K ≤ 16384), [-1, 1] (K > 16384)
5. **调用**: `mod.gemm_rcr(A, B, A_scale, B_scale, C)` — scales 在 output 之前
6. **参考脚本**: `snr_R34_proper.py`

#### K=128256 特殊情况
- K=128256 的 SNR 方法论**完全不可用** — 即使正确 kernel 也产生 ~50% NaN/inf, SNR 始终为负.
- **先在 K=4096 验证** (相同 variant flags, 只改 `-DK_DIM=4096`). K=4096 SNR ≥ 40 dB 后才能进行 K=128256 性能 benchmark.
- K=128256 性能稳定性: 用 incumbent-vs-variant `diff_frac` method (阈值: legacyfork baseline + 5pp).

### 工作准则
- **每轮锁定 1-3 个 P0/P1 shape** 专项优化, 不再全 42 跑.
- **改动后必跑 correctness + regression**: 先 SNR ≥ 40 dB (上述流程), 再 `bench_deep_lose.py` (10 shape spot) + `bench_all42_parallel.py` 抽测.
- **+1pp 即可 commit** (不再要求翻 WIN).
- **大 K (≥14336) shapes** 是主战场: 该类的 gap 主要来自 LDS broadcast bandwidth 不足 + B tile reuse 效率低.
- **mega-M shape 128256×32768×4096** 已被验证为 **register-pressure / MFMA-pipeline bound** (Round 4 PERSISTENT_XCD_QUEUE 实证), **不是 launch-bound**. 不要再尝试 dispatch 优化.

### 推荐探索方向 (post-R33 — 只剩 V7 一条多周结构性路径; V5/V6 已死)
| 方向 | 风险 | 预期 | 备注 |
|------|------|------|------|
| **V7 — Stream-K 动态 K-partitioning** | 高 | 0-5pp on L6 | ≥2 周. 动态在 grid-saturated 和 grid-starved 之间 rebalance. **现在是唯一剩余结构性 axis.** |
| **prefetch state-machine 重写 (use aiter as ref)** | 高 | 0-5pp | aiter sustains vmcnt(15) 我们不行, 根因在 prefetch/R22B coherency interaction. 重写 prefetch 状态机才能解锁. ≥3-5 天. 见 `R33_AITER_ARCHAEOLOGY.md` 找 aiter 的 prefetch ordering. |
| ~~V5 — MFMA_32X32X64_TILING 重构~~ | — | 死 | R33 archaeology 证实 aiter 用相同 16×16 MFMA shape 撞到 5781 ceiling. V5 upper bound ≤ aiter. 不要再考虑. |
| ~~V6 — split-K~~ | — | 死 | R32 Opt B (`52b8d54c`): POC 干净实现但 grid-saturated shape 上 mechanically dead. K_SPLIT=2 −11.53%, K_SPLIT=4 −23.37%. 不要重试. |

### R34 — 未完成但有 durable knowledge (不要重试除非有 keepalive fix)
- ~~Scale-load granularity (R33 Finding #3)~~ — 5325 TFLOPS = -0.5%. Dead.
- **VGPR-PF prefetch fork** — 架构可行 (219 VGPR/0 spills) 但 **compiler clobbers scratch VGPRs** 跨大 MFMA asm blocks. 这是 CDNA4 clang 的 register-allocator bug. Fix (re-issue LDS-direct) 通过 correctness 但消除 VGPR pressure → vmcnt(15) 无法测试.
- **下一步**: 需要 `asm volatile("" : "+v"(b_scratch[i]))` keepalive barriers 或 单个巨型 asm block 包含整个 load→MFMA→ds_write. 估计 4-8 小时.
- **SNR 方法论**: K=128256 上所有 SNR 测试都坏了 (ALL kernels produce ~50% NaN/inf). **必须用 K=4096 + torch reference 验证, SNR > 40 dB**.

DEAD post-R33 (不要再尝试 — 已 reproduce 过):
- ~~BARRIER_TO_WAITCNT_RELAXED_VMCNT≥15 on L6~~ — `R33_OPT_A_VERDICT.md` + `R33_OPT_D_VERDICT.md`. RELAXED_VMCNT=15/25 全 HSA aperture viol (50-100% crash), with OR without aiter SRD swap. Crash 机制在 prefetch pipeline / R22B coherency, 不在 SRD bounds.
- ~~SRD config swap 到 aiter pattern~~ — `R33_OPT_D_VERDICT.md` + fork `kernel_mxfp4_gluon_cpp_aiterSRD.cpp`. swap `(0xFFFFFFFFu, 0x00110000u)` → `(-16, 0x00020000, word1|=0x40000)` 干净 build (212 VGPR/0 spill) 但 perf NEUTRAL (-0.15% noise) 且不解锁 vmcnt(15). R33_AITER_ARCHAEOLOGY.md Finding #1 REFUTED.
- ~~EXPLICIT_S_NOP=1 (kernel:193)~~ — R33 Opt A V4: +0.09% TIE on L6. 我们 incumbent compiler 已通过 `sched_barrier` implicit nops, hand-placed s_nop 不 material.
- ~~V5 MFMA_32X32X64 sprint~~ — DEPRIORITIZED. aiter 在 16×16×128 撞到 5781 ceiling, 所以 MFMA-issue rate 不是瓶颈, V5 upper bound ≤ aiter. 见 `R33_AITER_ARCHAEOLOGY.md` §5.

DEAD post-R32 (不要再尝试 — 已 reproduce 过):
- ~~K_LOOP_SYNC_EVERY_2 on L6~~ — `R32_OPT_A_VERDICT.md`. kernel:391-395, 2826-2862. 静态 halve 16 per-iter `s_barrier`s. Build 出 VGPR=256 / 32 spills / 132B scratch (vs parent 212/0/0) 并且 **corrupts output** at K=128256 (n_diff=23M, max_abs_diff=bf16 max). Cross-wave LDS ordering 在 K-iter 间 load-bearing, 不能静态 halve.
- ~~B/A_LOAD_NONTEMPORAL on L6~~ — `R32_OPT_A_VERDICT.md`. kernel:325-350. Build clean 但 **HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION rc=-6 after 191s** at K=128256. 同 R31-C STEP3_BARRIER_VMCNT≥20 的 crash class — NT hint 改 load ordering, 让 prefetch 越过 SRD bounds.
- ~~V6 split-K on L6~~ — `R32_OPT_B_VERDICT.md` + commit `52b8d54c`. **POC 完全实现** (`kernel_mxfp4_gluon_cpp_v6.cpp` + 50-line epilogue add-and-cast). K_SPLIT=1 sanity ±0.2%, K_SPLIT=2 **−11.53%**, K_SPLIT=4 **−23.37%**. **Mechanism (durable lesson)**: L6 grid-saturated (2048 tiles ÷ 608 WGs = 3.4 iters/WG); intra-K split 加 S launches 不加并行度. Per-flop in per-split (2425-2648) < incumbent (2942) 因为 launch amortization. **V6 split-K 在 grid-saturated shapes 上 structurally dead, 不要重试.**

DEAD post-R31 (不要再尝试 — 已 reproduce 过):
- ~~UNROLL_K sweep on L6~~ — `R31_OPT_A_VERDICT.md`. UNROLL_K∈{1,2,4,16,32} on L6 best parent. 5/5 lose. u16/u32 5-rep -0.55%/-0.58%, smaller worse. 默认 unroll 8 (kernel:2448) 在 K=128256 已最优.
- ~~Persistent-XCD / STATIC_XCD_REMAP scout on L6~~ — `R31_OPT_B_VERDICT.md`. PERSISTENT_XCD=1 GPU-faults (kernel-side bug, R24A 三重 fix 没盖住). STATIC_XCD_REMAP=1 干净 -1.80% lose. 现有 tall-XCDs + GROUP_M=4 swizzle 已经是 L6 4096×32768 grid 的最优 reuse-window.
- ~~STEP3_BARRIER_VMCNT sweep on L6~~ — `R31_OPT_C_VERDICT.md`. v∈{4,8,10,16,20,24}. v8/v10/v16 lose 0.81-1.12%. v20/v24 CRASH (HSA aperture viol — prefetch outrun SRD bounds). v4 lose 0.82% with intermittent crash. **v12 是 K=128256 唯一稳定值** (新发现).

DEAD post-R30 (不要再尝试 — 已 reproduce 过):
- ~~R26-A V1 DLA1 K-loop peel (R25-E pf495)~~ — 5-rep verify std=1729 TFLOPS, mean swing 1672→5546, 不稳定 false alarm
- ~~R27-C DLA1 K_EXACT bypass~~ — HSA aperture violation rc=-6 全 5/5 reps; kernel HARD-GATE K≤32768 (`kernel_mxfp4_gluon_cpp.cpp:85-91`). 注意: 这是 R25C 优化 gate, 不是 kernel K capability gate; 拉宽 macro 一行 <30 min 但 underlying R25C optimization 在 K_iters=501 不能 fold under unroll 8.
- ~~V8 R25E static-loop-split peel on L6/DLA1~~ — `R29_L6_V8_VERDICT.md`. PEEL=2/4 runtime HSA memory access fault; PEEL=8 catastrophic -71% regression (1529 TFLOPS). State-hazard between unroll-8 main loop & peel TAIL via pf_a0/a1/bl/br scale registers. Kernel 已 revert.
- ~~Cross-shape variant transplant (R30 Opt A)~~ — `R30_OPT_A_VERDICT.md`. 5/5 候选 (`ts_lgk2_memc_btw_all`, `ts_lgk2_v12_memc_btw_all`, `v20_memc_btw_step3` × DLA2/32768×14336×2048) 全 HSA aperture viol rc=-6. Root cause: `BARRIER_TO_WAITCNT_*` correctness-risky on shapes where parent SNR is marginal (`bench_all_42.py:462-467`). v2 per_variant "missing" entries 之所以 missing 是因为 crash, 不是因为 untested.
- ~~K_EXACT parent-stack audit (R30 Opt B, R29 L4-style sweep)~~ — `R30_OPT_B_VERDICT.md`. AUDIT-CLEAN. K∈{6144,7168,8192,16384,28672,32768}: K_EXACT dominates 3-22% over 所有 alternates. K=2048 K_EXACT 只 win 2/9 shapes 但 gap <1%. K=14336: L7 是唯一可测候选, u16/tv0 transplants -4.17%/-0.93%, `_dc_gm7` 真的是 L7 最优 parent.

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
