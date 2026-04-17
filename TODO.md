# MXFP8 优化 TODO

目标：
1. MXFP8 RCR 追平 FP8 per-tensor（长期）
2. **MXFP8 RRR / CRR 达到 MXFP8 RCR 的 95%**（新增，优先）

协议：`test_mxfp8_python.py` / `test_python.py` 内的 per-iteration sync + `output.zero_()`，warmup=100 iters=200。

## 当前 baseline（GPU7，per-iter sync，8192^3）

| 版本 | TFLOPS | SNR | Spills | 相对 MXFP8 RCR |
| --- | ---: | --- | ---: | ---: |
| **FP8 per-tensor RCR (长期目标)** | **3070.93** | 49.61 dB PASS | 0 | 105.0% |
| **MXFP8 8-wave RCR KPAIR+SRD+SCALE_PIPE+HOIST_HI (当前最佳)** | **2926.61** | 49.60 dB PASS | 0 | 100.0% |
| **MXFP8 8-wave RRR (默认 flag)** | **2794.26** | 49.59 dB PASS | — | **95.48%** ✅ |
| **MXFP8 8-wave CRR + PIPELINE_SCALE (新默认 flag)** | **2740.55** | 49.60 dB PASS | 0 | **93.64%** ❌ 差 1.46% |
| MXFP8 8-wave CRR (旧默认，PIPELINE_SCALE=0) | 2733.91 | 49.60 dB PASS | 0 | 93.42% |

### RRR / CRR 95% gate

目标线：2926.61 × 0.95 = **2780.28 TFLOPS**

- RRR：2794.26 ≥ 2780.28 → **已达标**（+13.98 over gate），本轮不动
- CRR：2740.55 vs 2780.28 → **差 39.73 TFLOPS (1.43%)**（PIPELINE_SCALE 默认开后）；HOIST_HI 路径已证架构性不可行（见 R7-R9），单 flag 无法到达 gate

reviewer 验收数据（GPU7，warmup=100 iters=200 per-iter sync）：
- HOIST_HI formal (with SNR + det 3/3 gate): 2925.64 TFLOPS PASS
- 同 GPU 同条件 A/B 4 次 mean：baseline 2914.87 → HOIST_HI 2932.67，**Δ +17.80 TFLOPS (+0.61%)**
- Dev B 在 GPU1 上 A/B 5 次 mean：baseline 2954.30 → HOIST_HI 2979.97，Δ +25.67 (+0.87%) (GPU1 噪声更低)

构建 flag（MXFP8 当前最佳）：
```
-DMXFP8_RCR_EXACT_8WAVE_FAST_ENABLE=1
-DMXFP8_RCR_EXACT_PQ_KPAIR_LOOP_ENABLE=1
-DMXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE=1
-DMXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE=1
```

## 已完成

- [x] 创建 `feat/mxfp8-only` 分支
- [x] 删除所有 MXFP4 / Gluon kernel、测试、rewriter、产物（48 个文件）
- [x] 建立 FP8 / MXFP8 baseline
- [x] 决策者汇编级差距分析（见下方）
- [x] 第一轮 agent team 派活（Dev A / Dev B / Dev C）— 被中断
- [x] 半成品改动 stash 保存（`stash@{0}`，含 Dev A 的 4-wave KPAIR/SCALE_PIPE 骨架、Dev B 的 HOIST_HI opsel helper、Dev C 的 8-wave asm rewriter）
- [x] 第二轮 agent team 产出评审：采纳 Dev B (HOIST_HI)，**拒绝 Dev C (ASM rewriter)** 和 Dev A (4-wave KPAIR/SCALE_PIPE)
- [x] 清理 `.s` / per-run `.json` 等生成物，加强 `.gitignore`
- [x] 整理 `.cursor/skills`：删除 deprecated `fp8-strict-layout-tuning`，重命名 `mxfp8-mxfp4-layout-tuning` → `mxfp8-layout-tuning`，清除 mxfp4 知识，加入 commit-time 工作流

## 差距分析（8-wave inner loop 每 kpair，64 MFMAs）

| metric | FP8 | MXFP8 (前) | MXFP8 (HOIST_HI) | Δ vs FP8 |
| --- | ---: | ---: | ---: | ---: |
| 总行数 | 403 | 431 | **422** | +19 |
| MFMAs | 64 | 64 | 64 | 0 |
| `buffer_load` | 16 | 22 | 22 | +6 (scale loads) |
| `ds_read` | 48 | 48 | 48 | 0 |
| `s_waitcnt` | 10 | 12 | 12 | +2 |
| `s_barrier` | 16 | 16 | 16 | 0 |
| `v_lshrrev_b32` | 0 | 6 | **0** | 0 (op_sel 替代) |
| VGPR | 252 | 256 | 254 | +2 |
| LDS | 131 KB | 135 KB | 131 KB | 0 |
| Occupancy | 2 | 2 | 2 | 0 |
| Spills | 0 | 3 | 0 | 0 |

**HOIST_HI 成功消除了 6 v_lshr**（通过 `op_sel` + `op_sel_hi` 让 MFMA 硬件直接从 32-bit scale pack 中 byte-select 两个 scale 字节）。

**关键经验**：
1. K_PHASE 必须是编译期常量（通过 C++20 templated lambda 实现），否则 `if (k_phase == 0) opsel<0> else opsel<1>` 在 tail 会生成双份 MFMA 代码路径，导致 +64 额外 MFMA + 31 spills + 20 scratch accesses，性能倒退 ~2.8%。
2. Tail 区域的 `k_phase` 是 runtime，HOIST_HI 路径不能在 tail 用（会爆代码）。tail 直接 fallback 到 `rcr_mma_scaled_from_packs_exact` 即可，因为 tail 每 block 只跑一次，v_lshr 不在关键路径。

## 剩余差距（2925.64 → 3070.93 约 145 TFLOPS / 4.7%）

Main loop 已无 v_lshr，结构上与 FP8 几乎一致（仅多 6 个 scale buffer_load）。剩余差距主要来自：
- 6 个 scale `buffer_load` 的 issue 开销
- 额外 2 个 `s_waitcnt`
- 无法继续压缩 VGPR（256 hard limit，254 已接近极限）

## 进行中 / 下一轮

**第三轮评审 (2026-04-17) — 三条路径全 REJECT，不 commit 代码**

| 路径 | 结果 | 采纳？ |
|---|---|---|
| Dev A — scale cross-iter prefetch (MXFP8_RCR_EXACT_PQ_SCALE_PREFETCH_N1_ENABLE) | scope=0 full ring: 4 spills, A/B −2.26%; scope=1 B-only spill-free: A/B −1.58%。真实 VGPR 254，2 VGPR headroom 根本不够。 | **拒绝** |
| Dev B — tail compile-time K_PHASE dispatch (MXFP8_RCR_EXACT_PQ_TAIL_DISPATCH_ENABLE) | 正确性+资源全 clean（VGPR 254 / Spills 0 / LDS 128 KB / Occ 2 不变），asm 少 12 tail v_lshr；但 reviewer GPU7 A/B 10+20 rounds Δ = −0.04% ~ +0.18%，低于 +0.25% 噪声门槛 | **拒绝**（技术正确但噪声级） |
| Dev C — KPAIR 2× unroll (MXFP8_RCR_EXACT_PQ_KPAIR_UNROLL2_ENABLE) | body 翻倍 → live-range 爆 256 VGPR，51 spills，A/B −54.87% | **拒绝** |

**关键纠错**：真实 8-wave PQ scaled kernel = **VGPR 254 / LDS 131 KB**（不是之前决策者读错的 212 VGPR / 139 KB，那是 outer dispatcher）。headroom 仅 ~2 VGPR。

**第四轮评审 (2026-04-17) — 两条路径 REJECT**

| 路径 | 结果 | 采纳？ |
|---|---|---|
| Dev D — LDS-cached scales `MXFP8_RCR_EXACT_PQ_SCALE_LDS_ENABLE` 实测 | 编译过，VGPR 243 / LDS 132 KB / occ 2；smoke 256 PASS；但 8192 formal A/B −0.76%（5 runs 全 slower）且 **determinism FAIL**（max abs 1.43）。SCALE_LDS 设计是替代 PIPELINE_SCALE，叠加反而在 tail 加 ds_write/s_barrier 拖慢；LDS 同步仅靠 compiler fence 不足 | **拒绝（regression + det fail）** |
| Dev E — sched V2 松 `TK_WAIT_VMCNT(6→8)` 在 HOIST_HI 之上 | 正确 / 资源不变；GPU1 A/B 10 runs Δ +0.11%，Welch-t 0.58，纯噪声 | **拒绝（marginal）** |

**第五轮评审 (2026-04-17) — 三条路径 REJECT**

| 路径 | 结果 | 采纳？ |
|---|---|---|
| Dev F — AGPR accumulator per-MFMA `"+a"` (MXFP8_RCR_EXACT_PQ_AGPR_ACC_ENABLE) | VGPR 254→128 / AGPR 0→128 但 **Spills 0→13 / Scratch 56 B/lane**；285 V↔A shuffles per kpair；GPU3 A/B 10 runs Δ −2.19%。要破局需重写 `_row`/`_impl` 为单 asm 块 fuse 8-32 MFMA，不是本轮 scope | **拒绝（regression）** |
| Dev G — scale L2 cache-policy hint sc0 (MXFP8_RCR_EXACT_PQ_SCALE_L2_HINT_ENABLE) | 12/238 buffer_load 用 sc0，正确+资源不变；GPU4 A/B 15/20 清理后 Δ +0.066% Welch-t 0.20。sc1/nt 更差。cache-policy 轴已饱和 | **拒绝（marginal）** |
| Dev G2 — scale buffer_load b32×2 → b64 合并 (MXFP8_RCR_EXACT_PQ_SCALE_LOAD_B64_ENABLE) | 6 scale dwords 分布在 6 个独立 SRD，最小间距 8192 B，b64 需要 X 与 X+4 同 SRD → **结构不可行**；需重设计 `preshuffle_scale_matrix_mfma16` 影响所有 MXFP8 变体 | **拒绝（broken）** |

**第六轮评审 (2026-04-17) — 一条路径 REJECT**

| 路径 | 结果 | 采纳？ |
|---|---|---|
| Dev H — 强制 occupancy=1 (MXFP8_RCR_EXACT_PQ_FORCE_OCC1_ENABLE) | **编译器忽略 request**（occ 仍是 2 waves/SIMD）：一个 512-thread block 在 4 SIMD CU 上**算术最小** occupancy 就是 2 waves/SIMD，无法再降。Step 1 A/B +0.023% 噪声；Step 2 尝试叠加 pipeline 扩展 → 63 spills / −63.36%。Occupancy-knob 轴已饱和 | **拒绝（broken）** |

### 还没被验证为死路的方向（下一轮唯一剩余）

- [ ] **SCALE_LDS 替代式实现**（非叠加）：彻底替代 PIPELINE_SCALE 的 SGPR-SRD path；用真正的 `__builtin_amdgcn_s_barrier()` 前后包 ds_write/ds_read 解决 determinism；<100 行改不完，高风险结构改造
- [ ] **AGPR accumulator fused asm block**（8-32 MFMA 融进单 asm）：破 Dev F 的 per-MFMA 边界开销；改写 `rcr_mma_scaled_from_packs_opsel_phase_{row,impl}`；会破坏 HOIST_HI 的 templated-lambda 约定；高风险大改造
- [ ] **preshuffle_scale_matrix_mfma16 layout 重设计**：让 6 scale dwords 相邻 8-byte 组装载入 → `buffer_load_b64` 可用；影响 RRR/CRR/4-wave + Python 参考
- ~~主动下调到 occupancy=1~~ **已证明不可能**：8-wave 512-thread block 在 4-SIMD CU 上算术最小 occ 就是 2 waves/SIMD，不是 flag 能改的。要 occ=1 需换成 4-wave 256-thread block（另一个 kernel），或 2-wave 128-thread block（完全重写）
- [ ] **bank conflict / MFMA utilization profiling**（`rocprofv3 -i`）：145 TFLOPS 里有多少是 MFMA 利用率，多少是 latency stall

### RCR 本轮结论
MXFP8 RCR 从 `feat/mxfp8-only` 分支的起点 2737 TFLOPS 一路推到 2925.64 TFLOPS，**已达到当前结构约束下可微调的上限**。剩余 145 TFLOPS 差距只能靠**结构性重构**（任选一条高风险大改造）去摸。非结构性的调度/cache/小 flag/occupancy 尝试全部饱和。

---

## RRR / CRR 95% 任务（进行中）

### GPU7 实测 baseline (2026-04-17)

RRR PQ：**2794.26 TFLOPS** (95.48% of RCR) → 已达 95%，本轮不动

CRR PQ：**2737.94 TFLOPS** (93.55% of RCR) → 差 **42.34 TFLOPS (1.55%)** 才到 95% gate (2780.28)

### CRR 差距根因

`crr_mxfp8_exact_8wave_fastpath.inc` L407-461 主循环在每奇数 k 对 6 个 scale packs 做 C++ 层 `>> 16` shift（L411-420），这映射到 6 × `v_lshrrev_b32` per kpair —— **跟 RCR pre-HOIST_HI 完全同构**。

`crr_mma_scaled_base<opsel_a, opsel_b>` 已经支持 2-bit opsel（byte-select 在 bit 1），`crr_exact_cA_with_b1_interleave_raw_phase<K_PHASE, ...>` 和 `crr_mma_scaled_phase<K_PHASE>(...)` 等 compile-time 模板化 helper **代码里已经有**，只是主循环没用。

### CRR 方向

- [x] **PIPELINE_SCALE 默认开**（commit `8934e95c`）：reviewer GPU7 验收 OFF 2733.91 → ON 2740.55 (+0.243%)，VGPR 247→232（−15），spills 0→0，occ 2，SNR 49.60 dB，det 3/3。**未到 gate**（差 39.73 TFLOPS）但是 strict win 且为后续优化释放 15 VGPR headroom
- [x] **R7-R9：HOIST_HI 路径架构性不可行**（5 attempts: Dev A, B, C, H, I 同 256 VGPR ceiling）。详见下方 R7-R9 评审记录
- [ ] **未来方向（结构性，本会话不做）**：CRR 4-accumulator pattern 重构（合并 cA/cB/cC/cD 减寄存器）；或 KPAIR_LOOP 移植 CRR；或重做 `crr_exact_cA_with_b1_interleave` helper（拆掉 8 个独立 MFMA）。这些都是大型重写
- [ ] RRR 保持观察，若后续因编译器变化跌破 95% 再补

---

## 成功条件

### 长期（RCR）
- MXFP8 RCR ≥ 3070.93 TFLOPS（per-iter 协议）
- SNR > 48 dB
- 3 次 determinism 一致
- FP8 baseline 无回归

### 本轮（RRR / CRR）
- **RRR PQ 8192³ ≥ 2780.28 TFLOPS**（当前最佳 MXFP8 RCR × 0.95）
- **CRR PQ 8192³ ≥ 2780.28 TFLOPS**
- SNR > 48 dB
- 3 次 determinism 一致
- 不回归 RCR / FP8

## 运行记录

- `0a3eafb6` Remove all MXFP4 and Gluon kernels on mxfp8-only branch
- `bc0081e5` Tidy repo: skills, gitignore, agent team runbook
- `f943af92` HOIST_HI opsel 消除 main-loop v_lshr（reviewer GPU7 验收 2925.64，A/B +17.80；GPU1 head-to-head +25.67）。main-loop `v_lshr` 0，spills 0，VGPR 256→254，occupancy 2。构建 flag 加 `-DMXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE=1`。
- **第三轮 (2026-04-17)**：三条路径（scale prefetch n+1 / tail compile-time dispatch / KPAIR 2× unroll）全 reject。Baseline 稳定在 2917–2932 TFLOPS。无代码 commit，仅文档修正 baseline VGPR 数字（254，不是 212）+ 写入三条新 dead-end。
- `b964c110` Round-3 dead-ends: correct baseline VGPR = 254, not 212（仅文档 commit，code 不变）
- **第四轮 (2026-04-17)**：两条路径全 reject。Dev D 实测 SCALE_LDS 叠加：−0.76% 且 determinism FAIL（此前仅"未验证"，现有硬数据）。Dev E 实测 sched_barrier v2 `vmcnt(6→8)`：+0.11% 噪声级。写入 SKILL dead-ends，不 commit 代码。
- `3fe9c759` Round-4 dead-ends: SCALE_LDS measured (regression + det fail), SCHED V2 noise（仅文档 commit）
- **第五轮 (2026-04-17)**：三条路径全 reject。Dev F 实测 AGPR per-MFMA `"+a"`：−2.19%（per-MFMA 边界 V↔A 切换爆 285 次 shuffle + 13 spills）。Dev G 实测 sc0 cache hint：+0.066% 噪声。Dev G2 证明 `buffer_load_b64` 合并在当前 scale layout 下**结构不可行**（6 SRD 间距 8192 B）。剩余只能靠结构性重构。
- `5d31c742` Round-5 dead-ends: AGPR per-MFMA, scale L2 hint, b64 merge broken（仅文档 commit）
- **第六轮 (2026-04-17)**：Dev H 证明 occupancy=1 在 512-thread 8-wave block 上**架构性不可能**（CU 只有 4 SIMD，一个 512-thread block 最少占 2 waves/SIMD）。加 pipeline 扩展反而 63 spills / −63%。Occupancy 轴彻底关闭。
- `77370d3f` Round-6 dead-end: occupancy=1 architecturally impossible for 8-wave（仅文档 commit）
- **第七轮起 (2026-04-17)**：任务转向 RRR / CRR 95%-of-RCR gate。GPU7 实测三 layout：RCR 2926.61 / RRR 2794.26 (95.48%，已达标) / CRR 2737.94 (93.55%，差 42.34)。CRR 差距根因：主循环每奇数 k 做 6 × `scale_pack >> 16` → `v_lshrrev_b32`，跟 RCR pre-HOIST_HI 同构。计划：移植 HOIST_HI opsel 思路到 CRR 主循环（flag `MXFP8_CRR_EXACT_PQ_HOIST_HI_ENABLE`）。Dev CRR-A 已派活（worktree `/tmp/wt-crr-a`，GPU0），被打断未完成。
- **第七轮 round-1 重启 (2026-04-17)**：续派 Dev A/B/C 三 HOIST_HI 变体（CRR HOIST_HI K_PHASE templated lambda）—— 全 FAIL：CRR 4-accumulator (cA/cB/cC/cD) + 重 `crr_exact_cA_with_b1_interleave` 在 K_PHASE 模板化时 inlined codegen 翻倍，VGPR 247→256+ 含 53–173 spills，A/B −54% 到 −67%。
- **第七轮 round-2 (2026-04-17)**：Dev D/E/F/G/H 五个新方向：
  - **Dev D — PIPELINE_SCALE only**：✅ +0.243% on GPU7（详见 commit `8934e95c`），VGPR 247→232（−15），spills 0
  - **Dev E — sched_barrier (no body change)**：噪声级，Δ ≈ 0%。**结论：v_lshr 不在 critical path**
  - **Dev F — `__noinline__` outlined helper**：catastrophic：correctness 46% / scratch 800–888 B/lane / −97%。AMDGPU calling convention 无法跨 noinline 边界保持 4 个 accumulator live
  - **Dev G — runtime branch HOIST_HI**：3 变体全 FAIL gate；V1 `if/else` 256 VGPR + 13 spills，V2 manual unroll 256 + 347 spills，V3 with scopes 同 V2；A/B −94% / correctness FAIL
  - **Dev H — PIPELINE+HOIST combo**：256 VGPR + 29 spills；PIPELINE 的 SRSRC（24×32-bit）与 HOIST_HI 双 phase packs live 互相挤兑
- **第七轮 round-3 (Dev I) (2026-04-17)**：HOIST_HI + PIPELINE_SCALE + `CRR_EXACT_INTERLEAVE_B1_LDS=0`（删掉重 8-MFMA interleave，按 RRR 简单 4-MMA 结构走）：FAIL，VGPR 256 + 145 spills + 332 B/lane scratch。**确认架构性 ceiling**：CRR baseline 247 VGPR 只有 7 headroom，K_PHASE 模板化 4 MMA × 2 phase = 8 inlined MMA blocks 必然吃掉 9–25 VGPR
- `8934e95c` **MXFP8 CRR PIPELINE_SCALE default ON**（+0.243%，frees 15 VGPR）— 含 R7–R9 dead-end 总结
- **R7-R9 关键架构发现**：HOIST_HI K_PHASE templating 与 CRR 4-accumulator main loop **根本不兼容**。任何 templated body doubling 都会越过 254 VGPR cap，与是否叠加 PIPELINE_SCALE / 是否关 INTERLEAVE 无关。已在 5 个独立尝试（Dev A/B/C/H/I）观察到同一 256-VGPR 上限。CRR 要破 gate 必须做**结构性重构**（合并 accumulator / 或换 kernel 结构），非微调可达。
