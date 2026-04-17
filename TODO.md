# MXFP8 优化 TODO

目标：MXFP8 RCR 追平 FP8 per-tensor，协议为 `test_mxfp8_python.py` / `test_python.py` 内的 per-iteration sync + `output.zero_()`。

## 当前 baseline（per-iter sync，8192^3，RCR）

| 版本 | TFLOPS | SNR | Spills | 差距 |
| --- | ---: | --- | ---: | --- |
| **FP8 per-tensor RCR (target)** | **3070.93** | 49.61 dB PASS | 0 | 目标线 |
| **MXFP8 8-wave KPAIR+SRD+SCALE_PIPE+HOIST_HI(opsel) (当前最佳)** | **2925.64** | 49.60 dB PASS | 0 | −145.29 TFLOPS (−4.73%) |
| MXFP8 8-wave KPAIR+SRD+SCALE_PIPE (前 baseline) | 2897.66 | 49.60 dB PASS | 0 | −173.27 TFLOPS (−5.64%) |

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

- [ ] **4-wave 路径**（Dev A）— 有 14–44 VGPR headroom，可上 scale 软流水，是结构性突破可能的唯一方向
- ~~Dev C `.s` rewriter~~ — **已拒绝**：本轮实测 GPU2 A/B 仅 +5.94 TFLOPS (+0.18%)，且 HOIST_HI 已经从源头消除 6 个 `v_lshr`（彻底抹平 rewriter 的优化空间）。继续养 rewriter 维护成本不划算，关闭该方向，不再派活。
- [ ] 继续压缩剩余 6 scale buffer_load（考虑 LDS 缓存 scale，但 131 KB LDS 已接近 160 KB CU 限制）

## 成功条件

- MXFP8 RCR ≥ 3070.93 TFLOPS（per-iter 协议）
- SNR > 48 dB
- 3 次 determinism 一致
- FP8 baseline 无回归

## 运行记录

- `0a3eafb6` Remove all MXFP4 and Gluon kernels on mxfp8-only branch
- Dev B HOIST_HI opsel 消除 main-loop v_lshr（reviewer GPU7 验收 2925.64，A/B +17.80；GPU1 head-to-head +25.67）。main-loop `v_lshr` 0，spills 0，VGPR 256→254，occupancy 2。构建 flag 加 `-DMXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE=1`。
- `stash@{0}` WIP: agent team half-finished MXFP8 attempts (未测完，不要乱 pop)
