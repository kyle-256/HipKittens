# MXFP8 优化 TODO

目标：MXFP8 RCR 追平 FP8 per-tensor，协议为 `test_mxfp8_python.py` / `test_python.py` 内的 per-iteration sync + `output.zero_()`。

## 当前 baseline（per-iter sync，8192^3，RCR）

| 版本 | TFLOPS | SNR | Spills | 差距 |
| --- | ---: | --- | ---: | --- |
| **FP8 per-tensor RCR (target)** | **3070.93** | 49.61 dB PASS | 0 | 目标线 |
| **MXFP8 8-wave KPAIR+SRD+SCALE_PIPE (当前最佳)** | **2897.66** | 49.60 dB PASS | 0 | −173.27 TFLOPS (−5.64%) |

构建 flag（MXFP8 当前最佳）：
```
-DMXFP8_RCR_EXACT_8WAVE_FAST_ENABLE=1
-DMXFP8_RCR_EXACT_PQ_KPAIR_LOOP_ENABLE=1
-DMXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE=1
```

## 已完成

- [x] 创建 `feat/mxfp8-only` 分支
- [x] 删除所有 MXFP4 / Gluon kernel、测试、rewriter、产物（48 个文件）
- [x] 建立 FP8 / MXFP8 baseline
- [x] 决策者汇编级差距分析（见下方）
- [x] 第一轮 agent team 派活（Dev A / Dev B / Dev C）— 被中断
- [x] 半成品改动 stash 保存（`stash@{0}`，含 Dev A 的 4-wave KPAIR/SCALE_PIPE 骨架、Dev B 的 HOIST_HI opsel helper、Dev C 的 8-wave asm rewriter）
- [x] 清理 `.s` / per-run `.json` 等生成物，加强 `.gitignore`
- [x] 整理 `.cursor/skills`：删除 deprecated `fp8-strict-layout-tuning`，重命名 `mxfp8-mxfp4-layout-tuning` → `mxfp8-layout-tuning`，清除 mxfp4 知识，加入 commit-time 工作流

## 差距分析（8-wave inner loop 每 kpair，64 MFMAs）

| metric | FP8 | MXFP8 | Δ |
| --- | ---: | ---: | ---: |
| 总行数 | 403 | 431 | +28 |
| MFMAs | 64 | 64 | 0 |
| `buffer_load` | 16 | 22 | **+6 (scale loads)** |
| `ds_read` | 48 | 48 | 0 |
| `s_waitcnt` | 10 | 12 | +2 |
| `s_barrier` | 16 | 16 | 0 |
| `v_lshrrev_b32` | 0 | 6 | **+6 (scale remap，在关键路径 gap)** |
| VGPR | 252 | **256 (HARD LIMIT)** | +4 |
| LDS | 131 KB | 135 KB | +4 KB |
| Occupancy | 2 | 2 | 0 |

**结论**：MXFP8 每 iter 多 6 个 scale `buffer_load` + 6 个 `v_lshrrev_b32` remap。`v_lshr` 的输出被紧随其后的 MFMA 立即消费（RAW），不能下移；但可尝试上移到**前一 phase** 的 MFMA shadow。

## 进行中 / 下一轮

- [ ] **恢复 stash（第二轮）** — 用户要求继续 MXFP8 优化时再 `git stash pop`，逐个评估 Dev A/B/C 的半成品：
  - Dev A 4-wave KPAIR/SCALE_PIPE：编译 + formal benchmark（目标 > 2897）
  - Dev B `MXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE` opsel hoisting：build + smoke + formal
  - Dev C `rewrite_mxfp8_8wave.py` + `build_rewrite_8wave.sh`：完整 pipeline 试跑
- [ ] Reviewer 对每条路径跑门禁：smoke → formal → FP8 regression → SNR → determinism
- [ ] 胜出者 commit，一并更新 TODO.md + agent_prompt.md +（必要时）SKILL

## 成功条件

- MXFP8 RCR ≥ 3070.93 TFLOPS（per-iter 协议）
- SNR > 48 dB
- 3 次 determinism 一致
- FP8 baseline 无回归

## 运行记录

- `0a3eafb6` Remove all MXFP4 and Gluon kernels on mxfp8-only branch
- `stash@{0}` WIP: agent team half-finished MXFP8 attempts (未测完，不要乱 pop)
