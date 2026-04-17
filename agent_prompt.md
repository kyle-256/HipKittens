# Agent Team Runbook — MXFP8 RCR → FP8 per-tensor Parity

## 总则

- 分支：`feat/mxfp8-only`
- 工作目录：`analysis/fp8_gemm/mi350x`
- GPU：8× MI355X (gfx950)，**formal 验收固定 `HIP_VISIBLE_DEVICES=7`**
- Skill：
  - `.cursor/skills/mxfp8-layout-tuning/SKILL.md`
  - `.cursor/skills/fp8-per-tensor-layout-tuning/SKILL.md`
- 测试协议：`test_mxfp8_python.py` / `test_python.py` 内的 **per-iteration `torch.cuda.synchronize()` + `output.zero_()`**，`warmup=100, iters=200`

## 核心约束

1. 不破坏 FP8 per-tensor baseline（每次 formal 必须附 FP8 回归确认）
2. 只改 MXFP8 相关文件（`kernel_mxfp8_layouts.cpp`，`*_mxfp8_*.inc`，`kernel_mxfp8_4wave_rewrite.cpp`，`rewrite_mxfp8*.py`，`build_rewrite*.sh`）
3. 每次改动必须过门禁：smoke OK → formal 8192^3 OK → SNR > 48 dB → 3 次 determinism 一致
4. 有提升才 commit，Commit 时**必须同步更新** `TODO.md` + `agent_prompt.md` +（如有 durable finding）SKILL
5. 禁止提交 `*.so`、`*.s`、`*_layout_results_*.json`、`.bak*`、`gpucore.*`、`__pycache__` 等（`.gitignore` 已覆盖）
6. 每个子 agent 使用不同 `HIP_VISIBLE_DEVICES` 以免 GPU 冲突：Dev A → 0，Dev B → 1，Dev C → 2，Reviewer/formal → 7

## Baseline (2026-04-17)

| 版本 | TFLOPS | SNR | VGPR / AGPR / Spills / LDS |
| --- | ---: | --- | --- |
| **FP8 RCR (target)** | **3070.93** | 49.61 | 252 / 0 / 0 / 131 KB |
| **MXFP8 8-wave KPAIR+SRD+SCALE_PIPE+HOIST_HI(opsel)** | **2925.64** (GPU7 reviewer, SNR+det PASS) | 49.60 | 254 / 0 / 0 / 131 KB |
| MXFP8 8-wave KPAIR+SRD+SCALE_PIPE (前 baseline) | 2897.66 | 49.60 | 256 / 0 / 0 / 135 KB |
| 差距 (当前最佳 vs FP8) | −145.29 (−4.73%) | | |

> **VGPR 读数陷阱**：`-Rpass-analysis=kernel-resource-usage` 会为每个符号各报一次；MXFP8 RCR PQ 路径的真实 hot kernel 是 `rcr_exact_8wave_scaled_kernel<Lb1>`（VGPR **254** / LDS **131 KB**）。外壳 `gemm_kernel<Layout0,*>` 只是 dispatcher，显示 VGPR 212 / LDS 139 KB，**不是**可用于 headroom 推断的数字。任何基于「40 VGPR headroom」的优化提案都是错的，请以 scaled kernel 符号的 remark 为准。

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
- **已有开关**（按优先级实验）：
  - `MXFP8_RCR_EXACT_PQ_PHASE_U16_CACHE_ENABLE=1`
  - `MXFP8_RCR_EXACT_PQ_REMAP_ONCE_ENABLE=1`
  - `MXFP8_RCR_EXACT_PQ_SCALAR_PHASE_PACKS_ENABLE=1`
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
