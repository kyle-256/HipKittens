# HK fp8 grouped V2 Plan (Campaign D)

工作目录: `analysis/fp8_gemm/mi350x/`
新代码: `kernel_fp8_layouts2.cpp` (与 v1 `kernel_fp8_layouts.cpp` 并存)
日期: 2026-05-23 (locked)

## §0 任务条件 (locked, user 拍板)

- **A1**: `spill=0` hard 目标, 范围 = 所有 BN=256/128 实例化路径 (RCR/RRR/CRR var_k × FUSED ktail 变体)。autotune 选不选到不豁免。
- **B**: baseline = `hk_gemm_fp8` (HK dense fp8)。
- **C**: vs `hk_gemm_fp8` **geomean ≤ 3% gap** (worst-shape 可放宽, geomean 是硬指标)。
- **wgrad 目标**: 比当前 HK var_k 再 +15% → vs Triton ≥ **1.79× geomean** (当前 1.556×)。
- **fwd / dgrad 目标**: vs Triton ≥ **1.15× geomean** (当前 RCR 0.943×, RRR 0.955×)。
- **持 8-warp MMA** (mandate, 不退 4-warp); persistent + CPU sync free (现 v1 已具备 architecture parity, P0 已确认)。
- **dual-run**: v1 / v2 并存, 直到 P4 删旧。
- **做一点 benchmark 一点**: 每 milestone 端点必须出 chi2811 KPI 表 + spill 元数据, 不达标不进下一 milestone。

## §1 Raw delta vs 目标 (来自 Bench agent 2026-05-23)

| op | raw geomean | 目标 | gap |
|---|---|---|---|
| RCR fwd vs Triton | 0.943× | 1.15× | +22pp |
| RRR dgrad vs Triton | 0.955× | 1.15× | +20pp |
| CRR var_k wgrad vs Triton | 1.556× | 1.79× | +15pp |
| RCR fwd vs hk_dense | ~0.72× | ≥0.97× | +25pp |
| RRR dgrad vs hk_dense | ~0.74× | ≥0.97× | +23pp |
| spill (BN=256 主路径) | 37/67 VGPR | 0 | 结构性 |

完整 24-shape × 3-op KPI: 远程 chi2811 `/tmp/bench_hk_vs_triton.log`。

## §2 路线图

### Phase P1 — RCR fwd (task #2, #20, #21, #22)

| ms | 改造 | 端点 KPI 门槛 |
|---|---|---|
| P1.0 | dual-run 骨架: `kernel_fp8_layouts2.cpp` 1:1 复制 `grouped_rcr_kernel_body` + PT adapter `hk_grouped_rcr_fp8_new` + Python `_grouped_rcr_v2` | 双跑 SNR ≥ 30 dB + perf parity ±3% |
| P1.1 | Ref-trick 池可移植子集 (Ref agent 表 #2/#4/#7/#12/#13/#14): outer-hoist per-group ptr, two-step prologue, store-side unpad, vmcnt<3> partial drain, readfirstlane 显式 pin, s_nop / sched_barrier 末尾。 | vs Triton geomean ≥ 0.97× |
| P1.2 | **spill=0 改造** (BN=256, ~400-600 LOC): mfma 16×16×128 → **32×32×128** (per-warp 64×128 → 64×64), AGPR 128→64, V 释放 64 dwords; K-loop body 联合改 (避 `[[fp8-rrr-32x32-flawed-premise]]` 单换 mfma 不降 spill 的坑); LDS layout 同步调。 | spill=0 且 geomean 不退 ≥ 0.97× Triton |
| P1.3 | **algorithmic** (closes hk_dense gap, ~100→800 LOC 递进): 先试 (b) dispatcher 路由分流 (K≥4096 → BN=256, K<4096 → BN=128); (b) 不够再上 (a) split-K cross-group B share。 | RCR fwd geomean ≥ 1.15× Triton **AND** vs hk_dense geomean ≤ 3% |
| P1.4 | autotune sweep + 决定 v1 替换 / 共存 | 全 24 shape SNR ≥ 25 dB |

### Phase P2 — RRR dgrad (task #3)

复用 P1 wrapper, 同 5 milestone。关键: P2.2 spill=0 还要解 RRR FUSED_KTAIL=true 67/272 worst-spill, 走 32×32 + FUSED 分支折叠成无独立 a_kt0/a_kt1 (类似 bf16 path)。

### Phase P3 — CRR var_k wgrad (task #4)

P3.0 dual-run; P3.1 trick 池; **P3.2 spill=0**: 现 32-41 / 132-168, bf16 ref 同 topology 仅 1 spill → fp8 操作数排布是源, 改 ST_v2 layout + fragment 类型对齐 bf16 path; P3.3 +15% (Triton 1.556× → 1.79×): 主要靠 spill=0 释放 occupancy + LDS 双 buffer 改 triple buffer (var_k K 短, prefetch 收益大)。

### Phase P4 — 删旧 v1 (task #5)
全 24 shape P1-P3 都达标后, 删 v1 dispatcher / kernel body / adapter / binding。

### Phase P5 — final bench (task #6)
Triton / hipBLASLt / CK 三方对照 + production smoke (MoE forward+backward end-to-end)。

## §3 技术路径细节

### §3.1 spill=0 (BN=256) lever

当前 BN=256 mfma 16×16×128 per-warp 64×128 acc = 128 floats / lane = 128 AGPR; 8w V+A ≤ 256 → V 仅 128 可用, 装不下 A/B fragment + scratch → spill 37。

换 **32×32×128**: per-warp 64×64 acc = 64 AGPR → V 192 可用, 双 fragment + scratch 全装下 → spill 0。

注意 `[[fp8-rrr-32x32-flawed-premise]]` 警告: 单换 mfma wrapper 不够 (per-warp output 面积 fixed 时 acc reg 不降), 必须 K-loop body 联合改 (per-tile streaming + LDS pre-cache)。所以 P1.2 = mfma wrapper + K-loop nest + LDS 三处协同。

### §3.2 ≤3% gap vs hk_dense (geomean) lever

raw 拉低 geomean 的 3 大 shape (dsv3-up B=4 M=4096 / dsv3-down B=16 M=4096 / qwen235b-down) 都是 bandwidth-bound (`[[fp8-rrr-h14]]` H14 已证 worst-shape physics-bound)。

algorithmic 候选 (按 user 拍板顺序试):
- **(b) dispatcher 路由分流** (~100 LOC, 低风险): K≥4096 → BN=256 path, K<4096 → BN=128 path。bn128 现已 spill=0 + race-fix 稳定; Triton 短 K 也走 BLK=128×128, 应能拉 geomean。
- **(a) split-K cross-group B share** (~500-800 LOC, 中风险): 小-M-per-group 时多 group 共享同 B tile, 减 HBM B 重 stream — 直接打 bandwidth ceiling。需新 launcher + PT-side grouped_gemm_fp8.py 改 group_offs 接口。
- **(c) chunk_size + GRID_MUL 联动**: 与 (a)/(b) 正交, P1.4 autotune sweep 内自然覆盖。

### §3.3 autotune A1 enforcement (user 拍 i)

P1.2 落地前: autotune **hard 删除** BN=256 实例化路径 (不仅 warn)。BN=128 现已 spill=0 可保留。这会暂时让 RCR fwd geomean 掉到 ~0.85× (BN=128 perf 较弱), 直到 P1.2 重开 BN=256 with spill=0。

## §4 文件骨架

```
HipKittens/analysis/fp8_gemm/mi350x/
  kernel_fp8_layouts.cpp        # v1 (现有 4641 LOC, 三 dispatcher 已 inline 化防 ODR)
  kernel_fp8_layouts2.cpp       # v2 NEW (P1.0 起步)
  PLAN_V2.md                    # 本文件

Primus-Turbo/csrc/kernels/grouped_gemm/HipKittens/grouped_gemm_fp8_hipkitten.cpp
  # 加 3 个 adapter: hk_grouped_{rcr,rrr,var_k}_fp8_new
  # include kernel_fp8_layouts2.cpp

Primus-Turbo/csrc/pybind/bindings_pytorch.cpp
  # 注册 3 个新 binding: _grouped_{rcr,rrr,var_k_crr}_v2
```

v1 dispatcher 已 inline 1 个 (`dispatch_grouped_rcr`); P1.0 起步前补 inline `dispatch_grouped_rrr` + `dispatch_grouped_var_k_fp8`。

## §5 风险点 (按概率)

1. **P1.2 mfma 32×32 单换不降 spill** — `[[fp8-rrr-32x32-flawed-premise]]` 已警告; 必须 K-loop body 一起改, 否则掉回 P1.1。
2. **P1.3 split-K cross-group share 改 dispatcher 接口** — PT-side `grouped_gemm_fp8.py` group_offs 语义可能不兼容, 需新 launcher。先走 (b) 风险低。
3. **autotune A1 enforcement 暂时退化** — P1.2 落地前 BN=256 全禁用会让 RCR geomean 掉到 ~0.85×; 这是 user 拍 (i) 的 acceptable 代价。
4. **wgrad +15% (Triton 1.556→1.79)** — var_k 现已 1.03-2.11× 跨 shape, 拉 geomean 23% 需 P3.2/P3.3 都成功; 若 spill=0 后 perf 不增 (释放 occupancy 但 LDS 没满 → bw bound), 需算法层补。
5. **dual-run 阶段 .so 体积膨胀** — v1 + v2 共存约多 2-3 MB; 不挡 P5 前。

## §6 不做的事 (call out)

- 不动 bf16 grouped 路径
- 不动 mxfp8 / mxfp4 路径
- 不引 CK fallback (`[[no-ck-fallback]]`)
- 不靠常数 sweep (`[[no-constant-sweep]]`) — 必须 disassemble + ISA-level 分析后才调
- 不动 `dev/kyle_mxfp8_gg_pr` 远程分支 (只读 reference)
- 不动 git config / 不 push / 不 --no-verify

## §7 Task 映射

- task #1 [completed] P0 plan
- task #2 [in_progress] P1.0 RCR v2 dual-run skeleton
- task #20 P1.1 RCR v2 trick 池
- task #21 P1.2 RCR v2 spill=0 (mfma 32×32 + K-loop)
- task #22 P1.3 RCR v2 algorithmic + 决定替换 v1
- task #3 P2 RRR dgrad (P2.0-P2.4)
- task #4 P3 CRR var_k wgrad (P3.0-P3.4)
- task #5 P4 删旧 v1
- task #6 P5 final bench

## §8 端点 KPI 模板 (每 milestone 必填)

| 项 | 数值 |
|---|---|
| shape coverage | 24 / 24 |
| SNR min (dB) | ≥ 25 |
| spill (V/A scratch) | 0 / 0 / 0 |
| vs hk_dense geomean | ≥ ? |
| vs Triton geomean | ≥ ? |
| chi2811 commit (HK + PT) | sha / sha |
| KPI raw | `/tmp/<milestone>.log` 路径 |

### P1.0 端点 KPI (2026-05-23, PASS)

| 项 | 数值 |
|---|---|
| shape × bn | 5 shape × 2 bn = 10 / 10 PASS |
| correctness | 9 bit-equal + 1 SNR 47.53 dB (gpt_oss bn=128) |
| perf parity ratio v2/v1 | ∈ [0.996, 1.020], max |Δ|=2% (median-of-7 trials, 50 warmup) |
| spill (v2 same as v1) | BN=256 RCR/RRR 37 / BN=128 全 0 (P1.2 范围) |
| smoke script | `Primus-Turbo/scripts/_smoke_p1_0_rcr_v2.py` |
| 新增文件 | HK `analysis/fp8_gemm/mi350x/kernel_fp8_layouts2.cpp` (1:1 cp v1, 4735 LOC) |
| 新增 binding | `hk_grouped_rcr_fp8_new` (PT inner adapter + pytorch wrapper + extension decl + m.def/m.impl, hip + non-hip 双写) |
| 修复 build 阻塞 | rm orphan `Primus-Turbo/csrc/pytorch/grouped_gemm/turbo_grouped_gemm_hip.cpp` (Campaign C revert 残留, gitignored 未自动清) |
| 下一动作 | task #20 P1.1 起步 (Ref-trick 池 6 item) |
