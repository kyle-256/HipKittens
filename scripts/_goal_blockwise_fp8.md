# Goal — HipKittens FP8 blockwise GEMM (MI300X)

## 终极目标

**54 个生产形状（18 (M,N,K) × 3 段）每个 HK ≥ 1.25 × 本机 Triton**，且不破 49 dB SNR。

- Kernel: `kernels/gemm/fp8fp32/mi300x/blockwise_8192/blockwise.cpp`
- 段: fwd (RCR) / dgrad (RRR, 走 fwd) / wgrad (CRR, 走 fwd K-contig)

## 当前状态

实测分数会随 GPU VF / ROCm 版本 / Triton baseline 漂移, **跑 metric 拿当前数**:

```
python3 scripts/_metric_blockwise_fp8_target_shapes.py    # 全 54 形状
python3 scripts/_metric_blockwise_fp8_loser_shapes.py     # loser 子集 (~8× sensitivity)
```

最近一次 reference (snapshot, 会 rot):
- score ~860 / 1000, fwd ~77% / dgrad ~99% / wgrad ~82% (vs 1.25× 本机 Triton target)
- 总体 ~109% 本机 Triton (fwd 95% / dgrad 132% / wgrad 104%)
- 8192³ default: 49.59 dB SNR, ~770 TFLOPS, 236 VGPR / 2 wave/SIMD / 0 spill

### 18 个生产形状 (M, N, K) — 每个 × {fwd, dgrad, wgrad} = 54

| Model | Op | M | N | K |
|---|---|---:|---:|---:|
| LFM2-8B-A1B | GateUP | 8192 | 3584 | 2048 |
| LFM2-8B-A1B | GateUP | 16384 | 3584 | 2048 |
| LFM2-8B-A1B | GateUP | 32768 | 3584 | 2048 |
| LFM2-8B-A1B | Down | 8192 | 2048 | 1792 |
| LFM2-8B-A1B | Down | 16384 | 2048 | 1792 |
| LFM2-8B-A1B | Down | 32768 | 2048 | 1792 |
| Qwen3-235B-A22B | GateUP | 8192 | 3072 | 4096 |
| Qwen3-235B-A22B | GateUP | 16384 | 3072 | 4096 |
| Qwen3-235B-A22B | GateUP | 32768 | 3072 | 4096 |
| Qwen3-235B-A22B | Down | 8192 | 4096 | 1536 |
| Qwen3-235B-A22B | Down | 16384 | 4096 | 1536 |
| Qwen3-235B-A22B | Down | 32768 | 4096 | 1536 |
| DeepSeek-V3 | GateUP | 8192 | 4096 | 7168 |
| DeepSeek-V3 | GateUP | 16384 | 4096 | 7168 |
| DeepSeek-V3 | GateUP | 32768 | 4096 | 7168 |
| DeepSeek-V3 | Down | 8192 | 7168 | 2048 |
| DeepSeek-V3 | Down | 16384 | 7168 | 2048 |
| DeepSeek-V3 | Down | 32768 | 7168 | 2048 |

来源: `scripts/_shapes_target.py:SHAPES`。

## 已穷尽（设计决策记录 — 别再扫）

单旋钮空间已彻底闭。任何继续单旋钮 sweep 都会在噪声内 ±5 分波动。

| 旋钮 | 结论 |
|---|---|
| BLOCK_M / BLOCK_N | 已在 TUNED_REGISTRY 找到每形状最优 |
| NUM_WARPS | 4 / 8 之间已 per-shape 选择 |
| CHIPLET_CHUNK | 1 / 2 / 4 / 8 已 per-shape 选择 |
| BW_RAW_DRAIN (float2 raw partial) | 少数形状有 +5% 收益, 已 per-shape |
| BW_PRESCALE_BS (cluster-6 b_s 预折) | 多数 BM=128 fwd 配置受益, 已 per-shape |
| BLOCK_K=64 | K=7168 fwd 54-56% Triton (vs 当前 95%), FALSIFIED |
| KBPT=2 unpipelined (BW_KBPT2_A) | -2.26 dB 结构 bug, 已 close |
| KBPT=2 INTERLEAVED | correctness OK 但 perf 不及 KBPT=1 (最高 85% Tri vs KBPT=1 95%) |
| WGM > 1 | 全形状 FALSIFIED |
| MMA_QUAD_PRIO_RESET / HOIST_BS | FALSIFIED |
| PERSISTENT kernel | VGPR spill 爆掉, 要重写 per-tile state |

## 剩下值得攻（多周结构）

按 ceiling × 可能性排:

1. **KBPT=2 interleaved correctness 已 unblock 但 perf FALSIFIED**: rounds 118-122
   correctness 跑通到 49.59 dB SNR (round-119 发现 kittens `st<fp8, *, 256>`
   swizzle 路径有结构 bug, round-120 用 TWO 128-wide tile pair 绕过)。
   但 perf 在所有测试 geometry 下都打不过 KBPT=1 baseline。如继续攻: 试
   BLOCK_N=64 + REG_M=32 (更密 grid + 更小 partial), 或 fix 真正的 kittens
   swizzle bug 让原生 BW_KBPT2_A 路径直接 work (比 TWO-tile workaround
   高效)。

2. **Dedicated wgrad kernel** (低风险): 把 `dispatch_micro_wgrad` 从 fwd-routing
   拆成自己的 BM=128 native body。打 3 个 wgrad losers, 不动 fwd。3-4 轮。

3. **Persistent kernel** (高风险): round-73 因 VGPR spill 爆掉, 要重写降低
   per-tile state。

4. **32×32×16 MFMA flavor** (kittens 模板改动): 加 `mma_ABt(rt_fl<32,N,col>, ...)`
   overload, 解锁 BM=64。窄目标 (LFM2 wgrad)。

## 策略

1. **不再做单旋钮 sweep** — 已确认无收益。
2. **优先 attack #1** (KBPT=2 interleaved) — ceiling 最高且打主要 loser 家族。
3. **维护 `LOSER_PAIRS`**: 全 metric 跑后剔除 ≥105% 的对。
4. **始终保 49 dB SNR gate** — 破了就 revert。

## 分数估算

- 1 个 loser 95% → 110%: loser metric +8% (≈64 分), full metric +0.5% (≈5 分)
- 12 losers 全到 110%: loser ≈960, full ≈960
- 12 losers 全到 125%: loser 1000, full 980+

单旋钮 ceiling: **820-830 loser / 920-930 full**。
KBPT=2 interleaved 跑通: 估计 **900+ loser / 950+ full**。

## 文件

| Path | 用途 |
|---|---|
| `blockwise.cpp` / `test_python.py` / `Makefile` | kernel + harness + build |
| `scripts/_metric_blockwise_fp8_target_shapes.py` | 全 54 metric |
| `scripts/_metric_blockwise_fp8_loser_shapes.py` | loser-only metric (12 对) |
| `scripts/_task_blockwise_fp8{,_losers_extra}.md` | daemon 任务说明 |
| `scripts/auto_optimize_blockwise_fp8.py` | daemon (默认 loser metric + task) |
| `scripts/launch_auto_optimize_blockwise_fp8.sh` | nohup launcher |
| `scripts/_shapes_target.py` | 18 形状 + Triton baseline |
