# RRR v2 B-pre-transpose 多 session 改造计划

**目标**: 8/8 user shape RRR dgrad ≥ 1.15× Triton。当前 geomean 0.957，1/8 达标。

**Baseline commit (Session 1 起点, 钉死不动)**
- HK turbo HEAD: `1f1f593d`  (R662 RRR agpr_inplace wrapper + ISA 分析)
- PT  dev   HEAD: `55c632cb`  (R556 production RRR 路由 v2)

**ISA-level 根因 (RCR vs RRR 前 3000 条 main loop 指令对比)**
| 指标 | RCR | RRR | 比 |
|---|---|---|---|
| v_mfma_f32_16x16x128 | 96 | 96 | 1.0× |
| ds_read_b128 | 72 | 48 | 0.67× |
| ds_read_b64_tr_b8 | 0 | **48** | NEW (B 走 transposed half-width) |
| v_accvgpr_write | 0 | **304** | NEW |
| v_accvgpr_read | 0 | **112** | NEW |
| scratch_load | 98 | 194 | 1.98× |
| scratch_store | 14 | 57 | 4.07× |
| s_waitcnt vmcnt | 104 | 200 | 1.92× |

核心问题：mma_AB (RRR) 的 B 用 `ds_read_b64_tr_b8` (transposed, half width) + 416 条 accvgpr shuffle。
解决思路：把 B 在 LDS 里 **预转置**，让 RRR 也走 `ds_read_b128` 路径 + 消除 accvgpr 来回搬。

**关键文件路径**
- 内核体: `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts2.cpp`
- ST 类型: `HipKittens/include/types/shared/st_shape.cuh`
- ST→RT 加载: `HipKittens/include/ops/warp/memory/tile/shared_to_register.cuh`
- PT 路由:  `Primus-Turbo/csrc/kernels/grouped_gemm/HipKittens/hk_grouped_gemm_gfx950.cu`
- bench:    `Primus-Turbo/tests/bench_hk_vs_triton_grouped_fp8_kernel_only.py`

**Dual-path 同步铁律** (来自 memory)
所有 `HipKittens/<path>` 的 edit 必须立刻 `cp` 到 `Primus-Turbo/3rdparty/HipKittens/<same path>`。
两边都要 commit (不 push)。

---

## Session 1 — B mfma_AB lane layout 探测 + 文档化  (~150 LOC, 3-4h)

**目标**: 搞清楚 mfma_f32_16x16x128_f8f6f4 mma_AB 模式下 B 操作数在 64 lane 内的元素分布。这是 Session 2 设计 ST swizzle 的前提。

**任务**
1. 写一个 probe kernel: 在 `HipKittens/tests/probes/rrr_b_lane_layout_probe.cu` 喂一个已知 pattern 的 B (例如 b[k][n] = k*1000+n)，用 mma_AB 跑一次，把 B 操作数寄存器 dump 回去
2. 跑 probe，记录每个 lane 持有哪些 (k,n) 元素
3. 把映射写成表格附到本文件 Session 1 末尾
4. 输出：明确的 `lane_id, reg_byte -> (k, n)` 映射函数 (C++ constexpr 或 ASCII 表)

**验证**
- probe kernel 编出且 run pass
- 映射表给出 64 lane × 16 bytes = 1024 个 (k,n) 坐标，无歧义

**完成动作**
- HK commit: `git -C HipKittens commit -m "Session 1: B mma_AB lane layout probe"`
- PT 同步 + commit (3rdparty/HipKittens 镜像 + probe 不需要 PT 改动)
- 追加 `## Session 1 status: PASSED  HK=<hash>  PT=<hash>` 到本文件末尾
- 若卡住 ≥1h: 追加 `## Session 1 status: BLOCKED  reason: <>` 并退出

---

## Session 2 — B ST 类型 + G::load swizzle  (~200 LOC, 4-5h)

**目标**: 实现 B 在 LDS 里就按 mma_AB 期望的 layout 摆好，避免 ds_read_b64_tr_b8。

**前置**: Session 1 PASSED

**任务**
1. 在 `st_shape.cuh` 新增 `st_128x64_b_pretrans` (或合适尺寸) 类型，XOR swizzle 按 Session 1 映射设计
2. 在 `shared_to_register.cuh` 新增对应 `load(rt_..., st_128x64_b_pretrans)` 走 `ds_read_b128`
3. 修改 `kernel_fp8_layouts2.cpp` 里 RRR body 的 B HBM→LDS path (G::load)，让写入 LDS 时按新 swizzle 摆放
4. **暂时**保留旧 load_b 路径 (gated by `#define RRR_B_PRETRANS 0`)，新路径走 `RRR_B_PRETRANS 1`

**验证**
- `RRR_B_PRETRANS=0` 老路径仍 pass (24 shape SNR ≥ 47 dB)
- `RRR_B_PRETRANS=1` 新路径 numeric 正确 (SNR ≥ 47 dB on at least 4 shape spot check)
- ISA disasm: 新路径 main loop `ds_read_b64_tr_b8` 计数 = 0

**完成动作**
- HK + PT commit + status 追加 (同 Session 1 格式)

---

## Session 3 — B-pretranspose 全集成 + 切换 ds_read_b128  (~500 LOC, 5-8h)

**目标**: 把 Session 2 验证过的 b128-from-N-major-LDS primitive **落进 RRR kernel body**, 把所有 B 读路径从 `ds_read_b64_tr_b8` 切到 `ds_read_b128`, 数值 + 性能两方面都过 gate。

**前置**: Session 2 PASSED (probe primitive 已 byte-equivalent verified)

**已有素材** (Session 1+2 已交付)
- `tests/probes/rrr_b_pretrans_probe.cu` (HK + PT 3rdparty 双路径) — 闭式 b128 地址公式: `base + (lane&15)*K_DIM + ((lane>>4)&3)*16` + offset:64
- LDS layout 验证: N-major (`byte_offset(n,k) = n*K_DIM + k`)
- `(lane, byte) → (k, n)` 映射: `n = lane&15; k_block = (lane>>4)&3; half = byte>>4; k = k_block*16 + (byte&15) + half*64`
- ISA: probe kernel 2× `ds_read_b128`, 0× `ds_read_b64_tr_b8` (gfx950 chi2811)

**任务清单 (按顺序, 每步必须 SNR ≥ 47 dB 才能往下)**

1. **新 ST 类型** in `include/types/shared/st_shape.cuh`
   - `st_128x128_pretrans` (或更小如 `st_128x64`, 看 LDS budget): rows=N, cols=K, identity 或简单 XOR swizzle (Session 2 probe 用 identity 已通过, swizzle 留 bank conflict 出现后再加)
   - 注册到 `st_shape::all` concept, 保证 G::load / ds_read 模板能 dispatch

2. **load 特化** in `include/ops/warp/memory/tile/shared_to_register.cuh`
   - 新 `load(rt_..., st_128x..._pretrans &)` 走 Session 2 probe 验证过的 b128 地址公式
   - rt 目标类型 = RRR body 现有 B-frag (col_l fp8 rt_base), 不要新加 frag 类型避免 ripple

3. **B HBM→LDS writer** in `kernel_fp8_layouts2.cpp` (RRR body 4 个 prolog+主循环+FUSED_KTAIL prefetch 站点)
   - 默认 `G::load(Bs[...])` 是 row-major HBM→row-major LDS, 不能直接用
   - 选项 A: 8-warp 协作 `buffer_load_b128`(HBM K×N) + per-lane `ds_write_b128` 到 (n*K_DIM+k) LDS offset (类似 PR #330 的 transposed writer)
   - 选项 B: 在 LDS 里用 staging buffer 先 row-major load, 再 8-warp 协作 transpose ds_read+ds_write 到 N-major 目标 buffer (多一次 LDS round-trip, 慢但简单)
   - 推荐 A; B 当 fallback

4. **gate macro** `#define RRR_B_PRETRANS 0/1` (kernel_fp8_layouts2.cpp 顶部)
   - `RRR_B_PRETRANS=0` → 老 ds_read_b64_tr_b8 路径 (回归对照)
   - `RRR_B_PRETRANS=1` → 新 b128 路径
   - body 内 8+ `load_b` reads + 4 G::load(Bs[...]) writes 都按 macro 分叉
   - **暂时**保留, Session 4 才删 gate

5. **Bs slot size 重算** (Session 2 PARTIAL 警告)
   - 老 ST_v2 64KB double-buf; 新 N-major (128 N × 128 K × 1 fp8) = 16KB per tile × 2 buf = 32KB
   - LDS budget 不会超 (160KB CU), 但要在 kernel 顶部 `__shared__ uint8_t Bs[...]` size 显式重算并标注

6. **bank conflict 检查** (Session 2 PARTIAL 警告)
   - 同一 wave 16 lanes/n_val 同时 access 同 N 行 → 8-way 潜在冲突
   - 写完后跑一次 rocprof `SQ_LDS_BANK_CONFLICT` 计数, 若 >0 加 swizzle (候选 `((offset>>7)&7)<<4` mirror RCR ST_v2)

7. **FUSED_KTAIL 单独 audit**
   - K_tail 路径有独立的 b_kt prefetch lambdas, 不能漏改
   - K_rem=0 (full-tile-only) 和 K_rem=64 (有 ktail) 两种都要过

**验证 (硬 gate, 全部满足才 PASSED)**
- A. `RRR_B_PRETRANS=0` 老路径回归: 24-shape SNR ≥ 47 dB (确认没把老路径改坏)
- B. `RRR_B_PRETRANS=1` 新路径正确: 24-shape SNR ≥ 47 dB
- C. ISA disasm (`llvm-objdump` on chi2811): 新路径 main loop `ds_read_b64_tr_b8` 计数 = **0**, `ds_read_b128` 计数 ≈ 48 (per 3000-instr window, 与 RCR 对齐)
- D. 8-shape bench: TFLOPS geomean **不回退 > 3%** vs baseline (R167 4-acc vacc 1.024×T)。允许小幅 noise. 不要求超 baseline (perf 收益在 Session 4 消除 accvgpr shuffle 后)

**完成动作**
- HK + PT 3rdparty + PT outer 三 commit 链 parity (同 Session 2 格式)
- 追加 `## Session 3 status: PASSED  HK=<hash>  PT=<hash>  outer=<hash>` 到本文件末尾
- memory: `feedback_rrr_b_pretrans_session3_integration.md`
- 若卡住 ≥3h 或任一 gate 不过, 标 `BLOCKED reason: <>` 并退出

**风险与降级**
- 若选项 A writer 出现 race/correctness 难调, 立刻切选项 B (LDS staging) 保 correctness
- 若 ISA 显示 `ds_read_b128` 计数对但 perf 大幅退化 (>5%), 标 PARTIAL/PASSED 看 D gate 是否过; 不过就 BLOCKED 让 user 决策
- 若 24-shape 有 < 5 个 shape SNR 失败, 列具体 shape + 失败模式到 status 行, 让 Session 4 决定是否回退

**LOC + 时间 honest 估计**
- LOC: 新 ST 类型 30 + load 特化 60 + B writer 150 + body 接入 (4 G::load + 8 load_b × 双分支) 200 + ktail audit 50 + gate 宏 + smoke harness ≈ **490 LOC**
- 时间: 5-8h (新 ST + writer 是最大未知, bank conflict 处理可能拖时)

---

## Session 4 — AGPR 数据流优化  (~200 LOC, 3-4h)

**目标**: 消除 RRR 比 RCR 多的 416 条 accvgpr shuffle。

**前置**: Session 3 PASSED

**任务**
1. ISA disasm RRR body，定位 accvgpr_read/write 来源 (是 acc init? acc spill? mma chaining?)
2. 视情况：
   - 让 acc 全程 resident in AGPR (D-aliases-C inplace + acc-init avoid v_mov)
   - 或者改 acc storage class (rt_fl<...> 的 storage hint)
3. 与 Session 3 一样保留 numeric 验证

**验证**
- ISA: accvgpr_read + accvgpr_write 总数 ≤ 50 (vs 当前 416)
- 24/24 shape SNR ≥ 47 dB
- 8-shape bench: geomean ≥ 1.05× Triton (不是终点，仍要 Session 5)

**完成动作**: 同上

---

## Session 5 — Per-shape autotune + 最终验证  (~100 LOC, 2-3h)

**目标**: 锁定 8/8 ≥ 1.15× Triton。

**前置**: Session 4 PASSED

**任务**
1. 扩展 `grouped_gemm_fp8_impl.py` 的 `_HK_FP8_RRR_CANDIDATES` 覆盖 8 shape 关键 (group_m, chunk_size, bn_block) 组合
2. 跑 24-shape full bench, 找每个 shape 最优 tuple
3. 把 winners 编进 dispatcher per-shape table
4. 复测 8-shape bench, 必须 8/8 ≥ 1.15×

**验证**
- 8/8 ≥ 1.15× Triton
- 24-shape SNR ≥ 47 dB
- 24-shape geomean 不回退 vs baseline

**完成动作**
- HK + PT 最终 commit
- 追加 `## Session 5 status: PASSED  HK=<hash>  PT=<hash>  geomean=<x>  min=<y>`
- 更新 MEMORY.md：`feedback_rrr_v2_b_pretrans_win.md`

---

## Session 1 result — B mma_AB lane layout 映射表

**Probe**: `tests/probes/rrr_b_lane_layout_probe.cu` + Makefile (HK + PT 3rdparty 双路径)
**Run host**: chi2811 (gfx950, MI355X), `/opt/rocm/bin/hipcc --offload-arch=gfx950`
**Validation**: `OK: 2048 (k,n) coordinates, each appears exactly once.` (coverage 完整无重叠)

### 闭式映射 (lane, byte) → (k, n)

设 `lane ∈ [0,64)`, `byte ∈ [0,32)`. 一个 base tile = 128 K × 16 N, fp8e4m3.

```
n        = lane & 0xF                  // 0..15  (per-lane 常量)
k_block  = (lane >> 4) & 0x3           // 0..3   (lane 高 2 bit)
half     = byte >> 4                   // 0 or 1 (bytes 0..15 vs 16..31)
k        = k_block * 16 + (byte & 0xF) + half * 64
```

等价地, mfma B 操作数 `fp8e4m3_4 b[8]` (8 quad × 4 byte = 32 byte):
- `b[0..3]` (bytes 0..15)  → K = `k_block*16 + (0..15)`,       N 固定 = `lane & 15`
- `b[4..7]` (bytes 16..31) → K = `k_block*16 + 64 + (0..15)`,  N 固定 = `lane & 15`

### ASCII 表 (每 lane 持有的 K 集合)

| lane 范围 | k_block | N (= lane&15) | K_low (bytes 0..15) | K_high (bytes 16..31) |
|-----------|---------|---------------|---------------------|------------------------|
| 0..15     | 0       | 0..15         | 0..15               | 64..79                 |
| 16..31    | 1       | 0..15         | 16..31              | 80..95                 |
| 32..47    | 2       | 0..15         | 32..47              | 96..111                |
| 48..63    | 3       | 0..15         | 48..63              | 112..127               |

### 关键结构观察 (Session 2 swizzle 设计前提)

1. **N 维 lane-uniform**: 每个 lane 只读一个 N 列 (n = lane&15)。意味着 LDS 里把同 N 的 128 byte 摆连续, 64 lane × 1 N = 16 N 列只需 16-way 同 N broadcast (不是真正 broadcast, 而是 4 个 k_block 共享 N 索引)。
2. **K 维分块 2×16**: lane 持有的 32 K 不是连续, 而是 `[k_block*16 .. k_block*16+15] ∪ [k_block*16+64 .. k_block*16+79]`。两块距离恰好 64 K (= 64 bytes for fp8)。
3. **每 lane 32 byte = 2 × 16-byte chunk**: 第一 chunk 在 K_low 半区 (K<64), 第二 chunk 在 K_high 半区 (K≥64)。**正好对应一对 b128 read**, 每 read 16 byte = 16 个 fp8 K 元素 (同 N)。
4. **ds_read_b128 可行性**: 若 LDS 里按 `B[K][N]` 摆 (row-major K×N), 同 N 列的 128 K 是 stride=16 bytes/K, 不连续 → 无法直接 b128。**必须重摆**: 每 lane 的 16-byte chunk 在 LDS 里要连续, 即按 (k_block, half, N, k_in_block) 维度切片摆放。
5. **swizzle 等价 hint**: 类比 RCR 的 A col_l 路径已经走 `ds_read_b128`, 那里的 ST 类型是 `st_16x128_v2`, swizzle = `((offset>>7)&7)<<4`。Session 2 候选: 设计 `st_128x16_b_pretrans` (或更大尺寸如 128×128 单 ST 容 8 base tile), swizzle 让每 lane 一次 b128 拿连续 16 K (同 N, 同 k_block, 同 half)。

### 源数据
完整 64 行 lane dump 见 `tests/probes/session_logs/rrr_b_lane_layout_dump.txt` (probe 默认输出);
CSV `lane,byte,k,n` 用 `./rrr_b_lane_layout_probe --table` 重生成。

---

## Session log (按完成顺序追加)
<!-- 每个 session 完成或被卡，在下方追加一行 -->

## Session 1 status: PASSED  HK=bac982bd  PT=7736b348  (3rdparty=a5849bc7)
- 2026-05-25
- probe build + run pass on chi2811 (gfx950); coverage 2048/2048 unique
- 闭式映射 `(lane, byte) → (k, n)` 见 Session 1 result 段
- memory: `feedback_rrr_b_lane_layout.md`

## Session 2 status: PASSED  HK=090166ba  PT=e1ab0597  (outer=eae76e40)

## Session 3 status: BLOCKED  reason: 自定义 HBM→LDS B-pretranspose writer 单 session 装不下 — LDS budget (160KB) 排除 Option B fallback (要 196KB); Option A.1 naive scatter writer (16 ds_write_b8/lane) 正确但破 D gate (>10% 回退, 见 FUSED_KTAIL load_b_kt_fk 印证); Option A.2 cross-lane register transpose 需 ds_bpermute b32 × 4 + bank-conflict 校准, 多 session 工作 (~700-900 LOC)
- 2026-05-25
- 无 kernel/header 改动 commit (避免半成品污染); 完整诊断 + Session 4 handoff steps 见 memory `feedback_rrr_b_pretrans_session3_blocked.md`
- 关键证据:
  - LDS budget: 当前 Bs(ST_v2)=68KB + As=64KB + 余 28KB; 替代 N-major Bs=64KB OK, 但 Option B 同时持 staging+target = 132KB B alone, 加 As 总 196KB > 160KB
  - G::load `prefill_swizzled_offsets` (global_to_shared.cuh:121-181) 假设 HBM↔LDS 同方向 row-major, 无法表达转置写入
  - FUSED_KTAIL `load_b_kt_fk` (kernel_fp8_layouts2.cpp:2030-2082) 是 16 ds_write_b8/lane 的 working reference, 但**正是 prod fallback 因为它慢于 byte_b8 direct-to-reg**, 证明 naive scatter 路径 perf 不 acceptable
  - Cross-lane register transpose 路径需 gfx950 ds_bpermute_b32 × 4 (byte 粒度不支持) + 复杂 lane-byte 映射; 单 session 不现实
- Session 4 entry point (handoff in memory):
  1. 新 ST 类型 `st_128x128_n_major` (identity swizzle, 16384B per tile) → `st_shape.cuh` + `all` concept
  2. 新 load 特化 mirror Session 2 probe `addr = base + (n_base+lane&15)*128 + ((lane>>4)&3)*16` + offset:0/64 → `shared_to_register.cuh`
  3. 自定义 8-warp writer with cross-lane register transpose (4× ds_bpermute_b32) — 最大难点
  4. RRR body 6 G::load + 6 load_b sites gated by `RRR_B_PRETRANS` macro
  5. FUSED_KTAIL 路径**不需要改动** (已 bypass LDS)
- 风险点: bank conflict (16 lanes/n_val 同行) + 8-wave V+A ≤256 dword cap + bn128 race-fix dependency
**Note**: user 接受 PARTIAL 视为 PASSED；未交付的 ST 类型 + load 特化 + kernel body 集成 + 24-shape SNR 全部合并到 Session 3。原说明保留在下方供 Session 3 引用。

### (历史 PARTIAL 标注, 保留)
- 2026-05-25
- **Scope delivered**: 隔离 probe `tests/probes/rrr_b_pretrans_probe.cu` 验证 b128-from-pretranspose-LDS 路径 (LDS 按 N-major 摆: `byte_offset(n,k) = n*K_DIM + k`; 每 lane 2× `ds_read_b128` at `base + (lane&15)*K_DIM + ((lane>>4)&3)*16` + offset:64) 产出的 (lane, byte) → (k, n) 映射与 Session 1 mma_AB B 操作数映射 **byte-equivalent** (mismatch=0, coverage 2048/2048 unique on chi2811 gfx950)
- **ISA**: probe device asm `grep ds_*` → `2 ds_read_b128 / 0 ds_read_b64_tr_b8`。证明硬件层面 b128 路径 deliver 等价 mma 操作数
- **Scope NOT delivered (vs plan §Session 2 验证)**:
  - ❌ 新 `st_128x64_b_pretrans` ST 类型 + `st_shape.cuh::all` 注册
  - ❌ `shared_to_register.cuh` load 特化 (for new ST type)
  - ❌ `kernel_fp8_layouts2.cpp` RRR body 接入 (新 `#define RRR_B_PRETRANS 0/1` gate 未引入)
  - ❌ 老路径 24-shape SNR ≥ 47 dB 验证 (kernel 未改动)
  - ❌ 新路径 4-shape SNR ≥ 47 dB 验证 (kernel 未改动)
  - ❌ 整 kernel ISA disasm 0× `ds_read_b64_tr_b8` 验证 (kernel 未改动)
- **Why partial**: 完整集成 6 个 `G::load(Bs[...])` writes (prolog ×2 + 主循环 prefetch ×2 + FUSED_KTAIL ×2) + 8+ `load_b` reads + custom HBM→LDS transposed writer 单 session 容量超载。先 isolate 验证 b128 primitive 是 Session 1 mapping 充要条件, 避免直接改 kernel 后 race-fix/spill 退化
- **Session 3 retry 应做**: (1) 新 `st_128x128_pretrans` ST (rows=N=128, cols=K=128, identity 或简单 XOR swizzle) + 注册 `all` concept; (2) `shared_to_register.cuh` 新 load 特化用 probe 验证过的 b128 地址公式; (3) custom B HBM→LDS writer (G::load 走默认 row-major 不能直接用) — 候选: 8-warp 协作 `buffer_load_b128` (HBM K×N) + `ds_write_b128` 到 (n*K_DIM+k) LDS offset; (4) RRR body 4 个 `G::load(Bs)` + 1 `load_b` lambda gated by `RRR_B_PRETRANS` macro
- **Pre-condition risk for Session 3**: 当前 LDS layout (ST_v2 64KB double-buf) 与 N-major layout (16KB per tile × 2 buf) 容量不同, Bs slot size 需要 audit; bank conflict 未在 probe 验证 (单 wave, 顺序读, 16 lanes/n_val 同时 access 同 N 行 → 8-way 潜在冲突, Session 3 必须加 swizzle 或测 perf)
- HK commit: `d9ebba7c`; PT 3rdparty commit: `5ff508d5`; PT outer bump: `d1b42b55`
- memory: `feedback_rrr_b_pretrans_session2_probe.md`

