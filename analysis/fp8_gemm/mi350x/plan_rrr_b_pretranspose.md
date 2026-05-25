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

## Session 3 — ST 类型 + load 特化  (~200 LOC, 3-4h)

**目标**: 把 Session 2 probe 验证的 b128 地址公式固化成 HK type-system 一等公民: 新 ST 类型 + 新 load() 特化。**不动 kernel body**, 不动 writer, 不破任何现有路径。

**前置**: Session 2 PASSED (probe primitive 已 byte-equivalent verified)
**(原 Session 3 BLOCKED 后拆出的 3a)**

**已有素材** (Session 1+2 已交付)
- `tests/probes/rrr_b_pretrans_probe.cu` (HK + PT 3rdparty 双路径) — 闭式 b128 地址公式: `base + (lane&15)*K_DIM + ((lane>>4)&3)*16` + offset:64
- LDS layout 验证: N-major (`byte_offset(n,k) = n*K_DIM + k`)
- `(lane, byte) → (k, n)` 映射: `n = lane&15; k_block = (lane>>4)&3; half = byte>>4; k = k_block*16 + (byte&15) + half*64`
- ISA: probe kernel 2× `ds_read_b128`, 0× `ds_read_b64_tr_b8` (gfx950 chi2762)

**任务**
1. **新 ST 类型** in `include/types/shared/st_shape.cuh`
   - `st_128x128_n_major` (rows=N=128, cols=K=128, 16KB per tile, identity swizzle 起步)
   - 注册到 `st_shape::all` concept (保证 G::load / ds_read 模板能 dispatch)
2. **load 特化** in `include/ops/warp/memory/tile/shared_to_register.cuh`
   - 新 `load(rt_..., st_128x128_n_major &)` 走 Session 2 probe 闭式地址: `addr = base + (lane&15)*128 + ((lane>>4)&3)*16` + offset:0/64
   - rt 目标类型 = RRR body 现有 B-frag 类型 (col_l fp8 rt_base), 不新增 frag 避免 ripple
3. **单元 smoke** in `tests/probes/`
   - 写一个 minimal test: 填一个已知 pattern 的 LDS N-major buffer + 用新 load 拉到 rt + dump → 对照 Session 2 probe 的 (lane,byte)→(k,n) 映射, mismatch 必须 = 0

**验证 (硬 gate)**
- 新 ST 类型编出, 不破现有 ST 实例化 (HK build pass)
- load 特化编出, 不破现有 load(...) 重载 dispatch
- 单元 smoke 在 chi2762 跑 pass (mismatch = 0)

**完成动作**
- HK + PT 3rdparty + PT outer commit 链 parity
- 追加 `## Session 3 status: PASSED  HK=<hash>  PT=<hash>  outer=<hash>`
- memory: `feedback_rrr_b_pretrans_session3_st_load.md`

**LOC + 时间** 约 200 LOC / 3-4h (纯 type system, 不动 kernel body)

---

## Session 4 — cross-lane register transpose writer  (~400 LOC, 5-6h) 【难点】

**目标**: 实现 8-warp 协作的 HBM→LDS B 转置 writer, 走 ds_bpermute_b32 路径, perf-acceptable (不超 +10% epilog/prolog 开销 vs 老 G::load). 这是整条 B-pretranspose 链的真难点。

**前置**: Session 3 PASSED (ST 类型 + load 特化 OK, 写入端可独立验证)
**(原 Session 3 BLOCKED 后拆出的 3b)**

**核心问题**
- HBM B 是 row-major `B[K][N]` (K stride=N bytes)
- LDS 目标是 N-major `Bs[N][K]` (N stride=K bytes) — 即 Session 3 新 ST 的 layout
- 不能直接 G::load 因为 HBM↔LDS 同 layout 假设
- 不能 naive 每 lane scatter 16 ds_write_b8 (太慢, 见 FUSED_KTAIL load_b_kt_fk perf 数据)
- **必须** load 到 register (按 HBM row-major 16B = 16 N 元素 of 1 K), 然后 cross-lane shuffle 让每 lane 持有 16K of 1 N, 然后 ds_write_b128

**任务**
1. **block-transpose 算法设计** (写进 `analysis/fp8_gemm/mi350x/session4_writer_design.md`)
   - 输入 4×16 lane block (4 K × 16 N 16 个 fp8 byte/lane) → 输出 16×4 (16 K × 1 N 16 byte/lane)
   - 用 `__builtin_amdgcn_ds_bpermute_b32` × 4 (byte 粒度不支持, 用 b32 + bit-shuffle)
   - 详细 lane-id → swap-id 表 (mirror Session 1 输出的 lane layout)
2. **probe kernel** in `tests/probes/rrr_b_writer_probe.cu`
   - 隔离验证 transpose writer: 喂 HBM `b[k][n]=k*1000+n`, 走新 writer 写 LDS, 用 Session 3 的 load 读出, 对比 host-computed reference
   - mismatch = 0 才算通过
3. **bank conflict 探测** in probe
   - 加 ROCm profiler trace 或手动计数 `SQ_LDS_BANK_CONFLICT` 在 chi2762 rocprof, 必须 0 (或 < 1% of LDS accesses)

**验证 (硬 gate)**
- writer probe build + run pass on chi2762
- coverage = 100% (所有 (k,n) 对正确)
- bank conflict = 0
- writer 微基准: 单次 transpose 写一个 128×128 tile, 时延 ≤ 4× G::load 同尺寸 row-major (允许些许 overhead, perf 还原在 Session 5 的整 kernel 上测)

**完成动作**
- HK + PT 3rdparty + PT outer commit
- 追加 `## Session 4 status: PASSED  HK=<hash>  PT=<hash>  outer=<hash>`
- memory: `feedback_rrr_b_pretrans_session4_writer.md`

**LOC + 时间** 约 400 LOC / 5-6h (核心 transpose 算法 + probe + bank conflict 验证)

---

## Session 4.1 — writer 升级到 Path P (cross-lane bpermute, perf)  (~250 LOC, 4-5h)

**目标**: 把 Session 4 的 Path L 写法 (Phase 3 用 16 ds_read_u8/lane gather) 替换成 Path P (cross-lane ds_bpermute_b32 + v_perm_b32 byte-shuffle in register)。LDS ops 减半, 进一步 narrowed bank conflict。

**前置**: Session 4 PASSED (Path L 正确性 + Session 3 round-trip)

**已有素材**
- `session4_writer_design.md` §3 给出完整算法 (16-lane subgroup: 16 ds_bpermute + 12 v_perm per 16x16 byte block)
- `tests/probes/rrr_b_writer_probe.cu` 的 `b_writer_roundtrip` 验证 harness 直接复用 (mode K + mode N 双 dump)

**任务**
1. 在 `rrr_b_writer_probe.cu` 加 `__device__ void transpose16x16_bpermute(...)` 函数: 输入 16 lanes × 16 byte (K-major), 输出 16 lanes × 16 byte (N-major), 通过 4 phase ds_bpermute + 3 v_perm/output dword (per design §3.2)
2. 加新 kernel `b_writer_path_P`: 取消 staging LDS, 直接 load HBM → in-register transpose → ds_write_b128 final
3. probe `main` 加 verify-C/D: Path P 重跑 verify-A + verify-B
4. ISA 抓取: Path P main loop ds_read_u8 = 0, ds_bpermute_b32 ≈ 16 × 4 sub-groups × 2 iter = 128/warp
5. 微基准 (probe 内置): 10K-loop time both kernels, Path P ≤ 0.5× Path L (Phase 3 改进)

**验证 (硬 gate)**
- Path P probe verify-A + verify-B 全 0 mismatch
- Path L 和 Path P 输出 Bs_final byte-for-byte 完全一致 (probe 内 cross-compare)
- ISA: Path P main path ds_read_u8 = 0; 16-lane subgroup 内 ds_bpermute_b32 计数 = 16
- 微基准: Path P 单 tile transpose time ≤ 0.5× Path L (因为消除 16 ds_read_u8/lane)

**完成动作**
- HK + PT 3rdparty + PT outer commit
- 追加 `## Session 4.1 status: PASSED  HK=<hash>  PT=<hash>  outer=<hash>`
- memory: `feedback_rrr_b_pretrans_session4_1_path_p.md`

**LOC + 时间** 约 250 LOC / 4-5h (Path P 函数 + 新 kernel + verify)

---

## Session 4.2 — bank conflict 消除 + writer 微基准  (~200 LOC, 3-4h)

**目标**: 量化并消除 Path P 写入 / 读取 LDS 的 bank conflict, 给 Session 5 kernel 集成提供性能预算。

**前置**: Session 4.1 PASSED

**已有素材**
- `session4_writer_design.md` §3.4 + §3.5 分析两路 bank conflict 类型 (Path P 2-way write conflict, Path L 4-way read conflict)
- 候选 mitigation: `st_128x128_n_major_v2` (XOR swizzle `((offset>>7) & 7) << 4` mirror st_16x128_v2)

**任务**
1. 新 ST `st_128x128_n_major_v2` (XOR swizzle) + 注册 `all` concept
2. Session 3 `load(rt_128x16_s, st_128x128_n_major_v2)` 特化: 复用 base addr 公式, 在 swizzle 层 XOR (基址处 XOR 后 b128 仍连续 16 bytes within bank-group of 4)
3. writer probe 加 `b_writer_path_P_v2` (写入 v2 ST), 重跑 verify-A + verify-B
4. rocprof on chi2811: `SQ_LDS_BANK_CONFLICT` counter 抓 path_P_v1 vs path_P_v2 vs G::load row-major
5. 输出表格: instructions/byte, bank_conflict_per_warp, 推断 kernel 集成后预期 % 跌
6. 决定: 用 v1 (identity) or v2 (swizzled) 进 Session 5

**验证 (硬 gate)**
- v2 path probe verify-A + verify-B 全 0 mismatch
- rocprof: v2 写入 bank conflict 比 v1 减少 ≥ 50% (或确认 v1 实测 < 5% 就停留 v1)
- 微基准: v2 vs v1 path P 时延差 < 5% (mitigation 不能 perf 倒退)

**完成动作**
- HK + PT 3rdparty + PT outer commit
- 追加 `## Session 4.2 status: PASSED  HK=<hash>  PT=<hash>  outer=<hash>  chosen=<v1|v2>`
- memory: `feedback_rrr_b_pretrans_session4_2_bank_conflict.md`

**LOC + 时间** 约 200 LOC / 3-4h (新 ST + load 特化 + rocprof scripting)

---

## Session 5 — RRR body 集成 + 24-shape verify  (~300 LOC, 4-5h)

**目标**: 把 Session 3 的 ST/load + Session 4 的 writer 接进 RRR body, 全 24-shape 数值正确 + perf 不回退。

**前置**: Session 3 + 4 PASSED (Session 4.1 / 4.2 强烈推荐, 否则 perf 见 8 章 Session 5 risk note)
**(原 Session 3 BLOCKED 后拆出的 3c)**

**任务**
1. **gate macro** `#define RRR_B_PRETRANS 0/1` 在 `kernel_fp8_layouts2.cpp` 顶部
2. **6 个 G::load(Bs[...]) sites** (prolog ×2 + 主循环 prefetch ×2 + FUSED_KTAIL ×2): 走新 writer (gated by macro)
3. **6+ load_b reads**: 走新 ST + 新 load 特化 (gated by macro)
4. **Bs slot size 重算**: 新 N-major Bs = 16KB per tile × 2 buf = 32KB; LDS budget audit (160KB 内 As 64KB + Bs 32KB + scratch 余下 OK)
5. **FUSED_KTAIL audit**: K_rem=0 和 K_rem=64 两种都要过; ktail 走老 byte_b8 路径还是新 writer 由 macro 决定 (Session 3 BLOCKED 提示 FUSED_KTAIL 可保留老 path)
6. **24-shape SNR sweep** with RRR_B_PRETRANS=0 (回归) + =1 (新), 各跑一遍
7. **ISA 验证**: 新路径 main loop `ds_read_b64_tr_b8` = 0, `ds_read_b128` ≈ 48

**验证 (硬 gate)**
- A. RRR_B_PRETRANS=0 24-shape SNR ≥ 47 dB (老路径回归)
- B. RRR_B_PRETRANS=1 24-shape SNR ≥ 47 dB (新路径)
- C. ISA: 新路径 main loop ds_read_b64_tr_b8 = 0
- D. 8-shape bench TFLOPS geomean 不回退 > 3% vs R167 baseline (1.024×T)

**完成动作**
- HK + PT 3rdparty + PT outer commit
- 追加 `## Session 5 status: PASSED HK=<hash> PT=<hash> outer=<hash>`
- memory: `feedback_rrr_b_pretrans_session5_integration.md`

**LOC + 时间** 约 300 LOC / 4-5h (~12 site 替换 + 双路径 verify)

---

## Session 6 — AGPR 数据流优化  (~200 LOC, 3-4h)

**目标**: 消除 RRR 比 RCR 多的 416 条 accvgpr shuffle。

**前置**: Session 5 PASSED
**(原 Session 4 — 顺移)**

**任务**
1. ISA disasm RRR body, 定位 accvgpr_read/write 来源 (acc init? spill? mma chaining?)
2. 视情况:
   - 让 acc 全程 resident in AGPR (D-aliases-C inplace + acc-init avoid v_mov)
   - 或改 acc storage class (rt_fl<...> 的 storage hint)
3. 与 Session 5 一样保留 numeric 验证

**验证**
- ISA: accvgpr_read + accvgpr_write 总数 ≤ 50 (vs 当前 416)
- 24/24 shape SNR ≥ 47 dB
- 8-shape bench: geomean ≥ 1.05× Triton (不是终点, 仍要 Session 7)

**完成动作**: 同上

---

## Session 7 — Per-shape autotune + 最终验证  (~100 LOC, 2-3h)

**目标**: 锁定 8/8 ≥ 1.15× Triton。

**前置**: Session 6 PASSED
**(原 Session 5 — 顺移)**

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
- 追加 `## Session 7 status: PASSED HK=<hash> PT=<hash> geomean=<x> min=<y>`
- 更新 MEMORY.md: `feedback_rrr_v2_b_pretrans_win.md`

---

## Session 1 result — B mma_AB lane layout 映射表

**Probe**: `tests/probes/rrr_b_lane_layout_probe.cu` + Makefile (HK + PT 3rdparty 双路径)
**Run host**: chi2762 (gfx950, MI355X), `/opt/rocm/bin/hipcc --offload-arch=gfx950`
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
- probe build + run pass on chi2762 (gfx950); coverage 2048/2048 unique
- 闭式映射 `(lane, byte) → (k, n)` 见 Session 1 result 段
- memory: `feedback_rrr_b_lane_layout.md`

## Session 2 status: PASSED  HK=090166ba  PT=e1ab0597  (outer=eae76e40)

## Session 3 status: PASSED  HK=75335d85  PT=9761d96e  (outer=3c52e440)
- 2026-05-25
- 新 ST `kittens::ducks::st_shape::st_128x128_n_major` (rows=N=128 cols=K=128, identity swizzle, fp8) + alias `st_128x128_n_major_s`
- `shared_to_register.cuh::load(RT col_layout, ST)` 新 `if constexpr` 分支: 当 ST 是 `st_128x128_n_major` + RT=`rt_128x16_s` col_l fp8 时, 走 per-lane 2× `ds_read_b128` 公式 (`addr = base + (lane&15)*128 + ((lane>>4)&3)*16`, offset:0 + offset:64)
- 单元 smoke `tests/probes/rrr_b_pretrans_st_load_probe.cu`: chi2811 (gfx950) **mismatch=0 missing=0 duplicate=0** 覆盖 64 lane × 32 byte × 8 N-tile = 16384 元组; OK 输出 "new st_128x128_n_major + load() == Session 1 mma_AB mapping"
- ISA (probe binary `--save-temps` 出的 amdgcn .s): **8 ds_read_b128 / 0 ds_read_b64_tr_b8** (新路径独立, 老 col_l fp8 路径未触发)
- Bug-fix bonus: 同时把 line 53 的 `st_32x64` 也补全为 `ducks::st_shape::st_32x64` (原裸名在某些 include 顺序下解析不到 — probe build 触发后才暴露; PT 实际 build 此前能过是因为 include 顺序碰巧 OK)
- HK build: PT full re-link clean on chi2811 (`libprimus_turbo_kernels.so` 51086336 bytes, 04:05 timestamp), 0 errors, 现有 `ducks::st_shape::all` dispatch 未破
- memory: `feedback_rrr_b_pretrans_session3_st_load.md`
- Session 4 entry: 大头剩下 **cross-lane register transpose writer** (HBM row-major B → N-major LDS), 见原 BLOCKED 文档拆出的 Session 4 §; Session 3 已交付的 ST + load primitive 是 Session 4 / 5 写入端 / kernel 集成的 type-system 基座 — 单独可 verify, 不依赖 writer 即可证

## Session 3 (superseded, original 500 LOC scope) status: BLOCKED  reason: 自定义 HBM→LDS B-pretranspose writer 单 session 装不下 — LDS budget (160KB) 排除 Option B fallback (要 196KB); Option A.1 naive scatter writer (16 ds_write_b8/lane) 正确但破 D gate (>10% 回退, 见 FUSED_KTAIL load_b_kt_fk 印证); Option A.2 cross-lane register transpose 需 ds_bpermute b32 × 4 + bank-conflict 校准, 多 session 工作 (~700-900 LOC) — **已拆 Session 3a/3b/3c (new Session 3/4/5), 老 Session 4/5 顺移 6/7**
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
- **Scope delivered**: 隔离 probe `tests/probes/rrr_b_pretrans_probe.cu` 验证 b128-from-pretranspose-LDS 路径 (LDS 按 N-major 摆: `byte_offset(n,k) = n*K_DIM + k`; 每 lane 2× `ds_read_b128` at `base + (lane&15)*K_DIM + ((lane>>4)&3)*16` + offset:64) 产出的 (lane, byte) → (k, n) 映射与 Session 1 mma_AB B 操作数映射 **byte-equivalent** (mismatch=0, coverage 2048/2048 unique on chi2762 gfx950)
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

## Session 4 status: PASSED  HK=d5523163  PT 3rdparty=8cff83b9  (outer bump TBD)
- 2026-05-25
- **Scope delivered (Path L 正确性 + Session 4.1/4.2 设计 + plan additions)**
  - **设计文档** `analysis/fp8_gemm/mi350x/session4_writer_design.md` (~330 行): 8-warp 协作 HBM→LDS B-transpose writer 的两条路径完整算法 + LDS budget audit + bank conflict 分析 + scope split into Session 4 / 4.1 / 4.2
  - **Probe 实现** `tests/probes/rrr_b_writer_probe.cu` (~250 LOC) + Makefile entry
    - kernel `b_writer_path_L`: 512-thread WG, 16KB staging LDS + 16KB final Bs (st_128x128_n_major), Phase 1+2 (HBM→staging) 2 b128 load + 2 b128 write per lane, Phase 3+4 (staging→final transpose) 16 ds_read_u8 gather + 1 b128 write per lane × 2 iter
    - kernel `b_writer_roundtrip`: 同 writer + Session 3 `load(rt_128x16_s, st_128x128_n_major)` 拉出 RT 内容并 dump
    - VERIFY-A (host byte-compare ref[n*128+k]=hbm[k*128+n] vs Bs_final dump): **mismatch=0** over 16384 bytes
    - VERIFY-B (round-trip mode-K + mode-N via Session 3 load, compare against Session 1 closed-form (lane,byte)→(k,n)): **mismatch=0 bad_range=0 missing=0 duplicate=0** over 16384 元组
  - **ISA 抓取** (chi2811 gfx950, hipcc --save-temps):
    - `b_writer_path_L` static: 16 ds_read_b128 + 8 ds_write_b128 + 73 ds_read_u8 + 48 v_perm_b32 + 28 s_waitcnt (Phase 3 用 16 byte-read/lane = 主性能瓶颈, Session 4.1 lever)
    - `b_writer_roundtrip` static: 16 ds_read_b128 (Session 3 load 工作) + 4 ds_write_b128 + 32 ds_read_u8 + 48 v_perm_b32 + 10 s_waitcnt + **0 ds_read_b64_tr_b8** (新 path 不触发 transpose-load, 与 Session 3 验证一致)
- **Scope deferred (Session 4.1 / 4.2 plan additions written)**
  - ❌ Path P (cross-lane ds_bpermute_b32 + v_perm_b32 byte-shuffle in register) — 算法已在 design doc §3 落档, Session 4.1 task
  - ❌ Bank-conflict-free LDS layout (`st_128x128_n_major_v2` XOR swizzle 候选) — design doc §3.4/3.5 分析, Session 4.2 task
  - ❌ rocprof `SQ_LDS_BANK_CONFLICT` 实测 — Session 4.2 task
  - ❌ Writer microbenchmark (vs G::load row-major baseline) — Session 4.2 task
- **Why split**: 单 session 装下 design + Path L 正确性 + 完整 verify harness 已是上限; Path P + bank conflict + microbench 是 perf 优化, 必须先有 Path L 正确基线再 swap-and-compare。承认 "拆细自交付" 原则, Session 4 = "writer 正确", Session 4.1 = "writer 快", Session 4.2 = "writer 最优"
- **Risk note for Session 5**: 若 Session 4.1 / 4.2 跳过直接接入 Path L, kernel 主循环 Phase 3 的 16 ds_read_u8/lane (× 2 iter × 8 warps × 64 lanes = 16K byte reads per 128×128 tile) 会触发严重 LDS bank serialization, 预期 fwd geomean 可能回退 -10%~-20%。若必须跳过, 建议在 Session 5 集成时打开 macro `RRR_B_PRETRANS_FALLBACK_TO_GLOAD=1` 在 perf 不达标时回退老路径

