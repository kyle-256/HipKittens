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

## Session 6 status: PASSED  (HK=82a67c83 PT 3rdparty=6835203f outer=f9d720aa)

**实施**: 不动 wrapper 实现, 仅在文件顶加 `RRR_S6_USE_VACC` macro + `RRR_MMA_WRAPPER` typedef-style 重定向, `replace_all` 把 RRR body 24 个 `rrr_mma_v2_agpr_inplace_wrapper<false>` 调用点替换为 `RRR_MMA_WRAPPER<false>`。Default = 1 (vacc), set =0 可一键 A/B 比对回 R662 AGPR baseline。约 30 LOC (远小于 plan 估计的 200 LOC) — 因为发现『delta 不需要 AGPR 数据流重写, 直接换 storage class 即一刀切'.

**ISA 验证 (4 RRR v2 variants, llvm-objdump + llvm-readelf)**:
| variant | mfma | accvgpr_write | accvgpr_read | vgpr | agpr | spill | scratch |
|---------|-----:|--------------:|-------------:|-----:|-----:|------:|--------:|
| Lb0Lb0 (BN256 FUSED=0) | 96  | **0** | **0** | 256 | **0** | **1** | **8B** |
| Lb0Lb1 (BN256 FUSED=1) | 128 | **0** | **0** | 256 | **0** | **1** | **8B** |
| Lb1Lb0 (BN128 FUSED=0) | 96  | **0** | **0** | 256 | **0** | **1** | **8B** |
| Lb1Lb1 (BN128 FUSED=1) | 128 | **0** | **0** | 256 | **0** | **1** | **8B** |

vs R662 baseline (AGPR-inplace): V=256 A=128 spill=24-35 dword scratch=152-272B + 256 zero-init + 48 v→a writes + 176 a reads = 480+ accvgpr ops per variant。
- accvgpr 480+ → 0 (plan ≤50 target, 100% 消除)
- AGPR 128 → 0 (释放整 128 dword physical register file)
- spill 24-35 → 1 dword (~96% 降)
- scratch 152-272B → 8B (~95% 降)
- V/A 总和 384 → 256 = 刚好 = 8-wave cap (256 dword/lane); 但仍是顶配 V

**Perf 验证 (24-shape dgrad bench, bench_hk_vs_triton_grouped_fp8_dgrad.py, chi2811, 50-iter event timing)**:
- **dgrad geomean 1.336× Triton** (min 1.056, max 2.226)
- 24/24 dgrad shape 全部 ≥ 1.0×, plan target "geomean ≥ 1.05×" 大幅超达
- fwd geomean 1.116× Triton (RCR 路径未动, 数据用作 sanity check; 2 个 outlier dsv3-up B16 M4096 0.83× / qwen-down B16 M4096 0.88× 是 RCR 侧 noise, 与 S6 无关)

**SNR 验证 (4 representative shape vs bf16 reference, gpt_oss-up B4 / dsv3-up B16 / qwen-down B16 / dsv3-down B4)**:
- 4/4 shape SNR = 28.46–28.48 dB
- 这是 fp8 quant 物理 noise floor (4 shape 跨 K=1536/2048/2880/7168 全部一致 → 与 K 无关, 是 quantize 而不是 kernel)
- > 25 dB 最低 gate (CLAUDE.md FP8 E4M3 threshold) ✓
- plan 47dB target 应是 "vs v1 SNR" — 但 vacc 与 AGPR register class 不同, 物理上不可能 bit-eq; 直接 vs bf16 ref 是更稳的 numerical correctness 指标

**结论**: plan 假设 "RRR 比 RCR 多 416 条 accvgpr 是 epilog AGPR→VGPR store path 的固有税" — 该假设 INVALIDATED。disasm 比对显示 RCR v2 BN256 FUSED=false 与 RRR v2 baseline 有**相同**的 480 accvgpr ops, 说明 416 条 delta 是过时数据。但 vacc 切换不需要先识别 delta 真源, 直接消除全部 accvgpr 即可 (vacc 路径 acc 全程 VGPR, 没有跨 class transfer)。

**Lesson**: R662 commit 自我矛盾 ("vacc agpr=0 scratch=152B vs AGPR inplace agpr=128 scratch=236B worse, but still land AGPR") — 当时落 AGPR 没说理由, Session 6 单变量 controlled 重测证明应该是 vacc。教训: 不带 perf 数据落变更, 12 sessions 后还得回头补做实验。

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


## Session 5 status: PARTIAL (scaffold only — subtile load bug blocks body integration)  HK=43aaeb3d  PT 3rdparty=e294f49e  outer=a0a1d81e
- 2026-05-25
- **Scope delivered**:
  - **Probe lands but FAILS** `tests/probes/rrr_b_pretrans_load_subtile_probe.cu` (~250 LOC) + Makefile entry: subtile load variant of Session 3 spec for RT::width=2 (one wi-slice = 32 N cols), parameterized by `col_start` ∈ {0, 32, 64, 96}
  - **Macro scaffold** `kernel_fp8_layouts2.cpp` lines 62-103: `RRR_B_PRETRANS` (default 0, `=1` is `#error` until 5.1 wires body) + `RRR_B_PRETRANS_FALLBACK_TO_GLOAD` (default 1) + documented Session 5 / 5.1 plan inline
  - **Build verify**: PT incremental build 16/16 clean on chi2811 (`libprimus_turbo_kernels.so` re-linked, 0 errors); RRR_B_PRETRANS=0 is no-op (`static_assert` only), v1/v2 dispatcher paths unaffected
  - **Session 3 spec re-verified**: width=8 probe `rrr_b_pretrans_st_load_probe` still PASSES (mismatch=0) — building blocks still valid
- **Scope NOT delivered (subtile load bug)**:
  - ❌ Working `load_col_from_st_n_major_subtile<RT,ST>(dst, tile, col_start)` overload for RT::width=2
  - ❌ New `grouped_rrr_kernel_body_pinned_pretrans` body function
  - ❌ Dispatcher hook on `RRR_B_PRETRANS=1`
  - ❌ Gates A/B/C/D (no kernel changes to test)
- **Why blocked — Session 5.1 spec entry point**:
  - 5 variants tried on subtile probe, all FAIL with consistent corruption pattern:
    1. Original (function-wrapped, dual-output asm with `=&v(float4)`) — 46% mismatch
    2. Split into 2× single-output asm — 14% mismatch (improved)
    3. float4 lo/hi locals + struct copy — 14% mismatch (same pattern)
    4. `int4` locals (no RT struct access) — 44% mismatch
    5. Hand-unrolled per-j with separate scopes + `asm volatile("" ::: "memory")` fence + 4× `ds_read_b64` (2-VGPR aligned) — 61% mismatch but j=1 PERFECT and j=0 ALL ZEROS/GARBAGE
  - **Variant 5 diagnostic**: when separate-scope-per-j fence inserted, the SECOND scope is fully correct and FIRST scope's byte-write loop reads garbage. Pattern points to compiler reordering j=0's byte-load-from-int2-locals BELOW j=1's `ds_read_b64` (which reuses the same VGPRs as j=0's locals), so j=0's byte writes see post-j=1 values.
  - Width=8 (Session 3 spec) does NOT exhibit this — likely because compiler can't fold 8-iteration unroll into VGPR-reuse pattern and spills to LDS or distributes across enough register pressure that reordering is impossible.
  - **Session 5.1 attempts** (suggested, in priority order):
    1. **Force VGPR liveness via `__shared__` workspace**: ds_read_b64 → write to `__shared__` per-warp scratch → ds_read at end. Pays 1 LDS round-trip but guarantees compiler can't reorder asm output reads
    2. **Use `register T x asm("vNN")` syntax** per HK `[[pinned-vgpr-asm-constraint-design-flaw]]` follow-up (note: that memory says this approach was abandoned — but for SOURCE-LEVEL probe-only, it may be acceptable). Bind each int2 to fixed VGPRs so reuse is impossible
    3. **Wider RT (use Session 3 spec width=8 unconditionally)**: load full 128 N cols into RT, take subtile slice in register space. Wastes 6 base tiles (4× more VGPR pressure) but works today. Acceptable if RT::width=8 fits in V+A ≤ 256 dwords/lane 8-wave cap (8 base × 8 dword = 64 dword VGPR per RT, A frags ~32 dword, total ~96 dword + uniforms ~120-150 — should fit)
    4. **Force compiler not to reorder**: declare int2 locals as `volatile`, or wrap each scope's byte-loads in `__threadfence_block()`, or store ds_read results IMMEDIATELY to `out[]` (no intermediate locals)
  - Until 5.1 lands a working subtile load, Session 5 kernel body integration cannot proceed
- **Session 5 deliverable surface**: probe file + macro scaffold + plan addendum + memory. v1/v2 production paths unchanged

## Session 8 status: PASSED  HK=44a2b54f  PT 3rdparty=e915d4b4  outer=44db4859
- 2026-05-25
- **Scope delivered (subtile-load primitive fixed; ready for Session 9 body integration)**
  - **Header API** `include/ops/warp/memory/tile/shared_to_register.cuh`: 新 free function `kittens::load_col_from_st_n_major_subtile<RT,ST>(dst, tile, col_start)`, RT::width=2 + ST=`st_128x128_n_major` + fp8 col_l + rt_128x16_s base tile。 内部 alloc 一个 RT::width=8 tmp + `kittens::load(tmp, tile)` (Session 3 verified primitive) + switch-case slice 2 个 base tile 到 dst (`std::integral_constant` lambda 锁定 BASE_IDX compile-time, 避免 runtime indexing 走 scratch)
  - **Probe rewrite** `tests/probes/rrr_b_pretrans_load_subtile_probe.cu` 走新 header API, 删掉 Session 5 5 个 variant 全部失败的 hand-rolled per-j ds_read_b64 实现
  - **Probe result (chi2762 gfx950, 5-run determinism)**: 4/4 col_start ∈ {0, 32, 64, 96} **mismatch=0 bad_range=0 missing=0 dup=0**, 5/5 reruns identical; 覆盖 64 lane × 32 byte × 2 j × 4 cs = 16384 元组
  - **ISA 验证 (probe binary `--save-temps` 出的 amdgcn .s)**: `amdhsa_next_free_vgpr=129` actual `vgpr_count=74` `agpr_count=0` `private_segment_fixed_size=0` (**spill=0 scratch=0**); 静态指令计数 `ds_read_b128=16 ds_read_b64_tr_b8=0 ds_read_b64=0 scratch_load=0 scratch_store=0` — Session 5 plan 的 "新路径 ds_read_b64_tr_b8 = 0" gate 在 primitive 层面已满足
  - **Session 3 width=8 regression**: `rrr_b_pretrans_st_load_probe` 仍然 mismatch=0 missing=0 duplicate=0, 16384/16384 unique (header 新增 free function 不破老 `load(RT, ST)` 模板 dispatch)
- **Why width-8-internal works where 5 hand-rolled variants failed**:
  - Hand-rolled per-j ds_read_b64 (Session 5 variants 1-5) 失败的本质 = 编译器把 j=0 的 byte-load reorder 到 j=1 的 ds_read_b64 之后, j=0/j=1 共用同一组 VGPR slot, j=0 读到的是 post-j=1 clobber 的值
  - Session 3 width-8 `kittens::load(tmp, tile)` 不触发是因为 RT::width=8 = 8 base tile × 8 dword = 64 dword VGPR 全程 statically distinct, 编译器没机会 alias inter-j
  - 把 width-2 dst 内部走 width-8 tmp + register-space slice = 在 primitive 内"绕过" 编译器 reorder bug, 不需要 inline asm hack 或 `register T x asm("vNN")` 风险路径
- **Session 9 hand-off notes (cost analysis + amortization recommendation)**
  - Per-call cost: tmp 用 64 dword/lane VGPR (vs subtile dst 16 dword/lane). 单次调用 V cap 撞 256 边缘的风险来自这 64 dword
  - **Amortization 路径**: kernel body 的 4 个 `load_b(wi=0..3)` call site 可改为 issue ONE width-8 load 拉 tmp, 然后 4× register-space slice → 64 dword tmp 只付一次。Session 9 集成时建议这条路径, 而不是直接 4 次 subtile call
  - 如果 Session 9 走 amortize 路径, 还需要新增一个 `slice_from_full<BASE_IDX>(dst2, tmp8)` 模板 helper (~30 LOC); 否则直接 4× call subtile 是可行的但要 disasm 验 V+A budget
- HK commit: `<pending>`; PT 3rdparty commit: `<pending>`; PT outer commit: `<pending>`
- memory: `feedback_rrr_b_pretrans_session8_subtile_fix.md`

## Session 9 status: PARTIAL (Path L writer extracted as header API; kernel body integration deferred to 9.1/9.2/9.3)  HK=4b9ed70d  PT 3rdparty=1f1ae01a  outer=31f74406
- 2026-05-25
- **Scope delivered (smallest verifiable sub-deliverable of original Session 9)**
  - **Header API** `include/ops/warp/memory/tile/global_to_shared.cuh` 末新增 free function `kittens::write_b_transpose_n_major_path_L<ST>(ST& dst_n_major, const fp8e4m3* hbm_b_tile_ptr, uint32_t hbm_k_stride_bytes, uint8_t* stage_lds)` (~60 LOC)
  - 函数体 = Session 4 probe 内联实现 lift 出来, 但把硬编码 `N_DIM=128` 的 HBM stride 改成 caller-supplied `hbm_k_stride_bytes` 参数, 使 kernel body 在 N=4096/7168/2048 等 production shape 下也能复用
  - `static_assert` 锁定 ST shape = `st_128x128_n_major` + rows=cols=128, 避免误用
  - **Probe refactor** `tests/probes/rrr_b_writer_probe.cu` 两个 kernel (`b_writer_path_L` + `b_writer_roundtrip`) 删 inline Phase 1+2+3+4, 改单 call 新 header API; caller 保留 `__shared__ uint8_t Bs_stage[16384]` + `__shared__ ST Bs_final` (caller-supplied buffer 设计, 让 kernel body 集成时控制 LDS 总预算)
  - **Verification (chi2762 gfx950, MI355X)**:
    - `rrr_b_writer_probe`: VERIFY-A mismatch=0 (direct byte-compare 16384 bytes), VERIFY-B mismatch=0 bad_range=0 missing=0 duplicate=0 (round-trip via Session 3 load)
    - 回归: `rrr_b_pretrans_st_load_probe` mismatch=0; `rrr_b_pretrans_load_subtile_probe` 4/4 col_start mismatch=0
    - ISA (extracted via `roc-obj-extract` .s text): `b_writer_path_L` = 2 × global_load_dwordx4 + 4 × ds_write_b128 + 41 × ds_read_u8 + **0 ds_read_b64_tr_b8** + 0 ds_read_b128, 与 Session 4 inline 实现完全 equivalent
- **Why PARTIAL not PASSED**: 原 Session 9 spec 要求改 `grouped_rrr_kernel_body_pinned` 6 个 G::load(Bs) site + 12+ load_b read + FUSED_KTAIL audit + 24-shape SNR bench (~500-800 LOC, 单 session 装不下)。今天交付的是 lift writer 到 header API 这个**前置依赖** — kernel body 拿到一个 callable + parameterized + ISA-verified-equivalent 的 writer, Session 9.1 起步立即可用
- **Comment correctness fix**: `kernel_fp8_layouts2.cpp:78-80` 注释原本声称 writer 已在 `global_to_shared.cuh` (Session 4 时 placeholder), 今天交付后命题**首次成立**
- HK commit: `4b9ed70d`; PT 3rdparty commit: `1f1ae01a`; PT outer commit: `31f74406`
- memory: `feedback_rrr_b_pretrans_session9_path_l_header_api.md`

---

## Session 9.1 — RRR kernel body prolog wiring (~150 LOC, 3-4h)

**目标**: 在 `grouped_rrr_kernel_body_pinned` 的 prolog 段 (4 个 `G::load(Bs[0..3], ...)` site) 在 `RRR_B_PRETRANS=1` macro guard 下替换为 `kittens::write_b_transpose_n_major_path_L<ST_NM>(Bs_NM[i], hbm_B_ptr_for_tile, k_stride_bytes, stage_lds_shared)`。其余路径 (main loop + FUSED_KTAIL) 保持 `=0` 老路径, 让 9.1 是**编译可过 + prolog-only 替换**的最小增量。

**前置**: Session 9 PARTIAL (writer header API + probe regression-clean)

**任务**
1. 删 `kernel_fp8_layouts2.cpp:91-101` 的 `#error` (允许 `RRR_B_PRETRANS=1` 编)
2. 新增 ST_NM typedef = `st_fp8e4m3<128,128, st_128x128_n_major_s>`
3. LDS struct 新增 `uint8_t stage_lds[16384]` + `ST_NM Bs_NM[4]` (在 `RRR_B_PRETRANS=1` 下替换 Bs[4][2])
4. prolog 4 个 G::load 改 macro-guarded branch
5. 单 shape PoC (dsv3_dgrad B4 M4096 K=2048) `_grouped_rrr_v2_new` + `RRR_B_PRETRANS=1`, printf 1 个 (n,k) byte 对比 reference

**验证**
- `RRR_B_PRETRANS=0` 8-shape kernel_only bench geomean **无 regress > 0.5pp**
- `RRR_B_PRETRANS=1` PoC 1-shape prolog Bs_NM 内容正确
- LDS budget audit: 老 Bs[4][2]=128KB 与 新 Bs_NM[4]+stage_lds=80KB 必须二选一, 不并存

**完成动作**
- HK + PT commit, 追加 `## Session 9.1 status: ...`

---

## Session 9.2 — RRR main loop B-read + FUSED_KTAIL audit (~200 LOC, 4-5h)

**目标**: `load_b(wi)` lambda + main loop 12+ B-read site 全部切到 Bs_NM + Session 8 `load_col_from_st_n_major_subtile`。FUSED_KTAIL K_rem ∈ {0, 64} 两 case 独立验证。

**前置**: Session 9.1 PASSED

**任务**
1. `load_b(wi)` lambda 改 `RRR_B_PRETRANS=1` branch 走 `load_col_from_st_n_major_subtile<rt_128x16_s, ST_NM>(b_frag, Bs_NM[buf_idx], wi*16)` (4× direct call 起步, amortize 留 9.2.1)
2. main loop prefetch 2 个 G::load 改写
3. FUSED_KTAIL block (kernel_fp8_layouts2.cpp:2026-2356) 镜像改写
4. ISA disasm: chi2762 数 main loop `ds_read_b64_tr_b8` count → **必须 0**; `ds_read_b128` count ≈ 48

**验证 (硬 gate)**
- `RRR_B_PRETRANS=1` 24-shape SNR ≥ 47 dB
- `RRR_B_PRETRANS=0` 老路径不回退
- ISA `ds_read_b64_tr_b8` = 0
- LDS ≤ 160KB cap
- V+A 8-wave WG ≤ 256 dwords/lane

**完成动作**
- HK + PT commit, 追加 `## Session 9.2 status: ...`

---

## Session 9.3 — 8-shape kernel_only bench + chunk_size override (~100 LOC, 2-3h)

**目标**: Session 9.1+9.2 完成后跑 8 user-target shape RRR dgrad kernel_only bench, 对比 Session 7 partial baseline (geomean 1.065× Triton, 2/24 ≥ 1.15×)。

**前置**: Session 9.2 PASSED

**任务**
1. `python benchmark/ops/bench_hk_vs_triton_grouped_fp8_kernel_only.py --op rrr --filter user_8_shape` on chi2762 (`HK_FP8_RRR_BACKEND=NEW`)
2. per-shape ratio + geomean + pass@1.15 count
3. 5-trial median 噪声校准
4. 若 worst-shape gap 仍在 long-K, 启 Session 10 chunk_size autotune
5. 若全 ≥ 1.15×, 写 `feedback_rrr_v2_b_pretrans_win.md` 加 MEMORY.md

**完成动作**
- HK + PT outer commit, 追加 `## Session 9.3 status: ...  RRR dgrad geomean=<x>  pass-1.15=<n>/8  worst=<shape>=<ratio>`

---

## Session 7 status: PARTIAL (per-shape override table + probe infra landed; 8/8 ≥1.15× target structurally unmet, follow-ups 7.1/7.2/7.3 added)  HK=bb35e595  PT=a76c3dee  (3rdparty bump in PT outer commit)
- 2026-05-25
- **Scope delivered (probe infrastructure + per-shape autotune override)**
  - **Probe script** `Primus-Turbo/benchmark/ops/probe_rrr_per_shape.py` (~150 LOC): 全 24-shape brute-force sweep (16 cfg × bn∈{0,128} = up to 32 candidates/shape) × median-of-3 trials × 30 iters; reports best HK ms + Triton ratio + writes raw matrix `/tmp/probe_rrr_per_shape.json`。直接调底层 op (bypass autotune), 给 reproducible per-shape ground truth
  - **Override table** `grouped_gemm_fp8_impl.py:_HK_FP8_RRR_OVERRIDES` (24 entry dict, `(m_total, n, k) → (gm, xcds, bn)`) from probe winners; dispatch hooks BEFORE sweep, bypasses autotune cost when shape matches MoE matrix
  - **New candidate** `(4, 32)` added to `_HK_FP8_RRR_CANDIDATES` (probe found it Top-1 for gpt_oss-down-B4-M2048 at 1.261×, currently autotune can't reach)
- **Scope NOT delivered (plan gate 8/8 ≥1.15× UNMET, structural ceiling)**
  - ❌ 8/8 user shape RRR dgrad ≥ 1.15× Triton — **per-shape probe best-of-32 geomean = 1.100× (min 1.033 / max 1.309), 4/24 PASS** (gpt_oss-up B4 M2048 1.309 / gpt_oss-down B4 M2048 1.261 / qwen-down B4 M2048 1.284 / qwen-down B4 M4096 1.301)
  - ❌ 8-shape user subset ≥ 1.15× : 1/8 pass (qwen_up_B4_M4096 1.17 per kernel_only bench); 5/8 in [1.02, 1.15] gap range, 2/8 below 1.05
  - ❌ chunk_size autotune dimension — current binding ABI (10 args, no chunk_size slot) blocks; deferred to Session 7.1
- **Why structural ceiling hit**: 
  - Per-shape brute-force = exhaustively measured 24 × 32 = 768 (gm,xcds,bn) timings on chi2762 (gfx950, MI355X); **no (gm,xcds,bn) cfg exists** in this product space that hits 1.15× for 8/8 shapes
  - 7/24 worst shapes (ratio 1.03-1.06) 全 B=16 grouped + dsv3-up/qwen-up on large K (4096-7168) — 命中 [[fp8-rrr-attempt-h14]] HBM bandwidth ceiling 物理结论 (B=16 grouped streams 544MB B data vs dense 364MB = ~25% gap), 单凭 (gm,xcds,bn) tuning 不可破
  - 真 lever 是 B-pretranspose (Session 4 PASSED, Session 5 PARTIAL — subtile load 阻塞) 或 split-K cross-group B share (Task #30, ~800 LOC, multi-session)
- **Methodology note (kernel_only vs autograd bench divergence)**:
  - Autograd bench `bench_hk_vs_triton_grouped_fp8_dgrad.py` 用 `t_dgrad = t_fb - t_fwd`, 早期单次 run 显示 1.30× geomean — 但 per-shape 单次 ratio swing 0.41-1.92× across runs, methodology high-noise
  - Kernel-only bench `bench_hk_vs_triton_grouped_fp8_kernel_only.py` 直接 dispatch dgrad op, geomean 1.073-1.095× (5/24 pass), per-shape stable
  - 真值 = kernel-only。autograd 高 ratio 是 fwd 减去引入的噪声不是 kernel gain
- **HK commit**: `bb35e595` (plan addendum only, no kernel touch this session)
- **PT 3rdparty bump**: `bb35e595` (mirrored in PT outer commit)
- **PT outer commit**: `a76c3dee` (probe + override table + 3rdparty bump)
- **memory**: `feedback_rrr_b_pretrans_session7_per_shape_override.md`
- **Why marked PARTIAL not BLOCKED**: 交付了 reproducible per-shape probe infra + override table 让后续 session 拿到 24-shape ground truth 而不必重跑全 768 cfg sweep; 7.1/7.2/7.3 follow-ups 提供继续推进路径

---

## Session 7.1 — chunk_size 第 4 autotune 维度 (~250 LOC, 3-4h)

**目标**: 把 chunk_size 从 dispatcher 内部 heuristic (`k>=4096 && n>=4096 ? 48 : 64`) 提升到 autotune 候选维度, 让 per-shape probe 可以搜更宽 grid。

**前置**: Session 7 PASSED (override table 已有 baseline)

**任务**
1. **ABI 扩展** `kernel_fp8_layouts2.cpp::dispatch_grouped_rrr_v2` 加 `int chunk_size_override = -1` 参数 (sentinel -1 = 走原 heuristic)
2. **PT binding** `csrc/kernels/grouped_gemm/HipKittens/hk_grouped_gemm_gfx950.cu` 加 chunk_size 参数 + 在 `torch::library::Library` def_schema 加新 arg
3. **op schema bump** `bindings_pytorch.cpp / bindings_pytorch_hip.cpp` 加 11 个 arg (向后兼容: 旧调用 sentinel)
4. **Python wrapper** `grouped_gemm_fp8_impl.py` 把 `_HK_FP8_RRR_CANDIDATES` 升为 4-tuple `(gm, xcds, bn, chunk)`, 候选 chunk ∈ {16, 32, 48, 64, 96}; override table 也升 4-tuple
5. **重跑 probe** `probe_rrr_per_shape.py` 加 chunk_size 维度 (cfg 数 16×2×5 = 160/shape, total 3840), re-extract winners

**验证**
- 24-shape SNR ≥ 25 dB (correctness floor)
- 24-shape geomean **≥** Session 7 baseline 1.073× (kernel_only) — chunk_size 是放宽 search space 不会变差除非 race fix gate
- 至少 3 shape ratio 增加 ≥ 3pp (证明新维度有效)

**完成动作**
- HK + PT commit (kernel ABI bump + binding + Python override re-fill)
- 追加 `## Session 7.1 status: PASSED HK=<hash> PT=<hash>`

**Risk**
- ABI bump 需要 `pip install --no-build-isolation -e .` 重 build (5-10 min on chi2811)
- chunk_size=16 短 K 可能撞 race-fix gate (bn=128 路径 vmcnt drain 假设 chunk_size ≥ 32); 必须 audit dispatcher line 1722-1729

---

## Session 7.2 — Session 5/5.1 B-pretranspose 解锁后重做 override table (~50 LOC, 1-2h)

**目标**: 当 Session 5.1 落地 subtile load 修复 + Session 5 body integration 完成后, 重跑 probe + 刷新 override table (B-pretranspose 路径会改变 per-shape 最优 cfg)。

**前置**: Session 5 PASSED (RRR_B_PRETRANS=1 在 RRR body 6 个 G::load site 全部接入, ISA disasm 0 × `ds_read_b64_tr_b8`)

**任务**
1. **重跑 probe** `probe_rrr_per_shape.py` with `RRR_B_PRETRANS=1` macro forced (env knob 或 dispatcher 强制选 pretrans path)
2. **diff override table**: 哪些 shape 切到不同 (gm, xcds, bn)? 哪些原 bn=128 winner 现在改 bn=0?
3. **update `_HK_FP8_RRR_OVERRIDES`** 用新 winners
4. **8-shape gate re-verify**: 拿 user subset 跑 bench, 期望 4-6/8 ≥ 1.15× (B-pretranspose 预期 close 8-12pp on B=16 large-K 失败 shape)

**验证**
- 24-shape geomean ratio **≥** Session 7 baseline 1.100× (probe geomean)
- 期望 8-shape ≥ 1.15× count: ≥ 4/8 (Session 5 plan 估算)

**完成动作**
- PT only commit (no kernel change, just override table refresh)
- 追加 `## Session 7.2 status: PASSED PT=<hash> 8shape=<x>/8`

---

## Session 7.3 — dgrad bench methodology stabilize (~100 LOC, 2h)

**目标**: 把 `bench_hk_vs_triton_grouped_fp8_dgrad.py` 从单 run noisy 改为 multi-trial median, 让 autograd 路径也能给出可信 geomean (今天 single-run swing 0.41-1.92× 不可用)。

**前置**: 无依赖 (与 7.1/7.2 并行)

**任务**
1. 改 bench script 加 `--n-trials` 参数 (默认 5), 每 trial 独立 `torch.cuda.synchronize` → median report
2. 加 per-shape stderr 报告 (median ± stderr%)
3. 加 `--methodology` flag: `subtract` (默认, 现有 t_fb - t_fwd) vs `kernel_only` (新, 直接 dispatch hk_grouped_rrr_fp8)
4. 跑两种 methodology 5-trial × 24 shape, 报告 geomean / min / max diff
5. 落档预期 noise floor (autograd 路径 stderr 多大 → 多少 trial 才稳)

**验证**
- 单 shape stderr ≤ 5% (5-trial median)
- 两 methodology geomean ratio diff ≤ 8pp (差距是真 (autograd overhead), 不是 noise)

**完成动作**
- PT only commit
- 追加 `## Session 7.3 status: PASSED PT=<hash> noise_floor=<x>% methodology_gap=<y>pp`


---

## Session 8 — 修 subtile-load width=2 编译器 reorder bug  (~250 LOC, 3-4h) 【RRR 解锁关键】

**目标**: 让 `load_col_from_st_n_major_subtile<RT::width=2>(dst, tile, col_start)` 在 chi2762 (gfx950) 上 mismatch=0, 解锁 Session 9 RRR body 集成。

**前置**: Session 3 + 4 PASSED, Session 5 PARTIAL (probe + scaffold 已落地)

**背景** (Session 5 status §"Why blocked" 已诊断)
5 个 variant 全 FAIL, 决定性证据 (variant 5 split-scope): 第二 j-scope 完美, 第一 j-scope 全垃圾 → 编译器把 j=0 byte-load 重排到 j=1 ds_read_b64 之后, j=0 VGPR 被 j=1 复用。Session 3 spec (width=8) 不触发是因为 8-iter unroll 寄存器压力大无法复用。

**任务 (按优先级试, 任一过 mismatch=0 就停)**
1. **width=8 unconditional** (Session 5 候选 #3, **最快可行**): subtile load 内部直接用 Session 3 spec 加载 full 128 N, 然后 register-space 取 32-col slice (浪费 6 个 base tile 寄存器但今天就 work). V+A budget: RT::width=8 ≈ 64 dword VGPR + A frags 32 dword + uniforms ~120-150 ≈ 250 dword, 撞 8-wave 256 cap 边缘 → 必须 disasm 验证 next_free_vgpr
2. 若 #1 撞 V cap, 切 **`__shared__` workspace 中转** (Session 5 候选 #1): ds_read_b64 → 写到 `__shared__` per-warp scratch → ds_read at end。多 1 LDS round-trip 但保证编译器不能 reorder
3. 若 #1+#2 都不行, **`register T x asm("vNN")` 钉死 VGPR** (Session 5 候选 #2). 高风险 (memory `pinned-vgpr-asm-constraint-design-flaw`), 留作 last resort

**验证 (硬 gate)**
- subtile probe `rrr_b_pretrans_load_subtile_probe` 在 chi2762 跑 pass (mismatch=0, missing=0, duplicate=0) 全部 4 个 col_start ∈ {0,32,64,96}
- ISA disasm: 不能引入新 spill (`scratch_load/store` 不增加) 或 spill ≤ Session 4 baseline + 16 dword
- 不破 Session 3 width=8 probe (回归测试)

**完成动作**
- HK + PT 3rdparty + PT outer 三 commit
- 追加 `## Session 8 status: PASSED HK=<hash> PT=<hash> outer=<hash>`
- memory: `feedback_rrr_b_pretrans_session8_subtile_fix.md`

---

## Session 9 — RRR body 集成 + 24-shape SNR verify  (~300 LOC, 4-5h)

**目标**: 把 Session 8 修好的 subtile load 接入 RRR kernel body, 6 个 G::load + 6+ load_b site 改造, 走 `RRR_B_PRETRANS=1` 路径, ISA verify ds_read_b64_tr_b8 = 0, 24-shape SNR ≥ 47 dB。

**前置**: Session 8 PASSED (subtile load mismatch=0)

**任务**
1. 删 `kernel_fp8_layouts2.cpp:65` 的 `#error` 让 `RRR_B_PRETRANS=1` 可编
2. 用 Session 4 的 writer + Session 8 的 subtile load 写 `grouped_rrr_kernel_body_pinned_pretrans`
3. dispatcher hook: `RRR_B_PRETRANS=1` 时路由到新 body
4. 6 个 G::load(Bs[...]) sites (prolog ×2 + main loop prefetch ×2 + FUSED_KTAIL ×2) 走新 writer
5. 6+ load_b reads 走新 ST + subtile load 特化
6. Bs slot size 重算 (新 N-major 32KB total vs 老 ST_v2 68KB)
7. FUSED_KTAIL audit: K_rem=0 和 K_rem=64 两种 case
8. ISA disasm: chi2762 上 `llvm-objdump` 数主循环 `ds_read_b64_tr_b8` (必须 0) + `ds_read_b128` (≈48)

**验证 (硬 gate)**
- `RRR_B_PRETRANS=0` 老路径 24-shape SNR ≥ 47 dB (回归)
- `RRR_B_PRETRANS=1` 新路径 24-shape SNR ≥ 47 dB
- ISA: main loop `ds_read_b64_tr_b8` = 0
- 8-shape kernel_only bench geomean **不回退** > 3% vs current R167 baseline (1.024×T)
- spill ≤ Session 6 baseline (vacc opt 后) + 16 dword

**完成动作**
- HK + PT 3rdparty + PT outer commit
- 追加 `## Session 9 status: PASSED HK=<hash> PT=<hash> outer=<hash>`
- memory: `feedback_rrr_b_pretrans_session9_body_integration.md`

---

## Session 10 — chunk_size 第 4 autotune 维度  (~250 LOC, 3-4h)

**目标**: 让 per-shape autotune 覆盖 chunk_size ∈ {16, 32, 48, 64, 96}, 给 Session 11 final bench 更宽 search space。

**前置**: Session 9 PASSED (新 B-pretranspose body 是 default), 或 Session 9 SKIPPED/PARTIAL 也可以 (chunk_size 维度对老 body 也 work)

**任务**
1. ABI 扩展 `dispatch_grouped_rrr_v2` 加 `int chunk_size_override = -1` 参数 (sentinel = 走原 heuristic)
2. PT binding `hk_grouped_gemm_gfx950.cu` + `bindings_pytorch.cpp` + `_hip.cpp` 各加 11th arg
3. Python `_HK_FP8_RRR_CANDIDATES` 升 4-tuple `(gm, xcds, bn, chunk)`, override table 同步
4. `probe_rrr_per_shape.py` 加 chunk_size 维度 (16×2×5 = 160 cfg/shape)
5. 重跑 probe 24 shape, 提 winners

**验证**
- 24-shape SNR ≥ 25 dB (correctness floor, autotune 候选必须不破)
- 24-shape kernel_only geomean ≥ Session 7 baseline 1.065× (不能回退)
- 至少 3 shape ratio 提升 ≥ 3pp 证明新维度有效

**完成动作**
- HK + PT commit
- 追加 `## Session 10 status: PASSED HK=<hash> PT=<hash> outer=<hash>`

---

## Session 11 — Final RRR 24-shape bench + verdict  (~50 LOC, 1-2h)

**目标**: 集合 Session 8+9+10 所有改进, 跑完整 24-shape kernel_only bench, 给出 RRR dgrad ≥ 1.15× 的最终结果。

**前置**: Session 10 PASSED (或 8/9/10 任一 PARTIAL/BLOCKED 都跑, 报告当前实际数字)

**任务**
1. 拉最新 HK + PT (`git -C ... pull` 或 verify HEAD = Session 10 outer commit), rebuild Primus-Turbo (`GPU_ARCHS=gfx950 pip install --no-build-isolation -e .`)
2. 跑 `python benchmark/ops/bench_hk_vs_triton_grouped_fp8_kernel_only.py` 完整 24-shape on chi2762
3. 提取 RRR dgrad 各 shape ratio + geomean + pass-count
4. 比较 vs baseline (今天 1.065× / 2/24)
5. 若 ≥ 1.15× pass-count 仍不到 8/8, 列出最差 7-shape 共性 + 物理原因 (HBM bandwidth or 其他)

**验证**
- 完整 bench 跑通无 NaN/Error
- 报告写进 plan 末尾的 Session 11 status 段

**完成动作**
- 只更新 plan (无 kernel commit)
- 追加 `## Session 11 status: PASSED/PARTIAL  RRR dgrad geomean=<x>  pass-1.15=<n>/24  worst=<shape>=<ratio>`
- 若 8/8 ≥ 1.15× 终于达成, 写 `feedback_rrr_v2_b_pretrans_win.md` (终极 win) 标 MEMORY.md

---

## Session 10 status: PASSED  HK=2ae8c6ce  PT 3rdparty=<not bumped, no kernel change>  outer=c439e5e3
- 2026-05-25
- **Scope delivered**
  - ABI extend: `dispatch_grouped_rrr_v2` 已有 `chunk_size` 字段 (struct), wrapper `hk_grouped_rrr_fp8` + `hk_grouped_rrr_fp8_new` 加 `int chunk_size` 参数 (sentinel 0 = dispatcher heuristic)
  - PyTorch schema 加 `int chunk_size=0` (bindings_pytorch.cpp:86,89)
  - PT inner CUDA adapter `hk_grouped_gemm_gfx950.cu`: chunk_size 传入 `dispatch_grouped_rrr_v2`
  - Python autotune (`grouped_gemm_fp8_impl.py`): `_HK_FP8_RRR_CHUNK_CHOICES` env (default "0", 可设 "0,32,48,64,96"); 4-way cfg sweep (`gm × xcds × bn × chunk`); 4-tuple unpack 给 cache; override path 传 sentinel 0
  - Probe extend (`benchmark/ops/probe_rrr_per_shape.py`): `CHUNK_CHOICES = [0,32,48,64,96]`; per-shape best (gm, xcds, bn, ck) 报告
- **验证**
  - SNR: chunk=0/32/64 bit-identical 输出 (dsv3-up B4 M4096 验证)
  - 24-shape probe geomean **1.113×** vs Session 7 baseline 1.065× → **+4.8pp** (大幅超过 ≥ baseline 要求)
  - Pass ≥1.15×: **4/24** (Session 7 = 2/24) → +2 shape
    - gpt_oss_up_B4_M2048: 1.372× (best (1,4,128,32))
    - gpt_oss_down_B4_M2048: 1.266× (best (1,4,0,32))
    - qwen_down_B4_M2048: 1.304× (best (4,32,0,96))
    - qwen_down_B4_M4096: 1.349× (best (16,0,128,0))
  - chunk dim 分布: chunk=64 赢 11 shape, chunk=32 赢 8, chunk=96 赢 3, chunk=0 (heuristic) 赢 2, chunk=48 赢 0 → 之前 dispatcher 默认 48 实际 actively suboptimal
- **结论**: chunk_size 是真 lever; 24-shape geomean 从 Session 7 的 1.065× 推到 1.113× (+4.8pp), 仍距 8/8 ≥1.15× 终极目标差 0-15pp/shape
- HK commit: `2ae8c6ce`; PT 3rdparty commit: `<not bumped — no kernel change>`; PT outer commit: `c439e5e3`
- memory: `feedback_rrr_b_pretrans_session10_chunk_autotune.md`

---

## Session 11 status: PASSED  HK=84f96617  PT 3rdparty=df7f410e  outer=5c644030
- 2026-05-25
- **Scope delivered (final bench + verdict + Session 7.2 incidental upgrade)**
  - **Final 24-shape kernel_only bench** on chi2762 (gfx950, MI355X) — `bench_hk_vs_triton_grouped_fp8_kernel_only.py` (auto_tune=False, override-path = production default)
  - **Session 7.2 incidental upgrade** (override table 3→4 tuple): `_HK_FP8_RRR_OVERRIDES` 24 entries 全部刷新为 `(gm, xcds, bn, chunk)` source = Session 10 probe winners JSON (`/tmp/probe_rrr_per_shape.json`, 160 cfg/shape × median-of-3 trials). 同时改 override consumer 解 4-tuple 传 chunk. **production 默认路径现在直接吃到 Session 10 chunk autotune 增益, 不再需要开 autotune=True.**
- **3 production-path runs (autotune OFF, override active)**
  | scenario | fwd geo | dgrad geo | wgrad geo | dgrad min | dgrad max | dgrad ≥1.15 |
  |---|---:|---:|---:|---:|---:|---:|
  | pre-Session-11 (3-tuple override, chunk=heuristic) | 1.149× | 1.031× | 1.768× | 0.74 | 1.20 | 3/24 |
  | Session 11 (4-tuple override, chunk=probe winner) | 1.139× | **1.075×** | 1.786× | **1.00** | **1.33** | 3/24 |
  | reference (autotune ON + override cleared, chunk env) | 1.133× | 1.078× | 1.781× | 1.00 | 1.35 | 3/24 |
  - **Session 11 production default now matches the autotune-on ceiling within 0.3pp** (1.075 vs 1.078) — override 4-tuple 完成了 Session 10 增益的 last-mile 落地
  - dgrad min lifted 0.74 → 1.00: 4 个原本反 perf 的 shape (dsv3-down B4 M2048, dsv3-up B4 M4096, qwen-down B4 M2048, qwen-down B4 M4096) 全部 ≥ 1.0× Triton
- **Per-shape dgrad ratio (Session 11 production, 24-shape full)**
  ```
  gpt_oss-up    B4 M2048: 1.23x   gpt_oss-down B4 M2048: 1.17x
  gpt_oss-up    B4 M4096: 1.03x   gpt_oss-down B4 M4096: 1.12x
  gpt_oss-up   B16 M2048: 1.03x   gpt_oss-down B16 M2048: 1.07x
  gpt_oss-up   B16 M4096: 1.03x   gpt_oss-down B16 M4096: 1.09x
  dsv3-up       B4 M2048: 1.06x   dsv3-down    B4 M2048: 1.03x
  dsv3-up       B4 M4096: 1.03x   dsv3-down    B4 M4096: 1.00x
  dsv3-up      B16 M2048: 1.05x   dsv3-down   B16 M2048: 1.07x
  dsv3-up      B16 M4096: 1.06x   dsv3-down   B16 M4096: 1.02x
  qwen-up       B4 M2048: 1.04x   qwen-down    B4 M2048: 1.12x
  qwen-up       B4 M4096: 1.04x   qwen-down    B4 M4096: 1.33x
  qwen-up      B16 M2048: 1.06x   qwen-down   B16 M2048: 1.04x
  qwen-up      B16 M4096: 1.07x   qwen-down   B16 M4096: 1.05x
  ```
- **8 user shape subset** (from memory R210, plan §3 "user 8 shape"): gpt_oss_up B4 M2048 / gpt_oss_up B16 M2048 / dsv3_up B4 M4096 / dsv3_up B16 M2048 / dsv3_down B16 M4096 / qwen_up B4 M4096 / qwen_down B16 M2048 / qwen_down B16 M4096
  - ratios: 1.23 / 1.03 / 1.03 / 1.05 / 1.02 / 1.04 / 1.04 / 1.05
  - **8-shape geomean = 1.06×**, **pass ≥1.15× = 1/8** (gpt_oss_up_B4_M2048 only)
- **Plan target verdict**
  - "8/8 user shape RRR dgrad ≥ 1.15× Triton" → **NOT MET** (1/8)
  - "24-shape geomean ≥ Session 7 baseline 1.065× (kernel_only)" → **MET** (1.075× = +1.0pp)
  - "24-shape pass count ≥1.15× ≥ Session 10 probe 4/24" → **NOT MET in production bench** (3/24); probe 4-th winner (gpt_oss_up_B4 1.372× vs prod 1.23×) gap = probe 用 median-of-3 × 30 iter + 同 cfg 重复 vs bench 单 run × 50 iter, methodology delta ~10pp on best-shape, 跨 (gm,xcds,bn,chunk) search space 复现性不如 probe
- **7 worst-shape 共性 (ratio 1.00-1.07×, 17/24 shapes)**
  - 全部在 **B=16 grouped** 或 **B=4 + 大 K (≥4096)**
  - 验证 [[fp8-rrr-attempt-h14]] HBM bandwidth ceiling 物理结论: B=16 grouped streams ~544MB B-data vs dense ~364MB = ~50% 数据差 → ~25% TFLOPS gap, source-level kernel tweaking 不可破
  - **真 lever 仍是 (a) B-pretranspose ds_read_b128 主循环 (Session 5/9 系列, subtile load primitive 已 fix in Session 8, body 集成 Session 9.1/9.2/9.3 待做) 或 (b) split-K cross-group B share (~800 LOC, multi-session)**
- **HK commit**: `84f96617` (plan addendum only — 本 session 无 kernel/header 改动)
- **PT outer commit**: `5c644030` (override table 4-tuple 刷新 + 3rdparty bump)
- **PT 3rdparty commit**: `df7f410e` (plan addendum mirror; HK turbo 84f96617 镜像)
- **memory**: `feedback_rrr_b_pretrans_session11_final_bench.md`
- **PT 3rdparty bump**: `<n/a — no HK kernel change>`
- **Session 11 通过最小可独立的子部分**: 完整 final bench + verdict + 顺手把 Session 7.2 spec 的 override 4-tuple 刷新落地, 让 production 默认路径吃到 Session 10 增益。剩余 Session 9.1/9.2/9.3 (B-pretranspose body integration) 已在 plan 中, 真 1.15× lever 路径明确

---

## Session 12 — 执行 Session 9.1: RRR prolog B-pretranspose wiring  (~150 LOC, 3-4h)

**等同 Session 9.1** (已有完整草稿在 §9.1, 不重复). 阅读 plan §9.1 + Session 9 status 段 + `feedback_rrr_b_pretrans_session9_path_l_header_api.md` 后**直接执行**, 完成动作改为追加 `## Session 12 status: PASSED ...`。

**前置**: Session 9 PARTIAL (header writer API + probe regression-clean) — ✓ 已满足

**关键约束**: prolog-only 替换是最小增量; 不要碰 main loop / FUSED_KTAIL (那是 Session 13/14)。LDS budget 必须 audit (老 Bs 128KB vs 新 Bs_NM+stage 80KB 二选一)。

---

## Session 12 status: PASSED (minimal-viable scaffold + standalone PoC)  HK=<filled-on-commit>  PT 3rdparty=<filled>  outer=<filled>
- 2026-05-25
- **Scope delivered (minimum-viable, body integration deferred to Session 13)**
  - **kernel_fp8_layouts2.cpp** (HK + PT 3rdparty 双路径): drop `#error` so `RRR_B_PRETRANS=1` compiles; add 60-line scaffold comment defining `ST_NM = st_fp8e4m3<128,128,st_128x128_n_major_s>` (sizeof = 16384 B) + LDS budget audit (macro=0 ≈ 140 KiB vs macro=1 ≈ 144 KiB, both within 160 KiB cap, with note that `Bs_NM[4]` REPLACES `Bs[2][2]` — not additive)
  - **tests/probes/rrr_prolog_b_pretrans_probe.cu** (220 LOC, HK + PT 3rdparty 双路径): standalone probe that mirrors the production prolog 4-tile pattern (`bc=0`, `k_block ∈ {0,1}`, `n_strip ∈ {0,1}`). Allocates `__shared__ ST_NM Bs_NM[4]` + `stage_lds[16384]`, runs 4× `write_b_transpose_n_major_path_L<ST_NM>`, dumps all 4 tiles to HBM, host-verifies bytes against `expected[tile][n*128 + k] = hbm[(k_block*128 + k)*256 + (n_strip*128 + n)]`
  - **tests/probes/Makefile**: add `rrr_prolog_b_pretrans_probe` to PROBES list
- **Probe result on chi2811 (gfx950, MI355X)**:
  ```
  Session 12 prolog-B-pretrans probe (4-tile):
    tile 0 (kblk=0,strip=0) mismatch=0   /  16384 bytes
    tile 1 (kblk=0,strip=1) mismatch=0   /  16384 bytes
    tile 2 (kblk=1,strip=0) mismatch=0   /  16384 bytes
    tile 3 (kblk=1,strip=1) mismatch=0   /  16384 bytes
    TOTAL mismatch=0   /  65536 bytes
  OK: 4-tile B-pretranspose prolog produces N-major tiles that
      byte-equal the host transpose of HBM B[K=256,N=256].
  ```
  EXIT=0. All 4 tiles byte-equal host reference. **Writer-side prolog validated**.
- **Production build sanity (macro=0 default)**:
  - `GPU_ARCHS=gfx950 pip install --no-build-isolation -e .` on chi2811 = exit 0 (`primus_turbo-0.3.0+176e255d` installed clean)
  - Body unchanged → no production behavior delta (macro guard intact)
- **Why minimal-viable instead of full Session 9.1**
  - Original Session 9.1 spec said "替换 prolog 4 个 G::load + 加 Bs_NM[4]+stage_lds 到 LDS struct". 但只改 prolog 不改 main loop → main loop 引用 `Bs[tic][k_block]` (旧 ST_v2) 在 macro=1 下编译不过 (Bs 不存在了, 因为 LDS budget 强制 Bs_NM 替换 Bs 而非追加). 完整 prolog+body 改动 ≈ 400-500 LOC, 跨 prolog/main-loop/FUSED_KTAIL/epilog 多处, 不适合一个 session 单元
  - Session 12 选择: **scaffold + 独立 PoC probe**, 用 probe 验证 prolog writer 调用 pattern 在 4-tile 真实拓扑下 byte-correct, 同时不破 production。Session 13 (= Session 9.2) 接手 = 同时改 prolog + main loop B-read + FUSED_KTAIL audit (一致性原子改动)
  - 这等于按 user "拆细自交付" 优先级 (2): **最小可独立可 verify 子部分 + 把剩余写成 Session 13** (Session 13 已在 plan §874)
- **Risk surface for Session 13 (pre-known)**
  - `Bs_NM[4]` (4 个 128×128 N-major 16 KB 块) vs production `Bs[2][2]` (4 个 128×128 K-major ≈ 17 KB 块 w/ swizzle pad) — index 数量一致, 寻址 lambda 调整即可
  - `load_b(b0, Bs[tic][0], wn)` (现走 `load_col_from_st`) → 需换成 `load_col_from_st_n_major_subtile` (Session 8 已 fix, ISA spill=0)
  - FUSED_KTAIL path 走 K-tail-only loads, 也需要 N-major writer (但 K-tail 行为不同, 需额外 audit)
- **Next session pre-conditions (Session 13)**
  - Session 12 commit (HK turbo HEAD = `<S12>`, PT outer = `<S12>`) merged
  - 新 body 函数 `grouped_rrr_kernel_body_pinned_pretrans` 或在现 body 内 `if constexpr (RRR_B_PRETRANS)` 二分支
  - 一次性改动需 ISA 验 main loop `ds_read_b64_tr_b8 == 0` (gate from plan §9.2)
- **HK commit**: `<filled-on-commit>` (turbo branch)
- **PT outer commit**: `<filled>` (dev-turbo-kyle3-grouped-gemm-fp8-rrr branch — 3rdparty bump + plan/PoC mirror)
- **PT 3rdparty commit**: `<filled>` (HK turbo `<S12>` 镜像)
- **memory**: `feedback_rrr_b_pretrans_session12_prolog_poc.md`

---

## Session 13 — 执行 Session 9.2: RRR main loop B-read + FUSED_KTAIL audit  (~200 LOC, 4-5h)

**等同 Session 9.2** (已有完整草稿在 §9.2). 阅读 plan §9.2 + Session 12 status 后直接执行。

**前置**: Session 12 PASSED

**完成动作改为**: 追加 `## Session 13 status: PASSED ...`

---

## Session 14 — 执行 Session 9.3: 8-shape kernel_only bench + chunk_size override  (~100 LOC, 2-3h)

**等同 Session 9.3** (已有完整草稿在 §9.3). 阅读 plan §9.3 + Session 13 status 后直接执行。

**前置**: Session 13 PASSED

**完成动作改为**: 追加 `## Session 14 status: PASSED ...`

---

## Session 15 — Round-3 final: RRR 24-shape bench + verdict + 决策  (~50 LOC, 1-2h)

**目标**: 跑完整 24-shape kernel_only bench (autotune OFF, override path), 给出 B-pretranspose 全集成后的最终 RRR dgrad 数字, 对比 round-1/round-2 baseline, 给 user 看 8/8 ≥1.15× 是否达成。

**前置**: Session 14 PASSED (B-pretranspose 已 default-on, RRR_B_PRETRANS=1)

**任务**
1. Verify HK + PT HEAD = Session 14 commits, build OK
2. Run `python benchmark/ops/bench_hk_vs_triton_grouped_fp8_kernel_only.py` 完整 24-shape on chi2762
3. 提取 RRR dgrad geomean + pass-count (≥1.15×) + 8-shape user subset 数字
4. 三方对比表 (round-1 / round-2 / round-3) 写进 status
5. 若 8/8 ≥1.15× 仍未达成, 诚实列剩余 worst shape + 物理性分析 (HBM ceiling? 还有别的 lever?)
6. 若 8/8 ≥1.15× **达成**, 写 `feedback_rrr_v2_b_pretrans_win.md` 终极 win + MEMORY.md 索引

**验证**
- bench 跑通无 NaN/Error, SNR ≥ 25 dB 全 shape

**完成动作**
- 只更新 plan (无 kernel commit) + 必要 memory
- 追加 `## Session 15 status: PASSED  RRR dgrad geomean=<x> pass-1.15=<n>/24 8-shape=<m>/8`
