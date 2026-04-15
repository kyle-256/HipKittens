# MXFP4 GEMM 优化 — Agent提示词

你是MXFP4 GEMM优化项目的技术负责人。项目在 /shared_nfs/kyle/test/HipKittens, branch: mxfp4。

## 目标
- **主测shape**: 4096×32768×128256
- **aiter基线**: 5653 TFLOPS (MI355X)
- **目标**: 97% of aiter = **5484 TFLOPS**
- **当前最佳**: 4926 TFLOPS (原版C++ GM=8, 87.1% of aiter)

## 当前状态 (2026-04-15)

### 生产kernel (原版C++)
- `kernel_mxfp4_gluon_cpp.cpp`: 4926T (GM=8), 252 VGPRs, 0 spills
- 这是C++框架的结构极限，无法通过增量优化突破

### ART kernel (实验性)
- `kernel_mxfp4_art.cpp`: 3185T, 256V+256A, 0 spills, direct-A loading
- 单monolithic asm block (128 MFMAs + 32 loads + barrier)
- Bit-exact correct vs 原版

### 差距分析 (4926T → 5484T, 需要+558T = +11.3%)
| 因素 | 我们 | aiter | 影响 |
|------|------|-------|------|
| A tile loading | LDS round-trip (48 ds_reads) | Direct global (20 ds_reads) | ~5-8% |
| Loop VALU | 8 v_cndmask + overhead | 0 VALU | ~3-5% |
| Store | 256× scalar 2B stores | 32× vectorized 16B stores | ~2-5% |
| Instruction scheduling | Compiler-managed | Hand-scheduled ASM | ~5-10% |

## 已穷尽的方向 (严禁重试)
- ASM inline kernel参数化 (严禁使用)
- Gluon LLIR kernel直接集成 (不允许)
- XOR double-buffer toggle → 无效 (compiler已优化)
- Compiler flags (-mllvm options) → 无效
- UNROLL_K sweep → 无效 (K=128256太大)
- Vectorized store (LDS transpose) → 反而更慢
- Half-direct-A (A0 direct, A1 LDS) → 反而更慢 (global load延迟>LDS)
- Direct-A C++ (无ART) → 23 spills杀性能

## 唯一可行路径: Full HipKittens3 ART Rewrite

### Phase A: MFMA operand swap
- 交换MFMA A/B operands: B-matrix→MFMA-A, A-matrix→MFMA-B
- 每个thread得到4个连续COLUMN → row-oriented output
- 使能 v_permlane16_swap + buffer_store_dwordx4 vectorized store

### Phase B: Full hand-scheduled inner loop
- 用HipKittens3 ART framework (`art<>` tiles)
- 所有128 MFMAs + loads + scales + barrier在一个hand-scheduled序列
- 0 VALU in loop
- Direct-A loading with perfect latency hiding (关键！)
- 参考: /shared_nfs/kyle/HipKittens3/kernels/attn/gqa_causal_backwards/attn_bkwd_causal.cpp

### Phase C: 42-shape validation
- bench_all_42.py跑全量对比
- 每shape auto-tune GROUP_SIZE_M
- 所有shape ≥ 97% of aiter

## 关键发现 (必读)
1. **Direct-A without latency hiding = SLOWER** — buffer_load 200+cy vs ds_read 20-40cy
2. **Compiler clobbers AGPRs during store** → 必须在单asm block读完所有64 AGPRs
3. **`"=&v"` early-clobber** → 所有64个output operand必须用`"=&v"`防止alias
4. **UNROLL_K>1 + K=128256** → ART kernel crash (code size limit), 用UNROLL_K=1
5. **aiter输出确认是标准row-major** (不是transposed)
6. **v_mfma_scale_f32_16x16x128_f8f6f4 macro** 已添加到HipKittens3 macros.cuh

## 环境
- MI355X (gfx950), 8 GPUs
- benchmark规则: warmup=200, iters=500, trimmed mean 10%, HIP_VISIBLE_DEVICES=N
- triton_for_gluon: /shared_nfs/kyle/test/triton_for_gluon (matmul_4waves branch)
- aiter: /shared_nfs/kyle/test/aiter
- HipKittens3 ART框架: /shared_nfs/kyle/HipKittens3

## 工作方式
- **所有agent必须使用opus模型** (model: opus)
- 不要sleep()轮询, 用子agent监控
- 每个改动: 编译→正确性→性能→commit或revert
- commit用: `git -c user.name="kyle-256" -c user.email="Kyle.Zhao@amd.com" commit`
- 完整优化路线图: `OPTIMIZATION_ROADMAP.md`

## 关键文件
| 文件 | 用途 | 性能 |
|------|------|------|
| `kernel_mxfp4_gluon_cpp.cpp` | 生产C++内核 | 4926T (GM=8) |
| `kernel_mxfp4_art.cpp` | ART实验内核 | 3185T |
| `kernel_mxfp4_half_direct.cpp` | Half-direct实验 | 3703T |
| `bench_all_42.py` | 42-shape benchmark | — |
| `bench_aiter_a4w4_42.py` | aiter竞品benchmark | — |
| `test_mxfp4_gluon_cpp.py` | 快速正确性测试 | — |
| `art_register_map.md` | ART寄存器分配 | — |
| `OPTIMIZATION_ROADMAP.md` | 完整优化路线图 | — |
