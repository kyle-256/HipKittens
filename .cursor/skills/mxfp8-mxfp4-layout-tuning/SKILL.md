---
name: mxfp8-mxfp4-layout-tuning
description: Tune HipKittens MXFP8 and MXFP4 microscaling GEMM kernels on gfx950/MI350X. Use when working on analysis/fp8_gemm/mi350x MXFP8/MXFP4 RCR performance, preshuffle-quant scale packing, scale-pack scheduling, buffer_load SRD, KPAIR_LOOP, mfma_scale, SNR, or determinism.
---
# MXFP8 / MXFP4 Microscaling Layout Tuning

## When To Use
- User asks to debug or optimize `analysis/fp8_gemm/mi350x` **MXFP8** or **MXFP4** GEMM.
- User mentions `mxfp8`, `mxfp4`, `microscaling`, `block-scale`, `preshuffle-quant`, `mfma_scale`, `scale-pack`, `E8M0`, `KPAIR_LOOP`, `buffer_load SRD`, or `sched_group_barrier` in the context of scaled FP8/FP4.
- For per-tensor FP8 work, use the `fp8-per-tensor-layout-tuning` skill instead.

## First Read
- Read `analysis/fp8_gemm/mi350x/README.md` for the current task state and validated numbers.
- Read this file's reference section for known dead ends and durable findings.

## Hard Rules
- Formal shape is `8192x8192x8192`.
- `preshuffle-quant` is valid for scale tensors only. `preshuffle-ab` is **not allowed**.
- Both `A_scale` and `B_scale` are preshuffled in the current PQ path.
- Success means: numerical correctness pass, `SNR > 48 dB`, deterministic output, TFLOPS within target.
- Benchmark with batch timing: warmup 100 iterations, measure 200 iterations contiguously (no per-iteration `torch.cuda.synchronize()`).
- Do not claim a win from short runs alone.
- Only create a git commit when the user explicitly asks.
- Do not commit generated artifacts (`.tmp_*.json`, ISA dumps, `.co` files, `.bak*`).

## Primary Files
- `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp` — main MXFP8 kernel (8-wave + 4-wave)
- `analysis/fp8_gemm/mi350x/rcr_mxfp8_4wave_fastpath.inc` — 4-wave MXFP8 RCR kernel (inline ASM)
- `analysis/fp8_gemm/mi350x/kernel_mxfp8_4wave_rewrite.cpp` — 4-wave MXFP8 pure builtin (2853 TFLOPS)
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_colwise.cpp` — MXFP4 colwise KPAIR kernel (4-wave, 4241 TFLOPS)
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp` — MXFP4 Gluon-arch C++ reimplementation (4-wave, 4412 TFLOPS)
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_hybrid.cpp` — MXFP4 hybrid kernel (4-wave, 4166 TFLOPS)
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_asm.s` — gluon reference MXFP4 .s (5109 TFLOPS)
- `analysis/fp8_gemm/mi350x/test_mxfp4_gluon_cpp.py` — Gluon C++ benchmark/correctness harness
- `analysis/fp8_gemm/mi350x/kernel_mxfp4_asm_inline.cpp` — gluon wrapper with pybind11
- `analysis/fp8_gemm/mi350x/test_mxfp8_python.py` — MXFP8 benchmark/correctness harness
- `analysis/fp8_gemm/mi350x/test_mxfp4_hybrid.py` — MXFP4 benchmark/correctness harness
- `analysis/fp8_gemm/mi350x/rewrite_mxfp8.py` — Python .s rewriter for MXFP8
- `analysis/fp8_gemm/mi350x/rewrite_mxfp4.py` — Python .s rewriter for MXFP4
- `analysis/fp8_gemm/mi350x/build_rewrite.sh` — end-to-end C++ → .s → Python → .so pipeline
- `analysis/fp8_gemm/mi350x/Makefile` — build with CPPFLAGS macros
- `include/ops/warp/memory/util/util.cuh` — `make_srsrc`, `llvm_amdgcn_raw_buffer_load_b32`

## Build Commands
### MXFP8 current best (8-wave, buffer_load + KPAIR_LOOP)
```bash
cd analysis/fp8_gemm/mi350x
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) ROCM_PATH=/opt/rocm \
  CPPFLAGS='-DM_DIM=8192 -DN_DIM=8192 -DK_DIM=8192 -DMXFP8_RCR_EXACT_8WAVE_FAST_ENABLE=1 -DMXFP8_RCR_EXACT_PQ_KPAIR_LOOP_ENABLE=1 -DMXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE=1' \
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp
```

### MXFP4 hybrid (4-wave)
```bash
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) ROCM_PATH=/opt/rocm \
  CPPFLAGS='-DM_DIM=8192 -DN_DIM=8192 -DK_DIM=8192' \
  make -B TARGET=tk_mxfp4_hybrid SRC=kernel_mxfp4_hybrid.cpp
```

### Smoke test
```bash
# MXFP8
MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr MXFP8_WARMUP=5 MXFP8_ITERS=10 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
  python3 test_mxfp8_python.py 256 256 256
# MXFP4
MXFP4_PRESHUFFLE_QUANT=1 MXFP4_WARMUP=5 MXFP4_ITERS=10 MXFP4_CHECK=1 MXFP4_DETERMINISM_RUNS=3 \
  python3 test_mxfp4_hybrid.py 256 256 256
```

### Formal benchmark
```bash
# MXFP8
MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr MXFP8_WARMUP=100 MXFP8_ITERS=200 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=5 \
  python3 test_mxfp8_python.py 8192 8192 8192
# MXFP4
MXFP4_PRESHUFFLE_QUANT=1 MXFP4_WARMUP=100 MXFP4_ITERS=200 MXFP4_CHECK=1 MXFP4_DETERMINISM_RUNS=5 \
  python3 test_mxfp4_hybrid.py 8192 8192 8192
```

## Current Performance (batch timing, 8192^3)

### MXFP8
| Version | TFLOPS (with zero) | TFLOPS (pure) | Spills | SNR | vs FP8 |
| --- | ---: | ---: | ---: | --- | --- |
| FP8 per-tensor | 3335 | 3335 | 0 | - | 100% |
| MXFP8 8-wave (KPAIR+SRD) | 2848 | 3011 | 3 | 49.60 dB | 90.3% |
| **MXFP8 4-wave builtin** | **2853** | ~3020 | **1** | 49.60 dB | 90.5% |
| MXFP8 4-wave (old inline ASM) | 2118 | ~2250 | 0 | 49.60 dB | 67.5% |

### MXFP4
| Version | TFLOPS (with zero) | TFLOPS (pure) | Spills | SNR | vs Gluon |
| --- | ---: | ---: | ---: | --- | --- |
| Gluon reference (.s) | 5109 | ~5300 | 0 | 55.62 dB | 100% |
| aiter asm (128×256) | — | — | 0 | — | ~100% |
| **MXFP4 Gluon C++** | **4412** | ~4650 | **0** | ~25 dB* | **86.4%** |
| MXFP4 colwise KPAIR | 4241 | ~4510 | 0 | 49.62 dB | 83.0% |
| MXFP4 hybrid (4-wave) | 4166 | ~4440 | 0 | 49.62 dB | 81.5% |
| MXFP4 builtin (4-wave) | 4299 | 4575 | 0* | 49.62 dB | 84.1% |
| MXFP4 V2 (4-wave, pre-hybrid) | 4032 | 4257 | 0 | 49.60 dB | 78.8% |
| MXFP4 8-wave | 3466 | 3631 | 0 | 49.60 dB | 67.2% |

*builtin has 144 bytes scratch from fp4_intx4_t array spills, but no register spills.

## Architecture Comparison: 4-wave vs 8-wave

| Property | 4-wave | 8-wave |
| --- | --- | --- |
| Warp grid | 2×2 | 2×4 |
| Threads | 256 | 512 |
| Occupancy | 1 (or 2 if VGPRs allow) | 2 |
| LDS per workgroup | 128–139 KB (double-buffer) | 64 KB (single-buffer) |
| AGPRs per warp | 256 available | 256 available |
| Latency hiding | Double-buffer pipeline | Wave-level parallelism |
| Scale pipeline room | Yes (if VGPRs < 256) | No (VGPRs = 256) |

**Key finding**: 4-wave consistently outperforms 8-wave for MXFP4 (+26%). For MXFP8, 8-wave currently leads because 4-wave lacks KPAIR_LOOP and inline ASM optimizations.

## MXFP4 Hybrid Architecture (kernel_mxfp4_hybrid.cpp)

### Key Optimizations (validated, +6.4% over V2)
1. **Barrier repositioned**: After all tile data extracted to VGPRs (not after Phase 1). Lets 96 MFMAs overlap with tile prefetch.
2. **Intra-block MFMA + tile prefetch**: 4 asm blocks × 4 MFMA with `emit_one_pf` (compiler intrinsic `llvm_amdgcn_raw_buffer_load_lds`) between blocks. All 16 AGPRs declared `"+a"` per block to prevent hipcc AGPR reuse bug.
3. **Custom `make_pf_params`**: Pre-computes SRD/soffset/LDS addrs for prefetch, avoiding repeated `readfirstlane`.

### hipcc Compiler Limitations Discovered
1. **`#pragma unroll 1` breaks correctness**: hipcc loses VGPR liveness tracking for `asm volatile` blocks without unrolling. Must use `#pragma unroll 2`.
2. **Split asm blocks cause AGPR reuse**: 4 separate 4-MFMA blocks each declaring only their 4 AGPRs → compiler reuses same AGPRs across blocks. Fix: declare all 16 AGPRs as `"+a"` in every block.
3. **`buffer_load_dwordx4 ... lds` in inline ASM**: `"s"(int32x4_t)` constraint does not correctly generate 4-SGPR range for SRD. Fix: use compiler intrinsic `llvm_amdgcn_raw_buffer_load_lds` instead.
4. **`coord<>{}` vs brace-init**: `coord<>{0,0,r,c}` creates untyped coord where `unit_coord<2,3>()` doesn't scale by tile dimensions. Use brace-init `{0,0,r,c}` to let the compiler deduce `coord<ST>`.
5. **Scale pipeline across loop boundary**: Moving `load_scale_buffer` before/after the KPAIR loop causes 150–367 VGPR spills on the 8-wave MXFP8 kernel (256 VGPR limit, zero headroom). On 4-wave MXFP4 (14–44 VGPR headroom) this works fine.

## MXFP8 4-Wave Optimization Plan (Next Priority)

The MXFP8 4-wave kernel (`rcr_mxfp8_4wave_fastpath.inc`) is the most promising path for exceeding 8-wave performance. Current state:
- **VGPRs: 212–242, AGPRs: 0, Spills: 0** — 14–44 VGPRs headroom
- Accumulators in VGPRs (not AGPRs) → extra VGPR↔AGPR copy overhead per MFMA
- No KPAIR_LOOP, no buffer_load SRD, no inline ASM MFMAs

Planned optimizations (in priority order):
1. **Inline ASM MFMAs with AGPR accumulators**: Move acc from VGPRs to AGPRs via `"+a"` constraints. Eliminates copy overhead, frees ~64 VGPRs.
2. **KPAIR_LOOP**: Manual 2× unroll of K-pair loop (same technique as 8-wave, +13% expected).
3. **buffer_load SRD for scales**: Same SRD technique as 8-wave, reduces scale load overhead.
4. **Scale prefetch pipeline**: With AGPR accumulators freeing VGPRs, there's room to prefetch k_pair+1 scales during k_pair MFMAs (impossible on 8-wave due to 256 VGPR limit).
5. **Tile prefetch interleaving**: Same `emit_one_pf` technique from MXFP4 hybrid — issue `buffer_load_to_lds` between split MFMA asm blocks.

Expected outcome: close to or exceeding 8-wave's 3011 TFLOPS, with potential for further gains from scale pipeline.

## Durable Findings

### MXFP8
- KPAIR_LOOP (2× manual unroll) is the single biggest optimization for 8-wave (+13%).
- buffer_load with SGPR SRD reduces spills from 8 to 3 and adds +1%.
- The remaining ~9% gap to FP8 per-tensor: scale vmcnt stall (~5%), scale remap v_lshr_b32 (~2.5%), mfma_scale I-cache overhead (~1.6%).
- Software pipelining of scale loads on 8-wave causes 150–367 VGPR spills (256 limit, zero headroom). **This does NOT apply to 4-wave** (14–44 VGPRs headroom).
- LDS caching of scales is infeasible: tile double-buffers already consume 64–128 KB LDS.
- `sched_group_barrier` variants tested neutral to slightly worse vs `sched_barrier(0)`.
- **Pure `__builtin` MFMAs (no inline ASM) match inline ASM performance** on the 4-wave kernel: 2853 TFLOPS vs 2848 (8-wave KPAIR+SRD). The builtin approach eliminates ACC16 macro complexity, AGPR reuse bugs, and `#pragma unroll` correctness issues. File: `kernel_mxfp8_4wave_rewrite.cpp`.
- The compiler's AGPR shuffle overhead (404 ops/loop) is fully hidden in MFMA latency (128 MFMAs × 64 cycles >> 404 × 1 cycle). Optimizing AGPR scheduling yields zero performance gain.
- SALU instructions (`s_mov_b32` etc.) dual-issue with MFMA on gfx950 — they are zero-cost and NOT a performance bottleneck.

### MXFP4
- 4-wave with 128KB double-buffer outperforms 8-wave with occupancy 2 by 26%.
- Intra-block tile prefetch (between split 4-MFMA asm blocks) gives +3.4% over inter-block.
- Distributing prefetch loads more evenly (2+2+2+2+4+4 vs 4+4+4+4+0+0) does NOT help — more code paths hurt compiler optimization.
- N-split reordering (DOT_left/DOT_right) causes non-deterministic errors with hipcc (suspected register tracking bug across reordered asm blocks).
- **hipcc `"=v"` constraint does NOT enforce early-clobber correctly** — ds_read outputs can share VGPRs with MFMA inputs even with `"=&v"`. Changing ds_read:MFMA interleaving ratio from 2:1 to 1:1 within an asm block causes correctness failure (SNR 9.56 dB). The Python .s rewriter CAN safely do this transformation (VGPRs already allocated in .s). This is the one proven case where .s post-processing adds value C++ cannot.
- **gfx950 register budget**: VGPRs + AGPRs share 512 physical registers. With 256 AGPRs for accumulators, max VGPRs = 256. Gluon's `.vgpr_count: 512` = 256 VGPRs + 256 AGPRs combined. Both gluon and hybrid use identical register budgets.
- **Gluon's 21% advantage (5234 vs 4303 TFLOPS) comes from column-first MFMA ordering**: gluon fixes A data and cycles B columns (8 VGPRs/col), allowing 1:1 ds_read:MFMA interleaving with 8-MFMA groups. Hybrid uses row-first ordering (full B tiles, 32 VGPRs), requiring 2:1 ratio and lgkmcnt stalls between phases. Changing sub-tile GROUP order (A0×Bl→A0×Br vs A0×Bl→A1×Bl) has no effect — the inline ASM blocks internally use row-first.
- **Gluon uses 135 KB LDS** (138144 bytes, via dynamic allocation), 7 KB more than hybrid's 128 KB. Extra space used for LDS-based scale broadcast (ds_write + ds_read_b64_tr_b8), reducing global scale loads from 16 to 6 per loop body.
- Pure `__builtin` MFMAs on MXFP4 match inline ASM performance (4299 vs 4303 TFLOPS). The ACC16 `"+a"` constraint eliminates AGPR shuffling (0 AGPR ops in hybrid vs 404 in MXFP8 builtin), but this doesn't translate to measurable performance difference.

### C++ → .s → Python Pipeline
- **End-to-end pipeline validated**: `hipcc --offload-device-only -S` → Python rewrite → `clang -x assembler` → `ld.lld` → `clang-offload-bundler` → `clang -cc1 -fcuda-include-gpubinary` → `ld.lld` → `.so`. Build script: `build_rewrite.sh`.
- **Python .s rewriter's proven value**: bypasses hipcc's `"=v"` VGPR reuse bug for ds_read/MFMA reordering. C++ inline ASM cannot do 1:1 ds_read interleaving safely; Python .s can.
- **Python .s rewriter's limitations**: instruction scheduling changes (AGPR redistribution, cross-barrier MFMA movement, ds_read timing) yield 0% improvement on both MXFP8 and MXFP4. The compiler's scheduling is already near-optimal for the given data flow. The bottleneck is algorithmic (MFMA ordering pattern, scale loading architecture), not scheduling.

### General
- Per-iteration `torch.cuda.synchronize()` causes GPU DVFS clock drops; always use batch timing.
- `C.zero_()` inside the timed loop adds ~5–7% overhead; pure GEMM numbers are the fair comparison.
- hipcc `#pragma unroll 2` is critical for correctness with split asm volatile blocks.
- SALU instructions (s_mov_b32, s_add_i32, etc.) dual-issue with MFMA/VALU on CDNA4 — they are effectively free and should NOT be counted as overhead.
- `amdgpu_num_vgpr(N)` attribute sets upper limit only, does not force allocation. Compiler allocates based on code demand.

## Known Dead Ends
- `4de0a032` buffer_load scale path: fast but invalid at large K (correctness/determinism fail).
- Shared/LDS scale cache: correctness problems, no validated win.
- `phase1` `op_sel_hi` asm variants: correctness works but long-run performance regressed.
- Scalar phase-pack caching and remap micro-optimizations: did not beat current schedule.
- Inline asm async prefetch (8-wave): achieved 2929–3052 TFLOPS but failed correctness.
- L2 prefetch with `buffer_load_dword` discard VGPR: caused 190 spills.
- `sched_group_barrier` replacing `sched_barrier(0)`: -0.8% regression.
- MXFP8 8-wave scale pipeline (any form): 256 VGPR hard limit prevents cross-iteration liveness.
- MXFP4 `buffer_load_dwordx4 ... lds` in single asm block: SRD `"s"(int32x4_t)` constraint broken.
- MXFP4 N-split DOT_left/DOT_right reorder: non-deterministic correctness failure.
- MXFP4 changing ds_read:MFMA ratio from 2:1 to 1:1 in C++ inline ASM: hipcc `"=v"`/`"=&v"` does not prevent VGPR sharing → correctness failure. Must use Python .s rewriter for this transformation.
- MXFP4 4-tile upfront loading (all ds_reads before MFMAs): 4261 TFLOPS, SLOWER than interleaved (4303), because ds_read/MFMA overlap is lost.
- MXFP4 removing `#pragma unroll 2`: no effect (compiler still unrolls).
- MXFP4 removing scale prefetch (halving VMEM loads): no effect (VMEM bandwidth not the bottleneck).
- MXFP4 VGPR double-buffering (prev+cur tiles simultaneously): 439 spills because VGPRs+AGPRs=512 is a HARD limit, not expandable. Performance = baseline despite spills (scratch hidden in MFMA latency).
- MXFP4 sub-tile group ordering (A0×Bl→A1×Bl vs A0×Bl→A0×Br): no effect — inline ASM blocks internally use row-first regardless of call order.
- MXFP4 Gluon C++ swap-based LDS address ping-pong: 4100 TFLOPS (WORSE than 4268 baseline). 8 uint32_t swaps = 28 v_mov_b32 overhead per iteration.
- MXFP4 Gluon C++ barrier before Step 1 (instead of Steps 2→3): 4370 TFLOPS (-40 from baseline). Barrier latency fully exposed on critical path; the Step 2→3 position allows MFMA pipeline drain to hide barrier cost.
- MXFP4 Gluon C++ scale loads after Step 2: 4073 TFLOPS (-340 from baseline). Moving 8 buffer_load_dword between Step 2 and barrier increases vmcnt(8) stall before the barrier.
- MXFP8/MXFP4 Python .s rewriter scheduling optimizations (AGPR redistribution, cross-barrier MFMA split, ds_read early issue): all yield 0% improvement. Compiler scheduling is already near-optimal.
- MXFP4 colwise KPAIR with BK=128 upfront loading: eliminates remap_phase VALU (+1.8% over hybrid) but BK=128 means 32 K-iterations with 2 barriers each. Cannot match Gluon's BK=256 (16 iterations) without column-first incremental B loading, which requires per-column LDS reads and single-buffer architecture.

## MXFP4 Column-First KPAIR Kernel (kernel_mxfp4_colwise.cpp)

### Gluon C++ reimplementation: 4412 TFLOPS (86.4% of Gluon .s)
- Reimplemented Gluon Python kernel architecture in C++ with ThunderKittens framework
- 0 spills, 256 VGPRs + 256 AGPRs, 0 scratch
- **SNR ~25 dB** (speed prioritized over correctness per user directive; scale vmcnt issue suspected)
- Key scheduling optimizations applied:
  - **ds_read_b128 front-loading**: all 8 LDS reads interleaved 1:1 with first 8 MFMAs in each kpair block
  - **Named LDS variables + v_cndmask**: eliminated compiler-generated ds_read_b64 stalls from runtime-indexed arrays. 16 static named variables with ternary selection (8 v_cndmask per iteration, 0 ds_read_b64)
  - **Hoisted pf_params**: make_pf_params computation moved before Steps 1-2, overlapping with MFMAs. Reduced Step2→Step3 gap from ~80 ASM instructions to 5 (lgkmcnt + vmcnt + barrier)
- **Remaining 13.6% gap root cause**: algorithmic, not scheduling. Our 4-step × 32-iteration loop vs Gluon's column-first BK=256 × 16-iteration structure. Gluon achieves better MFMA utilization through fewer loop iterations, finer-grained barrier placement, and tighter prefetch scheduling.

### MXFP4 colwise KPAIR: 4240 TFLOPS (+1.8% over hybrid 4166)
- SNR 49.62 dB, determinism PASS (5 runs), 0 spills, 256 VGPRs + 256 AGPRs

### Architecture
- **KPAIR encoding**: `op_sel[0/1]` = A/B sub-group, `op_sel_hi[0/1]` = K-phase (0=lo bytes 0-1, 1=hi bytes 2-3)
- **Preshuffle scale layout**: 4-byte dword = `[sub0-kp0, sub1-kp0, sub0-kp1, sub1-kp1]`
  - `op_sel_hi` selects which 16-bit half (K-phase)
  - `op_sel` selects byte within half (sub-group)
  - Hybrid uses `remap_phase(src, k_phase) = src >> (kphase*16)` to shift K-phase into low bytes + `op_sel` for sub-group
  - KPAIR skips `remap_phase` entirely: `op_sel_hi:[0,0,0]` for lo, `op_sel_hi:[1,1,0]` for hi
- **0 VALU in main loop**: no `remap_phase` v_lshrrev_b32 operations (vs hybrid's 16 per phase)
- **Upfront loading**: all 4 tiles (32 ds_reads) loaded to VGPRs before MFMA phase, same as hybrid's 4-tile upfront pattern (4261 TFLOPS known dead end re-confirmed: upfront loses ds_read:MFMA overlap)
- **128 MFMAs per K-iteration** (BK=128): 8 B-cols × 4 A-rows × 2 K-phases (lo+hi KPAIR) × 2 A-halves (A0, A1)

### Remaining 19% gap to Gluon (5234 TFLOPS) — root cause analysis
Gluon's main loop: 256 MFMAs + 64 ds_read + 32 buffer_load, 0 VALU, 8 barriers per iteration (BK=256).

| Factor | Colwise (BK=128) | Gluon (BK=256) | Impact |
| --- | --- | --- | --- |
| Loop iterations (K=8192) | 32 | 16 | ~7% (2× barrier/SALU overhead) |
| ds_read:MFMA interleaving | None (upfront) | 1:1 column-level pipeline | ~2% |
| LDS scale broadcast | No (16 global loads) | Yes (6 loads + LDS) | ~2-3% |
| B-column VGPR budget | 64 VGPRs (all 8 cols loaded) | 16 VGPRs (1 col + 1 prefetch) | Enables BK=256 |
| Total | ~81% of Gluon | 100% | ~19% |

### Next priority: BK=256 single-buffer + column-first B loading
Required to close the BK-related overhead gap:
1. **BK=256 single-buffer LDS**: 4 tiles × 64 rows × 256 bytes = 64 KB (fits 160 KB at occ=1)
2. **Per-column LDS loading**: load B sub-tile rows individually (2 ds_reads per 16×128 sub-row), cycle through 8 B columns. A tiles loaded upfront (64 ds_reads for BK=256).
3. **Column-level software pipeline**: while computing A×B_col_j (8 MFMAs), ds_read loads B_col_{j+1}. This requires `fp4_load_one_column()` helper that targets a specific sub-tile row in LDS.
4. **VGPR budget**: A0+A1 = 128 VGPRs, B current+next = 32 VGPRs, scales+addrs = ~50 → ~210 total (fits 256)
5. After all B-columns processed, issue tile prefetch to LDS (safe because all data in VGPRs).

## Debug Workflow
1. Read `analysis/fp8_gemm/mi350x/README.md` for current task state.
2. One change at a time. No mixed loader + schedule + waitcnt experiments.
3. Rebuild, smoke test, then formal benchmark if smoke is clean.
4. After throughput changes, inspect VGPRs, spills, occupancy, LDS in compile remarks.
5. Only keep changes that survive both correctness gate and performance gate.
6. For inline ASM MFMA blocks: always declare ALL accumulator AGPRs as `"+a"` in every split block.
7. For coord arguments to `make_pf_params`: use brace-init `{0,0,r,c}`, never `coord<>{...}`.
