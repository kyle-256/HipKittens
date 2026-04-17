---
name: mxfp8-layout-tuning
description: Tune HipKittens MXFP8 microscaling GEMM kernels on gfx950/MI350X toward FP8 per-tensor parity. Use when working on analysis/fp8_gemm/mi350x MXFP8 RCR/RRR/CRR performance, preshuffle-quant scale packing, scale-pack scheduling, buffer_load SRD, KPAIR_LOOP, mfma_scale, SNR, determinism, or agent-team orchestration.
---
# MXFP8 Microscaling Layout Tuning

## When To Use
- Debugging or optimizing `analysis/fp8_gemm/mi350x` **MXFP8** GEMM kernels.
- Keywords: `mxfp8`, `microscaling`, `block-scale`, `preshuffle-quant`, `mfma_scale`, `scale-pack`, `E8M0`, `KPAIR_LOOP`, `buffer_load SRD`, `sched_group_barrier` (MXFP8 context only).
- For per-tensor FP8 work → `fp8-per-tensor-layout-tuning`.
- MXFP4 kernels have been removed from this branch; don't try to resurrect them here.

## First Read
1. `TODO.md` — live progress, current baseline, active agent assignments.
2. `agent_prompt.md` — team protocol, role definitions, runbook.
3. `analysis/fp8_gemm/mi350x/README.md` — longer-form task handoff.
4. This file's "Durable Findings" and "Known Dead Ends".

## Mission
Close the MXFP8 RCR vs FP8 per-tensor gap. Current status (see `TODO.md` for live numbers):
- **FP8 RCR baseline**: 3070.93 TFLOPS (SNR 49.61 dB)
- **MXFP8 8-wave KPAIR+SRD+SCALE_PIPE+HOIST_HI(opsel)**: 2925.64 TFLOPS (SNR 49.60 dB) — reviewer GPU7 formal
- **Gap**: +145.29 TFLOPS / +4.73% (A/B on GPU7 is +17.80 TFLOPS vs KPAIR+SRD+SCALE_PIPE alone)

## Hard Rules
- Branch: `feat/mxfp8-only`. Don't commit directly to `main`.
- Formal shape: `8192x8192x8192`. Smoke shape: `256x256x256`.
- `preshuffle-quant` is allowed **only for scale tensors**. `preshuffle-ab` is forbidden.
- Both `A_scale` and `B_scale` are preshuffled in the current PQ path.
- Acceptance gates (all four mandatory):
  1. Correctness pass (pass rate 100%)
  2. `SNR > 48 dB`
  3. Deterministic across 3+ runs
  4. Formal TFLOPS must survive the `warmup=100, iters=200` protocol on `HIP_VISIBLE_DEVICES=7`
- No regressions allowed on FP8 per-tensor RCR baseline.
- No host-side workarounds: no `.t().contiguous()`, no Python-side padding.

## Commit-Time Workflow (MANDATORY)
Every commit that changes performance characteristics **must** update the following in the same commit, in this order:
1. **`TODO.md`** — move completed items, add new dead-ends, refresh baseline table.
2. **`agent_prompt.md`** — update the baseline row, any new rule learned.
3. **This SKILL** — only if a durable finding or known dead-end was discovered (not for every perf delta).
4. **The code change** itself.

Commit message template:
```
MXFP8 RCR <what changed>: <TFLOPS> TFLOPS (<+delta%>)

SNR: XX.XX dB, determinism: PASS (N runs)
VGPR: X, AGPR: Y, spills: Z, LDS: W KB
```

Do NOT commit:
- `*.s`, `*.so`, `*.hipfb`, `*.hipi`, `*.bc`, `*.co` (generated)
- `*_layout_results_*.json`, `gpucore.*`, `pmc_*/` (per-run outputs)
- `.venv*/`, `__pycache__/`, `*.bak*`
- `build_probe_*/`, `work_probe_*/`, etc. (experiment scratch)

`.gitignore` already covers these; don't weaken it.

## Primary Files
| Path | Role |
| --- | --- |
| `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp` | Main MXFP8 kernel (8-wave + 4-wave paths) |
| `analysis/fp8_gemm/mi350x/rcr_mxfp8_4wave_fastpath.inc` | 4-wave RCR fastpath |
| `analysis/fp8_gemm/mi350x/rrr_mxfp8_4wave_fastpath.inc` | 4-wave RRR |
| `analysis/fp8_gemm/mi350x/crr_mxfp8_4wave_fastpath.inc` | 4-wave CRR |
| `analysis/fp8_gemm/mi350x/rrr_mxfp8_exact_8wave_fastpath.inc` | 8-wave RRR exact PQ path |
| `analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc` | 8-wave CRR exact PQ path |
| `analysis/fp8_gemm/mi350x/kernel_mxfp8_4wave_rewrite.cpp` | 4-wave pure `__builtin` MFMA variant (for rewriter pipeline) |
| `analysis/fp8_gemm/mi350x/test_mxfp8_python.py` | Benchmark + correctness + determinism harness |
| `analysis/fp8_gemm/mi350x/rewrite_mxfp8.py` | Python `.s` rewriter (AGPR-focused, currently only touches the 4-wave rewrite variant) |
| `analysis/fp8_gemm/mi350x/build_rewrite.sh` | C++ → `.s` → Python rewrite → `.so` pipeline |
| `analysis/fp8_gemm/mi350x/Makefile` | hipcc entry point |
| `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` | FP8 per-tensor reference (do NOT edit) |
| `analysis/fp8_gemm/mi350x/test_python.py` | FP8 regression harness |
| `include/ops/warp/memory/util/util.cuh` | `make_srsrc`, `llvm_amdgcn_raw_buffer_load_b32` |

## Build Commands

### Current best MXFP8 (8-wave, KPAIR_LOOP + SGPR SRD + scale pipeline + HOIST_HI opsel)
```bash
cd analysis/fp8_gemm/mi350x
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) ROCM_PATH=/opt/rocm \
  CPPFLAGS='-DM_DIM=8192 -DN_DIM=8192 -DK_DIM=8192 \
    -DMXFP8_RCR_EXACT_8WAVE_FAST_ENABLE=1 \
    -DMXFP8_RCR_EXACT_PQ_KPAIR_LOOP_ENABLE=1 \
    -DMXFP8_RCR_EXACT_PQ_PIPELINE_SCALE_ENABLE=1 \
    -DMXFP8_RCR_EXACT_PQ_HOIST_HI_ENABLE=1' \
  make -B TARGET=tk_mxfp8_layouts SRC=kernel_mxfp8_layouts.cpp
```

### FP8 per-tensor reference
```bash
cd analysis/fp8_gemm/mi350x
THUNDERKITTENS_ROOT=$(git rev-parse --show-toplevel) ROCM_PATH=/opt/rocm \
  make -B TARGET=tk_fp8_layouts SRC=kernel_fp8_layouts.cpp
```

### Smoke test
```bash
# MXFP8
HIP_VISIBLE_DEVICES=7 MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr \
  MXFP8_WARMUP=5 MXFP8_ITERS=10 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
  python3 test_mxfp8_python.py 256 256 256
```

### Formal benchmark
```bash
# MXFP8
HIP_VISIBLE_DEVICES=7 MXFP8_PRESHUFFLE_QUANT=1 MXFP8_LAYOUTS=rcr \
  MXFP8_WARMUP=100 MXFP8_ITERS=200 MXFP8_CHECK=1 MXFP8_DETERMINISM_RUNS=3 \
  python3 test_mxfp8_python.py 8192 8192 8192

# FP8 regression check
HIP_VISIBLE_DEVICES=7 FP8_LAYOUTS=rcr FP8_WARMUP=100 FP8_ITERS=200 FP8_CHECK=1 FP8_DETERMINISM_RUNS=3 \
  python3 test_python.py 8192 8192 8192
```

> The test harnesses use **per-iteration** `torch.cuda.synchronize()` + `output.zero_()`. All baseline numbers in this skill / `TODO.md` are measured under that exact protocol. Do NOT switch to batch timing unless you also restate every baseline in the new protocol.

## Agent Team Protocol
The MXFP8 → FP8 parity work is organized as an agent team (see `agent_prompt.md` for the long-form prompts):

- **Decision maker (main agent)** — analyzes asm diffs, maintains baseline, dispatches dev agents, runs reviewer, decides what to commit.
- **Dev A** — 4-wave upgrade path. Has ~14–44 VGPR headroom the 8-wave path lacks; the only path that can house a scale pipeline cross-iteration.
- **Dev B** — 8-wave micro-optimization. Target: hide/eliminate the 6 per-iter `v_lshrrev_b32` scale remaps.
- ~~**Dev C** — Python `.s` rewriter~~ **Closed**: A dedicated 8-wave rewriter draft was written and measured (GPU2 A/B +5.94 TFLOPS / +0.18%). Because HOIST_HI already eliminates the 6 `v_lshr` at source, the rewriter has no meaningful remaining target on this kernel. Do not reopen this role unless you can first demonstrate a new structural target that only a post-`.s` pass can reach.
- **Reviewer** — smoke → formal → FP8 regression → SNR → determinism. Any gate fail → reject.

Parallel agents SHOULD use distinct `HIP_VISIBLE_DEVICES` (Dev A → 0, Dev B → 1, Dev C → 2, Reviewer/formal → 7) to avoid GPU collisions.

## Current Kernel Resource Picture (8-wave PQ=1, HOIST_HI opsel enabled, from compile remarks)
- VGPR: **254** (down from 256 — the HOIST_HI opsel path frees 2 regs by eliminating hi_packs arrays)
- AGPR: 0 — accumulators already in VGPRs
- Spills: 0, Scratch: 0
- LDS: 131072 bytes (double-buffer A+B tiles)
- Occupancy: 2 waves/SIMD

## Inner-Loop ASM Diff (8-wave, per kpair)
| metric | FP8 | MXFP8 pre-HOIST_HI | MXFP8 w/ HOIST_HI (opsel) | Δ vs FP8 |
| --- | ---: | ---: | ---: | ---: |
| total lines | 403 | 431 | **422** | +19 |
| MFMAs | 64 | 64 | 64 | 0 |
| `v_mfma_scale` | 0 | 64 | 64 | +64 |
| `buffer_load` | 16 | 22 | 22 | +6 (scale loads) |
| `ds_read` | 48 | 48 | 48 | 0 |
| `s_waitcnt` | 10 | 12 | 12 | +2 |
| `s_barrier` | 16 | 16 | 16 | 0 |
| `v_lshrrev_b32` | 0 | 6 | **0** | 0 (op_sel replaces) |
| `sched_barrier` | 4 | 4 | 4 | 0 |

HOIST_HI eliminates the 6 scale-remap `v_lshr` by using the MFMA builtin's `op_sel_hi` byte-select: `K_PHASE=0` reads low 16-bits of the 32-bit scale pack (`op_sel_hi:[0,0,0]`), `K_PHASE=1` reads high 16-bits (`op_sel_hi:[1,1,0]`). Net perf gain: reviewer-verified +17.80 TFLOPS (+0.61%, GPU7 A/B, formal SNR+det PASS) on the 8192³ benchmark.

## Durable Findings
- **KPAIR_LOOP** (2× manual unroll of the k-pair loop) is the biggest optimization on 8-wave (~+13% historically).
- **buffer_load with SGPR SRD** + `readfirstlane` + `soffset` reduces spills from 8 → 3 and adds +1%.
- **HOIST_HI (op_sel byte-select)**: adds a reviewer-verified +17.80 TFLOPS (+0.61%, GPU7 A/B 2914.87 → 2932.67, formal SNR+det gate 2925.64). Uses the `v_mfma_scale_f32_16x16x128_f8f6f4` builtin's `op_sel` + `op_sel_hi` fields with a compile-time `K_PHASE` template parameter to let the hardware pick the correct 2 scale bytes out of the 32-bit scale pack — eliminates all 6 `v_lshrrev_b32` from the main inner loop without spills (VGPR 256→254, Spills 0, Occupancy 2). **K_PHASE must be compile-time constant** — we use a C++20 templated lambda (`[&]<int K_PHASE>(int k)`) with manually unrolled `template operator()<0>` / `<1>` calls. **Tail iterations must stay on `rcr_mma_scaled_from_packs_exact`** (runtime k_phase); routing tail through opsel generated +64 MFMA code and 31 spills on the first attempt.
- **Scale software pipelining on 8-wave is infeasible**: 256 VGPR hard limit + any cross-iteration live range for scale_packs → 150–200 spills.
- **LDS caching of scales on 8-wave is infeasible**: tile double-buffers already consume 128 KB LDS.
- **4-wave has headroom**: 14–44 VGPRs free, 128 KB LDS double-buffer possible, occupancy 1 (or 2 if VGPRs allow). Scale pipelining works here where it fails on 8-wave.
- **Pure `__builtin` MFMAs match inline ASM** on the 4-wave kernel (2853 TFLOPS either way). Builtin eliminates ACC16 macro complexity, AGPR reuse bugs, and `#pragma unroll` correctness issues.
- **SALU dual-issue with MFMA on gfx950**: measured ZERO benefit from scheduling SALU into MFMA streams (A/B tested on MXFP4 Gluon; Phase 2 of that rewriter moved 30 SALU, delivered 0%). Treat SALU as free on paper but don't expect wins from rescheduling it.
- **hipcc AGPR reuse bug**: split inline-ASM blocks each declaring only "their" AGPRs → compiler reuses AGPRs across blocks. Fix: declare ALL accumulator AGPRs as `"+a"` in every split block.
- **hipcc `#pragma unroll 2` is required** for correctness when using split `asm volatile` blocks; `unroll 1` breaks VGPR liveness tracking.
- Batch-timing vs per-iter timing: the FP8 skill quotes 3335 TFLOPS (batch), this skill quotes 3070.93 (per-iter sync, as measured in `test_mxfp8_python.py`'s harness). Do NOT mix protocols.

## Known Dead Ends
- `4de0a032` buffer_load scale path: fast but invalid at large K.
- Shared/LDS scale cache: correctness issues, never validated.
- `phase1 op_sel_hi` **inline-ASM** variants: semantics correct but long-run perf regressed. (The later **__builtin-based** opsel path under HOIST_HI works — see Durable Findings — but ASM-level attempts prior to it did not.)
- Scalar phase-pack caching micro-optimizations: didn't beat the current schedule.
- Inline-ASM async prefetch (8-wave): hit 2929–3052 TFLOPS but failed correctness.
- L2 prefetch with `buffer_load_dword` discard-VGPR: 190 spills.
- `sched_group_barrier` in place of `sched_barrier(0)`: −0.8%.
- MXFP8 8-wave scale pipeline (any form): 256 VGPR limit kills it.
- AGPR-redistribution rewriter (`rewrite_mxfp8.py`) on the 8-wave PQ kernel: no-op because that kernel has 0 AGPRs.
- Python `.s` rewriter scheduling changes (AGPR shuffle, cross-barrier MFMA split, ds_read early issue): all 0% on MXFP8. The compiler is already near-optimal within each asm block; the remaining gap is structural.
- 8-wave-dedicated `.s` rewriter (`rewrite_mxfp8_8wave.py`) that interleaves the 6 scale-remap `v_lshr` into preceding MFMA shadows: measured GPU2 A/B +5.94 TFLOPS (+0.18%). HOIST_HI opsel already eliminates the 6 `v_lshr` entirely, so this pass has nothing left to transform on the current best kernel. Draft not kept in tree; do not resurrect unless you find a fresh structural target.
- MXFP8 4-wave upgraded with KPAIR_LOOP + SGPR SRD + scale pipeline (Dev A, 2nd round): GPU0 A/B 4-wave bare 2878 → 4-wave KPAIR+PIPE 2900 (+22, +0.76%). Still ~90 TFLOPS below 8-wave KPAIR+SRD+SCALE_PIPE+HOIST_HI on the same GPU. The 4-wave inline-ASM MFMA path burns 256 AGPR + 256 VGPR → occupancy 1, versus 8-wave's 0 AGPR + 256 VGPR → occupancy 2. Cannot beat 8-wave until 4-wave is moved off inline-ASM onto `__builtin_amdgcn_mfma_scale_*` (which frees the AGPRs) AND a structural advantage beyond 8-wave shows up. Not worth chasing until that precondition is solved.

## Debug Workflow
1. Smoke test before every formal run. Formal ≥ 4 min; smoke ~2 s.
2. One change at a time. No mixed loader + schedule + waitcnt experiments.
3. After any throughput-affecting change, inspect VGPRs / spills / occupancy / LDS in compile remarks.
4. Only keep changes that pass both correctness and performance gates.
5. For inline ASM MFMA blocks: ALL accumulator AGPRs must be `"+a"` in EVERY split block.
6. For `make_pf_params` coord args: brace-init `{0,0,r,c}`, never `coord<>{...}`.
7. Profiling: `rocprofv3 -i <pmc_file>` (v1 doesn't support gfx950). Key counters: `MfmaUtil`, `SQ_INSTS_MFMA`, `SQ_WAVE_CYCLES`, `SQ_WAIT_ANY`, `SQ_WAIT_INST_LDS`, `SQ_BUSY_CYCLES`.
8. Kernel symbols to look for in device `.s`:
   - `_Z22rcr_exact_8wave_kernel*` — FP8 RCR 8-wave
   - `_Z29rcr_exact_8wave_scaled_kernelILb1EEv*` — MXFP8 8-wave PQ=1
   - `_Z29rcr_exact_8wave_scaled_kernelILb0EEv*` — MXFP8 8-wave PQ=0
   - Main inner loop label: `.LBB14_4:` (for current 8-wave PQ=1 build)
