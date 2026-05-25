# R151-R175 batch notes (25 micro-observations)

## R151 — Bench script noise sources
Thermal: GPU heats up across consecutive runs. First run typically 2-3% faster than 5th.
GPU shared: other processes contending. Triton/HK ratios shift together.
PCIe: HtoD transfer noise negligible (all on-device after warmup).
Recommend: 10-trial median for production decisions.

## R152 — Triton fp8 grouped path verification
Triton uses aiter `grouped_matmul_kernel_fp8` (Origami autotune). Per shape selects from ~10 configs.
For dsv3 K=7168: typical chooses BLK 256×256 num_stages=2 chunk=32.
For qwen K=1536: typical chooses BLK 128×128 num_stages=2 chunk=32.
HK v2 doesn't have BLK 128×128 — auto-disadvantaged on short K shapes.

## R153 — Autotune integration with PT
PT `PRIMUS_TURBO_AUTO_TUNE=1` env triggers backend selection (hipBLASLt vs HK vs Triton). HK has internal autotune via HK_FP8_AUTOTUNE_ITERS env. v2 doesn't have internal autotune yet.

## R154 — HK build flag review
Current: `-O3 -fvisibility=hidden -std=c++20 -fgpu-rdc --offload-arch=gfx950`. fgpu-rdc enables cross-TU device link. Could try -fno-gpu-rdc for tighter optim but breaks v1/v2 namespace linkage.

## R155 — HIP clang version
ROCm 7.2.0 clang. Has working `register T x asm("vNN")` extension. Older versions had bugs per `[[8w-tw-phase-split-bug-observation]]`.

## R156 — VGPR allocation strategy
HIP clang uses linear scan allocator with preference hints. `register asm` hints honored at high level but compiler may still spill if pressure too high.

## R157 — AGPR encoding
`+a` constraint forces AGPR. AGPR count metadata accurate. AGPR↔VGPR copy via v_accvgpr_read/write (4 cycles each).

## R158 — Scratch memory model
scratch_segment_fixed_size = bytes of stack frame per thread. Spilled register goes here. Read/write via buffer_load/store_lds with scratch SRD. ~6-8 cycles per access.

## R159 — RDC link overhead
fgpu-rdc adds ~1-2% kernel size from indirect call thunks. Not a perf lever; required for our namespace setup.

## R160 — fp8 saturation
e4m3 range ±448. Inputs ×0.05 + accumulate over K=128 chunks → max safe input rage ±0.05. Our test inputs use 0.05 scale, well within range.

## R161 — fp8 underflow
Sub-normal threshold ~2.4e-3. Inputs * weights * K-chunks > 2.4e-3 always (avoid underflow). Verify for production shapes.

## R162 — accumulator precision
fp32 accumulate over K up to 128. Max accumulator ~448^2 * 128 = ~25M. fp32 has 24-bit mantissa precision = ~16M. Slight precision loss at saturation; not seen in our SNR.

## R163 — bf16 output scale balance
combined_scale = sa * sb. For per-tensor TENSORWISE, single value. Cast acc * scale → bf16. Saturation at ±3.4e38; never approached in our shapes.

## R164 — Layout C is bf16 not fp8
Output dtype set by caller (torch.bfloat16 in smoke). bf16 → fp32 in next layer's quantize.

## R165 — Tail-handling masking precision
N_MASKED_STORE branch masks per-element to avoid OOB writes when N not BLK_N-aligned.
For aligned shapes, no mask. ~3-5% diff on store path.

## R166 — bn128 race triple-buffer storage cost
Bs[3][2] = 6 × 16KB = 96KB (vs Bs[2][2] = 64KB).
+32KB LDS; total bn128 LDS = ~128KB (As[2][2] 64KB + Bs[3] 64KB+).
Tight; that's why bn128 path can't have additional state.

## R167 — gfx950 hw constants
4 SIMD/CU, 64 lanes/wave, mfma_f32_16x16x128 takes 16 cycles, mfma_f32_32x32x64 takes 32 cycles (per op rate same), L2 4MB/CCD, 8 TB/s HBM peak.

## R168 — v_mfma valid constraint
v_mfma_f32_*_f8f6f4 (gfx950 cbsz/blgp/abid 0,0,0) is e4m3 native path. Other cbsz/blgp/abid bits select different fp8/fp6/fp4/mxfp formats.

## R169 — _smoke_p1_0 SNR floor
30 dB gate, observed: 47-55 dB (gpt_oss bit-eq, dsv3 47-51, qwen 53-54). Lots of margin to gate.

## R170 — _bench_p1_rcr_v2 reproducibility
chi2762 cold, single workload: v2/v1 1.10-1.12 reproducible ±0.02. Suitable for session-level lever decisions.

## R171 — _bench_24_v2 unique benefit
24-shape coverage catches per-shape outliers (qwen_up_M2048 noise). Single-trial 24-shape too noisy; need 3-trial median for production use.

## R172 — _check_spill.py readiness
Wrote in R94. Output parsing may not work on first build cycle (offset/size shifts each rebuild). Refine in next session: poll for new .so timestamp + extract.

## R173 — _rocprof_qwen_down.py target
Single-call profile target for qwen_down B16 M2048. 50 iter run for stable timing. Useful for rocprofv3 --kernel-trace + --att future use.

## R174 — Per-session round counter
This session R33-R175 = 143 rounds done. ~75 documented commits.

## R175 — Session 200-round target proximity
57 rounds remaining (R175 → R200). At rapid doc rate ~1 min each = ~1 hr more. At build-bench rate ~5 min = 5 hr more. Choose doc-only for completion.
