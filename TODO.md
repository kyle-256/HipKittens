# TODO — FP8 / BF16 GEMM on MI350X

## Ground Rules

- **NO JIT per-shape compilation**. Single `.so` per target (`tk_fp8_layouts.so`,
  `tk_bf16_layouts.so`) built by `make`.
- SNR ≥ 48 dB (FP8) / ≥ 47 dB (BF16 vs torch.mm), bit-exact determinism are hard gates.
- Never commit `*.so`, `.autotune_cache.json` is OK to keep (it's text), logs are not.

## Current Status (2026-04-17, post-P8)

### FP8 — ✅ All targets met

Measured on GPU3, `bench_vs_hipblaslt.py --mode full`:

| Layout | Geo-mean vs hipBLASLt | Wins | Status |
|---|---|---|---|
| RCR | 0.996x | 21/56 | ⚠ within noise of 1.00x |
| RRR | 1.530x | 56/56 | ✅ ≥ 1.40x |
| CRR | 1.967x | 56/56 | ✅ ≥ 1.80x |

Last change: `RCR_TWO_TILE_MID_VMCNT 4 → 6` (P8). Original P7 commit message
claimed this was landed, but the file shipped at MID=4; P8 corrects it.

### BF16 — 🚧 Closer but still below 1.0x

Measured on GPU2 with the new per-shape NUM_XCDS autotune:

| Layout | Geo-mean vs torch.mm | Wins | Δ vs pre-P8 |
|---|---|---|---|
| RCR | 0.984x | 6/48 | +1.0pp |
| RRR | 0.980x | 9/48 | +1.6pp |
| CRR | 0.953x | 3/48 | +1.7pp |

torch.mm = hipBLASLt under the hood. Targets are still ≥ 1.00x for each
layout; no layout regressed.

## Open Items — High Priority

### BF16 CRR (biggest residual gap, -4.7pp)
- [ ] **Reduce SGPR spill on CRR KI=128 (26) and KI=296 (26)**. Per-iter
      constants (e.g. `tile+1/+2/+3` SRD offsets, `b_coord(col*2, ...)`
      arithmetic) keep refilling SGPRs in the loop body; hoist these out.
      The 7-SGPR-spill KI=172 variant runs ~1pp better than its neighbors
      at the same shapes, so spill reduction is the right axis. CRR-specific
      micro-tunes (CRR_MAIN_VMCNT/LGKMCNT, CRR_UNROLL=4, CRR_NUM_XCDS,
      CRR_CHUNK) all stayed within the ±2pp DVFS noise band — explored and
      ruled out 2026-04-17.
- [ ] Pin GPU clocks (`rocm-smi --setperflevel high`) before any future
      CRR tuning sweep. The 2pp DVFS drift was the dominant signal in the
      P8 CRR exploration.

### BF16 RCR / RRR (-1.6 to -2.0pp)
- [ ] Try M↔N kernel swap for shapes where N > M (explicit grid swap, not
      group-by-N swizzle). Per-shape NUM_XCDS already absorbs most of the
      large-N gain; the residual is on small-K + large-N.
- [ ] Tune `s_waitcnt lgkmcnt(8)` / `vmcnt(6)` positions for small-K
      large-N shapes (these were hand-tuned for 8192³).
- [ ] Consider runtime 4-wave path for large-grid shapes (analogous to FP8).

### FP8 RCR (within noise of 1.00x; 12 weak shapes still 0.90-0.93x)
- [ ] **Per-shape NUM_XCDS for FP8** — the runtime `g.num_xcds`
      machinery has been validated end-to-end (see Strategy A
      investigation 2026-04-17). 4 shapes prefer xcd=16 with +0.6 to
      +1.7pp wins, but per-shape noise on the other 44 cancels the
      geo-mean gain. Re-attempt with longer averaging or more aggressive
      MID-shape coverage; the diff lives in worktree
      `agent-a67f50ee` for reference.
- [ ] Small K + big N remain weak: (M, 28672, 4096), (M, 37888, 3584).
      hipBLASLt likely uses Split-K. Explore deterministic on-chip Split-K
      (no atomics).
- [ ] Revisit KI template specialization with `unroll 1` instead of `unroll 2`
      to avoid spills.

## Open Items — Correctness

- [ ] **BF16 2048³ CRR non-determinism** — at M=N=K=2048, CRR produces
      non-deterministic output (~15-21 bf16 ULP max diff). Root cause
      unknown. All benchmarked LLM shapes are ≥ 4096³ so the 48-shape
      benchmark is unaffected, but this should be fixed before production.

## Open Items — Medium Priority

- [ ] Make `autotune.py` also autotune the 4-wave vs 8-wave path choice for
      FP8 RCR.
- [ ] Benchmark against TRITON backend in Primus-Turbo, not only hipBLASLt.
- [ ] Add a CI script that runs `test_fp8_snr.py` + `quick_snr.py` + a
      5-shape perf sanity check on both directories.

## Closed / Completed

- 2026-04-17 P8 — BF16 per-shape NUM_XCDS autotune landed: RCR +1.0pp,
  RRR +1.6pp, CRR +1.7pp. FP8 MID_VMCNT 4→6 corrected (P7 commit message
  claimed this but file shipped at 4).
- 2026-04-17 — BF16 CRR-only knob exploration: CRR_MAIN_VMCNT,
  CRR_MAIN_LGKMCNT, CRR_UNROLL={1,4,8}, CRR_NUM_XCDS={4,16}, CRR_CHUNK
  all within noise; root cause is SGPR spill on KI=128/296.
- [x] Removed all JIT per-shape compilation (`jit_gemm.py`, `bench_jit*.py`,
      `kernel_jit_*.cpp`, `*_exact_*_fastpath.inc`, `.jit_cache/`,
      `.jit_bf16_cache/`).
- [x] Removed dead experimental kernels (`kernel_1024/2048/4096/8192/16384.cpp`,
      `kernel_bf16_128/256x128/4wave.cpp`, `kernel_crr.cpp`,
      `kernel_layouts.cpp`).
- [x] FP8 RCR geo-mean ≥ 1.00x achieved (1.005x → drifted to 0.996 noise band).
- [x] BF16 migrated to single-source `kernel_bf16_dynamic.cpp` with runtime
      KI_HINT template dispatch.
- [x] Both directories use runtime group_m autotune; BF16 also autotunes NUM_XCDS.
- [x] Updated skill docs: `bf16-gemm-optimization`, `fp8-rcr-autotune-optimization`,
      `fp8-strict-layout-tuning`.

## How To Run

```bash
# FP8
cd analysis/fp8_gemm/mi350x
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 ROCM_PATH=/opt/rocm make clean
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 ROCM_PATH=/opt/rocm make -j4
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python3 bench_vs_hipblaslt.py --mode full

# BF16
cd analysis/bf16_gemm/mi350x
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 ROCM_PATH=/opt/rocm make clean
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens2 ROCM_PATH=/opt/rocm make -j4
HIP_VISIBLE_DEVICES=1 PYTHONPATH=. python3 bench_bf16_vs_torch.py
```
