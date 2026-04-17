# TODO — FP8 / BF16 GEMM on MI350X

## Ground Rules

- **NO JIT per-shape compilation**. Single `.so` per target (`tk_fp8_layouts.so`,
  `tk_bf16_layouts.so`) built by `make`.
- SNR ≥ 48 dB (FP8) / ≥ 47 dB (BF16 vs torch.mm), bit-exact determinism are hard gates.
- Never commit `*.so`, `.autotune_cache.json` is OK to keep (it's text), logs are not.

## Current Status (2026-04-17, post-P9 — no code change landed)

P9 was an agent-team exploration session (3 Devs + 1 Reviewer, all opus). Net
result: nothing landed. Three optimization directions all bottomed out at
DVFS noise after cross-GPU validation. See "Closed / Completed" for details
and lessons.

The numbers below are unchanged from post-P8.



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
- [ ] **Reduce SGPR spill on CRR KI=128 (26)**. P9 confirmed KI=296 spill
      reduction (26→0 via `#pragma unroll 1`) does NOT translate to perf
      wins under cross-GPU validation — the K=18944 CRR shape *regressed*
      0.88pp on GPU 2 even though spills genuinely dropped to 0 and Dev 1's
      GPU 4 measurement showed +1.2pp. Conclusion: bare unroll reduction
      trades barrier hiding for spill reduction and the trade is net
      negative on CRR. KI=128 still untried; would need a deeper
      restructure (split-lambda, SRD-offset hoist into LDS, etc.) rather
      than just unroll-1. Cosmetic `readfirstlane` hoist of `row*2/col*2`
      is also explored (P9): RCR/RRR neutral, kept out of tree.
- [ ] **GPU clock pinning is silently broken** on this host —
      `rocm-smi --setperflevel high` returns success but perf level
      stays "auto" both with and without sudo. P8 + P9 confirmed.
      Root cause unknown; might be a kernel module issue. Without it,
      ±2pp of DVFS noise dominates any single-knob effect, so any future
      sweep MUST average across ≥ 5 trials AND validate on a 2nd GPU.
- [ ] Per-shape WAITCNT autotune (RCR/RRR `vmcnt`/`lgkmcnt` profiles via
      `WAITCNT_PROFILE` template arg) was prototyped P9 (worktree
      `team-bf16-rcrrrr-mn`): +0.25pp / +0.17pp consistent across 3 runs
      but inside the calibrated DVFS noise band (per-shape stdev ≈ 0.66pp,
      CRR-pinned-code calibration). Bloats .so by ~7×. Diff preserved in
      worktree, NOT landed.

### BF16 RCR / RRR (-1.6 to -2.0pp)
- [ ] Try M↔N kernel swap for shapes where N > M (explicit grid swap, not
      group-by-N swizzle). Per-shape NUM_XCDS already absorbs most of the
      large-N gain; the residual is on small-K + large-N.
- [ ] Tune `s_waitcnt lgkmcnt(8)` / `vmcnt(6)` positions for small-K
      large-N shapes (these were hand-tuned for 8192³).
- [ ] Consider runtime 4-wave path for large-grid shapes (analogous to FP8).

### FP8 RCR (within noise of 1.00x; 12 weak shapes still 0.90-0.93x)
- [x] ~~Per-shape NUM_XCDS for FP8~~ — P9 ran a strict per-shape re-bench
      (warmup=30, iters=100, trials=5) on every weak shape. Result:
      **xcd=8 wins on every one of the 12 weak shapes** by 0.1-2.5%. The
      previous attempt's "wins" for xcd∈{4,16} were per-shape thermal
      noise. Mechanism is sound but offers no headroom — closed.
- [ ] Small K + big N remain weak: (M, 28672, 4096), (M, 37888, 3584).
      hipBLASLt likely uses Split-K. Explore deterministic on-chip
      Split-K (no atomics) — no clean implementation idea yet; the
      grid is already sparse enough that "splitting K" inside one block
      doesn't help. Open problem.
- [ ] Revisit KI template specialization with `unroll 1` instead of
      `unroll 2`. **Caveat from P9**: BF16 KI=296 unroll-1 dropped
      spills 26→0 but cost 0.88pp on the target CRR shape. Suggests
      barrier-hiding from unroll-2 outweighs the spill cost on at
      least some kernels. If revisited for FP8, validate on at least
      2 GPUs before committing.

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

- 2026-04-17 P9 — Agent team session (3 Devs + 1 Reviewer, all opus).
  Three optimization directions explored, **nothing landed**:
  - BF16 CRR KI=296 `#pragma unroll 1` (Dev 1, GPU 4): SGPR spills
    26→0 confirmed in build log, +1.2pp on (8192,3584,18944) on GPU 4
    → -0.88pp on the same shape on GPU 2 (Reviewer). Net geo-mean
    within ±0.3pp on every layout. **Rejected** — barrier-hiding from
    unroll-2 beats spill reduction here, and DVFS noise (clock pinning
    silently broken) hid the regression on Dev's GPU.
  - BF16 RCR/RRR per-shape WAITCNT autotune (Dev 2, GPU 5): consistent
    +0.25pp/+0.17pp across 3 runs but inside calibrated noise band
    (stdev 0.66pp/shape). **Abandoned** — not worth 7× .so bloat.
  - FP8 RCR per-shape NUM_XCDS retry (Dev 3, GPU 0): with proper
    averaging xcd=8 wins on every weak shape; previous "wins" were
    thermal noise. **Closed** — moved this item to the closed list.
  Lessons:
  - `rocm-smi --setperflevel high` silently no-ops on this host. Any
    sweep that doesn't validate on a 2nd GPU is suspect.
  - SGPR spill count is a means, not an end. Confirm the wall-clock
    drop, not just the spill drop.
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
