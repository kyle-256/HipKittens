# R31 OPT B — PERSISTENT_XCD / STATIC_XCD_REMAP Scout for L6

**Date**: 2026-04-18
**Shape**: L6 = 4096×32768×128256
**Incumbent**: `ts_lgk2_v12_memc_btw_all`, 5353.9 TFLOPS, 92.6% of comp 5781
**Bench params** (per `.claude/rules/benchmark-rules.md`): warmup=200, iters=500, trim=0.10
**GPU**: 7 (idle); GPU 6 used for early V1/V3 fault confirmation

## Verdict — DEAD END (4/4 variants fail)

| Variant | Build | Launch | TFLOPS (1-rep) | Ratio vs Inc | 3-rep mean | Verdict |
|---|---|---|---|---|---|---|
| V1 PERSISTENT_XCD=1 | PASS (256VGPR/35spill/144scratch) | **GPU FAULT** | — | — | — | **CRASH** |
| V2 STATIC_XCD_REMAP=1 | PASS (212VGPR/0spill/0scratch) | OK | 5268.9 | 98.41% | **5240.6 (-1.80%)** | **LOSE -1.80%** |
| V3 PERSISTENT_XCD=1 + PERSISTENT_BATCH=4 | PASS (256VGPR/35spill/144scratch) | **GPU FAULT** | — | — | — | **CRASH** |
| V4 STATIC_XCD_REMAP=1 + GROUP_SIZE_M=8 | PASS (212VGPR/0spill/0scratch) | OK warmup → mid-bench OOB | 0 (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION) | — | — | **CRASH** |

## Step 1 — Macro audit

Macros (defaults all OFF) at `kernel_mxfp4_gluon_cpp.cpp:160-181`:

| Macro | Default | Description |
|---|---|---|
| `PERSISTENT_XCD` | 0 | Atomic tile-counter persistent grid; gridDim=PERSISTENT_GRID; XCD-aware ordering preserved by `raw_bid % NUM_XCDS`. |
| `STATIC_XCD_REMAP` | 0 | Atomic-free: each XCD owns N-strip width `bpc/NUM_XCDS`. Falls back to default when `bpc % NUM_XCDS != 0`. |
| `PERSISTENT_GRID` | 608 | (8 XCDs × 38 CUs × 2 WG/CU). |
| `PERSISTENT_BATCH` | 1 | Tiles claimed per atomicAdd. |
| `GROUP_SIZE_M` | 4 | M-stripe width for L2 B-tile reuse. |

**Launcher contract** (`dispatch_gluon_cpp` lines 3219-3245):
- Already has Fix A (checked `hipGetSymbolAddress` + synchronous `hipMemset`).
- Already has Fix C (caps grid by `total_tiles`).
- The kernel body has Fix B (`__syncthreads()` before next claim, line 3210).
- **No kernel-signature change required**; macros are fully self-contained on the host side and inside the kernel body. Pure cppflag opt-in.

L6 dims: `bpc = N_DIM/BLK = 32768/256 = 128`, `bpr = M_DIM/BLK = 4096/256 = 16`, `total_tiles = 2048`.
- `bpc % NUM_XCDS = 0` ✓ (STATIC_XCD_REMAP eligible).
- `total_tiles (2048) > PERSISTENT_GRID (608)` ✓ (persistent grid would loop ~3.4 tiles/WG).

## Step 2 — Build sweep

All 4 variants compiled cleanly in 5.3 s (parallel). Resource usage:

| Variant | TotalSGPRs | VGPRs | AGPRs | Scratch (B/lane) | SGPRs Spill | VGPRs Spill | Occupancy |
|---|---|---|---|---|---|---|---|
| V1 PERSISTENT_XCD | 106 | 256 | 256 | 144 | 8 | 35 | 1 |
| V2 STATIC_XCD_REMAP | 93 | 212 | 256 | 0 | 0 | 0 | 1 |
| V3 PERSISTENT_XCD + B4 | 106 | 256 | 256 | 144 | 8 | 35 | 1 |
| V4 STATIC_XCD_REMAP + gm8 | 93 | 212 | 256 | 0 | 0 | 0 | 1 |

**Red flag pre-bench**: PERSISTENT_XCD inflates VGPRs from 212 → 256, induces 35 VGPR spills + 144 B/lane scratch. The atomic-counter loop and per-iter pre-fill state cannot be CSE'd around the `while(true)`, so the compiler conservatively widens live ranges.

## Step 3 — Single-rep bench + correctness probe

**Correctness sanity** (cross-check vs incumbent at L6 dims, single isolated launch on GPU 6):

- V1_pxcd: `Memory access fault by GPU node-8` immediately on first launch (independent of warmup). Same fault confirmed in 2 separate processes.
- V3_pxcd_b4: Same memory access fault.
- V2_sremap: launched OK, returned values (NaN due to bf16 overflow with random scales over K=128256, same NaN pattern as incumbent — not a correctness signal).
- V4_sremap_gm8: launched OK in single-shot probe; failed mid-bench during the 700-iter loop (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`), so the OOB is statistical or inputs-sensitive.

PERSISTENT_XCD's pre-existing fault was previously documented in `r24a_pxcd_debug.md` (R24, 2026-04-18). The kernel already carries Fixes A/B/C from that round, but the GPU-fault signature persists — there is at least one further bug not addressed by R24's three fixes. Investigating further is out of scope for an L6-only scout.

**Bench (warmup=200, iters=500, trim=0.10, GPU 7)**:

| Variant | wall (s) | TFLOPS (1-rep) | Ratio vs Inc 5354 |
|---|---|---|---|
| incumbent | 8.4 | 5365.5 | 100.22% |
| V2 STATIC_XCD_REMAP | 8.6 | 5268.9 | 98.41% |
| V4 STATIC_XCD_REMAP + gm8 | 201.9 (timeout-ish) | CRASH | — |

## Step 4 — 3-rep verify on V2 (winner threshold not met, but confirming regression)

GPU 7, back-to-back:

| Variant | rep1 | rep2 | rep3 | mean | min | max |
|---|---|---|---|---|---|---|
| incumbent | 5343.0 | 5333.4 | 5333.5 | **5336.6** | 5333.4 | 5343.0 |
| V2 STATIC_XCD_REMAP | 5245.8 | 5241.6 | 5234.5 | **5240.6** | 5234.5 | 5245.8 |

**Δ = -96 TFLOPS = -1.80%**. Tight per-rep clusters (spread <12 TFLOPS each), clean separation between distributions. The regression is real, not noise.

## Step 5 — Recommendation

**Add NOTHING to bench_all_42.py for L6.**

| Variant | Recommendation |
|---|---|
| PERSISTENT_XCD (V1, V3) | Do not pursue. Pre-existing GPU fault on launch survives R24's Fix A/B/C; needs deeper repair (out of L6-only scope). Also incurs +44 VGPRs, 35 spills, 144 B/lane scratch — even if fixed it would face significant headwind. |
| STATIC_XCD_REMAP V2 | Reject. Clean build (212 VGPR, 0 spill), launches successfully, **but loses 1.80% to incumbent**. The atomic-free static remap reorganises the bid→(br,bc) mapping but the existing default GROUP_M=4 swizzle (with the existing tall-XCDs remap) is already L2-optimal for L6 (B-tile is L2-resident after 2 K-iters per the persistent-XCD remap memory note). The static remap gives up the existing 4-row stripe and trades to an N-strip walk that increases B-tile working set per XCD. |
| STATIC_XCD_REMAP + GROUP_SIZE_M=8 (V4) | Reject. Compiles clean but mid-bench `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. Suspected bid-arithmetic edge case when `bpr % GROUP_M != 0` interacts with the static remap (bpr=16, GROUP_M=8, so this *should* be safe — but evidently isn't). Not worth chasing on L6 alone given V2's regression. |

## Unexpected findings

1. **Two of four PERSISTENT_XCD launches GPU-fault despite the documented R24A fixes being present in source.** Code at `kernel_mxfp4_gluon_cpp.cpp:3222-3240` includes all three fixes (`hipGetSymbolAddress` checked, `hipMemset` synchronous, grid capped). The fault therefore happens *inside* the kernel — not in the host setup. Likely candidates: the persistent loop's per-iter `prefill_swizzled_offsets` setup interacting badly with non-zero `bid` on the first iter when WGs claim tiles out of XCD-aware order; or scratch-spill interaction with the `while(true)` body. R24A's Fix B (`__syncthreads()` before next claim) does not address pre-claim state.
2. **V4 (GROUP_SIZE_M=8 over STATIC_XCD_REMAP) crashes mid-bench, not on first launch.** The single-shot probe succeeded; the OOB only manifests during the 200+500 iter loop. Suggests an iter-ordering or counter-state hazard, not a static dispatch bug.
3. **V2 (STATIC_XCD_REMAP, GROUP_SIZE_M=4) loses 1.80% on L6.** This is the cleanest possible "swap dispatch only, keep all other tuning" experiment. The result confirms that the existing tall-XCDs remap + GROUP_M=4 swizzle already exploits L2 B-tile reuse for L6's M=4096 / N=32768 / K=128256 geometry. Re-bucketing into N-strips reduces the per-XCD B-tile reuse window, costing more bandwidth than it saves on duplicate B-tile fetches.

## Conclusion

A4 axis (PERSISTENT_XCD / STATIC_XCD_REMAP) is **dead** for L6. No variant beats incumbent. Two variants GPU-fault. **No commit. No bench_all_42.py change.** L6's 92.6% ceiling will need to be approached from a different axis (e.g., the OptA UNROLL_K sweep or scale-load reorg).

## Artifacts

- `build_round31_optB_xcd.py` — build driver (4 variants in parallel)
- `bench_R31_optB.py` — single-rep / N-rep bench driver with optional SNR
- `R31_OPT_B_BUILD_RESULTS.json` — build resource usage
- `R31_OPT_B_BENCH_SREMAP.json` — 1-rep bench (incumbent + V2 + V4 crash)
- `R31_OPT_B_VERIFY_V2_3rep.json` — 3-rep verify (incumbent vs V2)
- `R31_OPT_B_SNR_V2.json` — V2 SNR probe (corner-decoded SNR uninformative due to bf16 overflow on random scales over K=128256)
- `R31_OPT_B_BUILD.log`, `R31_OPT_B_BENCH_SREMAP.log`, `R31_OPT_B_VERIFY_V2_3rep.log`
- `_R31B_correctness_check.py` (used to confirm V1/V3 GPU faults; unsuitable for SNR due to NaN-overflow on this shape)
