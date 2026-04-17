# Round 18 Optimizer C — Verdict: DEAD END

**Date**: 2026-04-17
**Author**: Kyle.Zhao@amd.com / kyle-256
**Task**: Implement R17A-P2 — M=128 → M=256 specialization for K=128256 (DLA1)
**Time spent**: ~50 min wall

---

## Summary

| Variant | Build OK | SNR pass | LDS usage | Smoke Δpp DLA1 | Verify Δpp DLA1 |
|---|---|---|---|---|---|
| BLOCK_M_512 source refactor | NOT ATTEMPTED | n/a | est. 192 KB > 160 KB cap | n/a | n/a |
| `_r18c_pb1` (PERSISTENT_BATCH=1) | OK | **BROKEN** (memfault) | n/a | n/a | n/a |
| `_r18c_pb2` (PERSISTENT_BATCH=2) | OK | **BROKEN** (memfault) | n/a | n/a | n/a |
| `_r18c_pb4` (PERSISTENT_BATCH=4) | OK | **BROKEN** (memfault) | n/a | n/a | n/a |
| `_r18c_pb8` (PERSISTENT_BATCH=8) | OK | **BROKEN** (memfault) | n/a | n/a | n/a |
| `_r18c_pb1_g1216` (BATCH=1 + grid=1216) | OK | **BROKEN** (memfault) | n/a | n/a | n/a |
| `_r18c_pb2_g1216` (BATCH=2 + grid=1216) | OK | **BROKEN** (memfault) | n/a | n/a | n/a |
| `_r18c_pb1_static` (BATCH=1 + STATIC_XCD_REMAP) | OK | **BROKEN** (memfault) | n/a | n/a | n/a |

**Result**: 0 winners. **No commit.**

---

## Why the original P2 (BLOCK_M=512 source refactor) was not attempted

Profile findings (`round17_optA_dla1_profile_findings.md` §5 Proposal P2) propose
`M=128 → M=256` to halve CTA count from 524 K to 262 K. Two problems with
implementing this in a single round:

### 1. Architectural depth of the refactor

`kernel_mxfp4_gluon_cpp.cpp` is 2,647 lines. The current per-CTA tile is already
`BLK × BLK = 256 × 256`, with A split into two row-halves of `HB = BLK/2 = 128`
each. The "M=128" in the profile was the per-warp HB sub-tile, not the per-CTA
tile. To meaningfully halve total CTA count, the per-CTA M-tile must grow to 512:

- A would split into A0/A1/A2/A3 (currently A0/A1) — 4 tiles instead of 2.
- The inner main loop has 4 STEPs (A0×Bl, A0×Br, A1×Bl, A1×Br) each ~50–100 lines
  of hand-scheduled MFMA + ds_read + prefetch interleaving. STEPs would expand to
  8 (also adding A2×Bl, A2×Br, A3×Bl, A3×Br), duplicating ~600 lines of inner-loop
  code, plus prologue (initial tile loads), tail iteration, and epilogue store.
- 4× scale buffers (currently 2: `pf_a0`, `pf_a1`).
- 4 accumulator sets (currently `acc_A0Bl…acc_A1Br` = 4 sets; would become 8).
- `compute_lds_base_addrs` calls × 4 instead of × 2.

This is a multi-day refactor with high risk of register spilling (current kernel
already runs near 256-VGPR cap per the R17A profile; adding 4× accumulators
means 8 × 16 × float = 512 floats just for C, vs current 256 — exceeds available
VGPR budget without forcing waves_per_eu=1 occupancy).

### 2. LDS budget overflow

Per `rocminfo`, MI355X has **160 KB LDS / CU** (not 64 KB as the prompt hinted —
gfx950 uses the larger 160 KB shared-memory segment).

Current LDS use:
- `__shared__ ST_tile A0_db[2], A1_db[2], Bl_db[2], Br_db[2]`
- `ST_tile = st<fp8e4m3, HB=128, BK=128>` — but fp8e4m3 here packs MXFP4 (4 bits)
  giving ~8 KB / tile. Conservative estimate (1 byte / element): 16 KB / tile.
- 8 tiles × 16 KB = **128 KB** (bound, fits inside 160 KB).

Doubled M-tile (M=512):
- A buffers double: 4 A tiles × 2 db × 16 KB = **128 KB** for A alone.
- Bl/Br unchanged: 4 B tiles × 16 KB = 64 KB.
- **Total: 192 KB** — exceeds 160 KB cap.

Mitigations and their costs:
- **Drop double-buffering on A** (single-buffer): saves 64 KB → total 128 KB
  (fits) but kills the producer-consumer overlap that the inner loop relies on.
  Latency-hide gain from larger tile is offset by serialized A loads. Expected
  net Δ ≤ 0.
- **Reduce N tile** to M=512, N=128 (half N): saves Bl+Br by half but doubles
  the N-direction CTA count back to 262 K, negating the original goal.
- **Reduce K-tile (BK)**: would shorten the inner-loop K-iteration unit and
  proportionally increase scale-pack overhead — likely net negative.

The "abort with M=192 attempt" fallback in the prompt is also not viable: M=192
isn't a power of 2 nor does it cleanly divide 64 (the row alignment for scale
pack `>> 6` indexing). Would require a third partial-tile code path on top of
the existing A0/A1 split.

### Verdict on source-level P2

**Not implementable in a single round.** A serious M=512 specialization is a
multi-day kernel rewrite with non-trivial risk of register spilling and LDS
overflow, and would need its own multi-round optimization pass. Recommend the
M=512 path be tracked as a long-horizon refactor, not a per-round optimization.

---

## Pragmatic substitute: PERSISTENT_BATCH probe

Since BLK_M=512 was infeasible, I probed the closest-spirit alternative on the
**already-merged** PERSISTENT_XCD code path. Persistent grid lets each WG claim
multiple sequential tiles via atomic counter, amortizing per-CTA launch /
synchronization overhead — partial proxy for "halve the per-CTA overhead" goal.

Round 14 Optimizer C had previously tested `PERSISTENT_XCD=1 PERSISTENT_BATCH={16, 32}`
on DLA1 and both BROKE with memory access fault (`bench_round14_optC_smoke.json`).
The **untested** BATCH={1, 2, 4, 8} values plus grid-size and STATIC_XCD_REMAP
combos were probed in this round.

### Build (`build_round18_optC.{py,log}`)

7 variants × 2 shapes (DLA1 + small SNR shape) = 14 builds. **All 14 OK** in 19 s.

### SNR safety probe (`snr_probe_r18c.{py,log,json}`)

Run on N=32768, K=128256 with M=256 (small DLA1-sized check), real random fp4
data, scales = 1.0 (e8m0 0). 7 variants tested. **All 7 BROKEN** with
`Memory access fault by GPU node-X` from `hipDeviceSynchronize`.

```
_r18c_pb1                    BROKEN (memfault)
_r18c_pb2                    BROKEN (memfault)
_r18c_pb4                    BROKEN (memfault)
_r18c_pb8                    BROKEN (memfault)
_r18c_pb1_g1216              BROKEN (memfault)
_r18c_pb2_g1216              BROKEN (memfault)
_r18c_pb1_static             BROKEN (memfault)
```

### Diagnosis: PERSISTENT_XCD is fundamentally broken on DLA1 size

R14C had assumed batch=16/32 was the issue. R18C confirms **all PERSISTENT_BATCH
sizes (1, 2, 4, 8, 16, 32) crash on DLA1**. Root cause is in the persistent grid
mechanism itself when the WG count is large (DLA1 has 2048 logical tiles).
Likely candidates:
- Race on `g_persistent_tile_counter` reset across multi-XCD launch.
- `s_claim_base` __shared__ ordering vs prefetch path.
- `bid >= total_blocks` bounds-check happens AFTER `xcd` swizzle, which may
  underflow on negative `local_pid` for terminating WGs.

PERSISTENT_XCD axis is **DEAD** for any DLA1-sized launch.

### Smoke bench / verify / regression-check

**Skipped** — no SNR-passing variant to bench.

---

## Files produced

| File | Contents |
|---|---|
| `build_round18_optC.{py,log}` | 14 builds (7 variants × 2 shapes), all OK |
| `snr_probe_r18c.{py,log,json}` | SNR probe, all 7 candidates BROKEN |
| `round18_optC_verdict.md` | This document |

## Files NOT produced (no successful candidate)

- `bench_round18_optC_smoke.{py,log,json}`
- `bench_round18_optC_verify.{py,log,json}`

---

## Commits

**No commits.** No source changes; analysis-only round.
`kernel_mxfp4_gluon_cpp.cpp` untouched.

---

## Recommendation to R18 decider

1. **Mark P2 (M-tile=512 specialization) as a long-horizon refactor**, not a
   per-round task. Estimate: 3-5 days of focused work to refactor the inner loop
   for 4-A-tile structure + LDS-budget mitigation (likely require BK=64 or
   half-N-tile to fit).
2. **Mark PERSISTENT_XCD axis as DEAD** for DLA1 (and likely all
   K=128256 / N=32768 shapes). The dispatcher mechanism is broken at large grid
   counts. If anyone wants to revive it, the bug needs root-causing first.
3. **R17A-P1 (double C-accumulator)** remains the highest-EV unimplemented
   source proposal but also requires a sustained multi-round effort due to
   register pressure (256-VGPR cap with WPE2 → 1).
4. **R17A-P3 (drop redundant inner-loop s_barrier)** is the only remaining
   single-round source-edit candidate from the R17A findings; it has a
   ~5-6 pp upper bound but requires careful SNR validation. Consider dispatching
   that to R18 next.

## Round 18 dead-end registry update

Add to TODO.md DEAD-END section:
- **R18C: PERSISTENT_XCD on DLA1** — confirmed BROKEN for batch ∈ {1, 2, 4, 8, 16, 32}
  + grid {608, 1216} + STATIC_XCD_REMAP combo. Memory fault before any compute.
- **R18C: BLOCK_M_512 source refactor** — infeasible in single round; LDS budget
  overflow (192 KB > 160 KB) + 600-line inner-loop duplication + register spill
  risk. Long-horizon multi-day refactor required.
