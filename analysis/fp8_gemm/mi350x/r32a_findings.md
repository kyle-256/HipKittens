# R32 Dev A Findings — Stage A2 rect-V2 CRR fastpath: NO SHIP

## TL;DR

**NO SHIP.** Stage A2c failed correctness gate at the 70B KV shape (4096×1024×8192):
- Pass rate: **33.53%** (gate: ≥99%)
- Determinism: **FAIL** (gate: 3/3 bit-equal)
- Root cause: the `load_col_from_v2_st_half` helper's `K_HALF` template parameter
  splits the **N-direction** (LDS row dim), not the K-direction. For rect mode
  with HB_N=64, K_HALF=1 requests N-rows 64..127 which is out-of-bounds for the
  64-row rect LDS tile. The R31 Stage A1 stub that duplicates K_HALF=0 data
  into the K_HALF=1 register slot remains the only valid scaffolding — the
  numerics are wrong by construction.

The kernel **builds clean (148 VGPRs, 104,448 B LDS, 2 waves/SIMD occupancy)**
and **runs without GPU fault** (R31 Stage A1a/A1b held). All R32 kernel/host
edits have been reverted to the R31 baseline. Stage A2d (paired BABA bench) is
gated on A2c PASS and is therefore **SKIPPED** per the task brief.

## Stage A2 deliverables status

| Stage | Description | Status |
|-------|-------------|--------|
| A2a   | `preshuffle_v2_b_rect` host function | **DONE** (in `r32a_bench.py`) |
| A2b   | Rewrite `load_col_from_v2_st_rect` to fetch both K_HALFs | **NO-FIX** (architectural blocker) |
| A2c   | SNR ≥ 48 dB + det 3/3 PASS @ 70B KV | **FAIL** — pass_rate 33.53%, det False |
| A2d   | Paired BABA bench rect vs square @ 70B KV | **SKIPPED** (gated on A2c) |

## Architectural root cause

The square fastpath uses `ST_v2 = st_fp8e4m3<HB=128, BK=128, st_16x128_v2_s>`
where the LDS tile has `rows = HB = N-direction` and `cols = BK = K-direction`.
This is documented at `kernel_mxfp8_layouts.cpp:438`:

```cpp
using ST_row = st_fp8e4m3<HB, BK, st_16x128_s>;    // 128×128, M/N rows × K cols
```

The `load_col_from_v2_st_half<RT, K_HALF>` helper computes:

```cpp
const int k_row = row_off + K_HALF * 64;          // indexes ROWS = N-direction
const uint32_t stidx = k_row >> 4;
const uint32_t base_k = tile_base + (stidx << 11) + (stidx << 7) + ((k_row & 15) << 7);
```

The variable name `k_row` is misleading: it indexes the **LDS row** dimension,
which is the **N-direction** of the matrix tile. With `K_HALF=1`, `k_row` ∈
[64, 127], which addresses N-rows 64..127 of the LDS tile.

For square `ST_v2` (HB=128 N-rows), this is in-bounds. For rect `ST_v2_rect`
with HB_N=64 (only 64 N-rows in the tile), `K_HALF=1` reads bytes from offset
~17152 into a tile that is only 8192 bytes — out-of-bounds, into adjacent LDS
tiles or scratch.

The R31 Stage A1 stub side-steps this by duplicating `K_HALF=0` data into the
`K_HALF=1` register slot (data[4..7] = data[0..3]). This guarantees no GPU
fault but produces wrong numerics: each MMA iteration uses the same K=0..63
data twice instead of K=0..63 then K=64..127.

## Two valid paths forward (out of R32 budget)

### Path 1: Square LDS tile + halve the kernel's N work (cheaper)

Use `ST_v2_rect = st_fp8e4m3<HB=128, BK=128, ...>` (same as square ST_v2) so
the existing helper works unchanged. Restructure the rect kernel to use **only
ONE N-half per BLK_N=128 ctile**:

- Drop `b1`, `cB`, `cD` — only `b0`, `cA`, `cC` remain (1 N-half, 2 M-halves)
- Change `s = bc * 2`/`bc * 2 + 1` → `s = bc` (one global_load_b per ctile)
- Halve the per-block work; expect 2x more blocks for same N (N/BLK_N vs N/BLK)

**Cost**: ~3-4h to restructure the steady-state loop and prologue/epilogue
unrolls. The B-side scale fetch reverts to square layout (slab_bytes_b=64*kp,
pack=2 with both packs filled) — the host preshuffle stays unchanged from
square.

**Risk**: register pressure may not drop as much as the "true rect" (148 →
~140 VGPRs vs square's 234), since the B register tile is still RBN_RECT=16
but the number of MMAs per loop is halved. Net: VGPR savings come from cB/cD
elimination (32 fewer FP32 accumulators × 8 packed_per_thread = ~64 VGPRs).
Estimated 110-130 VGPRs final.

### Path 2: Rewrite helper to fetch K-direction differently (more invasive)

Redesign `load_col_from_v2_st_half` so that `K_HALF` indexes the K-direction
(LDS col dim) rather than the N-direction (LDS row dim). This requires
re-deriving the lane sharding (currently 64 lanes × 16 cols = 1024 elements
per ds_read group) so that lanes cover K-direction of the LDS tile and the
inner j-loop covers N-direction.

**Cost**: ~6-8h plus extensive correctness validation against the existing
square fastpath (which depends on the current sharding shape).

**Risk**: high. The existing helper is used in 4 hot kernels (CRR/RRR/RCR
exact + 4-wave). Changing its sharding semantics breaks all of them; would
need a `_rect` variant.

**Recommended**: Path 1.

## Bench data

### Stage A2c failure run (4096×1024×8192, GPU0, 30s preheat, sclk 2308→2388 MHz)

```
[r32a] As.shape=torch.Size([32, 32768]) Bs.shape=torch.Size([32, 16384])
CORRECTNESS M=4096 N=1024 K=8192 snr_db=inf pass_rate_pct=33.53
C[0, :8] = [-0.94921875, -0.52734375, -0.87890625, 1.2265625, 0.5703125,
            -0.050048828125, -0.059326171875, -0.125]
C_ref[0, :8] = [-0.6951422691345215, -0.3869476318359375, -1.008007287979126,
                -0.8026106357574463, -1.0931396484375, 0.6643214225769043,
                0.12281346321105957, 1.4651551246643066]
DETERMINISM ok=False
RUN 0 avg_ms=0.0834 tflops=824.39
RUN 1 avg_ms=0.0815 tflops=843.66
RUN 2 avg_ms=0.0811 tflops=847.54
SUMMARY median=843.66 mean=838.53 stdev=12.40 runs=3
```

Notes:
- `snr_db=inf` is misleading — there are non-NaN values that match magnitudes
  but signs/values diverge from reference. The 33.5% pass rate (within 3.0 abs
  tol or 10% rel tol) is the meaningful number.
- `DETERMINISM ok=False` is concerning: same inputs give different outputs
  across runs. Hypothesis: the K_HALF=1 stub reads from out-of-bounds LDS
  region whose contents depend on prior tile state (Bs[2][2] double-buffer
  lifecycle). This is consistent with the architectural diagnosis above.
- TFLOPS data (~840 TFLOPS) is **not directly comparable** to square (1660+
  TFLOPS at this shape) because the rect kernel does HALF the per-block work
  with 2x blocks; per-TFLOP arithmetic is correct but the kernel is
  numerically wrong.

### Build resources (rect, R31 baseline)

```
TotalSGPRs: 53     VGPRs: 148    AGPRs: 0
Occupancy [waves/SIMD]: 2
LDS Size [bytes/block]: 104448
md5: f19e6eb331f5b4798f916f0b48dbc7a2  tk_mxfp8_layouts.cpython-310-x86_64-linux-gnu.so
```

vs square (R30 baseline reference):
```
TotalSGPRs: 81     VGPRs: 234
Occupancy [waves/SIMD]: 2
LDS Size [bytes/block]: 139264
md5: 8432a2a9de6ca1246e69a5c98790c154  (default build, byte-identical to R31 cell1)
```

The rect kernel is 36% smaller in VGPRs (148 vs 234) and 25% smaller in LDS
(104K vs 139K) but at the same occupancy (2 waves/SIMD). Without correct
numerics, this is irrelevant.

## R32 work artifacts

- **`r32a_bench.py`** — bench harness with new `preshuffle_v2_b_rect` host
  function (Stage A2a, complete). Uses 30s preheat, sclk monitoring with
  PHYS_GPU env, BABA-pattern-ready, n_runs configurable, GPU0-discount
  un-applied (raw values).
- **`r32a_orchestrate.sh`** — 3-cell orchestration: build square baseline
  byte-identity check, build rect, run Stage A2c correctness test.
- **`r32a_cell{1,2,3}_*.txt`** — orchestrate output logs.
- **`r32a_build_cell{1,2}.log`** — full build logs.

The `crr_mxfp8_exact_8wave_rect_fastpath.inc` and `kernel_mxfp8_layouts.cpp`
are **unchanged from the R31 baseline** — all my Stage A2b kernel edits were
reverted after the architectural diagnosis showed they could not produce
correct numerics without restructuring the kernel (Path 1 above).

## Recommendation for R33

Take **Path 1**: restructure rect kernel to use square LDS tile + drop
b1/cB/cD. This is achievable in ~3-4h and unlocks the rect VGPR/LDS savings
with correct numerics. The host preshuffle reverts to the existing square
`preshuffle_v2_b` (no new host code needed).

The R32 `r32a_bench.py`'s `preshuffle_v2_b_rect` function is **not needed for
Path 1** — it was designed for the (now-abandoned) square-equivalent slab
layout with rect ctile boundary. Save it as reference for Path 2 if anyone
revisits.

## Author identity

Author: Claude (Opus 4) acting as R32 Dev A
Co-Authored-By: Claude Opus 4 <noreply@anthropic.com>
