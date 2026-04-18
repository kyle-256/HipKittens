# R24A — PERSISTENT_XCD coverage bug analysis

## Code paths

- Persistent dispatcher loop: `analysis/fp8_gemm/mi350x/kernel_mxfp4_gluon_cpp.cpp:2086-2137`
  (also: counter declaration line 2052; loop close line 3029; host reset lines 3038-3046)
- Build flags: `bench/build_round23_optB.py` — `_pxcd_b1` = `-DPERSISTENT_XCD=1 -DPERSISTENT_BATCH=1`,
  `_pxcd_b4` = `-DPERSISTENT_XCD=1 -DPERSISTENT_BATCH=4`. **PERSISTENT_GRID stays at default 608**
  (8 XCDs * 38 CUs * 2 WG/CU). MI355X has 304 CUs.

## Symptoms recap (bench_round23_optB.log)

| shape | tiles total | _pxcd_b1 cov | _pxcd_b1 TFLOPS | _pxcd_b4 cov |
|-------|-------------|---|---|---|
| DLA1 (M=4096) | 16*128 = 2048 | rc=-6 (SIGABRT) | — | rc=-6 |
| DLA2 (M=128256) | 501*128 = 64128 | 6.4% (~4104 tiles) | 60012 (≈13× peak) | 6.4% |
| DLA7 (M=28672) | 112*128 = 14336 | 28.6% (~4100 tiles) | 12834 (≈3× peak) | 28.6% |

The **TFLOPS×coverage** product matches realistic peak (DLA2: 60012 × 0.064 ≈ 3840
TFLOPS, near baseline 4500). This confirms the kernel is correctly processing the
tiles it does claim — just not enough of them.

The constant ~4100-tile ceiling on both DLA2 and DLA7 (despite 4× different tile
counts) is the smoking gun. **Each WG averages ~6.7 tiles before its loop terminates**
(4100 / 608 ≈ 6.74), independent of total problem size.

## Root-cause hypotheses (ranked by likelihood)

### H1 (most likely): `g_persistent_tile_counter` is never reset between launches

Lines 3038-3046:
```cpp
unsigned int* counter_dev = nullptr;
hipGetSymbolAddress((void**)&counter_dev, HIP_SYMBOL(g_persistent_tile_counter));
hipMemsetAsync(counter_dev, 0, sizeof(unsigned int), 0);
```
- The return code of `hipGetSymbolAddress` is **never checked**. For a `__device__`
  variable that lives in a Python-loaded `.so` (via `importlib.util`), HIP may not
  register the symbol with the runtime → call returns an error and `counter_dev`
  stays NULL → `hipMemsetAsync(NULL, ...)` is a no-op (or worse).
- The static initializer `__device__ unsigned int g_persistent_tile_counter = 0;`
  fires **once at module load**, not per launch. So iter 1 starts at 0 and processes
  all tiles correctly; iter 2..N see counter == total_blocks + 608 (from prior run's
  drain) and every WG immediately fails `raw_bid >= total_blocks` and breaks.
- Across 200 warmup + 500 timed iters, only iter 1 actually writes C → coverage
  should be ~100%, not 6.4%. **So H1 alone doesn't fully explain coverage**, but
  it's almost certainly broken.

### H2 (explains the 6.4% number): `__launch_bounds__(_NUM_THREADS, 1)` + late WG launch race

`__launch_bounds__(_NUM_THREADS, 1)` permits min 1 WG/EU. Actual occupancy is
register-limited; for this kernel only ~1-2 WGs/CU → ~304-608 WGs co-resident.
- WGs in the **first wave** (~304-608) drain the counter rapidly.
- By the time the GPU scheduler tries to launch any **queued** WGs, the counter
  is already past total_blocks, so they `break` immediately on first claim.
- DLA2 (64128 tiles) — first wave alone can drain all tiles before queue starts.
  But coverage 6.4% × 64128 ≈ 4104 tiles ≈ 6.74 tiles/WG-in-wave1 (608 WGs).
- This implies that even WGs in the first wave are exiting early — there's
  another bug.

### H3 (cause of DLA1 SIGABRT): tile-count-vs-grid-size mismatch on small problems

DLA1 has only 2048 tiles but PERSISTENT_GRID=608. With ~3.4 tiles/WG, fine in theory,
but combined with H1/H2 the over-launched grid likely hits an out-of-bounds shared-mem
write or assert in correctness mode. (Need to inspect the exact assert line — possibly
`bid >= total_blocks` after the XCD remap re-mapping.)

### H4 (subtle race): missing barrier before re-entering the while loop

There is **no `__syncthreads()` between the kernel body's tail (stores at lines
3010-3026) and the loop top atomicAdd (line 2109)**. Thread 0 may issue the next
atomicAdd while other threads are still flushing buffer_stores. The next iteration's
prefill / prefetch then overlaps the previous tile's stores, corrupting LDS.
Result: store-side write races, possibly C corruption rather than coverage loss.
This may explain DLA1 SIGABRT (correctness-mode assert) but probably not the
DLA2/DLA7 6.4%/28.6% numbers.

## Proposed diff (safe — no behavior change when PERSISTENT_XCD==0)

### Fix A — robust counter reset (addresses H1)

```cpp
// kernel_mxfp4_gluon_cpp.cpp, replace lines 3038-3046
#if PERSISTENT_XCD
    unsigned int* counter_dev = nullptr;
    hipError_t err = hipGetSymbolAddress((void**)&counter_dev,
                                         HIP_SYMBOL(g_persistent_tile_counter));
    if (err != hipSuccess || counter_dev == nullptr) {
        // Hard failure - the persistent counter cannot be reset.
        fprintf(stderr, "PERSISTENT_XCD: hipGetSymbolAddress failed (%d) - aborting\n",
                (int)err);
        std::abort();
    }
    // Synchronous memset to guarantee the reset is visible before kernel launch.
    hipError_t merr = hipMemset(counter_dev, 0, sizeof(unsigned int));
    if (merr != hipSuccess) {
        fprintf(stderr, "PERSISTENT_XCD: hipMemset failed (%d)\n", (int)merr);
        std::abort();
    }
    const dim3 grid(PERSISTENT_GRID);
#else
    const dim3 grid((m / BLK) * (n / BLK));
#endif
```

(Switched `hipMemsetAsync` → `hipMemset` for ordering safety; the cost is negligible
versus a multi-ms GEMM, and it eliminates any stream-ordering ambiguity.)

### Fix B — barrier before next-tile claim (addresses H4)

```cpp
// kernel_mxfp4_gluon_cpp.cpp, immediately before line 3029 ( "} // end while(true)" )
#if PERSISTENT_XCD
    __syncthreads();   // ensure all stores for current tile are visible / threads
                       // have all left the store loop before next atomicAdd.
#endif
    } // end while(true) persistent loop
```

### Fix C — cap PERSISTENT_GRID by problem size (addresses H3)

In `dispatch_gluon_cpp` after computing `total_blocks`:

```cpp
#if PERSISTENT_XCD
    const int total_tiles = (m / BLK) * (n / BLK);
    const int grid_x = total_tiles < PERSISTENT_GRID ? total_tiles : PERSISTENT_GRID;
    const dim3 grid(grid_x);
#else
    ...
#endif
```

This prevents over-launching for small problems (DLA1) where 608 > 2048-tile budget
isn't a problem mathematically but creates many WGs that immediately exit, which
combined with H4 may surface a race.

## Safety analysis

All three fixes are gated by `#if PERSISTENT_XCD`. When PERSISTENT_XCD==0
(the default `_pxcd_baseline` build), the host code path is unchanged. The
`__syncthreads()` in Fix B sits inside `#if PERSISTENT_XCD` so the static-dispatch
build sees a no-op brace.

## Recommended test order after applying fixes

1. Apply **Fix A only** → smoke DLA2 _pxcd_b1. If coverage ~100%, H1 is confirmed.
2. If still <100%, apply **Fix B**. If DLA1 still SIGABRTs, apply **Fix C**.
3. Compare TFLOPS vs `_pxcd_baseline` — if >>5% drop, persistent overhead exceeds
   any benefit; PXCD is dead-end. If within 5%, profile next.
