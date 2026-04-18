# R29 Dev D — V2-CRR LDS Bank Conflict Audit

**Date:** 2026-04-18
**Branch:** r29-d (base feat/mxfp8-only @ 145ff766)
**Scope:** Audit all `ds_read*` and `ds_write*` calls in the V2-CRR exact 8-wave fastpath to determine whether LDS bank conflicts are a residual cause of V2-CRR's under-performance (8192³ ratio 0.9266; 70B Gate 0.8579 post-R28; 70B KV 0.8646; 70B Down 0.8459).

**Verdict:** **No bank conflicts in the V2 column-load (B-tile) or V2a column-load (A-tile) paths.** The existing swizzle `(nc ^ sw_k)` already distributes lane addresses across all 32 banks evenly within every 16-lane dispatch slot. No fix possible at the swizzle level. The LDS bank-conflict lever is **CLOSED** for V2-CRR.

---

## Hardware model (gfx950 / MI350x)

- **LDS banks:** 32 banks × 4 bytes per bank = 128 B per cycle peak.
- **Wavefront issue:** A `ds_read` instruction on a 64-lane wave issues over 4 cycles, 16 lanes per cycle (lanes 0-15, 16-31, 32-47, 48-63). Bank conflicts are evaluated **per cycle** (i.e. within each 16-lane group), not across the whole wave.
- **`ds_read_b64`:** each lane reads 8 bytes = 2 contiguous banks per lane. 16 lanes × 2 banks = 32 banks, exactly the bank count → conflict-free iff each lane's pair `(bank, bank+1)` is unique within the cycle.
- **`ds_read_b64_tr_b8`:** byte-transposed variant. Same per-lane address semantics for bank-conflict purposes (each lane provides one 8-byte address; lane-byte transpose happens after the bank read).

Bank function: `bank(addr) = (addr / 4) % 32`.

---

## Inventory: all `ds_*` instructions in the V2-CRR fastpath

Scanned `analysis/fp8_gemm/mi350x/kernel_mxfp8_layouts.cpp` and the exact-8-wave fastpath inc. The V2-CRR steady-state main loop emits the following LDS traffic:

| Source location | Instruction | Count per CTA-iteration | Notes |
|---|---|---|---|
| `kernel_mxfp8_layouts.cpp:440-441` (`load_col_from_v2_st_half`) | `ds_read_b64_tr_b8` × 2 | per j ∈ [0,RT::width), per K_HALF ∈ {0,1} | B-tile column read, used for `b0` and `b1` in the fastpath |
| `kernel_mxfp8_layouts.cpp:480-481` (`load_col_from_v2a_st_half`) | `ds_read_b64_tr_b8` × 2 | per j ∈ [0,RT::width), per K_HALF ∈ {0,1} | A-tile column read (RT::width = RBM/16 = 4) |
| Tile fill (G::load → kittens shared store) | `ds_write_b*` | once per buffer fill | Hidden inside `kittens::load`; uses ST swizzle defined in `include/types/shared/st_shape.cuh:267` |

The V2-CRR steady inner loop also issues `buffer_load_dword` for raw scale packs (`b32/b64/b128`), but these are VMEM, not LDS, and out of scope for bank-conflict analysis.

---

## Address derivation (B-tile column load, V2)

From `kernel_mxfp8_layouts.cpp:418-447`:

```cpp
const int laneid  = kittens::laneid();
const int row_off = ((laneid % 16) / 2) + ((laneid / 16) * 16);
const int col_off = (laneid % 2) * 8;
const int k_row   = row_off + K_HALF * 64;

const uint32_t stidx  = k_row >> 4;                           // subtile index (0..7 for HB=128)
const uint32_t base_k = tile_base
                      + (stidx << 11)                          // 2048 B/subtile data
                      + (stidx << 7)                           // 128 B subtile padding
                      + ((k_row & 15) << 7);                   // 128 B/row inside subtile
const uint32_t sw_k   = (k_row & 7) << 4;                      // swizzle 4-bit nibble at bits 4-7

for (j = 0; j < RT::width; j++) {
    const uint32_t nc   = col_start + j*16 + col_off;
    const uint32_t addr = base_k + (nc ^ sw_k);
    ds_read_b64_tr_b8 dst[idx],   addr offset:0
    ds_read_b64_tr_b8 dst[idx+2], addr offset:1024
}
```

`row_off` map (laneid → row_off):
- lanes 0-1 → 0, lanes 2-3 → 1, ..., lanes 14-15 → 7
- lanes 16-17 → 16, lanes 18-19 → 17, ..., lanes 30-31 → 23
- lanes 32-33 → 32, ..., lanes 46-47 → 39
- lanes 48-49 → 48, ..., lanes 62-63 → 55

So **lanes 0-15 cover rows 0-7**, lanes 16-31 cover rows 16-23, etc. Each 16-lane dispatch slot touches 8 unique `k_row` values × 2 lane-pair `col_off` values.

---

## Per-cycle bank computation (cycle 0 example, K_HALF=0, j=0, col_start=0)

| laneid | row_off | k_row | col_off | nc | sw_k | nc^sw_k | addr (− tile_base) | addr/4 | bank=(addr/4)%32 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
|  0 |  0 | 0 | 0 | 0 |  0 |  0 |    0 |   0 |  0 |
|  1 |  0 | 0 | 8 | 8 |  0 |  8 |    8 |   2 |  2 |
|  2 |  1 | 1 | 0 | 0 | 16 | 16 |  144 |  36 |  4 |
|  3 |  1 | 1 | 8 | 8 | 16 | 24 |  152 |  38 |  6 |
|  4 |  2 | 2 | 0 | 0 | 32 | 32 |  288 |  72 |  8 |
|  5 |  2 | 2 | 8 | 8 | 32 | 40 |  296 |  74 | 10 |
|  6 |  3 | 3 | 0 | 0 | 48 | 48 |  432 | 108 | 12 |
|  7 |  3 | 3 | 8 | 8 | 48 | 56 |  440 | 110 | 14 |
|  8 |  4 | 4 | 0 | 0 | 64 | 64 |  576 | 144 | 16 |
|  9 |  4 | 4 | 8 | 8 | 64 | 72 |  584 | 146 | 18 |
| 10 |  5 | 5 | 0 | 0 | 80 | 80 |  720 | 180 | 20 |
| 11 |  5 | 5 | 8 | 8 | 80 | 88 |  728 | 182 | 22 |
| 12 |  6 | 6 | 0 | 0 | 96 | 96 |  864 | 216 | 24 |
| 13 |  6 | 6 | 8 | 8 | 96 |104 |  872 | 218 | 26 |
| 14 |  7 | 7 | 0 | 0 |112 |112 | 1008 | 252 | 28 |
| 15 |  7 | 7 | 8 | 8 |112 |120 | 1016 | 254 | 30 |

`ds_read_b64` reads 8 bytes per lane → each lane occupies 2 banks: lane *L* uses banks `{bank(L), bank(L)+1}`.

Cycle-0 bank usage: `{0,1}, {2,3}, {4,5}, {6,7}, {8,9}, {10,11}, {12,13}, {14,15}, {16,17}, {18,19}, {20,21}, {22,23}, {24,25}, {26,27}, {28,29}, {30,31}` — **all 32 banks, each used exactly once. Zero conflicts.**

The same pattern holds for cycles 1, 2, 3 (lanes 16-31, 32-47, 48-63) because each 16-lane group's row span (rows 16-23, 32-39, 48-55) produces the identical `(k_row & 15) << 7` + `(k_row & 7) << 4` interaction (for `K_HALF=0` the lanes always satisfy `k_row & 7 == k_row & 15` since `row_off & 15 == row_off & 7` within each group of 8). The base offset per cycle changes by `2048 + 128 = 2176 B = 544 dwords`, which is `544 % 32 = 0` banks — **same bank pattern, no new conflicts**.

### Second instruction (`offset:1024`)

`offset:1024` ⇒ +256 dwords ⇒ +0 banks. Same bank pattern as the first instruction. **Zero conflicts.**

### Other `j` values

For `j ≥ 1`, `nc` increments by 16 (0x10). Since `sw_k ∈ {0, 16, 32, 48, 64, 80, 96, 112}` and `nc & 0xF0` may be set, `nc ^ sw_k` flips a single bit at position 4 of `sw_k`. The address shifts by ±16 B = ±4 banks. Per-cycle bank pattern is uniformly translated by ±4 banks → still all 32 banks unique → **zero conflicts**.

### `K_HALF=1`

`k_row += 64` ⇒ `stidx += 4` ⇒ `base_k += 4*(2048+128) = 8704 B = 2176 dwords = 0 banks (mod 32)`. Inside `K_HALF=1`, `(k_row & 7)` and `(k_row & 15)` cycle the same way as `K_HALF=0`. **Identical bank pattern. Zero conflicts.**

---

## A-tile column load (V2a, `load_col_from_v2a_st_half`)

`kernel_mxfp8_layouts.cpp:458-488` is byte-for-byte identical address arithmetic to the V2 helper above. Only `RT::width` differs (4 instead of 2). The four `j` iterations shift the bank pattern by 0/±4/±8/±12 banks — all permutations remain conflict-free by the same argument. **Zero conflicts.**

---

## ST_v2 / ST_v2a swizzle on the **store** side

`include/types/shared/st_shape.cuh:267` defines:

```cpp
const int swizzle = ((offset >> 7) & 7) << 4;
const int swizzled_offset = offset ^ swizzle;
```

This maps row-address bits 7-9 (`(r*128) >> 7 & 7 = r & 7`) into the col bits 4-6, exactly mirroring `(k_row & 7) << 4` in the load helper. The store path is the inverse of the load path and uses the same swizzle, so any per-cycle distribution argument applies identically. The default `kittens::load → ds_write_b128` issued by tile fill is also conflict-free under this swizzle (the same 16-lane-cycle argument applies, with each lane writing 16 B = 4 banks: `4 lanes × 4 banks × 2 sub-cycles = 32 banks` per cycle, distributed uniformly).

No store-side conflict either.

---

## What this means for V2-CRR's under-performance

The V2-CRR LDS path is **already optimally swizzled**. The 8.7% gap on the 8192³ shape (ratio 0.9266) and the 14-19% gaps on 70B Gate / Down are *not* caused by LDS bank contention. Profiling counters in prior cycles (R22-B, R23, R26-A) consistently showed `SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS < 1%` for the V2-CRR steady kernel, which empirically corroborates this static analysis (cross-reference: R22-B 5x bench notes — V2 had `SQ_INSTS_VMEM -15.38%` but no LDS-conflict regression vs V1).

**Lever closure recommendation:** mark "LDS bank conflict reduction" as a **DEAD-END** for V2-CRR alongside `s_setprio` (R28 Dev A), `split-K-along-K` (R28 Dev D corrected), and scale-LDS double-buffer (V2 has no scale LDS by design — VMEM→VGPR direct).

The V2-CRR shortfall must therefore originate from one of:

1. **VMEM bandwidth / SRD setup overhead** (V2 uses per-wave-tile SRDs; large-N shapes pay more SRD-setup cycles per CTA-iteration). Partially explored by R27 Dev A's cachepolicy auto-select (R28 SHIP +1.94%).
2. **Steady-state issue-rate stall** from scale b128/b64 buffer-load latency. Open lever: 2-deep prefetch queue for scales (cf. R26 Dev B exploration).
3. **VGPR / occupancy pressure** due to 8-wave layout × scale registers. Cross-cycle observation: VGPR usage at the per-CTA limit; reducing register footprint could enable 2 waves/CU instead of 1.
4. **Explicit `buffer_load_dword_lds` (VMEM→LDS direct)** — gfx950 supports this, V2-CRR currently goes VMEM→VGPR→LDS. Big restructure (the original alternative-2 lever in R29 Dev D's brief).

Of these, (3) and (4) are the highest-EV unexplored directions for a future R30 cycle. (1) and (2) are partially mined.

---

## Rejected sub-experiments (not pursued)

- **Pad ST_v2 row by 4 B** (e.g. cols=132 instead of 128). Would shift bank pattern by 1 bank/row, but since the current pattern is *already* conflict-free, padding wastes LDS without benefit and would increase per-CTA LDS footprint above the current 139264 B (already a known SPI launch-allocator concern — see TODO.md:999).
- **Re-mask `sw_k` to `(k_row & 15) << 4`** (extra bit). Would pivot the bank stripe by 16 banks/row instead of 8 banks/row, but the *per-cycle* row-set is `{8 contiguous rows}` so all row patterns ∈ `{0..7}` are already exercised — no improvement possible.
- **`__builtin_amdgcn_sched_barrier(0xff)`**: the fastpath already uses `sched_barrier(0)` (block all reordering) extensively (`kernel_mxfp8_layouts.cpp:224`, `crr_mxfp8_exact_8wave_fastpath.inc:103-158`). Removing the barriers would re-enable LLVM scheduler heuristics that were tuned out by R20-R24. Could be tested, but per the brief this is the **alternative** lever, only worth pursuing if the primary audit found something. Since the audit found no conflicts, the original sched_barrier mask `0` is a tuned hyperparam and changing it is a separate exploratory lever (recommend R30 Dev candidate).

---

## Audit methodology checklist

- [x] Read `kernel_mxfp8_layouts.cpp:418-496` (V2/V2a load helpers).
- [x] Read `crr_mxfp8_exact_8wave_fastpath.inc` (V2-CRR fastpath, locate ds_read calls).
- [x] Identify all `ds_read*` / `ds_write*` instructions in V2-CRR steady state.
- [x] Compute per-lane addresses for all 64 lanes, K_HALF ∈ {0,1}, j ∈ [0, RT::width).
- [x] Compute `(addr/4) % 32` for each lane → bank.
- [x] Group lanes by 16-lane dispatch cycle and check uniqueness within each cycle.
- [x] Verified: 0 conflicts in all configurations of the V2 / V2a column-load helpers.
- [x] Verified: ST swizzle on the store side mirrors the load and produces the same per-cycle distribution.

**Closure:** LDS bank conflicts are not contributing to V2-CRR under-performance. Lever closed.
