#!/usr/bin/env python3
"""
Reverse-engineer the LDS swizzle permutation by tracing both paths on CPU.

For each lane, compute:
  1. What bytes the buffer_load_lds writes to which LDS addresses
  2. What bytes the ds_read extracts from which LDS addresses
  3. The net permutation: for each (lane, byte_position), which global byte ends up there

This tells us exactly what register-side permutation we need to replicate.
"""
import numpy as np

# Constants from the kernel
BK = 128          # K-block size in bytes
HB = 128          # tile height (rows)
RBN = 64          # per-warp rows
NUM_THREADS = 256
WARP_THREADS = 64
NUM_WARPS = 4
BPM = 16 * NUM_THREADS  # 4096 bytes per prefetch iteration
PF_MPT = (HB * BK) // (16 * NUM_THREADS)  # = 4

# XOR swizzle function (from the kernel)
def xor_swizzle(offset):
    return offset ^ (((offset % (16 * 128)) >> 8) << 4)

def trace_buffer_load_lds():
    """Trace which global byte goes to which LDS byte for the FULL tile."""
    # The tile is HB x BK = 128 x 128 = 16384 bytes
    # buffer_load_lds loads 16 bytes per thread per prefetch iteration
    # Total: NUM_THREADS * 16 * PF_MPT = 256 * 16 * 4 = 16384 bytes

    # LDS array: lds[addr] = global_byte_offset
    lds = np.full(16384 + 4096, -1, dtype=np.int32)  # extra space for safety

    for warp in range(NUM_WARPS):
        warp_off = warp * (16 * WARP_THREADS)  # = warp * 1024
        for thread_in_warp in range(WARP_THREADS):
            global_thread = warp * WARP_THREADS + thread_in_warp
            for pf_iter in range(PF_MPT):
                lin = warp_off + pf_iter * BPM + global_thread * 16
                # Subtile ID for padding calculation
                sid = lin // (16 * BK)  # subtile_bytes = 16 * 128 = 2048
                lds_addr = lin + sid * 0  # subtile_padding = 0
                # Each thread loads 16 bytes
                for byte_off in range(16):
                    # The swizzled voffset determines which global bytes are loaded
                    # prefill_swizzled_offsets computes the voffset per PF_MPT iteration
                    # For simplicity, assume the global data is loaded linearly
                    # (the swizzle is in the LDS address, not the global address)
                    global_byte = lin + byte_off
                    lds_byte = xor_swizzle(lds_addr + byte_off)
                    if lds_byte < len(lds):
                        lds[lds_byte] = global_byte
    return lds

def trace_ds_read(wn=0):
    """Trace which LDS bytes each lane reads via ds_read_b128.

    For a B tile with wn=0 (first 64 rows), returns:
      reads[lane][read_idx] = list of 16 LDS byte offsets

    There are 8 ds_reads per kpair (d0..d7):
      d[0..3] from lds_a0 (phase 0)
      d[4..7] from lds_a1 (phase 1)
    """
    # compute_lds_base_addrs for B_row_reg:
    # base_tile_rows = 16, base_tile_stride = 16
    # base_tile_cols = 128 (rt_16x128)
    # base_tile_elements_per_stride_group = 4 * 16 = 64

    # The subtile_inplace<RBN, BK>(..., {wn, 0}) selects rows [wn*RBN, (wn+1)*RBN)
    # within the HB=128 row tile. For wn=0: rows 0-63.

    # compute_lds_base_addrs:
    # row_offset = laneid % 16
    # col_offset = 16 * (laneid / 16)  (base_tile_stride = 16)
    # subcols = 128 (ST underlying_subtile_cols)
    # off0 = src_ptr + row_offset * 128 + col_offset
    # addr_p0 = xor_swizzle(off0)
    # col1 = col_offset + 64  (base_tile_elements_per_stride_group = 64)
    # off1 = src_ptr + row_offset * 128 + col1
    # addr_p1 = xor_swizzle(off1)

    # The ds_reads use offsets 0, 2048, 4096, 6144 (= i * subtile_bytes)
    # These select the 4 subtile rows (0-15, 16-31, 32-47, 48-63)

    # For the warp's subtile within the tile:
    # src_ptr = base_lds_addr + wn * RBN * BK (subtile offset)
    # But wn is handled by subtile_inplace, which adjusts the pointer.

    subtile_base = wn * RBN * BK  # = 0 for wn=0

    reads = {}  # reads[(lane, read_idx)] = list of 16 LDS byte offsets

    for lane in range(WARP_THREADS):
        row_offset = lane % 16
        col_offset = 16 * (lane // 16)

        # Phase 0 (p0): columns [0, 64)
        off0 = subtile_base + row_offset * 128 + col_offset
        addr_p0 = xor_swizzle(off0)

        # Phase 1 (p1): columns [64, 128)
        col1 = col_offset + 64
        off1 = subtile_base + row_offset * 128 + col1
        addr_p1 = xor_swizzle(off1)

        # 4 ds_reads from p0 (subtiles 0-3)
        for i in range(4):
            subtile_off = i * 2048  # 16 * 128
            lds_start = addr_p0 + subtile_off
            reads[(lane, i)] = [lds_start + b for b in range(16)]

        # 4 ds_reads from p1 (subtiles 0-3)
        for i in range(4):
            subtile_off = i * 2048
            lds_start = addr_p1 + subtile_off
            reads[(lane, i + 4)] = [lds_start + b for b in range(16)]

    return reads

def trace_direct_load(wn=0):
    """What global bytes each lane gets from direct buffer_load.

    voff = (lid%16) * K_STRIDE + (lid/16) * 16
    For K=8192: K_STRIDE = 4096

    But within a tile, we consider K_STRIDE = BK = 128 (tile-local addressing).
    Actually for direct load, K_STRIDE is the GLOBAL row stride, but the
    data we care about is which tile-local bytes each lane gets.

    The tile covers: rows [n_start..n_start+64), K_cols [bt*BK..bt*BK+BK)
    Direct load for lane l reads from global:
      row = n_start + l%16 + subtile*16
      col = bt*BK + (l/16)*16    (lo phase)
      col = bt*BK + (l/16)*16 + 64  (hi phase)

    Tile-local offset: row_in_tile * BK + col_in_tile
      = (l%16 + subtile*16) * BK + (l/16)*16  (lo phase)
      = (l%16 + subtile*16) * BK + (l/16)*16 + 64  (hi phase)
    """
    reads = {}
    for lane in range(WARP_THREADS):
        for i in range(4):  # 4 subtiles
            # Lo phase: d[i]
            row_in_tile = (lane % 16) + i * 16
            col_in_tile = (lane // 16) * 16
            tile_off = row_in_tile * BK + col_in_tile
            reads[(lane, i)] = [tile_off + b for b in range(16)]

            # Hi phase: d[i+4]
            col_hi = col_in_tile + 64
            tile_off_hi = row_in_tile * BK + col_hi
            reads[(lane, i + 4)] = [tile_off_hi + b for b in range(16)]

    return reads

# Compute both paths
lds_contents = trace_buffer_load_lds()
lds_reads = trace_ds_read(wn=0)
direct_reads = trace_direct_load(wn=0)

# For each (lane, read_idx), compare what global bytes end up in the register
print("Comparing LDS path vs direct path for B tile (wn=0, first 64 rows):")
print(f"{'Lane':>4} {'Read':>4} | {'LDS global bytes':>40} | {'Direct global bytes':>40} | {'Match'}")
print("-" * 100)

mismatches = 0
for lane in [0, 1, 15, 16, 31, 32, 48, 63]:
    for read_idx in range(8):
        # LDS path: ds_read from swizzled address → get data from original global offset
        lds_addrs = lds_reads[(lane, read_idx)]
        lds_global = [lds_contents[a] if a < len(lds_contents) else -1 for a in lds_addrs]

        # Direct path: buffer_load from global offset
        direct_global = direct_reads[(lane, read_idx)]

        match = lds_global[:4] == direct_global[:4]
        if not match:
            mismatches += 1

        # Show first 4 bytes of each
        lds_str = str(lds_global[:4])
        dir_str = str(direct_global[:4])
        m = "✓" if match else "✗"
        if not match or read_idx == 0:  # show all mismatches + first read
            print(f"{lane:>4} {read_idx:>4} | {lds_str:>40} | {dir_str:>40} | {m}")

print(f"\nTotal mismatches: {mismatches} / {64 * 8} = {mismatches/(64*8)*100:.0f}%")

if mismatches > 0:
    # Determine the permutation
    print("\n=== PERMUTATION ANALYSIS ===")
    print("For each (lane, read_idx), which global bytes does LDS give vs direct?")

    # Build permutation map: for lane L, read R, byte B:
    #   direct gives global_byte D = direct_reads[(L,R)][B]
    #   LDS gives global_byte L = lds_global
    #   So the permutation is: register_position(L,R,B) needs global_byte L
    #   But direct gives global_byte D at that position
    #   So we need to permute: data[D] → data[L]

    # Check if the permutation is the same for all lanes in a sub-group
    for read_idx in range(8):
        perm = []
        for lane in range(WARP_THREADS):
            lds_addrs = lds_reads[(lane, read_idx)]
            lds_global = [lds_contents[a] for a in lds_addrs]
            direct_global = direct_reads[(lane, read_idx)]
            if lds_global != direct_global:
                # Find which direct bytes map to which LDS bytes
                perm.append((lane, lds_global[:4], direct_global[:4]))

        if perm:
            print(f"\nRead {read_idx}: {len(perm)} lanes differ")
            for lane, lg, dg in perm[:3]:
                print(f"  Lane {lane}: LDS={lg}, Direct={dg}")
                # The difference: LDS global byte X is at position where direct puts byte Y
                for b in range(4):
                    if lg[b] != dg[b]:
                        print(f"    byte {b}: LDS wants global[{lg[b]}], direct gives global[{dg[b]}]")
                        print(f"    diff: row {lg[b]//BK} col {lg[b]%BK} vs row {dg[b]//BK} col {dg[b]%BK}")
