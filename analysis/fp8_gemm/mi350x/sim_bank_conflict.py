"""
Simulate LDS bank access patterns for ds_read_b64_tr_b8 with different swizzle patterns.
Find a swizzle that eliminates bank conflicts.
"""
import numpy as np
from collections import Counter

def compute_addr_v2(laneid, k, j, col_start, tile_base=0):
    """Compute LDS address for load_col_from_v2_st"""
    row_off = ((laneid % 16) // 2) + ((laneid // 16) * 16)
    col_off = (laneid % 2) * 8
    k_row = row_off + k * 64
    
    base_k = tile_base + ((k_row >> 4) << 11) + ((k_row & 15) << 7)
    sw_k = ((k_row & 7)) << 4
    nc = col_start + j * 16 + col_off
    addr = base_k + (nc ^ sw_k)
    return addr

def compute_addr_row(laneid, k, j, col_start, tile_base=0):
    """Compute LDS address for load_col_from_row_st"""
    row_off = ((laneid % 16) // 2) + ((laneid // 16) * 16)
    col_off = (laneid % 2) * 8
    k_row = row_off + k * 64
    
    base_k = tile_base + ((k_row >> 4) << 11) + ((k_row & 15) << 7)
    sw_k = (((k_row & 15) >> 1) & 7) << 4
    nc = col_start + j * 16 + col_off
    addr = base_k + (nc ^ sw_k)
    return addr

def compute_addr_custom(laneid, k, j, col_start, swizzle_fn, tile_base=0):
    """Compute LDS address with custom swizzle"""
    row_off = ((laneid % 16) // 2) + ((laneid // 16) * 16)
    col_off = (laneid % 2) * 8
    k_row = row_off + k * 64
    
    base_k = tile_base + ((k_row >> 4) << 11) + ((k_row & 15) << 7)
    nc = col_start + j * 16 + col_off
    sw_bits = swizzle_fn(k_row)
    addr = base_k + (nc ^ sw_bits)
    return addr

def count_bank_conflicts(addrs, group_size=32):
    """
    Count bank conflicts for a set of 8-byte reads.
    Each addr starts an 8-byte read = 2 consecutive 4-byte bank entries.
    Process in groups of group_size lanes.
    """
    total_conflicts = 0
    for g in range(0, len(addrs), group_size):
        group = addrs[g:g+group_size]
        banks = []
        for a in group:
            banks.append((a // 4) % 32)
            banks.append(((a + 4) // 4) % 32)
        
        bank_counts = Counter(banks)
        for bank, cnt in bank_counts.items():
            if cnt > 1:
                total_conflicts += cnt - 1
    return total_conflicts

def analyze_swizzle(name, addr_fn, col_start=0, max_j=4):
    """Analyze bank conflicts for a swizzle pattern across all ds_read_b64_tr_b8 calls"""
    total_conflicts = 0
    total_reads = 0
    
    for k in range(2):
        for j in range(max_j):
            addrs = [addr_fn(lane, k, j, col_start) for lane in range(64)]
            # first ds_read
            conflicts = count_bank_conflicts(addrs)
            total_conflicts += conflicts
            total_reads += 1
            
            # second ds_read (k_next = k_row + 8)
            addrs_next = []
            for lane in range(64):
                row_off = ((lane % 16) // 2) + ((lane // 16) * 16)
                col_off = (lane % 2) * 8
                k_row = row_off + k * 64
                k_next = k_row + 8
                
                if 'v2' in name or 'custom' in name:
                    base_n = ((k_next >> 4) << 11) + ((k_next & 15) << 7)
                    if 'v2' in name:
                        sw_n = ((k_next & 7)) << 4
                    elif 'row' in name:
                        sw_n = (((k_next & 15) >> 1) & 7) << 4
                    else:
                        sw_n = 0
                else:
                    base_n = ((k_next >> 4) << 11) + ((k_next & 15) << 7)
                    sw_n = (((k_next & 15) >> 1) & 7) << 4
                
                nc = col_start + j * 16 + col_off
                addr = base_n + (nc ^ sw_n)
                addrs_next.append(addr)
            
            conflicts = count_bank_conflicts(addrs_next)
            total_conflicts += conflicts
            total_reads += 1
    
    return total_conflicts, total_reads

# Analyze existing swizzle patterns
print("=== Bank Conflict Analysis for ds_read_b64_tr_b8 ===\n")

for group_sz in [16, 32, 64]:
    print(f"\n--- Group size = {group_sz} lanes processed per cycle ---")
    
    for name, addr_fn in [("ST_row", compute_addr_row), ("ST_v2", compute_addr_v2)]:
        for width_name, max_j in [("A(width=4)", 4), ("B(width=2)", 2)]:
            total_c = 0
            total_r = 0
            for k in range(2):
                for j in range(max_j):
                    addrs = [addr_fn(lane, k, j, 0) for lane in range(64)]
                    c = count_bank_conflicts(addrs, group_sz)
                    total_c += c
                    total_r += 1
            print(f"  {name} {width_name}: {total_c} conflicts in {total_r} reads ({total_c/total_r:.1f}/read)")

# Now let's try to find a BETTER swizzle
print("\n\n=== Searching for optimal swizzle ===\n")

# The address formula is: addr = base_k + (nc ^ sw_bits)
# where base_k = ((k_row >> 4) << 11) + ((k_row & 15) << 7) is fixed
# nc is the column offset
# sw_bits depends on the row

# Bank = (addr / 4) % 32 = ((base_k + (nc ^ sw_bits)) / 4) % 32
# base_k/4 = ((k_row >> 4) << 9) + ((k_row & 15) << 5)
# Since base_k is always 128-byte aligned (multiple of 128), base_k/4 is multiple of 32
# So bank = ((nc ^ sw_bits) / 4) % 32

# For zero bank conflicts, we need all 64 lanes to access different banks.
# But with 32 banks and 64 lanes each reading 2 banks = 128 bank accesses,
# the minimum is 128/32 = 4 accesses per bank = 0 conflicts (if perfectly distributed).
# Wait, no. With 64 lanes and each reading 2 banks, and only 32 banks:
# Each lane accesses banks (nc^sw)/4%32 and (nc^sw+1)/4%32 (which is the same bank for 8-byte aligned, or adjacent)

# Let me reconsider: addr/4 % 32 gives the bank. For 8-byte read at address a:
# bytes a, a+1, ..., a+7 access banks a//4%32 and (a+4)//4%32
# Since addr is column-based, nc ^ sw_bits is the column after swizzle

# For group_size=16 (most likely on CDNA):
# 16 lanes simultaneously, each reading 2 bank entries = 32 entries
# With 32 banks, need all entries to go to different banks = 0 conflicts

# Let me just try all possible 3-bit XOR swizzle patterns
best_conflicts = float('inf')
best_pattern = None

for shift in range(0, 8):
    for mask_bits in range(0, 16):
        def make_sw(k_row, s=shift, m=mask_bits):
            return ((k_row >> s) & 0x7) << 4
        
        # Test with A (width=4, col_start=0)
        total_c = 0
        for k in range(2):
            for j in range(4):
                addrs = [compute_addr_custom(lane, k, j, 0, make_sw) for lane in range(64)]
                c = count_bank_conflicts(addrs, 16)
                total_c += c
        
        if total_c < best_conflicts:
            best_conflicts = total_c
            best_pattern = f"((k_row >> {shift}) & 7) << 4"
            if total_c == 0:
                print(f"  ZERO conflicts: sw = {best_pattern}")

print(f"\n  Best found: {best_pattern} with {best_conflicts} conflicts")

# Also try different XOR bit positions
print("\n\n=== Trying broader search: different XOR bit positions ===")
for xor_shift in range(0, 8):
    for row_shift in range(0, 5):
        for row_mask in [0x3, 0x7, 0xF]:
            def make_sw2(k_row, rs=row_shift, rm=row_mask, xs=xor_shift):
                return ((k_row >> rs) & rm) << xs
            
            total_c = 0
            for k in range(2):
                for j in range(4):
                    addrs = [compute_addr_custom(lane, k, j, 0, make_sw2) for lane in range(64)]
                    c = count_bank_conflicts(addrs, 16)
                    total_c += c
            
            if total_c == 0:
                print(f"  ZERO: sw = ((k_row >> {row_shift}) & {row_mask:#x}) << {xor_shift}")
            elif total_c < 4:
                print(f"  LOW({total_c}): sw = ((k_row >> {row_shift}) & {row_mask:#x}) << {xor_shift}")
