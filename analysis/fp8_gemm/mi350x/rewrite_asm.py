#!/usr/bin/env python3
"""Post-compilation assembly rewriter for MXFP4 V2 kernel.

Reorders the inner loop to interleave MFMAs with ds_reads and tile loads,
mimicking the LLIR scheduler's throughput-matched interleaving.

Usage:
  python3 rewrite_asm.py kernel_mxfp4_v2-hip-amdgcn-amd-amdhsa-gfx950.s > kernel_mxfp4_v2_rewritten.s
  /opt/rocm/lib/llvm/bin/clang -x assembler kernel_mxfp4_v2_rewritten.s -target amdgcn-amd-amdhsa -mcpu=gfx950 -o kernel_mxfp4_v2_rewritten.o
"""
import sys, re

def classify_inst(line):
    s = line.strip()
    if not s or s.startswith(';') or s.startswith('.') or s.endswith(':'):
        return 'other'
    if 'v_mfma_scale' in s:
        return 'mfma'
    if 'buffer_load_dwordx4' in s and 'lds' in s:
        return 'tile_load'
    if 'ds_read' in s:
        return 'ds_read'
    if 's_waitcnt' in s:
        return 'waitcnt'
    if 's_barrier' in s:
        return 'barrier'
    if 'sched_barrier' in s:
        return 'sched_barrier'
    if 'buffer_load' in s:
        return 'scale_load'
    if 'ds_write' in s:
        return 'ds_write'
    if 's_mov_b32 m0' in s:
        return 'm0_set'
    return 'scalar'

def find_loop(lines):
    """Find .LBB0_1 to s_cbranch...LBB0_1"""
    start = end = None
    for i, line in enumerate(lines):
        if '.LBB0_1:' in line and start is None:
            start = i
        if start and 's_cbranch' in line and 'LBB0_1' in line:
            end = i
            break
    return start, end

def parse_loop_body(lines, start, end):
    """Parse the loop body into classified instruction groups."""
    groups = []
    for i in range(start, end + 1):
        cls = classify_inst(lines[i])
        groups.append((i, cls, lines[i]))
    return groups

def rewrite_loop(groups):
    """Rewrite loop body with MFMA-memory interleaving.
    
    Strategy:
    - Phase 1: Interleave ds_reads with MFMAs at ~1:1 ratio
    - Phase 2: Interleave tile_loads with MFMAs at ~1:4 ratio
    - Preserve waitcnt, barrier, scalar instructions at original relative positions
    """
    mfmas = [(i, l) for i, cls, l in groups if cls == 'mfma']
    ds_reads = [(i, l) for i, cls, l in groups if cls == 'ds_read']
    tile_loads = [(i, l) for i, cls, l in groups if cls == 'tile_load']
    m0_sets = [(i, l) for i, cls, l in groups if cls == 'm0_set']
    scale_loads = [(i, l) for i, cls, l in groups if cls == 'scale_load']
    barriers = [(i, l) for i, cls, l in groups if cls == 'barrier']
    waitcnts = [(i, l) for i, cls, l in groups if cls == 'waitcnt']
    scalars = [(i, l) for i, cls, l in groups if cls == 'scalar']
    others = [(i, l) for i, cls, l in groups if cls in ('other', 'sched_barrier')]
    
    # Keep the loop label and first barrier
    result = []
    result.append(groups[0][2])  # .LBB0_1: label
    
    # Find the initial barrier+waitcnt (loop entry sync)
    for i, cls, line in groups[1:10]:
        if cls in ('barrier', 'waitcnt', 'scalar', 'other', 'sched_barrier'):
            result.append(line)
        else:
            break
    
    # Phase 1: Interleave ds_reads with scale_loads and first MFMAs
    # Issue all scale loads first (they're fast, SGPR-based)
    for _, l in scale_loads:
        result.append(l)
    
    # Interleave: 1 ds_read, 1 MFMA, 1 ds_read, 1 MFMA, ...
    ds_idx = 0
    mfma_idx = 0
    
    # First: issue first 8 ds_reads, then start MFMAs
    # (need some data before first MFMA)
    batch = min(8, len(ds_reads))
    for j in range(batch):
        result.append(ds_reads[ds_idx][1])
        ds_idx += 1
    
    # Wait for first batch
    result.append('\ts_waitcnt lgkmcnt(0) vmcnt(0)\n')
    
    # Now interleave remaining ds_reads with MFMAs
    while ds_idx < len(ds_reads) and mfma_idx < len(mfmas):
        # 1 ds_read
        result.append(ds_reads[ds_idx][1])
        ds_idx += 1
        # 2 MFMAs  
        for _ in range(2):
            if mfma_idx < len(mfmas):
                result.append(mfmas[mfma_idx][1])
                mfma_idx += 1
    
    # Drain remaining ds_reads
    while ds_idx < len(ds_reads):
        result.append(ds_reads[ds_idx][1])
        ds_idx += 1
    
    if ds_idx > 8:
        result.append('\ts_waitcnt lgkmcnt(0)\n')
    
    # Phase 2: Interleave tile_loads with remaining MFMAs
    # Need m0_set before each tile_load
    tile_idx = 0
    m0_idx = 0
    
    if tile_loads:
        # First barrier to protect LDS before tile loads
        result.append('\ts_barrier\n')
    
    while mfma_idx < len(mfmas):
        # 4 MFMAs
        for _ in range(4):
            if mfma_idx < len(mfmas):
                result.append(mfmas[mfma_idx][1])
                mfma_idx += 1
        
        # 1 tile load (with m0 setup)
        if tile_idx < len(tile_loads):
            if m0_idx < len(m0_sets):
                result.append(m0_sets[m0_idx][1])
                m0_idx += 1
            result.append(tile_loads[tile_idx][1])
            tile_idx += 1
    
    # Drain remaining tile loads
    while tile_idx < len(tile_loads):
        if m0_idx < len(m0_sets):
            result.append(m0_sets[m0_idx][1])
            m0_idx += 1
        result.append(tile_loads[tile_idx][1])
        tile_idx += 1
    
    # Add remaining scalars (loop counter, branch)
    for _, l in scalars[-10:]:  # last few scalar ops (loop counter, addresses)
        result.append(l)
    
    # Add the branch
    for i, cls, line in groups:
        if 's_cbranch' in line and 'LBB0_1' in line:
            result.append(line)
            break
    
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: rewrite_asm.py <input.s>", file=sys.stderr)
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        lines = f.readlines()
    
    start, end = find_loop(lines)
    if start is None or end is None:
        print("Could not find loop", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found loop at lines {start+1}-{end+1}", file=sys.stderr)
    
    groups = parse_loop_body(lines, start, end)
    
    # Count instructions
    from collections import Counter
    counts = Counter(cls for _, cls, _ in groups)
    print(f"Loop contents: {dict(counts)}", file=sys.stderr)
    
    # Rewrite
    new_loop = rewrite_loop(groups)
    
    # Output: before loop + rewritten loop + after loop
    for line in lines[:start]:
        sys.stdout.write(line)
    for line in new_loop:
        sys.stdout.write(line)
    for line in lines[end+1:]:
        sys.stdout.write(line)
    
    new_mfma = sum(1 for l in new_loop if 'v_mfma_scale' in l)
    new_ds = sum(1 for l in new_loop if 'ds_read' in l)
    new_tile = sum(1 for l in new_loop if 'buffer_load_dwordx4' in l and 'lds' in l)
    print(f"Rewritten loop: {new_mfma} MFMAs, {new_ds} ds_reads, {new_tile} tile_loads", file=sys.stderr)

if __name__ == '__main__':
    main()
