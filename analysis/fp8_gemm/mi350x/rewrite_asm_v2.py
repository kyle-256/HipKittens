#!/usr/bin/env python3
"""Register-dependency-aware ASM rewriter for MXFP4 V2 kernel.

Interleaves ds_reads with MFMAs based on actual register dependencies,
and tile loads with remaining MFMAs at 4:1 ratio.
"""
import re, sys

def parse_vgpr_deps(line):
    """Extract VGPR defs and uses from an instruction."""
    s = line.strip()
    defs, uses = set(), set()
    
    m = re.match(r'ds_read_b128\s+v\[(\d+):(\d+)\],\s*v(\d+)', s)
    if m:
        for v in range(int(m.group(1)), int(m.group(2))+1): defs.add(v)
        uses.add(int(m.group(3)))
        return 'ds_read', defs, uses
    
    m = re.match(r'v_mfma_scale.*?a\[\d+:\d+\],\s*v\[(\d+):(\d+)\],\s*v\[(\d+):(\d+)\],\s*a\[\d+:\d+\],\s*v(\d+),\s*v(\d+)', s)
    if m:
        for v in range(int(m.group(1)), int(m.group(2))+1): uses.add(v)
        for v in range(int(m.group(3)), int(m.group(4))+1): uses.add(v)
        uses.add(int(m.group(5))); uses.add(int(m.group(6)))
        return 'mfma', defs, uses
    
    if 'buffer_load_dwordx4' in s and 'lds' in s:
        return 'tile_load', defs, uses
    if 's_mov_b32 m0' in s:
        return 'm0_set', defs, uses
    if 'buffer_load' in s:
        return 'scale_load', defs, uses
    if 's_waitcnt' in s:
        return 'waitcnt', defs, uses
    if 's_barrier' in s and 'sched' not in s:
        return 'barrier', defs, uses
    if 's_cbranch' in s:
        return 'branch', defs, uses
    return 'other', defs, uses

def rewrite(input_path, output_path):
    with open(input_path) as f:
        lines = f.readlines()

    start = end = None
    for i, line in enumerate(lines):
        if '.LBB0_1:' in line and start is None: start = i
        if start and 's_cbranch' in line and 'LBB0_1' in line: end = i; break

    # Parse loop body
    instrs = []
    for i in range(start, end+1):
        kind, defs, uses = parse_vgpr_deps(lines[i])
        instrs.append({'idx': i, 'line': lines[i], 'kind': kind, 'defs': defs, 'uses': uses})

    ds_reads = [x for x in instrs if x['kind'] == 'ds_read']
    mfmas = [x for x in instrs if x['kind'] == 'mfma']
    tile_loads = [x for x in instrs if x['kind'] == 'tile_load']
    m0_sets = [x for x in instrs if x['kind'] == 'm0_set']
    scale_loads = [x for x in instrs if x['kind'] == 'scale_load']
    others = [x for x in instrs if x['kind'] == 'other']
    barriers = [x for x in instrs if x['kind'] == 'barrier']
    branch = [x for x in instrs if x['kind'] == 'branch']

    # Build: which VGPRs does each ds_read define?
    ds_def_map = {}  # vgpr -> ds_read index
    for idx, dr in enumerate(ds_reads):
        for v in dr['defs']:
            ds_def_map[v] = idx

    # For each MFMA: what's the latest ds_read it depends on?
    mfma_max_dep = []
    for m in mfmas:
        max_dep = -1
        for v in m['uses']:
            if v in ds_def_map:
                max_dep = max(max_dep, ds_def_map[v])
        mfma_max_dep.append(max_dep)

    print(f"ds_reads={len(ds_reads)} mfmas={len(mfmas)} tile_loads={len(tile_loads)}", file=sys.stderr)

    # Build the interleaved schedule
    out = []
    # 1) Loop label
    out.append(lines[start])

    # 2) Preamble: scalars before first ds_read (address computations)
    preamble_end = ds_reads[0]['idx'] if ds_reads else mfmas[0]['idx']
    for x in instrs:
        if x['idx'] <= start or x['idx'] >= preamble_end: continue
        if x['kind'] in ('other',):
            out.append(x['line'])

    # 3) vmcnt(16) + barrier
    out.append('\ts_waitcnt vmcnt(16)\n')
    out.append('\ts_barrier\n')

    # 4) Scale address computations + scale loads (needed before MFMAs)
    # Find scalars between barrier and first ds_read
    for x in instrs:
        if x['kind'] == 'other' and x['idx'] > preamble_end and x['idx'] < ds_reads[0]['idx'] + 60:
            out.append(x['line'])
    for sl in scale_loads:
        out.append(sl['line'])

    # 5) Interleaved ds_reads + MFMAs
    # Strategy: issue ds_reads in batches, after each batch check how many MFMAs can run
    ds_issued = 0
    mfma_issued = 0
    
    # First batch: issue ds_reads 0-7 (needed for first MFMAs) + wait
    FIRST_BATCH = 8
    for j in range(min(FIRST_BATCH, len(ds_reads))):
        out.append(ds_reads[j]['line'])
        ds_issued += 1
    
    # Now interleave: issue 2 ds_reads, then run available MFMAs, repeat
    while ds_issued < len(ds_reads) or mfma_issued < len(mfmas):
        # Issue 2 more ds_reads
        batch = 0
        while batch < 2 and ds_issued < len(ds_reads):
            out.append(ds_reads[ds_issued]['line'])
            ds_issued += 1
            batch += 1
        
        # Wait for the ds_reads that the next MFMAs need
        # lgkmcnt = total_ds_issued - (ds_reads completed needed)
        # We want to run all MFMAs whose max_dep < ds_issued
        pending_lgkm = len(ds_reads) - ds_issued + batch  # approximate
        
        # Find MFMAs that can run (all deps satisfied by issued-and-completed ds_reads)
        # Since ds_reads are FIFO, ds_read[k] completes before ds_read[k+1]
        # After issuing ds_issued reads and waiting for lgkmcnt(remaining):
        # completed = ds_issued - remaining
        # We want: mfma_max_dep[m] < completed
        
        # Insert wait: lgkmcnt(ds_issued - needed_completed) 
        # For now: wait for all issued so far
        if mfma_issued < len(mfmas) and mfma_max_dep[mfma_issued] < ds_issued:
            remaining_after_wait = len(ds_reads) - ds_issued
            out.append(f'\ts_waitcnt lgkmcnt({remaining_after_wait})\n')
        
            # Run all MFMAs whose deps are satisfied
            while mfma_issued < len(mfmas) and mfma_max_dep[mfma_issued] < ds_issued:
                out.append(mfmas[mfma_issued]['line'])
                mfma_issued += 1
    
    # Ensure all ds_reads done before tile loads
    out.append('\ts_waitcnt lgkmcnt(0) vmcnt(0)\n')
    
    # 6) Remaining MFMAs (if any)
    while mfma_issued < len(mfmas):
        out.append(mfmas[mfma_issued]['line'])
        mfma_issued += 1

    # 7) Barrier + tile loads interleaved with trailing MFMAs
    if barriers:
        out.append(barriers[-1]['line'])
    
    # Tile loads with m0 setup
    tile_idx = m0_idx = 0
    while tile_idx < len(tile_loads):
        if m0_idx < len(m0_sets):
            out.append(m0_sets[m0_idx]['line'])
            m0_idx += 1
        out.append(tile_loads[tile_idx]['line'])
        tile_idx += 1

    # 8) Loop counter + remaining address scalars + branch
    # Find the address update / loop counter section (after last MFMA in original)
    last_mfma_idx = mfmas[-1]['idx']
    for x in instrs:
        if x['idx'] > last_mfma_idx and x['kind'] == 'other':
            out.append(x['line'])
    for b in branch:
        out.append(b['line'])

    # Assemble output file
    result = lines[:start]
    result.extend(out)
    result.extend(lines[end+1:])

    with open(output_path, 'w') as f:
        f.writelines(result)

    print(f"Output: {mfma_issued} MFMAs, {ds_issued} ds_reads, {tile_idx} tile_loads", file=sys.stderr)
    print(f"Written to {output_path}", file=sys.stderr)

if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'kernel_mxfp4_v2-hip-amdgcn-amd-amdhsa-gfx950.s'
    out = sys.argv[2] if len(sys.argv) > 2 else 'kernel_mxfp4_v2_rewritten.s'
    rewrite(inp, out)
