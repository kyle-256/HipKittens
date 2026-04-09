#!/usr/bin/env python3
"""MXFP4 KPAIR .s rewriter: improve ds_read scheduling.

The KPAIR kernel has 4:1 MFMA:ds_read ratio within each column block:
  MFMA×8, DS_RD×2, MFMA×8, DS_RD×2, MFMA×8, DS_RD×2, MFMA×8, DS_RD×2

The last DS_RDs are issued ~1 cycle before the s_waitcnt lgkmcnt(0),
causing ~20 cycle stalls.

Fix: move DS_RDs before their MFMA group:
  DS_RD×2, MFMA×8, DS_RD×2, MFMA×8, DS_RD×2, MFMA×8, DS_RD×2, MFMA×8

This gives the last DS_RDs 32+ extra cycles of latency hiding.
"""

import re, sys


def classify(line):
    s = line.strip()
    if not s or s.startswith(';') or s.startswith('.') or s.endswith(':'):
        return 'meta'
    if 'v_mfma_scale' in s:
        return 'mfma'
    if 'ds_read_b128' in s:
        return 'ds_read'
    if 'buffer_load_dwordx4' in s and 'lds' in s:
        return 'tile_load'
    if 'buffer_load_dword' in s:
        return 'scale_load'
    if 's_waitcnt' in s:
        return 'waitcnt'
    if 's_barrier' in s and 'sched' not in s:
        return 'barrier'
    return 'other'


def find_loop(lines):
    best = (None, None)
    for i, line in enumerate(lines):
        m = re.search(r's_cbranch_\w+\s+(\.LBB\d+_\d+)', line)
        if m:
            target = m.group(1)
            for j in range(i):
                if target + ':' in lines[j] and j < i:
                    if best[0] is None or i - j > best[1] - best[0]:
                        best = (j, i)
    return best


def rewrite(input_path, output_path):
    with open(input_path) as f:
        lines = f.readlines()

    start, end = find_loop(lines)
    if start is None:
        print("ERROR: no loop found", file=sys.stderr)
        sys.exit(1)

    print(f"Loop: L{start+1}-{end+1} ({end-start+1} lines)", file=sys.stderr)

    orig_mfma = sum(1 for l in lines[start:end+1] if 'v_mfma_scale' in l)
    orig_ds = sum(1 for l in lines[start:end+1] if 'ds_read_b128' in l)
    orig_tile = sum(1 for l in lines[start:end+1]
                    if 'buffer_load_dwordx4' in l and 'lds' in l)

    new_lines = list(lines)
    transformed = 0

    i = start
    while i <= end:
        if classify(new_lines[i]) != 'mfma':
            i += 1
            continue

        group_start = i
        mfma_count = 0
        j = i
        while j <= end and classify(new_lines[j]) in ('mfma', 'meta'):
            if classify(new_lines[j]) == 'mfma':
                mfma_count += 1
            j += 1
        mfma_end = j

        ds_count = 0
        while j <= end and classify(new_lines[j]) in ('ds_read', 'meta'):
            if classify(new_lines[j]) == 'ds_read':
                ds_count += 1
            j += 1
        ds_end = j

        if mfma_count >= 4 and ds_count >= 1:
            mfma_block = new_lines[group_start:mfma_end]
            ds_block = new_lines[mfma_end:ds_end]
            new_lines[group_start:ds_end] = ds_block + mfma_block
            transformed += 1
            i = ds_end
        else:
            i = j if j > i else i + 1

    print(f"Phase 1: {transformed} groups reordered (DS_RD before MFMA)",
          file=sys.stderr)

    new_start, new_end = find_loop(new_lines)
    new_mfma = sum(1 for l in new_lines[new_start:new_end+1]
                   if 'v_mfma_scale' in l)
    new_ds = sum(1 for l in new_lines[new_start:new_end+1]
                 if 'ds_read_b128' in l)
    new_tile = sum(1 for l in new_lines[new_start:new_end+1]
                   if 'buffer_load_dwordx4' in l and 'lds' in l)

    print(f"MFMAs: {orig_mfma} → {new_mfma}, ds_reads: {orig_ds} → {new_ds}, "
          f"tiles: {orig_tile} → {new_tile}", file=sys.stderr)

    if new_mfma != orig_mfma or new_ds != orig_ds or new_tile != orig_tile:
        print("ERROR: instruction count mismatch!", file=sys.stderr)
        sys.exit(1)

    with open(output_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Written to {output_path}", file=sys.stderr)


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'mxfp4_kpair_device.s'
    out = sys.argv[2] if len(sys.argv) > 2 else 'mxfp4_kpair_opt.s'
    rewrite(inp, out)
