#!/usr/bin/env python3
"""MXFP4 .s rewriter: 2:1 → 1:1 ds_read interleaving.

The hybrid kernel uses 2:1 MFMA:ds_read ratio (2 MFMAs then 1 ds_read).
This causes ~28 cycle lgkmcnt stalls because the last ds_read is issued
just 2 cycles before the waitcnt.

Fix: change to 1:1 (1 MFMA, 1 ds_read, repeat). This gives the last
ds_read 8 more MFMAs (~32 cycles) to complete before the waitcnt.
No VGPR constraint issues because register allocation is already done.

Transformation pattern in .s:
  BEFORE (2:1):  MFMA, MFMA, ds_read, MFMA, MFMA, ds_read, ...
  AFTER  (1:1):  MFMA, ds_read, MFMA, MFMA, ds_read, MFMA, ...
                 (ds_reads shifted 1 MFMA earlier)
"""

import re, sys
from collections import Counter


def classify(line):
    s = line.strip()
    if not s or s.startswith(';') or s.startswith('.') or s.endswith(':'):
        return 'meta'
    if 'v_mfma_scale' in s: return 'mfma'
    if 'ds_read_b128' in s: return 'ds_read'
    if 'buffer_load_dwordx4' in s and 'lds' in s: return 'tile_load'
    if 'buffer_load_dword' in s: return 'scale_load'
    if 's_waitcnt' in s: return 'waitcnt'
    if 's_barrier' in s and 'sched' not in s: return 'barrier'
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
        print("ERROR: no loop", file=sys.stderr)
        sys.exit(1)

    print(f"Loop: L{start+1}-{end+1} ({end-start+1} lines)", file=sys.stderr)

    # Find 2:1 patterns: MFMA, MFMA, ds_read (with meta gaps)
    # Transform to: MFMA, ds_read, MFMA
    transformed = 0
    new_lines = list(lines)  # work on a copy

    i = start
    while i <= end:
        # Look for pattern: MFMA at i, MFMA at i+k, ds_read at i+k+m
        # where gaps are only meta lines
        i_kind = classify(new_lines[i])
        if i_kind != 'mfma':
            i += 1
            continue

        # Find next non-meta instruction
        j = i + 1
        while j <= end and classify(new_lines[j]) == 'meta':
            j += 1
        if j > end:
            i += 1
            continue
        j_kind = classify(new_lines[j])

        if j_kind != 'mfma':
            i += 1
            continue

        # Found MFMA at i, MFMA at j. Look for ds_read next.
        k = j + 1
        while k <= end and classify(new_lines[k]) == 'meta':
            k += 1
        if k > end:
            i += 1
            continue
        k_kind = classify(new_lines[k])

        if k_kind != 'ds_read':
            i += 1
            continue

        # Found pattern: MFMA(i), MFMA(j), ds_read(k)
        # Transform to: MFMA(i), ds_read(k), MFMA(j)
        # Move ds_read from position k to after position i (before j)

        # Extract the ds_read line and any trailing meta (;;#ASMEND)
        ds_line = new_lines[k]
        # Also grab trailing meta if it's ;;#ASMEND
        ds_trail = []
        if k + 1 <= end and ';;#ASMEND' in new_lines[k + 1]:
            ds_trail.append(new_lines[k + 1])

        # Remove ds_read (and trail) from original position
        for _ in range(1 + len(ds_trail)):
            new_lines.pop(k)
            end -= 1

        # Insert ds_read after MFMA at position i
        # Find insertion point: right after the MFMA line at i
        insert_at = i + 1
        # Skip any meta after the first MFMA
        while insert_at < j and classify(new_lines[insert_at]) == 'meta':
            insert_at += 1

        for idx, dl in enumerate(reversed([ds_line] + ds_trail)):
            new_lines.insert(insert_at, dl)
            end += 1

        transformed += 1
        # Advance past this group
        i = insert_at + 1 + len(ds_trail) + 1  # past inserted ds_read + second MFMA

    print(f"Transformed {transformed} patterns (2:1 → 1:1)", file=sys.stderr)

    # Verify counts
    orig_mfma = sum(1 for l in lines[start:end+1] if 'v_mfma_scale' in l)
    orig_ds = sum(1 for l in lines[start:end+1] if 'ds_read_b128' in l)
    # Recount in new_lines
    new_start, new_end = find_loop(new_lines)
    new_mfma = sum(1 for l in new_lines[new_start:new_end+1] if 'v_mfma_scale' in l)
    new_ds = sum(1 for l in new_lines[new_start:new_end+1] if 'ds_read_b128' in l)

    print(f"MFMAs: {orig_mfma} → {new_mfma}, ds_reads: {orig_ds} → {new_ds}", file=sys.stderr)
    if new_mfma != orig_mfma or new_ds != orig_ds:
        print("ERROR: count mismatch!", file=sys.stderr)
        sys.exit(1)

    with open(output_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Written to {output_path}", file=sys.stderr)


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'mxfp4_hybrid_device.s'
    out = sys.argv[2] if len(sys.argv) > 2 else 'mxfp4_hybrid_opt.s'
    rewrite(inp, out)
