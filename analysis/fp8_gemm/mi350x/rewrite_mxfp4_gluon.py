#!/usr/bin/env python3
"""MXFP4 Gluon C++ .s rewriter: redistribute PFs into MFMA streams.

The baseline kernel front-loads all 16 tile prefetches (buffer_load_dwordx4 ... lds)
in a concentrated block between the barrier and Step 3's first MFMA. This creates
~47 non-MFMA instructions of MFMA pipeline starvation after each barrier.

Fix: move the 16 PF pairs (s_mov_b32 m0 + buffer_load_dwordx4 ... lds) into the
pure-MFMA sections of Steps 3 and 4, interleaved 1:3 with MFMAs. The s_mov_b32 m0
is SALU (dual-issues with MFMA, free) and buffer_load uses VMEM (separate pipe).
MFMAs between m0 setup and buffer_load provide the required hazard delay,
eliminating s_nop instructions.

Keeps v_cndmask, s_mov s60, s_cmp in the post-barrier gap (minimal overhead).
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
        return 'tile_pf'
    if 's_mov_b32 m0' in s:
        return 'm0_set'
    if 's_nop' in s:
        return 'nop'
    if 'v_cndmask' in s:
        return 'cndmask'
    if 's_barrier' in s and 'sched' not in s:
        return 'barrier'
    if 's_waitcnt' in s:
        return 'waitcnt'
    if 's_cbranch' in s:
        return 'branch'
    return 'other'


def find_loop(lines):
    """Find the largest backward branch (main loop)."""
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


def find_barrier_in_loop(lines, start, end):
    """Find the barrier instruction within the loop. Return line index."""
    for i in range(start, end + 1):
        if classify(lines[i]) == 'barrier':
            return i
    return None


def find_pf_block(lines, barrier_idx, end):
    """Extract the PF block between barrier and the next ;;#ASMSTART.
    Returns (pf_start, pf_end) — range of lines containing PFs and related instrs."""
    pf_start = barrier_idx + 1
    pf_end = pf_start
    for i in range(pf_start, end + 1):
        if ';;#ASMSTART' in lines[i]:
            pf_end = i - 1
            break
    return pf_start, pf_end


def extract_pf_pairs(lines, pf_start, pf_end):
    """Extract PF pairs and non-PF instructions from the PF block.

    A PF sequence is: s_mov_b32 m0, sXX → buffer_load_dwordx4 ... lds
    The first buffer_load may not have a preceding m0 in this block (uses prev value).

    Returns: (pf_groups, keep_instrs)
      pf_groups: list of (m0_line_or_None, bufload_line) tuples
      keep_instrs: list of non-PF instruction lines to keep after barrier
    """
    pf_groups = []
    keep_instrs = []
    pending_m0 = None

    for i in range(pf_start, pf_end + 1):
        kind = classify(lines[i])
        if kind == 'tile_pf':
            pf_groups.append((pending_m0, lines[i]))
            pending_m0 = None
        elif kind == 'm0_set':
            if pending_m0 is not None:
                keep_instrs.append(pending_m0)
            pending_m0 = lines[i]
        elif kind == 'nop':
            continue
        elif kind == 'meta':
            continue
        else:
            keep_instrs.append(lines[i])

    if pending_m0 is not None:
        keep_instrs.append(pending_m0)

    return pf_groups, keep_instrs


def find_pure_mfma_section(lines, asm_start, asm_end):
    """Find where ds_reads end and pure MFMAs begin within an asm block.
    Returns index of the first pure MFMA (no ds_read following it)."""
    last_ds = None
    for i in range(asm_start, asm_end + 1):
        if classify(lines[i]) == 'ds_read':
            last_ds = i
    if last_ds is None:
        return asm_start
    for i in range(last_ds + 1, asm_end + 1):
        if classify(lines[i]) == 'mfma':
            return i
    return asm_end


def find_asm_blocks_after_barrier(lines, pf_end, loop_end):
    """Find the two asm blocks (Steps 3 and 4) after the PF block.
    Returns list of (start, end) tuples for each asm block."""
    blocks = []
    i = pf_end + 1
    while i <= loop_end and len(blocks) < 2:
        if ';;#ASMSTART' in lines[i]:
            block_start = i
            j = i + 1
            while j <= loop_end:
                if ';;#ASMEND' in lines[j]:
                    mfma_count = sum(1 for k in range(block_start, j + 1)
                                     if classify(lines[k]) == 'mfma')
                    if mfma_count >= 30:
                        blocks.append((block_start, j))
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return blocks


def interleave_pfs_into_mfmas(lines, pure_mfma_start, asm_end, pf_groups):
    """Insert PF pairs into the pure-MFMA section.

    Pattern per PF (Gluon-style):
      s_mov_b32 m0, sXX     ; SALU, before MFMA (dual-issues)
      v_mfma_scale ...       ; MFMA provides m0 hazard delay
      buffer_load_dwordx4    ; VMEM, uses m0

    Distributes PFs evenly across the available MFMAs.
    """
    mfma_indices = []
    for i in range(pure_mfma_start, asm_end + 1):
        if classify(lines[i]) == 'mfma':
            mfma_indices.append(i)

    n_mfma = len(mfma_indices)
    n_pf = len(pf_groups)
    if n_pf == 0 or n_mfma == 0:
        return lines

    spacing = max(n_mfma // n_pf, 2)
    insert_points = []
    for k in range(n_pf):
        idx = min(k * spacing, n_mfma - 1)
        insert_points.append(mfma_indices[idx])

    offset = 0
    for k, (m0_line, bufload_line) in enumerate(pf_groups):
        mfma_pos = insert_points[k] + offset
        ins = []
        if m0_line is not None:
            ins.append(m0_line)
        ins_before = len(ins)
        lines.insert(mfma_pos, *[]) if False else None

        for idx_off, line in enumerate(ins):
            lines.insert(mfma_pos + idx_off, line)
        offset += ins_before

        after_mfma = mfma_pos + ins_before + 1
        bufload = bufload_line if bufload_line.endswith('\n') else bufload_line + '\n'
        lines.insert(after_mfma, bufload)
        offset += 1

    return lines


def rewrite(input_path, output_path):
    with open(input_path) as f:
        lines = f.readlines()

    loop_start, loop_end = find_loop(lines)
    if loop_start is None:
        print("ERROR: no loop found", file=sys.stderr)
        sys.exit(1)
    print(f"Loop: L{loop_start+1}-{loop_end+1} ({loop_end-loop_start+1} lines)",
          file=sys.stderr)

    barrier_idx = find_barrier_in_loop(lines, loop_start, loop_end)
    if barrier_idx is None:
        print("ERROR: no barrier in loop", file=sys.stderr)
        sys.exit(1)
    print(f"Barrier at L{barrier_idx+1}", file=sys.stderr)

    pf_start, pf_end = find_pf_block(lines, barrier_idx, loop_end)
    print(f"PF block: L{pf_start+1}-{pf_end+1}", file=sys.stderr)

    pf_groups, keep_instrs = extract_pf_pairs(lines, pf_start, pf_end)
    print(f"Extracted {len(pf_groups)} PF pairs, {len(keep_instrs)} keep instrs",
          file=sys.stderr)

    asm_blocks = find_asm_blocks_after_barrier(lines, pf_end, loop_end)
    if len(asm_blocks) < 2:
        print(f"ERROR: expected 2 asm blocks after PF, found {len(asm_blocks)}",
              file=sys.stderr)
        sys.exit(1)

    for idx, (s, e) in enumerate(asm_blocks):
        mfma_cnt = sum(1 for i in range(s, e+1) if classify(lines[i]) == 'mfma')
        ds_cnt = sum(1 for i in range(s, e+1) if classify(lines[i]) == 'ds_read')
        print(f"  Step {idx+3}: L{s+1}-{e+1} ({mfma_cnt} MFMAs, {ds_cnt} ds_reads)",
              file=sys.stderr)

    n_pf = len(pf_groups)
    pf_step3 = pf_groups  # ALL PFs in Step 3 for maximum latency hiding
    pf_step4 = []
    print(f"  Distributing: {len(pf_step3)} PFs → Step 3, {len(pf_step4)} PFs → Step 4",
          file=sys.stderr)

    replacement = []
    for line in keep_instrs:
        replacement.append(line if line.endswith('\n') else line + '\n')

    orig_pf_lines = lines[pf_start:pf_end + 1]
    lines[pf_start:pf_end + 1] = replacement

    shift = len(replacement) - len(orig_pf_lines)
    adjusted_blocks = [(s + shift, e + shift) for s, e in asm_blocks]
    loop_end += shift

    for block_idx in reversed(range(2)):
        s, e = adjusted_blocks[block_idx]
        pfs = pf_step4 if block_idx == 1 else pf_step3
        if not pfs:
            continue

        mfma_indices = [i for i in range(s, e + 1)
                        if classify(lines[i]) == 'mfma']

        if not mfma_indices:
            print(f"  WARNING: no pure MFMAs in block {block_idx}", file=sys.stderr)
            continue

        n_m = len(mfma_indices)
        n_p = len(pfs)
        spacing = max(n_m // n_p, 1)

        offset = 0
        for k, (m0_line, bufload_line) in enumerate(pfs):
            target_mfma = min(k * spacing, n_m - 1)
            pos = mfma_indices[target_mfma] + offset

            to_insert = []
            if m0_line is not None:
                ml = m0_line.rstrip('\n') + '\n'
                to_insert.append(ml)

            for idx_off, il in enumerate(to_insert):
                lines.insert(pos + idx_off, il)
            offset += len(to_insert)

            after_mfma = pos + len(to_insert) + 1
            bl = bufload_line.rstrip('\n') + '\n'
            lines.insert(after_mfma, bl)
            offset += 1

    orig_loop_start, orig_loop_end = find_loop(lines)
    orig_mfma = sum(1 for l in lines[orig_loop_start:orig_loop_end+1]
                    if 'v_mfma_scale' in l)
    orig_pf = sum(1 for l in lines[orig_loop_start:orig_loop_end+1]
                  if 'buffer_load_dwordx4' in l and 'lds' in l)
    orig_ds = sum(1 for l in lines[orig_loop_start:orig_loop_end+1]
                  if 'ds_read_b128' in l)
    orig_nop = sum(1 for l in lines[orig_loop_start:orig_loop_end+1]
                   if 's_nop' in l.strip())

    print(f"\nResult: {orig_mfma} MFMAs, {orig_pf} PFs, {orig_ds} ds_reads, "
          f"{orig_nop} s_nop (should be 0)", file=sys.stderr)

    with open(output_path, 'w') as f:
        f.writelines(lines)
    print(f"Written to {output_path}", file=sys.stderr)


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'mxfp4_gluon_cpp_device.s'
    out = sys.argv[2] if len(sys.argv) > 2 else 'mxfp4_gluon_cpp_opt.s'
    rewrite(inp, out)
