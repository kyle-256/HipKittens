#!/usr/bin/env python3
"""MXFP4 Gluon C++ .s rewriter — Phase 1 + Phase 2.

Phase 1: Move 16 PF pairs from concentrated post-barrier block into Step 3's MFMA stream.
Phase 2: Pipeline the header SALU into Step 4's MFMA stream (dual-issue = free).
         - Insert prologue copy of header SALU before loop label (first iteration setup)
         - Move header SALU from loop start into Step 4 (subsequent iterations)
         - Adjust branch constant +1 (pipeline shifts exit check by one iteration)
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
    if 'v_mov_b32' in s:
        return 'vmov'
    if re.match(r'\s*buffer_load_dword\s', s) and 'lds' not in s:
        return 'scale_load'
    if s.startswith('s_') and not any(x in s for x in ['barrier', 'nop', 'waitcnt', 'cbranch']):
        return 'salu'
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


def find_barrier_in_loop(lines, start, end):
    for i in range(start, end + 1):
        if classify(lines[i]) == 'barrier':
            return i
    return None


def find_large_asm_blocks(lines, start, end):
    blocks = []
    i = start
    while i <= end:
        if ';;#ASMSTART' in lines[i]:
            bs = i
            j = i + 1
            while j <= end:
                if ';;#ASMEND' in lines[j]:
                    mc = sum(1 for k in range(bs, j + 1) if classify(lines[k]) == 'mfma')
                    if mc >= 30:
                        blocks.append((bs, j))
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return blocks


# ─── Phase 1: PF redistribution ─────────────────────────────────────────────

def phase1_redistribute_pfs(lines):
    loop_start, loop_end = find_loop(lines)
    if loop_start is None:
        print("ERROR: no loop found", file=sys.stderr); sys.exit(1)
    print(f"Phase 1 — Loop: L{loop_start+1}-{loop_end+1}", file=sys.stderr)

    barrier_idx = find_barrier_in_loop(lines, loop_start, loop_end)
    if barrier_idx is None:
        print("ERROR: no barrier in loop", file=sys.stderr); sys.exit(1)

    pf_start = barrier_idx + 1
    pf_end = pf_start
    for i in range(pf_start, loop_end + 1):
        if ';;#ASMSTART' in lines[i]:
            pf_end = i - 1
            break

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
        elif kind in ('nop', 'meta'):
            continue
        else:
            keep_instrs.append(lines[i])
    if pending_m0 is not None:
        keep_instrs.append(pending_m0)

    print(f"  {len(pf_groups)} PF pairs, {len(keep_instrs)} keep instrs", file=sys.stderr)

    asm_blocks = []
    i = pf_end + 1
    while i <= loop_end and len(asm_blocks) < 2:
        if ';;#ASMSTART' in lines[i]:
            bs = i; j = i + 1
            while j <= loop_end:
                if ';;#ASMEND' in lines[j]:
                    mc = sum(1 for k in range(bs, j + 1) if classify(lines[k]) == 'mfma')
                    if mc >= 30:
                        asm_blocks.append((bs, j))
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    if len(asm_blocks) < 2:
        print(f"ERROR: need 2 asm blocks after PF, found {len(asm_blocks)}", file=sys.stderr)
        sys.exit(1)

    replacement = [l if l.endswith('\n') else l + '\n' for l in keep_instrs]
    orig_pf = lines[pf_start:pf_end + 1]
    lines[pf_start:pf_end + 1] = replacement
    shift = len(replacement) - len(orig_pf)
    step3 = (asm_blocks[0][0] + shift, asm_blocks[0][1] + shift)

    mfma_idx = [i for i in range(step3[0], step3[1] + 1) if classify(lines[i]) == 'mfma']
    n_m, n_p = len(mfma_idx), len(pf_groups)
    if n_m > 0 and n_p > 0:
        spacing = max(n_m // n_p, 1)
        off = 0
        for k, (m0_line, bl_line) in enumerate(pf_groups):
            t = min(k * spacing, n_m - 1)
            pos = mfma_idx[t] + off
            ins = []
            if m0_line is not None:
                ins.append(m0_line.rstrip('\n') + '\n')
            for ix, il in enumerate(ins):
                lines.insert(pos + ix, il)
            off += len(ins)
            lines.insert(pos + len(ins) + 1, bl_line.rstrip('\n') + '\n')
            off += 1

    ls, le = find_loop(lines)
    nm = sum(1 for l in lines[ls:le+1] if 'v_mfma_scale' in l)
    np = sum(1 for l in lines[ls:le+1] if 'buffer_load_dwordx4' in l and 'lds' in l)
    nn = sum(1 for l in lines[ls:le+1] if 's_nop' in l.strip())
    print(f"  → {nm} MFMAs, {np} PFs, {nn} s_nop", file=sys.stderr)
    return lines


# ─── Phase 2: Header SALU pipelining ────────────────────────────────────────

def phase2_pipeline_header(lines):
    loop_start, loop_end = find_loop(lines)
    if loop_start is None:
        return lines
    print(f"\nPhase 2 — Loop: L{loop_start+1}-{loop_end+1}", file=sys.stderr)

    all_blocks = find_large_asm_blocks(lines, loop_start, loop_end)
    if len(all_blocks) < 4:
        print(f"  WARNING: need 4 asm blocks, found {len(all_blocks)}. Skip.", file=sys.stderr)
        return lines

    step1_start = all_blocks[0][0]
    step4_s, step4_e = all_blocks[3]

    # --- Extract header SALU ---
    header_salu = []
    header_keep = []
    for i in range(loop_start + 1, step1_start):
        kind = classify(lines[i])
        if kind in ('meta', 'nop'):
            continue
        elif kind == 'salu':
            header_salu.append(lines[i])
        else:
            header_keep.append(lines[i])

    if not header_salu:
        print("  No SALU to move.", file=sys.stderr)
        return lines
    print(f"  Header: {len(header_salu)} SALU to pipeline, {len(header_keep)} keep",
          file=sys.stderr)

    # --- Find branch exit condition (s_cmp_eq_u32 sXX, N) ---
    barrier_idx = find_barrier_in_loop(lines, loop_start, loop_end)
    exit_cmp_reg = None
    exit_cmp_val = None
    exit_cmp_idx = None
    for i in range(barrier_idx, loop_end + 1):
        m = re.search(r's_cmp_eq_u32\s+(s\d+),\s+(\d+)', lines[i].strip())
        if m:
            exit_cmp_reg = m.group(1)
            exit_cmp_val = int(m.group(2))
            exit_cmp_idx = i
            break
    if exit_cmp_reg is None:
        print("  WARNING: no exit s_cmp found. Skip.", file=sys.stderr)
        return lines
    print(f"  Exit cmp: {exit_cmp_reg} == {exit_cmp_val} at L{exit_cmp_idx+1}", file=sys.stderr)

    # ====== Apply transforms (bottom-up to preserve indices) ======

    # 2a. Insert adjusted branch condition before s_cbranch
    branch_idx = loop_end  # s_cbranch is the last line of the loop
    new_cmp = f'\ts_cmp_eq_u32 {exit_cmp_reg}, {exit_cmp_val + 1}\n'
    lines.insert(branch_idx, new_cmp)
    loop_end += 1
    print(f"  Added exit cmp: {exit_cmp_reg} == {exit_cmp_val + 1} before branch", file=sys.stderr)

    # 2b. Insert header SALU into Step 4's MFMA stream (dual-issue with MFMAs)
    # Re-find Step 4 after the branch insertion
    loop_start2, loop_end2 = find_loop(lines)
    all_blocks2 = find_large_asm_blocks(lines, loop_start2, loop_end2)
    if len(all_blocks2) < 4:
        print("  WARNING: lost Step 4 after branch insert.", file=sys.stderr)
        return lines
    s4s, s4e = all_blocks2[3]
    mfma_idx = [i for i in range(s4s, s4e + 1) if classify(lines[i]) == 'mfma']
    n_m = len(mfma_idx)
    n_s = len(header_salu)

    if n_m > 0 and n_s > 0:
        start = max(0, n_m - n_s)
        off = 0
        for k, sl in enumerate(header_salu):
            t = min(start + k, n_m - 1)
            pos = mfma_idx[t] + off
            lines.insert(pos, sl.rstrip('\n') + '\n')
            off += 1
        print(f"  Inserted {n_s} SALU into Step 4 (MFMAs {start}-{min(start+n_s-1, n_m-1)})",
              file=sys.stderr)

    # 2c. Replace the original header with just header_keep
    loop_start3, loop_end3 = find_loop(lines)
    all_blocks3 = find_large_asm_blocks(lines, loop_start3, loop_end3)
    step1_start3 = all_blocks3[0][0] if all_blocks3 else step1_start

    new_header = [lines[loop_start3]]  # loop label
    for hl in header_keep:
        new_header.append(hl if hl.endswith('\n') else hl + '\n')

    orig_hdr = lines[loop_start3:step1_start3]
    lines[loop_start3:step1_start3] = new_header
    hdr_shift = len(new_header) - len(orig_hdr)
    print(f"  Header: {len(orig_hdr)} → {len(new_header)} lines (shift {hdr_shift})",
          file=sys.stderr)

    # 2d. Insert prologue copy of header SALU before the loop label
    loop_start4, _ = find_loop(lines)
    prologue = [sl.rstrip('\n') + '\n' for sl in header_salu]
    for idx, pl in enumerate(prologue):
        lines.insert(loop_start4 + idx, pl)
    print(f"  Prologue: {len(prologue)} SALU inserted before loop", file=sys.stderr)

    # --- Final stats ---
    ls, le = find_loop(lines)
    if ls is not None:
        total = lines[ls:le+1]
        def cnt(tag): return sum(1 for l in total if tag in l)
        def cntc(c): return sum(1 for l in total if classify(l) == c)
        print(f"\n  Final loop: L{ls+1}-{le+1} ({le-ls+1} lines)", file=sys.stderr)
        print(f"  {cnt('v_mfma_scale')} MFMAs, "
              f"{sum(1 for l in total if 'buffer_load_dwordx4' in l and 'lds' in l)} PFs, "
              f"{cnt('ds_read_b128')} ds_reads", file=sys.stderr)
        print(f"  SALU in loop: {cntc('salu')}, v_mov: {cntc('vmov')}, "
              f"v_cndmask: {cntc('cndmask')}, scale_load: {cntc('scale_load')}",
              file=sys.stderr)

    return lines


def rewrite(input_path, output_path):
    with open(input_path) as f:
        lines = f.readlines()

    lines = phase1_redistribute_pfs(lines)
    lines = phase2_pipeline_header(lines)

    with open(output_path, 'w') as f:
        f.writelines(lines)
    print(f"\nWritten to {output_path}", file=sys.stderr)


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'mxfp4_gluon_cpp_device.s'
    out = sys.argv[2] if len(sys.argv) > 2 else 'mxfp4_gluon_cpp_opt.s'
    rewrite(inp, out)
