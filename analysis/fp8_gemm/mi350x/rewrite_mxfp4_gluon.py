#!/usr/bin/env python3
"""MXFP4 Gluon C++ .s rewriter — Phase 1 + Phase 2 + Phase 3.

Phase 1: Move 16 PF pairs from concentrated post-barrier block into Step 3's MFMA stream.
Phase 2: Pipeline the header SALU into Step 4's MFMA stream (dual-issue = free).
         - Insert prologue copy of header SALU before loop label (first iteration setup)
         - Move header SALU from loop start into Step 4 (subsequent iterations)
         - Adjust branch constant +1 (pipeline shifts exit check by one iteration)
Phase 3: Pipeline post-Step4 vmcnt+vmov pairs into Step 4's MFMA stream.
         - Registers not used in Step 4 (v142, v144): move to Step 4's first 2 free MFMA slots
         - LDS pointer regs (v160, v161): move to just after their last ds_read in Step 4
         - Register v140: move to just after its last MFMA use in Step 4
         - Register v143: move to just after its last MFMA use in Step 4
         - Reduces post-Step4 sequential tail (saves 6 of 8 vmov pairs from critical path)
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


# ─── Phase 3: vmcnt+vmov pipelining into Step 4 ─────────────────────────────

def phase3_pipeline_vmov(lines):
    """Move select vmcnt+vmov pairs from post-Step4 into Step 4 MFMA stream.

    Strategy:
    - v142, v144: NOT used in Step 4 → insert before MFMA #1 and #2 of Step 4
    - v160: last ds_read using it is ds_read_b128 vX, v160 → insert after that
    - v161: last ds_read using it is ds_read_b128 vX, v161 → insert after that
    - v140: last MFMA using it (check scale arg) → insert after that MFMA
    - v143: last MFMA using it → insert after that MFMA
    All remaining (v141, v145) stay at the post-Step4 tail.
    """
    loop_start, loop_end = find_loop(lines)
    if loop_start is None:
        return lines

    all_blocks = find_large_asm_blocks(lines, loop_start, loop_end)
    if len(all_blocks) < 4:
        print("  Phase 3: need 4 asm blocks, skip.", file=sys.stderr)
        return lines

    s4s, s4e = all_blocks[3]
    print(f"\nPhase 3 — Step 4: L{s4s+1}-{s4e+1}", file=sys.stderr)

    # --- Find post-Step4 vmcnt+vmov pairs ---
    # After Step 4 ASMEND, look for the block: lgkmcnt(0) block + vmcnt+vmov pairs
    post_start = s4e + 1
    vmcnt_vmov = []   # list of (vmcnt_val, dst_vreg, src_vreg, line_idx_in_lines, pair_lines)
    i = post_start
    while i <= loop_end:
        s = lines[i].strip()
        m_wc = re.match(r's_waitcnt\s+vmcnt\((\d+)\)', s)
        if m_wc and i + 1 <= loop_end:
            vmcnt_val = int(m_wc.group(1))
            s2 = lines[i+1].strip()
            m_mov = re.match(r'v_mov_b32_e32\s+(v\d+),\s+(v\d+)', s2)
            if m_mov:
                vmcnt_vmov.append((vmcnt_val, m_mov.group(1), m_mov.group(2), i, [lines[i], lines[i+1]]))
                i += 2
                continue
        i += 1

    if not vmcnt_vmov:
        print("  Phase 3: no vmcnt+vmov pairs found, skip.", file=sys.stderr)
        return lines

    print(f"  Found {len(vmcnt_vmov)} vmcnt+vmov pairs:", file=sys.stderr)
    for vc, dst, src, idx, _ in vmcnt_vmov:
        print(f"    vmcnt({vc}) → {dst} ← {src} at L{idx+1}", file=sys.stderr)

    # --- Identify which registers are used in Step 4 ---
    step4_lines = lines[s4s:s4e+1]
    def reg_used_in_step4(reg):
        for l in step4_lines:
            # Check if reg appears as a scale arg in MFMA or as ds_read base
            if reg + ',' in l or reg + ' ' in l or l.strip().endswith(reg):
                if 'v_mfma_scale' in l or 'ds_read' in l:
                    return True
        return False

    def last_use_in_step4(reg, as_ds_base=False):
        """Return the ABSOLUTE line index of the last use of reg in Step 4."""
        last = None
        # Pattern for MFMA scale args:
        # v_mfma_scale_... a[acc], vA, vB, a[acc], vAscale, vBscale op_sel...
        # The 5th and 6th comma-separated tokens (0-indexed) are the scale regs
        mfma_scale_pat = re.compile(
            r'v_mfma_scale\S*\s+\S+,\s*\S+,\s*\S+,\s*\S+,\s*(' + reg + r'),\s*\S+'
            r'|v_mfma_scale\S*\s+\S+,\s*\S+,\s*\S+,\s*\S+,\s*\S+,\s*(' + reg + r')\s'
        )
        ds_base_pat = re.compile(r'ds_read_b128\s+\S+,\s*' + reg + r'\s+')
        for j in range(s4s, s4e+1):
            l = lines[j].strip()
            if as_ds_base:
                if ds_base_pat.search(l):
                    last = j
            else:
                if 'v_mfma_scale' in l and mfma_scale_pat.search(l):
                    last = j
        return last

    # Categorize each pair
    to_move_step4_start = []   # (pair_data, insert_before_mfma_idx) - v142, v144
    to_move_after_ds = []      # (pair_data, vreg, after_line_idx) - v160, v161
    to_move_after_mfma = []    # (pair_data, vreg, after_line_idx) - v140, v143
    keep_at_end = []           # v141, v145 (used at last MFMA)

    mfma_idx_s4 = [j for j in range(s4s, s4e+1) if classify(lines[j]) == 'mfma']

    for vc, dst, src, orig_idx, pair_lines in vmcnt_vmov:
        # Check if dst is used as ds_read base in Step 4
        last_ds = last_use_in_step4(dst, as_ds_base=True)
        # Check if dst is used as MFMA scale arg in Step 4
        last_mfma = last_use_in_step4(dst, as_ds_base=False)

        if last_ds is None and last_mfma is None:
            # Not used in Step 4 at all → can go at start (before MFMA #1 or #2)
            to_move_step4_start.append((vc, dst, src, orig_idx, pair_lines))
            print(f"  → {dst}: not in Step 4, will dual-issue with first MFMAs", file=sys.stderr)
        elif last_ds is not None and (last_mfma is None or last_ds >= last_mfma):
            # Used as ds_read base, last use is ds_read
            to_move_after_ds.append((vc, dst, src, orig_idx, pair_lines, last_ds))
            print(f"  → {dst}: last ds_read at L{last_ds+1}", file=sys.stderr)
        elif last_mfma is not None:
            last_mfma_pos = last_mfma
            # Check if it's truly the last MFMA
            n_mfma = len(mfma_idx_s4)
            if last_mfma_pos == mfma_idx_s4[-1]:
                # Last MFMA → keep at end
                keep_at_end.append((vc, dst, src, orig_idx, pair_lines))
                print(f"  → {dst}: last MFMA is at end, keep at tail", file=sys.stderr)
            else:
                to_move_after_mfma.append((vc, dst, src, orig_idx, pair_lines, last_mfma_pos))
                print(f"  → {dst}: last MFMA at L{last_mfma_pos+1}", file=sys.stderr)
        else:
            keep_at_end.append((vc, dst, src, orig_idx, pair_lines))
            print(f"  → {dst}: keep at tail", file=sys.stderr)

    # Collect which pairs to remove from original location
    to_remove_orig = set()
    to_remove_orig.update(orig_idx for _, _, _, orig_idx, _ in to_move_step4_start)
    to_remove_orig.update(orig_idx for _, _, _, orig_idx, _, _ in to_move_after_ds)
    to_remove_orig.update(orig_idx for _, _, _, orig_idx, _, _ in to_move_after_mfma)

    if not to_remove_orig:
        print("  Phase 3: nothing to move, skip.", file=sys.stderr)
        return lines

    # ====== Apply transforms bottom-up to preserve indices ======

    # Step 3c: Insert "after last MFMA" pairs (working in Step 4 area)
    # Do bottom-up: process highest orig_idx first
    all_insertions = []  # (insert_after_line_idx, pair_lines)

    for vc, dst, src, orig_idx, pair_lines, last_mfma_pos in to_move_after_mfma:
        all_insertions.append((last_mfma_pos, pair_lines))  # insert after this line

    for vc, dst, src, orig_idx, pair_lines, last_ds in to_move_after_ds:
        all_insertions.append((last_ds, pair_lines))  # insert after this ds_read

    # Sort insertions bottom-up (highest line first) to preserve indices
    all_insertions.sort(key=lambda x: -x[0])

    for insert_after, pair_lines in all_insertions:
        for k, pl in enumerate(pair_lines):
            lines.insert(insert_after + 1 + k, pl.rstrip('\n') + '\n')

    # Step 3b: Insert "start of Step 4" pairs (before MFMA #1 and #2)
    # Re-find Step 4 after insertions
    loop_s2, loop_e2 = find_loop(lines)
    all_blocks2 = find_large_asm_blocks(lines, loop_s2, loop_e2)
    if len(all_blocks2) >= 4:
        s4s2, s4e2 = all_blocks2[3]
        mfma_idx2 = [j for j in range(s4s2, s4e2+1) if classify(lines[j]) == 'mfma']
        slot = 0
        off2 = 0
        for vc, dst, src, orig_idx, pair_lines in to_move_step4_start:
            if slot < len(mfma_idx2):
                pos = mfma_idx2[slot] + off2
                for k, pl in enumerate(pair_lines):
                    lines.insert(pos + k, pl.rstrip('\n') + '\n')
                off2 += len(pair_lines)
                slot += 1

    # Step 3a: Remove original pairs from post-Step4 (bottom-up)
    # Re-find the original locations by scanning for the pairs
    # Since indices shifted, re-scan for the original vmcnt+vmov pattern in post-Step4
    loop_s3, loop_e3 = find_loop(lines)
    all_blocks3 = find_large_asm_blocks(lines, loop_s3, loop_e3)
    if all_blocks3:
        s4e3 = all_blocks3[3][1] if len(all_blocks3) >= 4 else loop_e3
    else:
        s4e3 = loop_e3

    # Collect pairs to remove (those that were moved)
    moved_dsts = set()
    moved_dsts.update(dst for _, dst, _, _, _ in to_move_step4_start)
    moved_dsts.update(dst for _, dst, _, _, _, _ in to_move_after_ds)
    moved_dsts.update(dst for _, dst, _, _, _, _ in to_move_after_mfma)

    # Scan post-Step4 area for vmcnt+vmov pairs matching moved_dsts and remove them
    i = s4e3 + 1
    remove_lines = []
    while i <= loop_e3:
        s = lines[i].strip()
        m_wc = re.match(r's_waitcnt\s+vmcnt\((\d+)\)', s)
        if m_wc and i + 1 <= loop_e3:
            s2 = lines[i+1].strip()
            m_mov = re.match(r'v_mov_b32_e32\s+(v\d+),\s+(v\d+)', s2)
            if m_mov and m_mov.group(1) in moved_dsts:
                remove_lines.extend([i, i+1])
                i += 2
                continue
        i += 1

    # Remove bottom-up
    for idx in sorted(remove_lines, reverse=True):
        del lines[idx]

    # --- Final stats ---
    ls, le = find_loop(lines)
    if ls is not None:
        ab = find_large_asm_blocks(lines, ls, le)
        if len(ab) >= 4:
            s4s_f, s4e_f = ab[3]
            post_vmov = sum(1 for l in lines[s4e_f+1:le+1] if 'v_mov_b32' in l)
            moved_count = len(to_move_step4_start) + len(to_move_after_ds) + len(to_move_after_mfma)
            print(f"  Moved {moved_count} pairs into Step 4; {post_vmov} vmov remain at tail",
                  file=sys.stderr)

    return lines


# ─── Phase 4: Header VALU pipelining into Step 4 tail ───────────────────────

# VALU instruction patterns found in the loop header
_VALU_PATS = ['v_lshl_or', 'v_add_u32', 'v_lshrrev_b32', 'v_bitop3']


def _is_header_valu(line):
    s = line.strip()
    return classify(line) == 'cndmask' or any(p in s for p in _VALU_PATS)


def phase4_pipeline_header_valu(lines):
    """Move loop-header VALU chain into Step 4 tail (before lgkmcnt block).

    The VALU chain computes v66, v67 (ds_read bases for Step 1) and v110, v126
    (ds_read bases for Steps 2/3). All their inputs (s63, vcc, v138, v147, v158,
    v159, s55) are already available at Step 4's end via Phase 2's SALU pipeline.
    Moving them here eliminates the ~40-cycle stall at the loop header before Step 1.
    """
    loop_start, loop_end = find_loop(lines)
    if loop_start is None:
        return lines

    all_blocks = find_large_asm_blocks(lines, loop_start, loop_end)
    if len(all_blocks) < 4:
        print("  Phase 4: need 4 asm blocks, skip.", file=sys.stderr)
        return lines

    step1_start = all_blocks[0][0]
    s4s, s4e = all_blocks[3]

    print(f"\nPhase 4 — Step 4: L{s4s+1}-{s4e+1}", file=sys.stderr)

    # --- Extract header VALU ---
    header_valu = []
    header_keep = []
    for i in range(loop_start + 1, step1_start):
        kind = classify(lines[i])
        if _is_header_valu(lines[i]):
            header_valu.append(lines[i])
        elif kind in ('meta', 'nop'):
            continue
        else:
            header_keep.append(lines[i])

    if not header_valu:
        print("  Phase 4: no VALU in header, skip.", file=sys.stderr)
        return lines

    print(f"  Header VALU: {len(header_valu)} instrs to move, {len(header_keep)} keep",
          file=sys.stderr)

    # ====== Apply transforms bottom-up to preserve indices ======

    # 4a. Insert VALU after Step 4's last MFMA block (before the lgkmcnt ASMBLOCK)
    #     The lgkmcnt block is at s4e+1: ";;#ASMSTART\n  s_waitcnt lgkmcnt(0)\n;;#ASMEND"
    #     Insert VALU BEFORE that block.
    insert_pos = s4e + 1  # right after ;;#ASMEND of Step 4
    for k, vl in enumerate(header_valu):
        lines.insert(insert_pos + k, vl.rstrip('\n') + '\n')
    shift_a = len(header_valu)

    # 4b. Replace loop header with just header_keep (remove VALU from header)
    # Re-find loop after insertion
    loop_s2, loop_e2 = find_loop(lines)
    ab2 = find_large_asm_blocks(lines, loop_s2, loop_e2)
    step1_s2 = ab2[0][0] if ab2 else (step1_start + shift_a)

    new_header = [lines[loop_s2]]  # keep loop label
    for hl in header_keep:
        new_header.append(hl if hl.endswith('\n') else hl + '\n')
    orig_hdr = lines[loop_s2:step1_s2]
    lines[loop_s2:step1_s2] = new_header
    hdr_shift = len(new_header) - len(orig_hdr)
    print(f"  Header: {len(orig_hdr)} → {len(new_header)} lines (shift {hdr_shift})",
          file=sys.stderr)

    # 4c. Insert prologue copy of VALU before loop label (for first iteration)
    loop_s3, _ = find_loop(lines)
    prologue = [vl.rstrip('\n') + '\n' for vl in header_valu]
    for k, pl in enumerate(prologue):
        lines.insert(loop_s3 + k, pl)
    print(f"  Prologue: {len(prologue)} VALU inserted before loop", file=sys.stderr)

    # --- Final stats ---
    ls, le = find_loop(lines)
    if ls is not None:
        total = lines[ls:le+1]
        print(f"\n  Final loop: L{ls+1}-{le+1} ({le-ls+1} lines)", file=sys.stderr)
        n_valu = sum(1 for l in total if _is_header_valu(l))
        print(f"  VALU in loop: {n_valu}", file=sys.stderr)

    return lines


def phase5_move_vmcnt_header(lines):
    """Move s_waitcnt vmcnt(N) from loop header to just before lgkmcnt(0) between Step 1 and Step 2.

    The vmcnt wait in the loop header blocks until tile-prefetch buffer_load_dwordx4 lds ops
    from the previous iteration's Step 3 complete. Step 1's MFMAs don't depend on those loads,
    so we can defer the wait until right before the ds_reads of Step 2 that actually use the
    LDS data. This gives ~2048 cycles of Step 1 MFMAs for the loads to complete, eliminating
    the stall at the loop header.
    """
    loop_start, loop_end = find_loop(lines)
    if loop_start is None:
        return lines

    all_blocks = find_large_asm_blocks(lines, loop_start, loop_end)
    if len(all_blocks) < 2:
        print("  Phase 5: need >=2 asm blocks, skip.", file=sys.stderr)
        return lines

    step1_start, step1_end = all_blocks[0]

    # Find and remove vmcnt wait in loop header (between loop_start and step1_start)
    vmcnt_line = None
    vmcnt_idx = None
    for i in range(loop_start + 1, step1_start):
        m = re.match(r'\s*s_waitcnt\s+vmcnt\((\d+)\)\s*$', lines[i])
        if m:
            vmcnt_line = lines[i]
            vmcnt_idx = i
            break

    if vmcnt_line is None:
        print("  Phase 5: no vmcnt wait in header, skip.", file=sys.stderr)
        return lines

    val = re.search(r'vmcnt\((\d+)\)', vmcnt_line).group(1)
    print(f"\nPhase 5 — Move vmcnt({val}) from header (L{vmcnt_idx+1}) to before Step1→Step2 lgkmcnt",
          file=sys.stderr)

    # Remove from header
    lines.pop(vmcnt_idx)

    # Re-find positions after removal
    loop_start2, _ = find_loop(lines)
    ab2 = find_large_asm_blocks(lines, loop_start2, find_loop(lines)[1])
    s1s2, s1e2 = ab2[0]

    # Find the lgkmcnt(0) ASMSTART block right after Step 1's ASMEND
    # It should be at s1e2+1 (;;#ASMSTART) and s1e2+2 (s_waitcnt lgkmcnt(0))
    insert_pos = s1e2 + 1  # right after Step 1 ASMEND, before the lgkmcnt ASMSTART
    lines.insert(insert_pos, vmcnt_line)
    print(f"  Inserted vmcnt({val}) at new L{insert_pos+1} (before lgkmcnt block)", file=sys.stderr)

    return lines


def phase6_remove_redundant_vmcnt(lines):
    """Remove vmcnt(8) between Step 1 ds_reads and Step 2 MFMAs.

    Step 2 uses A-tiles from prev-iter ds_reads (lgkmcnt-gated) and B-tiles from
    Step 1 paired ds_reads (lgkmcnt(0) at L823). The 8 scale buffer_loads from
    the loop header don't feed any Step 2 register. Removing this wait saves the
    stall cost (≤1 load completion) and one instruction.
    """
    loop_start, loop_end = find_loop(lines)
    if loop_start is None:
        return lines

    all_blocks = find_large_asm_blocks(lines, loop_start, loop_end)
    if len(all_blocks) < 2:
        return lines

    s1s, s1e = all_blocks[0]
    s2s, s2e = all_blocks[1]

    # Find vmcnt(N) between Step 1 end and Step 2 start (outside any ASMSTART block)
    removed = 0
    i = s1e + 1
    while i < s2s:
        m = re.match(r'\s*s_waitcnt\s+vmcnt\((\d+)\)\s*$', lines[i])
        if m:
            val = m.group(1)
            print(f"  Phase 6: remove vmcnt({val}) at L{i+1} (between Step1 and Step2)",
                  file=sys.stderr)
            lines.pop(i)
            removed += 1
            # don't increment i since we popped
        else:
            i += 1

    if removed == 0:
        print("  Phase 6: no free-standing vmcnt found between Step1 and Step2", file=sys.stderr)

    return lines


def phase7_remove_prebarrier_vmcnt(lines):
    """Remove vmcnt(N) ASMBLOCK immediately before s_barrier if no LDS-dest loads are outstanding.

    The vmcnt before s_barrier was protecting against buffer_load→lds outstanding ops.
    Those tile-prefetch loads happen in Step 3, AFTER the barrier. Only scalar register
    buffer_loads (scale loads) are outstanding by this point, and they complete well before
    s_barrier is reached (Step 1 + Step 2 = ~4000 cycles of compute).
    """
    loop_start, loop_end = find_loop(lines)
    if loop_start is None:
        return lines

    # Find s_barrier inside loop
    barrier_idx = None
    for i in range(loop_start, loop_end + 1):
        if classify(lines[i]) == 'barrier':
            barrier_idx = i
            break
    if barrier_idx is None:
        return lines

    # Look backwards from barrier for ASMSTART..vmcnt..ASMEND block
    removed = 0
    i = barrier_idx - 1
    while i > loop_start:
        if ';;#ASMEND' in lines[i]:
            end_idx = i
            # scan back for ASMSTART
            j = i - 1
            while j > loop_start and ';;#ASMSTART' not in lines[j]:
                j -= 1
            if ';;#ASMSTART' in lines[j]:
                # Check if block contains only vmcnt
                block = [lines[k].strip() for k in range(j+1, end_idx)]
                block = [b for b in block if b]
                if len(block) == 1 and re.match(r's_waitcnt\s+vmcnt\(', block[0]):
                    val = re.search(r'vmcnt\((\d+)\)', block[0]).group(1)
                    print(f"  Phase 7: remove vmcnt({val}) ASMBLOCK at L{j+1}-{end_idx+1} before barrier",
                          file=sys.stderr)
                    del lines[j:end_idx+1]
                    removed += len(range(j, end_idx+1))
                    break
            i = j - 1
        else:
            break

    if removed == 0:
        print("  Phase 7: no vmcnt ASMBLOCK found before barrier", file=sys.stderr)
    return lines


def phase8_move_scale_loads_to_tail(lines):
    """Move 8 scale buffer_load_dword from loop header to loop tail.

    Scale loads are issued at the loop header and used in Step 4 via vmcnt+vmov (Phase 3).
    They complete within ~500-1500 cycles; Step 4 is ~6000 cycles later, so there's no
    actual stall. But issuing them in the header means 8 instructions run serially before
    Step 1's first MFMA. Moving them to the tail alongside the VALU chain allows the GPU
    to execute them concurrently with VALU (different execution units), saving ~8 cycles/iter.
    A prologue copy is inserted before the loop for the first iteration.
    """
    loop_start, loop_end = find_loop(lines)
    if loop_start is None:
        return lines

    all_blocks = find_large_asm_blocks(lines, loop_start, loop_end)
    if not all_blocks:
        return lines
    step1_start = all_blocks[0][0]

    # Collect scale loads from loop header
    scale_loads = []
    header_other = []
    for i in range(loop_start + 1, step1_start):
        if classify(lines[i]) == 'scale_load':
            scale_loads.append(lines[i])
        elif classify(lines[i]) not in ('meta', 'nop'):
            header_other.append(lines[i])

    if not scale_loads:
        print("  Phase 8: no scale loads in header, skip.", file=sys.stderr)
        return lines

    print(f"\nPhase 8 — Move {len(scale_loads)} scale loads from header to tail", file=sys.stderr)

    # Find tail insertion point: after vmcnt+vmov for v141/v145, before s_cbranch
    branch_idx = loop_end
    # Insert scale loads just before s_cbranch (the branch closes the loop)
    insert_tail = branch_idx  # insert AT branch_idx shifts it down

    # 1. Insert prologue before loop (for first iteration)
    loop_s2, _ = find_loop(lines)
    for k, sl in enumerate(scale_loads):
        lines.insert(loop_s2 + k, sl.rstrip('\n') + '\n')
    pro_shift = len(scale_loads)

    # 2. Re-find loop after prologue insertion
    loop_s3, loop_e3 = find_loop(lines)
    ab3 = find_large_asm_blocks(lines, loop_s3, loop_e3)
    step1_s3 = ab3[0][0] if ab3 else step1_start + pro_shift

    # 3. Remove scale loads from new header position
    i = loop_s3 + 1
    while i < step1_s3:
        if classify(lines[i]) == 'scale_load':
            lines.pop(i)
        else:
            i += 1

    # 4. Re-find loop and insert scale loads in tail before branch
    loop_s4, loop_e4 = find_loop(lines)
    branch_i4 = loop_e4  # loop_end is the branch
    for k, sl in enumerate(scale_loads):
        lines.insert(branch_i4 + k, sl.rstrip('\n') + '\n')

    ls, le = find_loop(lines)
    print(f"  Final loop: L{ls+1}-{le+1} ({le-ls+1} lines)", file=sys.stderr)
    return lines


def phase9_remove_tail_vmcnt(lines):
    """Remove vmcnt waits from the loop tail (between lgkmcnt and branch).

    After Phase 3+4, the tail has:  lgkmcnt(0) + vmcnt(20)+vmov(v141) + vmcnt(16)+vmov(v145)
    The scale loads (v165, v169) were issued ~6000+ cycles before, well within their
    ~500-1500 cycle latency. By the tail, they're definitely done. Removing the redundant
    vmcnt waits saves 2 serial instruction slots from the critical tail path.
    """
    loop_start, loop_end = find_loop(lines)
    if loop_start is None:
        return lines

    # Find the lgkmcnt(0) block near the end of the loop
    lgkm_idx = None
    for i in range(loop_end - 1, loop_start, -1):
        if 's_waitcnt lgkmcnt(0)' in lines[i]:
            lgkm_idx = i
            break
    if lgkm_idx is None:
        return lines

    removed = 0
    i = lgkm_idx + 1
    while i < loop_end:
        m = re.match(r'\s*s_waitcnt\s+vmcnt\((\d+)\)\s*$', lines[i])
        if m:
            val = m.group(1)
            print(f"  Phase 9: remove tail vmcnt({val}) at L{i+1}", file=sys.stderr)
            lines.pop(i)
            removed += 1
        else:
            i += 1

    if removed == 0:
        print("  Phase 9: no tail vmcnt found after lgkmcnt", file=sys.stderr)
    else:
        print(f"  Phase 9: removed {removed} tail vmcnt waits", file=sys.stderr)
    return lines


def phase10_merge_asm_blocks(lines):
    """Merge adjacent asm blocks in Steps 3-4 to eliminate gaps.

    The compiler emits Steps 3-4 as 4 separate 8-MFMA asm blocks each, with
    PF instructions (s_mov m0 + buffer_load ... lds) and s_nop between them.
    By removing the ;;#ASMEND / ;;#ASMSTART markers between adjacent blocks,
    the assembler treats them as one continuous stream. The PF and s_nop
    instructions (now inside the merged block) execute alongside MFMAs via
    dual-issue, eliminating the ~5-cycle gaps between blocks.
    """
    loop_start, loop_end = find_loop(lines)
    if loop_start is None:
        return lines

    barrier_idx = find_barrier_in_loop(lines, loop_start, loop_end)
    if barrier_idx is None:
        return lines

    print(f"\nPhase 10 — Merge asm blocks after barrier (Steps 3-4)", file=sys.stderr)

    # Find all ;;#ASMEND / ;;#ASMSTART pairs after the barrier
    # These are the gaps between 8-MFMA blocks in Steps 3-4
    merged = 0
    i = barrier_idx + 1
    while i < loop_end:
        # Look for pattern: ;;#ASMEND followed (possibly with gap) by ;;#ASMSTART
        if ';;#ASMEND' in lines[i]:
            end_idx = i
            # Scan forward for the next ;;#ASMSTART
            j = i + 1
            while j <= loop_end and ';;#ASMSTART' not in lines[j]:
                j += 1
            if j <= loop_end and ';;#ASMSTART' in lines[j]:
                start_idx = j
                # Check that the gap between doesn't contain anything critical
                # (s_barrier, s_cbranch, or lgkmcnt would indicate a Step boundary)
                gap_lines = [lines[k].strip() for k in range(end_idx + 1, start_idx)]
                gap_has_barrier = any('s_barrier' in g for g in gap_lines)
                gap_has_lgkm = any('lgkmcnt' in g for g in gap_lines)
                gap_has_branch = any('s_cbranch' in g for g in gap_lines)

                if not gap_has_barrier and not gap_has_lgkm and not gap_has_branch:
                    # Safe to merge: remove ;;#ASMEND, gap content stays, remove ;;#ASMSTART
                    # Keep the gap content (PF, s_nop, salu) - they'll be inside the merged block
                    lines[end_idx] = ''  # remove ;;#ASMEND
                    lines[start_idx] = ''  # remove ;;#ASMSTART
                    merged += 1
                    i = start_idx + 1
                    continue
            i = j + 1
        else:
            i += 1

    # Clean up empty lines
    lines = [l for l in lines if l != '']

    if merged > 0:
        ls, le = find_loop(lines)
        nops = sum(1 for l in lines[ls:le+1] if 's_nop' in l.strip())
        print(f"  Merged {merged} asm block pairs; {nops} s_nop remain in loop", file=sys.stderr)
    else:
        print("  No blocks merged", file=sys.stderr)

    return lines


def rewrite(input_path, output_path):
    with open(input_path) as f:
        lines = f.readlines()

    lines = phase1_redistribute_pfs(lines)
    lines = phase2_pipeline_header(lines)
    lines = phase3_pipeline_vmov(lines)
    lines = phase4_pipeline_header_valu(lines)
    lines = phase5_move_vmcnt_header(lines)
    lines = phase6_remove_redundant_vmcnt(lines)
    lines = phase7_remove_prebarrier_vmcnt(lines)
    lines = phase8_move_scale_loads_to_tail(lines)
    lines = phase10_merge_asm_blocks(lines)
    # lines = phase9_remove_tail_vmcnt(lines)  # BAD: causes non-determinism

    with open(output_path, 'w') as f:
        f.writelines(lines)
    print(f"\nWritten to {output_path}", file=sys.stderr)


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'mxfp4_gluon_cpp_device.s'
    out = sys.argv[2] if len(sys.argv) > 2 else 'mxfp4_gluon_cpp_opt.s'
    rewrite(inp, out)
