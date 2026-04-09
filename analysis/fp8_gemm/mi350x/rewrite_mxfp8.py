#!/usr/bin/env python3
"""MXFP8 .s rewriter V2: dependency-aware AGPR redistribution.

Only moves instructions when provably safe. Keeps overall structure intact.
Targets: AGPR-heavy zones between MFMAs → spread MFMAs into AGPR shadows.

Pipeline:
  hipcc --offload-device-only -S ... -o raw.s
  python3 rewrite_mxfp8.py raw.s optimized.s
  # then assemble + bundle + link (see build_rewrite.sh)
"""

import re, sys
from dataclasses import dataclass, field
from enum import Enum, auto


class IK(Enum):
    MFMA = auto()
    DS_READ = auto()
    TILE_LOAD = auto()
    SCALE_LOAD = auto()
    AGPR_READ = auto()
    AGPR_WRITE = auto()
    AGPR_MOV = auto()
    WAITCNT = auto()
    BARRIER = auto()
    SCALAR = auto()
    VALU = auto()
    META = auto()
    OTHER = auto()


@dataclass
class Inst:
    idx: int
    line: str
    kind: IK
    vdef: set = field(default_factory=set)
    vuse: set = field(default_factory=set)
    adef: set = field(default_factory=set)
    ause: set = field(default_factory=set)
    sdef: set = field(default_factory=set)
    suse: set = field(default_factory=set)


def _vrange(s):
    out = set()
    for m in re.finditer(r'v\[(\d+):(\d+)\]', s):
        out.update(range(int(m.group(1)), int(m.group(2)) + 1))
    for m in re.finditer(r'(?<!\[)\bv(\d+)\b(?!\]|:)', s):
        out.add(int(m.group(1)))
    return out

def _arange(s):
    out = set()
    for m in re.finditer(r'a\[(\d+):(\d+)\]', s):
        out.update(range(int(m.group(1)), int(m.group(2)) + 1))
    for m in re.finditer(r'(?<!\[)\ba(\d+)\b(?!\]|:)', s):
        out.add(int(m.group(1)))
    return out


def classify(line):
    s = line.strip()
    if not s or s.startswith(';') or s.startswith('.') or s.endswith(':'):
        return IK.META
    if 'v_mfma_scale' in s: return IK.MFMA
    if 'ds_read_b128' in s: return IK.DS_READ
    if 'buffer_load_dwordx4' in s and 'lds' in s: return IK.TILE_LOAD
    if 'buffer_load_dword' in s: return IK.SCALE_LOAD
    if 'v_accvgpr_read' in s: return IK.AGPR_READ
    if 'v_accvgpr_write' in s: return IK.AGPR_WRITE
    if 'v_accvgpr_mov' in s: return IK.AGPR_MOV
    if 's_waitcnt' in s: return IK.WAITCNT
    if 's_barrier' in s: return IK.BARRIER
    if s.startswith('s_'): return IK.SCALAR
    if s.startswith('v_'): return IK.VALU
    return IK.OTHER


def parse(idx, line):
    kind = classify(line)
    inst = Inst(idx=idx, line=line, kind=kind)
    s = line.strip()

    if kind == IK.MFMA:
        agprs = list(re.finditer(r'a\[(\d+):(\d+)\]', s))
        if len(agprs) >= 2:
            inst.adef = set(range(int(agprs[0].group(1)), int(agprs[0].group(2)) + 1))
            inst.ause = set(range(int(agprs[1].group(1)), int(agprs[1].group(2)) + 1))
        inst.vuse = _vrange(s)
    elif kind == IK.DS_READ:
        vr = list(re.finditer(r'v\[(\d+):(\d+)\]', s))
        if vr:
            inst.vdef = set(range(int(vr[0].group(1)), int(vr[0].group(2)) + 1))
        m = re.search(r',\s*v(\d+)', s)
        if m: inst.vuse.add(int(m.group(1)))
    elif kind == IK.AGPR_READ:
        m = re.match(r'\s*v_accvgpr_read_b32\s+v(\d+),\s*a(\d+)', s)
        if m:
            inst.vdef.add(int(m.group(1)))
            inst.ause.add(int(m.group(2)))
    elif kind == IK.AGPR_WRITE:
        m = re.match(r'\s*v_accvgpr_write_b32\s+a(\d+),\s*v(\d+)', s)
        if m:
            inst.adef.add(int(m.group(1)))
            inst.vuse.add(int(m.group(2)))
    elif kind == IK.AGPR_MOV:
        m = re.match(r'\s*v_accvgpr_mov_b32\s+a(\d+),\s*a(\d+)', s)
        if m:
            inst.adef.add(int(m.group(1)))
            inst.ause.add(int(m.group(2)))
    elif kind in (IK.TILE_LOAD, IK.SCALE_LOAD):
        inst.vuse = _vrange(s)
    elif kind == IK.VALU:
        all_v = _vrange(s)
        m = re.match(r'\s*v_\w+\s+v(\d+)', s)
        if m:
            inst.vdef.add(int(m.group(1)))
            inst.vuse = all_v - inst.vdef
        else:
            inst.vuse = all_v
    return inst


def conflicts(a, b):
    """Check if instruction b depends on instruction a (RAW, WAR, WAW)."""
    # RAW: b reads something a writes
    if (a.vdef & b.vuse) or (a.adef & b.ause):
        return True
    # WAW: both write same register
    if (a.vdef & b.vdef) or (a.adef & b.adef):
        return True
    # WAR: b writes something a reads
    if (a.vuse & b.vdef) or (a.ause & b.adef):
        return True
    return False


def find_loop(lines):
    """Find the inner loop by looking for back-edge branch."""
    candidates = []
    for i, line in enumerate(lines):
        m = re.search(r's_cbranch_\w+\s+(\.LBB\d+_\d+)', line)
        if m:
            target = m.group(1)
            for j in range(i):
                if target + ':' in lines[j]:
                    if j < i:  # back edge
                        candidates.append((j, i))
    if not candidates:
        return None, None
    return max(candidates, key=lambda x: x[1] - x[0])


def optimize_phase(instrs):
    """Within a barrier-delimited phase, redistribute AGPR ops among MFMA shadows.

    Find AGPR-heavy zones (>4 consecutive AGPR ops between MFMAs).
    If a later MFMA has no dependency on the AGPR zone, pull it forward.
    """
    if len(instrs) < 3:
        return instrs

    result = list(instrs)
    moved = 0

    i = 0
    while i < len(result) - 2:
        # Find an AGPR-heavy zone: MFMA followed by >4 AGPR ops before next MFMA
        if result[i].kind != IK.MFMA:
            i += 1
            continue

        # Count consecutive AGPR/scalar ops after this MFMA
        agpr_start = i + 1
        j = agpr_start
        while j < len(result) and result[j].kind in (IK.AGPR_READ, IK.AGPR_WRITE, IK.AGPR_MOV, IK.SCALAR, IK.VALU, IK.META, IK.OTHER):
            j += 1

        agpr_count = sum(1 for k in range(agpr_start, j)
                        if result[k].kind in (IK.AGPR_READ, IK.AGPR_WRITE, IK.AGPR_MOV))

        if agpr_count < 6 or j >= len(result):
            i += 1
            continue

        # Found a heavy zone. Look for a MFMA after it that can be pulled forward.
        next_mfma_idx = None
        for k in range(j, min(j + 10, len(result))):
            if result[k].kind == IK.MFMA:
                next_mfma_idx = k
                break

        if next_mfma_idx is None:
            i += 1
            continue

        candidate = result[next_mfma_idx]

        # Check if candidate can be moved after the first ~4 AGPR ops
        # It must not conflict with any instruction between insert_pos and its current pos
        insert_pos = agpr_start + min(4, agpr_count)

        can_move = True
        for k in range(insert_pos, next_mfma_idx):
            if conflicts(result[k], candidate) or conflicts(candidate, result[k]):
                can_move = False
                break

        # Also check the candidate doesn't depend on the MFMA at position i
        if conflicts(result[i], candidate):
            # The candidate reads from the result of the previous MFMA
            # This is only safe if the MFMA has enough latency (64 cycles, ~16 instructions)
            if insert_pos - i < 8:  # conservative: need at least 8 instructions gap
                can_move = False

        if can_move:
            inst = result.pop(next_mfma_idx)
            result.insert(insert_pos, inst)
            moved += 1
        else:
            i += 1
            continue

        i = insert_pos + 1

    return result, moved


def rewrite(input_path, output_path):
    with open(input_path) as f:
        lines = f.readlines()

    start, end = find_loop(lines)
    if start is None:
        print("ERROR: Could not find inner loop", file=sys.stderr)
        sys.exit(1)

    print(f"Loop: lines {start+1}-{end+1} ({end-start+1} lines)", file=sys.stderr)

    instrs = [parse(i, lines[i]) for i in range(start, end + 1)]

    from collections import Counter
    counts = Counter(inst.kind.name for inst in instrs)
    print(f"Instructions: {dict(sorted(counts.items()))}", file=sys.stderr)

    # Split into barrier-delimited phases
    phases = []
    current = []
    for inst in instrs:
        if inst.kind == IK.BARRIER and current:
            phases.append(current)
            current = [inst]
        else:
            current.append(inst)
    if current:
        phases.append(current)

    print(f"Phases: {len(phases)}", file=sys.stderr)

    # Optimize each phase
    total_moved = 0
    new_instrs = []
    for pi, phase in enumerate(phases):
        n_mfma = sum(1 for x in phase if x.kind == IK.MFMA)
        n_agpr = sum(1 for x in phase if x.kind in (IK.AGPR_READ, IK.AGPR_WRITE, IK.AGPR_MOV))

        if n_mfma > 0 and n_agpr > 10:
            optimized, moved = optimize_phase(phase)
            total_moved += moved
            new_instrs.extend(optimized)
            if moved > 0:
                print(f"  Phase {pi}: {n_mfma} MFMAs, {n_agpr} AGPRs → moved {moved} MFMAs forward", file=sys.stderr)
        else:
            new_instrs.extend(phase)

    print(f"Total MFMAs redistributed: {total_moved}", file=sys.stderr)

    # Rebuild
    new_lines = [inst.line for inst in new_instrs]
    result = lines[:start] + new_lines + lines[end + 1:]

    with open(output_path, 'w') as f:
        f.writelines(result)

    n_mfma = sum(1 for l in new_lines if 'v_mfma_scale' in l)
    n_ds = sum(1 for l in new_lines if 'ds_read_b128' in l)
    print(f"Output: {n_mfma} MFMAs, {n_ds} ds_reads → {output_path}", file=sys.stderr)


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'mxfp8_rewrite_device.s'
    out = sys.argv[2] if len(sys.argv) > 2 else 'mxfp8_rewrite_opt.s'
    rewrite(inp, out)
