#!/usr/bin/env python3
"""
gen_monolithic_kstep.py — Post-compile ISA optimizer for MXFP4 Gluon C++ kernel.

Reads the compiler-generated .s file, optimizes the main K-loop by:
  1. Stripping all ;;#ASMSTART / ;;#ASMEND markers (compiler inline-asm boundaries)
  2. Removing all s_nop 0 instructions (compiler-inserted no-ops)
  3. Moving 8 buffer_load_dword (scale loads) from each K-step's loop header
     INTO the last 8 MFMA slots of the previous K-step's Step 4, interleaved 1:1
  4. For the first K-step's scale loads: software pipelining via back-edge
     (duplicate into last 8 MFMAs) + prologue copy for first pass
  5. Moving s_add_i32 (scale offset computations) into free MFMA slots for
     SALU+MFMA dual-issue

Then assembles/links/bundles to produce the final .so.
"""

import re
import subprocess
import sys
import os

WORKDIR = "/shared_nfs/kyle/HipKittens/analysis/fp8_gemm/mi350x"
SRC_S = os.path.join(WORKDIR, "kernel_mxfp4_gluon_cpp-hip-amdgcn-amd-amdhsa-gfx950.s")
OPT_S = os.path.join(WORKDIR, "mxfp4_gluon_cpp_opt.s")
HOST_O = os.path.join(WORKDIR, "kernel_mxfp4_gluon_cpp-host-x86_64-unknown-linux-gnu.o")
OUT_SO = os.path.join(WORKDIR, "tk_mxfp4_gluon_cpp.cpython-310-x86_64-linux-gnu.so")

LLVM = "/opt/rocm/lib/llvm/bin"
HIPCC = "/opt/rocm/bin/hipcc"


def read_lines(path):
    with open(path, "r") as f:
        return f.readlines()


def write_lines(path, lines):
    with open(path, "w") as f:
        f.writelines(lines)


def is_asmstart(line):
    return line.strip() == ";;#ASMSTART"


def is_asmend(line):
    return line.strip() == ";;#ASMEND"


def is_snop0(line):
    return line.strip() == "s_nop 0"


def is_mfma(line):
    return "v_mfma_scale_f32_16x16x128_f8f6f4" in line


def is_buffer_load_dword_single(line):
    """Match buffer_load_dword (single-dword scale loads), NOT buffer_load_dwordx4."""
    s = line.strip()
    return s.startswith("buffer_load_dword ") and "buffer_load_dwordx4" not in s


def is_s_add_i32(line):
    return line.strip().startswith("s_add_i32 ")


def is_blank_or_noise(line):
    s = line.strip()
    return s == "" or s == ";;#ASMSTART" or s == ";;#ASMEND" or s == "s_nop 0"


def find_loop_bounds(lines):
    """Find the main loop: .LBB0_1 to s_cbranch_scc0 .LBB0_1"""
    loop_start = None
    loop_end = None
    for i, line in enumerate(lines):
        if ".LBB0_1:" in line:
            loop_start = i
        if "s_cbranch_scc0 .LBB0_1" in line:
            loop_end = i
    return loop_start, loop_end


def find_scale_load_groups(loop_lines):
    """Find all groups of exactly 8 consecutive buffer_load_dword (scale loads)."""
    groups = []
    i = 0
    while i < len(loop_lines):
        if is_buffer_load_dword_single(loop_lines[i]):
            group_start = i
            group = []
            while i < len(loop_lines) and is_buffer_load_dword_single(loop_lines[i]):
                group.append((i, loop_lines[i]))
                i += 1
            if len(group) == 8:
                # Look backwards for associated s_add_i32 (scale offset computation)
                pre_salu = []
                j = group_start - 1
                while j >= 0 and j >= group_start - 5:
                    if is_s_add_i32(loop_lines[j]):
                        pre_salu.append((j, loop_lines[j]))
                    elif not is_blank_or_noise(loop_lines[j]):
                        break
                    j -= 1
                # Look forward for s_add_i32 right after the group
                fwd_salu = []
                j = group[7][0] + 1
                while j < len(loop_lines) and j <= group[7][0] + 5:
                    l = loop_lines[j].strip()
                    if is_s_add_i32(loop_lines[j]):
                        fwd_salu.append((j, loop_lines[j]))
                    elif not is_blank_or_noise(loop_lines[j]) and not l.startswith("s_mov_b32"):
                        break
                    j += 1
                groups.append({
                    'start': group_start,
                    'lines': [g[1] for g in group],
                    'indices': [g[0] for g in group],
                    'pre_salu': pre_salu,
                    'post_salu': fwd_salu,
                })
        else:
            i += 1
    return groups


def find_last_n_mfmas(loop_lines, before_idx, n):
    """Find the last N MFMAs before before_idx, returned in chronological order."""
    positions = []
    j = before_idx - 1
    while j >= 0 and len(positions) < n:
        if is_mfma(loop_lines[j]):
            positions.append(j)
        j -= 1
    return list(reversed(positions))


def main():
    print("=" * 70)
    print("MXFP4 Gluon C++ Kernel ISA Optimizer")
    print("=" * 70)

    # Step 1: Read the .s file
    print(f"\n[1] Reading {SRC_S}")
    lines = read_lines(SRC_S)
    print(f"  {len(lines)} lines")

    # Step 2: Find the main loop
    print("\n[2] Finding main loop bounds")
    loop_start, loop_end = find_loop_bounds(lines)
    if loop_start is None or loop_end is None:
        print("  ERROR: Could not find main loop!")
        sys.exit(1)
    print(f"  Loop: lines {loop_start + 1} to {loop_end + 1} ({loop_end - loop_start + 1} lines)")

    loop_lines = lines[loop_start:loop_end + 1]

    n_asmstart = sum(1 for l in loop_lines if is_asmstart(l))
    n_snop = sum(1 for l in loop_lines if is_snop0(l))
    n_mfma = sum(1 for l in loop_lines if is_mfma(l))
    n_buf = sum(1 for l in loop_lines if is_buffer_load_dword_single(l))
    print(f"  {n_asmstart} ASMSTART/END, {n_snop} s_nop 0")
    print(f"  {n_mfma} MFMAs, {n_buf} buffer_load_dword (scale loads)")

    # Step 3: Find scale load groups
    print("\n[3] Finding scale load groups")
    groups = find_scale_load_groups(loop_lines)
    print(f"  Found {len(groups)} groups of 8 buffer_load_dword")

    # Step 4: Plan scale load optimization
    print("\n[4] Planning scale load moves")

    indices_to_remove = set()
    insertions = {}  # loop_line_idx -> [lines to insert after]
    prologue_additions = []

    all_mfma_indices = [i for i, l in enumerate(loop_lines) if is_mfma(l)]

    for sg_idx, sg in enumerate(groups):
        group_start = sg['start']
        last_8 = find_last_n_mfmas(loop_lines, group_start, 8)

        if len(last_8) < 8:
            # Group 0: back-edge optimization
            print(f"  Group {sg_idx}: back-edge (last 8 MFMAs of loop)")
            last_8_loop = all_mfma_indices[-8:]

            for idx in sg['indices']:
                indices_to_remove.add(idx)
            salu_lines = []
            for idx, line in sg['pre_salu']:
                indices_to_remove.add(idx)
                salu_lines.append(line)
            for idx, line in sg['post_salu']:
                indices_to_remove.add(idx)
                salu_lines.append(line)

            for k in range(8):
                mfma_idx = last_8_loop[k]
                if mfma_idx not in insertions:
                    insertions[mfma_idx] = []
                insertions[mfma_idx].append(sg['lines'][k])

            if salu_lines:
                earlier = find_last_n_mfmas(loop_lines, last_8_loop[0], len(salu_lines))
                for k, salu_line in enumerate(salu_lines):
                    if k < len(earlier):
                        mid = earlier[k]
                        if mid not in insertions:
                            insertions[mid] = []
                        insertions[mid].append(salu_line)

            # Add to prologue for first pass
            for _, salu_line in sg['pre_salu']:
                prologue_additions.append(salu_line)
            for line in sg['lines']:
                prologue_additions.append(line)
            for _, salu_line in sg['post_salu']:
                prologue_additions.append(salu_line)
        else:
            for idx in sg['indices']:
                indices_to_remove.add(idx)
            salu_lines = []
            for idx, line in sg['pre_salu']:
                indices_to_remove.add(idx)
                salu_lines.append(line)
            for idx, line in sg['post_salu']:
                indices_to_remove.add(idx)
                salu_lines.append(line)

            for k in range(8):
                mfma_idx = last_8[k]
                if mfma_idx not in insertions:
                    insertions[mfma_idx] = []
                insertions[mfma_idx].append(sg['lines'][k])

            if salu_lines:
                earlier = find_last_n_mfmas(loop_lines, last_8[0], len(salu_lines))
                for k, salu_line in enumerate(salu_lines):
                    if k < len(earlier):
                        mid = earlier[k]
                        if mid not in insertions:
                            insertions[mid] = []
                        insertions[mid].append(salu_line)

    print(f"  Removing {len(indices_to_remove)} lines, inserting after {len(insertions)} MFMA positions")
    print(f"  Adding {len(prologue_additions)} lines to prologue")

    # Step 5: Rebuild the loop
    print("\n[5] Rebuilding loop body")
    new_loop = []
    for i, line in enumerate(loop_lines):
        if i in indices_to_remove:
            continue
        if is_asmstart(line) or is_asmend(line):
            continue
        if is_snop0(line):
            continue
        new_loop.append(line)
        if i in insertions:
            for ins_line in insertions[i]:
                ins = ins_line.strip()
                if ins:
                    new_loop.append("\t" + ins + "\n")

    # Step 6: Rebuild full file
    print("\n[6] Rebuilding file")
    new_lines = []
    for line in lines[:loop_start]:
        if is_asmstart(line) or is_asmend(line):
            continue
        if is_snop0(line):
            continue
        new_lines.append(line)

    for pl in prologue_additions:
        ins = pl.strip()
        if ins:
            new_lines.append("\t" + ins + "\n")

    new_lines.extend(new_loop)

    for line in lines[loop_end + 1:]:
        if is_asmstart(line) or is_asmend(line):
            continue
        if is_snop0(line):
            continue
        new_lines.append(line)

    # Verify
    new_loop_start, new_loop_end = find_loop_bounds(new_lines)
    if new_loop_start and new_loop_end:
        new_body = new_lines[new_loop_start:new_loop_end + 1]
        new_n_mfma = sum(1 for l in new_body if is_mfma(l))
        new_n_buf = sum(1 for l in new_body if is_buffer_load_dword_single(l))
        print(f"\n  After optimization:")
        print(f"  {new_n_mfma} MFMAs, {new_n_buf} buffer_load_dword")
        print(f"  Loop: {new_loop_end - new_loop_start + 1} lines (was {loop_end - loop_start + 1})")

    # Step 7: Write
    print(f"\n[7] Writing {OPT_S}")
    write_lines(OPT_S, new_lines)
    print(f"  {len(new_lines)} lines")

    # Step 8: Assemble
    print("\n[8] Assembling")
    r = subprocess.run([
        f"{LLVM}/clang", "-cc1as",
        "-triple", "amdgcn-amd-amdhsa",
        "-target-cpu", "gfx950",
        "-filetype", "obj",
        "-o", "/tmp/gluon_opt.o",
        OPT_S
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED:\n{r.stderr}")
        sys.exit(1)
    print("  OK")

    # Step 9: Link
    print("\n[9] Linking")
    r = subprocess.run([
        f"{LLVM}/ld.lld",
        "--no-undefined", "-shared",
        "-plugin-opt=-amdgpu-internalize-symbols",
        "-plugin-opt=mcpu=gfx950",
        "-o", "/tmp/gluon_opt.hsaco",
        "/tmp/gluon_opt.o"
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED:\n{r.stderr}")
        sys.exit(1)
    print("  OK")

    # Step 10: Bundle
    print("\n[10] Bundling")
    r = subprocess.run([
        f"{LLVM}/clang-offload-bundler",
        "-type=o",
        "-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx950",
        f"-input={HOST_O}",
        "-input=/tmp/gluon_opt.hsaco",
        "-output=/tmp/gluon_bundled.o"
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED:\n{r.stderr}")
        sys.exit(1)
    print("  OK")

    # Step 11: Final link
    print("\n[11] Final link")
    r = subprocess.run(["python3-config", "--ldflags"], capture_output=True, text=True)
    ldflags = r.stdout.strip().replace("-lcrypt", "")
    r = subprocess.run(
        f"{HIPCC} /tmp/gluon_bundled.o -shared -o {OUT_SO} {ldflags}",
        shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED:\n{r.stderr}")
        sys.exit(1)
    print(f"  OK -> {OUT_SO}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
