"""R20 milestone-1 byte-equivalence test for preshuffle V2 layout.

Drives `preshuffle_scale_matrix_mfma16_v2` and verifies that, for every
physical (row, k_block) cell, the byte fetched via the V2 lane-offset
formula equals the byte fetched via the V1 lane-offset formula from the
existing `preshuffle_scale_matrix_mfma16` output.

Run:
    python3 verify_preshuffle_v2.py
"""

import math
import sys

import torch

# Reuse the shipped helpers from test_mxfp8_python.py without executing
# its top-level kernel benchmark code.
import importlib.util

THIS_DIR = "/tmp/wt-r20-a/analysis/fp8_gemm/mi350x"
spec = importlib.util.spec_from_file_location("_mxfp8_pytest", f"{THIS_DIR}/test_mxfp8_python.py")
# We cannot just import it because it eagerly runs benchmarks; instead pull
# the two pure-Python helpers we need by re-implementing here. The functions
# below are intentionally identical to the ones in test_mxfp8_python.py.

def encode_scale_matrix_raw(scale_exp):
    raw = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    return torch.where(
        scale_exp == -128,
        torch.full_like(raw, 0xFF, dtype=torch.uint8),
        raw,
    )


def preshuffle_scale_matrix_mfma16(scale_exp):
    rows, k_blocks_local = scale_exp.shape
    padded_rows = math.ceil(rows / 32) * 32
    padded_k_blocks = math.ceil(k_blocks_local / 8) * 8
    raw = torch.full(
        (padded_rows, padded_k_blocks),
        0x7F,
        dtype=torch.uint8,
        device=scale_exp.device,
    )
    raw[:rows, :k_blocks_local] = encode_scale_matrix_raw(scale_exp)
    shuffled = raw.view(padded_rows // 32, 2, 16, padded_k_blocks // 8, 2, 4, 1)
    shuffled = shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    return shuffled.view(padded_rows // 32, padded_k_blocks * 32)


def preshuffle_scale_matrix_mfma16_v2(scale_exp, pack_count):
    if pack_count <= 0:
        raise ValueError("pack_count must be positive")
    rows, k_blocks_local = scale_exp.shape
    padded_rows = math.ceil(rows / 32) * 32
    padded_k_blocks = math.ceil(k_blocks_local / 8) * 8
    row_groups = padded_rows // 32
    if row_groups % pack_count != 0:
        row_groups = math.ceil(row_groups / pack_count) * pack_count
        padded_rows = row_groups * 32
    num_slabs = row_groups // pack_count

    raw = torch.full(
        (padded_rows, padded_k_blocks),
        0x7F,
        dtype=torch.uint8,
        device=scale_exp.device,
    )
    raw[:rows, :k_blocks_local] = encode_scale_matrix_raw(scale_exp)

    kp_count = padded_k_blocks // 8
    rows_view = raw.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)
    shuffled = rows_view.permute(0, 4, 6, 3, 1, 5, 2).contiguous()
    return shuffled.view(num_slabs, pack_count * 32 * padded_k_blocks)


def v1_lane_offset(k_pair, lane_kblk, lane_nonk):
    """Mirrors load_scale_pair_pack_16x128_preshuffled in the kernel."""
    return ((k_pair * 4 + lane_kblk) * 16 + lane_nonk) * 4


def v2_lane_offset(k_pair, lane_kblk, lane_nonk, pack_count):
    """V2 byte offset within a wave-tile slab.

    For pack_count=4: matches the spec
       lane_byte_offset_v2 = lane_kblk*256 + lane_nonk*16
       kpair_off            = k_pair*1024 + lane_byte_offset_v2

    Generalized for any pack_count.
    """
    return k_pair * (pack_count * 256) + lane_kblk * (pack_count * 64) + lane_nonk * (pack_count * 4)


def verify(rows, k_cols, pack_count, *, device="cpu", seed=0):
    torch.manual_seed(seed)
    k_blocks = (k_cols + 31) // 32
    scale_exp = torch.randint(
        low=-3, high=4, size=(rows, k_blocks), dtype=torch.int8, device=device
    )

    v1 = preshuffle_scale_matrix_mfma16(scale_exp)
    v2 = preshuffle_scale_matrix_mfma16_v2(scale_exp, pack_count)

    padded_rows = math.ceil(rows / 32) * 32
    padded_k_blocks = math.ceil(k_blocks / 8) * 8
    row_groups = padded_rows // 32
    # Bring row_groups up to a multiple of pack_count to mirror v2's padding.
    row_groups_v2 = math.ceil(row_groups / pack_count) * pack_count
    num_slabs = row_groups_v2 // pack_count
    kp_count = padded_k_blocks // 8

    # For every (slab, pack, k_pair, lane_kblk, lane_nonk), build the V1 dword
    # and the V2 dword and compare bytewise.
    #
    # V1 fetch: row_group = slab * pack_count + pack; offset =
    # ((k_pair*4+lane_kblk)*16+lane_nonk)*4; read 4 bytes from v1[row_group, offset:offset+4].
    # V2 fetch: slab_base = slab * (pack_count*32*padded_k_blocks);
    # dword_offset = slab_base + v2_lane_offset(...) + pack*4; read 4 bytes from v2.flat.
    flat_v2 = v2.contiguous().view(-1)

    mismatches = 0
    total = 0
    first_mismatch = None
    for slab in range(num_slabs):
        for pack in range(pack_count):
            row_group = slab * pack_count + pack
            if row_group >= row_groups:
                # Padding row_group (no real data); v1 doesn't have this entry,
                # but v2 may have padded with 0x7F. Skip.
                continue
            for k_pair in range(kp_count):
                for lane_kblk in range(4):
                    for lane_nonk in range(16):
                        v1_off = v1_lane_offset(k_pair, lane_kblk, lane_nonk)
                        v1_bytes = v1[row_group, v1_off:v1_off + 4]
                        slab_base = slab * (pack_count * 32 * padded_k_blocks)
                        v2_off = (
                            slab_base
                            + v2_lane_offset(k_pair, lane_kblk, lane_nonk, pack_count)
                            + pack * 4
                        )
                        v2_bytes = flat_v2[v2_off:v2_off + 4]
                        total += 1
                        if not torch.equal(v1_bytes, v2_bytes):
                            if first_mismatch is None:
                                first_mismatch = (
                                    slab, pack, k_pair, lane_kblk, lane_nonk,
                                    v1_bytes.tolist(), v2_bytes.tolist(),
                                )
                            mismatches += 1

    return total, mismatches, first_mismatch


def main():
    cases = [
        # (rows, k_cols, pack_count)
        (256, 256, 4),
        (256, 256, 2),
        (1024, 1024, 4),
        (1024, 1024, 2),
        # Tile sizes that would exercise actual A/B usage.
        (8192, 8192, 4),
        (8192, 8192, 2),
    ]
    failed = False
    for rows, k_cols, pc in cases:
        total, mm, first = verify(rows, k_cols, pc)
        status = "PASS" if mm == 0 else "FAIL"
        msg = f"  rows={rows} k_cols={k_cols} pc={pc}: {status} ({total - mm}/{total} dwords matched)"
        if mm:
            failed = True
            msg += f"\n    first mismatch: {first}"
        print(msg)
    if failed:
        print("\nOVERALL: FAIL")
        sys.exit(1)
    print("\nOVERALL: PASS")


if __name__ == "__main__":
    main()
