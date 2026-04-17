"""R20 milestone-1 256^3 correctness gate for the V2 kernel-side consumer.

Builds A/B scale tensors at 256^3 size, encodes them via V1 preshuffle and V2
preshuffle (pack_count=4 for A, pack_count=2 for B), then invokes the
`verify_preshuffle_v2_consumer` device kernel which reads the V2 layout via
b128 / b64 and compares against V1-loaded packs lane-by-lane.

PASS requires zero mismatches.

Run:
    HIP_VISIBLE_DEVICES=1 python3 test_preshuffle_v2_consumer.py
"""

import math
import sys

import torch

import tk_mxfp8_layouts


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


def main(rows=256, k_cols=256):
    if not hasattr(tk_mxfp8_layouts, "verify_preshuffle_v2_consumer"):
        print("FAIL: tk_mxfp8_layouts.verify_preshuffle_v2_consumer is not "
              "compiled in. Build with -DMXFP8_RCR_PRESHUFFLE_V2_ENABLE=1.")
        sys.exit(2)

    torch.manual_seed(0)
    device = "cuda"
    k_blocks = (k_cols + 31) // 32

    A_scale_exp = torch.randint(low=-3, high=4, size=(rows, k_blocks),
                                dtype=torch.int8, device=device)
    B_scale_exp = torch.randint(low=-3, high=4, size=(rows, k_blocks),
                                dtype=torch.int8, device=device)

    v1_a = preshuffle_scale_matrix_mfma16(A_scale_exp)
    v1_b = preshuffle_scale_matrix_mfma16(B_scale_exp)
    v2_a = preshuffle_scale_matrix_mfma16_v2(A_scale_exp, pack_count=4)
    v2_b = preshuffle_scale_matrix_mfma16_v2(B_scale_exp, pack_count=2)

    padded_rows = math.ceil(rows / 32) * 32
    padded_k_blocks = math.ceil(k_blocks / 8) * 8
    row_groups = padded_rows // 32
    num_slabs_a = row_groups // 4
    num_slabs_b = row_groups // 2
    num_kpairs = padded_k_blocks // 8

    print(f"=== Verifying V2 consumer at rows={rows}, k_cols={k_cols} ===")
    print(f"  V1 A shape={tuple(v1_a.shape)} dtype={v1_a.dtype}")
    print(f"  V1 B shape={tuple(v1_b.shape)} dtype={v1_b.dtype}")
    print(f"  V2 A shape={tuple(v2_a.shape)} dtype={v2_a.dtype}  (pc=4)")
    print(f"  V2 B shape={tuple(v2_b.shape)} dtype={v2_b.dtype}  (pc=2)")
    print(f"  num_slabs_a={num_slabs_a}, num_slabs_b={num_slabs_b}, "
          f"num_kpairs={num_kpairs}, padded_k_blocks={padded_k_blocks}")

    mismatches = tk_mxfp8_layouts.verify_preshuffle_v2_consumer(
        v1_a, v1_b, v2_a, v2_b,
        padded_k_blocks, num_slabs_a, num_slabs_b, num_kpairs)

    total_compares = num_slabs_a * num_slabs_b * 64 * num_kpairs * (4 + 2)
    print(f"  Mismatches: {mismatches} / {total_compares}")
    if mismatches == 0:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
