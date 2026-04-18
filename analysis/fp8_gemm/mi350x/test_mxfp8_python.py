import json
import math
import os
import random
import sys

import torch

torch.manual_seed(0)
random.seed(0)

import tk_mxfp8_layouts


def parse_problem_size(argv):
    if len(argv) == 2:
        n = int(argv[1])
        return n, n, n
    if len(argv) == 4:
        return int(argv[1]), int(argv[2]), int(argv[3])
    return 8192, 8192, 8192


M, N, K = parse_problem_size(sys.argv)
build_M = int(os.environ.get("MXFP8_BUILD_M", str(M)))
build_N = int(os.environ.get("MXFP8_BUILD_N", str(N)))
build_K = int(os.environ.get("MXFP8_BUILD_K", str(K)))

if build_M < M or build_N < N or build_K < K:
    raise ValueError(
        f"Build shape ({build_M}, {build_N}, {build_K}) must cover "
        f"problem shape ({M}, {N}, {K})"
    )

num_warmup = int(os.environ.get("MXFP8_WARMUP", "10"))
num_iters = int(os.environ.get("MXFP8_ITERS", "20"))
determinism_runs = max(1, int(os.environ.get("MXFP8_DETERMINISM_RUNS", "1")))
snr_threshold_db = float(os.environ.get("MXFP8_SNR_THRESHOLD_DB", "48.0"))
use_preshuffle_quant = os.environ.get("MXFP8_PRESHUFFLE_QUANT", "0") != "0"
# R21 milestone-2 V2-RCR runtime gate: when set, RCR-PQ uses the wave-tile
# reordered V2 layout (option a) so the kernel can issue buffer_load_b128 /
# buffer_load_b64 for scales.
use_v2_rcr = os.environ.get("MXFP8_RCR_PRESHUFFLE_V2_RUNTIME", "1") != "0"
# R22 milestone-1 V2-RRR runtime gate: when set, RRR-PQ uses the same V2
# wave-tile reordered scale layout as RCR. Default ON after R22-A 5x A/B
# benchmark passed (+168.64 TFLOPS / +5.94% / Welch t=4.76).
use_v2_rrr = os.environ.get("MXFP8_RRR_PRESHUFFLE_V2_RUNTIME", "1") != "0"
requested_layouts = {
    layout.strip().lower()
    for layout in os.environ.get("MXFP8_LAYOUTS", "rcr,rrr,crr").split(",")
    if layout.strip()
}
check_results = os.environ.get("MXFP8_CHECK", "1") != "0"

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = 2 * M * N * K
k_blocks = (build_K + 31) // 32


def default_output_path():
    suffix = "_pq" if use_preshuffle_quant else ""
    if M == N == K:
        return f"mxfp8_layout_results_{M}{suffix}.json"
    return f"mxfp8_layout_results_{M}x{N}x{K}{suffix}.json"


def generate_fp8_matrix(total_rows, total_cols, valid_rows, valid_cols):
    x = torch.zeros(total_rows, total_cols, dtype=torch.float32, device="cuda")
    x[:valid_rows, :valid_cols] = (
        torch.randn(valid_rows, valid_cols, dtype=torch.float32, device="cuda") * 0.05
    )
    return x.to(torch.float8_e4m3fn)


def generate_scale_matrix(total_rows, total_k_blocks, valid_rows):
    # MXFP8 E8M0 is modeled as an int8 power-of-two exponent.
    scales = torch.zeros(total_rows, total_k_blocks, dtype=torch.int8, device="cuda")
    scales[:valid_rows, :] = torch.randint(
        low=-2,
        high=3,
        size=(valid_rows, total_k_blocks),
        dtype=torch.int8,
        device="cuda",
    )
    return scales


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
    """R20 V2 layout: interleave `pack_count` consecutive row_groups within
    each wave-tile slab so that the per-(lane, k_pair) dword group becomes
    contiguous, enabling buffer_load_b{64,128} on the kernel side.

    Per-slab byte order (outer -> inner stride):
        slab(pc*32 rows) -> k_pair(K/8) -> lane_kblk(4) -> lane_nonk(16)
            -> pack(pack_count) -> byte_in_dword(4 = k_phase_lo*2 + half)

    Strides (bytes), for parameter pack_count = PC:
        slab            = PC * 32 * padded_k_blocks
        k_pair          = PC * 256
        lane_kblk       =  PC *  64
        lane_nonk       =  PC *   4
        pack            =        4
        byte_in_dword   =        1   (encodes (k_phase_lo, half) like V1)

    The 4 bytes within one (slab, k_pair, lane_kblk, lane_nonk, pack)
    dword are identical to V1's 32-bit pack for the corresponding
    physical row_group = slab * pack_count + pack.

    Consumer can issue a single buffer_load_b{32 * PC} at byte offset
        slab_base + k_pair * (PC*256) + lane_kblk * (PC*64) + lane_nonk * (PC*4)
    to retrieve all `pack_count` dwords for the wave-tile in one VMEM op.
    For PC=4 this matches the R19 spec (k_pair*1024 + lane_kblk*256 +
    lane_nonk*16); for PC=2 it yields k_pair*512 + lane_kblk*128 +
    lane_nonk*8 enabling buffer_load_b64.
    """
    if pack_count <= 0:
        raise ValueError("pack_count must be positive")
    rows, k_blocks_local = scale_exp.shape
    padded_rows = math.ceil(rows / 32) * 32
    padded_k_blocks = math.ceil(k_blocks_local / 8) * 8
    row_groups = padded_rows // 32
    if row_groups % pack_count != 0:
        # Pad row_groups up to a multiple of pack_count.
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

    # raw[r, k] with r = (slab * PC + pack) * 32 + h * 16 + lane_nonk
    #              and k = k_pair * 8 + k_phase_lo * 4 + lane_kblk
    # View raw as [num_slabs, PC, 2(h), 16(ln), num_kpairs, 2(kp), 4(lk)].
    kp_count = padded_k_blocks // 8
    rows_view = raw.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)

    # Permute so byte ordering becomes:
    #   [slab, kpair, lk, ln, pack, kp, h]
    # which gives the V2 strides described above.
    shuffled = rows_view.permute(0, 4, 6, 3, 1, 5, 2).contiguous()

    # Per-slab flat byte count:
    #   PC * 32 (rows) * padded_k_blocks (cols) bytes
    return shuffled.view(num_slabs, pack_count * 32 * padded_k_blocks)


def preshuffle_scale_matrix_mfma16_v2_rcr_a(scale_exp,
                                            blk=256, hb=128, rbm=64, warps_m=2):
    """R21 milestone-2 V2-RCR-A layout (option a wave-tile reorder).

    Repacks the A-side scale matrix so that each (br, wm) wave-tile owns one
    contiguous slab of pack_count=4 row_groups in the order
    {a0p0, a1p0, a0p1, a1p1}. This matches the kernel's pack-load order
    (a0_scale_packs[i], a1_scale_packs[i] alternated as pack_idx=0,1) and
    makes the four dwords needed per (k_pair, lane) contiguous in memory,
    enabling a single buffer_load_b128 to fetch them all.

    Layout: per-slab byte order is identical to V2 with PC=4, namely
        slab(4*32 rows) -> k_pair -> lane_kblk -> lane_nonk -> pack(4) ->
            byte_in_dword(4 = k_phase_lo*2 + half).
    Slab indexing: slab(br, wm) = br * warps_m + wm.

    The four 32-byte dwords inside slab(br, wm) at any (k_pair, lane) are
    the V1 packs for physical row_groups
        a0p0 = (br*BLK + 0  + wm*RBM + 0 ) >> 5
        a1p0 = (br*BLK + HB + wm*RBM + 0 ) >> 5
        a0p1 = (br*BLK + 0  + wm*RBM + 32) >> 5
        a1p1 = (br*BLK + HB + wm*RBM + 32) >> 5
    in that 'pack' index order (0..3).
    """
    pack_count = 4
    rows, k_blocks_local = scale_exp.shape
    padded_rows = math.ceil(rows / blk) * blk
    padded_k_blocks = math.ceil(k_blocks_local / 8) * 8
    if padded_rows % blk:
        raise ValueError("padded_rows must be multiple of blk")
    num_ctiles = padded_rows // blk
    num_slabs = num_ctiles * warps_m

    # Build raw per-CTA, per-wm wave-tile permutation.  rgs_per_ctile = blk/32
    rgs_per_ctile = blk // 32  # 8 with blk=256
    pack_a = rbm // 32  # 2 with rbm=64

    raw = torch.full(
        (padded_rows, padded_k_blocks),
        0x7F,
        dtype=torch.uint8,
        device=scale_exp.device,
    )
    raw[:rows, :k_blocks_local] = encode_scale_matrix_raw(scale_exp)

    # Build a contiguous tensor whose row order is the wave-tile permutation.
    perm_rows = torch.empty_like(raw)
    rg_view = raw.view(num_ctiles, rgs_per_ctile, 32, padded_k_blocks)
    perm_view = perm_rows.view(num_ctiles, num_slabs // num_ctiles,
                               pack_count, 32, padded_k_blocks)
    # For each (br, wm) build the [a0p0, a1p0, a0p1, a1p1] slab.
    for wm in range(warps_m):
        rg_base = wm * (rbm // 32)  # in row_groups within the ctile
        rg_hi = (hb // 32) + rg_base  # row offset for half=1
        for pidx in range(pack_a):
            # pack index 2*pidx -> a0p<pidx>; 2*pidx+1 -> a1p<pidx>
            perm_view[:, wm, 2 * pidx, :, :]     = rg_view[:, rg_base + pidx, :, :]
            perm_view[:, wm, 2 * pidx + 1, :, :] = rg_view[:, rg_hi   + pidx, :, :]

    # Now perm_rows is shaped (num_slabs * pack_count * 32, padded_k_blocks)
    # with row order matching V2 slab+pack ordering. Apply the same byte
    # reshuffle as preshuffle_scale_matrix_mfma16_v2 (PC=4).
    kp_count = padded_k_blocks // 8
    rows_view = perm_rows.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)
    shuffled = rows_view.permute(0, 4, 6, 3, 1, 5, 2).contiguous()
    return shuffled.view(num_slabs, pack_count * 32 * padded_k_blocks)


def preshuffle_scale_matrix_mfma16_v2_rcr_b(scale_exp,
                                            blk=256, hb=128, rbn=32, warps_n=4):
    """R21 milestone-2 V2-RCR-B layout (option a wave-tile reorder).

    Repacks the B-side scale matrix so that each (bc, wn) wave-tile owns one
    contiguous slab of pack_count=2 row_groups in the order {b0p0, b1p0}.
    This matches the kernel's iteration order (b0_scale_packs[0],
    b1_scale_packs[0]) and makes the two dwords contiguous, enabling
    buffer_load_b64.

    Slab indexing: slab(bc, wn) = bc * warps_n + wn.
    The two dwords inside slab(bc, wn) are the V1 packs for row_groups
        b0p0 = (bc*BLK + 0  + wn*RBN) >> 5
        b1p0 = (bc*BLK + HB + wn*RBN) >> 5
    in that 'pack' index order (0,1).
    """
    pack_count = 2
    rows, k_blocks_local = scale_exp.shape
    padded_rows = math.ceil(rows / blk) * blk
    padded_k_blocks = math.ceil(k_blocks_local / 8) * 8
    if padded_rows % blk:
        raise ValueError("padded_rows must be multiple of blk")
    num_ctiles = padded_rows // blk
    num_slabs = num_ctiles * warps_n
    rgs_per_ctile = blk // 32  # 8 with blk=256

    raw = torch.full(
        (padded_rows, padded_k_blocks),
        0x7F,
        dtype=torch.uint8,
        device=scale_exp.device,
    )
    raw[:rows, :k_blocks_local] = encode_scale_matrix_raw(scale_exp)

    perm_rows = torch.empty_like(raw)
    rg_view = raw.view(num_ctiles, rgs_per_ctile, 32, padded_k_blocks)
    perm_view = perm_rows.view(num_ctiles, warps_n, pack_count, 32,
                               padded_k_blocks)
    rbn_rg = rbn // 32  # 1 with rbn=32
    rg_hi_offset = hb // 32  # 4 with hb=128
    for wn in range(warps_n):
        rg_base = wn * rbn_rg
        perm_view[:, wn, 0, :, :] = rg_view[:, rg_base, :, :]
        perm_view[:, wn, 1, :, :] = rg_view[:, rg_base + rg_hi_offset, :, :]

    kp_count = padded_k_blocks // 8
    rows_view = perm_rows.view(num_slabs, pack_count, 2, 16, kp_count, 2, 4)
    shuffled = rows_view.permute(0, 4, 6, 3, 1, 5, 2).contiguous()
    return shuffled.view(num_slabs, pack_count * 32 * padded_k_blocks)


def expand_row_scales(scale_exp, cols):
    return torch.pow(2.0, scale_exp.float()).repeat_interleave(32, dim=1)[:, :cols]


def expand_col_scales(scale_exp, rows):
    return torch.pow(2.0, scale_exp.float()).transpose(0, 1).repeat_interleave(32, dim=0)[:rows, :]


def benchmark_kernel(fn, output, warmup=num_warmup, iters=num_iters):
    for _ in range(warmup):
        output.zero_()
        fn()
    timings = []
    for _ in range(iters):
        output.zero_()
        torch.cuda.synchronize()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))
    return timings


def compute_snr_db(C_test, C_ref):
    C_test_f32 = C_test.float()
    C_ref_f32 = C_ref.float()
    noise = C_test_f32 - C_ref_f32
    signal_power = torch.sum(C_ref_f32 * C_ref_f32).item()
    noise_power = torch.sum(noise * noise).item()
    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal_power / noise_power)


def check_determinism(fn, output, label):
    if determinism_runs <= 1:
        return True, 0.0

    output.zero_()
    fn()
    reference = output[:M, :N].clone()
    max_abs_diff = 0.0
    deterministic = True
    for _ in range(determinism_runs - 1):
        output.zero_()
        fn()
        candidate = output[:M, :N]
        if not torch.equal(candidate, reference):
            deterministic = False
            max_abs_diff = max(
                max_abs_diff,
                (candidate.float() - reference.float()).abs().max().item(),
            )

    status = "PASS" if deterministic else "FAIL"
    print(f"  [{label}] Determinism ({determinism_runs} runs): {status}")
    if not deterministic:
        print(f"    Max abs diff across runs: {max_abs_diff:.6f}")
    return deterministic, max_abs_diff


def check_correctness(C_test, C_ref, label):
    C_test_f32 = C_test.float()
    C_ref_f32 = C_ref.float()
    diff = (C_test_f32 - C_ref_f32).abs()
    scale = C_ref_f32.abs().clamp(min=1.0)
    rel_diff = diff / scale
    atol = 3.0
    rtol = 0.10
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()
    snr_db = compute_snr_db(C_test, C_ref)
    snr_ok = snr_db > snr_threshold_db
    pass_count = ((diff <= atol) | (rel_diff <= rtol)).sum().item()
    total = C_test.numel()
    pass_rate = pass_count / total * 100
    print(f"  [{label}] Correctness (rtol={rtol:.4f}, atol={atol:.1f}):")
    print(f"    Max  abs error: {max_abs:.4f}, Mean abs error: {mean_abs:.4f}")
    print(f"    Max  rel error: {max_rel:.4f}, Mean rel error: {mean_rel:.4f}")
    print(f"    SNR: {snr_db:.2f} dB (threshold {snr_threshold_db:.1f} dB)")
    print(f"    Pass rate: {pass_count}/{total} ({pass_rate:.2f}%)")
    ok = pass_rate >= 99.0 and snr_ok
    print(f"    Result: {'PASS' if ok else 'FAIL'}")
    return {"snr_db": snr_db, "ok": ok}


print(
    f"=== MXFP8 GEMM Layout Benchmark ({'preshuffle-quant' if use_preshuffle_quant else 'reference scale layout'}): "
    f"M={M}, N={N}, K={K} ==="
)
if (build_M, build_N, build_K) != (M, N, K):
    print(f"Build shape={build_M}x{build_N}x{build_K} (zero-padded)")
print(f"Warmup={num_warmup}, iters={num_iters}, correctness={'on' if check_results else 'off'}")
if check_results:
    print(f"SNR threshold={snr_threshold_db:.1f} dB, determinism_runs={determinism_runs}")
print(f"Layouts={','.join(sorted(requested_layouts))}\n")

results = {f"{M}x{N}x{K}": {}}
result_key = f"{M}x{N}x{K}"


def record_result(layout, timings, output, reference, runner):
    avg_ms = sum(timings) / len(timings)
    tflops = flops_ref / (avg_ms * 1e9)
    print(f"  Avg time: {avg_ms:.4f} ms, TFLOPS: {tflops:.2f}")
    results[result_key][layout] = {"avg_ms": avg_ms, "tflops": tflops}
    if check_results:
        quality = check_correctness(output[:M, :N], reference, layout.upper())
        det_ok, det_max_abs = check_determinism(runner, output, layout.upper())
        results[result_key][layout].update(
            {
                "snr_db": quality["snr_db"],
                "deterministic": det_ok,
                "determinism_max_abs_diff": det_max_abs,
                "quality_ok": quality["ok"] and det_ok,
            }
        )
    print()


if "rcr" in requested_layouts:
    print("--- RCR Layout: C = A @ B^T ---")
    A = generate_fp8_matrix(build_M, build_K, M, K)
    B = generate_fp8_matrix(build_N, build_K, N, K)
    A_scale_exp = generate_scale_matrix(build_M, k_blocks, M)
    B_scale_exp = generate_scale_matrix(build_N, k_blocks, N)
    if use_preshuffle_quant:
        if use_v2_rcr:
            A_scale = preshuffle_scale_matrix_mfma16_v2_rcr_a(A_scale_exp)
            B_scale = preshuffle_scale_matrix_mfma16_v2_rcr_b(B_scale_exp)
            run = lambda: tk_mxfp8_layouts.gemm_rcr_pq_v2(A, B, A_scale, B_scale, C)
        else:
            A_scale = preshuffle_scale_matrix_mfma16(A_scale_exp)
            B_scale = preshuffle_scale_matrix_mfma16(B_scale_exp)
            run = lambda: tk_mxfp8_layouts.gemm_rcr_pq(A, B, A_scale, B_scale, C)
    else:
        A_scale = A_scale_exp
        B_scale = B_scale_exp
        run = lambda: tk_mxfp8_layouts.gemm_rcr(A, B, A_scale, B_scale, C)
    C = torch.zeros(build_M, build_N, dtype=torch.bfloat16, device="cuda")
    timings = benchmark_kernel(run, C)
    A_ref = A[:M, :K].float() * expand_row_scales(A_scale_exp[:M, :k_blocks], K)
    B_ref = B[:N, :K].float() * expand_row_scales(B_scale_exp[:N, :k_blocks], K)
    C_ref = A_ref @ B_ref.T
    record_result("rcr", timings, C, C_ref, run)

if "rrr" in requested_layouts:
    print("--- RRR Layout: C = A @ B ---")
    A = generate_fp8_matrix(build_M, build_K, M, K)
    B = generate_fp8_matrix(build_K, build_N, K, N)
    A_scale_exp = generate_scale_matrix(build_M, k_blocks, M)
    B_scale_exp = generate_scale_matrix(build_N, k_blocks, N)
    if use_preshuffle_quant:
        if use_v2_rrr:
            # R22 V2-RRR: reuse RCR's wave-tile reordered preshuffle (A/B
            # scale base formulas are identical between RCR and RRR).
            A_scale = preshuffle_scale_matrix_mfma16_v2_rcr_a(A_scale_exp)
            B_scale = preshuffle_scale_matrix_mfma16_v2_rcr_b(B_scale_exp)
            run = lambda: tk_mxfp8_layouts.gemm_rrr_pq_v2(A, B, A_scale, B_scale, C)
        else:
            A_scale = preshuffle_scale_matrix_mfma16(A_scale_exp)
            B_scale = preshuffle_scale_matrix_mfma16(B_scale_exp)
            run = lambda: tk_mxfp8_layouts.gemm_rrr_pq(A, B, A_scale, B_scale, C)
    else:
        A_scale = A_scale_exp
        B_scale = B_scale_exp
        run = lambda: tk_mxfp8_layouts.gemm_rrr(A, B, A_scale, B_scale, C)
    C = torch.zeros(build_M, build_N, dtype=torch.bfloat16, device="cuda")
    timings = benchmark_kernel(run, C)
    A_ref = A[:M, :K].float() * expand_row_scales(A_scale_exp[:M, :k_blocks], K)
    B_ref = B[:K, :N].float() * expand_col_scales(B_scale_exp[:N, :k_blocks], K)
    C_ref = A_ref @ B_ref
    record_result("rrr", timings, C, C_ref, run)

if "crr" in requested_layouts:
    print("--- CRR Layout: C = A^T @ B ---")
    A = generate_fp8_matrix(build_K, build_M, K, M)
    B = generate_fp8_matrix(build_K, build_N, K, N)
    A_scale_exp = generate_scale_matrix(build_M, k_blocks, M)
    B_scale_exp = generate_scale_matrix(build_N, k_blocks, N)
    if use_preshuffle_quant:
        A_scale = preshuffle_scale_matrix_mfma16(A_scale_exp)
        B_scale = preshuffle_scale_matrix_mfma16(B_scale_exp)
        run = lambda: tk_mxfp8_layouts.gemm_crr_pq(A, B, A_scale, B_scale, C)
    else:
        A_scale = A_scale_exp
        B_scale = B_scale_exp
        run = lambda: tk_mxfp8_layouts.gemm_crr(A, B, A_scale, B_scale, C)
    C = torch.zeros(build_M, build_N, dtype=torch.bfloat16, device="cuda")
    timings = benchmark_kernel(run, C)
    A_ref = A[:K, :M].float() * expand_col_scales(A_scale_exp[:M, :k_blocks], K)
    B_ref = B[:K, :N].float() * expand_col_scales(B_scale_exp[:N, :k_blocks], K)
    C_ref = A_ref.T @ B_ref
    record_result("crr", timings, C, C_ref, run)


outfile = os.environ.get("MXFP8_OUTPUT", default_output_path())
with open(outfile, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {outfile}")

quality_checks = [
    results[result_key][layout].get("quality_ok", False)
    for layout in ("rcr", "rrr", "crr")
    if layout in results[result_key] and check_results
]
if quality_checks and not all(quality_checks):
    sys.exit(1)
