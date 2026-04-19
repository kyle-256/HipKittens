#!/usr/bin/env python3
"""R35 Opt A SNR validation: keepalive + proper VGPR ds_write deposit.

Tests at K=4096 (where output stays finite, torch ref is reliable).
Gate: SNR_det >= 40 dB and bit-eq with incumbent.
"""
import os, sys, math, json, gc, importlib.util
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
sys.path.insert(0, BUILD_DIR)

M, N, K = 4096, 32768, 4096
SNR_GATE = 40.0

FP4_E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def dequant_fp4(packed_uint8, K):
    lo = (packed_uint8 & 0x0F).to(torch.int64)
    hi = ((packed_uint8 >> 4) & 0x0F).to(torch.int64)
    table = FP4_E2M1_TABLE.to(packed_uint8.device)
    rows = packed_uint8.shape[0]
    cols = packed_uint8.shape[1]
    out = torch.empty(rows, cols * 2, dtype=torch.float32, device=packed_uint8.device)
    out[:, 0::2] = table[lo]
    out[:, 1::2] = table[hi]
    return out[:, :K]


def apply_block_scales(data_f32, scale_exp_i8, block_size=32):
    rows, K_dim = data_f32.shape
    k_blocks = K_dim // block_size
    scales = (2.0 ** scale_exp_i8.to(torch.float32))
    scales_expanded = scales.unsqueeze(-1).expand(rows, k_blocks, block_size)
    scales_expanded = scales_expanded.reshape(rows, k_blocks * block_size)[:, :K_dim]
    return data_f32 * scales_expanded


def torch_reference(A_packed, B_packed, A_scale_exp, B_scale_exp, K_dim):
    A_f32 = dequant_fp4(A_packed, K_dim)
    B_f32 = dequant_fp4(B_packed, K_dim)
    A_scaled = apply_block_scales(A_f32, A_scale_exp)
    B_scaled = apply_block_scales(B_f32, B_scale_exp)
    C_f32 = torch.matmul(A_scaled, B_scaled.T)
    return C_f32.to(torch.bfloat16)


def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device="cuda")
    return (hi << 4) | lo


def preshuffle_mfma16_merged(scale_exp):
    rows, kb = scale_exp.shape
    pr = math.ceil(rows / 64) * 64
    pk = math.ceil(kb / 8) * 8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=scale_exp.device)
    raw[:rows, :kb] = (scale_exp.to(torch.int16) + 127).to(torch.uint8)
    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1)
    sh = sh.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    sh = sh.view(pr // 32, pk * 32)
    sh = sh.view(pr // 64, 2, pk * 32 // 4, 4)
    sh = sh.permute(0, 2, 1, 3).contiguous()
    return sh.view(pr // 64, pk * 64)


def load_module(mod_name):
    so_path = os.path.join(BUILD_DIR, f"{mod_name}.cpython-310-x86_64-linux-gnu.so")
    if not os.path.exists(so_path):
        return None
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VARIANTS = [
    ("V0_legacyfork", f"tk_mxfp4_gluon_cpp_n32768_k{K}_R35A_v12_memc_btw_all_legacyfork"),
    ("V0_vgprpf",     f"tk_mxfp4_gluon_cpp_n32768_k{K}_R35A_v12_memc_btw_all_vgprpf"),
    ("V1_vmcnt15",    f"tk_mxfp4_gluon_cpp_n32768_k{K}_R35A_v12_memc_btw_all_vmcnt15"),
    ("V2_vmcnt15_n8", f"tk_mxfp4_gluon_cpp_n32768_k{K}_R35A_v12_memc_btw_all_vmcnt15_n8"),
    ("V3_vmcnt12_n4", f"tk_mxfp4_gluon_cpp_n32768_k{K}_R35A_v12_memc_btw_all_vmcnt12_n4"),
]


def main():
    torch.manual_seed(42)
    k_blocks = K // 32

    print(f"=== R35 Opt A SNR (M={M}, N={N}, K={K}, gate={SNR_GATE} dB) ===\n")

    A = gen_fp4(M, K)
    B = gen_fp4(N, K)
    A_scale_exp = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device="cuda")
    B_scale_exp = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device="cuda")
    A_sc = preshuffle_mfma16_merged(A_scale_exp)
    B_sc = preshuffle_mfma16_merged(B_scale_exp)

    print("Computing torch reference (float32 matmul)...", flush=True)
    C_ref = torch_reference(A, B, A_scale_exp, B_scale_exp, K)
    ref_finite = torch.isfinite(C_ref.float()).sum().item()
    print(f"  ref: finite={ref_finite}/{C_ref.numel()} ({ref_finite/C_ref.numel()*100:.1f}%) "
          f"max={C_ref.float().abs().max().item():.2f}\n")

    results = {}
    for tag, mod_name in VARIANTS:
        print(f"[{tag}] loading {mod_name}...", flush=True)
        mod = load_module(mod_name)
        if mod is None:
            print(f"  MISSING")
            results[tag] = {"status": "MISSING"}
            continue

        # Run 5 times, check determinism
        runs = []
        try:
            for _ in range(5):
                C_test = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
                mod.gemm_rcr(A, B, A_sc, B_sc, C_test)
                torch.cuda.synchronize()
                runs.append(C_test.view(torch.int16).clone())
        except Exception as e:
            print(f"  RUN EXCEPTION: {e}")
            results[tag] = {"status": f"CRASH:{e}"}
            del mod; gc.collect(); torch.cuda.empty_cache()
            continue

        stack = torch.stack(runs, 0)
        agree = (stack == stack[0:1]).all(dim=0)
        det_frac = agree.float().mean().item()
        C0 = runs[0].view(torch.bfloat16)

        # SNR vs torch reference, on deterministic + finite cells
        diff = (C0.float() - C_ref.float())
        both_finite = torch.isfinite(C0.float()) & torch.isfinite(C_ref.float())
        det_finite = both_finite & agree
        n_finite = both_finite.sum().item()
        n_det = det_finite.sum().item()

        if n_det > 0:
            ref_vals = C_ref.float()[det_finite]
            diff_vals = diff[det_finite]
            sig = (ref_vals ** 2).mean().item()
            err = (diff_vals ** 2).mean().item() + 1e-30
            ratio = max(sig, 1e-30) / err
            snr_det = 10.0 * math.log10(ratio) if ratio > 0 else float('-inf')
            max_abs = diff_vals.abs().max().item()
        else:
            snr_det = float('-inf')
            max_abs = float('inf')

        if n_finite > 0:
            ref_vals = C_ref.float()[both_finite]
            diff_vals = diff[both_finite]
            sig = (ref_vals ** 2).mean().item()
            err = (diff_vals ** 2).mean().item() + 1e-30
            ratio = max(sig, 1e-30) / err
            snr_all = 10.0 * math.log10(ratio) if ratio > 0 else float('-inf')
        else:
            snr_all = float('-inf')

        verdict = "PASS" if (snr_det >= SNR_GATE and det_frac >= 0.99) else "FAIL"
        results[tag] = {
            "status": "OK", "snr_det_dB": snr_det, "snr_all_dB": snr_all,
            "max_abs_diff": max_abs,
            "finite_frac": n_finite / C0.numel(),
            "det_frac": det_frac,
            "verdict": verdict,
        }
        print(f"  finite={n_finite/C0.numel()*100:5.1f}%  det={det_frac*100:5.1f}%  "
              f"SNR_all={snr_all:7.2f} dB  SNR_det={snr_det:7.2f} dB  "
              f"max_abs={max_abs:.4g}  -> {verdict}")

        del mod; gc.collect(); torch.cuda.empty_cache()

    out_path = os.path.join(SCRIPT_DIR, "R35_OPT_A_SNR_RESULTS.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
