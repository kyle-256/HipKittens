#!/usr/bin/env python3
"""R35 Opt A SNR validation: K=4096 (smaller, no spilling), incumbent comparison."""
import os, sys, math, json, gc, importlib.util, subprocess, time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
sys.path.insert(0, BUILD_DIR)
TK_ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=SCRIPT_DIR).decode().strip()

# Build incumbent at K=4096 N=32768 if missing.
M, N, K = 4096, 32768, 4096
INC_NAME = f"tk_mxfp4_gluon_cpp_n{N}_k{K}_R35A_v12_memc_btw_all_legacyfork"  # use legacyfork (VGPR_PF_MODE=0) as ref


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
    ("V0_legacyfork", f"tk_mxfp4_gluon_cpp_n{N}_k{K}_R35A_v12_memc_btw_all_legacyfork"),
    ("V0_vgprpf",     f"tk_mxfp4_gluon_cpp_n{N}_k{K}_R35A_v12_memc_btw_all_vgprpf"),
    ("V1_vmcnt15",    f"tk_mxfp4_gluon_cpp_n{N}_k{K}_R35A_v12_memc_btw_all_vmcnt15"),
    ("V2_vmcnt15_n8", f"tk_mxfp4_gluon_cpp_n{N}_k{K}_R35A_v12_memc_btw_all_vmcnt15_n8"),
    ("V3_vmcnt12_n4", f"tk_mxfp4_gluon_cpp_n{N}_k{K}_R35A_v12_memc_btw_all_vmcnt12_n4"),
]


def main():
    torch.manual_seed(42)
    k_blocks = K // 32

    print(f"=== R35 Opt A SNR (KERNEL-vs-KERNEL) at M={M} N={N} K={K} ===\n")

    A = gen_fp4(M, K)
    B = gen_fp4(N, K)
    # Use small NEGATIVE scales to keep output in finite bf16 range
    A_scale_exp = torch.full((M, k_blocks), -4, dtype=torch.int8, device="cuda")
    B_scale_exp = torch.full((N, k_blocks), -4, dtype=torch.int8, device="cuda")
    A_sc = preshuffle_mfma16_merged(A_scale_exp)
    B_sc = preshuffle_mfma16_merged(B_scale_exp)

    # Use legacyfork (VGPR_PF_MODE=0) as the reference (it = incumbent)
    print(f"[reference={INC_NAME}]", flush=True)
    inc_mod = load_module(INC_NAME)
    if inc_mod is None:
        print(f"  MISSING reference!")
        sys.exit(1)
    inc_runs = []
    for _ in range(5):
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        inc_mod.gemm_rcr(A, B, A_sc, B_sc, C)
        torch.cuda.synchronize()
        inc_runs.append(C.view(torch.int16).clone())
    inc_stack = torch.stack(inc_runs, 0)
    inc_agree = (inc_stack == inc_stack[0:1]).all(dim=0)
    inc_det_frac = inc_agree.float().mean().item()
    inc_C0 = inc_runs[0].view(torch.bfloat16).float()
    inc_finite = torch.isfinite(inc_C0)
    print(f"  ref: finite={inc_finite.float().mean()*100:5.1f}%  det={inc_det_frac*100:5.1f}%  "
          f"max_abs={inc_C0.abs()[inc_finite].max():.2f}")
    del inc_mod; gc.collect(); torch.cuda.empty_cache()

    results = {"reference": INC_NAME,
               "ref_det_frac": inc_det_frac,
               "ref_finite_frac": inc_finite.float().mean().item()}

    for tag, mod_name in VARIANTS:
        if mod_name == INC_NAME:
            continue
        print(f"\n[{tag}]", flush=True)
        mod = load_module(mod_name)
        if mod is None:
            print(f"  MISSING {mod_name}")
            results[tag] = {"status": "MISSING"}
            continue

        try:
            v_runs = []
            for _ in range(5):
                C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
                mod.gemm_rcr(A, B, A_sc, B_sc, C)
                torch.cuda.synchronize()
                v_runs.append(C.view(torch.int16).clone())
        except Exception as e:
            print(f"  CRASH: {e}")
            results[tag] = {"status": f"CRASH:{e}"}
            del mod; gc.collect(); torch.cuda.empty_cache()
            continue

        v_stack = torch.stack(v_runs, 0)
        v_agree = (v_stack == v_stack[0:1]).all(dim=0)
        v_det_frac = v_agree.float().mean().item()
        v_C0 = v_runs[0].view(torch.bfloat16).float()
        v_finite = torch.isfinite(v_C0)

        # Bit-equal vs incumbent on cells deterministic+finite in both
        both_det = inc_agree & v_agree
        both_finite = inc_finite & v_finite
        valid = both_det & both_finite
        n_valid = valid.sum().item()
        if n_valid > 0:
            bit_eq = (inc_runs[0][valid] == v_runs[0][valid]).float().mean().item()
            ref = inc_C0[valid]
            got = v_C0[valid]
            err = got - ref
            sig = (ref ** 2).mean().item()
            err_pow = (err ** 2).mean().item() + 1e-30
            snr_db = 10.0 * math.log10(max(sig, 1e-30) / err_pow) if err_pow > 0 else float('inf')
            max_abs = err.abs().max().item()
        else:
            bit_eq = 0.0
            snr_db = float('-inf')
            max_abs = float('inf')

        verdict = "PASS" if (snr_db >= 40.0 and v_det_frac >= 0.99 * inc_det_frac) else "FAIL"
        results[tag] = {
            "status": "OK", "snr_dB": snr_db, "max_abs_diff": max_abs,
            "bit_eq_frac": bit_eq,
            "v_det_frac": v_det_frac,
            "v_finite_frac": v_finite.float().mean().item(),
            "n_valid_cells": n_valid,
            "verdict": verdict,
        }
        print(f"  finite={v_finite.float().mean()*100:5.1f}% det={v_det_frac*100:5.1f}% "
              f"valid={n_valid/(M*N)*100:5.1f}% "
              f"bit_eq={bit_eq*100:6.2f}% SNR={snr_db:7.2f} dB max_abs={max_abs:.4g} -> {verdict}")

        del mod; gc.collect(); torch.cuda.empty_cache()

    out_path = os.path.join(SCRIPT_DIR, "R35_OPT_A_SNR_smallK_RESULTS.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
