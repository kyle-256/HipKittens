#!/usr/bin/env python3
"""Run SNR for ONE variant (subprocess-isolated to survive HSA faults)."""
import os, sys, math, gc, importlib.util
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
sys.path.insert(0, BUILD_DIR)

INC = "tk_mxfp4_gluon_cpp_n32768_k128256_ts_lgk2_v12_memc_btw_all"


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


def load_mod(mod_name):
    so_path = os.path.join(BUILD_DIR, f"{mod_name}.cpython-310-x86_64-linux-gnu.so")
    if not os.path.exists(so_path):
        return None
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if len(sys.argv) < 2:
        print("Usage: snr_R35_one.py <module_name>")
        sys.exit(1)
    mod_name = sys.argv[1]
    M, N, K = 4096, 32768, 128256
    k_blocks = K // 32
    torch.manual_seed(42)
    A = gen_fp4(M, K); B = gen_fp4(N, K)
    A_sc_exp = torch.zeros((M, k_blocks), dtype=torch.int8, device="cuda")
    B_sc_exp = torch.zeros((N, k_blocks), dtype=torch.int8, device="cuda")
    A_sc = preshuffle_mfma16_merged(A_sc_exp)
    B_sc = preshuffle_mfma16_merged(B_sc_exp)

    mod = load_mod(mod_name)
    if mod is None:
        print(f"MISSING {mod_name}"); sys.exit(2)

    runs = []
    for _ in range(5):
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        mod.gemm_rcr(A, B, A_sc, B_sc, C)
        torch.cuda.synchronize()
        runs.append(C.view(torch.int16).clone())
    stack = torch.stack(runs, 0)
    agree = (stack == stack[0:1]).all(dim=0)
    det = agree.float().mean().item()
    C0 = runs[0].view(torch.bfloat16).float()
    finite = torch.isfinite(C0).float().mean().item()
    print(f"{mod_name}  finite={finite*100:.2f}%  det={det*100:.2f}%  max|C0[finite]|={C0[torch.isfinite(C0)].abs().max():.2f}")

    # If incumbent available, also report bit_eq
    try:
        inc_mod = load_mod(INC)
        if inc_mod is not None:
            inc_runs = []
            for _ in range(5):
                C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
                inc_mod.gemm_rcr(A, B, A_sc, B_sc, C)
                torch.cuda.synchronize()
                inc_runs.append(C.view(torch.int16).clone())
            inc_stack = torch.stack(inc_runs, 0)
            inc_agree = (inc_stack == inc_stack[0:1]).all(dim=0)
            inc_C0 = inc_runs[0].view(torch.bfloat16).float()
            inc_finite = torch.isfinite(inc_C0)
            v_finite = torch.isfinite(C0)
            both_det = agree & inc_agree
            both_fin = inc_finite & v_finite
            valid = both_det & both_fin
            n_valid = valid.sum().item()
            if n_valid > 0:
                bit_eq = (runs[0][valid] == inc_runs[0][valid]).float().mean().item()
                print(f"  vs incumbent: valid={n_valid/(M*N)*100:.2f}%  bit_eq={bit_eq*100:.4f}%")
    except Exception as e:
        print(f"  inc compare error: {e}")

if __name__ == "__main__":
    main()
