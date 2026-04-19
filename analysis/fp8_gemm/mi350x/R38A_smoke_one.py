#!/usr/bin/env python3
"""Quick smoke test: load one R38A module, run it once, report kernel_finite."""
import sys, math, json, importlib.util, os, sysconfig
import torch
torch.manual_seed(0)

EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda') << 4) | \
           torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda')


def preshuffle(se):
    r, kb = se.shape
    pr = math.ceil(r / 64) * 64
    pk = math.ceil(kb / 8) * 8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=se.device)
    raw[:r, :kb] = (se.to(torch.int16) + 127).to(torch.uint8)
    sh = raw.view(pr // 32, 2, 16, pk // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(pr // 32, pk * 32)
    sh = sh.view(pr // 64, 2, pk * 32 // 4, 4).permute(0, 2, 1, 3).contiguous()
    return sh.view(pr // 64, pk * 64)


def main():
    M = int(sys.argv[1]); N = int(sys.argv[2]); K = int(sys.argv[3])
    module_name = sys.argv[4]
    so_dir = sys.argv[5] if len(sys.argv) > 5 else os.path.join(SCRIPT_DIR, "build_R38A")
    so_path = os.path.join(so_dir, f"{module_name}{EXT_SUFFIX}")
    print(f"Loading {so_path}", flush=True)

    spec = importlib.util.spec_from_file_location(module_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    A = gen_fp4(M, K); B = gen_fp4(N, K)
    sc_a = torch.full((M, K // 32), -4, dtype=torch.int8, device='cuda')
    sc_b = torch.full((N, K // 32), -4, dtype=torch.int8, device='cuda')
    A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

    mod.gemm_rcr(A, B, A_sc, B_sc, C)
    torch.cuda.synchronize()

    Cf = C.float()
    finite = float(torch.isfinite(Cf).sum().item()) / float(Cf.numel())
    nan_frac = float(torch.isnan(Cf).sum().item()) / float(Cf.numel())
    inf_frac = float(torch.isinf(Cf).sum().item()) / float(Cf.numel())
    print(f"M={M} N={N} K={K}  kernel_finite={finite:.6f}  nan={nan_frac:.6f}  inf={inf_frac:.6f}")
    print(f"PASS" if finite >= 0.995 else "FAIL")


if __name__ == "__main__":
    main()
