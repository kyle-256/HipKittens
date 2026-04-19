#!/usr/bin/env python3
"""R44 Opt C minimal CRASH repro for (4096, 32768, 28672) on the p2b_ctrl .so.

Designed to be wrapped with HSA_DEBUG=1 / AMD_LOG_LEVEL=4 / rocgdb / etc.
No bench. No verification. Just call the kernel once and let it crash.
"""
import sys, os, math, importlib.util, torch

SO = os.environ.get(
    "REPRO_SO",
    "/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R43A/"
    "tk_mxfp4_gluon_cpp_n32768_k28672_ts_lgk2_gm7_pfoff104_kx28672_btw_all_R43A_p2b_ctrl"
    ".cpython-310-x86_64-linux-gnu.so",
)
M = int(os.environ.get("REPRO_M", "4096"))
N = int(os.environ.get("REPRO_N", "32768"))
K = int(os.environ.get("REPRO_K", "28672"))

print(f"R44_OPT_C_repro: SO={SO}", flush=True)
print(f"R44_OPT_C_repro: M={M} N={N} K={K}", flush=True)

torch.manual_seed(42)
# Module name MUST match PyInit symbol embedded in the .so.
MOD_NAME = os.path.basename(SO).split('.')[0]
print(f"R44_OPT_C_repro: MOD_NAME={MOD_NAME}", flush=True)
spec = importlib.util.spec_from_file_location(MOD_NAME, SO)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("R44_OPT_C_repro: module loaded", flush=True)

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

A = gen_fp4(M, K)
B = gen_fp4(N, K)
sc_a = torch.randint(-2, 3, (M, K // 32), dtype=torch.int8, device='cuda')
sc_b = torch.randint(-2, 3, (N, K // 32), dtype=torch.int8, device='cuda')
A_sc = preshuffle(sc_a)
B_sc = preshuffle(sc_b)
C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

print("R44_OPT_C_repro: tensors built; calling gemm_rcr...", flush=True)
sys.stdout.flush()
sys.stderr.flush()
mod.gemm_rcr(A, B, A_sc, B_sc, C)
torch.cuda.synchronize()
print("R44_OPT_C_repro: completed without crash (UNEXPECTED)", flush=True)
