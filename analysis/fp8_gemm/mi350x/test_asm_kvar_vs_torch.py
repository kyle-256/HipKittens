#!/usr/bin/env python3
"""Compare K-specialized ASM kernel against the torch reference used by
test_mxfp4_gluon_cpp.py (verified to match the gluon kernel)."""
import sys, os, math, subprocess
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SO = {
    8192:  os.path.join(SCRIPT_DIR, "build_asm_kvar", "tk_mxfp4_asm_inline_k8192.cpython-310-x86_64-linux-gnu.so"),
    14336: os.path.join(SCRIPT_DIR, "build_asm_kvar", "tk_mxfp4_asm_inline_k14336.cpython-310-x86_64-linux-gnu.so"),
    16384: os.path.join(SCRIPT_DIR, "build_asm_kvar", "tk_mxfp4_asm_inline_k16384.cpython-310-x86_64-linux-gnu.so"),
    28672: os.path.join(SCRIPT_DIR, "build_asm_kvar", "tk_mxfp4_asm_inline_k28672.cpython-310-x86_64-linux-gnu.so"),
    32768: os.path.join(SCRIPT_DIR, "build_asm_kvar", "tk_mxfp4_asm_inline_k32768.cpython-310-x86_64-linux-gnu.so"),
}

def correctness(K, M=256, N=256):
    so = SO[K]
    script = f"""
import torch, math, importlib.util
torch.manual_seed(0)
M, N, K = {M}, {N}, {K}
k_blocks = K // 32

FP4_LUT = torch.tensor(
    [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0],
    dtype=torch.float32, device='cuda'
)

def gen_fp4(rows, K):
    cols = K // 2
    lo = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
    hi = torch.randint(0, 16, (rows, cols), dtype=torch.uint8, device='cuda')
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

A = gen_fp4(M, K)
B = gen_fp4(N, K)
sc_exp_a = torch.randint(-2, 3, (M, k_blocks), dtype=torch.int8, device='cuda')
sc_exp_b = torch.randint(-2, 3, (N, k_blocks), dtype=torch.int8, device='cuda')
A_sc = preshuffle_mfma16_merged(sc_exp_a)
B_sc = preshuffle_mfma16_merged(sc_exp_b)
C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

mod_name = 'tk_mxfp4_asm_inline_k{K}'
spec = importlib.util.spec_from_file_location(mod_name, '{so}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.gemm_rcr(A, B, A_sc, B_sc, C)
torch.cuda.synchronize()

def unpack(p, K):
    lo = (p & 0x0F).to(torch.int64); hi = ((p >> 4) & 0x0F).to(torch.int64)
    out = torch.empty(p.shape[0], K, dtype=torch.float32, device='cuda')
    out[:, 0::2] = FP4_LUT[lo]; out[:, 1::2] = FP4_LUT[hi]
    return out
def expand(exp, K):
    return torch.pow(2.0, exp.float()).repeat_interleave(32, dim=1)[:, :K]

Af = unpack(A, K) * expand(sc_exp_a, K)
Bf = unpack(B, K) * expand(sc_exp_b, K)
Cref = Af @ Bf.T
noise = C.float() - Cref.float()
sig = (Cref.float()**2).sum().item()
noi = (noise**2).sum().item()
snr = 10*math.log10(sig/noi) if noi > 0 else float('inf')
max_err = noise.abs().max().item()
print(f"K={K} M={M} N={N}: SNR {{snr:.2f}} dB, max_err {{max_err:.2f}}, C_ref_max {{Cref.float().abs().max().item():.2f}}, C_kernel_max {{C.float().abs().max().item():.2f}}")
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    print(r.stdout.strip() if r.stdout.strip() else f"K={K} (no output)")
    if r.returncode != 0:
        print(f"  STDERR: {r.stderr[-1000:]}")

if __name__ == "__main__":
    for K in [8192, 14336, 16384, 28672, 32768]:
        correctness(K, M=256, N=256)
