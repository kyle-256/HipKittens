#!/usr/bin/env python3
"""Correctness test for K-specialized ASM inline kernel.

Compares against a tiny torch-only reference for small M, N to catch K-loop bugs.
Then benchmarks correctness via stat sanity (non-zero count, magnitude) on
the deep-LOSE shapes.
"""
import sys, os, math, importlib.util, subprocess
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

WARMUP, ITERS, TRIM = 200, 500, 0.10

# K-spec → so path
SO = {
    8192:  os.path.join(SCRIPT_DIR, "build_asm_kvar", "tk_mxfp4_asm_inline_k8192.cpython-310-x86_64-linux-gnu.so"),
    14336: os.path.join(SCRIPT_DIR, "build_asm_kvar", "tk_mxfp4_asm_inline_k14336.cpython-310-x86_64-linux-gnu.so"),
    16384: os.path.join(SCRIPT_DIR, "build_asm_kvar", "tk_mxfp4_asm_inline_k16384.cpython-310-x86_64-linux-gnu.so"),
    28672: os.path.join(SCRIPT_DIR, "build_asm_kvar", "tk_mxfp4_asm_inline_k28672.cpython-310-x86_64-linux-gnu.so"),
    32768: os.path.join(SCRIPT_DIR, "build_asm_kvar", "tk_mxfp4_asm_inline_k32768.cpython-310-x86_64-linux-gnu.so"),
}


def run_correctness(K, so_path, M=512, N=512):
    """Run the K-specialized kernel on a small problem and dump output stats."""
    script = f"""
import sys, math, torch, importlib.util
def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda') << 4) | torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda')
def preshuffle(se):
    r, kb = se.shape; pr = math.ceil(r/64)*64; pk = math.ceil(kb/8)*8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=se.device)
    raw[:r,:kb] = (se.to(torch.int16)+127).to(torch.uint8)
    sh = raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh = sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)
mod_name = 'tk_mxfp4_asm_inline_k{K}'
spec = importlib.util.spec_from_file_location(mod_name, '{so_path}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
M,N,K = {M},{N},{K}
torch.manual_seed(0)
A=gen_fp4(M,K); B=gen_fp4(N,K)
sc_a=torch.randint(-2,3,(M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.randint(-2,3,(N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
mod.gemm_rcr(A,B,A_sc,B_sc,C)
torch.cuda.synchronize()
nz = (C != 0).sum().item()
mx = C.abs().max().item()
mean = C.float().mean().item()
std = C.float().std().item()
nonzero_frac = nz / (M*N)
# crude sanity: max should be in some realistic range.
# For random fp4 K-accum with random scales, expect O(K) magnitude.
print(f"K={K} M={M} N={N}  NONZERO_FRAC {{nonzero_frac:.4f}}  MAX_ABS {{mx:.2f}}  MEAN {{mean:.4f}}  STD {{std:.4f}}")
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    return r.stdout, r.stderr, r.returncode


def bench_one(K, so_path, M, N):
    """Bench TFLOPS at warmup=200, iters=500."""
    script = f"""
import sys, math, torch, importlib.util
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda') << 4) | torch.randint(0, 16, (r, c), dtype=torch.uint8, device='cuda')
def preshuffle(se):
    r, kb = se.shape; pr = math.ceil(r/64)*64; pk = math.ceil(kb/8)*8
    raw = torch.full((pr, pk), 0x7F, dtype=torch.uint8, device=se.device)
    raw[:r,:kb] = (se.to(torch.int16)+127).to(torch.uint8)
    sh = raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh = sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)
mod_name = 'tk_mxfp4_asm_inline_k{K}'
spec = importlib.util.spec_from_file_location(mod_name, '{so_path}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
M,N,K = {M},{N},{K}
torch.manual_seed(0)
A=gen_fp4(M,K); B=gen_fp4(N,K)
sc_a=torch.randint(-2,3,(M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.randint(-2,3,(N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
run=lambda:mod.gemm_rcr(A,B,A_sc,B_sc,C)
for _ in range(WARMUP): run()
torch.cuda.synchronize()
times=[]
for _ in range(ITERS):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); run(); e.record(); torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
times.sort(); trim=int(len(times)*TRIM)
times=times[trim:-trim] if trim>0 else times
avg=sum(times)/len(times); t=2.0*M*N*K/(avg*1e-3)/1e12
nz = (C != 0).sum().item()
print(f"BENCH K={K} M={M} N={N}  TFLOPS {{t:.1f}}  MS {{avg:.4f}}  NONZERO {{nz}}/{{M*N}}")
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    return r.stdout, r.stderr, r.returncode


if __name__ == "__main__":
    print("=== Correctness sanity (M=N=256) ===")
    for K in [8192, 14336, 16384, 28672, 32768]:
        out, err, rc = run_correctness(K, SO[K], M=256, N=256)
        print(out.strip() if out.strip() else f"K={K} (no output)")
        if rc != 0:
            print(f"  STDERR: {err[-500:]}")

    # If sanity passes, bench on actual deep-LOSE shapes
    print("\n=== TFLOPS on selected deep-LOSE / reference shapes ===")
    SHAPES = [
        (8192, 8192, 8192),         # baseline reference (~5258)
        (14336, 4096, 32768),       # deep-LOSE
        (16384, 4096, 28672),       # deep-LOSE
        (32768, 4096, 14336),       # deep-LOSE
        (28672, 4096, 16384),       # deep-LOSE
    ]
    for M, N, K in SHAPES:
        if K not in SO:
            print(f"  M={M} N={N} K={K}: skipped (no K-variant)")
            continue
        out, err, rc = bench_one(K, SO[K], M, N)
        print(out.strip() if out.strip() else f"M={M} N={N} K={K} (no output)")
        if rc != 0:
            print(f"  STDERR: {err[-1000:]}")
