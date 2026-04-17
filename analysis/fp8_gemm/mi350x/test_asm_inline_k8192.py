#!/usr/bin/env python3
"""Quick correctness + perf test of the gluon-derived ASM inline kernel.

Tests on K=8192 (the K it was originally compiled for) — should hit ~5084 TFLOPS.
Also runs gluon_cpp default for the same shape as a comparison.
"""
import sys, os, math, importlib.util, time, subprocess
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WARMUP, ITERS, TRIM = 200, 500, 0.10

def build_default_gluon(m, n, k):
    """Build default gluon_cpp for this shape for comparison."""
    so = os.path.join(SCRIPT_DIR, "build_all42", f"tk_mxfp4_gluon_cpp_n{n}_k{k}.cpython-310-x86_64-linux-gnu.so")
    return so if os.path.exists(so) else None

def bench(so_path, mod_name, m, n, k):
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
spec = importlib.util.spec_from_file_location('{mod_name}', '{so_path}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
M,N,K = {m},{n},{k}
torch.manual_seed(0)
A=gen_fp4(M,K); B=gen_fp4(N,K)
sc_a=torch.randint(-2,3,(M,K//32),dtype=torch.int8,device='cuda')
sc_b=torch.randint(-2,3,(N,K//32),dtype=torch.int8,device='cuda')
A_sc=preshuffle(sc_a); B_sc=preshuffle(sc_b)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
run=lambda:mod.gemm_rcr(A,B,A_sc,B_sc,C)
# Correctness sanity: make sure it doesn't all-zero
run()
torch.cuda.synchronize()
import torch as _t
nz = (C != 0).sum().item()
mx = C.abs().max().item()
print(f"NONZERO_COUNT {{nz}} MAX_ABS {{mx:.4f}}")
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
print(f"TFLOPS {{t:.1f}} MS {{avg:.4f}}")
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    return r.stdout, r.stderr, r.returncode

def main():
    M, N, K = 8192, 8192, 8192
    asm_so = os.path.join(SCRIPT_DIR, "tk_mxfp4_asm_inline.cpython-310-x86_64-linux-gnu.so")

    print(f"=== ASM inline on M={M} N={N} K={K} ===")
    out, err, rc = bench(asm_so, "tk_mxfp4_asm_inline", M, N, K)
    print("STDOUT:", out)
    if rc != 0:
        print("STDERR:", err[-2000:])
        sys.exit(1)

    print(f"\n=== Gluon CPP default on same shape (reference) ===")
    cpp_so = build_default_gluon(M, N, K)
    if cpp_so:
        out, err, rc = bench(cpp_so, f"tk_mxfp4_gluon_cpp_n{N}_k{K}", M, N, K)
        print("STDOUT:", out)
        if rc != 0:
            print("STDERR:", err[-2000:])
    else:
        print("(no prebuilt default gluon_cpp .so found)")

if __name__ == "__main__":
    main()
