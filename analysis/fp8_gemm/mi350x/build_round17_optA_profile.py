#!/usr/bin/env python3
"""DLA1 profiling driver for Round 17 Optimizer A.

DLA1: M=4096, N=32768, K=128256
Best variant: ts_pf6_6_v12_memc

Loads pre-built .so and runs N profile iterations under rocprof.
warmup=200, iters=100 (lighter than 500 for profiling).
"""
import sys, math, os, importlib.util, json, torch
torch.manual_seed(0)

WARMUP = 200
ITERS  = 100   # lighter for profiling

M, N, K = 4096, 32768, 128256
SCRIPT_DIR = "/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x"
SO = os.path.join(SCRIPT_DIR, "build_all42",
    "tk_mxfp4_gluon_cpp_n32768_k128256_ts_pf6_6_v12_memc.cpython-310-x86_64-linux-gnu.so")

def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')<<4) \
         | torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')

def preshuffle(se):
    r,kb=se.shape; pr=math.ceil(r/64)*64; pk=math.ceil(kb/8)*8
    raw=torch.full((pr,pk),0x7F,dtype=torch.uint8,device=se.device)
    raw[:r,:kb]=(se.to(torch.int16)+127).to(torch.uint8)
    sh=raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh=sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)

def main():
    mod_name = "tk_mxfp4_gluon_cpp_n32768_k128256_ts_pf6_6_v12_memc"
    spec = importlib.util.spec_from_file_location(mod_name, SO)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    print(f"[+] loaded {SO}", flush=True)
    print(f"[+] shape M={M} N={N} K={K}  warmup={WARMUP} iters={ITERS}", flush=True)

    A    = gen_fp4(M, K)
    B    = gen_fp4(N, K)
    sc_a = torch.randint(-2,3,(M, K//32),dtype=torch.int8,device='cuda')
    sc_b = torch.randint(-2,3,(N, K//32),dtype=torch.int8,device='cuda')
    A_sc = preshuffle(sc_a); B_sc = preshuffle(sc_b)
    C    = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

    run = lambda: mod.gemm_rcr(A, B, A_sc, B_sc, C)
    for _ in range(WARMUP): run()
    torch.cuda.synchronize()
    print(f"[+] warmup done; starting {ITERS} timed iters", flush=True)

    times = []
    for i in range(ITERS):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); run(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort(); trim = int(len(times)*0.10)
    times_t = times[trim:-trim] if trim>0 else times
    avg = sum(times_t)/len(times_t)
    t  = 2.0 * M * N * K / (avg * 1e-3) / 1e12
    print(f"[+] avg={avg:.4f} ms  tflops={t:.1f}", flush=True)

if __name__ == "__main__":
    main()
