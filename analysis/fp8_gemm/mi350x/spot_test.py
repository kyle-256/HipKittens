#!/usr/bin/env python3
"""Spot-test a single shape with all auto-tune variants. Runs on GPU specified by HIP_VISIBLE_DEVICES."""
import sys, os, subprocess, json, math, importlib.util, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

WARMUP = 200
ITERS = 500
TRIM = 0.10

VARIANTS = [
    ("default", ""),
    ("gm1", "-DGROUP_SIZE_M=1"),
    ("gm2", "-DGROUP_SIZE_M=2"),
    ("gm8", "-DGROUP_SIZE_M=8"),
    ("gm16", "-DGROUP_SIZE_M=16"),
    ("u8", "-DUNROLL_K=8"),
    ("u16", "-DUNROLL_K=16"),
    ("u32", "-DUNROLL_K=32"),
    ("gm2u8", "-DGROUP_SIZE_M=2 -DUNROLL_K=8"),
    ("gm2u16", "-DGROUP_SIZE_M=2 -DUNROLL_K=16"),
    ("gm8u8", "-DGROUP_SIZE_M=8 -DUNROLL_K=8"),
    ("gm8u16", "-DGROUP_SIZE_M=8 -DUNROLL_K=16"),
    ("gm16u8", "-DGROUP_SIZE_M=16 -DUNROLL_K=8"),
    ("gm16u16", "-DGROUP_SIZE_M=16 -DUNROLL_K=16"),
    ("gm1u16", "-DGROUP_SIZE_M=1 -DUNROLL_K=16"),
    ("ts", "-DTAIL_SPLIT=1"),
    ("ts_gm8", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8"),
    ("ts_u16", "-DTAIL_SPLIT=1 -DUNROLL_K=16"),
    ("ts_gm2", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2"),
    ("ts_gm2u16", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DUNROLL_K=16"),
    ("swap", "-DSWAP_STEP34_MAIN=1 -DSWAP_STEP12_MAIN=1"),
    ("swap_gm8", "-DSWAP_STEP34_MAIN=1 -DSWAP_STEP12_MAIN=1 -DGROUP_SIZE_M=8"),
    ("ts_v12", "-DTAIL_SPLIT=1 -DSTEP3_BARRIER_VMCNT=12"),
    ("ts_gm8_v12", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12"),
    ("ts_gm2u8", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=2 -DUNROLL_K=8"),
    ("spread_gm2u8", "-DSPREAD_LDS=1 -DGROUP_SIZE_M=2 -DUNROLL_K=8"),
    ("spread_gm2u8_v12", "-DSPREAD_LDS=1 -DGROUP_SIZE_M=2 -DUNROLL_K=8 -DSTEP3_BARRIER_VMCNT=12"),
    ("spread_gm8", "-DSPREAD_LDS=1 -DGROUP_SIZE_M=8"),
    ("v12", "-DSTEP3_BARRIER_VMCNT=12"),
    ("gm8_v12", "-DGROUP_SIZE_M=8 -DSTEP3_BARRIER_VMCNT=12"),
    ("no_nvs", "-DNONVOLATILE_SCALE_X2_POC=0"),
    ("pf4", "-DSTEP3_PF_N=4 -DSTEP4_PF_N=4"),
    ("ts_pf4", "-DTAIL_SPLIT=1 -DSTEP3_PF_N=4 -DSTEP4_PF_N=4"),
    ("gm2_v12", "-DGROUP_SIZE_M=2 -DSTEP3_BARRIER_VMCNT=12"),
    ("ext_br", "-DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("ts_ext_br", "-DTAIL_SPLIT=1 -DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("gm8_ext_br", "-DGROUP_SIZE_M=8 -DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("ts_gm8_ext_br", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=8 -DSTEP4_EXTERNAL_BR_PREFETCH=1"),
    ("gm32", "-DGROUP_SIZE_M=32"),
    ("gm64", "-DGROUP_SIZE_M=64"),
    ("ts_gm16", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=16"),
    ("ts_gm32", "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=32"),
]

def build(n, k, tag, flags, build_dir):
    so_name = f"tk_mxfp4_gluon_cpp_{tag}.cpython-310-x86_64-linux-gnu.so"
    so_path = os.path.join(build_dir, so_name)
    if os.path.exists(so_path):
        return so_path
    cmd = (
        f"/opt/rocm/bin/hipcc {SCRIPT_DIR}/kernel_mxfp4_gluon_cpp.cpp "
        f"--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
        f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
        f"-I/opt/rocm/include/hip "
        f"-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
        f"-shared -fPIC -std=c++20 -w "
        f"-DN_DIM={n} -DK_DIM={k} {flags} "
        f"-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm "
        f"-o {so_path}"
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None
    return so_path

def bench_one(so_path, m, n, k, comp):
    """Run benchmark in subprocess to avoid symbol conflicts."""
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
EXT = '.cpython-310-x86_64-linux-gnu.so'
spec = importlib.util.spec_from_file_location('tk_mxfp4_gluon_cpp', '{so_path}')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
M,N,K = {m},{n},{k}
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
import json; print(json.dumps({{"tflops": round(t,1), "ms": round(avg,4)}}))
"""
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=300
        )
        if r.returncode != 0:
            return None
        return json.loads(r.stdout.strip())
    except:
        return None

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} M N K [comp_tflops]")
        sys.exit(1)

    m, n, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    comp = float(sys.argv[4]) if len(sys.argv) > 4 else 0

    build_dir = os.path.join(SCRIPT_DIR, f"build_spot_{n}_{k}")
    os.makedirs(build_dir, exist_ok=True)

    print(f"Shape: {m}x{n}x{k}, comp={comp}")
    print(f"Building {len(VARIANTS)} variants...")

    # Build all variants
    sos = {}
    for tag, flags in VARIANTS:
        so = build(n, k, tag, flags, build_dir)
        if so:
            sos[tag] = so
        else:
            print(f"  FAIL: {tag}")

    print(f"Built {len(sos)}/{len(VARIANTS)}. Benchmarking...")
    print(f"{'Tag':>15} {'TFLOPS':>8} {'Ratio':>7} {'Status':>6}")
    print("-" * 42)

    best_t, best_tag = 0, ""
    for tag, _ in VARIANTS:
        if tag not in sos:
            continue
        r = bench_one(sos[tag], m, n, k, comp)
        if r is None:
            print(f"{tag:>15}    CRASH")
            continue
        t = r["tflops"]
        ratio = t / comp * 100 if comp > 0 else 0
        status = "WIN" if ratio >= 100 else "LOSE"
        marker = " <<<" if t > best_t else ""
        print(f"{tag:>15} {t:>8.1f} {ratio:>6.1f}% {status:>6}{marker}")
        if t > best_t:
            best_t, best_tag = t, tag

    print(f"\nBest: {best_tag} = {best_t:.1f}T", end="")
    if comp > 0:
        print(f" ({best_t/comp*100:.1f}% of {comp})")
    else:
        print()

if __name__ == "__main__":
    main()
