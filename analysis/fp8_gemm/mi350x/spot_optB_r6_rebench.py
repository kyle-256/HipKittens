#!/usr/bin/env python3
"""Re-bench Phase 4b suspect entries SERIALLY on a single idle GPU.

Suspect entries from spot_optB_r6_unroll.log:
  - 16384x4096x2048      _v16_wpe2 (FAILED — module path likely uses default _v16_wpe2 build)
  - 4096x128256x32768    _v16_wpe2 (FAILED)
  - 4096x4096x8192       _v16_wpe2 (FAILED)
  - 16384x4096x2048      _optB_r6_u16_lgk2_dc_v16_wpe2 (281.4 TFLOPS = 9.4% — bogus)
  - 4096x128256x32768    _optB_r6_u16_v16_wpe2 / u8_v16_wpe2_memc / u16_lgk2_dc_v16_wpe2  (162% bogus)
  - 4096x4096x8192       _optB_r6_u16_v16_wpe2 (713 TFLOPS = 18% bogus)

Build _v16_wpe2 baseline `.so` for those (n,k) pairs if missing,
then re-bench serially. warmup=200 iters=500 trim=10%.
"""
import os, sys, json, math, time, subprocess, sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TK_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")
LOG_DIR = os.path.join(SCRIPT_DIR, "build_logs_optB_r6")
os.makedirs(LOG_DIR, exist_ok=True)

WARMUP, ITERS, TRIM = 200, 500, 0.10
GPU = int(os.environ.get("BENCH_GPU", "4"))

BASELINE_VARIANT = "_v16_wpe2"
BASELINE_FLAGS = "-DSTEP3_BARRIER_VMCNT=16 -DWAVES_PER_EU_2=1"

WINNERS = [
    "_optB_r6_u16_v16_wpe2",
    "_optB_r6_u8_v16_wpe2_memc",
    "_optB_r6_u16_lgk2_dc_v16_wpe2",
]

# (M, N, K, comp_baseline_TFLOPS) — re-bench all "WIN sample + suspect" rows
RECHECK_SHAPES = [
    (16384, 4096, 2048, 2995.0),
    (4096, 128256, 32768, 3195.3),
    (4096, 4096, 8192, 3959.9),
]

BASE_HIPCC = (
    "/opt/rocm/bin/hipcc {wrapper_src} "
    "--offload-arch=gfx950 -DKITTENS_CDNA4 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math "
    f"-I/opt/rocm/include/rocrand -I{TK_ROOT}/include -I{TK_ROOT}/prototype "
    "-I/opt/rocm/include/hip "
    "-I/usr/include/python3.10 -I/opt/venv/lib/python3.10/site-packages/pybind11/include "
    "-shared -fPIC -std=c++20 -w "
    "-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl -lm "
    "-DK_DIM={k} -DN_DIM={n} {flags} -o {so}"
)


def ensure_build(n, k, suffix, flags):
    module_name = f"tk_mxfp4_gluon_cpp_n{n}_k{k}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if os.path.exists(so_path):
        return so_path, module_name, "cached"
    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n}_k{k}{suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)
    cmd = BASE_HIPCC.format(wrapper_src=wrapper_src, k=k, n=n, flags=flags, so=so_path)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(so_path):
        return None, module_name, f"BUILD FAIL ({dt:.1f}s) {r.stderr[-300:]}"
    return so_path, module_name, f"built ({dt:.1f}s)"


def bench(m, n, k, suffix, gpu_id):
    module_name = f"tk_mxfp4_gluon_cpp_n{n}_k{k}{suffix}"
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return None
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
M, N, K = {m}, {n}, {k}
def gen_fp4(r, K):
    c = K // 2
    return (torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')<<4)|torch.randint(0,16,(r,c),dtype=torch.uint8,device='cuda')
def preshuffle(se):
    r,kb=se.shape; pr=math.ceil(r/64)*64; pk=math.ceil(kb/8)*8
    raw=torch.full((pr,pk),0x7F,dtype=torch.uint8,device=se.device)
    raw[:r,:kb]=(se.to(torch.int16)+127).to(torch.uint8)
    sh=raw.view(pr//32,2,16,pk//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(pr//32,pk*32)
    sh=sh.view(pr//64,2,pk*32//4,4).permute(0,2,1,3).contiguous()
    return sh.view(pr//64,pk*64)
spec=importlib.util.spec_from_file_location('{module_name}','{so_path}')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
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
print(json.dumps({{"tflops":round(t,1),"ms":round(avg,4)}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            return {"err": "run_failed", "stderr": r.stderr[-300:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"err": str(e)}


def main():
    print(f"=== Round 6 Optimizer B re-bench (SERIAL, GPU={GPU}) ===")
    print(f"warmup={WARMUP} iters={ITERS} trim={TRIM}")
    print()

    # 1. Ensure baseline _v16_wpe2 built for each (n,k)
    print("--- Build phase: ensure baseline + winners exist ---")
    for (m, n, k, comp) in RECHECK_SHAPES:
        for suffix, flags in [(BASELINE_VARIANT, BASELINE_FLAGS)]:
            so, mod, msg = ensure_build(n, k, suffix, flags)
            print(f"  n={n} k={k} {suffix:35s} {msg}")
    # Winners are already built per Phase 4 of the earlier script.

    # 2. Bench serially: baseline + each winner per shape
    print("\n--- Bench phase (serial) ---")
    results = {}
    for (m, n, k, comp) in RECHECK_SHAPES:
        shape_key = f"{m}x{n}x{k}"
        results[shape_key] = {}
        for suffix in [BASELINE_VARIANT] + WINNERS:
            r = bench(m, n, k, suffix, GPU)
            results[shape_key][suffix] = r
            if r and "tflops" in r:
                pct = r["tflops"] / comp * 100
                print(f"  {shape_key:25s} {suffix:35s} {r['tflops']:7.1f} ({pct:.1f}% of comp {comp})")
            else:
                err = r.get("err", "?") if r else "no_so"
                print(f"  {shape_key:25s} {suffix:35s} FAILED ({err})")

    # 3. Compare deltas
    print("\n--- Delta vs baseline ---")
    print(f"{'shape':<25s} {'baseline':>10s}  " + "  ".join(f"{w[-22:]:>22s}" for w in WINNERS))
    for (m, n, k, comp) in RECHECK_SHAPES:
        shape_key = f"{m}x{n}x{k}"
        rb = results[shape_key].get(BASELINE_VARIANT)
        bt = rb["tflops"] if rb and "tflops" in rb else None
        line = f"{shape_key:<25s} {bt or 'FAIL':>10}"
        for w in WINNERS:
            rw = results[shape_key].get(w)
            wt = rw["tflops"] if rw and "tflops" in rw else None
            if bt and wt:
                d = (wt - bt) / comp * 100
                line += f"  {wt:7.1f}({d:+.2f}pp){' ':>4}"
            else:
                line += f"  {wt or 'FAIL':>22}"
        print(line)

    out = os.path.join(SCRIPT_DIR, "spot_optB_r6_rebench_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
