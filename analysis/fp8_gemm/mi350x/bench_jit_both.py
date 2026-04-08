import subprocess, json, os, math, sys

DIR = os.path.dirname(os.path.abspath(__file__))
TK = os.environ.get("THUNDERKITTENS_ROOT", os.path.abspath(os.path.join(DIR, "..", "..", "..")))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
PY_INC = subprocess.check_output(["python3", "-m", "pybind11", "--includes"], text=True).strip()
PY_LD = subprocess.check_output(["python3-config", "--ldflags"], text=True).strip().replace("-lcrypt","")
EXT = subprocess.check_output(["python3-config", "--extension-suffix"], text=True).strip()

shapes = [
    (4096,28672,4096),(8192,28672,4096),(8192,16384,16384),(16384,16384,16384),
    (8192,57344,8192),(4096,57344,8192),(16384,106496,16384),(8192,16384,53248),
]
configs = [
    ("4wave", "-DRCR_USE_EXACT_4WAVE_FASTPATH=1 -DRCR_USE_EXACT_8WAVE_FASTPATH=0"),
    ("8wave", "-DRCR_USE_EXACT_4WAVE_FASTPATH=0 -DRCR_USE_EXACT_8WAVE_FASTPATH=1"),
]

with open(os.path.join(DIR, "bench_vs_hipblaslt_clean_gpu4.json")) as f:
    bl_data = {f'{r["M"]}_{r["N"]}_{r["K"]}': r["bl_tflops"]
               for r in json.load(f)["results"] if r["layout"] == "rcr"}

bench = 'import torch,tk_fp8_layouts;M,N,K={M},{N},{K};A=(torch.randn(M,K,device="cuda")*0.1).to(torch.float8_e4m3fn);B=(torch.randn(N,K,device="cuda")*0.1).to(torch.float8_e4m3fn);C=torch.zeros(M,N,dtype=torch.bfloat16,device="cuda");fn=lambda:tk_fp8_layouts.gemm_rcr(A,B,C,1.0,1.0,4);[fn()for _ in range(50)];se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[];[(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))for _ in range(80)];print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'

results = {c[0]: {} for c in configs}
for cname, cflags in configs:
    for M, N, K in shapes:
        outdir = f"/tmp/jit_{cname}"
        os.makedirs(outdir, exist_ok=True)
        so = os.path.join(outdir, f"tk_fp8_layouts{EXT}")
        cmd = ["/opt/rocm/bin/hipcc", os.path.join(DIR, "kernel_jit_rcr.cpp"),
               "-DKITTENS_CDNA4", "--offload-arch=gfx950", "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
               "-ffast-math", "-I/opt/rocm/include/rocrand",
               f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DK_DIM={K}", "-DRCR_STEADY_VMCNT=8",
               *cflags.split(), "-std=c++20", "-w", "-shared", "-fPIC",
               f"-I{TK}/include", f"-I{TK}/prototype", "-I/opt/rocm/include/hip",
               *PY_INC.split(), *PY_LD.split(), "-o", so]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"[{cname}] ({M},{N},{K}) COMPILE FAIL")
            continue
        env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": outdir}
        r = subprocess.run(["python3", "-c", bench.format(M=M,N=N,K=K)],
                          capture_output=True, text=True, cwd=outdir, env=env, timeout=60)
        if r.returncode != 0:
            print(f"[{cname}] ({M},{N},{K}) RUN FAIL")
            continue
        tf = float(r.stdout.strip())
        results[cname][f"{M}_{N}_{K}"] = tf
        bl = bl_data.get(f"{M}_{N}_{K}", 0)
        ratio = tf/bl if bl > 0 else 0
        w = "**" if ratio >= 1.0 else "  "
        print(f"[{cname:>5}] ({M:>5},{N:>6},{K:>5}) {tf:>7.1f}  BL={bl:>7.1f}  {ratio:.3f}x {w}")

print(f"\n{'Shape':>25}", end="")
for c,_ in configs:
    print(f" {c:>10}", end="")
print(f" {'hipBLASLt':>10}")
print("-"*60)
for M,N,K in shapes:
    k = f"{M}_{N}_{K}"
    print(f"({M:>5},{N:>6},{K:>5})", end="")
    for c,_ in configs:
        print(f" {results[c].get(k,0):>9.1f}", end="")
    print(f" {bl_data.get(k,0):>9.1f}")
