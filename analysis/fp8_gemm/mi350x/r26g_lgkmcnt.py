#!/usr/bin/env python3
"""R26-G: STEP12_BR_LGKMCNT × R25 stack on 4 high-value shapes.

Hypothesis: STEP12_BR_LGKMCNT (s_waitcnt lgkmcnt(N) between Step 12 LDS read for
next K-pair and the consumer MFMA) gates LDS-read vs MFMA-consumption pairing.
Default in current best variants is 2 (lgk2 family). Values {1, 3} are unexplored.

For DLA7/SD/SE (already have STEP12_BR_LGKMCNT=2): test lgk=1, lgk=3 + verify lgk=2.
For SC (no STEP12_BR_LGKMCNT, kernel default 0): test lgk=1, lgk=2, lgk=3 + verify lgk=0.

Standard MXFP4 bench: warmup=200, iters=500, trim=10%.
"""
import json
import os
import subprocess
import sys
import sysconfig
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
KERNEL_SRC = os.path.join(SCRIPT_DIR, "kernel_mxfp4_gluon_cpp.cpp")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_all42")
TK_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
)

WARMUP = 200
ITERS = 500
TRIM = 0.10
N_REPS = 5
GPU_ID = 4
VARIANT_TIMEOUT = 600

# 4 shapes (M, N, K, comp_tflops) with their R25-best variants.
# default_lgk: kernel-current value when no -DSTEP12_BR_LGKMCNT=... overrides.
# 0 = kernel default (no lgk macro in base flags), 2 = base flags include LGK=2.
SHAPES = [
    {
        "name": "DLA7",
        "M": 28672, "N": 32768, "K": 4096, "comp": 4466.6,
        "base_suffix": "_ts_lgk2_gm7_v12_memc_pfoff14",
        # base_flags WITHOUT the LGK macro — we add -DSTEP12_BR_LGKMCNT=N below
        "base_flags_no_lgk": (
            "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 "
            "-DR25C_TAIL_PF_OFF_ITERS=14 -DR25C_K_LIMIT=32768 "
            "-mllvm -amdgpu-sched-strategy=max-memory-clause"
        ),
        "default_lgk": 2,
        "test_lgk": [1, 2, 3],  # 2 is verify
    },
    {
        "name": "SD",
        "M": 16384, "N": 4096, "K": 28672, "comp": 5525.3,
        "base_suffix": "_ts_lgk2_gm7_memc_pfoff104_kx28672_btw_all",
        "base_flags_no_lgk": (
            "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 "
            "-DR25C_TAIL_PF_OFF_ITERS=104 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=28672 "
            "-mllvm -amdgpu-sched-strategy=max-memory-clause "
            "-DBARRIER_TO_WAITCNT_ALL=1"
        ),
        "default_lgk": 2,
        "test_lgk": [1, 2, 3],
    },
    {
        "name": "SC",
        "M": 4096, "N": 32768, "K": 6144, "comp": 4548.6,
        "base_suffix": "_ts_v12_gm7_memc_pfoff19_kx6144_btw_all",
        # SC has NO -DSTEP12_BR_LGKMCNT in base flags → default is kernel default 0.
        "base_flags_no_lgk": (
            "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 -DSTEP3_BARRIER_VMCNT=12 "
            "-DR25C_TAIL_PF_OFF_ITERS=19 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=6144 "
            "-mllvm -amdgpu-sched-strategy=max-memory-clause "
            "-DBARRIER_TO_WAITCNT_ALL=1"
        ),
        "default_lgk": 0,  # kernel default since base_flags don't override
        "test_lgk": [0, 1, 2, 3],  # 0 is verify (no macro), test 1/2/3
    },
    {
        "name": "SE",
        "M": 28672, "N": 4096, "K": 16384, "comp": 5409.1,
        "base_suffix": "_ts_lgk2_gm7_memc_pfoff56_kx16384_btw_all",
        "base_flags_no_lgk": (
            "-DTAIL_SPLIT=1 -DGROUP_SIZE_M=7 "
            "-DR25C_TAIL_PF_OFF_ITERS=56 -DR25C_K_LIMIT=32768 -DR25C_K_EXACT=16384 "
            "-mllvm -amdgpu-sched-strategy=max-memory-clause "
            "-DBARRIER_TO_WAITCNT_ALL=1"
        ),
        "default_lgk": 2,
        "test_lgk": [1, 2, 3],
    },
]


def variant_suffix(shape, lgk):
    return f"{shape['base_suffix']}_r26g_lgk{lgk}"


def variant_flags(shape, lgk):
    base = shape["base_flags_no_lgk"]
    if lgk == 0:
        # No macro override → kernel default (which is 0). For shapes whose default is 0
        # this matches their existing best; for shapes whose default is 2 we still
        # explicitly omit the macro to test lgk=0 if needed (not in test_lgk for them).
        return base
    return f"{base} -DSTEP12_BR_LGKMCNT={lgk}"


def module_name_for(shape, lgk):
    return f"tk_mxfp4_gluon_cpp_n{shape['N']}_k{shape['K']}{variant_suffix(shape, lgk)}"


def build_one(shape, lgk):
    n_dim = shape["N"]
    k_dim = shape["K"]
    suffix = variant_suffix(shape, lgk)
    cppflags = variant_flags(shape, lgk)
    module_name = module_name_for(shape, lgk)
    out_file = f"{module_name}{EXT_SUFFIX}"
    out_path = os.path.join(BUILD_DIR, out_file)

    if os.path.exists(out_path):
        return ("CACHED", shape["name"], lgk, out_path, 0.0, "")

    with open(KERNEL_SRC, "r") as f:
        src = f.read()
    patched = src.replace(
        "PYBIND11_MODULE(tk_mxfp4_gluon_cpp,",
        f"PYBIND11_MODULE({module_name},"
    )
    wrapper_src = os.path.join(BUILD_DIR, f"wrap_n{n_dim}_k{k_dim}{suffix}.cpp")
    with open(wrapper_src, "w") as f:
        f.write(patched)

    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = TK_ROOT

    cmd = (
        f'make -C {SCRIPT_DIR} TARGET={os.path.join(BUILD_DIR, module_name)} '
        f'SRC={wrapper_src} '
        f'CPPFLAGS="-DK_DIM={k_dim} -DN_DIM={n_dim} {cppflags}"'
    )
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0 or not os.path.exists(out_path):
        return ("FAIL", shape["name"], lgk, out_path, elapsed, result.stderr[-2000:])
    return ("OK", shape["name"], lgk, out_path, elapsed, "")


def bench_one(shape, lgk):
    suffix = variant_suffix(shape, lgk)
    module_name = module_name_for(shape, lgk)
    so_path = os.path.join(BUILD_DIR, f"{module_name}{EXT_SUFFIX}")
    if not os.path.exists(so_path):
        return None

    M, N, K = shape["M"], shape["N"], shape["K"]
    script = f"""
import sys, math, torch, importlib.util, json
torch.manual_seed(0)
WARMUP, ITERS, TRIM = {WARMUP}, {ITERS}, {TRIM}
M, N, K = {M}, {N}, {K}
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
print(json.dumps({{"tflops":round(t,2),"ms":round(avg,4)}}))
"""
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(GPU_ID)
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=VARIANT_TIMEOUT, env=env)
        if r.returncode != 0:
            sys.stderr.write(f"[bench fail lgk={lgk}] {r.stderr[-500:]}\n")
            return None
        return json.loads(r.stdout.strip())
    except Exception as e:
        sys.stderr.write(f"[bench exc lgk={lgk}] {e}\n")
        return None


def do_build():
    os.makedirs(BUILD_DIR, exist_ok=True)
    tasks = [(s, lgk) for s in SHAPES for lgk in s["test_lgk"]]
    print(f"=== R26-G BUILD: {len(tasks)} variants ===")
    print(f"Build dir: {BUILD_DIR}")

    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(build_one, s, lgk): (s["name"], lgk) for s, lgk in tasks}
        for fut in as_completed(futs):
            name, lgk = futs[fut]
            try:
                status, sh_name, lgk_v, path, elapsed, err = fut.result()
                if status == "FAIL":
                    print(f"  [{sh_name} lgk{lgk_v}] FAIL ({elapsed:.1f}s)")
                    print(f"    stderr: {err[-500:]}")
                else:
                    print(f"  [{sh_name} lgk{lgk_v}] {status} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  [{name} lgk{lgk}] EXC: {e}")


def do_bench():
    print(f"\n=== R26-G BENCH: GPU {GPU_ID}, {N_REPS} reps, warmup={WARMUP} iters={ITERS} trim={TRIM} ===")
    results = {}
    for s in SHAPES:
        for lgk in s["test_lgk"]:
            results[(s["name"], lgk)] = []

    for rep in range(N_REPS):
        print(f"\n--- rep {rep+1}/{N_REPS} ---")
        for s in SHAPES:
            for lgk in s["test_lgk"]:
                b = bench_one(s, lgk)
                if b is None:
                    print(f"  [{s['name']} lgk{lgk}] FAIL", flush=True)
                else:
                    results[(s["name"], lgk)].append(b["tflops"])
                    pct = b["tflops"] / s["comp"] * 100
                    print(f"  [{s['name']} lgk{lgk}] {b['tflops']:7.2f} TFLOPS ({pct:.2f}%)", flush=True)

    print("\n" + "=" * 80)
    print("R26-G SUMMARY (mean / std / min / max over reps; * = current default)")
    print("=" * 80)
    summary = {}
    for s in SHAPES:
        print(f"\n{s['name']}  ({s['M']}x{s['N']}x{s['K']})  comp={s['comp']:.1f}  default_lgk={s['default_lgk']}")
        print(f"  {'lgk':>4s} {'mean':>8s} {'std':>7s} {'min':>8s} {'max':>8s} {'%comp':>8s}  reps")
        sh_results = []
        for lgk in s["test_lgk"]:
            ts = results[(s["name"], lgk)]
            if not ts:
                print(f"  {lgk:>4d}    FAIL")
                continue
            mean = sum(ts) / len(ts)
            std = (sum((x - mean) ** 2 for x in ts) / len(ts)) ** 0.5 if len(ts) > 1 else 0.0
            pct = mean / s["comp"] * 100
            marker = " *" if lgk == s["default_lgk"] else "  "
            print(f"  {lgk:>4d}{marker}{mean:>7.1f} {std:>7.1f} {min(ts):>8.1f} {max(ts):>8.1f} {pct:>7.2f}%  {ts}")
            sh_results.append({"lgk": lgk, "mean": mean, "std": std, "min": min(ts), "max": max(ts), "pct_comp": pct, "reps": ts})
        summary[s["name"]] = {
            "shape": [s["M"], s["N"], s["K"]],
            "comp": s["comp"],
            "default_lgk": s["default_lgk"],
            "results": sh_results,
        }

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    for s in SHAPES:
        sh = summary.get(s["name"])
        if not sh or not sh["results"]:
            print(f"  [{s['name']}] DEAD (no results)")
            continue
        cur = next((r for r in sh["results"] if r["lgk"] == s["default_lgk"]), None)
        if cur is None:
            print(f"  [{s['name']}] no default_lgk result")
            continue
        cur_pct = cur["pct_comp"]
        best = max((r for r in sh["results"] if r["lgk"] != s["default_lgk"]), key=lambda r: r["mean"], default=None)
        if best is None:
            print(f"  [{s['name']}] DEAD (only default tested)")
            continue
        delta_pp = best["pct_comp"] - cur_pct
        if delta_pp >= 1.5 and best["std"] <= 25.0:
            print(f"  [{s['name']}] WIN: lgk{best['lgk']} {best['mean']:.1f} TFLOPS ({best['pct_comp']:.2f}%) vs default lgk{s['default_lgk']} {cur['mean']:.1f} ({cur_pct:.2f}%) Δ=+{delta_pp:.2f}pp std={best['std']:.1f}")
        else:
            print(f"  [{s['name']}] DEAD: best lgk{best['lgk']} {best['mean']:.1f} ({best['pct_comp']:.2f}%) vs default lgk{s['default_lgk']} {cur['mean']:.1f} ({cur_pct:.2f}%) Δ={delta_pp:+.2f}pp std={best['std']:.1f}")

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": GPU_ID, "warmup": WARMUP, "iters": ITERS, "trim": TRIM, "reps": N_REPS,
        "summary": summary,
    }
    out_path = os.path.join(SCRIPT_DIR, "r26g_lgkmcnt_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("build", "all"):
        do_build()
    if mode in ("bench", "all"):
        do_bench()


if __name__ == "__main__":
    main()
