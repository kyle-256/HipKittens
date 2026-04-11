"""
Benchmark already-compiled extended CRR params from .crr_ext_sweep/.
Tests all L/I/V combos for shapes already compiled (from killed sweep_crr_insert_ext.py).
Quick params: 20w/30i.
"""
import subprocess, os, sys, json, itertools

DIR = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "4")
EXT = subprocess.check_output(["python3-config", "--extension-suffix"], text=True).strip()

TMPDIR = os.path.join(DIR, ".crr_ext_sweep")
CACHE  = os.path.join(DIR, ".jit_cache")
WARMUP, ITERS = 20, 30

bench_tpl = '''import torch,tk_fp8_layouts as m
M,N,K={M},{N},{K}
A=(torch.randn(K,M,device='cuda')*0.1).to(torch.float8_e4m3fn)
B=(torch.randn(K,N,device='cuda')*0.1).to(torch.float8_e4m3fn)
C=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
fn=lambda:m.gemm_crr(A,B,C,1.0,1.0,{gm})
for _ in range({w}):C.zero_();fn()
se=torch.cuda.Event(enable_timing=True);ee=torch.cuda.Event(enable_timing=True);ts=[]
for _ in range({it}):(torch.cuda.synchronize(),se.record(),fn(),ee.record(),torch.cuda.synchronize(),ts.append(se.elapsed_time(ee)))
print(f"{{2.0*M*N*K/(sum(ts)/len(ts)*1e9):.1f}}")'''

CRR_GROUP_M = {
    ( 4096,   4096, 14336):  8,
    ( 4096,   6144,  4096):  8,
    ( 4096,   8192,  8192):  8,
    ( 4096,  10240,  8192):  8,
    ( 4096,  22016,  4096):  8,
    ( 4096,  28672,  4096):  8,
    ( 8192,   3584, 18944):  8,
    ( 8192,   8192,  8192):  8,
    ( 8192,   8192, 29568):  8,
    ( 8192,  10240,  8192):  8,
    ( 8192,  12288,  4096):  8,
    ( 8192,  37888,  3584):  8,
    (16384,   3584,  3584):  8,
    (16384,   4096,  4096):  8,
    (16384,   4608,  3584):  8,
    (16384,   6144,  4096):  8,
    (16384,  10240,  8192):  8,
    (16384,  28672,  4096):  8,
    (16384,  37888,  3584):  8,
    (16384,  59136,  8192):  8,
}

CRR_SHAPE_PARAMS = {
    ( 8192,  16384, 16384): (2, 3, 5),
    ( 8192,  16384, 53248): (2, 3, 5),
    ( 8192, 106496, 16384): (4, 3, 4),
    (16384,  16384, 53248): (3, 4, 5),
    (16384, 106496, 16384): (2, 3, 5),
}

def get_current_so(M, N, K):
    opt = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave_opt")
    if os.path.exists(os.path.join(opt, f"tk_fp8_layouts{EXT}")):
        return opt, "opt"
    jit = os.path.join(CACHE, f"crr_{M}x{N}x{K}_8wave")
    if os.path.exists(os.path.join(jit, f"tk_fp8_layouts{EXT}")):
        return jit, "8wave"
    return os.path.join(CACHE, "crr_shared"), "shared"

def bench(M, N, K, so_dir, gm):
    script = bench_tpl.format(M=M, N=N, K=K, gm=gm, w=WARMUP, it=ITERS)
    env = {**os.environ, "HIP_VISIBLE_DEVICES": GPU, "PYTHONPATH": so_dir}
    r = subprocess.run(["python3", "-c", script], capture_output=True, text=True,
                       cwd=so_dir, env=env, timeout=90)
    if r.returncode == 0:
        try:
            return float(r.stdout.strip())
        except ValueError:
            pass
    return 0.0

# Find which shapes have compiled SOs in TMPDIR
compiled = {}
for name in os.listdir(TMPDIR):
    so = os.path.join(TMPDIR, name, f"tk_fp8_layouts{EXT}")
    if not os.path.exists(so):
        continue
    # Parse: crr_{M}x{N}x{K}_L{l}_I{i}_V{v}
    parts = name.replace("crr_", "").split("_")
    try:
        dims = parts[0].split("x")
        M, N, K = int(dims[0]), int(dims[1]), int(dims[2])
        l = int(parts[1][1:])
        i = int(parts[2][1:])
        v = int(parts[3][1:])
        compiled.setdefault((M, N, K), []).append((l, i, v))
    except (IndexError, ValueError):
        pass

shapes = sorted(compiled.keys())
print(f"Found {len(shapes)} shapes with compiled SOs in {TMPDIR}")
for s in shapes:
    print(f"  {s[0]}x{s[1]}x{s[2]}: {len(compiled[s])} combos")
print()

# Load reference results
results_file = os.path.join(DIR, "bench_jit_full_results.json")
ref_crr, ref_rcr = {}, {}
if os.path.exists(results_file):
    with open(results_file) as f:
        data = json.load(f)
        ref_crr = data.get("crr", {})
        ref_rcr = data.get("rcr", {})

gains = {}
for M, N, K in shapes:
    gm = CRR_GROUP_M.get((M, N, K), 4)
    cur_so, cur_type = get_current_so(M, N, K)
    cur_params = CRR_SHAPE_PARAMS.get((M, N, K), None)
    key = f"{M}_{N}_{K}"
    ref = ref_crr.get(key, 0)
    rcr = ref_rcr.get(key, 0)

    print(f"\n=== {M}x{N}x{K} (gm={gm}, ref_crr={ref:.1f}, rcr={rcr:.1f}, ratio={ref/rcr:.3f}x) ===")
    if cur_params:
        print(f"  Current params: L={cur_params[0]} I={cur_params[1]} V={cur_params[2]}")
    else:
        print(f"  Current params: (4,4,4) default")

    r_cur = bench(M, N, K, cur_so, gm)
    print(f"  Current SO: {r_cur:.1f} TFLOPS")

    best_tf = r_cur
    best_combo = None
    results_table = []

    for l, i, v in sorted(compiled[(M, N, K)]):
        so_dir = os.path.join(TMPDIR, f"crr_{M}x{N}x{K}_L{l}_I{i}_V{v}")
        tf = bench(M, N, K, so_dir, gm)
        delta = tf - r_cur
        results_table.append((l, i, v, tf, delta))
        if tf > best_tf:
            best_tf = tf
            best_combo = (l, i, v)

    # Print top 10 combos
    results_table.sort(key=lambda x: -x[3])
    print(f"  Top 10 combos:")
    for l, i, v, tf, delta in results_table[:10]:
        flag = " ← BEST" if (l, i, v) == best_combo else ""
        flag2 = " ← GAIN" if delta > 15 else (" ← REGRESS" if delta < -15 else "")
        print(f"    L={l} I={i} V={v}: {tf:.1f} ({delta:+.1f}){flag}{flag2}")
    print(f"  ... worst: {results_table[-1][3]:.1f} ({results_table[-1][4]:+.1f})")

    if best_combo:
        delta = best_tf - r_cur
        print(f"  WINNER: L={best_combo[0]} I={best_combo[1]} V={best_combo[2]} → +{delta:.1f} TFLOPS")
        gains[(M, N, K)] = (best_combo, best_tf, r_cur, delta)
    else:
        print(f"  No improvement found")
    sys.stdout.flush()

print("\n" + "=" * 60)
print("SUMMARY: Shapes with gain > 15 TFLOPS:")
for (M, N, K), ((l, i, v), best, cur, delta) in sorted(gains.items(), key=lambda x: -x[1][3]):
    print(f"  {M}x{N}x{K}: L={l} I={i} V={v} → {cur:.1f}→{best:.1f} (+{delta:.1f} T, +{delta/cur*100:.2f}%)")
    print(f"    ({M:>5}, {N:>6}, {K:>5}): ({l}, {i}, {v}),")
if not gains:
    print("  None found — INSERT=6,7,8 / VMCNT=7,8 provide no gains over current params")
