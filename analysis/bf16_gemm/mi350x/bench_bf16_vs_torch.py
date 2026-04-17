"""BF16 GEMM: HipKittens vs torch.matmul (hipBLASLt) on LLM shapes.

Single compiled .so per node; K-specialization via template dispatch (no per-shape JIT).
Runtime autotune of group_m ∈ {1,2,4,8,16} per (shape, layout).
"""
import torch, math, json, os, sys, time
torch.manual_seed(42)

sys.path.insert(0, os.path.dirname(__file__))
import tk_bf16_layouts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../fp8_gemm/mi350x"))
from bench_vs_hipblaslt import DenseModelConfigs, gen_gemm_test_cases

WARMUP, ITERS = 20, 40
GM_SEARCH = [1, 2, 4, 8, 16]
# Best-of-N repeats to reject launch/DVFS noise: per-gm we time ITERS kernel
# launches NREPEAT times and take the min of each set, then the min across
# repeats. Min is used because GEMM time is lower-bounded by hardware and any
# variance adds latency (never removes it).
NREPEAT = 3
BLK = tk_bf16_layouts.BLOCK_SIZE
K_STEP = tk_bf16_layouts.K_STEP


def _time(run):
    for _ in range(WARMUP): run()
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(NREPEAT):
        ts = []
        for _ in range(ITERS):
            torch.cuda.synchronize(); se.record(); run(); ee.record()
            torch.cuda.synchronize(); ts.append(se.elapsed_time(ee))
        # Use 20th-percentile to reject outlier tails on both ends.
        ts.sort()
        lo = ts[len(ts)//5]
        if lo < best: best = lo
    return best


def bench_tk(M, N, K, layout):
    if layout == "rcr":
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    elif layout == "rrr":
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    else:
        A = torch.randn(K, M, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    fn_map = {"rcr": tk_bf16_layouts.gemm_rcr, "rrr": tk_bf16_layouts.gemm_rrr,
              "crr": tk_bf16_layouts.gemm_crr}
    fn = fn_map[layout]

    best_ms, best_gm = float("inf"), 4
    for gm in GM_SEARCH:
        run = lambda g=gm: fn(A, B, C, g)
        ms = _time(run)
        if ms < best_ms:
            best_ms, best_gm = ms, gm
    return 2.0 * M * N * K / (best_ms * 1e9), best_gm


def bench_torch(M, N, K, layout):
    if layout == "rcr":
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        run = lambda: torch.mm(A, B.T)
    elif layout == "rrr":
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        run = lambda: torch.mm(A, B)
    else:
        A = torch.randn(K, M, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        run = lambda: torch.mm(A.T, B)
    ms = _time(run)
    return 2.0 * M * N * K / (ms * 1e9)


all_shapes = set()
for config in DenseModelConfigs.values():
    for mbs in [1, 2]:
        for _, seq, n, k in gen_gemm_test_cases(config):
            M = seq * mbs
            if M % BLK == 0 and n % BLK == 0 and k % K_STEP == 0:
                all_shapes.add((M, n, k))
shapes = sorted(all_shapes)

LAYOUTS = ["rcr", "rrr", "crr"]
print(f"BF16 GEMM benchmark: {len(shapes)} shapes × {len(LAYOUTS)} layouts vs torch.mm (hipBLASLt)")
print(f"{'M':>5} {'N':>6} {'K':>5}", end="")
for lay in LAYOUTS:
    print(f" | {lay.upper():>3} {'TK':>7} {'gm':>2} {'torch':>7} {'ratio':>6}", end="")
print()
print("-" * 110)

stats_by_layout = {l: [] for l in LAYOUTS}
all_rows = []
for M, n, k in shapes:
    row = {"M": M, "N": n, "K": k}
    line = f"{M:>5} {n:>6} {k:>5}"
    for layout in LAYOUTS:
        tk_tf, gm = bench_tk(M, n, k, layout)
        torch_tf = bench_torch(M, n, k, layout)
        ratio = tk_tf / torch_tf if torch_tf > 0 else 0
        stats_by_layout[layout].append(ratio)
        win = "*" if ratio >= 1.0 else " "
        row[f"{layout}_tk"] = tk_tf
        row[f"{layout}_torch"] = torch_tf
        row[f"{layout}_gm"] = gm
        row[f"{layout}_ratio"] = ratio
        line += f" | {layout.upper():>3} {tk_tf:>7.1f} {gm:>2} {torch_tf:>7.1f} {ratio:>.3f}x{win}"
    all_rows.append(row)
    print(line, flush=True)

print()
for layout in LAYOUTS:
    s = stats_by_layout[layout]
    if s:
        geo = math.exp(sum(math.log(v) for v in s) / len(s))
        wins = sum(1 for v in s if v >= 1.0)
        print(f"{layout.upper()} geo-mean vs torch.mm: {geo:.4f}x  ({wins}/{len(s)} wins)")

with open(os.path.join(os.path.dirname(__file__), "bench_bf16_no_jit_final.json"), "w") as f:
    summary = {"rows": all_rows,
               "summary": {lay: {
                   "geo_mean": math.exp(sum(math.log(v) for v in stats_by_layout[lay]) / len(stats_by_layout[lay])),
                   "wins": sum(1 for v in stats_by_layout[lay] if v >= 1.0),
                   "total": len(stats_by_layout[lay]),
               } for lay in LAYOUTS}}
    json.dump(summary, f, indent=2)
print("Saved bench_bf16_no_jit_final.json")
