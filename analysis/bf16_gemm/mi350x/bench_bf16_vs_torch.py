"""BF16 GEMM: HipKittens vs torch.matmul (hipBLASLt) on LLM shapes."""
import torch, math, json, os, sys, time

torch.manual_seed(42)

sys.path.insert(0, os.path.dirname(__file__))
import tk_bf16_layouts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../fp8_gemm/mi350x"))
from bench_vs_hipblaslt import DenseModelConfigs, gen_gemm_test_cases

WARMUP, ITERS = 15, 30
BLK = tk_bf16_layouts.BLOCK_SIZE
K_STEP = tk_bf16_layouts.K_STEP

def bench_tk(M, N, K, layout="rcr", gm=8):
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
    run = lambda: fn(A, B, C, gm)

    for _ in range(WARMUP): run()
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(ITERS):
        torch.cuda.synchronize(); se.record(); run(); ee.record()
        torch.cuda.synchronize(); ts.append(se.elapsed_time(ee))
    return 2.0 * M * N * K / (sum(ts) / len(ts) * 1e9)


def bench_torch(M, N, K, layout="rcr"):
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

    for _ in range(WARMUP): run()
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(ITERS):
        torch.cuda.synchronize(); se.record(); run(); ee.record()
        torch.cuda.synchronize(); ts.append(se.elapsed_time(ee))
    return 2.0 * M * N * K / (sum(ts) / len(ts) * 1e9)


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
print(f"{'Model':>16} {'Op':>12} {'MBS':>3} {'M':>5} {'N':>6} {'K':>5} {'Lay':>4} {'TK':>8} {'torch':>8} {'ratio':>7}")
print("-" * 93)

stats_by_layout = {l: [] for l in LAYOUTS}
for model_name in DenseModelConfigs:
    config = DenseModelConfigs[model_name]
    for mbs in [1, 2]:
        for op_name, seq, n, k in gen_gemm_test_cases(config):
            M = seq * mbs
            if M % BLK != 0 or n % BLK != 0 or k % K_STEP != 0:
                continue
            for layout in LAYOUTS:
                tk_tf = bench_tk(M, n, k, layout)
                torch_tf = bench_torch(M, n, k, layout)
                ratio = tk_tf / torch_tf if torch_tf > 0 else 0
                stats_by_layout[layout].append(ratio)
                win = "*" if ratio >= 1.0 else " "
                print(f"{model_name:>16} {op_name:>12} {mbs:>3} {M:>5} {n:>6} {k:>5} {layout:>4} {tk_tf:>8.1f} {torch_tf:>8.1f} {ratio:>6.3f}x{win}")

print()
for layout in LAYOUTS:
    s = stats_by_layout[layout]
    if s:
        geo = math.exp(sum(math.log(v) for v in s) / len(s))
        wins = sum(1 for v in s if v >= 1.0)
        print(f"{layout.upper()} geo-mean vs torch.mm: {geo:.4f}x  ({wins}/{len(s)} wins)")
