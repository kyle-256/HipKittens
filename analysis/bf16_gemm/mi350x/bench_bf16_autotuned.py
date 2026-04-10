"""BF16 GEMM: HipKittens (autotuned group_m) vs torch.matmul on LLM shapes."""
import torch, math, json, os, sys, time

torch.manual_seed(42)

sys.path.insert(0, os.path.dirname(__file__))
import tk_bf16_layouts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../fp8_gemm/mi350x"))
from bench_vs_hipblaslt import DenseModelConfigs, gen_gemm_test_cases

WARMUP, ITERS = 30, 50
BLK = tk_bf16_layouts.BLOCK_SIZE
K_STEP = tk_bf16_layouts.K_STEP


def _bench_one(fn, M, N, K):
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP): fn()
    ts = []
    for _ in range(ITERS):
        torch.cuda.synchronize(); se.record(); fn(); ee.record()
        torch.cuda.synchronize(); ts.append(se.elapsed_time(ee))
    return 2.0 * M * N * K / (sum(ts) / len(ts) * 1e9)


def autotune_gm(M, N, K, A, B, C):
    best_gm, best_tf = 4, 0
    for gm in [1, 2, 4, 8, 16]:
        fn = lambda: tk_bf16_layouts.gemm_rcr(A, B, C, gm)
        for _ in range(10): fn()
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(15):
            torch.cuda.synchronize(); se.record(); fn(); ee.record()
            torch.cuda.synchronize(); ts.append(se.elapsed_time(ee))
        tf = 2.0 * M * N * K / (sum(ts) / len(ts) * 1e9)
        if tf > best_tf:
            best_tf, best_gm = tf, gm
    return best_gm


all_shapes = set()
for config in DenseModelConfigs.values():
    for mbs in [1, 2]:
        for _, seq, n, k in gen_gemm_test_cases(config):
            M = seq * mbs
            if M % BLK == 0 and n % BLK == 0 and k % K_STEP == 0:
                all_shapes.add((M, n, k))
shapes = sorted(all_shapes)

print(f"BF16 GEMM benchmark (autotuned): {len(shapes)} unique shapes, RCR layout")
print(f"{'Model':>16} {'Op':>12} {'MBS':>3} {'M':>5} {'N':>6} {'K':>5} {'gm':>3} {'TK':>8} {'torch':>8} {'ratio':>7}")
print("-" * 90)

gm_cache = {}
stats = []

for model_name in DenseModelConfigs:
    config = DenseModelConfigs[model_name]
    for mbs in [1, 2]:
        for op_name, seq, n, k in gen_gemm_test_cases(config):
            M = seq * mbs
            if M % BLK != 0 or n % BLK != 0 or k % K_STEP != 0:
                continue

            A = torch.randn(M, k, dtype=torch.bfloat16, device="cuda")
            B = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
            C = torch.zeros(M, n, dtype=torch.bfloat16, device="cuda")

            key = f"{M}_{n}_{k}"
            if key not in gm_cache:
                gm_cache[key] = autotune_gm(M, n, k, A, B, C)
            gm = gm_cache[key]

            tk_tf = _bench_one(lambda: tk_bf16_layouts.gemm_rcr(A, B, C, gm), M, n, k)
            torch_tf = _bench_one(lambda: torch.mm(A, B.T), M, n, k)

            ratio = tk_tf / torch_tf if torch_tf > 0 else 0
            stats.append(ratio)
            win = "*" if ratio >= 1.0 else " "
            print(f"{model_name:>16} {op_name:>12} {mbs:>3} {M:>5} {n:>6} {k:>5} {gm:>3} {tk_tf:>8.1f} {torch_tf:>8.1f} {ratio:>6.3f}x{win}")
            del A, B, C; torch.cuda.empty_cache()

print("=" * 90)
if stats:
    geo = math.exp(sum(math.log(v) for v in stats) / len(stats))
    wins = sum(1 for v in stats if v >= 1.0)
    print(f"Geo-mean vs torch.mm: {geo:.4f}x  ({wins}/{len(stats)} wins)")
    print(f"group_m cache: {json.dumps(gm_cache, indent=2)}")
