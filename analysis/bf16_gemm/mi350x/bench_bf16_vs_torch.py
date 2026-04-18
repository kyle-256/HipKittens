"""BF16 GEMM: HipKittens vs torch.matmul (hipBLASLt) on LLM shapes.

Single compiled .so per node; K-specialization via template dispatch (no per-shape JIT).
Runtime autotune of group_m ∈ {1,2,4,8,16} per (shape, layout).
"""
import torch, math, json, os, sys, time
torch.manual_seed(42)

sys.path.insert(0, os.path.dirname(__file__))
import tk_bf16_layouts

# Inlined from analysis/fp8_gemm/mi350x/bench_vs_hipblaslt.py to avoid the
# tk_fp8_layouts .so import dependency (FP8 build not required for BF16 bench).
DenseModelConfigs = {
    "Llama-2-7B":     {"seqlen": 4096, "hidden_size": 4096,  "intermediate_size": 11008,
                       "num_attention_heads": 32,  "num_key_value_heads": 32, "head_dim": 128},
    "Llama-2-70B":    {"seqlen": 4096, "hidden_size": 8192,  "intermediate_size": 28672,
                       "num_attention_heads": 64,  "num_key_value_heads": 8,  "head_dim": 128},
    "Llama-3.1-8B":   {"seqlen": 8192, "hidden_size": 4096,  "intermediate_size": 14336,
                       "num_attention_heads": 32,  "num_key_value_heads": 8,  "head_dim": 128},
    "Llama-3.1-405B": {"seqlen": 8192, "hidden_size": 16384, "intermediate_size": 53248,
                       "num_attention_heads": 128, "num_key_value_heads": 8,  "head_dim": 128},
    "Qwen2.5-7B":     {"seqlen": 8192, "hidden_size": 3584,  "intermediate_size": 18944,
                       "num_attention_heads": 28,  "num_key_value_heads": 4,  "head_dim": 128},
    "Qwen2.5-72B":    {"seqlen": 8192, "hidden_size": 8192,  "intermediate_size": 29568,
                       "num_attention_heads": 64,  "num_key_value_heads": 8,  "head_dim": 128},
    "Mistral-7B":     {"seqlen": 4096, "hidden_size": 4096,  "intermediate_size": 14336,
                       "num_attention_heads": 32,  "num_key_value_heads": 8,  "head_dim": 128},
}

def gen_gemm_test_cases(config):
    seq = config["seqlen"]; hs = config["hidden_size"]; inter = config["intermediate_size"]
    nah = config["num_attention_heads"]; nkv = config["num_key_value_heads"]; hd = config["head_dim"]
    return [
        ("attn_qkv",    seq, int((nah + 2 * nkv) * hd), hs),
        ("attn_out",    seq, hs, hs),
        ("mlp_gate_up", seq, int(2 * inter), hs),
        ("mlp_down",    seq, hs, inter),
    ]

WARMUP, ITERS = 20, 40
# P10 (Dev2): Expanded autotune search.
# Discovery: many tall-N losing shapes find +1-6pp wins at gm=24 (a value
# outside the original {1,2,4,8,16} search). Several also benefit from
# xcd=2 or xcd=32 (outside the original {4,8,16}).
# Expanded: gm += {6, 24}, xcd += {2, 32}. Bench runtime grows ~1.7x.
GM_SEARCH = [1, 2, 4, 6, 8, 16, 24]
XCD_SEARCH = [2, 4, 8, 16, 32]
# Best-of-N repeats to reject launch/DVFS noise: per-gm we time ITERS kernel
# launches NREPEAT times and take the min of each set, then the min across
# repeats. Min is used because GEMM time is lower-bounded by hardware and any
# variance adds latency (never removes it).
NREPEAT = 3
BLK = tk_bf16_layouts.BLOCK_SIZE
K_STEP = tk_bf16_layouts.K_STEP

# P23 Step 5: end-to-end MFMA correctness gate. Catches silent operand-mapping
# errors (per Dev C §2.5: a Step-4 implementation that compiles, builds, and
# benches but produces wrong MFMA outputs because ds_read_b128's lane payload
# is rotated 90° vs MFMA col_l A-operand expectation).
#
# Smoke shape: M=N=256, K=128. Notes on shape selection:
#   - Dev C §6.1 sketched (M=N=128, K=64); empirically that shape returns
#     `hipErrorInvalidConfiguration` because BLOCK_SIZE=256 (one tile floor).
#   - Module exports K_STEP=64, but the kernel's actual K-correctness floor
#     is 128 — at K=64, gemm_rcr returns silent garbage (max_abs ~ 33 on
#     unit-variance inputs vs ref). The smoke gate explicitly avoids that
#     trap; Step-4 Devs should treat K=64 as untested.
#   - (M=256, N=256, K=128) is the smallest valid shape: 1 row-block × 1
#     col-block × 1 K-tile, deterministic, ~120 µs round-trip.
#
# Tolerance: rtol=1e-2, atol=K*1e-2 (Dev C §6.2). Reference is FP32-promoted
# torch.mm to remove BF16-vs-BF16 accumulator coupling — we want to catch
# kernel-side mapping errors, not hipBLASLt's BF16 accumulator drift.
SMOKE_M, SMOKE_N, SMOKE_K = 256, 256, 128


def verify_correctness(M, N, K, layout, gm=1, xcd=4, _log_handle=None):
    """Run TK kernel once and compare vs FP32-promoted torch.mm.

    Returns True on pass. Raises RuntimeError on mismatch with first ~10
    mismatched indices, max abs/rel err, and the failing (M,N,K,layout).
    """
    assert M % BLK == 0 and N % BLK == 0, f"smoke shape (M={M},N={N}) not aligned to BLOCK_SIZE={BLK}"
    assert K % 128 == 0, f"smoke shape K={K} below kernel correctness floor (128); see harness comment"
    if layout == "rcr":
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        ref = torch.mm(A.float(), B.float().T)
        fn = tk_bf16_layouts.gemm_rcr
    elif layout == "rrr":
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        ref = torch.mm(A.float(), B.float())
        fn = tk_bf16_layouts.gemm_rrr
    elif layout == "crr":
        A = torch.randn(K, M, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        ref = torch.mm(A.float().T, B.float())
        fn = tk_bf16_layouts.gemm_crr
    else:
        raise ValueError(f"unknown layout {layout!r}")

    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    fn(A, B, C, gm, xcd)
    torch.cuda.synchronize()

    rtol, atol = 1e-2, K * 1e-2
    Cf = C.float()
    diff = (Cf - ref).abs()
    if not torch.allclose(Cf, ref, rtol=rtol, atol=atol):
        max_abs = diff.max().item()
        rel = diff / (ref.abs() + 1e-9)
        max_rel = rel.max().item()
        # First ~10 mismatched indices (use the same allclose mask).
        bad = (diff > (atol + rtol * ref.abs()))
        idxs = torch.nonzero(bad, as_tuple=False)
        n_show = min(10, idxs.shape[0])
        msg_lines = [
            f"[CORRECTNESS] FAIL layout={layout} M={M} N={N} K={K} gm={gm} xcd={xcd}",
            f"[CORRECTNESS]   max_abs={max_abs:.4f}  max_rel={max_rel:.4e}  atol={atol:.3f}  rtol={rtol}",
            f"[CORRECTNESS]   {idxs.shape[0]} mismatched / {Cf.numel()} elements; first {n_show}:",
        ]
        for k in range(n_show):
            i, j = int(idxs[k, 0]), int(idxs[k, 1])
            msg_lines.append(f"[CORRECTNESS]     C[{i},{j}] got={Cf[i,j].item():.4f} expected={ref[i,j].item():.4f}")
        msg = "\n".join(msg_lines)
        print(msg, flush=True)
        raise RuntimeError(msg)

    line = f"[CORRECTNESS] M={M} N={N} K={K} layout={layout} PASS  (max_abs={diff.max().item():.4f}, atol={atol:.3f})"
    print(line, flush=True)
    if _log_handle is not None:
        _log_handle.write(line + "\n")
        _log_handle.flush()
    return True


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

    best_ms, best_gm, best_xcd = float("inf"), 4, 8
    for xcd in XCD_SEARCH:
        for gm in GM_SEARCH:
            run = lambda g=gm, x=xcd: fn(A, B, C, g, x)
            ms = _time(run)
            if ms < best_ms:
                best_ms, best_gm, best_xcd = ms, gm, xcd
    return 2.0 * M * N * K / (best_ms * 1e9), best_gm, best_xcd


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

# P23 Step 5: MANDATORY correctness gate — runs BEFORE any perf timing.
# Aborts the bench if any layout silently produces wrong MFMA outputs.
# Do NOT make conditional or opt-out: this is the trap-door for Step 4.
print(f"[CORRECTNESS] P23 Step 5 gate: smoke shape M={SMOKE_M} N={SMOKE_N} K={SMOKE_K}")
for _layout in LAYOUTS:
    verify_correctness(SMOKE_M, SMOKE_N, SMOKE_K, _layout)
print(f"[CORRECTNESS] all {len(LAYOUTS)} layouts PASSED gate; proceeding to perf bench")

print(f"BF16 GEMM benchmark: {len(shapes)} shapes × {len(LAYOUTS)} layouts vs torch.mm (hipBLASLt)")
print(f"{'M':>5} {'N':>6} {'K':>5}", end="")
for lay in LAYOUTS:
    print(f" | {lay.upper():>3} {'TK':>7} {'gm':>2} {'xc':>2} {'torch':>7} {'ratio':>6}", end="")
print()
print("-" * 130)

stats_by_layout = {l: [] for l in LAYOUTS}
all_rows = []
for M, n, k in shapes:
    row = {"M": M, "N": n, "K": k}
    line = f"{M:>5} {n:>6} {k:>5}"
    for layout in LAYOUTS:
        tk_tf, gm, xcd = bench_tk(M, n, k, layout)
        torch_tf = bench_torch(M, n, k, layout)
        ratio = tk_tf / torch_tf if torch_tf > 0 else 0
        stats_by_layout[layout].append(ratio)
        win = "*" if ratio >= 1.0 else " "
        row[f"{layout}_tk"] = tk_tf
        row[f"{layout}_torch"] = torch_tf
        row[f"{layout}_gm"] = gm
        row[f"{layout}_xcd"] = xcd
        row[f"{layout}_ratio"] = ratio
        line += f" | {layout.upper():>3} {tk_tf:>7.1f} {gm:>2} {xcd:>2} {torch_tf:>7.1f} {ratio:>.3f}x{win}"
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
