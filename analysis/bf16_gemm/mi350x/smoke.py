"""Quick smoke test on a few shapes - 3 layouts, with group_m autotune."""
import torch, math, sys, os, time
torch.manual_seed(42)
sys.path.insert(0, os.path.dirname(__file__))
import tk_bf16_layouts as m

WARMUP, ITERS = 10, 20
GM_SEARCH = [1, 2, 4, 8, 16]
SHAPES = [
    # Square
    (4096,  4096,  4096),
    (8192,  8192,  8192),
    # Tall-N (mlp_gate_up)
    (4096, 22016,  4096),  # Llama-2-7B
    (4096, 28672,  4096),
    (4096, 28672,  8192),
    (8192, 28672,  4096),
    (8192, 28672,  8192),
    # Wide-K (mlp_down)
    (4096,  4096, 11008),
    (8192,  4096, 14336),
    (4096,  4096, 14336),
    # Large-K
    (8192, 16384, 16384),
    # Extreme tall-N
    (8192, 57344,  8192),
    (8192, 53248,  8192),  # Llama-3.1-405B
]
LAYOUTS = ["rcr","rrr","crr"]

def _time(run):
    for _ in range(WARMUP): run()
    se=torch.cuda.Event(enable_timing=True); ee=torch.cuda.Event(enable_timing=True)
    ts=[]
    for _ in range(ITERS):
        torch.cuda.synchronize(); se.record(); run(); ee.record()
        torch.cuda.synchronize(); ts.append(se.elapsed_time(ee))
    return sum(ts)/len(ts)

def bench_tk(M,N,K,lay):
    if lay=="rcr":
        A = torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
        B = torch.randn(N,K,dtype=torch.bfloat16,device="cuda")
    elif lay=="rrr":
        A = torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
        B = torch.randn(K,N,dtype=torch.bfloat16,device="cuda")
    else:
        A = torch.randn(K,M,dtype=torch.bfloat16,device="cuda")
        B = torch.randn(K,N,dtype=torch.bfloat16,device="cuda")
    C = torch.zeros(M,N,dtype=torch.bfloat16,device="cuda")
    fn = {"rcr":m.gemm_rcr, "rrr":m.gemm_rrr, "crr":m.gemm_crr}[lay]
    best_ms, best_gm = float("inf"), 4
    for gm in GM_SEARCH:
        run = lambda g=gm: fn(A,B,C,g)
        ms = _time(run)
        if ms < best_ms:
            best_ms, best_gm = ms, gm
    return 2.0*M*N*K/(best_ms*1e9), best_gm

def bench_torch(M,N,K,lay):
    if lay=="rcr":
        A = torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
        B = torch.randn(N,K,dtype=torch.bfloat16,device="cuda")
        run = lambda: torch.mm(A,B.T)
    elif lay=="rrr":
        A = torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
        B = torch.randn(K,N,dtype=torch.bfloat16,device="cuda")
        run = lambda: torch.mm(A,B)
    else:
        A = torch.randn(K,M,dtype=torch.bfloat16,device="cuda")
        B = torch.randn(K,N,dtype=torch.bfloat16,device="cuda")
        run = lambda: torch.mm(A.T,B)
    ms = _time(run)
    return 2.0*M*N*K/(ms*1e9)

print(f"{'M':>6} {'N':>6} {'K':>6} {'lay':>3}  {'tk':>7} gm  {'tor':>7}  ratio")
for M,N,K in SHAPES:
    for lay in LAYOUTS:
        tk_tf, gm = bench_tk(M,N,K,lay)
        tor_tf = bench_torch(M,N,K,lay)
        r = tk_tf/tor_tf
        marker = '*' if r >= 1.0 else ''
        print(f"{M:>6} {N:>6} {K:>6} {lay:>3}  {tk_tf:>7.1f} {gm:>2}  {tor_tf:>7.1f}  {r:.3f}x{marker}")
