"""Quick SNR + determinism check."""
import torch, math, sys, os
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
sys.path.insert(0, os.path.dirname(__file__))
import tk_bf16_layouts as m

SHAPES = [
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (4096, 28672, 4096),
    (8192, 28672, 4096),
    (8192, 16384, 16384),
]
LAYOUTS = ["rcr", "rrr", "crr"]

print(f"{'M':>5} {'N':>5} {'K':>5} {'L':>3} {'SNR':>7} {'Det':>3} {'gm':>2}")
all_pass = True
for M,N,K in SHAPES:
    for lay in LAYOUTS:
        for gm in [1, 4, 8]:
            torch.manual_seed(123)
            if lay == "rcr":
                A = torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
                B = torch.randn(N,K,dtype=torch.bfloat16,device="cuda")
                ref = torch.mm(A, B.T)
                fn = lambda C: m.gemm_rcr(A,B,C,gm)
            elif lay == "rrr":
                A = torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
                B = torch.randn(K,N,dtype=torch.bfloat16,device="cuda")
                ref = torch.mm(A, B)
                fn = lambda C: m.gemm_rrr(A,B,C,gm)
            else:
                A = torch.randn(K,M,dtype=torch.bfloat16,device="cuda")
                B = torch.randn(K,N,dtype=torch.bfloat16,device="cuda")
                ref = torch.mm(A.T.contiguous(), B)
                fn = lambda C: m.gemm_crr(A,B,C,gm)
            C = torch.zeros(M,N,dtype=torch.bfloat16,device="cuda")
            # Warm up: first call can be off because some internal state
            # (e.g. per-shape kernel cache, hipFuncSetAttribute) is set up.
            for _ in range(2):
                C.zero_()
                fn(C)
            torch.cuda.synchronize()
            ref_f = ref.float()
            c_f = C.float()
            sp = (ref_f**2).sum().item()
            np = ((c_f - ref_f)**2).sum().item()
            snr = 10*math.log10(sp/(np+1e-30))

            # determinism across 4 back-to-back runs (post-warmup)
            runs = []
            for _ in range(4):
                C.zero_()
                fn(C)
                torch.cuda.synchronize()
                runs.append(C.clone())
            ok_det = all(torch.equal(runs[0], r) for r in runs[1:])

            # BF16 precision upper bound is ~48 dB when comparing against
            # torch.mm(bf16) reference. Use 47 dB threshold.
            snr_ok = snr >= 47.0
            passed = snr_ok and ok_det
            all_pass = all_pass and passed
            flag = "" if passed else " !"
            print(f"{M:>5} {N:>5} {K:>5} {lay:>3} {snr:>7.2f} {'OK' if ok_det else 'FAIL':>3} {gm:>2}{flag}")

print("ALL PASS" if all_pass else "FAIL")
