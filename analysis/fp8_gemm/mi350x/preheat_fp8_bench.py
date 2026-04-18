"""R26 Dev A — preheat then run test_python.py (FP8) in same process."""
import os, sys, time
import torch

print("[preheat] starting...", file=sys.stderr, flush=True)
a = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
b = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
t0 = time.time()
while time.time() - t0 < 6.0:
    c = a @ b
torch.cuda.synchronize()
del a, b, c
torch.cuda.empty_cache()
print("[preheat] done", file=sys.stderr, flush=True)

sys.argv = ["test_python.py"] + sys.argv[1:]
exec(open(os.path.join(os.path.dirname(__file__), "test_python.py")).read())
