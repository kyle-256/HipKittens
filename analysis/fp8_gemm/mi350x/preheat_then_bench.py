"""R26 Dev A — preheat GPU3 to ramp DPM, then exec the benchmark in same process.
Usage: HIP_VISIBLE_DEVICES=3 MXFP8_LAYOUTS=crr python3 preheat_then_bench.py M N K
"""
import os
import sys
import time

import torch

# 8-second sustained heavy workload to ramp DPM to peak sclk.
print("[preheat] starting sustained heavy workload...", file=sys.stderr, flush=True)
a = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
b = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
t0 = time.time()
i = 0
while time.time() - t0 < 8.0:
    c = a @ b
    i += 1
torch.cuda.synchronize()
print(f"[preheat] done ({i} iters in {time.time() - t0:.1f}s)", file=sys.stderr, flush=True)
del a, b, c
torch.cuda.empty_cache()

# Now exec the actual benchmark in the SAME process so DPM stays high.
sys.argv = ["test_mxfp8_python.py"] + sys.argv[1:]
exec(open(os.path.join(os.path.dirname(__file__), "test_mxfp8_python.py")).read())
