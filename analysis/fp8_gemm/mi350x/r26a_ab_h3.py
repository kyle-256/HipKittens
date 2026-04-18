"""R26 Dev A — A/B test V2-CRR (default) vs V1-CRR (H3 runtime gate=0).
Same process, preheat once, alternate runs to neutralize drift.

Usage: HIP_VISIBLE_DEVICES=3 python3 r26a_ab_h3.py M N K
"""
import os
import sys
import time
import statistics
import json
import subprocess

import torch

torch.manual_seed(0)

# Preheat: sustained heavy workload to ramp DPM (kept warm by reusing tensors).
print("[preheat] starting sustained heavy load...", file=sys.stderr, flush=True)
pa = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
pb = torch.randn(16384, 16384, device="cuda", dtype=torch.float16)
t0 = time.time()
while time.time() - t0 < 6.0:
    pc = pa @ pb
torch.cuda.synchronize()
print("[preheat] done", file=sys.stderr, flush=True)

# Inline run of the test_mxfp8_python.py logic, configurable via env.
M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
K = int(sys.argv[3]) if len(sys.argv) > 3 else 8192

# We want to repeatedly invoke the same "benchmark V2-CRR" code path 5x
# and "benchmark V1-CRR" 5x in alternating order, while keeping pa/pb resident
# so DPM doesn't drop between runs.
import tk_mxfp8_layouts as tk
print("API:", [x for x in dir(tk) if not x.startswith('_')], file=sys.stderr)
sys.exit(0)
