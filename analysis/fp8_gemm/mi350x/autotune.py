"""
FP8 GEMM autotuner for HipKittens.

For each (M, N, K, layout) combination, finds the best `group_m` parameter
by running a quick benchmark grid search. Results are cached to disk.

Usage:
    from autotune import AutotunedGEMM
    gemm = AutotunedGEMM()
    gemm.rcr(A, B, C, scale_a, scale_b)  # auto-selects best group_m
"""

import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import tk_fp8_layouts

CACHE_FILE = os.path.join(os.path.dirname(__file__), ".autotune_cache.json")

GROUP_M_CANDIDATES = [1, 2, 4, 8, 16]
TUNE_WARMUP = 5
TUNE_ITERS = 12

_LAYOUT_FNS = {
    "rcr": tk_fp8_layouts.gemm_rcr,
    "rrr": tk_fp8_layouts.gemm_rrr,
    "crr": tk_fp8_layouts.gemm_crr,
}


def _cache_key(M, N, K, layout):
    return f"{layout}_{M}_{N}_{K}"


class AutotunedGEMM:
    def __init__(self, cache_file=CACHE_FILE, verbose=False):
        self._cache_file = cache_file
        self._cache = {}
        self._verbose = verbose
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save_cache(self):
        try:
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except IOError:
            pass

    def _tune(self, M, N, K, layout, A, B):
        """Benchmark all group_m candidates, return best."""
        fn = _LAYOUT_FNS[layout]
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        best_gm = GROUP_M_CANDIDATES[0]
        best_ms = float("inf")
        results = {}

        for gm in GROUP_M_CANDIDATES:
            for _ in range(TUNE_WARMUP):
                C.zero_()
                fn(A, B, C, 1.0, 1.0, gm)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(TUNE_ITERS):
                C.zero_()
                fn(A, B, C, 1.0, 1.0, gm)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / TUNE_ITERS * 1000.0

            results[gm] = elapsed
            if elapsed < best_ms:
                best_ms = elapsed
                best_gm = gm

        if self._verbose:
            flops = 2.0 * M * N * K
            print(f"[autotune] {layout.upper()} ({M},{N},{K}): ", end="")
            for gm, ms in sorted(results.items()):
                tf = flops / (ms * 1e9)
                marker = " <--" if gm == best_gm else ""
                print(f"gm={gm}:{tf:.0f}", end=marker + "  ")
            print()

        key = _cache_key(M, N, K, layout)
        self._cache[key] = {"group_m": best_gm, "ms": round(best_ms, 4)}
        self._save_cache()
        return best_gm

    def get_group_m(self, M, N, K, layout, A=None, B=None):
        """Get tuned group_m; tune on cache miss if tensors provided."""
        key = _cache_key(M, N, K, layout)
        if key in self._cache:
            return self._cache[key]["group_m"]
        if A is not None and B is not None:
            return self._tune(M, N, K, layout, A, B)
        return tk_fp8_layouts.DEFAULT_GROUP_M

    def _call(self, layout, A, B, C, scale_a, scale_b):
        M, N = C.shape[0], C.shape[1]
        if layout == "rcr":
            K = A.shape[1]
        elif layout == "rrr":
            K = A.shape[1]
        else:
            K = B.shape[0]
        gm = self.get_group_m(M, N, K, layout, A, B)
        _LAYOUT_FNS[layout](A, B, C, scale_a, scale_b, gm)

    def rcr(self, A, B, C, scale_a=1.0, scale_b=1.0):
        self._call("rcr", A, B, C, scale_a, scale_b)

    def rrr(self, A, B, C, scale_a=1.0, scale_b=1.0):
        self._call("rrr", A, B, C, scale_a, scale_b)

    def crr(self, A, B, C, scale_a=1.0, scale_b=1.0):
        self._call("crr", A, B, C, scale_a, scale_b)

    def tune_shapes(self, shapes, layouts=("rcr", "rrr", "crr")):
        """Pre-tune a list of (M, N, K) shapes."""
        for M, N, K in shapes:
            for layout in layouts:
                key = _cache_key(M, N, K, layout)
                if key in self._cache:
                    continue
                if layout == "rcr":
                    A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
                    B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
                elif layout == "rrr":
                    A = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
                    B = (torch.randn(K, N, device="cuda") * 0.1).to(torch.float8_e4m3fn)
                else:
                    A = (torch.randn(K, M, device="cuda") * 0.1).to(torch.float8_e4m3fn)
                    B = (torch.randn(K, N, device="cuda") * 0.1).to(torch.float8_e4m3fn)
                self._tune(M, N, K, layout, A, B)
                del A, B
                torch.cuda.empty_cache()


def tune_for_models(model_configs, batch_sizes=(1, 2), layouts=("rcr", "rrr", "crr"),
                    verbose=True):
    """Tune all shapes from model configs."""
    tuner = AutotunedGEMM(verbose=verbose)
    shapes = set()
    for name, cfg in model_configs.items():
        seq = cfg["seqlen"]
        hs = cfg["hidden_size"]
        inter = cfg["intermediate_size"]
        nah = cfg["num_attention_heads"]
        nkv = cfg["num_key_value_heads"]
        hd = cfg["head_dim"]
        ops = [
            (seq, int((nah + 2 * nkv) * hd), hs),
            (seq, hs, hs),
            (seq, int(2 * inter), hs),
            (seq, hs, inter),
        ]
        for mbs in batch_sizes:
            for s, n, k in ops:
                shapes.add((s * mbs, n, k))

    shapes = sorted(shapes)
    print(f"Tuning {len(shapes)} unique shapes × {len(layouts)} layouts...")
    tuner.tune_shapes(shapes, layouts)
    print(f"Done. Cache: {tuner._cache_file}")
    return tuner
