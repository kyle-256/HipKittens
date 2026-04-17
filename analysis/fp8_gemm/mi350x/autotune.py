"""
FP8 GEMM autotuner for HipKittens.

For each (M, N, K, layout) combination, finds the best `group_m` parameter
(and, for RCR, the best kernel variant: 4-wave vs 8-wave) by running a quick
benchmark grid search. Results are cached to disk.

Usage:
    from autotune import AutotunedGEMM
    gemm = AutotunedGEMM()
    gemm.rcr(A, B, C, scale_a, scale_b)  # auto-selects best group_m (+ kernel)
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

GROUP_M_CANDIDATES = [1, 2, 4, 8, 16, 32]
# For RCR we also pick between the 8-wave and 4-wave dynamic kernels. The
# 4-wave kernel has 2×2 warp grid (vs 2×4) and 2 blocks/CU, so its scheduling
# profile is quite different; some large-N / medium-K shapes win with 4-wave.
RCR_KERNELS = ["8", "4"]  # "default" == "8" here; we enumerate both explicitly
TUNE_WARMUP = 15
TUNE_ITERS = 50
# Run multiple independent trials and take the min time per (kernel, gm) to
# reject transient GPU load noise (e.g. thermal/memory-clock oscillations).
TUNE_TRIALS = 3

_LAYOUT_FNS = {
    "rcr": tk_fp8_layouts.gemm_rcr,
    "rrr": tk_fp8_layouts.gemm_rrr,
    "crr": tk_fp8_layouts.gemm_crr,
}


def _cache_key(M, N, K, layout):
    return f"{layout}_{M}_{N}_{K}"


def _rcr_kernel_allowed(kernel: str, M: int, N: int, K: int) -> bool:
    """Don't bother trying 4-wave on shapes the kernel intentionally skips.

    Mirrors the C++ heuristic (`RCR_4WAVE_MIN_GRID`, `RCR_4WAVE_MAX_K`) so we
    don't waste autotune iterations on obviously-bad configurations.
    """
    if kernel == "8":
        return True
    assert kernel == "4"
    if M % 256 != 0 or N % 256 != 0 or K % 128 != 0:
        return False
    return True


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

    def _bench_single(self, fn, A, B, C, gm):
        # CUDA events give us kernel-accurate timing with lower host jitter
        # than time.perf_counter. We take the min across TUNE_TRIALS runs.
        for _ in range(TUNE_WARMUP):
            C.zero_()
            fn(A, B, C, 1.0, 1.0, gm)
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        best = float("inf")
        for _ in range(TUNE_TRIALS):
            torch.cuda.synchronize()
            start_ev.record()
            for _ in range(TUNE_ITERS):
                C.zero_()
                fn(A, B, C, 1.0, 1.0, gm)
            end_ev.record()
            torch.cuda.synchronize()
            avg = start_ev.elapsed_time(end_ev) / TUNE_ITERS
            if avg < best:
                best = avg
        return best

    def _tune(self, M, N, K, layout, A, B):
        """Benchmark all (group_m, kernel) candidates, return best."""
        fn = _LAYOUT_FNS[layout]
        C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        best_gm = GROUP_M_CANDIDATES[0]
        best_ms = float("inf")
        best_kernel = "8"
        results = {}

        # For RCR, sweep both 4-wave and 8-wave; for RRR/CRR, there is only
        # one kernel path.
        if layout == "rcr":
            kernels = RCR_KERNELS
        else:
            kernels = ["8"]

        for kernel in kernels:
            if layout == "rcr" and not _rcr_kernel_allowed(kernel, M, N, K):
                continue
            if kernel == "8":
                os.environ["TK_RCR_FORCE_KERNEL"] = "8"
            elif kernel == "4":
                os.environ["TK_RCR_FORCE_KERNEL"] = "4"
            for gm in GROUP_M_CANDIDATES:
                try:
                    elapsed = self._bench_single(fn, A, B, C, gm)
                except Exception as e:
                    continue
                results[(kernel, gm)] = elapsed
                if elapsed < best_ms:
                    best_ms = elapsed
                    best_gm = gm
                    best_kernel = kernel
        # Clear the override so runtime dispatch uses its built-in heuristic
        # when we're not explicitly running an autotune iteration.
        if "TK_RCR_FORCE_KERNEL" in os.environ:
            del os.environ["TK_RCR_FORCE_KERNEL"]

        if self._verbose:
            flops = 2.0 * M * N * K
            print(f"[autotune] {layout.upper()} ({M},{N},{K}): ", end="")
            keys = sorted(results.keys(), key=lambda x: (x[0], x[1]))
            for (kk, gm), ms in ((k, results[k]) for k in keys):
                tf = flops / (ms * 1e9)
                tag = f"{kk}w/gm={gm}:{tf:.0f}"
                if (kk, gm) == (best_kernel, best_gm):
                    tag += " <--"
                print(tag, end="  ")
            print()

        key = _cache_key(M, N, K, layout)
        entry = {"group_m": best_gm, "ms": round(best_ms, 4)}
        if layout == "rcr":
            entry["kernel"] = best_kernel
        self._cache[key] = entry
        self._save_cache()
        return best_gm, best_kernel

    def _get_entry(self, M, N, K, layout, A=None, B=None):
        key = _cache_key(M, N, K, layout)
        if key in self._cache:
            cached = self._cache[key]
            kernel = cached.get("kernel", "8") if layout == "rcr" else "8"
            return cached["group_m"], kernel
        if A is not None and B is not None:
            return self._tune(M, N, K, layout, A, B)
        return tk_fp8_layouts.DEFAULT_GROUP_M, "8"

    def get_group_m(self, M, N, K, layout, A=None, B=None):
        """Get tuned group_m; tune on cache miss if tensors provided."""
        gm, _ = self._get_entry(M, N, K, layout, A, B)
        return gm

    def _call(self, layout, A, B, C, scale_a, scale_b):
        M, N = C.shape[0], C.shape[1]
        if layout == "rcr":
            K = A.shape[1]
        elif layout == "rrr":
            K = A.shape[1]
        else:
            K = B.shape[0]
        gm, kernel = self._get_entry(M, N, K, layout, A, B)
        # For RCR, apply the cached kernel choice via the env-var override
        # used by the C++ dispatch code. RRR/CRR have a single kernel path.
        if layout == "rcr":
            prev = os.environ.get("TK_RCR_FORCE_KERNEL")
            os.environ["TK_RCR_FORCE_KERNEL"] = kernel
            try:
                _LAYOUT_FNS[layout](A, B, C, scale_a, scale_b, gm)
            finally:
                if prev is None:
                    os.environ.pop("TK_RCR_FORCE_KERNEL", None)
                else:
                    os.environ["TK_RCR_FORCE_KERNEL"] = prev
        else:
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
