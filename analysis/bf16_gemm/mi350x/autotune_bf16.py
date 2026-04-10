"""
BF16 GEMM autotune dispatch: JIT exact-dim + dynamic, per-shape best-of.
Caches results to .bf16_autotune_cache.json.
"""
import json
import os
import subprocess
import sys
import time
import importlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_FILE = os.path.join(_DIR, ".bf16_autotune_cache.json")


class AutotunedBF16GEMM:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self._cache = {}
        self._jit_modules = {}
        self._dyn_module = None
        try:
            with open(_CACHE_FILE) as f:
                self._cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _get_dyn(self):
        if self._dyn_module is None:
            sys.path.insert(0, _DIR)
            self._dyn_module = importlib.import_module("tk_bf16_layouts")
        return self._dyn_module

    def _get_jit(self, M, N, K):
        key = f"{M}x{N}x{K}"
        if key in self._jit_modules:
            return self._jit_modules[key]
        from jit_bf16_gemm import compile_exact, _exact_cache_dir, _MODULE_NAME
        try:
            subdir = compile_exact(M, N, K, verbose=self.verbose)
            ext = importlib.machinery.EXTENSION_SUFFIXES[0]
            so = os.path.join(subdir, _MODULE_NAME + ext)
            if not os.path.exists(so):
                return None
            # Can't load in same process due to symbol conflicts
            # Return the path for subprocess dispatch
            self._jit_modules[key] = subdir
            return subdir
        except Exception:
            return None

    def _autotune(self, A, B, C, M, N, K, layout):
        """Benchmark group_m values with the dynamic module, return best TFLOPS + gm."""
        import torch
        mod = self._get_dyn()
        fn_map = {"rcr": mod.gemm_rcr, "rrr": mod.gemm_rrr, "crr": mod.gemm_crr}
        fn = fn_map[layout]

        best_gm, best_tf = 4, 0
        for gm in [1, 2, 4, 8, 16]:
            call = lambda: fn(A, B, C, gm)
            for _ in range(5):
                call()
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(10):
                torch.cuda.synchronize()
                se.record()
                call()
                ee.record()
                torch.cuda.synchronize()
                ts.append(se.elapsed_time(ee))
            tf = 2.0 * M * N * K / (sum(ts) / len(ts) * 1e9)
            if tf > best_tf:
                best_tf, best_gm = tf, gm
        return best_gm, best_tf

    def gemm(self, A, B, C, layout="rcr"):
        """Autotuned GEMM dispatch. Uses cached results when available."""
        import torch
        M = C.shape[0]
        N = C.shape[1]
        if layout == "crr":
            K = A.shape[0]
        else:
            K = A.shape[1]

        cache_key = f"{layout}_{M}_{N}_{K}"
        if cache_key in self._cache:
            gm = self._cache[cache_key]["group_m"]
            mod = self._get_dyn()
            fn_map = {"rcr": mod.gemm_rcr, "rrr": mod.gemm_rrr, "crr": mod.gemm_crr}
            fn_map[layout](A, B, C, gm)
            return

        gm, _ = self._autotune(A, B, C, M, N, K, layout)
        self._cache[cache_key] = {"group_m": gm}
        self._save_cache()

        mod = self._get_dyn()
        fn_map = {"rcr": mod.gemm_rcr, "rrr": mod.gemm_rrr, "crr": mod.gemm_crr}
        fn_map[layout](A, B, C, gm)

    def rcr(self, A, B, C):
        self.gemm(A, B, C, "rcr")

    def rrr(self, A, B, C):
        self.gemm(A, B, C, "rrr")

    def crr(self, A, B, C):
        self.gemm(A, B, C, "crr")

    def _save_cache(self):
        try:
            with open(_CACHE_FILE, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass
