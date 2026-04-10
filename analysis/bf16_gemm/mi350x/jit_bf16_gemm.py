"""
JIT compiler for shape-specific BF16 GEMM kernels.
Compiles exact-dimension 8-wave RCR kernels with compile-time M/N/K.
Uses subprocess execution to avoid .so symbol conflicts.
"""

import importlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_DIR = os.path.dirname(os.path.abspath(__file__))
_TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT",
                           os.path.abspath(os.path.join(_DIR, "..", "..", "..")))
_CACHE_DIR = os.path.join(_DIR, ".jit_bf16_cache")
_EXT_SUFFIX = importlib.machinery.EXTENSION_SUFFIXES[0]
_MODULE_NAME = "tk_bf16_layouts"
_HIPCXX = "/opt/rocm/bin/hipcc"
_PY_INCLUDES = None
_PY_LDFLAGS = None

BLOCK_SIZE = 256
K_STEP = 64


def _get_py_flags():
    global _PY_INCLUDES, _PY_LDFLAGS
    if _PY_INCLUDES is None:
        _PY_INCLUDES = subprocess.check_output(
            ["python3", "-m", "pybind11", "--includes"], text=True).strip()
        _PY_LDFLAGS = subprocess.check_output(
            ["python3-config", "--ldflags"], text=True).strip().replace("-lcrypt", "")
    return _PY_INCLUDES, _PY_LDFLAGS


def _can_jit(M: int, N: int, K: int) -> bool:
    return (M % BLOCK_SIZE == 0 and N % BLOCK_SIZE == 0 and
            K % K_STEP == 0 and K >= 2 * K_STEP)


def _exact_cache_dir(M: int, N: int, K: int) -> str:
    return os.path.join(_CACHE_DIR, f"exact_{M}x{N}x{K}")


def compile_exact(M: int, N: int, K: int, verbose: bool = False) -> str:
    """Compile an exact-dimension .so and return its directory."""
    subdir = _exact_cache_dir(M, N, K)
    so_path = os.path.join(subdir, _MODULE_NAME + _EXT_SUFFIX)
    if os.path.exists(so_path):
        return subdir

    os.makedirs(subdir, exist_ok=True)
    if verbose:
        print(f"[JIT-BF16] Compiling {M}x{N}x{K}...", end=" ", flush=True)
    t0 = time.time()

    py_inc, py_ld = _get_py_flags()
    src = os.path.join(_CACHE_DIR, "kernel_exact.cpp")

    cmd = [
        _HIPCXX, src,
        "-DKITTENS_CDNA4", "--offload-arch=gfx950",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
        "-I/opt/rocm/include/rocrand",
        f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DK_DIM={K}",
        "-std=c++20", "-w", "-shared", "-fPIC",
        f"-I{_TK_ROOT}/include", f"-I{_TK_ROOT}/prototype",
        "-I/opt/rocm/include/hip",
        *py_inc.split(), *py_ld.split(),
        "-o", so_path,
    ]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"JIT failed for {M}x{N}x{K}:\n{r.stderr[-300:]}")

    if verbose:
        print(f"{time.time() - t0:.1f}s")
    return subdir


def warmup_shapes(shapes, verbose=True, max_workers=4):
    """Pre-compile all shapes in parallel."""
    _get_py_flags()
    tasks = [(M, N, K) for M, N, K in shapes
             if _can_jit(M, N, K)
             and not os.path.exists(os.path.join(
                 _exact_cache_dir(M, N, K), _MODULE_NAME + _EXT_SUFFIX))]

    if not tasks:
        if verbose:
            print("[JIT-BF16] All shapes already compiled.")
        return

    n = min(max_workers, len(tasks))
    if verbose:
        print(f"[JIT-BF16] Compiling {len(tasks)} exact kernels (workers={n})...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(compile_exact, M, N, K, False): (M, N, K)
                for M, N, K in tasks}
        for fut in futs:
            M, N, K = futs[fut]
            try:
                fut.result()
                if verbose:
                    print(f"  {M}x{N}x{K} done")
            except Exception as e:
                print(f"  {M}x{N}x{K} FAILED: {e}")

    if verbose:
        print(f"[JIT-BF16] All compiled in {time.time()-t0:.1f}s")
