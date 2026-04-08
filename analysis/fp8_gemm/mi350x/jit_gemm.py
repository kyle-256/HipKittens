"""
JIT compiler for shape-specific FP8 GEMM kernels.
Compiles 4-wave exact fastpath kernels with compile-time M/N/K dimensions.
"""

import hashlib
import importlib
import os
import subprocess
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
_TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT",
                           os.path.abspath(os.path.join(_DIR, "..", "..", "..")))
_ROCM = os.environ.get("ROCM_PATH", "/opt/rocm")
_CACHE_DIR = os.path.join(_DIR, ".jit_cache")
_EXT_SUFFIX = importlib.machinery.EXTENSION_SUFFIXES[0]

_loaded_modules = {}


def _shape_key(M: int, N: int, K: int) -> str:
    return f"{M}x{N}x{K}"


_MODULE_NAME = "tk_fp8_layouts"


def _cache_subdir(M: int, N: int, K: int) -> str:
    return os.path.join(_CACHE_DIR, f"{M}x{N}x{K}")


def _can_use_4wave(M: int, N: int, K: int) -> bool:
    return M % 256 == 0 and N % 256 == 0 and K % 128 == 0 and M > 0 and N > 0 and K >= 256


_JIT_SRC = "kernel_jit_all.cpp"
_LAYOUT_IDS = {"rcr": 1, "rrr": 2, "crr": 3}
_HIPCXX = "/opt/rocm/bin/hipcc"
_PY_INCLUDES = None
_PY_LDFLAGS = None

VARIANT_4WAVE = "4wave"
VARIANT_8WAVE = "8wave"
VARIANT_BOTH = "both"

_VARIANT_FLAGS = {
    VARIANT_4WAVE: "-DRCR_USE_EXACT_4WAVE_FASTPATH=1 -DRCR_USE_EXACT_8WAVE_FASTPATH=0",
    VARIANT_8WAVE: "-DRCR_USE_EXACT_4WAVE_FASTPATH=0 -DRCR_USE_EXACT_8WAVE_FASTPATH=1",
    VARIANT_BOTH:  "-DRCR_USE_EXACT_4WAVE_FASTPATH=1 -DRCR_USE_EXACT_8WAVE_FASTPATH=1",
}


def _get_py_flags():
    global _PY_INCLUDES, _PY_LDFLAGS
    if _PY_INCLUDES is None:
        import subprocess as sp
        _PY_INCLUDES = sp.check_output(["python3", "-m", "pybind11", "--includes"],
                                        text=True).strip()
        _PY_LDFLAGS = sp.check_output(["python3-config", "--ldflags"],
                                       text=True).strip().replace("-lcrypt", "")
    return _PY_INCLUDES, _PY_LDFLAGS


def _cache_subdir_v(M: int, N: int, K: int, variant: str, layout: str = "rcr") -> str:
    return os.path.join(_CACHE_DIR, f"{layout}_{M}x{N}x{K}_{variant}")


def compile_for_shape(M: int, N: int, K: int, variant: str = VARIANT_BOTH,
                      layout: str = "rcr", verbose: bool = False) -> str:
    """Compile a shape-specific .so and return its directory."""
    subdir = _cache_subdir_v(M, N, K, variant, layout)
    so_path = os.path.join(subdir, _MODULE_NAME + _EXT_SUFFIX)

    if os.path.exists(so_path):
        return subdir

    os.makedirs(subdir, exist_ok=True)
    if verbose:
        print(f"[JIT] Compiling {layout.upper()} {_shape_key(M, N, K)} ({variant})...",
              end=" ", flush=True)
    t0 = time.time()

    py_inc, py_ld = _get_py_flags()
    vflags = _VARIANT_FLAGS.get(variant, _VARIANT_FLAGS[VARIANT_BOTH])
    layout_id = _LAYOUT_IDS.get(layout, 0)
    extra = []
    if layout == "rrr":
        extra = ["-DRRR_MAIN_UNROLL=2"]
    cmd = [
        _HIPCXX, os.path.join(_DIR, _JIT_SRC),
        "-DKITTENS_CDNA4", "--offload-arch=gfx950",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-ffast-math",
        "-I/opt/rocm/include/rocrand",
        f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DK_DIM={K}",
        f"-DJIT_LAYOUT={layout_id}",
        "-DRCR_STEADY_VMCNT=8", *vflags.split(), *extra,
        "-std=c++20", "-w", "-shared", "-fPIC",
        f"-I{_TK_ROOT}/include", f"-I{_TK_ROOT}/prototype",
        "-I/opt/rocm/include/hip",
        *py_inc.split(), *py_ld.split(),
        "-o", so_path,
    ]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"JIT compilation failed for {layout} {_shape_key(M, N, K)}:\n{r.stderr[-500:]}")

    if verbose:
        print(f"{time.time() - t0:.1f}s")
    return subdir


def get_module(M: int, N: int, K: int, variant: str = VARIANT_BOTH,
               layout: str = "rcr", verbose: bool = False):
    """Get the JIT-compiled module for a given shape and layout."""
    key = f"{layout}_{_shape_key(M, N, K)}_{variant}"
    if key in _loaded_modules:
        return _loaded_modules[key]

    if not _can_use_4wave(M, N, K):
        return None

    subdir = compile_for_shape(M, N, K, variant=variant, layout=layout, verbose=verbose)
    so_path = os.path.join(subdir, _MODULE_NAME + _EXT_SUFFIX)

    saved = sys.modules.pop(_MODULE_NAME, None)
    try:
        spec = importlib.util.spec_from_file_location(
            _MODULE_NAME, so_path,
            submodule_search_locations=[subdir])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules[_MODULE_NAME] = saved
        else:
            sys.modules.pop(_MODULE_NAME, None)

    _loaded_modules[key] = mod
    return mod


class JITGemm:
    """Drop-in replacement for tk_fp8_layouts with JIT compilation.
    
    NOTE: Due to dynamic linker symbol conflicts, JIT-compiled modules cannot
    coexist with the default tk_fp8_layouts in the same process. Use one of:
    1. Only JIT: don't import tk_fp8_layouts before creating JITGemm
    2. Subprocess: use bench_jit.py which runs each shape in a subprocess
    """

    def __init__(self, verbose: bool = True, fallback_module=None):
        self.verbose = verbose
        self._fallback = fallback_module

    def gemm_rcr(self, A, B, C, scale_a, scale_b, group_m=4):
        M, K_a = A.shape
        N, K_b = B.shape
        K = K_a
        mod = get_module(M, N, K, verbose=self.verbose)
        if mod is not None:
            mod.gemm_rcr(A, B, C, scale_a, scale_b, group_m)
        elif self._fallback is not None:
            self._fallback.gemm_rcr(A, B, C, scale_a, scale_b, group_m)
        else:
            raise RuntimeError(f"No JIT module for ({M},{N},{K}) and no fallback")

    def gemm_rrr(self, A, B, C, scale_a, scale_b, group_m=4):
        if self._fallback is not None:
            self._fallback.gemm_rrr(A, B, C, scale_a, scale_b, group_m)
        else:
            raise RuntimeError("RRR not supported without fallback")

    def gemm_crr(self, A, B, C, scale_a, scale_b, group_m=4):
        if self._fallback is not None:
            self._fallback.gemm_crr(A, B, C, scale_a, scale_b, group_m)
        else:
            raise RuntimeError("CRR not supported without fallback")


def warmup_shapes(shapes, variants=None, layouts=None, verbose=True, max_workers=8):
    """Pre-compile all shapes × variants × layouts in parallel."""
    import concurrent.futures
    if variants is None:
        variants = [VARIANT_4WAVE, VARIANT_8WAVE]
    if layouts is None:
        layouts = ["rcr"]

    tasks = []
    for M, N, K in shapes:
        if not _can_use_4wave(M, N, K):
            continue
        for lay in layouts:
            vlist = variants if lay == "rcr" else [VARIANT_8WAVE]
            for v in vlist:
                if not os.path.exists(os.path.join(
                        _cache_subdir_v(M, N, K, v, lay), _MODULE_NAME + _EXT_SUFFIX)):
                    tasks.append((M, N, K, v, lay))

    if not tasks:
        if verbose:
            print("[JIT] All shapes already compiled.")
        return

    _get_py_flags()

    n_workers = min(max_workers, len(tasks))
    if verbose:
        print(f"[JIT] Compiling {len(tasks)} kernels (workers={n_workers})...")
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(compile_for_shape, M, N, K, v, lay, False): (M, N, K, v, lay)
                for M, N, K, v, lay in tasks}
        for fut in concurrent.futures.as_completed(futs):
            M, N, K, v, lay = futs[fut]
            try:
                fut.result()
                if verbose:
                    print(f"  [JIT] {lay.upper()} {_shape_key(M,N,K)} ({v}) done")
            except Exception as e:
                print(f"  [JIT] {lay.upper()} {_shape_key(M,N,K)} ({v}) FAILED: {e}")

    if verbose:
        print(f"[JIT] All compiled in {time.time()-t0:.1f}s")
