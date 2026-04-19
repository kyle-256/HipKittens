#!/usr/bin/env python3
"""R50 Opt D — build the aiter f4gemm .co dlopen shim into a Python-loadable .so.

Self-contained: only depends on hip_runtime + pybind11 + torch C++ headers.
Does NOT depend on aiter at link time.
"""
import json
import os
import subprocess
import sys
import sysconfig
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "R50D_aiter_dlopen.cpp")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build_R50D")
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
OUT = os.path.join(BUILD_DIR, f"R50D_aiter_shim{EXT_SUFFIX}")

os.makedirs(BUILD_DIR, exist_ok=True)


def python_include():
    return sysconfig.get_path("include")


def torch_includes():
    import torch
    base = os.path.dirname(torch.__file__)
    return [
        os.path.join(base, "include"),
        os.path.join(base, "include", "torch", "csrc", "api", "include"),
    ]


def torch_libdir():
    import torch
    return os.path.join(os.path.dirname(torch.__file__), "lib")


def pybind11_include():
    try:
        import pybind11
        return pybind11.get_include()
    except Exception:
        return ""


def main():
    t0 = time.time()
    cxx = os.environ.get("HIPCC", "/opt/rocm/bin/hipcc")
    if not os.path.exists(cxx):
        cxx = "hipcc"

    incs = [python_include(), pybind11_include()] + torch_includes()
    includes = " ".join(f"-I{p}" for p in incs if p)
    libdir = torch_libdir()
    # Match torch ABI on _GLIBCXX_USE_CXX11_ABI
    import torch
    abi = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"

    cmd = (
        f"{cxx} -O3 -fPIC -shared -std=c++17 "
        f"-D_GLIBCXX_USE_CXX11_ABI={abi} "
        f"-DTORCH_API_INCLUDE_EXTENSION_H "
        f"-DPYBIND11_COMPILER_TYPE='\"_gcc\"' "
        f"-DPYBIND11_STDLIB='\"_libstdcpp\"' "
        f"-DPYBIND11_BUILD_ABI='\"_cxxabi1011\"' "
        f"-D__HIP_PLATFORM_AMD__ "
        f"--offload-arch=gfx950 "
        f"{includes} "
        f"-L{libdir} -ltorch -ltorch_cpu -ltorch_python -lc10 "
        f"-Wl,-rpath,{libdir} "
        f"-o {OUT} {SRC}"
    )
    print("CMD:", cmd, flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0
    print("STDOUT:\n", p.stdout, flush=True)
    print("STDERR:\n", p.stderr, flush=True)
    print(f"rc={p.returncode}  elapsed={elapsed:.1f}s  out={OUT}", flush=True)

    manifest = {
        "round": "R50_optD",
        "src": SRC,
        "out": OUT,
        "rc": p.returncode,
        "elapsed_s": round(elapsed, 1),
        "exists": os.path.exists(OUT),
    }
    with open(os.path.join(SCRIPT_DIR, "R50D_BUILD_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
