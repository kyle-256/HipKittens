from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pybind11
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "analysis" / "fp8_gemm" / "mi350x" / "kernel_fp8_layouts.cpp"


def _rocm_path() -> Path:
    return Path(os.environ.get("ROCM_PATH", "/opt/rocm"))


def _detect_gpu_arch() -> str:
    return "gfx950"


def _python_include_flags() -> list[str]:
    return subprocess.check_output(["python3-config", "--includes"], text=True).split()


def _python_ldflags() -> list[str]:
    flags = subprocess.check_output(["python3-config", "--ldflags"], text=True).split()
    return [flag for flag in flags if flag != "-lcrypt"]


class HipKittensBuildExt(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if ext.name != "hipkittens._tk_fp8_layouts":
            raise RuntimeError(f"Unexpected extension name: {ext.name}")

        rocm_path = _rocm_path()
        hipcc = Path(os.environ.get("HIPCXX", str(rocm_path / "bin" / "hipcc")))
        if not hipcc.is_file():
            raise RuntimeError(f"hipcc not found: {hipcc}")

        ext_path = Path(self.get_ext_fullpath(ext.name))
        ext_path.parent.mkdir(parents=True, exist_ok=True)

        arch = _detect_gpu_arch()
        cmd = [
            str(hipcc),
            str(SOURCE),
            f"--offload-arch={arch}",
            "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
            "-ffast-math",
            "-std=c++20",
            "-w",
            "-shared",
            "-fPIC",
            "-Rpass-analysis=kernel-resource-usage",
            f"-DTK_FP8_LAYOUTS_MODULE_NAME={ext.name.rsplit('.', 1)[-1]}",
            f"-I{ROOT / 'include'}",
            f"-I{ROOT / 'prototype'}",
            f"-I{rocm_path / 'include' / 'hip'}",
            f"-I{rocm_path / 'include' / 'rocrand'}",
            *_python_include_flags(),
            f"-I{pybind11.get_include()}",
            *_python_ldflags(),
            "-o",
            str(ext_path),
        ]
        if arch.startswith("gfx95"):
            cmd.insert(2, "-DKITTENS_CDNA4")

        self.spawn(cmd)


setup(
    name="hipkittens",
    version="0.1.0",
    description="HipKittens FP8 GEMM Python package",
    packages=find_packages(include=["hipkittens", "hipkittens.*"]),
    ext_modules=[Extension("hipkittens._tk_fp8_layouts", sources=[])],
    cmdclass={"build_ext": HipKittensBuildExt},
    include_package_data=True,
    install_requires=["pybind11"],
    python_requires=">=3.10",
)
