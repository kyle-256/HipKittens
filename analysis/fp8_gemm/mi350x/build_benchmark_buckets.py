#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


RECOMMENDED_BUCKETS = [
    {"name": "rcr_m4096_wide_n", "build_shape": (4096, 57344, 8192), "layouts": ("rcr",)},
    {"name": "rcr_m4096_wide_k", "build_shape": (4096, 8192, 28672), "layouts": ("rcr",)},
    {"name": "rcr_m8192_wide_n", "build_shape": (8192, 106496, 16384), "layouts": ("rcr",)},
    {"name": "rcr_m8192_wide_k", "build_shape": (8192, 16384, 53248), "layouts": ("rcr",)},
    {"name": "rcr_m16384_wide_n", "build_shape": (16384, 106496, 16384), "layouts": ("rcr",)},
    {"name": "rcr_m16384_wide_k", "build_shape": (16384, 16384, 53248), "layouts": ("rcr",)},
    {"name": "rcr_m32768_wide_n", "build_shape": (32768, 106496, 16384), "layouts": ("rcr",)},
    {"name": "rcr_m32768_wide_k", "build_shape": (32768, 16384, 53248), "layouts": ("rcr",)},
    {"name": "rrr_m4096_wide_k", "build_shape": (4096, 8192, 57344), "layouts": ("rrr",)},
    {
        "name": "rrr_m4096_wide_n",
        "build_shape": (4096, 28672, 8192),
        "layouts": ("rrr",),
        "macros": {"RRR_MAIN_UNROLL": 2, "RRR_PREFETCH_LGKM": 8},
    },
    {"name": "rrr_m8192_wide_k", "build_shape": (8192, 16384, 106496), "layouts": ("rrr",)},
    {
        "name": "rrr_m8192_wide_n",
        "build_shape": (8192, 53248, 16384),
        "layouts": ("rrr",),
        "macros": {"RRR_MAIN_UNROLL": 2, "RRR_PREFETCH_LGKM": 8},
    },
    {"name": "rrr_m16384_wide_k", "build_shape": (16384, 16384, 106496), "layouts": ("rrr",)},
    {
        "name": "rrr_m16384_wide_n",
        "build_shape": (16384, 53248, 16384),
        "layouts": ("rrr",),
        "macros": {"RRR_MAIN_UNROLL": 2, "RRR_PREFETCH_LGKM": 8},
    },
    {"name": "rrr_m32768_wide_k", "build_shape": (32768, 16384, 106496), "layouts": ("rrr",)},
    {
        "name": "rrr_m32768_wide_n",
        "build_shape": (32768, 53248, 16384),
        "layouts": ("rrr",),
        "macros": {"RRR_MAIN_UNROLL": 2, "RRR_PREFETCH_LGKM": 8},
    },
    {"name": "crr_m3584_main", "build_shape": (3584, 18944, 32768), "layouts": ("crr",)},
    {"name": "crr_m4096_main", "build_shape": (4096, 14336, 32768), "layouts": ("crr",)},
    {"name": "crr_m4608_main", "build_shape": (4608, 3584, 32768), "layouts": ("crr",)},
    {"name": "crr_m6144_main", "build_shape": (6144, 4096, 32768), "layouts": ("crr",)},
    {"name": "crr_m8192_main", "build_shape": (8192, 29696, 32768), "layouts": ("crr",)},
    {"name": "crr_m10240_main", "build_shape": (10240, 8192, 32768), "layouts": ("crr",)},
    {"name": "crr_m12288_main", "build_shape": (12288, 4096, 16384), "layouts": ("crr",)},
    {"name": "crr_m16384_main", "build_shape": (16384, 53248, 32768), "layouts": ("crr",)},
    {"name": "crr_m18432_main", "build_shape": (18432, 16384, 32768), "layouts": ("crr",)},
    {"name": "crr_m22016_main", "build_shape": (22016, 4096, 16384), "layouts": ("crr",)},
    {"name": "crr_m28672_main", "build_shape": (28672, 4096, 32768), "layouts": ("crr",)},
    {"name": "crr_m37888_main", "build_shape": (37888, 3584, 32768), "layouts": ("crr",)},
    {"name": "crr_m57344_main", "build_shape": (57344, 8192, 16384), "layouts": ("crr",)},
    {"name": "crr_m59136_main", "build_shape": (59136, 8192, 32768), "layouts": ("crr",)},
    {"name": "crr_m106496_main", "build_shape": (106496, 16384, 32768), "layouts": ("crr",)},
]


def module_name(prefix: str, build_shape: tuple[int, int, int]) -> str:
    m_dim, n_dim, k_dim = build_shape
    return f"{prefix}_{m_dim}x{n_dim}x{k_dim}"


def extension_suffix() -> str:
    return subprocess.check_output(
        [sys.executable, "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')"],
        text=True,
    ).strip()


def parse_csv_filter(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def select_buckets(layouts: set[str] | None, names: set[str] | None) -> list[dict]:
    selected = []
    for bucket in RECOMMENDED_BUCKETS:
        bucket_layouts = {layout.lower() for layout in bucket["layouts"]}
        if layouts is not None and bucket_layouts.isdisjoint(layouts):
            continue
        if names is not None and bucket["name"].lower() not in names:
            continue
        selected.append(bucket)
    return selected


def build_bucket(bucket: dict, args, ext_suffix: str, workdir: Path, output_dir: Path) -> None:
    build_shape = tuple(bucket["build_shape"])
    mod_name = module_name(args.module_prefix, build_shape)
    target = output_dir / mod_name
    artifact = target.with_name(target.name + ext_suffix)
    if args.skip_existing and artifact.exists():
        print(f"[skip] {bucket['name']} -> {artifact.name}")
        return

    cppflags = [
        f"-DM_DIM={build_shape[0]}",
        f"-DN_DIM={build_shape[1]}",
        f"-DK_DIM={build_shape[2]}",
        f"-DTK_FP8_LAYOUTS_MODULE_NAME={mod_name}",
    ]
    for macro_name, macro_value in bucket.get("macros", {}).items():
        cppflags.append(f"-D{macro_name}={macro_value}")
    command = [
        "make",
        f"TARGET={target}",
        f"CPPFLAGS={' '.join(cppflags)}",
    ]
    print(f"[build] {bucket['name']} -> {artifact.name}")
    if args.dry_run:
        print("        " + " ".join(command))
        return

    subprocess.run(command, cwd=workdir, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build recommended HipKittens benchmark bucket modules.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "buckets",
        help="Directory for compiled bucket modules.",
    )
    parser.add_argument(
        "--module-prefix",
        default="tk_fp8_layouts",
        help="Python module name prefix used by Primus bucket dispatch.",
    )
    parser.add_argument(
        "--layouts",
        default=None,
        help="Comma-separated layout filter, e.g. rrr,crr.",
    )
    parser.add_argument(
        "--names",
        default=None,
        help="Comma-separated bucket-name filter.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip buckets whose extension module already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print make commands without running them.",
    )
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ext_suffix = extension_suffix()

    layouts = parse_csv_filter(args.layouts)
    names = parse_csv_filter(args.names)
    buckets = select_buckets(layouts, names)
    if not buckets:
        print("No buckets selected.")
        return 0

    for bucket in buckets:
        build_bucket(bucket, args, ext_suffix, workdir, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
