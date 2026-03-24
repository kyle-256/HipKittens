#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


LAYOUT_MAP = {
    ("T", "N"): "rcr",
    ("T", "T"): "rrr",
    ("N", "T"): "crr",
}


CANDIDATES = {
    "rcr": [
        {
            "name": "rcr_base",
            "macros": {
                "RCR_PREFETCH_LGKM": 8,
                "RCR_MAIN_UNROLL": 2,
            },
        },
        {
            "name": "rcr_lgkm6",
            "macros": {
                "RCR_PREFETCH_LGKM": 6,
                "RCR_MAIN_UNROLL": 2,
            },
        },
    ],
    "rrr": [
        {
            "name": "rrr_col_u4_l8",
            "macros": {
                "RRR_ROW_SHARED_TRANSPOSE": 0,
                "RRR_MAIN_UNROLL": 4,
                "RRR_PREFETCH_LGKM": 8,
            },
        },
        {
            "name": "rrr_col_u4_l5",
            "macros": {
                "RRR_ROW_SHARED_TRANSPOSE": 0,
                "RRR_MAIN_UNROLL": 4,
                "RRR_PREFETCH_LGKM": 5,
            },
        },
        {
            "name": "rrr_col_u2_l8",
            "macros": {
                "RRR_ROW_SHARED_TRANSPOSE": 0,
                "RRR_MAIN_UNROLL": 2,
                "RRR_PREFETCH_LGKM": 8,
            },
        },
        {
            "name": "rrr_col_u2_l5",
            "macros": {
                "RRR_ROW_SHARED_TRANSPOSE": 0,
                "RRR_MAIN_UNROLL": 2,
                "RRR_PREFETCH_LGKM": 5,
            },
        },
    ],
    "crr": [
        {
            "name": "crr_col_base",
            "macros": {
                "CRR_ROW_SHARED_TRANSPOSE": 0,
                "CRR_PREFETCH_LGKM": 3,
                "CRR_INIT1_VMCNT": 6,
                "CRR_STEADY_VMCNT": 6,
                "CRR_ENABLE_SCHED_BARRIER": 0,
                "CRR_BATCHED_PAIR_MMA": 1,
            },
        },
        {
            "name": "crr_col_init1_4",
            "macros": {
                "CRR_ROW_SHARED_TRANSPOSE": 0,
                "CRR_PREFETCH_LGKM": 3,
                "CRR_INIT1_VMCNT": 4,
                "CRR_STEADY_VMCNT": 6,
                "CRR_ENABLE_SCHED_BARRIER": 0,
                "CRR_BATCHED_PAIR_MMA": 1,
            },
        },
        {
            "name": "crr_col_lgkm4",
            "macros": {
                "CRR_ROW_SHARED_TRANSPOSE": 0,
                "CRR_PREFETCH_LGKM": 4,
                "CRR_INIT1_VMCNT": 6,
                "CRR_STEADY_VMCNT": 6,
                "CRR_ENABLE_SCHED_BARRIER": 0,
                "CRR_BATCHED_PAIR_MMA": 1,
            },
        },
        {
            "name": "crr_col_steady4",
            "macros": {
                "CRR_ROW_SHARED_TRANSPOSE": 0,
                "CRR_PREFETCH_LGKM": 3,
                "CRR_INIT1_VMCNT": 6,
                "CRR_STEADY_VMCNT": 4,
                "CRR_ENABLE_SCHED_BARRIER": 0,
                "CRR_BATCHED_PAIR_MMA": 1,
            },
        },
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune HipKittens FP8 layouts for all shapes in mnk.yaml."
    )
    parser.add_argument(
        "--mnk-file",
        default="/shared_nfs/kyle/triton_bench/mnk.yaml",
        help="Path to input mnk.yaml",
    )
    parser.add_argument(
        "--output",
        default="hipk_tuned_mnk.yaml",
        help="Output YAML file with the best config per shape",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup iterations per candidate benchmark",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Measured iterations per candidate benchmark",
    )
    parser.add_argument(
        "--check-warmup",
        type=int,
        default=25,
        help="Warmup iterations for best-config correctness recheck",
    )
    parser.add_argument(
        "--check-iters",
        type=int,
        default=5,
        help="Measured iterations for best-config correctness recheck",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=0,
        help="Limit the number of shapes for a debug run (0 = all)",
    )
    parser.add_argument(
        "--skip-check-best",
        action="store_true",
        help="Skip the best-config correctness recheck",
    )
    return parser.parse_args()


def align_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def layout_for_item(item):
    key = (item["rowMajorA"], item["rowMajorB"])
    if key not in LAYOUT_MAP:
        raise ValueError(f"Unsupported layout combination: {key}")
    return LAYOUT_MAP[key]


def result_key(item):
    return (
        item["M"],
        item["N"],
        item["K"],
        item["rowMajorA"],
        item["rowMajorB"],
    )


def load_existing_results(path):
    output_path = Path(path)
    if not output_path.exists():
        return []
    with output_path.open() as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a YAML list")
    return data


def save_results(path, results):
    with open(path, "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)


def macros_to_cppflags(build_m, build_n, build_k, macros):
    merged = {
        "M_DIM": build_m,
        "N_DIM": build_n,
        "K_DIM": build_k,
    }
    merged.update(macros)
    return " ".join(f"-D{name}={value}" for name, value in merged.items())


def run_make(analysis_dir, repo_root, cppflags):
    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = str(repo_root)
    env["CPPFLAGS"] = cppflags
    cmd = ["make", "-B"]
    return subprocess.run(
        cmd,
        cwd=analysis_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def run_bench(
    analysis_dir,
    gpu,
    layout,
    shape,
    build_shape,
    warmup,
    iters,
    check,
):
    m, n, k = shape
    build_m, build_n, build_k = build_shape
    output_path = analysis_dir / "tmp_fp8_mnk_result.json"
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    env["FP8_LAYOUTS"] = layout
    env["FP8_WARMUP"] = str(warmup)
    env["FP8_ITERS"] = str(iters)
    env["FP8_CHECK"] = "1" if check else "0"
    env["FP8_BUILD_M"] = str(build_m)
    env["FP8_BUILD_N"] = str(build_n)
    env["FP8_BUILD_K"] = str(build_k)
    env["FP8_OUTPUT"] = str(output_path)
    proc = subprocess.run(
        [sys.executable, "test_python.py", str(m), str(n), str(k)],
        cwd=analysis_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    key = f"{m}x{n}x{k}"
    with output_path.open() as f:
        data = json.load(f)[key]
    layout_result = data[layout]
    return {
        "ok": True,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "avg_ms": layout_result["avg_ms"],
        "tflops": layout_result["tflops"],
        "pct_of_rcr": layout_result.get("pct_of_rcr"),
        "target_met": data.get("target_met"),
        "check_pass": "Result: PASS" in proc.stdout and "Result: FAIL" not in proc.stdout,
    }


def tune_one_shape(args, analysis_dir, repo_root, item):
    m = item["M"]
    n = item["N"]
    k = item["K"]
    layout = layout_for_item(item)
    build_m = align_up(m, 256)
    build_n = align_up(n, 256)
    build_k = align_up(k, 128)
    padded = (build_m, build_n, build_k) != (m, n, k)

    best = None
    failures = []
    for candidate in CANDIDATES[layout]:
        cppflags = macros_to_cppflags(build_m, build_n, build_k, candidate["macros"])
        build_proc = run_make(analysis_dir, repo_root, cppflags)
        if build_proc.returncode != 0:
            failures.append(
                {
                    "candidate": candidate["name"],
                    "stage": "build",
                    "stderr": build_proc.stderr[-2000:],
                }
            )
            continue

        bench = run_bench(
            analysis_dir=analysis_dir,
            gpu=args.gpu,
            layout=layout,
            shape=(m, n, k),
            build_shape=(build_m, build_n, build_k),
            warmup=args.warmup,
            iters=args.iters,
            check=False,
        )
        if not bench["ok"]:
            failures.append(
                {
                    "candidate": candidate["name"],
                    "stage": "bench",
                    "stderr": bench["stderr"][-2000:],
                }
            )
            continue

        current = {
            "candidate_name": candidate["name"],
            "macros": dict(candidate["macros"]),
            "cppflags": cppflags,
            "avg_ms": bench["avg_ms"],
            "tflops": bench["tflops"],
            "pct_of_rcr": bench.get("pct_of_rcr"),
        }
        if best is None or current["tflops"] > best["tflops"]:
            best = current

    if best is None:
        return {
            "M": m,
            "N": n,
            "K": k,
            "rowMajorA": item["rowMajorA"],
            "rowMajorB": item["rowMajorB"],
            "hipkittens_layout": layout,
            "build_M": build_m,
            "build_N": build_n,
            "build_K": build_k,
            "padded": padded,
            "status": "failed",
            "failures": failures,
        }

    check_pass = None
    if not args.skip_check_best:
        best_build = run_make(analysis_dir, repo_root, best["cppflags"])
        if best_build.returncode == 0:
            check = run_bench(
                analysis_dir=analysis_dir,
                gpu=args.gpu,
                layout=layout,
                shape=(m, n, k),
                build_shape=(build_m, build_n, build_k),
                warmup=args.check_warmup,
                iters=args.check_iters,
                check=True,
            )
            check_pass = check["ok"] and check["check_pass"]
        else:
            check_pass = False

    result = {
        "M": m,
        "N": n,
        "K": k,
        "rowMajorA": item["rowMajorA"],
        "rowMajorB": item["rowMajorB"],
        "hipkittens_layout": layout,
        "build_M": build_m,
        "build_N": build_n,
        "build_K": build_k,
        "padded": padded,
        "status": "ok",
        "candidate_name": best["candidate_name"],
        "avg_ms": round(best["avg_ms"], 6),
        "TFLOPS": round(best["tflops"], 3),
        "pct_of_rcr": None if best["pct_of_rcr"] is None else round(best["pct_of_rcr"], 3),
        "cppflags": best["cppflags"],
        "correctness_pass": check_pass,
    }
    result.update(best["macros"])
    return result


def main():
    args = parse_args()
    analysis_dir = Path(__file__).resolve().parent
    repo_root = analysis_dir.parents[2]

    with open(args.mnk_file) as f:
        items = yaml.safe_load(f)

    if args.max_shapes > 0:
        items = items[: args.max_shapes]

    existing = load_existing_results(args.output)
    completed = {result_key(item) for item in existing}
    results = list(existing)

    total = len(items)
    start = time.time()
    for idx, item in enumerate(items, 1):
        key = result_key(item)
        layout = layout_for_item(item)
        if key in completed:
            print(
                f"[{idx}/{total}] skip M={item['M']} N={item['N']} K={item['K']} "
                f"{item['rowMajorA']}{item['rowMajorB']} ({layout})",
                flush=True,
            )
            continue

        print(
            f"[{idx}/{total}] tune M={item['M']} N={item['N']} K={item['K']} "
            f"{item['rowMajorA']}{item['rowMajorB']} ({layout})",
            flush=True,
        )
        result = tune_one_shape(args, analysis_dir, repo_root, dict(item))
        results.append(result)
        completed.add(key)
        save_results(args.output, results)

        status = result["status"]
        if status == "ok":
            padded_suffix = ""
            if result["padded"]:
                padded_suffix = (
                    f" padded->{result['build_M']}x{result['build_N']}x{result['build_K']}"
                )
            print(
                f"    best={result['candidate_name']} "
                f"TFLOPS={result['TFLOPS']:.3f} "
                f"time_ms={result['avg_ms']:.6f}"
                f"{padded_suffix}",
                flush=True,
            )
        else:
            print("    failed to find a valid candidate", flush=True)

    elapsed = time.time() - start
    print(
        f"Finished {len(results)} results in {elapsed / 3600:.2f} hours. "
        f"Saved to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
