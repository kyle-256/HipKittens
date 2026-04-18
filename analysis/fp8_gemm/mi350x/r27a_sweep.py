"""R27 Dev A H1 sweep: cachepolicy 0/1/2/3 for V2 scale buffer_load on a single shape.

Single process: builds 4 variants, then for each runs a 5x interleaved A/B
benchmark using preheat-in-process. Outputs median, std, Welch t vs cp=0.

Usage:
    HIP_VISIBLE_DEVICES=0 python3 r27a_sweep.py <layout> <M> <N> <K>
        layout = rcr | rrr | crr
"""
import json
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def build(layout, M, N, K, cachepolicy):
    layout_upper = layout.upper()
    macro = f"-DMXFP8_{layout_upper}_V2_SCALE_CACHEPOLICY={cachepolicy}"
    log = os.path.join(HERE, f"build_cp{cachepolicy}_{layout}_{M}x{N}x{K}.log")
    cmd = [
        "make", "-j8",
        "TARGET=tk_mxfp8_layouts", "SRC=kernel_mxfp8_layouts.cpp",
        f"CXXFLAGS=-w -DM_DIM={M} -DN_DIM={N} -DK_DIM={K} {macro}",
    ]
    env = os.environ.copy()
    env["THUNDERKITTENS_ROOT"] = ROOT
    # Force a clean build for reliability.
    subprocess.run(["make", "clean"], cwd=HERE, env=env,
                   capture_output=True)
    print(f"[build] cp={cachepolicy} layout={layout} {M}x{N}x{K} ...", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    with open(log, "w") as f:
        f.write(p.stdout); f.write("\n=STDERR=\n"); f.write(p.stderr)
    if p.returncode != 0:
        print(f"  BUILD FAILED rc={p.returncode}, see {log}")
        return False
    print(f"  ok ({dt:.1f}s)", flush=True)
    return True


def bench(layout, M, N, K, iters=100, warmup=50):
    """Run preheat_then_bench.py once. Returns (tflops, snr_db) or (None,None) on fail."""
    env = os.environ.copy()
    env.setdefault("HIP_VISIBLE_DEVICES", "0")
    env["MXFP8_LAYOUTS"] = layout
    env["MXFP8_PRESHUFFLE_QUANT"] = "1"
    env["MXFP8_WARMUP"] = str(warmup)
    env["MXFP8_ITERS"] = str(iters)
    p = subprocess.run(
        ["python3", "preheat_then_bench.py", str(M), str(N), str(K)],
        cwd=HERE, env=env, capture_output=True, text=True,
    )
    out = p.stdout
    # Parse "TFLOPS: X" and "SNR: Y dB"
    tflops = None
    snr = None
    pass_status = None
    for line in out.splitlines():
        if "TFLOPS:" in line:
            try:
                tflops = float(line.split("TFLOPS:")[-1].strip())
            except ValueError:
                pass
        elif "SNR:" in line:
            try:
                snr = float(line.split("SNR:")[-1].split("dB")[0].strip())
            except ValueError:
                pass
        elif "Result:" in line:
            pass_status = "PASS" in line
    return tflops, snr, pass_status, out


def welch_t(a, b):
    n_a, n_b = len(a), len(b)
    m_a, m_b = statistics.mean(a), statistics.mean(b)
    v_a = statistics.variance(a) if n_a > 1 else 0.0
    v_b = statistics.variance(b) if n_b > 1 else 0.0
    se = math.sqrt(v_a / n_a + v_b / n_b)
    if se == 0:
        return float("inf") if m_a != m_b else 0.0
    return (m_a - m_b) / se


def main():
    if len(sys.argv) != 5:
        print("usage: r27a_sweep.py <layout> M N K"); sys.exit(2)
    layout = sys.argv[1].lower()
    M, N, K = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    # Disable preshuffle V2 of OTHER layouts to keep build identical (we only
    # care about `layout`). Actually preheat_then_bench picks one layout via
    # MXFP8_LAYOUTS, so that's fine.

    cps = [0, 1, 2, 3]
    results = {}
    for cp in cps:
        ok = build(layout, M, N, K, cp)
        if not ok:
            results[cp] = {"build_ok": False}
            continue
        # Run 5x; abort cell if first run fails correctness.
        tflops_list = []
        snr_list = []
        for r in range(5):
            tflops, snr, ok, out = bench(layout, M, N, K)
            if tflops is None:
                print(f"  [cp={cp} run={r}] PARSE FAIL\n{out[-500:]}")
                continue
            tflops_list.append(tflops)
            snr_list.append(snr)
            print(f"  [cp={cp} run={r}] TFLOPS={tflops:.2f} SNR={snr:.2f} dB pass={ok}")
            if r == 0 and not ok:
                print("  Correctness FAIL on first run, aborting cp=", cp)
                break
        results[cp] = {
            "build_ok": True,
            "tflops": tflops_list,
            "snr": snr_list,
            "median": statistics.median(tflops_list) if tflops_list else None,
            "stdev": statistics.stdev(tflops_list) if len(tflops_list) > 1 else None,
            "snr_min": min(snr_list) if snr_list else None,
        }

    # Welch t vs cp=0
    base = results.get(0, {}).get("tflops") or []
    print("\n=== SUMMARY ===")
    for cp in cps:
        r = results.get(cp, {})
        if not r.get("build_ok") or not r.get("tflops"):
            print(f"cp={cp}: NO DATA")
            continue
        med = r["median"]; sd = r["stdev"]; snr_min = r["snr_min"]
        if cp == 0:
            print(f"cp=0 (BASELINE): median={med:.2f} std={sd:.2f} snr_min={snr_min:.2f} dB")
        else:
            t = welch_t(r["tflops"], base) if base else None
            delta = med - results[0]["median"] if results.get(0, {}).get("median") else None
            print(f"cp={cp}: median={med:.2f} std={sd:.2f} delta={delta:+.2f} TFLOPS Welch t={t:+.2f} snr_min={snr_min:.2f} dB")

    out_json = os.path.join(HERE, f"r27a_sweep_{layout}_{M}x{N}x{K}.json")
    with open(out_json, "w") as f:
        json.dump({"layout": layout, "M": M, "N": N, "K": K, "results": results}, f, indent=2)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
