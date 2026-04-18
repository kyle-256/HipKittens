#!/usr/bin/env python3
"""R31-A 5-rep verify: incumbent + u16 + u32 on a SINGLE idle GPU (sequential).

Each variant: 5 separate fresh subprocess invocations of bench_R31_optA.py for one variant.
Compute mean ± std and decide winner.
"""
import os, sys, json, subprocess, time, statistics, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_one(tag, gpu):
    cmd = ["python3", os.path.join(SCRIPT_DIR, "bench_R31_optA.py"),
           "--gpu", str(gpu), "--variants", tag]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    # Parse the JSON file output
    j = json.load(open(os.path.join(SCRIPT_DIR, "R31_OPT_A_BENCH_RESULTS.json")))
    return j["results"][0].get("tflops", None), j["results"][0].get("avg_ms", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--variants", nargs="+", default=["incumbent", "u16", "u32"])
    args = ap.parse_args()

    out = {}
    print(f"R31-A 5-rep verify on GPU {args.gpu}, variants={args.variants}, reps={args.reps}")
    for tag in args.variants:
        tfs = []
        mss = []
        for i in range(args.reps):
            t0 = time.time()
            tf, ms = run_one(tag, args.gpu)
            dt = time.time() - t0
            print(f"  [{tag}] rep{i+1}: {tf} TFLOPS  ({ms} ms)  wall={dt:.1f}s", flush=True)
            tfs.append(tf); mss.append(ms)
        mean = statistics.mean(tfs)
        std = statistics.stdev(tfs) if len(tfs) > 1 else 0.0
        out[tag] = {"reps": tfs, "mean_tflops": round(mean, 2),
                    "std_tflops": round(std, 2),
                    "min": min(tfs), "max": max(tfs),
                    "mean_ms": round(statistics.mean(mss), 4)}
        print(f"  [{tag}] MEAN={mean:.2f} ± {std:.2f} TFLOPS\n", flush=True)

    out_path = os.path.join(SCRIPT_DIR, "R31_OPT_A_VERIFY5_RESULTS.json")
    with open(out_path, "w") as f:
        json.dump({"gpu_id": args.gpu, "reps": args.reps, "results": out}, f, indent=2)
    print(f"\nResults: {out_path}")
    # Pretty summary
    inc = out.get("incumbent", {}).get("mean_tflops", 0)
    print("\n=== Summary ===")
    for tag, d in out.items():
        ratio = d["mean_tflops"] / inc * 100 if inc else 0
        print(f"  {tag:10s} {d['mean_tflops']:>7.1f} ± {d['std_tflops']:5.1f}  ({ratio:6.2f}% vs incumbent)  [reps: {d['reps']}]")


if __name__ == "__main__":
    main()
